# src/model_wrapper.py
"""
Model wrapper for Gemma with LoRA adapters, ready for MPS (Apple Silicon) and CUDA.
"""

import copy
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


BASE_MODEL = "google/gemma-2b"

# Cache tokenizer y modelo base para evitar recargas repetidas
_tokenizer_cache = {}
_base_model_cache = {}


def _detect_device(explicit=None):
    if explicit:
        return explicit

    # 1 — CUDA sempre é prioridade se estiver disponível
    if torch.cuda.is_available():
        return "cuda"

    # 2 — MPS só deve ser usado em ambiente Apple Silicon real
    if torch.backends.mps.is_available():
        try:
            # Tenta "testar" o backend para garantir que funciona
            torch.zeros(1).to("mps")
            return "mps"
        except Exception:
            pass  # fallback para cpu

    # 3 — CPU como fallback universal
    return "cpu"


def _build_quant_config(device, quant_bits):
    if device != "cuda":
        return None
    if quant_bits not in (4, 8):
        return None
    return BitsAndBytesConfig(
        load_in_4bit=quant_bits == 4,
        load_in_8bit=quant_bits == 8,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )


def _lora_targets(base_model):
    # Gemma/LLaMA style
    if "gemma" in base_model.lower() or "llama" in base_model.lower():
        return ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], ["layers"]
    # Fallback for otros modelos pequeños (p.ej., GPT-2 tiny)
    return "all-linear", None


def _set_dropout_rate(model, pdrop):
    """Set dropout probability across nn.Dropout modules to pdrop."""
    if pdrop is None:
        return
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.p = float(pdrop)


def load_model(hparams, base_model=BASE_MODEL, device=None, apply_lora=True):
    # Aseguramos que el device sea el string completo (ej: "cuda:7") si viene explícito
    # 1. Sanitización del device
    device_name = device if device else _detect_device()
    
    # Si por error llega "mps" pero no estamos en Mac, forzamos CUDA o CPU
    if device_name == "mps" and not torch.backends.mps.is_available():
        print("[Model Wrapper] WARNING: 'mps' requested but not available. Falling back to CUDA/CPU.")
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Extraer el ID del dispositivo si es cuda:X para pasarlo al device_map
    if isinstance(device_name, str) and device_name.startswith("cuda:"):
        # Mapear todo el modelo a esa GPU específica directamente
        dev_map = {"": device_name}
    else:
        dev_map = "auto" if device_name != "cpu" else None

    print(f"[Model Wrapper] Loading base model on {device_name} with map {dev_map}...")

    # Tokenizer cache
    if base_model in _tokenizer_cache:
        tokenizer = _tokenizer_cache[base_model]
    else:
        tokenizer = AutoTokenizer.from_pretrained(base_model)
        _tokenizer_cache[base_model] = tokenizer

    tokenizer.model_max_length = hparams["seq_len"]
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    orig_requested_quant = hparams.get("quant")
    requested_quant = orig_requested_quant
    quant_config = _build_quant_config(device_name, requested_quant) # Usar device_name normalizado
    dtype = torch.bfloat16 if "cuda" in device_name or "mps" in device_name else torch.float32

    # Lógica de carga
    try:
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            device_map=dev_map,
            dtype=dtype,
            quantization_config=quant_config,
        )
    except ImportError as e:
        # Fallback si falla bitsandbytes: cargamos sin cuantización
        if quant_config is not None:
            print(f"[Model Wrapper] Quantization failed ({e}); retrying without quant.")
            quant_config = None
            requested_quant = None
            model = AutoModelForCausalLM.from_pretrained(
                base_model,
                device_map=dev_map,
                dtype=dtype,
                quantization_config=None,
            )
        else:
            raise

    # --- CORRECCIÓN CRÍTICA AQUÍ ABAJO ---
    if quant_config is not None:
        # Aquí es donde DEBE estar el False para evitar el crash en multiprocessing
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)

    if apply_lora:
        total_layers = getattr(model.config, "num_hidden_layers", None)
        target_modules, layers_pattern = _lora_targets(base_model)

        layers_to_transform = None
        if isinstance(target_modules, list) and total_layers:
            if hparams["k_layers"] < 1.0:
                keep = max(1, int(total_layers * hparams["k_layers"]))
                start = max(total_layers - keep, 0)  # aplicar en las últimas capas
                layers_to_transform = list(range(start, total_layers))
            else:
                layers_to_transform = list(range(total_layers))
            # layers_pattern solo si tenemos capas definidas
            layers_pattern = ["layers"]
        else:
            layers_pattern = None

        lora_cfg = LoraConfig(
            r=hparams["lora_r"],
            lora_alpha=hparams["alpha"],
            lora_dropout=hparams["p_lora"],
            target_modules=target_modules,
            bias="none",
            task_type="CAUSAL_LM",
            layers_to_transform=layers_to_transform if isinstance(target_modules, list) else None,
            layers_pattern=layers_pattern,
        )

        try:
            model = get_peft_model(model, lora_cfg)
        except ValueError as e:
            # Fallback para modelos que no exponen los módulos esperados
            print(f"[Model Wrapper] LoRA target modules failed ({e}); retrying with all-linear")
            lora_cfg.target_modules = "all-linear"
            lora_cfg.layers_to_transform = None
            lora_cfg.layers_pattern = None
            model = get_peft_model(model, lora_cfg)

        if layers_to_transform and total_layers:
            print(f"[Model Wrapper] Applying LoRA to first {len(layers_to_transform)} layers out of {total_layers}")
        elif total_layers:
            print(f"[Model Wrapper] Applying LoRA to all {total_layers} layers")
    
    _set_dropout_rate(model, hparams.get("pdrop"))

    # Solo movemos manualmente si NO hay cuantización.
    if quant_config is None and device_name != "auto" and device_name != "cpu":
         model.to(device_name) # Aquí device_name ya está saneado, no fallará

    return model, tokenizer