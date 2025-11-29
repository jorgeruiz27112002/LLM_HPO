# run_preview.py
"""
Preview answers from the best GA individual (or provided hparams) on the test set.
Uses the same prompt format as evaluation.py and prints a few samples.
"""

import argparse
import csv
import json
import torch
from pathlib import Path

from src.model_wrapper import load_model
from src.dataset_generator import load_test_data
from src.search_space import search_space


def _detect_device(explicit=None):
    if explicit:
        return explicit
    
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _load_best_hparams(log_path: Path):
    if not log_path.exists():
        raise FileNotFoundError(f"No se encontró el archivo de logs: {log_path}")
        
    rows = list(csv.DictReader(log_path.open()))
    if not rows:
        raise ValueError(f"No rows found in {log_path}")
    
    # Buscar el mejor fitness (asumiendo que fitness es la columna de score)
    best = max(rows, key=lambda r: float(r["fitness"]))
    print(f"[Preview] Best individual found (Gen {best['generation']}): Fitness {best['fitness']}")
    
    h = {}
    for k, space in search_space.items():
        if k in best:
            raw = best[k]
            if isinstance(space, tuple):
                h[k] = float(raw)
            else:
                # Discrete: assume int, unless it looks like a float (e.g. k_layers 0.5)
                try:
                    val = float(raw)
                    if val.is_integer() and all(isinstance(x, int) for x in space):
                         h[k] = int(val)
                    else:
                         h[k] = val
                except ValueError:
                    h[k] = raw # string fallback
    return h


def _build_prompt(example):
    context = example.get("context", "")
    question = example.get("question", "")
    return (
        "Responde usando SOLO texto copiado literalmente del contexto. No añadas nada más.\n"
        f"Context: {context}\nQuestion: {question}\nAnswer:"
    )


def preview_answers(model, tokenizer, data, device_arg, max_samples=5, gen_max_new_tokens=64):
    samples = data[:max_samples]
    results = []
    
    # Detectar dónde está realmente el modelo para enviar los inputs allí
    if hasattr(model, "device"):
        target_device = model.device
    else:
        try:
            target_device = next(model.parameters()).device
        except StopIteration:
            target_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
    # Fallback si el modelo reporta CPU pero queríamos GPU
    if target_device.type == 'cpu' and device_arg and "cuda" in str(device_arg):
        target_device = torch.device(device_arg)

    print(f"[Preview] Inference running on: {target_device}")

    model.eval()
    # NOTA: No hacemos model.to(device) porque rompe modelos cuantizados (bitsandbytes)
    
    with torch.no_grad():
        for ex in samples:
            prompt = _build_prompt(ex)
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=tokenizer.model_max_length,
            ).to(target_device) # <--- CORRECCIÓN: Usar target_device detectado
            
            gen_ids = model.generate(
                **inputs,
                max_new_tokens=min(gen_max_new_tokens, 128),
                do_sample=False,
                num_beams=1,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            prompt_len = inputs["input_ids"].shape[1]
            pred_answer = tokenizer.decode(gen_ids[0][prompt_len:], skip_special_tokens=True).strip()
            
            # Limpieza básica igual que en evaluation.py
            pred_answer = pred_answer.lstrip('\n').lstrip('"').strip()
            if "\n" in pred_answer:
                 pred_answer = pred_answer.split("\n")[0].strip()

            results.append(
                {
                    "question": ex.get("question", ""),
                    "reference": ex.get("answer", ""),
                    "prediction": pred_answer,
                }
            )
    return results


def parse_args():
    ap = argparse.ArgumentParser(description="Preview model answers on test set")
    ap.add_argument("--ga-log", default="logs/ga_metrics.csv", help="CSV con métricas GA para tomar el mejor individuo")
    ap.add_argument("--model-name", default="google/gemma-2b", help="Modelo base a cargar")
    ap.add_argument("--device", default=None, help="mps/cuda/cpu (auto si no se especifica)")
    ap.add_argument("--samples", type=int, default=5, help="Nº de ejemplos de test a mostrar")
    ap.add_argument("--hparams-json", default=None, help="Ruta a JSON con hiperparámetros (opcional, si no se usa GA log)")
    return ap.parse_args()


def main():
    args = parse_args()
    log_path = Path(args.ga_log)
    
    print(f"[Preview] Reading configuration...")
    if args.hparams_json:
        hparams = json.load(open(args.hparams_json))
        print(f"[Preview] Loaded from JSON: {args.hparams_json}")
    else:
        try:
            hparams = _load_best_hparams(log_path)
            print(f"[Preview] Loaded best hparams from GA log.")
        except Exception as e:
            print(f"[Preview] Error loading GA log: {e}")
            print("[Preview] Using default search space values for testing...")
            hparams = {k: (v[0] if isinstance(v, list) else v[0]) for k, v in search_space.items()}

    device = _detect_device(args.device)
    print(f"[Preview] Requested Device: {device}")
    
    print(f"[Preview] Loading model {args.model_name}...")
    model, tokenizer = load_model(hparams, base_model=args.model_name, device=device)
    
    print("[Preview] Loading test data...")
    test_data = load_test_data()
    
    print("[Preview] Generating answers...")
    results = preview_answers(model, tokenizer, test_data, device, max_samples=args.samples)
    
    print("\n" + "="*60)
    for i, r in enumerate(results, 1):
        print(f"\n--- Sample {i} ---")
        print(f"Q:    {r['question']}")
        print(f"Ref:  {r['reference']}")
        print(f"Pred: \033[92m{r['prediction']}\033[0m") # Green color for prediction
    print("\n" + "="*60)


if __name__ == "__main__":
    main()