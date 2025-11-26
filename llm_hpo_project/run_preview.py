"""
Preview answers from the best GA individual (or provided hparams) on the test set.
Uses the same prompt format as evaluation.py and prints a few samples.
"""

import argparse
import csv
import json
from pathlib import Path

from src.model_wrapper import load_model
from src.dataset_generator import load_test_data
from src.search_space import search_space


def _detect_device(explicit=None):
    if explicit:
        return explicit
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _load_best_hparams(log_path: Path):
    rows = list(csv.DictReader(log_path.open()))
    if not rows:
        raise ValueError(f"No rows found in {log_path}")
    best = max(rows, key=lambda r: float(r["fitness"]))
    h = {}
    for k, space in search_space.items():
        raw = best[k]
        if isinstance(space, tuple):
            h[k] = float(raw)
        else:
            # Discrete: assume int
            h[k] = int(float(raw))
    return h


def _build_prompt(example):
    context = example.get("context", "")
    question = example.get("question", "")
    return (
        "Responde usando SOLO texto copiado literalmente del contexto. No añadas nada más.\n"
        f"Context: {context}\nQuestion: {question}\nAnswer:"
    )


def preview_answers(model, tokenizer, data, device, max_samples=5, gen_max_new_tokens=64):
    samples = data[:max_samples]
    results = []
    import torch

    model.eval()
    model.to(device)
    with torch.no_grad():
        for ex in samples:
            prompt = _build_prompt(ex)
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=tokenizer.model_max_length,
            ).to(device)
            gen_ids = model.generate(
                **inputs,
                max_new_tokens=min(gen_max_new_tokens, 48),
                do_sample=False,
                num_beams=1,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            prompt_len = inputs["input_ids"].shape[1]
            pred_answer = tokenizer.decode(gen_ids[0][prompt_len:], skip_special_tokens=True).strip()
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
    if args.hparams_json:
        hparams = json.load(open(args.hparams_json))
    else:
        hparams = _load_best_hparams(log_path)
    device = _detect_device(args.device)
    print(f"[Preview] Device: {device}")
    print(f"[Preview] Loading model {args.model_name} with best hparams from {args.ga_log}")
    model, tokenizer = load_model(hparams, base_model=args.model_name, device=device)
    test_data = load_test_data()
    results = preview_answers(model, tokenizer, test_data, device, max_samples=args.samples)
    for i, r in enumerate(results, 1):
        print(f"\n--- Sample {i} ---")
        print("Q:", r["question"])
        print("Ref:", r["reference"])
        print("Pred:", r["prediction"])


if __name__ == "__main__":
    main()
