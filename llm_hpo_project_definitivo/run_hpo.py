# run_hpo.py
"""
Main launcher for the Hyperparameter Optimization experiment.
This script:
1. Builds dataset (can be skipped if already cached)
2. Runs Genetic Algorithm
3. Prints best hyperparameters
"""

import argparse
import os

from src.dataset_generator import build_dataset, load_validation_data
from src.genetic_algorithm import run_ga
from src.model_wrapper import load_model
from src.evaluation import evaluate_model
from src.search_space import search_space

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


def parse_args():
    parser = argparse.ArgumentParser(description="Run HPO with GA + LoRA on Gemma")
    parser.add_argument("--pdf-path", default="data/raw_pdf/source_book.pdf", help="Ruta al PDF fuente para generar el dataset")
    parser.add_argument("--output-dir", default="data/processed", help="Directorio donde guardar/cargar el dataset procesado")
    parser.add_argument("--skip-dataset", action="store_true", help="No regenerar dataset si ya existe")
    parser.add_argument("--force-rebuild", action="store_true", help="Forzar regenerar el dataset aunque exista")
    parser.add_argument("--lang", default="es", help="Idioma de las preguntas generadas (es/en). Por defecto español")
    parser.add_argument("--model-name", default="google/gemma-2b", help="Modelo base a cargar (usa uno abierto para pruebas)")
    parser.add_argument("--device", default=None, help="Forzar dispositivo (mps/cuda/cpu). Por defecto autodetección")
    parser.add_argument("--pop-size", type=int, default=None, help="Población GA (override)")
    parser.add_argument("--n-gen", type=int, default=None, help="Número de generaciones GA (override)")
    parser.add_argument("--eval-samples", type=int, default=8, help="Número de ejemplos de validación a evaluar por individuo")
    parser.add_argument("--baseline-only", action="store_true", help="Evalúa el modelo base con hiperparámetros por defecto y termina")
    parser.add_argument("--train-steps", type=int, default=150, help="Pasos de fine-tune por individuo")
    parser.add_argument("--log-metrics", action="store_true", help="Guardar métricas por individuo en CSV")
    parser.add_argument("--log-path", default="logs/ga_metrics.csv", help="Ruta del CSV de métricas si se activa --log-metrics")
    parser.add_argument("--plot", action="store_true", help="Generar gráfica de fitness por generación (requiere matplotlib)")
    return parser.parse_args()


def _default_hparams():
    h = {}
    for k, space in search_space.items():
        if isinstance(space, tuple):
            h[k] = sum(space) / 2.0
        else:
            h[k] = space[0]
    return h


def run_baseline(args):
    print("[BASELINE] Evaluando modelo base sin LoRA/GA...")
    hparams = _default_hparams()
    # Baseline: sin cuantización ni LoRA
    hparams["quant"] = None
    model, tokenizer = load_model(hparams, base_model=args.model_name, device=args.device, apply_lora=False)
    val_data = load_validation_data()
    metrics = evaluate_model(model, tokenizer, val_data, device=args.device or "auto", max_samples=args.eval_samples)
    Acc, F1, EM, PPL = metrics
    print(f"[BASELINE] Acc={Acc:.4f} F1={F1:.4f} EM={EM:.4f} PPL={PPL:.2f}")
    return metrics


def plot_fitness(csv_path, out_path="logs/fitness.png"):
    if plt is None:
        print("[PLOT] matplotlib no está disponible. Instala matplotlib para generar la gráfica.")
        return
    if not os.path.exists(csv_path):
        print(f"[PLOT] No se encontró {csv_path}")
        return
    import csv
    gens = {}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            g = int(row["generation"])
            fit = float(row["fitness"])
            gens.setdefault(g, []).append(fit)
    xs = sorted(gens.keys())
    ys = [max(gens[g]) for g in xs]
    plt.figure()
    plt.plot(xs, ys, marker="o")
    plt.xlabel("Generación")
    plt.ylabel("Fitness (máx)")
    plt.title("Evolución GA")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path)
    print(f"[PLOT] Gráfica guardada en {out_path}")


def main():
    args = parse_args()

    print("\n========================================")
    print("  LLM Hyperparameter Optimization (HPO)")
    print("  Using Genetic Algorithms + QLoRA")
    print("========================================\n")

    if not args.skip_dataset:
        print("[MAIN] Building dataset...")
        build_dataset(
            pdf_path=args.pdf_path,
            output_dir=args.output_dir,
            force_rebuild=args.force_rebuild,
            language=args.lang,
        )
        print("\n[MAIN] Dataset ready!")
    else:
        print("[MAIN] Skipping dataset build (flag --skip-dataset)")

    if args.baseline_only:
        run_baseline(args)
        return

    print("\n[MAIN] Running Genetic Algorithm...\n")
    best_hparams, best_fit, log = run_ga(
        model_name=args.model_name,
        device=args.device,
        pop_size=args.pop_size,
        n_gen=args.n_gen,
        max_eval_samples=args.eval_samples,
        max_train_steps=args.train_steps,
        log_metrics=args.log_metrics,
        log_path=args.log_path,
    )

    if args.log_metrics and args.plot:
        plot_fitness(args.log_path)

    print("\n========================================")
    print(" BEST HYPERPARAMETERS FOUND BY GA")
    print("========================================")
    for k, v in best_hparams.items():
        print(f"{k:15s} : {v}")

    print(f"\nBest fitness: {best_fit:.4f}")
    print("========================================\n")


if __name__ == "__main__":
    main()
