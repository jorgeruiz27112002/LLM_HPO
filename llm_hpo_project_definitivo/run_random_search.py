# run_random_search.py
"""
Launcher for Random Search HPO.
Replicates run_hpo.py interface but uses Random Search instead of GA.
"""

import argparse
import os

from src.dataset_generator import build_dataset, load_validation_data
from src.random_search import run_random_search
from src.model_wrapper import load_model
from src.evaluation import evaluate_model
from src.search_space import search_space

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


def parse_args():
    parser = argparse.ArgumentParser(description="Run HPO with Random Search + LoRA on Gemma")
    parser.add_argument("--pdf-path", default="data/raw_pdf/source_book.pdf", help="Ruta al PDF fuente para generar el dataset")
    parser.add_argument("--output-dir", default="data/processed", help="Directorio donde guardar/cargar el dataset procesado")
    parser.add_argument("--skip-dataset", action="store_true", help="No regenerar dataset si ya existe")
    parser.add_argument("--force-rebuild", action="store_true", help="Forzar regenerar el dataset aunque exista")
    parser.add_argument("--lang", default="es", help="Idioma de las preguntas generadas (es/en). Por defecto español")
    parser.add_argument("--model-name", default="google/gemma-2b", help="Modelo base a cargar")
    parser.add_argument("--device", default=None, help="Forzar dispositivo (mps/cuda/cpu)")
    parser.add_argument("--n-iter", type=int, default=10, help="Número de iteraciones de búsqueda aleatoria")
    parser.add_argument("--eval-samples", type=int, default=8, help="Número de ejemplos de validación a evaluar")
    parser.add_argument("--train-steps", type=int, default=150, help="Pasos de fine-tune por individuo")
    parser.add_argument("--log-metrics", action="store_true", help="Guardar métricas en CSV")
    parser.add_argument("--log-path", default="logs/rs_metrics.csv", help="Ruta del CSV de métricas")
    parser.add_argument("--plot", action="store_true", help="Generar gráfica básica (scatter fitness)")
    return parser.parse_args()


def plot_results(csv_path, out_path="logs/rs_fitness.png"):
    if plt is None:
        print("[PLOT] matplotlib no está disponible.")
        return
    if not os.path.exists(csv_path):
        print(f"[PLOT] No se encontró {csv_path}")
        return
    import csv
    iters = []
    fits = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            iters.append(int(row["iteration"]))
            fits.append(float(row["fitness"]))
            
    plt.figure()
    plt.scatter(iters, fits, c='blue', alpha=0.6, label='Random Run')
    # draw max line
    if fits:
        max_fit = max(fits)
        plt.axhline(y=max_fit, color='r', linestyle='--', label=f'Best: {max_fit:.4f}')
    
    plt.xlabel("Iteración")
    plt.ylabel("Fitness")
    plt.title("Random Search Results")
    plt.legend()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path)
    print(f"[PLOT] Gráfica guardada en {out_path}")


def main():
    args = parse_args()

    print("\n========================================")
    print("  LLM Hyperparameter Optimization (HPO)")
    print(f"  Using Random Search ({args.n_iter} iters)")
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
        print("[MAIN] Skipping dataset build")

    print("\n[MAIN] Running Random Search...\n")
    best_hparams, best_fit = run_random_search(
        n_iter=args.n_iter,
        model_name=args.model_name,
        device=args.device,
        max_eval_samples=args.eval_samples,
        max_train_steps=args.train_steps,
        log_metrics=args.log_metrics,
        log_path=args.log_path,
    )

    if args.log_metrics and args.plot:
        plot_results(args.log_path)

    print("\n========================================")
    print(" BEST HYPERPARAMETERS FOUND BY RS")
    print("========================================")
    for k, v in best_hparams.items():
        print(f"{k:15s} : {v}")

    print(f"\nBest fitness: {best_fit:.4f}")
    print("========================================\n")


if __name__ == "__main__":
    main()
