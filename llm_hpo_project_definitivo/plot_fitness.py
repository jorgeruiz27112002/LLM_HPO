import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_ga_evolution(csv_path="logs/ga_metrics.csv", output_path="logs/fitness_evolution.png"):
    if not os.path.exists(csv_path):
        print(f"ERROR: No se encontró el archivo {csv_path}")
        return

    # Leer el CSV
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"ERROR al leer el CSV: {e}")
        return

    # Verificar columnas necesarias
    if "generation" not in df.columns or "fitness" not in df.columns:
        print("ERROR: El CSV no tiene las columnas 'generation' o 'fitness'.")
        print("Columnas encontradas:", df.columns)
        return

    # Agrupar por generación
    # Calculamos el Máximo (Mejor individuo) y la Media (Salud de la población)
    stats = df.groupby("generation")["fitness"].agg(["max", "mean", "min"]).reset_index()

    # Configurar gráfica
    plt.figure(figsize=(10, 6))
    
    # Línea del mejor fitness (Max)
    plt.plot(stats["generation"], stats["max"], marker="o", linestyle="-", color="b", label="Mejor Fitness (Max)", linewidth=2)
    
    # Línea de la media (Mean)
    plt.plot(stats["generation"], stats["mean"], marker="s", linestyle="--", color="orange", label="Fitness Promedio", alpha=0.7)

    # Línea del peor (Min) - Opcional, para ver diversidad
    plt.fill_between(stats["generation"], stats["min"], stats["max"], color="gray", alpha=0.1, label="Rango de Población")

    plt.title("Evolución del Entrenamiento (GA + QLoRA)", fontsize=14)
    plt.xlabel("Generación", fontsize=12)
    plt.ylabel("Fitness (Maximizando)", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    
    # Guardar
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    print(f"✅ Gráfica guardada exitosamente en: {output_path}")
    print("\nResumen de progreso:")
    print(stats)

if __name__ == "__main__":
    plot_ga_evolution()
