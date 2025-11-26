# search_space.py
"""
Hyperparameter search space for the Genetic Algorithm.
This matches the ranges and categorical sets described in the paper.
"""

search_space = {
    # Continuous hyperparameters
    "eta": (1e-6, 1e-4),                  # learning rate: ampliamos techo para permitir algo más de adaptación
    "pdrop": (0.0, 0.3),                  # dropout
    "weight_decay": (0.0, 0.05),          # AdamW weight decay reducido
    "p_lora": (0.05, 0.2),                # LoRA dropout (evitar 0 para regularizar capas finales)

    # Discrete hyperparameters
    "batch_size": [1, 2, 4, 8],
    "epochs": [1, 2, 3, 4, 5, 6],
    "lora_r": [4, 8, 16],                 # rangos más estables para capas finales
    "alpha": [8, 16, 32],                 # alpha moderado
    "k_layers": [0.1, 0.25, 0.4],         # % de capas finales a adaptar (algo más de capacidad sin llegar a 50%)
    "seq_len": [512, 1024, 2048],
    "quant": [4, 8],                      # quantization bits (QLoRA)
}
