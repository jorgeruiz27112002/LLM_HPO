# search_space.py
"""
Hyperparameter search space for the Genetic Algorithm.
This matches the ranges and categorical sets described in the paper.
"""

search_space = {
    # Continuous hyperparameters
    "eta": (5e-5, 2e-4),          # Learning rate un poco más alto para convergencia rápida
    "pdrop": (0.0, 0.2),
    "weight_decay": (0.0, 0.01),
    "p_lora": (0.05, 0.1),

    # Discrete hyperparameters - Optimizados para Apple Silicon M4 Max (64GB)
    "batch_size": [8, 16],        # Limitar para evitar OOM en MPS y usar grad accumulation
    "epochs": [3, 4],             # Más épocas pero acotadas
    "lora_r": [8, 16, 32],
    "alpha": [16, 32],
    "k_layers": [0.5, 0.75],      # Evitar forzar todas las capas en MPS
    "seq_len": [512],             # Fijar a 512 para estabilidad y throughput
    "quant": [4, 8],              # (El wrapper gestionará esto, MPS usará bf16 si es necesario)
}
