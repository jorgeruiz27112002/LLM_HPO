# fitness.py
"""
Fitness function used by the Genetic Algorithm.
Combines performance and efficiency metrics as described in the research paper.
"""

import torch
from math import log


def compute_fitness(metrics):
    """
    metrics = (Acc, F1, EM, PPL, C)

    Where:
    - Acc: accuracy on validation set
    - F1: F1 score
    - EM: exact match
    - PPL: perplexity
    - C: computational cost (normalized)

    Returns:
        A single scalar value to maximize.
    """

    Acc, F1, EM, PPL, C = metrics

    # Normalizar valores problemáticos y acotar PPL para evitar penalizaciones extremas
    if not torch.isfinite(torch.tensor(PPL)):
        PPL = 1e6
    PPL = max(1e-6, min(PPL, 1e6))
    if not torch.isfinite(torch.tensor(Acc)):
        Acc = 0.0
    if not torch.isfinite(torch.tensor(F1)):
        F1 = 0.0
    if not torch.isfinite(torch.tensor(EM)):
        EM = 0.0
    if not torch.isfinite(torch.tensor(C)):
        C = 1.0

    fitness_value = (
        0.3 * Acc +
        0.3 * F1 +
        0.2 * EM -
        0.1 * log(PPL) -
        0.1 * C
    )

    return fitness_value
