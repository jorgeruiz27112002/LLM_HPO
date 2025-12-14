# src/random_search.py
"""
Random Search for hyperparameter optimization of the LLM.
Simple alternative to Genetic Algorithm.
"""

import random
import csv
import os
import time
import torch
import torch.multiprocessing as mp

from .search_space import search_space
from .genetic_algorithm import (
    evaluate_individual, 
    decode_individual, 
    param_names,
    create_individual, # Reusing this as it does exactly what we need: random init
    _write_log
)

class RSIndividual(list):
    """Simple wrapper to hold metrics like Deap individuals do."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.metrics = None
        self.fitness = type("Fitness", (object,), {"values": (None,)})()

def worker_eval_rs(args):
    """
    Worker function for parallel random search.
    """
    ind_data, gpu_id, model_name, max_eval, max_steps = args
    device_str = f"cuda:{gpu_id}"
    
    ind_obj = RSIndividual(ind_data)
    
    try:
        # Reusing the robust evaluation logic from GA
        fitness = evaluate_individual(
            ind_obj,
            device=device_str,
            base_model=model_name,
            max_eval_samples=max_eval,
            cached=None,
            max_train_steps=max_steps
        )
        metrics = getattr(ind_obj, "metrics", None)
        return fitness, metrics
    except Exception as e:
        print(f"[RS Worker Error GPU {gpu_id}] {e}")
        return (-9999.0,), None

def run_random_search(
    n_iter=50,
    seed=42,
    model_name=None,
    device=None,
    max_eval_samples=8,
    max_train_steps=150,
    log_metrics=False,
    log_path="logs/rs_metrics.csv",
):
    """
    Runs Random Search for n_iter iterations.
    """
    # Multiprocessing setup
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    # Device detection
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    if device == "cuda" and torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print(f"[RS] Parallel Mode: Using {num_gpus} GPUs.")
    else:
        num_gpus = 0
        print(f"[RS] Serial Mode: Using {device}.")

    random.seed(seed)
    
    # Generate all individuals upfront
    individuals = [RSIndividual(create_individual()) for _ in range(n_iter)]
    
    best_ind = None
    best_fitness = -9999.0
    
    log_buffer = []

    print(f"[RS] Starting Random Search with {n_iter} iterations...")

    # Parallel Execution
    if num_gpus > 1:
        tasks = []
        for i, ind in enumerate(individuals):
            gpu_id = i % num_gpus
            tasks.append((ind[:], gpu_id, model_name, max_eval_samples, max_train_steps))
            
        with mp.Pool(processes=num_gpus) as pool:
            results = pool.map(worker_eval_rs, tasks)
            
        for ind, (fit, met) in zip(individuals, results):
            ind.fitness.values = fit
            ind.metrics = met # Re-attach metrics
            
    else:
        # Serial Execution
        for i, ind in enumerate(individuals):
            print(f"[RS] Iteration {i+1}/{n_iter}")
            ind.fitness.values = evaluate_individual(
                ind,
                device=device,
                base_model=model_name,
                max_eval_samples=max_eval_samples,
                cached=None, # Important: cached=None to force new load/config
                max_train_steps=max_train_steps
            )

            # Check for best
            fit_val = ind.fitness.values[0]
            if fit_val > best_fitness:
                best_fitness = fit_val
                best_ind = ind
            
            # Incremental Logging
            if log_metrics:
                ind_h = decode_individual(ind)
                if hasattr(ind, "metrics") and ind.metrics:
                    Acc, F1, EM, PPL, C = ind.metrics
                else:
                    Acc = F1 = EM = PPL = C = "" 
                
                row = {
                    "iteration": i,
                    "fitness": fit_val,
                    "Acc": Acc,
                    "F1": F1,
                    "EM": EM,
                    "PPL": PPL,
                    "C": C,
                    "is_best": 1 if ind is best_ind else 0,
                    **ind_h,
                }
                log_buffer.append(row)
                # Write immediately
                _write_log(log_buffer, log_path)



    best_hparams = decode_individual(best_ind) if best_ind else {}
    
    print("\n[RS] Best individual:", best_hparams)
    print("[RS] Best fitness:", best_fitness)
    
    return best_hparams, best_fitness
