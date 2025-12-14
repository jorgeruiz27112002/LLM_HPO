# utils.py
"""
Utility functions used by multiple modules.
"""

import time
import torch


def get_device():
    """
    Return device: mps (Apple Silicon) or cpu.
    """
    return "mps" if torch.backends.mps.is_available() else "cpu"


def time_block(label, func, *args, **kwargs):
    """
    Measures execution time of a block of code.
    """
    start = time.time()
    result = func(*args, **kwargs)
    end = time.time()
    print(f"[Timer] {label}: {end - start:.2f} seconds")
    return result


def normalize(value, min_val, max_val):
    """
    Normalizes a value to the range [0,1].
    """
    return (value - min_val) / (max_val - min_val)
