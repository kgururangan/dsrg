import os
import psutil
import numpy as np

def spatial_index(p):
    if p % 2 == 0:
        return int(p / 2)
    else:
        return int((p + 1) / 2)

def spin_label(p):
    if p % 2 == 0:
        return "B"
    else:
        return "A"

def regularized_denominator(x, s):
    z = np.sqrt(s) * x
    small = np.abs(z) <= 1.0e-09
    # For small z, use the Taylor approximation
    result = np.where(
        small,
        np.sqrt(s) * (z - z ** 3 / 2 + z ** 5 / 6),
        (1. - np.exp(-s * x ** 2)) / x
    )
    # result = (1. - np.exp(-s*x**2)) * np.reciprocal(x)
    return result

def regularized_denominator_2(x, s):
    z = np.sqrt(s) * x

    if np.abs(z) <= 1.0e-09:
        return np.sqrt(s)*(z - z**3/2 + z**5/6)

    return (1. - np.exp(-s * x**2)) / x

def get_memory_usage():
    """Returns the amount of memory currently used in MB. Useful for
    investigating the memory usages of various routines."""
    current_process = psutil.Process(os.getpid())
    memory = current_process.memory_info().rss
    return memory / (1024 * 1024)

