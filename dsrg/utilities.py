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
    '''Compute the denomiator factor [1 - exp(s*x^2)]/x. For small
    values of s*x^2, apply Taylor expansion of exp(s*x^2). This allows
    one to recover the s -> infty limit.'''
    z = np.sqrt(s) * x
    if abs(z) <= 1.0e-09:
        return np.sqrt(s)*(z - z**3/2 + z**5/6)
    return (1. - np.exp(-s * x**2)) / x

def get_memory_usage():
    """Returns the amount of memory currently used in MB. Useful for
    investigating the memory usages of various routines."""
    current_process = psutil.Process(os.getpid())
    memory = current_process.memory_info().rss
    return memory / (1024 * 1024)

