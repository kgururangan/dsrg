import os
import psutil
import numpy as np
import subprocess

def semicanonicalize_active(hbar_act, ref):
    def _rotate_1(U, F):
        return np.einsum("ij,ip,jq->pq", F, np.conj(U), U, optimize=True)

    def _rotate_2s(U, V):
        return np.einsum("ijkl,ip,jq,kr,ls->pqrs", V, np.conj(U), np.conj(U), U, U, optimize=True)

    def _rotate_2(Ua, Ub, V):
        return np.einsum("ijkl,ip,jq,kr,ls->pqrs", V, np.conj(Ua), np.conj(Ub), Ua, Ub, optimize=True)

    a = ref.orbspace['active_alpha']
    A = ref.orbspace['active_beta']
    # semi-canonicalize 1- and 2-body integrals
    hbar_act['a'] = _rotate_1(ref.U['a'][a, a], hbar_act['a'])
    hbar_act['b'] = _rotate_1(ref.U['b'][A, A], hbar_act['b'])
    hbar_act['aa'] = _rotate_2s(ref.U['a'][a, a], hbar_act['aa'])
    hbar_act['ab'] = _rotate_2(ref.U['a'][a, a], ref.U['b'][A, A], hbar_act['ab'])
    hbar_act['bb'] = _rotate_2s(ref.U['b'][A, A], hbar_act['bb'])
    return hbar_act

def flatten_dict_to_vector(d):
    return np.concatenate([v.ravel() for v in d.values()])

def unflatten_vector_to_dict(vec, shapes, sizes):
    out = {}
    i = 0
    for k in shapes:
        sz = sizes[k]
        out[k] = vec[i:i + sz].reshape(shapes[k])
        i += sz
    return out

def numel_in_dict(d):
    return sum([np.prod(v.shape) for v in d.values()])

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

def clean_up(fid, n):
    for i in range(n):
        remove_file(fid + "-" + str(i + 1) + ".npy")
    return

def remove_file(filePath):
    try:
        os.remove(filePath)
    except OSError:
        pass
    return

def get_git_commit_id():
    try:
        commit_id = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).strip().decode('utf-8')
        return commit_id
    except subprocess.CalledProcessError:
        return "N/A"
