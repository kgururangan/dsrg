import os
import psutil
import numpy as np

def rotate_1(U_L, U_R, F):
    return np.einsum("ij,ip,jq->pq", F, np.conj(U_L), U_R, optimize=True)

def rotate_2s(U_L, U_R, V):
    return np.einsum("ijkl,ip,jq,kr,ls->pqrs", V, np.conj(U_L), np.conj(U_L), U_R, U_R, optimize=True)

def rotate_2(Ua_L, Ub_L, Ua_R, Ub_R, V):
    return np.einsum("ijkl,ip,jq,kr,ls->pqrs", V, np.conj(Ua_L), np.conj(Ub_L), Ua_R, Ub_R, optimize=True)

def rotate_3s(U_L, U_R, W):
    return np.einsum("abcijk,ap,bq,cr,is,jt,ku->pqrstu", W, np.conj(U_L), np.conj(U_L), np.conj(U_L), U_R, U_R, U_R, optimize=True)

def rotate_3b(Ua_L, Ub_L, Ua_R, Ub_R, W):
    return np.einsum("abcijk,ap,bq,cr,is,jt,ku->pqrstu", W, np.conj(Ua_L), np.conj(Ua_L), np.conj(Ub_L), Ua_R, Ua_R, Ub_R, optimize=True)

def rotate_3c(Ua_L, Ub_L, Ua_R, Ub_R, W):
    return np.einsum("abcijk,ap,bq,cr,is,jt,ku->pqrstu", W, np.conj(Ua_L), np.conj(Ub_L), np.conj(Ub_L), Ua_R, Ub_R, Ub_R, optimize=True)

def inv_rotate_1(U_L, U_R, F):
    return np.einsum("ij,pi,qj->pq", F, np.conj(U_L), U_R, optimize=True)

def inv_rotate_2s(U_L, U_R, V):
    return np.einsum("ijkl,pi,qj,rk,sl->pqrs", V, np.conj(U_L), np.conj(U_L), U_R, U_R, optimize=True)

def inv_rotate_2(Ua_L, Ub_L, Ua_R, Ub_R, V):
    return np.einsum("ijkl,pi,qj,rk,sl->pqrs", V, np.conj(Ua_L), np.conj(Ub_L), Ua_R, Ub_R, optimize=True)

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

def numel_in_dict(d, nrank=-1):
    if nrank == -1:
        return sum([np.prod(v.shape) for v in d.values()])
    else:
        return sum([np.prod(v.shape) for k, v in d.items() if len(k) <= nrank])

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
