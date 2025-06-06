import time
import numpy as np
from dsrg.utilities import regularized_denominator

from dsrg.wicked_contractions.ricmrccsd_pert_t_contractions import *


def build_denominators(s, eps_a, eps_b, ref):
    n = np.newaxis
    h = ref.orbspace['hole_alpha']
    p = ref.orbspace['particle_alpha']
    H = ref.orbspace['hole_beta']
    P = ref.orbspace['particle_beta']

    reg_denom = {'aaa': (eps_a[n, n, n, h, n, n] + eps_a[n, n, n, n, h, n] + eps_a[n, n, n, n, n, h]
                     - eps_a[p, n, n, n, n, n] - eps_a[n, p, n, n, n, n] - eps_a[n, n, p, n, n, n]),
                'aab': (eps_a[n, n, n, h, n, n] + eps_a[n, n, n, n, h, n] + eps_b[n, n, n, n, n, H]
                        - eps_a[p, n, n, n, n, n] - eps_a[n, p, n, n, n, n] - eps_b[n, n, P, n, n, n]),
                'abb': (eps_a[n, n, n, h, n, n] + eps_b[n, n, n, n, H, n] + eps_b[n, n, n, n, n, H]
                        - eps_a[p, n, n, n, n, n] - eps_b[n, P, n, n, n, n] - eps_b[n, n, P, n, n, n]),
                'bbb': (eps_b[n, n, n, H, n, n] + eps_b[n, n, n, n, H, n] + eps_b[n, n, n, n, n, H]
                        - eps_b[P, n, n, n, n, n] - eps_b[n, P, n, n, n, n] - eps_b[n, n, P, n, n, n])}

    for key, value in reg_denom.items():
        reg_denom[key] = regularized_denominator(value, s)

    return reg_denom


def update_t(T, X, ref, reg_denom):
    # Slicing
    ha = ref.orbspace['hole_active_alpha']
    pa = ref.orbspace['particle_active_alpha']
    hA = ref.orbspace['hole_active_beta']
    pA = ref.orbspace['particle_active_beta']


    T['aaa'] = X['aaa'] * reg_denom['aaa']
    T['aaa'][pa, pa, pa, ha, ha, ha] = .0
    T['aab'] = X['aab'] * reg_denom['aab']
    T['aab'][pa, pa, pA, ha, ha, hA] = .0
    T['abb'] = X['abb'] * reg_denom['abb']
    T['abb'][pa, pA, pA, ha, hA, hA] = .0
    T['bbb'] = X['bbb'] * reg_denom['bbb']
    T['bbb'][pA, pA, pA, hA, hA, hA] = .0

    return T


def triples_correction(T, ref, hamiltonian, s):
    # Slicing
    h = ref.orbspace['hole_alpha']
    p = ref.orbspace['particle_alpha']
    H = ref.orbspace['hole_beta']
    P = ref.orbspace['particle_beta']

    ha = ref.orbspace['hole_active_alpha']
    pa = ref.orbspace['particle_active_alpha']
    hA = ref.orbspace['hole_active_beta']
    pA = ref.orbspace['particle_active_beta']

    nua, nub, noa, nob = ref.V['ab'][p, P, h, H].shape

    # Build n-body (regularized) MP denominators
    tic = time.time()
    reg_denom = build_denominators(
        s,
        np.real(np.diagonal(ref.F['a'])),
        np.real(np.diagonal(ref.F['b'])),
        ref,
    )
    toc = time.time()
    print(f"   ... build n-body regularized denominators: {toc - tic}s")


    # Initial value for the residual (0 commutators)
    X = {'0': 0.0,
         'aaa': np.zeros((nua, nua, nua, noa, noa, noa)),
         'aab': np.zeros((nua, nua, nub, noa, noa, nob)),
         'abb': np.zeros((nua, nub, nub, noa, nob, nob)),
         'bbb': np.zeros((nub, nub, nub, nob, nob, nob))}

    # 3-body
    # _t0 = time.time()
    X = H_T_ncomm1_nbody3(X, hamiltonian, T, ref.gam1, ref.eta1, ref.lambdas, ref.orbspace)
    # print(f"time for threebody {time.time() - _t0}")

    # antisymmetrize threebody
    X['aaa'] -= X['aaa'].transpose(0, 2, 1, 3, 4, 5)  # (bc)
    X['aaa'] -= X['aaa'].transpose(1, 0, 2, 3, 4, 5) + X['aaa'].transpose(2, 1, 0, 3, 4, 5)  # (a/bc)
    X['aaa'] -= X['aaa'].transpose(0, 1, 2, 3, 5, 4)  # (jk)
    X['aaa'] -= X['aaa'].transpose(0, 1, 2, 4, 3, 5) + X['aaa'].transpose(0, 1, 2, 5, 4, 3)  # (i/jk)

    X['aab'] -= X['aab'].transpose(1, 0, 2, 3, 4, 5)  # (ab)
    X['aab'] -= X['aab'].transpose(0, 1, 2, 4, 3, 5)  # (ij)

    X['abb'] -= X['abb'].transpose(0, 2, 1, 3, 4, 5)  # (bc)
    X['abb'] -= X['abb'].transpose(0, 1, 2, 3, 5, 4)  # (jk)

    X['bbb'] -= X['bbb'].transpose(0, 2, 1, 3, 4, 5)  # (bc)
    X['bbb'] -= X['bbb'].transpose(1, 0, 2, 3, 4, 5) + X['bbb'].transpose(2, 1, 0, 3, 4, 5)  # (a/bc)
    X['bbb'] -= X['bbb'].transpose(0, 1, 2, 3, 5, 4)  # (jk)
    X['bbb'] -= X['bbb'].transpose(0, 1, 2, 4, 3, 5) + X['bbb'].transpose(0, 1, 2, 5, 4, 3)  # (i/jk)

    # Build T3 amplitudes 
    T = update_t(T, X, ref, reg_denom)

    # 0-body (energy)
    # _t0 = time.time()
    X = H_T_ncomm1_nbody0(X, hamiltonian, T, ref.gam1, ref.eta1, ref.lambdas, ref.orbspace)
    X = H_T_ncomm2_nbody0(X, hamiltonian, T, ref.gam1, ref.eta1, ref.lambdas, ref.orbspace)
    # print(f"time for zerobody {time.time() - _t0}")

    return X['0']
