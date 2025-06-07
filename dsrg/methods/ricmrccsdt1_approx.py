import time
import numpy as np
from dsrg.utilities import regularized_denominator

from dsrg.wicked_contractions.ricmrccsdt1_approx_contractions import *
from dsrg.wicked_contractions.ricmrccsdt1_approx_hbar_active import *


def build_denominators(s, eps_a, eps_b, ref):
    n = np.newaxis
    h = ref.orbspace['hole_alpha']
    p = ref.orbspace['particle_alpha']
    H = ref.orbspace['hole_beta']
    P = ref.orbspace['particle_beta']

    denom = {'a': eps_a[n, h] - eps_a[p, n], 'b': eps_b[n, H] - eps_b[P, n],
             'aa': eps_a[n, n, h, n] + eps_a[n, n, n, h] - eps_a[p, n, n, n] - eps_a[n, p, n, n],
             'ab': eps_a[n, n, h, n] + eps_b[n, n, n, H] - eps_a[p, n, n, n] - eps_b[n, P, n, n],
             'bb': eps_b[n, n, H, n] + eps_b[n, n, n, H] - eps_b[P, n, n, n] - eps_b[n, P, n, n],
             'aaa': (eps_a[n, n, n, h, n, n] + eps_a[n, n, n, n, h, n] + eps_a[n, n, n, n, n, h]
                     - eps_a[p, n, n, n, n, n] - eps_a[n, p, n, n, n, n] - eps_a[n, n, p, n, n, n]),
             'aab': (eps_a[n, n, n, h, n, n] + eps_a[n, n, n, n, h, n] + eps_b[n, n, n, n, n, H]
                     - eps_a[p, n, n, n, n, n] - eps_a[n, p, n, n, n, n] - eps_b[n, n, P, n, n, n]),
             'abb': (eps_a[n, n, n, h, n, n] + eps_b[n, n, n, n, H, n] + eps_b[n, n, n, n, n, H]
                     - eps_a[p, n, n, n, n, n] - eps_b[n, P, n, n, n, n] - eps_b[n, n, P, n, n, n]),
             'bbb': (eps_b[n, n, n, H, n, n] + eps_b[n, n, n, n, H, n] + eps_b[n, n, n, n, n, H]
                     - eps_b[P, n, n, n, n, n] - eps_b[n, P, n, n, n, n] - eps_b[n, n, P, n, n, n])}

    reg_denom = {}
    for key, value in denom.items():
        reg_denom[key] = regularized_denominator(value, s)

    return denom, reg_denom


def initial_guess(ref, denom, reg_denom):
    # Slicing
    h = ref.orbspace['hole_alpha']
    p = ref.orbspace['particle_alpha']
    H = ref.orbspace['hole_beta']
    P = ref.orbspace['particle_beta']

    ha = ref.orbspace['hole_active_alpha']
    pa = ref.orbspace['particle_active_alpha']
    hA = ref.orbspace['hole_active_beta']
    pA = ref.orbspace['particle_active_beta']

    T = {}

    # 1st-order t2
    T['aa'] = ref.V['aa'][p, p, h, h] * reg_denom['aa']
    T['ab'] = ref.V['ab'][p, P, h, H] * reg_denom['ab']
    T['bb'] = ref.V['bb'][P, P, H, H] * reg_denom['bb']
    T['aa'][pa, pa, ha, ha] *= 0.
    T['ab'][pa, pA, ha, hA] *= 0.
    T['bb'][pA, pA, hA, hA] *= 0.

    # 1st-order t1
    T['a'] = (
            ref.F['a'][p, h]
            + np.einsum("axiu,ux,ux->ai", T['aa'][:, pa, :, ha], denom['a'][pa, ha], ref.gam1['a'],
                        optimize=True)
            + np.einsum("axiu,ux,ux->ai", T['ab'][:, pA, :, hA], denom['b'][pA, hA], ref.gam1['b'],
                        optimize=True)
    )
    T['a'] *= reg_denom['a']
    T['a'][pa, ha] *= 0.
    T['b'] = (
            ref.F['b'][P, H]
            + np.einsum("axiu,ux,ux->ai", T['bb'][:, pA, :, hA], denom['b'][pA, hA], ref.gam1['b'],
                        optimize=True)
            + np.einsum("xaui,ux,ux->ai", T['ab'][pA, :, hA, :], denom['a'][pa, ha], ref.gam1['a'],
                        optimize=True)
    )
    T['b'] *= reg_denom['b']
    T['b'][pA, hA] *= 0.

    # Set initial T3 to 0
    T['aaa'] = np.zeros_like(reg_denom['aaa'])
    T['aab'] = np.zeros_like(reg_denom['aab'])
    T['abb'] = np.zeros_like(reg_denom['abb'])
    T['bbb'] = np.zeros_like(reg_denom['bbb'])
    return T


def update_t(T, X, ref, denom, reg_denom, **kwargs):
    # Slicing
    ha = ref.orbspace['hole_active_alpha']
    pa = ref.orbspace['particle_active_alpha']
    hA = ref.orbspace['hole_active_beta']
    pA = ref.orbspace['particle_active_beta']

    # T_old = {}
    # T_old['a'] = T['a'].copy()
    # T_old['b'] = T['b'].copy()
    # T_old['aa'] = T['aa'].copy()
    # T_old['ab'] = T['ab'].copy()
    # T_old['bb'] = T['bb'].copy()
    T_old = {k: v for k, v in T.items()}

    T['a'] = (X['a'] + T['a'] * denom['a']) * reg_denom['a']
    T['a'][pa, ha] = .0
    T['b'] = (X['b'] + T['b'] * denom['b']) * reg_denom['b']
    T['b'][pA, hA] = .0
    T['aa'] = (X['aa'] + T['aa'] * denom['aa']) * reg_denom['aa']
    T['aa'][pa, pa, ha, ha] = .0
    T['ab'] = (X['ab'] + T['ab'] * denom['ab']) * reg_denom['ab']
    T['ab'][pa, pA, ha, hA] = .0
    T['bb'] = (X['bb'] + T['bb'] * denom['bb']) * reg_denom['bb']
    T['bb'][pA, pA, hA, hA] = .0
    T['aaa'] = X['aaa'] * reg_denom['aaa']
    T['aaa'][pa, pa, pa, ha, ha, ha] = .0
    T['aab'] = X['aab'] * reg_denom['aab']
    T['aab'][pa, pa, pA, ha, ha, hA] = .0
    T['abb'] = X['abb'] * reg_denom['abb']
    T['abb'][pa, pA, pA, ha, hA, hA] = .0
    T['bbb'] = X['bbb'] * reg_denom['bbb']
    T['bbb'][pA, pA, pA, hA, hA, hA] = .0

    # compute the change in T (residual)
    dT = {key: T[key] - T_old[key] for key in T_old.keys()}
    # dT = {}
    # dT['a'] = T['a'].copy()
    # dT['b'] = T['b'].copy()
    # dT['aa'] = T['aa'].copy()
    # dT['ab'] = T['ab'].copy()
    # dT['bb'] = T['bb'].copy()

    return T, dT


def compute_residual(hamiltonian, T, ref, herm):
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

    # Initial value for the residual (0 commutators)
    X = {'0': 0.0,
         'a': ref.F['a'][p, h].copy(),
         'b': ref.F['b'][P, H].copy(),
         'aa': 0.25 * ref.V['aa'][p, p, h, h].copy(),
         'ab': ref.V['ab'][p, P, h, H].copy(),
         'bb': 0.25 * ref.V['bb'][P, P, H, H].copy(),
         'aaa': np.zeros((nua, nua, nua, noa, noa, noa)),
         'aab': np.zeros((nua, nua, nub, noa, noa, nob)),
         'abb': np.zeros((nua, nub, nub, noa, nob, nob)),
         'bbb': np.zeros((nub, nub, nub, nob, nob, nob))}

    # 0-body (energy)
    # _t0 = time.time()
    X = H_T_ncomm1_nbody0(X, hamiltonian, T, ref.gam1, ref.eta1, ref.lambdas, ref.orbspace)
    X = H_T_ncomm2_nbody0(X, hamiltonian, T, ref.gam1, ref.eta1, ref.lambdas, ref.orbspace)
    # print(f"time for zerobody {time.time() - _t0}")
    # 1-body
    # _t0 = time.time()
    X = H_T_ncomm1_nbody1(X, hamiltonian, T, ref.gam1, ref.eta1, ref.lambdas, ref.orbspace)
    X = H_T_ncomm2_nbody1(X, hamiltonian, T, ref.gam1, ref.eta1, ref.lambdas, ref.orbspace)
    # print(f"time for onebody {time.time() - _t0}")
    # 2-body
    # _t0 = time.time()
    X = H_T_ncomm1_nbody2(X, hamiltonian, T, ref.gam1, ref.eta1, ref.lambdas, ref.orbspace)
    X = H_T_ncomm2_nbody2(X, hamiltonian, T, ref.gam1, ref.eta1, ref.lambdas, ref.orbspace)
    # print(f"time for twobody {time.time() - _t0}")
    # 3-body
    # _t0 = time.time()
    X = H_T_ncomm1_nbody3(X, hamiltonian, T, ref.gam1, ref.eta1, ref.lambdas, ref.orbspace)
    # print(f"time for threebody {time.time() - _t0}")

    # four-virtual blocks
    v = ref.orbspace['virt_alpha']
    V = ref.orbspace['virt_beta']
    X['aa'] = vvvv_t2aa_terms(X['aa'], hamiltonian['aa'][v, v, v, v], T, ref.orbspace)
    X['bb'] = vvvv_t2bb_terms(X['bb'], hamiltonian['bb'][V, V, V, V], T, ref.orbspace)
    X['ab'] = vvvv_t2ab_terms(X['ab'], hamiltonian['ab'][v, V, v, V], T, ref.orbspace)

    # antisymmetrize twobody
    X['aa'] -= X['aa'].transpose(1, 0, 2, 3)
    X['aa'] -= X['aa'].transpose(0, 1, 3, 2)
    X['bb'] -= X['bb'].transpose(1, 0, 2, 3)
    X['bb'] -= X['bb'].transpose(0, 1, 3, 2)

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
    return X


def compute_hbar_active(hamiltonian, T, ref, herm, reg_denom):
    # Slicing
    a = ref.orbspace['active_alpha']
    A = ref.orbspace['active_beta']
    # ha = ref.orbspace['hole_active_alpha']
    # pa = ref.orbspace['particle_active_alpha']
    # hA = ref.orbspace['hole_active_beta']
    # pA = ref.orbspace['particle_active_beta']

    # # Dimensions
    # nua, nub, noa, nob = ref.V['ab'][p, P, h, H].shape

    # # Initial value for the residual (0 commutators)
    # X = {'a': ref.F['a'][a, a].copy(),
    #      'b': ref.F['b'][A, A].copy(),
    #      'aa': 0.25 * ref.V['aa'][a, a, a, a].copy(),
    #      'ab': ref.V['ab'][a, A, a, A].copy(),
    #      'bb': 0.25 * ref.V['bb'][A, A, A, A].copy()}
    
    # X3 = {'aaa': np.zeros((nua, nua, nua, noa, noa, noa)),
    #       'aab': np.zeros((nua, nua, nub, noa, noa, nob)),
    #       'abb': np.zeros((nua, nub, nub, noa, nob, nob)),
    #       'bbb': np.zeros((nub, nub, nub, nob, nob, nob))}
    
    # # For consistency, recompute T3
    # X3 = H_T_ncomm1_nbody3(X3, hamiltonian, T, ref.gam1, ref.eta1, ref.lambdas, ref.orbspace)
    # T['aaa'] = X3['aaa'] * reg_denom['aaa']
    # T['aaa'][pa, pa, pa, ha, ha, ha] = .0
    # T['aab'] = X3['aab'] * reg_denom['aab']
    # T['aab'][pa, pa, pA, ha, ha, hA] = .0
    # T['abb'] = X3['abb'] * reg_denom['abb']
    # T['abb'][pa, pA, pA, ha, hA, hA] = .0
    # T['bbb'] = X3['bbb'] * reg_denom['bbb']
    # T['bbb'][pA, pA, pA, hA, hA, hA] = .0

    # 1-body
    _t0 = time.time()
    X = Hbar_ncomm1_nbody1(X, hamiltonian, T, ref.gam1, ref.eta1, ref.lambdas, ref.orbspace)
    X = Hbar_ncomm2_nbody1(X, hamiltonian, T, ref.gam1, ref.eta1, ref.lambdas, ref.orbspace)
    # print(f"time for onebody {time.time() - _t0}")
    # 2-body
    _t0 = time.time()
    X = Hbar_ncomm1_nbody2(X, hamiltonian, T, ref.gam1, ref.eta1, ref.lambdas, ref.orbspace)
    X = Hbar_ncomm2_nbody2(X, hamiltonian, T, ref.gam1, ref.eta1, ref.lambdas, ref.orbspace)
    # print(f"time for twobody {time.time() - _t0}")
    
    # antisymmetrize twobody
    X['aa'] -= X['aa'].transpose(1, 0, 2, 3)
    X['aa'] -= X['aa'].transpose(0, 1, 3, 2)
    X['bb'] -= X['bb'].transpose(1, 0, 2, 3)
    X['bb'] -= X['bb'].transpose(0, 1, 3, 2)
    return X