import time
import numpy as np
from dsrg.utilities import regularized_denominator
from dsrg.wicked_contractions.ricmrccsd_contractions import *


def build_denominators(s, eps_a, eps_b, ref):
    n = np.newaxis
    h = ref.orbspace['hole_alpha']
    p = ref.orbspace['particle_alpha']
    H = ref.orbspace['hole_beta']
    P = ref.orbspace['particle_beta']

    denom = {'a': eps_a[n, h] - eps_a[p, n], 'b': eps_b[n, H] - eps_b[P, n],
             'aa': eps_a[n, n, h, n] + eps_a[n, n, n, h] - eps_a[p, n, n, n] - eps_a[n, p, n, n],
             'ab': eps_a[n, n, h, n] + eps_b[n, n, n, H] - eps_a[p, n, n, n] - eps_b[n, P, n, n],
             'bb': eps_b[n, n, H, n] + eps_b[n, n, n, H] - eps_b[P, n, n, n] - eps_b[n, P, n, n]}

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
    return T


def update_t(T, X, ref, denom, reg_denom, **kwargs):
    
    # Slicing
    ha = ref.orbspace['hole_active_alpha']
    pa = ref.orbspace['particle_active_alpha']
    hA = ref.orbspace['hole_active_beta']
    pA = ref.orbspace['particle_active_beta']

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

    # compute the change in T (residual)
    dT = {key: T[key] - T_old[key] for key in T.keys()}

    return T, dT


def compute_residual(hamiltonian, T, ref, herm):
    
    # Slicing
    h = ref.orbspace['hole_alpha']
    p = ref.orbspace['particle_alpha']
    H = ref.orbspace['hole_beta']
    P = ref.orbspace['particle_beta']

    # Initial value for the residual (0 commutators)
    X = {'0': 0.0,
         'a': ref.F['a'][p, h].copy(),
         'b': ref.F['b'][P, H].copy(),
         'aa': 0.25 * ref.V['aa'][p, p, h, h].copy(),
         'ab': ref.V['ab'][p, P, h, H].copy(),
         'bb': 0.25 * ref.V['bb'][P, P, H, H].copy()}
    
    # 0-body (energy)
    _t0 = time.time()
    X = H_T_ncomm1_nbody0(X, hamiltonian, T, ref.gam1, ref.eta1, ref.lambdas, ref.orbspace)
    X = H_T_ncomm2_nbody0(X, hamiltonian, T, ref.gam1, ref.eta1, ref.lambdas, ref.orbspace)
    # print(f"time for zerobody {time.time() - _t0}")
    if herm:
        X['0'] *= 2.0
    # 1-body
    _t0 = time.time()
    X = H_T_ncomm1_nbody1(X, hamiltonian, T, ref.gam1, ref.eta1, ref.lambdas, ref.orbspace)
    X = H_T_ncomm2_nbody1(X, hamiltonian, T, ref.gam1, ref.eta1, ref.lambdas, ref.orbspace)
    # print(f"time for onebody {time.time() - _t0}")
    # 2-body
    _t0 = time.time()
    X = H_T_ncomm1_nbody2(X, hamiltonian, T, ref.gam1, ref.eta1, ref.lambdas, ref.orbspace)
    X = H_T_ncomm2_nbody2(X, hamiltonian, T, ref.gam1, ref.eta1, ref.lambdas, ref.orbspace)
    # print(f"time for twobody {time.time() - _t0}")
    # antisymmetrize twobody
    X['aa'] -= X['aa'].transpose(1, 0, 2, 3)
    X['aa'] -= X['aa'].transpose(0, 1, 3, 2)
    X['bb'] -= X['bb'].transpose(1, 0, 2, 3)
    X['bb'] -= X['bb'].transpose(0, 1, 3, 2)
    if herm:
        X['a'] += X['a'].T.conj()
        X['b'] += X['b'].T.conj()
        X['aa'] += X['aa'].transpose(2, 3, 0, 1).conj()
        X['ab'] += X['ab'].transpose(2, 3, 0, 1).conj()
        X['bb'] += X['bb'].transpose(2, 3, 0, 1).conj()
    return X