import time
import numpy as np
from dsrg.utilities import regularized_denominator
from dsrg.wicked_contractions.sqricmrccsd_contractions import *


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


def update_t(T, X, ref, denom, reg_denom):
    
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
    
    # T1 transformation via recursive commutator approximation
    hbar = recursive_commutator(hamiltonian, T, ref, herm, max_ncomm=12)

    # Initial value for the residual (0 commutators)
    X = {'0': hbar['0'][0],
         'a': hbar['a'][p, h].copy(),
         'b': hbar['b'][P, H].copy(),
         'aa': 0.25 * hbar['aa'][p, p, h, h].copy(),
         'ab': hbar['ab'][p, P, h, H].copy(),
         'bb': 0.25 * hbar['bb'][P, P, H, H].copy()}
    
    # 0-body (energy)
    _t0 = time.time()
    X = H_T_ncomm1_nbody0(X, hbar, T, ref.gam1, ref.eta1, ref.lambdas, ref.orbspace)
    X = H_T_ncomm2_nbody0(X, hbar, T, ref.gam1, ref.eta1, ref.lambdas, ref.orbspace)
    # print(f"time for zerobody {time.time() - _t0}")
    if herm:
        X['0'] *= 2.0
    # 1-body
    _t0 = time.time()
    X = H_T_ncomm1_nbody1(X, hbar, T, ref.gam1, ref.eta1, ref.lambdas, ref.orbspace)
    X = H_T_ncomm2_nbody1(X, hbar, T, ref.gam1, ref.eta1, ref.lambdas, ref.orbspace)
    # print(f"time for onebody {time.time() - _t0}")
    # 2-body
    _t0 = time.time()
    X = H_T_ncomm1_nbody2(X, hbar, T, ref.gam1, ref.eta1, ref.lambdas, ref.orbspace)
    X = H_T_ncomm2_nbody2(X, hbar, T, ref.gam1, ref.eta1, ref.lambdas, ref.orbspace)
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


def commutator_H_T1(o, o_old, T, ref, herm):
    # 0-body
    o = h_t_c0(o, o_old, T, ref.gam1, ref.eta1, ref.lambdas, ref.orbspace, scale=2.0 if herm else 1.0)
    # 1-body
    o = h1_t1_c1(o, o_old, T, ref.gam1, ref.eta1, ref.lambdas, ref.orbspace)
    o = h2_t1_c1(o, o_old, T, ref.gam1, ref.eta1, ref.lambdas, ref.orbspace)
    # 2-body
    _t0 = time.time()
    o = h2_t1_c2(o, o_old, T, ref.gam1, ref.eta1, ref.lambdas, ref.orbspace)
    # antisymmetrize twobody
    o['aa'] -= o['aa'].transpose(1, 0, 2, 3)
    o['aa'] -= o['aa'].transpose(0, 1, 3, 2)
    o['bb'] -= o['bb'].transpose(1, 0, 2, 3)
    o['bb'] -= o['bb'].transpose(0, 1, 3, 2)
    if herm:
        o['a'] += o['a'].T.conj()
        o['b'] += o['b'].T.conj()
        o['aa'] += o['aa'].transpose(2, 3, 0, 1)
        o['ab'] += o['ab'].transpose(2, 3, 0, 1)
        o['bb'] += o['bb'].transpose(2, 3, 0, 1)
    return o


def recursive_commutator(hamiltonian, T, ref, herm, max_ncomm):
    
    # Initial values
    o = {}
    o_old = {}
    hbar = {}
    for key in hamiltonian.keys():
        o[key] = np.zeros_like(hamiltonian[key])
        o_old[key] = hamiltonian[key].copy()
        hbar[key] = hamiltonian[key].copy()
    
    o['0'] = np.array([0.0])
    o_old['0'] = np.array([0.0])
    hbar['0'] = np.array([0.0])
    
    #
    ncomm = 0
    while ncomm < max_ncomm:
        o['0'] *= 0.0
        o = commutator_H_T1(o, o_old, T, ref, herm)
        # Increment commutator count
        ncomm += 1
        # Compute residual
        resid = 0.0
        for key, value in o.items():
            o[key] /= ncomm
            resid += np.linalg.norm(o[key].flatten())
            
        print(f"         E{ncomm} = {hbar['0'][0]:.10f}  |HBar(1)| = {resid:.10f}")
        
        if resid < 1e-12:
            break
        # Increment many-body components of Hbar
        for key, value in o.items():
            hbar[key] += value
        # Store old delta H
        for key, value in o.items():
            o_old[key] = o[key].copy()
        # reset delta H
        for key, value in o.items():
            o[key] = np.zeros_like(o_old[key])

    return hbar
