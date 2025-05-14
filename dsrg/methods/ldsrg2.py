import time
import numpy as np
from dsrg.utilities import regularized_denominator
from dsrg.wicked_contractions.ldsrg2_contractions import *



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


def update_t(T, hbar, ref, denom, reg_denom):
    # Slicing
    h = ref.orbspace['hole_alpha']
    p = ref.orbspace['particle_alpha']
    H = ref.orbspace['hole_beta']
    P = ref.orbspace['particle_beta']
    
    ha = ref.orbspace['hole_active_alpha']
    pa = ref.orbspace['particle_active_alpha']
    hA = ref.orbspace['hole_active_beta']
    pA = ref.orbspace['particle_active_beta']
    
    T['a'] = (hbar['a'][p, h] + T['a'] * denom['a']) * reg_denom['a']
    T['a'][pa, ha] = .0
    T['b'] = (hbar['b'][P, H] + T['b'] * denom['b']) * reg_denom['b']
    T['b'][pA, hA] = .0
    T['aa'] = (hbar['aa'][p, p, h, h] + T['aa'] * denom['aa']) * reg_denom['aa']
    T['aa'][pa, pa, ha, ha] = .0
    T['ab'] = (hbar['ab'][p, P, h, H] + T['ab'] * denom['ab']) * reg_denom['ab']
    T['ab'][pa, pA, ha, hA] = .0
    T['bb'] = (hbar['bb'][P, P, h, H] + T['bb'] * denom['bb']) * reg_denom['bb']
    T['bb'][pA, pA, hA, hA] = .0
    
    return T


def update_hbar(o, o_old, T, ref, herm):
    # 0-body (energy)
    _t0 = time.time()
    o = h2a_t2a_c0(o, o_old, T, ref.gam1, ref.eta1, ref.lambdas, ref.orbspace, scale=2.0 if herm else 1.0)
    o = h2a_t2b_c0(o, o_old, T, ref.gam1, ref.eta1, ref.lambdas, ref.orbspace, scale=2.0 if herm else 1.0)
    o = h2b_t2a_c0(o, o_old, T, ref.gam1, ref.eta1, ref.lambdas, ref.orbspace, scale=2.0 if herm else 1.0)
    o = h2b_t2b_c0(o, o_old, T, ref.gam1, ref.eta1, ref.lambdas, ref.orbspace, scale=2.0 if herm else 1.0)
    o = h2b_t2c_c0(o, o_old, T, ref.gam1, ref.eta1, ref.lambdas, ref.orbspace, scale=2.0 if herm else 1.0)
    o = h2c_t2b_c0(o, o_old, T, ref.gam1, ref.eta1, ref.lambdas, ref.orbspace, scale=2.0 if herm else 1.0)
    o = h2c_t2c_c0(o, o_old, T, ref.gam1, ref.eta1, ref.lambdas, ref.orbspace, scale=2.0 if herm else 1.0)
    o = h_t_c0(o, o_old, T, ref.gam1, ref.eta1, ref.lambdas, ref.orbspace, scale=2.0 if herm else 1.0)
    # print(f"time for zerobody {time.time() - _t0}")
    # 1-body
    _t0 = time.time()
    o = h1_t1_c1(o, o_old, T, ref.gam1, ref.eta1, ref.lambdas, ref.orbspace)
    o = h1_t2_c1(o, o_old, T, ref.gam1, ref.eta1, ref.lambdas, ref.orbspace)
    o = h2_t1_c1(o, o_old, T, ref.gam1, ref.eta1, ref.lambdas, ref.orbspace)
    o = h2_t2_c1(o, o_old, T, ref.gam1, ref.eta1, ref.lambdas, ref.orbspace)
    # print(f"time for onebody {time.time() - _t0}")
    # 2-body
    _t0 = time.time()
    o = h1_t2_c2(o, o_old, T, ref.gam1, ref.eta1, ref.lambdas, ref.orbspace)
    o = h2_t1_c2(o, o_old, T, ref.gam1, ref.eta1, ref.lambdas, ref.orbspace)
    o = h2_t2_c2(o, o_old, T, ref.gam1, ref.eta1, ref.lambdas, ref.orbspace)
    # print(f"time for twobody {time.time() - _t0}")
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