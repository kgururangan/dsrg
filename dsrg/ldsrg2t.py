import time
import numpy as np
from dsrg.utilities import regularized_denominator
from dsrg.wicked_contractions.ldsrg2t_contractions import *



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
    
    # lowest-order t3
    nua, nub, noa, nob = T['ab'].shape
    T['aaa'] = np.zeros((nua, nua, nua, noa, noa, noa))
    T['aab'] = np.zeros((nua, nua, nub, noa, noa, nob))
    T['abb'] = np.zeros((nua, nub, nub, noa, nob, nob))
    T['bbb'] = np.zeros((nub, nub, nub, nob, nob, nob))
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
    T['aaa'] = (hbar['aaa'][p, p, p, h, h, h] + T['aaa'] * denom['aaa']) * reg_denom['aaa']
    T['aaa'][pa, pa, pa, ha, ha, ha] = .0
    T['aab'] = (hbar['aab'][p, p, P, h, h, H] + T['aab'] * denom['aab']) * reg_denom['aab']
    T['aab'][pa, pa, pA, ha, ha, hA] = .0
    T['abb'] = (hbar['abb'][p, P, P, h, H, H] + T['abb'] * denom['abb']) * reg_denom['abb']
    T['abb'][pa, pA, pA, ha, hA, hA] = .0
    T['bbb'] = (hbar['bbb'][P, P, P, H, H, H] + T['bbb'] * denom['bbb']) * reg_denom['bbb']
    T['bbb'][pA, pA, pA, hA, hA, hA] = .0
    return T


def update_hbar(o, o_old, T, ref, herm):
    # 0-body (energy)
    o = h1a_t1a_c0(o, o_old, T, ref.gam1, ref.eta1, ref.lambdas, ref.orbspace,
                   scale=2.0 if herm else 1.0)
    o = h1b_t1b_c0(o, o_old, T, ref.gam1, ref.eta1, ref.lambdas, ref.orbspace,
                   scale=2.0 if herm else 1.0)
    o = h1a_t2a_c0(o, o_old, T, ref.gam1, ref.eta1, ref.lambdas, ref.orbspace,
                   scale=2.0 if herm else 1.0)
    o = h1a_t2b_c0(o, o_old, T, ref.gam1, ref.eta1, ref.lambdas, ref.orbspace,
                   scale=2.0 if herm else 1.0)
    o = h1b_t2b_c0(o, o_old, T, ref.gam1, ref.eta1, ref.lambdas, ref.orbspace,
                   scale=2.0 if herm else 1.0)
    o = h1b_t2c_c0(o, o_old, T, ref.gam1, ref.eta1, ref.lambdas, ref.orbspace,
                   scale=2.0 if herm else 1.0)
    o = h2a_t1a_c0(o, o_old, T, ref.gam1, ref.eta1, ref.lambdas, ref.orbspace,
                   scale=2.0 if herm else 1.0)
    o = h2b_t1a_c0(o, o_old, T, ref.gam1, ref.eta1, ref.lambdas, ref.orbspace,
                   scale=2.0 if herm else 1.0)
    o = h2b_t1b_c0(o, o_old, T, ref.gam1, ref.eta1, ref.lambdas, ref.orbspace,
                   scale=2.0 if herm else 1.0)
    o = h2c_t1b_c0(o, o_old, T, ref.gam1, ref.eta1, ref.lambdas, ref.orbspace,
                   scale=2.0 if herm else 1.0)
    o = h2a_t2a_c0(o, o_old, T, ref.gam1, ref.eta1, ref.lambdas, ref.orbspace,
                   scale=2.0 if herm else 1.0)
    o = h2a_t2b_c0(o, o_old, T, ref.gam1, ref.eta1, ref.lambdas, ref.orbspace,
                   scale=2.0 if herm else 1.0)
    o = h2b_t2a_c0(o, o_old, T, ref.gam1, ref.eta1, ref.lambdas, ref.orbspace,
                   scale=2.0 if herm else 1.0)
    o = h2b_t2b_c0(o, o_old, T, ref.gam1, ref.eta1, ref.lambdas, ref.orbspace,
                   scale=2.0 if herm else 1.0)
    o = h2b_t2c_c0(o, o_old, T, ref.gam1, ref.eta1, ref.lambdas, ref.orbspace,
                   scale=2.0 if herm else 1.0)
    o = h2c_t2b_c0(o, o_old, T, ref.gam1, ref.eta1, ref.lambdas, ref.orbspace,
                   scale=2.0 if herm else 1.0)
    o = h2c_t2c_c0(o, o_old, T, ref.gam1, ref.eta1, ref.lambdas, ref.orbspace,
                   scale=2.0 if herm else 1.0)
    o = h1a_t3a_c0(o, o_old, T, ref.gam1, ref.eta1, ref.lambdas, ref.orbspace,
                   scale=2.0 if herm else 1.0)
    o = h1a_t3b_c0(o, o_old, T, ref.gam1, ref.eta1, ref.lambdas, ref.orbspace,
                   scale=2.0 if herm else 1.0)
    o = h1a_t3c_c0(o, o_old, T, ref.gam1, ref.eta1, ref.lambdas, ref.orbspace,
                   scale=2.0 if herm else 1.0)
    o = h1b_t3b_c0(o, o_old, T, ref.gam1, ref.eta1, ref.lambdas, ref.orbspace,
                   scale=2.0 if herm else 1.0)
    o = h1b_t3c_c0(o, o_old, T, ref.gam1, ref.eta1, ref.lambdas, ref.orbspace,
                   scale=2.0 if herm else 1.0)
    o = h1b_t3d_c0(o, o_old, T, ref.gam1, ref.eta1, ref.lambdas, ref.orbspace,
                   scale=2.0 if herm else 1.0)
    o = h2a_t3a_c0(o, o_old, T, ref.gam1, ref.eta1, ref.lambdas, ref.orbspace,
                   scale=2.0 if herm else 1.0)
    o = h2a_t3b_c0(o, o_old, T, ref.gam1, ref.eta1, ref.lambdas, ref.orbspace,
                   scale=2.0 if herm else 1.0)
    o = h2a_t3c_c0(o, o_old, T, ref.gam1, ref.eta1, ref.lambdas, ref.orbspace,
                   scale=2.0 if herm else 1.0)
    o = h2b_t3a_c0(o, o_old, T, ref.gam1, ref.eta1, ref.lambdas, ref.orbspace,
                   scale=2.0 if herm else 1.0)
    o = h2b_t3b_c0(o, o_old, T, ref.gam1, ref.eta1, ref.lambdas, ref.orbspace,
                   scale=2.0 if herm else 1.0)
    o = h2b_t3c_c0(o, o_old, T, ref.gam1, ref.eta1, ref.lambdas, ref.orbspace,
                   scale=2.0 if herm else 1.0)
    o = h2b_t3d_c0(o, o_old, T, ref.gam1, ref.eta1, ref.lambdas, ref.orbspace,
                   scale=2.0 if herm else 1.0)
    o = h2c_t3b_c0(o, o_old, T, ref.gam1, ref.eta1, ref.lambdas, ref.orbspace,
                   scale=2.0 if herm else 1.0)
    o = h2c_t3c_c0(o, o_old, T, ref.gam1, ref.eta1, ref.lambdas, ref.orbspace,
                   scale=2.0 if herm else 1.0)
    o = h2c_t3d_c0(o, o_old, T, ref.gam1, ref.eta1, ref.lambdas, ref.orbspace,
                   scale=2.0 if herm else 1.0)
    ### onebody
    o = h1a_t1a_c1a(o, o_old, T, ref.gam1, ref.eta1, ref.lambdas, ref.orbspace)
    o = h1a_t2a_c1a(o, o_old, T, ref.gam1, ref.eta1, ref.lambdas, ref.orbspace)
    o = h1b_t2b_c1a(o, o_old, T, ref.gam1, ref.eta1, ref.lambdas, ref.orbspace)
    o = h2a_t1a_c1a(o, o_old, T, ref.gam1, ref.eta1, ref.lambdas, ref.orbspace)
    o = h2b_t1b_c1a(o, o_old, T, ref.gam1, ref.eta1, ref.lambdas, ref.orbspace)
    o = h2a_t2a_c1a(o, o_old, T, ref.gam1, ref.eta1, ref.lambdas, ref.orbspace)
    o = h2a_t2b_c1a(o, o_old, T, ref.gam1, ref.eta1, ref.lambdas, ref.orbspace)
    o = h2b_t2a_c1a(o, o_old, T, ref.gam1, ref.eta1, ref.lambdas, ref.orbspace)
    o = h2b_t2b_c1a(o, o_old, T, ref.gam1, ref.eta1, ref.lambdas, ref.orbspace)
    o = h2b_t2c_c1a(o, o_old, T, ref.gam1, ref.eta1, ref.lambdas, ref.orbspace)
    o = h2c_t2b_c1a(o, o_old, T, ref.gam1, ref.eta1, ref.lambdas, ref.orbspace)
    o = h1a_t3a_c1a(o, o_old, T, ref.gam1, ref.eta1, ref.lambdas, ref.orbspace)
    o = h1a_t3b_c1a(o, o_old, T, ref.gam1, ref.eta1, ref.lambdas, ref.orbspace)
    o = h1b_t3b_c1a(o, o_old, T, ref.gam1, ref.eta1, ref.lambdas, ref.orbspace)
    o = h1b_t3c_c1a(o, o_old, T, ref.gam1, ref.eta1, ref.lambdas, ref.orbspace)
    o = h2a_t3a_c1a(o, o_old, T, ref.gam1, ref.eta1, ref.lambdas, ref.orbspace)
    o = h2a_t3b_c1a(o, o_old, T, ref.gam1, ref.eta1, ref.lambdas, ref.orbspace)
    o = h2a_t3c_c1a(o, o_old, T, ref.gam1, ref.eta1, ref.lambdas, ref.orbspace)
    o = h2b_t3a_c1a(o, o_old, T, ref.gam1, ref.eta1, ref.lambdas, ref.orbspace)
    o = h2b_t3b_c1a(o, o_old, T, ref.gam1, ref.eta1, ref.lambdas, ref.orbspace)
    o = h2b_t3c_c1a(o, o_old, T, ref.gam1, ref.eta1, ref.lambdas, ref.orbspace)
    o = h2b_t3d_c1a(o, o_old, T, ref.gam1, ref.eta1, ref.lambdas, ref.orbspace)
    o = h2c_t3b_c1a(o, o_old, T, ref.gam1, ref.eta1, ref.lambdas, ref.orbspace)
    o = h2c_t3c_c1a(o, o_old, T, ref.gam1, ref.eta1, ref.lambdas, ref.orbspace)
    o = h1b_t1b_c1b(o, o_old, T, ref.gam1, ref.eta1, ref.lambdas, ref.orbspace)
    o = h1b_t2c_c1b(o, o_old, T, ref.gam1, ref.eta1, ref.lambdas, ref.orbspace)
    o = h1a_t2b_c1b(o, o_old, T, ref.gam1, ref.eta1, ref.lambdas, ref.orbspace)
    o = h2c_t1b_c1b(o, o_old, T, ref.gam1, ref.eta1, ref.lambdas, ref.orbspace)
    o = h2b_t1a_c1b(o, o_old, T, ref.gam1, ref.eta1, ref.lambdas, ref.orbspace)
    o = h2c_t2c_c1b(o, o_old, T, ref.gam1, ref.eta1, ref.lambdas, ref.orbspace)
    o = h2c_t2b_c1b(o, o_old, T, ref.gam1, ref.eta1, ref.lambdas, ref.orbspace)
    o = h2b_t2c_c1b(o, o_old, T, ref.gam1, ref.eta1, ref.lambdas, ref.orbspace)
    o = h2b_t2b_c1b(o, o_old, T, ref.gam1, ref.eta1, ref.lambdas, ref.orbspace)
    o = h2b_t2a_c1b(o, o_old, T, ref.gam1, ref.eta1, ref.lambdas, ref.orbspace)
    o = h2a_t2b_c1b(o, o_old, T, ref.gam1, ref.eta1, ref.lambdas, ref.orbspace)
    o = h1a_t3b_c1b(o, o_old, T, ref.gam1, ref.eta1, ref.lambdas, ref.orbspace)
    o = h1a_t3c_c1b(o, o_old, T, ref.gam1, ref.eta1, ref.lambdas, ref.orbspace)
    o = h1b_t3c_c1b(o, o_old, T, ref.gam1, ref.eta1, ref.lambdas, ref.orbspace)
    o = h1b_t3d_c1b(o, o_old, T, ref.gam1, ref.eta1, ref.lambdas, ref.orbspace)
    o = h2a_t3b_c1b(o, o_old, T, ref.gam1, ref.eta1, ref.lambdas, ref.orbspace)
    o = h2a_t3c_c1b(o, o_old, T, ref.gam1, ref.eta1, ref.lambdas, ref.orbspace)
    o = h2b_t3a_c1b(o, o_old, T, ref.gam1, ref.eta1, ref.lambdas, ref.orbspace)
    o = h2b_t3b_c1b(o, o_old, T, ref.gam1, ref.eta1, ref.lambdas, ref.orbspace)
    o = h2b_t3c_c1b(o, o_old, T, ref.gam1, ref.eta1, ref.lambdas, ref.orbspace)
    o = h2b_t3d_c1b(o, o_old, T, ref.gam1, ref.eta1, ref.lambdas, ref.orbspace)
    o = h2c_t3b_c1b(o, o_old, T, ref.gam1, ref.eta1, ref.lambdas, ref.orbspace)
    o = h2c_t3c_c1b(o, o_old, T, ref.gam1, ref.eta1, ref.lambdas, ref.orbspace)
    o = h2c_t3d_c1b(o, o_old, T, ref.gam1, ref.eta1, ref.lambdas, ref.orbspace)
    ### twobody
    o = h1a_t2a_c2a(o, o_old, T, ref.gam1, ref.eta1, ref.lambdas, ref.orbspace)
    o = h2a_t1a_c2a(o, o_old, T, ref.gam1, ref.eta1, ref.lambdas, ref.orbspace)
    o = h2a_t2a_c2a(o, o_old, T, ref.gam1, ref.eta1, ref.lambdas, ref.orbspace)
    o = h2b_t2b_c2a(o, o_old, T, ref.gam1, ref.eta1, ref.lambdas, ref.orbspace)
    o = h1a_t3a_c2a(o, o_old, T, ref.gam1, ref.eta1, ref.lambdas, ref.orbspace)
    o = h1b_t3b_c2a(o, o_old, T, ref.gam1, ref.eta1, ref.lambdas, ref.orbspace)
    o = h2a_t3a_c2a(o, o_old, T, ref.gam1, ref.eta1, ref.lambdas, ref.orbspace)
    o = h2a_t3b_c2a(o, o_old, T, ref.gam1, ref.eta1, ref.lambdas, ref.orbspace)
    o = h2b_t3a_c2a(o, o_old, T, ref.gam1, ref.eta1, ref.lambdas, ref.orbspace)
    o = h2b_t3b_c2a(o, o_old, T, ref.gam1, ref.eta1, ref.lambdas, ref.orbspace)
    o = h2b_t3c_c2a(o, o_old, T, ref.gam1, ref.eta1, ref.lambdas, ref.orbspace)
    o = h2c_t3b_c2a(o, o_old, T, ref.gam1, ref.eta1, ref.lambdas, ref.orbspace)
    o = h1a_t2b_c2b(o, o_old, T, ref.gam1, ref.eta1, ref.lambdas, ref.orbspace)
    o = h1b_t2b_c2b(o, o_old, T, ref.gam1, ref.eta1, ref.lambdas, ref.orbspace)
    o = h2b_t1a_c2b(o, o_old, T, ref.gam1, ref.eta1, ref.lambdas, ref.orbspace)
    o = h2b_t1b_c2b(o, o_old, T, ref.gam1, ref.eta1, ref.lambdas, ref.orbspace)
    o = h2a_t2b_c2b(o, o_old, T, ref.gam1, ref.eta1, ref.lambdas, ref.orbspace)
    o = h2b_t2a_c2b(o, o_old, T, ref.gam1, ref.eta1, ref.lambdas, ref.orbspace)
    o = h2b_t2b_c2b(o, o_old, T, ref.gam1, ref.eta1, ref.lambdas, ref.orbspace)
    o = h2b_t2c_c2b(o, o_old, T, ref.gam1, ref.eta1, ref.lambdas, ref.orbspace)
    o = h2c_t2b_c2b(o, o_old, T, ref.gam1, ref.eta1, ref.lambdas, ref.orbspace)
    o = h1a_t3b_c2b(o, o_old, T, ref.gam1, ref.eta1, ref.lambdas, ref.orbspace)
    o = h1b_t3c_c2b(o, o_old, T, ref.gam1, ref.eta1, ref.lambdas, ref.orbspace)
    o = h2a_t3b_c2b(o, o_old, T, ref.gam1, ref.eta1, ref.lambdas, ref.orbspace)
    o = h2a_t3c_c2b(o, o_old, T, ref.gam1, ref.eta1, ref.lambdas, ref.orbspace)
    o = h2b_t3a_c2b(o, o_old, T, ref.gam1, ref.eta1, ref.lambdas, ref.orbspace)
    o = h2b_t3b_c2b(o, o_old, T, ref.gam1, ref.eta1, ref.lambdas, ref.orbspace)
    o = h2b_t3c_c2b(o, o_old, T, ref.gam1, ref.eta1, ref.lambdas, ref.orbspace)
    o = h2b_t3d_c2b(o, o_old, T, ref.gam1, ref.eta1, ref.lambdas, ref.orbspace)
    o = h2c_t3b_c2b(o, o_old, T, ref.gam1, ref.eta1, ref.lambdas, ref.orbspace)
    o = h2c_t3c_c2b(o, o_old, T, ref.gam1, ref.eta1, ref.lambdas, ref.orbspace)
    o = h1b_t2c_c2c(o, o_old, T, ref.gam1, ref.eta1, ref.lambdas, ref.orbspace)
    o = h2c_t1b_c2c(o, o_old, T, ref.gam1, ref.eta1, ref.lambdas, ref.orbspace)
    o = h2c_t2c_c2c(o, o_old, T, ref.gam1, ref.eta1, ref.lambdas, ref.orbspace)
    o = h2b_t2b_c2c(o, o_old, T, ref.gam1, ref.eta1, ref.lambdas, ref.orbspace)
    o = h1a_t3c_c2c(o, o_old, T, ref.gam1, ref.eta1, ref.lambdas, ref.orbspace)
    o = h1b_t3d_c2c(o, o_old, T, ref.gam1, ref.eta1, ref.lambdas, ref.orbspace)
    o = h2a_t3c_c2c(o, o_old, T, ref.gam1, ref.eta1, ref.lambdas, ref.orbspace)
    o = h2b_t3b_c2c(o, o_old, T, ref.gam1, ref.eta1, ref.lambdas, ref.orbspace)
    o = h2b_t3c_c2c(o, o_old, T, ref.gam1, ref.eta1, ref.lambdas, ref.orbspace)
    o = h2b_t3d_c2c(o, o_old, T, ref.gam1, ref.eta1, ref.lambdas, ref.orbspace)
    o = h2c_t3c_c2c(o, o_old, T, ref.gam1, ref.eta1, ref.lambdas, ref.orbspace)
    o = h2c_t3d_c2c(o, o_old, T, ref.gam1, ref.eta1, ref.lambdas, ref.orbspace)
    # 3-body
    o = h2a_t2a_c3a(o, o_old, T, ref.gam1, ref.eta1, ref.lambdas, ref.orbspace)
    o = h2a_t2b_c3b(o, o_old, T, ref.gam1, ref.eta1, ref.lambdas, ref.orbspace)
    o = h2b_t2a_c3b(o, o_old, T, ref.gam1, ref.eta1, ref.lambdas, ref.orbspace)
    o = h2b_t2b_c3b(o, o_old, T, ref.gam1, ref.eta1, ref.lambdas, ref.orbspace)
    o = h2b_t2b_c3c(o, o_old, T, ref.gam1, ref.eta1, ref.lambdas, ref.orbspace)
    o = h2b_t2c_c3c(o, o_old, T, ref.gam1, ref.eta1, ref.lambdas, ref.orbspace)
    o = h2c_t2b_c3c(o, o_old, T, ref.gam1, ref.eta1, ref.lambdas, ref.orbspace)
    o = h2c_t2c_c3d(o, o_old, T, ref.gam1, ref.eta1, ref.lambdas, ref.orbspace)
    #
    o = h1a_t3a_c3a(o, o_old, T, ref.gam1, ref.eta1, ref.lambdas, ref.orbspace)
    o = h2a_t3a_c3a(o, o_old, T, ref.gam1, ref.eta1, ref.lambdas, ref.orbspace)
    o = h2b_t3b_c3a(o, o_old, T, ref.gam1, ref.eta1, ref.lambdas, ref.orbspace)
    o = h1a_t3b_c3b(o, o_old, T, ref.gam1, ref.eta1, ref.lambdas, ref.orbspace)
    o = h1b_t3b_c3b(o, o_old, T, ref.gam1, ref.eta1, ref.lambdas, ref.orbspace)
    o = h2a_t3b_c3b(o, o_old, T, ref.gam1, ref.eta1, ref.lambdas, ref.orbspace)
    o = h2b_t3a_c3b(o, o_old, T, ref.gam1, ref.eta1, ref.lambdas, ref.orbspace)
    o = h2b_t3b_c3b(o, o_old, T, ref.gam1, ref.eta1, ref.lambdas, ref.orbspace)
    o = h2b_t3c_c3b(o, o_old, T, ref.gam1, ref.eta1, ref.lambdas, ref.orbspace)
    o = h2c_t3b_c3b(o, o_old, T, ref.gam1, ref.eta1, ref.lambdas, ref.orbspace)
    o = h1a_t3c_c3c(o, o_old, T, ref.gam1, ref.eta1, ref.lambdas, ref.orbspace)
    o = h1b_t3c_c3c(o, o_old, T, ref.gam1, ref.eta1, ref.lambdas, ref.orbspace)
    o = h2a_t3c_c3c(o, o_old, T, ref.gam1, ref.eta1, ref.lambdas, ref.orbspace)
    o = h2b_t3b_c3c(o, o_old, T, ref.gam1, ref.eta1, ref.lambdas, ref.orbspace)
    o = h2b_t3c_c3c(o, o_old, T, ref.gam1, ref.eta1, ref.lambdas, ref.orbspace)
    o = h2b_t3d_c3c(o, o_old, T, ref.gam1, ref.eta1, ref.lambdas, ref.orbspace)
    o = h2c_t3c_c3c(o, o_old, T, ref.gam1, ref.eta1, ref.lambdas, ref.orbspace)
    o = h1b_t3d_c3d(o, o_old, T, ref.gam1, ref.eta1, ref.lambdas, ref.orbspace)
    o = h2b_t3c_c3d(o, o_old, T, ref.gam1, ref.eta1, ref.lambdas, ref.orbspace)
    o = h2c_t3d_c3d(o, o_old, T, ref.gam1, ref.eta1, ref.lambdas, ref.orbspace)

    # antisymmetrize
    o['aa'] -= o['aa'].transpose(1, 0, 2, 3)
    o['aa'] -= o['aa'].transpose(0, 1, 3, 2)
    o['bb'] -= o['bb'].transpose(1, 0, 2, 3)
    o['bb'] -= o['bb'].transpose(0, 1, 3, 2)

    o['aaa'] -= o['aaa'].transpose(0, 2, 1, 3, 4, 5)  # (bc)
    o['aaa'] -= o['aaa'].transpose(1, 0, 2, 3, 4, 5) + o['aaa'].transpose(2, 1, 0, 3, 4, 5)  # (a/bc)
    o['aaa'] -= o['aaa'].transpose(0, 1, 2, 3, 5, 4)  # (jk)
    o['aaa'] -= o['aaa'].transpose(0, 1, 2, 4, 3, 5) + o['aaa'].transpose(0, 1, 2, 5, 4, 3)  # (i/jk)

    o['aab'] -= o['aab'].transpose(1, 0, 2, 3, 4, 5)  # (ab)
    o['aab'] -= o['aab'].transpose(0, 1, 2, 4, 3, 5)  # (ij)

    o['abb'] -= o['abb'].transpose(0, 2, 1, 3, 4, 5)  # (bc)
    o['abb'] -= o['abb'].transpose(0, 1, 2, 3, 5, 4)  # (jk)

    o['bbb'] -= o['bbb'].transpose(0, 2, 1, 3, 4, 5)  # (bc)
    o['bbb'] -= o['bbb'].transpose(1, 0, 2, 3, 4, 5) + o['bbb'].transpose(2, 1, 0, 3, 4, 5)  # (a/bc)
    o['bbb'] -= o['bbb'].transpose(0, 1, 2, 3, 5, 4)  # (jk)
    o['bbb'] -= o['bbb'].transpose(0, 1, 2, 4, 3, 5) + o['bbb'].transpose(0, 1, 2, 5, 4, 3)  # (i/jk)

    if herm:
        o['a'] += o['a'].T.conj()
        o['b'] += o['b'].T.conj()
        o['aa'] += o['aa'].transpose(2, 3, 0, 1)
        o['ab'] += o['ab'].transpose(2, 3, 0, 1)
        o['bb'] += o['bb'].transpose(2, 3, 0, 1)
        o['aaa'] += o['aaa'].transpose(3, 4, 5, 0, 1, 2)
        o['aab'] += o['aab'].transpose(3, 4, 5, 0, 1, 2)
        o['abb'] += o['abb'].transpose(3, 4, 5, 0, 1, 2)
        o['bbb'] += o['bbb'].transpose(3, 4, 5, 0, 1, 2)
    return o