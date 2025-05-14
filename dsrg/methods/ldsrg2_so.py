import time
import numpy as np
from dsrg.utilities import regularized_denominator
from dsrg.wicked_contractions.ldsrg2_so_contractions import *



def build_denominators(s, eps, ref):

    n = np.newaxis
    h = ref.orbspace['hole']
    p = ref.orbspace['particle']

    denom = {'1': eps[n, h] - eps[p, n],
             '2': eps[n, n, h, n] + eps[n, n, n, h] - eps[p, n, n, n] - eps[n, p, n, n]}

    reg_denom = {}
    for key, value in denom.items():
        reg_denom[key] = regularized_denominator(value, s)

    return denom, reg_denom


def initial_guess(ref, denom, reg_denom):
    # Slicing
    h = ref.orbspace['hole']
    p = ref.orbspace['particle']

    ha = ref.orbspace['hole_active']
    pa = ref.orbspace['particle_active']

    T = {}

    # 1st-order t2
    T['2'] = ref.V[p, p, h, h] * reg_denom['2']
    T['2'][pa, pa, ha, ha] *= 0.

    # 1st-order t1
    T['1'] = (
            ref.F[p, h]
            + np.einsum("axiu,ux,ux->ai", T['2'][:, pa, :, ha], denom['1'][pa, ha], ref.gam1,
                        optimize=True)
    )
    T['1'] *= reg_denom['1']
    T['1'][pa, ha] *= 0.
    return T


def update_t(T, hbar, ref, denom, reg_denom):
    # Slicing
    h = ref.orbspace['hole']
    p = ref.orbspace['particle']
    
    ha = ref.orbspace['hole_active']
    pa = ref.orbspace['particle_active']

    T['1'] = (hbar['1'][p, h] + T['1'] * denom['1']) * reg_denom['1']
    T['1'][pa, ha] = .0

    T['2'] = (hbar['2'][p, p, h, h] + T['2'] * denom['2']) * reg_denom['2']
    T['2'][pa, pa, ha, ha] = .0
    
    return T


def update_hbar(o, o_old, T, ref, herm):
    o0 = o['0']
    o1 = o['1']
    o2 = o['2']
    
    o1_old = o_old['1']
    o2_old = o_old['2']
    
    t1 = T['1']
    t2 = T['2']
    # zerobody
    # _t0 = time.time()
    o0 = h1_t1_c0(o0, (o1_old, o2_old), (t1, t2), ref.gam1, ref.eta1, ref.lambdas,
                  ref.orbspace, scale=2.0 if herm else 1.0)
    o0 = h1_t2_c0(o0, (o1_old, o2_old), (t1, t2), ref.gam1, ref.eta1, ref.lambdas,
                  ref.orbspace, scale=2.0 if herm else 1.0)
    o0 = h2_t1_c0(o0, (o1_old, o2_old), (t1, t2), ref.gam1, ref.eta1, ref.lambdas,
                  ref.orbspace, scale=2.0 if herm else 1.0)
    o0 = h2_t2_c0(o0, (o1_old, o2_old), (t1, t2), ref.gam1, ref.eta1, ref.lambdas,
                  ref.orbspace, scale=2.0 if herm else 1.0)
    # print(f"energy took {time.time() - _t0}")
    # onebody
    # _t0 = time.time()
    o1 = h1_t1_c1(o1, (o1_old, o2_old), (t1, t2), ref.gam1, ref.eta1, ref.lambdas,
                  ref.orbspace)
    o1 = h1_t2_c1(o1, (o1_old, o2_old), (t1, t2), ref.gam1, ref.eta1, ref.lambdas,
                  ref.orbspace)
    o1 = h2_t1_c1(o1, (o1_old, o2_old), (t1, t2), ref.gam1, ref.eta1, ref.lambdas,
                  ref.orbspace)
    o1 = h2_t2_c1(o1, (o1_old, o2_old), (t1, t2), ref.gam1, ref.eta1, ref.lambdas,
                  ref.orbspace)
    # print(f"onebody took {time.time() - _t0}")
    # twobody
    # _t0 = time.time()
    o2 = h1_t2_c2(o2, (o1_old, o2_old), (t1, t2), ref.gam1, ref.eta1, ref.lambdas,
                  ref.orbspace)
    o2 = h2_t1_c2(o2, (o1_old, o2_old), (t1, t2), ref.gam1, ref.eta1, ref.lambdas,
                  ref.orbspace)
    o2 = h2_t2_c2(o2, (o1_old, o2_old), (t1, t2), ref.gam1, ref.eta1, ref.lambdas,
                  ref.orbspace)
    # print(f"twobody took {time.time() - _t0}")
    # antisymmetrize twobody
    o2 -= o2.transpose(1, 0, 2, 3)
    o2 -= o2.transpose(0, 1, 3, 2)
    if herm:
        o1 += o1.T.conj()
        o2 += o2.transpose(2, 3, 0, 1)

    o['0'] = o0
    o['1'] = o1
    o['2'] = o2
    return o
