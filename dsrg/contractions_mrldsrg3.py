import time
import numpy as np

def h_t_c0(O, h, t, gamma1, eta1, lambdas, orbspace, verbose=False, scale=1.0):
    c = orbspace['core_alpha']
    C = orbspace['core_beta']
    a = orbspace['active_alpha']
    A = orbspace['active_beta']
    v = orbspace['virt_alpha']
    V = orbspace['virt_beta']
    hc = orbspace['hole_core_alpha']
    hC = orbspace['hole_core_beta']
    ha = orbspace['hole_active_alpha']
    hA = orbspace['hole_active_beta']
    pa = orbspace['particle_active_alpha']
    pA = orbspace['particle_active_beta']
    pv = orbspace['particle_virt_alpha']
    pV = orbspace['particle_virt_beta']
    #
    hole_a = orbspace['hole_alpha']
    part_a = orbspace['particle_alpha']
    hole_b = orbspace['hole_beta']
    part_b = orbspace['particle_beta']

    # [H1, T1]
    O['0'] += scale * (
            np.einsum("me,em->", h['a'][c, v], t['a'][pv, hc], optimize=True)
            + np.einsum("xe,ey,xy->", h['a'][a, v], t['a'][pv, ha], gamma1['a'], optimize=True)
            + np.einsum("mx,ym,yx->", h['a'][c, a], t['a'][pa, hc], eta1['a'], optimize=True)
    )
    O['0'] += scale * (
            np.einsum("me,em->", h['b'][C, V], t['b'][pV, hC], optimize=True)
            + np.einsum("xe,ey,xy->", h['b'][A, V], t['b'][pV, hA], gamma1['b'], optimize=True)
            + np.einsum("mx,ym,yx->", h['b'][C, A], t['b'][pA, hC], eta1['b'], optimize=True)
    )

    # [H1, T2] + [H2, T1]
    temp = (
            np.einsum("xe,eyuv->xyuv", h['a'][a, v], t['aa'][pv, pa, ha, ha], optimize=True)
            - np.einsum("mv,xyum->xyuv", h['a'][c, a], t['aa'][pa, pa, ha, hc], optimize=True)
            + np.einsum("xyev,eu->xyuv", h['aa'][a, a, v, a], t['a'][pv, ha], optimize=True)
            - np.einsum("myuv,xm->xyuv", h['aa'][c, a, a, a], t['a'][pa, hc], optimize=True)
    )
    O['0'] += 0.5 * scale * np.einsum("xyuv,xyuv->", temp, lambdas['aa'], optimize=True) 

    temp = (
            np.einsum("xe,eyuv->xyuv", h['b'][A, V], t['bb'][pV, pA, hA, hA], optimize=True)
            - np.einsum("mv,xyum->xyuv", h['b'][C, A], t['bb'][pA, pA, hA, hC], optimize=True)
            + np.einsum("xyev,eu->xyuv", h['bb'][A, A, V, A], t['b'][pV, hA], optimize=True)
            - np.einsum("myuv,xm->xyuv", h['bb'][C, A, A, A], t['b'][pA, hC], optimize=True)
    )
    O['0'] += 0.5 * scale * np.einsum("xyuv,xyuv->", temp, lambdas['bb'], optimize=True)

    temp = (
            np.einsum("xe,eYuV->xYuV", h['a'][a, v], t['ab'][pv, pA, ha, hA], optimize=True)
            + np.einsum("YE,xEuV->xYuV", h['b'][A, V], t['ab'][pa, pV, ha, hA], optimize=True)
            - np.einsum("MV,xYuM->xYuV", h['b'][C, A], t['ab'][pa, pA, ha, hC], optimize=True)
            - np.einsum("mu,xYmV->xYuV", h['a'][c, a], t['ab'][pa, pA, hc, hA], optimize=True)
            + np.einsum("xYeV,eu->xYuV", h['ab'][a, A, v, A], t['a'][pv, ha], optimize=True)
            + np.einsum("xYuE,EV->xYuV", h['ab'][a, A, a, V], t['b'][pV, hA], optimize=True)
            - np.einsum("mYuV,xm->xYuV", h['ab'][c, A, a, A], t['a'][pa, hc], optimize=True)
            - np.einsum("xMuV,YM->xYuV", h['ab'][a, C, a, A], t['b'][pA, hC], optimize=True)

    )
    O['0'] += scale * np.einsum("xYuV,xYuV->", temp, lambdas['ab'], optimize=True)

    return O

def h1_t1_c1(O, h, t, gamma1, eta1, lambdas, orbspace, verbose=False, scale=1.0):
    t0 = time.time()
    c = orbspace['core_alpha']
    C = orbspace['core_beta']
    a = orbspace['active_alpha']
    A = orbspace['active_beta']
    v = orbspace['virt_alpha']
    V = orbspace['virt_beta']
    hc = orbspace['hole_core_alpha']
    hC = orbspace['hole_core_beta']
    ha = orbspace['hole_active_alpha']
    hA = orbspace['hole_active_beta']
    pa = orbspace['particle_active_alpha']
    pA = orbspace['particle_active_beta']
    pv = orbspace['particle_virt_alpha']
    pV = orbspace['particle_virt_beta']
    #
    hole_a = orbspace['hole_alpha']
    part_a = orbspace['particle_alpha']
    hole_b = orbspace['hole_beta']
    part_b = orbspace['particle_beta']

    O['a'][:, hole_a] += scale * np.einsum("pa,ai->pi", h['a'][:, part_a], t['a'], optimize=True)
    O['a'][part_a, :] -= scale * np.einsum("iq,ai->aq", h['a'][hole_a, :], t['a'], optimize=True)

    O['b'][:, hole_b] += scale * np.einsum("PA,AI->PI", h['b'][:, part_b], t['b'], optimize=True)
    O['b'][part_b, :] -= scale * np.einsum("IQ,AI->AQ", h['b'][hole_b, :], t['b'], optimize=True)
    return O

def h1_t2_c1(O, h, t, gamma1, eta1, lambdas, orbspace, verbose=False, scale=1.0):
    t0 = time.time()
    c = orbspace['core_alpha']
    C = orbspace['core_beta']
    a = orbspace['active_alpha']
    A = orbspace['active_beta']
    v = orbspace['virt_alpha']
    V = orbspace['virt_beta']
    hc = orbspace['hole_core_alpha']
    hC = orbspace['hole_core_beta']
    ha = orbspace['hole_active_alpha']
    hA = orbspace['hole_active_beta']
    pa = orbspace['particle_active_alpha']
    pA = orbspace['particle_active_beta']
    pv = orbspace['particle_virt_alpha']
    pV = orbspace['particle_virt_beta']
    #
    hole_a = orbspace['hole_alpha']
    part_a = orbspace['particle_alpha']
    hole_b = orbspace['hole_beta']
    part_b = orbspace['particle_beta']
    # a
    O['a'][part_a, hole_a] += scale * np.einsum("mb,abim->ai", h['a'][c, part_a], t['aa'][:, :, :, c], optimize=True)
    O['a'][part_a, hole_a] += scale * np.einsum("ub,uv,abiv->ai", h['a'][a, part_a], gamma1['a'], t['aa'][:, :, :, ha], optimize=True)
    O['a'][part_a, hole_a] -= scale * np.einsum("jv,uv,auij->ai", h['a'][hole_a, a], gamma1['a'], t['aa'][:, pa, :, :], optimize=True)
    O['a'][part_a, hole_a] += scale * np.einsum("MB,aBiM->ai", h['b'][C, part_b], t['ab'][:, :, :, hC], optimize=True)
    O['a'][part_a, hole_a] += scale * np.einsum("UB,UV,aBiV->ai", h['b'][A, part_b], gamma1['b'], t['ab'][:, :, :, hA], optimize=True)
    O['a'][part_a, hole_a] -= scale * np.einsum("JV,UV,aUiJ->ai", h['b'][hole_b, A], gamma1['b'], t['ab'][:, pA, :, :], optimize=True)
    # b
    O['b'][part_b, hole_b] += scale * np.einsum("mb,bAmI->AI", h['a'][c, part_a], t['ab'][:, :, hc, :], optimize=True)
    O['b'][part_b, hole_b] += scale * np.einsum("ub,uv,bAvI->AI", h['a'][a, part_a], gamma1['a'], t['ab'][:, :, ha, :], optimize=True)
    O['b'][part_b, hole_b] -= scale * np.einsum("jv,uv,uAjI->AI", h['a'][hole_a, a], gamma1['a'], t['ab'][pa, :, :, :], optimize=True)
    O['b'][part_b, hole_b] += scale * np.einsum("mb,abim->ai", h['b'][C, part_b], t['bb'][:, :, :, C], optimize=True)
    O['b'][part_b, hole_b] += scale * np.einsum("ub,uv,abiv->ai", h['b'][A, part_b], gamma1['b'], t['bb'][:, :, :, hA], optimize=True)
    O['b'][part_b, hole_b] -= scale * np.einsum("jv,uv,auij->ai", h['b'][hole_b, A], gamma1['b'], t['bb'][:, pA, :, :], optimize=True)
    return O

def h2_t1_c1(O, h, t, gamma1, eta1, lambdas, orbspace, verbose=False, scale=1.0):
    t0 = time.time()
    c = orbspace['core_alpha']
    C = orbspace['core_beta']
    a = orbspace['active_alpha']
    A = orbspace['active_beta']
    v = orbspace['virt_alpha']
    V = orbspace['virt_beta']
    hc = orbspace['hole_core_alpha']
    hC = orbspace['hole_core_beta']
    ha = orbspace['hole_active_alpha']
    hA = orbspace['hole_active_beta']
    pa = orbspace['particle_active_alpha']
    pA = orbspace['particle_active_beta']
    pv = orbspace['particle_virt_alpha']
    pV = orbspace['particle_virt_beta']
    #
    hole_a = orbspace['hole_alpha']
    part_a = orbspace['particle_alpha']
    hole_b = orbspace['hole_beta']
    part_b = orbspace['particle_beta']

    # a
    O['a'] += scale * np.einsum("pmqa,am->pq", h['aa'][:, c, :, part_a], t['a'][:, hc], optimize=True)
    O['a'] += scale * np.einsum("pyqe,yx,ex->pq", h['aa'][:, a, :, v], gamma1['a'], t['a'][pv, ha], optimize=True)
    O['a'] -= scale * np.einsum("pmqv,uv,um->pq", h['aa'][:, c, :, a], gamma1['a'], t['a'][pa, hc], optimize=True)
    O['a'] += scale * np.einsum("pMqA,AM->pq", h['ab'][:, C, :, part_b], t['b'][:, hC], optimize=True)
    O['a'] += scale * np.einsum("pYqE,YX,EX->pq", h['ab'][:, A, :, V], gamma1['b'], t['b'][pV, hA], optimize=True)
    O['a'] -= scale * np.einsum("pMqV,UV,UM->pq", h['ab'][:, C, :, A], gamma1['b'], t['b'][pA, hC], optimize=True)
    # b
    O['b'] += scale * np.einsum("mPaQ,am->PQ", h['ab'][c, :, part_a, :], t['a'][:, hc], optimize=True)
    O['b'] += scale * np.einsum("yPeQ,yx,ex->PQ", h['ab'][a, :, v, :], gamma1['a'], t['a'][pv, ha], optimize=True)
    O['b'] -= scale * np.einsum("mPvQ,uv,um->PQ", h['ab'][c, :, a, :], gamma1['a'], t['a'][pa, hc], optimize=True)
    O['b'] += scale * np.einsum("pmqa,am->pq", h['bb'][:, C, :, part_b], t['b'][:, hC], optimize=True)
    O['b'] += scale * np.einsum("pyqe,yx,ex->pq", h['bb'][:, A, :, V], gamma1['b'], t['b'][pV, hA], optimize=True)
    O['b'] -= scale * np.einsum("pmqv,uv,um->pq", h['bb'][:, C, :, A], gamma1['b'], t['b'][pA, hC], optimize=True)
    return O

def h2_t2_c1(O, h, t, gamma1, eta1, lambdas, orbspace, verbose=False, scale=1.0):
    t0 = time.time()
    c = orbspace['core_alpha']
    C = orbspace['core_beta']
    a = orbspace['active_alpha']
    A = orbspace['active_beta']
    v = orbspace['virt_alpha']
    V = orbspace['virt_beta']
    hc = orbspace['hole_core_alpha']
    hC = orbspace['hole_core_beta']
    ha = orbspace['hole_active_alpha']
    hA = orbspace['hole_active_beta']
    pa = orbspace['particle_active_alpha']
    pA = orbspace['particle_active_beta']
    pv = orbspace['particle_virt_alpha']
    pV = orbspace['particle_virt_beta']
    #
    hole_a = orbspace['hole_alpha']
    part_a = orbspace['particle_alpha']
    hole_b = orbspace['hole_beta']
    part_b = orbspace['particle_beta']
    #
    O['a'][:, hole_a] += 0.5 * scale * np.einsum("rmab,abim->ri", h['aa'][:, c, part_a, part_a], t['aa'][:, :, :, hc], optimize=True)
    O['a'][:, hole_a] += scale * np.einsum("rMaB,aBiM->ri", h['ab'][:, C, part_a, part_b], t['ab'][:, :, :, hC], optimize=True)
    O['b'][:, hole_b] += 0.5 * scale * np.einsum("RMAB,ABIM->RI", h['bb'][:, C, part_b, part_b], t['bb'][:, :, :, hC], optimize=True)
    O['b'][:, hole_b] += scale * np.einsum("mRaB,aBmI->RI", h['ab'][c, :, part_a, part_b], t['ab'][:, :, hc, :], optimize=True)

    O['a'][:, hole_a] += 0.5 * scale * np.einsum("ruab,abiv,uv->ri", h['aa'][:, a, part_a, part_a], t['aa'][:, :, :, ha], gamma1['a'], optimize=True)
    O['a'][:, hole_a] += scale * np.einsum("rUaB,aBiV,UV->ri", h['ab'][:, A, part_a, part_b], t['ab'][:, :, :, hA], gamma1['b'], optimize=True)
    O['b'][:, hole_b] += 0.5 * scale * np.einsum("ruab,abiv,uv->ri", h['bb'][:, A, part_b, part_b], t['bb'][:, :, :, hA], gamma1['b'], optimize=True)
    O['b'][:, hole_b] += scale * np.einsum("uRaB,aBvI,uv->RI", h['ab'][a, :, part_a, part_b], t['ab'][:, :, ha, :], gamma1['a'], optimize=True)

    O['a'][:, hole_a] += 0.5 * scale * np.einsum("rjvy,uxij,xy,uv->ri", h['aa'][:, hole_a, a, a], t['aa'][pa, pa, :, :], gamma1['a'], gamma1['a'], optimize=True)
    O['b'][:, hole_b] += 0.5 * scale * np.einsum("rjvy,uxij,xy,uv->ri", h['bb'][:, hole_b, A, A], t['bb'][pA, pA, :, :], gamma1['b'], gamma1['b'], optimize=True)
    temp = np.einsum("uXiJ,XY,uv->vYiJ", t['ab'][pa, pA, :, :], gamma1['b'], gamma1['a'], optimize=True)
    O['a'][:, hole_a] += scale * np.einsum("vYiJ,rJvY->ri", temp, h['ab'][:, hole_b, a, A], optimize=True)
    O['b'][:, hole_b] += scale * np.einsum("vYjI,jRvY->RI", temp, h['ab'][hole_a, :, a, A], optimize=True)

    O['a'][:, hole_a] -= scale * np.einsum("rmvb,ubim,uv->ri", h['aa'][:, c, a, part_a], t['aa'][pa, :, :, hc], gamma1['a'], optimize=True)
    O['a'][:, hole_a] -= scale * np.einsum("rMvB,uBiM,uv->ri", h['ab'][:, C, a, part_b], t['ab'][pa, :, :, hC], gamma1['a'], optimize=True)
    O['a'][:, hole_a] -= scale * np.einsum("rMbV,bUiM,UV->ri", h['ab'][:, C, part_a, A], t['ab'][:, pA, :, hC], gamma1['b'], optimize=True)
    O['b'][:, hole_b] -= scale * np.einsum("rmvb,ubim,uv->ri", h['bb'][:, C, A, part_b], t['bb'][pA, :, :, hC], gamma1['b'], optimize=True)
    O['b'][:, hole_b] -= scale * np.einsum("mRbV,bUmI,UV->RI", h['ab'][c, :, part_a, A], t['ab'][:, pA, hc, :], gamma1['b'], optimize=True)
    O['b'][:, hole_b] -= scale * np.einsum("mRvB,uBmI,uv->RI", h['ab'][c, :, a, part_b], t['ab'][pa, :, hc, :], gamma1['a'], optimize=True)

    O['a'][:, hole_a] -= scale * np.einsum("rxvb,ubiy,uv,xy->ri", h['aa'][:, a, a, part_a], t['aa'][pa, :, :, ha], gamma1['a'], gamma1['a'], optimize=True)
    O['a'][:, hole_a] -= scale * np.einsum("rXvB,uBiY,uv,XY->ri", h['ab'][:, A, a, part_b], t['ab'][pa, :, :, hA], gamma1['a'], gamma1['b'], optimize=True)
    O['a'][:, hole_a] -= scale * np.einsum("rXbV,bUiY,UV,XY->ri", h['ab'][:, A, part_a, A], t['ab'][:, pA, :, hA], gamma1['b'], gamma1['b'], optimize=True)
    O['b'][:, hole_b] -= scale * np.einsum("RXVB,UBIY,UV,XY->RI", h['bb'][:, A, A, part_b], t['bb'][pA, :, :, hA], gamma1['b'], gamma1['b'], optimize=True)
    O['b'][:, hole_b] -= scale * np.einsum("xRvB,uByI,uv,xy->RI", h['ab'][a, :, a, part_b], t['ab'][pa, :, ha, :], gamma1['a'], gamma1['a'], optimize=True)
    O['b'][:, hole_b] -= scale * np.einsum("xRbV,bUyI,UV,xy->RI", h['ab'][a, :, part_a, A], t['ab'][:, pA, ha, :], gamma1['b'], gamma1['a'], optimize=True)

    O['a'][part_a, :] -= 0.5 * scale * np.einsum("ijpe,aeij->ap", h['aa'][hole_a, hole_a, :, v], t['aa'][:, pv, :, :], optimize=True)
    O['a'][part_a, :] -= scale * np.einsum("iJpE,aEiJ->ap", h['ab'][hole_a, hole_b, :, V], t['ab'][:, pV, :, :], optimize=True)
    O['b'][part_b, :] -= 0.5 * scale * np.einsum("ijpe,aeij->ap", h['bb'][hole_b, hole_b, :, V], t['bb'][:, pV, :, :], optimize=True)
    O['b'][part_b, :] -= scale * np.einsum("iJeP,eAiJ->AP", h['ab'][hole_a, hole_b, v, :], t['ab'][pv, :, :, :], optimize=True)

    O['a'][part_a, :] -= 0.5 * scale * np.einsum("ijpv,auij,uv->ap", h['aa'][hole_a, hole_a, :, a], t['aa'][:, pa, :, :], eta1['a'], optimize=True)
    O['a'][part_a, :] -= scale * np.einsum("iJpV,aUiJ,UV->ap", h['ab'][hole_a, hole_b, :, A], t['ab'][:, pA, :, :], eta1['b'], optimize=True)
    O['b'][part_b, :] -= 0.5 * scale * np.einsum("ijpv,auij,uv->ap", h['bb'][hole_b, hole_b, :, A], t['bb'][:, pA, :, :], eta1['b'], optimize=True)
    O['b'][part_b, :] -= scale * np.einsum("iJvP,uAiJ,uv->AP", h['ab'][hole_a, hole_b, a, :], t['ab'][pa, :, :, :], eta1['a'], optimize=True)

    O['a'][part_a, :] -= 0.5 * scale * np.einsum("uxpb,abvy,uv,xy->ap", h['aa'][a, a, :, part_a], t['aa'][:, :, ha, ha], eta1['a'], eta1['a'], optimize=True)
    O['b'][part_b, :] -= 0.5 * scale * np.einsum("uxpb,abvy,uv,xy->ap", h['bb'][A, A, :, part_b], t['bb'][:, :, hA, hA], eta1['b'], eta1['b'], optimize=True)
    temp = np.einsum("aBvY,uv,XY->aBuX", t['ab'][:, :, ha, hA], eta1['a'], eta1['b'], optimize=True)
    O['a'][part_a, :] -= scale * np.einsum("uXpB,aBuX->ap", h['ab'][a, A, :, part_b], temp, optimize=True)
    O['b'][part_b, :] -= scale * np.einsum("uXbP,bAuX->AP", h['ab'][a, A, part_a, :], temp, optimize=True)

    O['a'][part_a, :] += scale * np.einsum("ujpe,aevj,uv->ap", h['aa'][a, hole_a, :, v], t['aa'][:, pv, ha, :], eta1['a'], optimize=True)
    O['a'][part_a, :] += scale * np.einsum("uJpE,aEvJ,uv->ap", h['ab'][a, hole_b, :, V], t['ab'][:, pV, ha, :], eta1['a'], optimize=True)
    O['a'][part_a, :] += scale * np.einsum("jUpE,aEjV,UV->ap", h['ab'][hole_a, A, :, V], t['ab'][:, pV, :, hA], eta1['b'], optimize=True)
    O['b'][part_b, :] += scale * np.einsum("ujpe,aevj,uv->ap", h['bb'][A, hole_b, :, V], t['bb'][:, pV, hA, :], eta1['b'], optimize=True)
    O['b'][part_b, :] += scale * np.einsum("uJeP,eAvJ,uv->AP", h['ab'][a, hole_b, v, :], t['ab'][pv, :, ha, :], eta1['a'], optimize=True)
    O['b'][part_b, :] += scale * np.einsum("jUeP,eAjV,UV->AP", h['ab'][hole_a, A, v, :], t['ab'][pv, :, :, hA], eta1['b'], optimize=True)

    O['a'][part_a, :] += scale * np.einsum("ujpy,axvj,uv,xy->ap", h['aa'][a, hole_a, :, a], t['aa'][:, pa, ha, :], eta1['a'], eta1['a'], optimize=True)
    O['a'][part_a, :] += scale * np.einsum("uJpY,aXvJ,uv,XY->ap", h['ab'][a, hole_b, :, A], t['ab'][:, pA, ha, :], eta1['a'], eta1['b'], optimize=True)
    O['a'][part_a, :] += scale * np.einsum("jUpY,aXjV,UV,XY->ap", h['ab'][hole_a, A, :, A], t['ab'][:, pA, :, hA], eta1['b'], eta1['b'], optimize=True)
    O['b'][part_b, :] += scale * np.einsum("ujpy,axvj,uv,xy->ap", h['bb'][A, hole_b, :, A], t['bb'][:, pA, hA, :], eta1['b'], eta1['b'], optimize=True)
    O['b'][part_b, :] += scale * np.einsum("uJyP,xAvJ,uv,xy->AP", h['ab'][a, hole_b, a, :], t['ab'][pa, :, ha, :], eta1['a'], eta1['a'], optimize=True)
    O['b'][part_b, :] += scale * np.einsum("jUyP,xAjV,UV,xy->AP", h['ab'][hole_a, A, a, :], t['ab'][pa, :, :, hA], eta1['b'], eta1['a'], optimize=True)

    O['a'][:, hole_a] += 0.25 * scale * np.einsum("rjuv,xyuv,xyij->ri", h['aa'][:, hole_a, a, a], lambdas['aa'], t['aa'][pa, pa, :, :], optimize=True) 
    O['b'][:, hole_b] += 0.25 * scale * np.einsum("rjuv,xyuv,xyij->ri", h['bb'][:, hole_b, A, A], lambdas['bb'], t['bb'][pA, pA, :, :], optimize=True) 
    temp = np.einsum("xYuV,xYiJ->uViJ", lambdas['ab'], t['ab'][pa, pA, :, :], optimize=True)
    O['a'][:, hole_a] += scale * np.einsum("rJuV,uViJ->ri", h['ab'][:, hole_b, a, A], temp, optimize=True)
    O['b'][:, hole_b] += scale * np.einsum("jRuV,uVjI->RI", h['ab'][hole_a, :, a, A], temp, optimize=True)

    O['a'][part_a, :] -= 0.25 * scale * np.einsum("xypb,abuv,xyuv->ap", h['aa'][a, a, :, part_a], t['aa'][:, :, ha, ha], lambdas['aa'], optimize=True)
    O['b'][part_b, :] -= 0.25 * scale * np.einsum("xypb,abuv,xyuv->ap", h['bb'][A, A, :, part_b], t['bb'][:, :, hA, hA], lambdas['bb'], optimize=True)
    temp = np.einsum("xYuV,aBuV->aBxY", lambdas['ab'], t['ab'][:, :, ha, hA], optimize=True)
    O['a'][part_a, :] -= scale * np.einsum("xYpB,aBxY->ap", h['ab'][a, A, :, part_b], temp, optimize=True)
    O['b'][part_b, :] -= scale * np.einsum("xYbP,bAxY->AP", h['ab'][a, A, part_a, :], temp, optimize=True)

    O['a'][:, hole_a] -= scale * np.einsum("rXuA,yXuV,yAiV->ri", h['ab'][:, A, a, part_b], lambdas['ab'], t['ab'][pa, :, :, hA], optimize=True)
    O['b'][:, hole_b] -= scale * np.einsum("xRaU,xYvU,aYvI->RI", h['ab'][a, :, part_a, A], lambdas['ab'], t['ab'][:, pA, ha, :], optimize=True)
    O['a'][part_a, :] += scale * np.einsum("xIpU,xYvU,aYvI->ap", h['ab'][a, hole_b, :, A], lambdas['ab'], t['ab'][:, pA, ha, :], optimize=True)
    O['b'][part_b, :] += scale * np.einsum("iXuP,yXuV,yAiV->AP", h['ab'][hole_a, A, a, :], lambdas['ab'], t['ab'][pa, :, :, hA], optimize=True)

    temp = (
            np.einsum("xyuv,ayiv->auix", lambdas['aa'], t['aa'][:, pa, :, ha], optimize=True)
            + np.einsum("xYuV,aYiV->auix", lambdas['ab'], t['ab'][:, pA, :, hA], optimize=True)
    )
    O['a'][:, hole_a] += scale * np.einsum("rxau,auix->ri", h['aa'][:, a, part_a, a], temp, optimize=True)
    O['a'][part_a, :] -= scale * np.einsum("ixpu,auix->ap", h['aa'][hole_a, a, :, a], temp, optimize=True)
    temp = (
            np.einsum("XYUV,aYiV->aUiX", lambdas['bb'], t['ab'][:, pA, :, hA], optimize=True)
            + np.einsum("yXvU,ayiv->aUiX", lambdas['ab'], t['aa'][:, pa, :, ha], optimize=True)
    )
    O['a'][:, hole_a] += scale * np.einsum("rXaU,aUiX->ri", h['ab'][:, A, part_a, A], temp, optimize=True)
    O['a'][part_a, :] -= scale * np.einsum("iXpU,aUiX->ap", h['ab'][hole_a, A, :, A], temp, optimize=True)
    temp = (
            np.einsum("xyuv,yAvI->uAxI", lambdas['aa'], t['ab'][pa, :, ha, :], optimize=True)
            + np.einsum("xYuV,AYIV->uAxI", lambdas['ab'], t['bb'][:, pA, :, hA], optimize=True)
    )
    O['b'][:, hole_b] += scale * np.einsum("xRuA,uAxI->RI", h['ab'][a, :, a, part_b], temp, optimize=True)
    O['b'][part_b, :] -= scale * np.einsum("xIuP,uAxI->AP", h['ab'][a, hole_b, a, :], temp, optimize=True)
    temp = (
            np.einsum("XYUV,AYIV->AUIX", lambdas['bb'], t['bb'][:, pA, :, hA], optimize=True)
            + np.einsum("yXvU,yAvI->AUIX", lambdas['ab'], t['ab'][pa, :, ha, :], optimize=True)
    )
    O['b'][:, hole_b] += scale * np.einsum("RXAU,AUIX->RI", h['bb'][:, A, part_b, A], temp, optimize=True)
    O['b'][part_b, :] -= scale * np.einsum("IXPU,AUIX->AP", h['bb'][hole_b, A, :, A], temp, optimize=True)

    temp = (
            0.5 * np.einsum("xyav,xyuv->ua",h['aa'][a, a, part_a, a], lambdas['aa'], optimize=True)
            + np.einsum("xYaV,xYuV->ua", h['ab'][a, A, part_a, A], lambdas['ab'], optimize=True)
    )
    O['a'][part_a, hole_a] += scale * np.einsum("ua,abuj->bj", temp, t['aa'][:, :, ha, :], optimize=True)
    O['b'][part_b, hole_b] += scale * np.einsum("ua,aBuJ->BJ", temp, t['ab'][:, :, ha, :], optimize=True)
    temp = (
            0.5 * np.einsum("XYAV,XYUV->UA", h['bb'][A, A, part_b, A], lambdas['bb'], optimize=True)
            + np.einsum("xYvA,xYvU->UA", h['ab'][a, A, a, part_b], lambdas['ab'], optimize=True)
    )
    O['a'][part_a, hole_a] += scale * np.einsum("bAjU,UA->bj", t['ab'][:, :, :, hA], temp, optimize=True)
    O['b'][part_b, hole_b] += scale * np.einsum("ABUJ,UA->BJ", t['bb'][:, :, hA, :], temp, optimize=True)

    temp = (
            0.5 * np.einsum("iyuv,xyuv->ix", h['aa'][hole_a, a, a, a], lambdas['aa'], optimize=True)
            + np.einsum("iYuV,xYuV->ix", h['ab'][hole_a, A, a, A], lambdas['ab'], optimize=True)
    )
    O['a'][part_a, hole_a] -= scale * np.einsum("xbij,ix->bj", t['aa'][pa, :, :, :], temp, optimize=True)
    O['b'][part_b, hole_b] -= scale * np.einsum("xBiJ,ix->BJ", t['ab'][pa, :, :, :], temp, optimize=True)
    temp = (
            0.5 * np.einsum("iyuv,xyuv->ix", h['bb'][hole_b, A, A, A], lambdas['bb'], optimize=True)
            + np.einsum("yIvU,yXvU->IX", h['ab'][a, hole_b, a, A], lambdas['ab'], optimize=True)
    )
    O['a'][part_a, hole_a] -= scale * np.einsum("bXjI,IX->bj", t['ab'][:, pA, :, :], temp, optimize=True)
    O['b'][part_b, hole_b] -= scale * np.einsum("XBIJ,IX->BJ", t['bb'][pA, :, :, :], temp, optimize=True)

    temp = (
            0.5 * np.einsum("xyuv,eyuv->ex", lambdas['aa'], t['aa'][pv, pa, ha, ha], optimize=True)
            + np.einsum("xYuV,eYuV->ex", lambdas['ab'], t['ab'][pv, pA, ha, hA], optimize=True)
    )
    O['a'] += scale * np.einsum("xseq,ex->sq", h['aa'][a, :, v, :], temp, optimize=True)
    O['b'] += scale * np.einsum("xSeQ,ex->SQ", h['ab'][a, :, v, :], temp, optimize=True)
    temp = (
            0.5 * np.einsum("XYUV,EYUV->EX", lambdas['bb'], t['bb'][pV, pA, hA, hA], optimize=True)
            + np.einsum("yXuV,yEuV->EX", lambdas['ab'], t['ab'][pa, pV, ha, hA], optimize=True)
    )
    O['a'] += scale * np.einsum("sXqE,EX->sq", h['ab'][:, A, :, V], temp, optimize=True)
    O['b'] += scale * np.einsum("XSEQ,EX->SQ", h['bb'][A, :, V, :], temp, optimize=True)

    temp = (
            0.5 * np.einsum("xyuv,xymv->um", lambdas['aa'], t['aa'][pa, pa, hc, ha], optimize=True)
            + np.einsum("xYuV,xYmV->um", lambdas['ab'], t['ab'][pa, pA, hc, hA], optimize=True)
    )
    O['a'] -= scale * np.einsum("msuq,um->sq", h['aa'][c, :, a, :], temp, optimize=True)
    O['b'] -= scale * np.einsum("mSuQ,um->SQ", h['ab'][c, :, a, :], temp, optimize=True)
    temp = (
            0.5 * np.einsum("XYUV,XYMV->UM", lambdas['bb'], t['bb'][pA, pA, hC, hA], optimize=True)
            + np.einsum("xYvU,xYvM->UM", lambdas['ab'], t['ab'][pa, pA, ha, hC], optimize=True)
    )
    O['a'] -= scale * np.einsum("sMqU,UM->sq", h['ab'][:, C, :, A], temp, optimize=True)
    O['b'] -= scale * np.einsum("MSUQ,UM->SQ", h['bb'][C, :, A, :], temp, optimize=True)
    return O

def h1_t2_c2(O, h, t, gamma1, eta1, lambdas, orbspace, verbose=False, scale=1.0):
    t0 = time.time()
    c = orbspace['core_alpha']
    C = orbspace['core_beta']
    a = orbspace['active_alpha']
    A = orbspace['active_beta']
    v = orbspace['virt_alpha']
    V = orbspace['virt_beta']
    hc = orbspace['hole_core_alpha']
    hC = orbspace['hole_core_beta']
    ha = orbspace['hole_active_alpha']
    hA = orbspace['hole_active_beta']
    pa = orbspace['particle_active_alpha']
    pA = orbspace['particle_active_beta']
    pv = orbspace['particle_virt_alpha']
    pV = orbspace['particle_virt_beta']
    #
    hole_a = orbspace['hole_alpha']
    part_a = orbspace['particle_alpha']
    hole_b = orbspace['hole_beta']
    part_b = orbspace['particle_beta']

    O['aa'][:, part_a, hole_a, hole_a] += 0.5 * scale * np.einsum("pa,abij->pbij", h['a'][:, part_a], t['aa'], optimize=True)
    O['aa'][part_a, part_a, :, hole_a] -= 0.5 * scale * np.einsum("iq,abij->abqj", h['a'][hole_a, :], t['aa'], optimize=True)

    O['ab'][:, part_b, hole_a, hole_b] += scale * np.einsum("pa,aBiJ->pBiJ", h['a'][:, part_a], t['ab'], optimize=True)
    O['ab'][part_a, :, hole_a, hole_b] += scale * np.einsum("PB,aBiJ->aPiJ", h['b'][:, part_b], t['ab'], optimize=True)
    O['ab'][part_a, part_b, :, hole_b] -= scale * np.einsum("iq,aBiJ->aBqJ", h['a'][hole_a, :], t['ab'], optimize=True)
    O['ab'][part_a, part_b, hole_a, :] -= scale * np.einsum("JQ,aBiJ->aBiQ", h['b'][hole_b, :], t['ab'], optimize=True) 

    O['bb'][:, part_b, hole_b, hole_b] += 0.5 * scale * np.einsum("pa,abij->pbij", h['b'][:, part_b], t['bb'], optimize=True)
    O['bb'][part_b, part_b, :, hole_b] -= 0.5 * scale * np.einsum("iq,abij->abqj", h['b'][hole_b, :], t['bb'], optimize=True)
    return O

def h2_t1_c2(O, h, t, gamma1, eta1, lambdas, orbspace, verbose=False, scale=1.0):
    t0 = time.time()
    c = orbspace['core_alpha']
    C = orbspace['core_beta']
    a = orbspace['active_alpha']
    A = orbspace['active_beta']
    v = orbspace['virt_alpha']
    V = orbspace['virt_beta']
    hc = orbspace['hole_core_alpha']
    hC = orbspace['hole_core_beta']
    ha = orbspace['hole_active_alpha']
    hA = orbspace['hole_active_beta']
    pa = orbspace['particle_active_alpha']
    pA = orbspace['particle_active_beta']
    pv = orbspace['particle_virt_alpha']
    pV = orbspace['particle_virt_beta']
    #
    hole_a = orbspace['hole_alpha']
    part_a = orbspace['particle_alpha']
    hole_b = orbspace['hole_beta']
    part_b = orbspace['particle_beta']

    O['aa'][:, :, hole_a, :] += 0.5 * scale * np.einsum("pqar,ai->pqir", h['aa'][:, :, part_a, :], t['a'], optimize=True)
    O['aa'][part_a, :, :, :] -= 0.5 * scale * np.einsum("iqrs,ai->aqrs", h['aa'][hole_a, :, :, :], t['a'], optimize=True)

    O['ab'][:, :, hole_a, :] += scale * np.einsum("pQaR,ai->pQiR", h['ab'][:, :, part_a, :], t['a'], optimize=True)
    O['ab'][:, :, :, hole_b] += scale * np.einsum("pQrA,AI->pQrI", h['ab'][:, :, :, part_b], t['b'], optimize=True)
    O['ab'][part_a, :, :, :] -= scale * np.einsum("iQrS,ai->aQrS", h['ab'][hole_a, :, :, :], t['a'], optimize=True)
    O['ab'][:, part_b, :, :] -= scale * np.einsum("pIrS,AI->pArS", h['ab'][:, hole_b, :, :], t['b'], optimize=True)

    O['bb'][:, :, hole_b, :] += 0.5 * scale * np.einsum("pqar,ai->pqir", h['bb'][:, :, part_b, :], t['b'], optimize=True)
    O['bb'][part_b, :, :, :] -= 0.5 * scale * np.einsum("iqrs,ai->aqrs", h['bb'][hole_b, :, :, :], t['b'], optimize=True)
    return O

def h2_t2_c2(O, h, t, gamma1, eta1, lambdas, orbspace, verbose=False, scale=1.0):
    t0 = time.time()
    c = orbspace['core_alpha']
    C = orbspace['core_beta']
    a = orbspace['active_alpha']
    A = orbspace['active_beta']
    v = orbspace['virt_alpha']
    V = orbspace['virt_beta']
    hc = orbspace['hole_core_alpha']
    hC = orbspace['hole_core_beta']
    ha = orbspace['hole_active_alpha']
    hA = orbspace['hole_active_beta']
    pa = orbspace['particle_active_alpha']
    pA = orbspace['particle_active_beta']
    pv = orbspace['particle_virt_alpha']
    pV = orbspace['particle_virt_beta']
    #
    hole_a = orbspace['hole_alpha']
    part_a = orbspace['particle_alpha']
    hole_b = orbspace['hole_beta']
    part_b = orbspace['particle_beta']

    # particle-particle contractions
    # (H2 * T2)_C
    O['aa'][:, :, hole_a, hole_a] += 0.125 * scale * np.einsum("rsab,abij->rsij", h['aa'][:, :, part_a, part_a], t['aa'], optimize=True)
    O['ab'][:, :, hole_a, hole_b] += scale * np.einsum("rSaB,aBiJ->rSiJ", h['ab'][:, :, part_a, part_b], t['ab'], optimize=True)
    O['bb'][:, :, hole_b, hole_b] += 0.125 * scale * np.einsum("rsab,abij->rsij", h['bb'][:, :, part_a, part_a], t['bb'], optimize=True)
    # -(T2 * H2)_C
    O['aa'][:, :, hole_a, hole_a] -= 0.25 * scale * np.einsum("rsyb,xbij,xy->rsij", h['aa'][:, :, a, part_a], t['aa'][pa, :, :, :], gamma1['a'], optimize=True) 
    O['ab'][:, :, hole_a, hole_b] -= scale * np.einsum("rSyB,xBiJ,xy->rSiJ", h['ab'][:, :, a, part_b], t['ab'][pa, :, :, :], gamma1['a'], optimize=True)
    O['ab'][:, :, hole_a, hole_b] -= scale * np.einsum("rSbY,bXiJ,XY->rSiJ", h['ab'][:, :, part_a, A], t['ab'][:, pA, :, :], gamma1['b'], optimize=True)
    O['bb'][:, :, hole_b, hole_b] -= 0.25 * scale * np.einsum("rsyb,xbij,xy->rsij", h['bb'][:, :, A, part_b], t['bb'][pA, :, :, :], gamma1['b'], optimize=True) 

    # hole-hole contractions
    # (H2 * T2)_C
    O['aa'][part_a, part_a, :, :] += 0.125 * scale * np.einsum("ijpq,abij->abpq", h['aa'][hole_a, hole_a, :, :], t['aa'], optimize=True)
    O['ab'][part_a, part_b, :, :] += scale * np.einsum("iJpQ,aBiJ->aBpQ", h['ab'][hole_a, hole_b, :, :], t['ab'], optimize=True)
    O['bb'][part_b, part_b, :, :] += 0.125 * scale * np.einsum("ijpq,abij->abpq", h['bb'][hole_b, hole_b, :, :], t['bb'], optimize=True)
    # -(H2 * T2)_C
    O['aa'][part_a, part_a, :, :] -= 0.25 * scale * np.einsum("xjpq,abyj,xy->abpq", h['aa'][a, hole_a, :, :], t['aa'][:, :, ha, :], eta1['a'], optimize=True)
    O['ab'][part_a, part_b, :, :] -= scale * np.einsum("xJpQ,aByJ,xy->aBpQ", h['ab'][a, hole_b, :, :], t['ab'][:, :, ha, :], eta1['a'], optimize=True)
    O['ab'][part_a, part_b, :, :] -= scale * np.einsum("jXpQ,aBjY,XY->aBpQ", h['ab'][hole_a, A, :, :], t['ab'][:, :, :, hA], eta1['b'], optimize=True)
    O['bb'][part_b, part_b, :, :] -= 0.25 * scale * np.einsum("XJPQ,ABYJ,XY->ABPQ", h['bb'][A, hole_b, :, :], t['bb'][:, :, hA, :], eta1['b'], optimize=True)

    # hole-particle contractions
    # (H2 * T2)_C
    O['aa'][:, part_a, :, hole_a] += scale * np.einsum("msaq,abmj->sbqj", h['aa'][c, :, part_a, :], t['aa'][:, :, hc, :], optimize=True)
    O['aa'][:, part_a, :, hole_a] += scale * np.einsum("xsaq,abyj,xy->sbqj", h['aa'][a, :, part_a, :], t['aa'][:, :, ha, :], gamma1['a'], optimize=True)
    O['aa'][:, part_a, :, hole_a] += scale * np.einsum("sMqA,bAjM->sbqj", h['ab'][:, C, :, part_b], t['ab'][:, :, :, hC], optimize=True)
    O['aa'][:, part_a, :, hole_a] += scale * np.einsum("sXqA,bAjY,XY->sbqj", h['ab'][:, A, :, part_b], t['ab'][:, :, :, hA], gamma1['b'], optimize=True)
    # -(T2 * H2)_C
    O['aa'][:, part_a, :, hole_a] -= scale * np.einsum("isyq,xbij,xy->sbqj", h['aa'][hole_a, :, a, :], t['aa'][pa, :, :, :], gamma1['a'], optimize=True)
    O['aa'][:, part_a, :, hole_a] -= scale * np.einsum("sIqY,bXjI,XY->sbqj", h['ab'][:, hole_b, :, A], t['ab'][:, pA, :, :], gamma1['b'], optimize=True)

    # (H2 * T2)_C
    O['bb'][:, part_b, :, hole_b] += scale * np.einsum("MSAQ,ABMJ->SBQJ", h['bb'][C, :, part_b, :], t['bb'][:, :, hC, :], optimize=True)
    O['bb'][:, part_b, :, hole_b] += scale * np.einsum("XSAQ,ABYJ,XY->SBQJ", h['bb'][A, :, part_b, :], t['bb'][:, :, hA, :], gamma1['b'], optimize=True)
    O['bb'][:, part_b, :, hole_b] += scale * np.einsum("mSaQ,aBmJ->SBQJ", h['ab'][c, :, part_a, :], t['ab'][:, :, hc, :], optimize=True)
    O['bb'][:, part_b, :, hole_b] += scale * np.einsum("xSaQ,aByJ,xy->SBQJ", h['ab'][a, :, part_a, :], t['ab'][:, :, ha, :], gamma1['a'], optimize=True)
    # -(T2 * H2)_C
    O['bb'][:, part_b, :, hole_b] -= scale * np.einsum("ISYQ,XBIJ,XY->SBQJ", h['bb'][hole_b, :, A, :], t['bb'][pA, :, :, :], gamma1['b'], optimize=True)
    O['bb'][:, part_b, :, hole_b] -= scale * np.einsum("iSyQ,xBiJ,xy->SBQJ", h['ab'][hole_a, :, a, :], t['ab'][pa, :, :, :], gamma1['a'], optimize=True)

    # (H2 * T2)_C
    O['ab'][:, part_b, :, hole_b] += scale * np.einsum("msaq,aBmJ->sBqJ", h['aa'][c, :, part_a, :], t['ab'][:, :, hc, :], optimize=True)
    O['ab'][:, part_b, :, hole_b] += scale * np.einsum("xsaq,aByJ,xy->sBqJ", h['aa'][a, :, part_a, :], t['ab'][:, :, ha, :], gamma1['a'], optimize=True)
    O['ab'][:, part_b, :, hole_b] += scale * np.einsum("sMqA,ABMJ->sBqJ", h['ab'][:, C, :, part_b], t['bb'][:, :, hC, :], optimize=True)
    O['ab'][:, part_b, :, hole_b] += scale * np.einsum("sXqA,ABYJ,XY->sBqJ", h['ab'][:, A, :, part_b], t['bb'][:, :, hA, :], gamma1['b'], optimize=True)
    # -(T2 * H2)_C
    O['ab'][:, part_b, :, hole_b] -= scale * np.einsum("isyq,xBiJ,xy->sBqJ", h['aa'][hole_a, :, a, :], t['ab'][pa, :, :, :], gamma1['a'], optimize=True)
    O['ab'][:, part_b, :, hole_b] -= scale * np.einsum("sIqY,XBIJ,XY->sBqJ", h['ab'][:, hole_b, :, A], t['bb'][pA, :, :, :], gamma1['b'], optimize=True)

    # (H2 * T2)_C
    O['ab'][:, part_b, hole_a, :] -= scale * np.einsum("sMaQ,aBiM->sBiQ", h['ab'][:, C, part_a, :], t['ab'][:, :, :, hC], optimize=True)
    O['ab'][:, part_b, hole_a, :] -= scale * np.einsum("sXaQ,aBiY,XY->sBiQ", h['ab'][:, A, part_a, :], t['ab'][:, :, :, hA], gamma1['b'], optimize=True)
    # -(T2 * H2)_C
    O['ab'][:, part_b, hole_a, :] += scale * np.einsum("sJyQ,xBiJ,xy->sBiQ", h['ab'][:, hole_b, a, :], t['ab'][pa, :, :, :], gamma1['a'], optimize=True)

    # (H2 * T2)_C
    O['ab'][part_a, :, :, hole_b] -= scale * np.einsum("mSqB,aBmJ->aSqJ", h['ab'][c, :, :, part_b], t['ab'][:, :, hc, :], optimize=True)
    O['ab'][part_a, :, :, hole_b] -= scale * np.einsum("xSqB,aByJ,xy->aSqJ", h['ab'][a, :, :, part_b], t['ab'][:, :, ha, :], gamma1['a'], optimize=True)
    # -(T2 * H2)_C
    O['ab'][part_a, :, :, hole_b] += scale * np.einsum("iSqY,aXiJ,XY->aSqJ", h['ab'][hole_a, :, :, A], t['ab'][:, pA, :, :], gamma1['b'], optimize=True)

    # (H2 * T2)_C
    O['ab'][part_a, :, hole_a, :] += scale * np.einsum("mSbQ,abim->aSiQ", h['ab'][c, :, part_a, :], t['aa'][:, :, :, hc], optimize=True)
    O['ab'][part_a, :, hole_a, :] += scale * np.einsum("xSbQ,abiy,xy->aSiQ", h['ab'][a, :, part_a, :], t['aa'][:, :, :, ha], gamma1['a'], optimize=True)
    O['ab'][part_a, :, hole_a, :] += scale * np.einsum("MSBQ,aBiM->aSiQ", h['bb'][C, :, part_b, :], t['ab'][:, :, :, hC], optimize=True)
    O['ab'][part_a, :, hole_a, :] += scale * np.einsum("XSBQ,aBiY,XY->aSiQ", h['bb'][A, :, part_b, :], t['ab'][:, :, :, hA], gamma1['b'], optimize=True)
    # -(T2 * H2)_C
    O['ab'][part_a, :, hole_a, :] -= scale * np.einsum("jSyQ,axij,xy->aSiQ", h['ab'][hole_a, :, a, :], t['aa'][:, pa, :, :], gamma1['a'], optimize=True)
    O['ab'][part_a, :, hole_a, :] -= scale * np.einsum("JSYQ,aXiJ,XY->aSiQ", h['bb'][hole_b, :, A, :], t['ab'][:, pA, :, :], gamma1['b'], optimize=True)
    return O
