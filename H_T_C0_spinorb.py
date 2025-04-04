import time
import numpy as np

def H_T_C0_flipped(F, V, T1, T2, gamma1, eta1, lambdas, orbspace, scale=1.0):
    # 24 lines
    t0 = time.time()
    hc = orbspace['hole_core']
    ha = orbspace['hole_active']
    pa = orbspace['particle_active']
    pv = orbspace['particle_virt']
    c = orbspace['core']
    a = orbspace['active']
    v = orbspace['virt']

    lambda2 = lambdas['2']

    C0 = 0.0
    C0 += scale * +1.00000000 * np.einsum('ui,iv,vu->', F[a,c], T1[hc,pa], eta1, optimize=True)
    C0 += scale * +1.00000000 * np.einsum('ai,ia->', F[v,c], T1[hc,pv], optimize=True)
    C0 += scale * +1.00000000 * np.einsum('au,va,uv->', F[v,a], T1[ha,pv], gamma1, optimize=True)

    C0 += scale * -0.50000000 * np.einsum('ui,ivwx,wxuv->', F[a,c], T2[hc,ha,pa,pa], lambda2, optimize=True)
    C0 += scale * -0.50000000 * np.einsum('au,vwxa,uxvw->', F[v,a], T2[ha,ha,pa,pv], lambda2, optimize=True)
    C0 += scale * -0.50000000 * np.einsum('iu,vwix,uxvw->', T1[hc,pa], V[a,a,c,a], lambda2, optimize=True)
    C0 += scale * -0.50000000 * np.einsum('ua,vawx,wxuv->', T1[ha,pv], V[a,v,a,a], lambda2, optimize=True)

    C0 += scale * +0.25000000 * np.einsum('ijuv,wxij,vx,uw->', T2[hc,hc,pa,pa], V[a,a,c,c], eta1, eta1, optimize=True)
    C0 += scale * +0.12500000 * np.einsum('ijuv,wxij,uvwx->', T2[hc,hc,pa,pa], V[a,a,c,c], lambda2, optimize=True)
    C0 += scale * +0.50000000 * np.einsum('iuvw,xyiz,wy,vx,zu->', T2[hc,ha,pa,pa], V[a,a,c,a], eta1, eta1, gamma1, optimize=True)
    C0 += scale * +1.00000000 * np.einsum('iuvw,xyiz,wy,vzux->', T2[hc,ha,pa,pa], V[a,a,c,a], eta1, lambda2, optimize=True)
    C0 += scale * +0.25000000 * np.einsum('iuvw,xyiz,zu,vwxy->', T2[hc,ha,pa,pa], V[a,a,c,a], gamma1, lambda2, optimize=True)
    # C0 += scale * +0.25000000 * np.einsum('iuvw,xyiz,vwzuxy->', T2[hc,ha,pa,pa], V[a,a,c,a], lambda3, optimize=True)
    C0 += scale * +0.50000000 * np.einsum('ijua,vaij,uv->', T2[hc,hc,pa,pv], V[a,v,c,c], eta1, optimize=True)
    C0 += scale * +1.00000000 * np.einsum('iuva,waix,vw,xu->', T2[hc,ha,pa,pv], V[a,v,c,a], eta1, gamma1, optimize=True)
    C0 += scale * +1.00000000 * np.einsum('iuva,waix,vxuw->', T2[hc,ha,pa,pv], V[a,v,c,a], lambda2, optimize=True)
    C0 += scale * +0.50000000 * np.einsum('uvwa,xayz,wx,zv,yu->', T2[ha,ha,pa,pv], V[a,v,a,a], eta1, gamma1, gamma1, optimize=True)
    C0 += scale * +0.25000000 * np.einsum('uvwa,xayz,wx,yzuv->', T2[ha,ha,pa,pv], V[a,v,a,a], eta1, lambda2, optimize=True)
    C0 += scale * +1.00000000 * np.einsum('uvwa,xayz,zv,wyux->', T2[ha,ha,pa,pv], V[a,v,a,a], gamma1, lambda2, optimize=True)
    # C0 += scale * -0.25000000 * np.einsum('uvwa,xayz,wyzuvx->', T2[ha,ha,pa,pv], V[a,v,a,a], lambda3, optimize=True)
    C0 += scale * +0.25000000 * np.einsum('ijab,abij->', T2[hc,hc,pv,pv], V[v,v,c,c], optimize=True)
    C0 += scale * +0.50000000 * np.einsum('iuab,abiv,vu->', T2[hc,ha,pv,pv], V[v,v,c,a], gamma1, optimize=True)
    C0 += scale * +0.25000000 * np.einsum('uvab,abwx,xv,wu->', T2[ha,ha,pv,pv], V[v,v,a,a], gamma1, gamma1, optimize=True)
    C0 += scale * +0.12500000 * np.einsum('uvab,abwx,wxuv->', T2[ha,ha,pv,pv], V[v,v,a,a], lambda2, optimize=True)

    t1 = time.time()
    print("H_T_C0 took {:.4f} seconds to run.".format(t1-t0))
    return C0

def H_T_C0(F, V, T1, T2, gamma1, eta1, lambdas, orbspace, scale=1.0):
    # 24 lines
    t0 = time.time()
    hc = orbspace['hole_core']
    ha = orbspace['hole_active']
    pa = orbspace['particle_active']
    pv = orbspace['particle_virt']
    c = orbspace['core']
    a = orbspace['active']
    v = orbspace['virt']

    lambda2 = lambdas['2']

    C0 = 0.
    C0 += scale * +1.00000000 * np.einsum('iu,vi,vu->', F[c,a], T1[pa,hc], eta1, optimize=True)
    C0 += scale * -0.50000000 * np.einsum('iu,vwix,vwux->', F[c,a], T2[pa,pa,hc,ha], lambda2, optimize=True)
    C0 += scale * +1.00000000 * np.einsum('ia,ai->', F[c,v], T1[pv,hc], optimize=True)
    C0 += scale * +1.00000000 * np.einsum('ua,av,uv->', F[a,v], T1[pv,ha], gamma1, optimize=True)
    C0 += scale * -0.50000000 * np.einsum('ua,vawx,uvwx->', F[a,v], T2[pa,pv,ha,ha], lambda2, optimize=True)
    C0 += scale * -0.50000000 * np.einsum('ui,ivwx,uvwx->', T1[pa,hc], V[c,a,a,a], lambda2, optimize=True)
    C0 += scale * -0.50000000 * np.einsum('au,vwxa,vwux->', T1[pv,ha], V[a,a,a,v], lambda2, optimize=True)
    C0 += scale * +0.25000000 * np.einsum('uvij,ijwx,vx,uw->', T2[pa,pa,hc,hc], V[c,c,a,a], eta1, eta1, optimize=True)
    C0 += scale * +0.12500000 * np.einsum('uvij,ijwx,uvwx->', T2[pa,pa,hc,hc], V[c,c,a,a], lambda2, optimize=True)
    C0 += scale * +0.50000000 * np.einsum('uviw,ixyz,vz,uy,xw->', T2[pa,pa,hc,ha], V[c,a,a,a], eta1, eta1, gamma1, optimize=True)
    C0 += scale * +1.00000000 * np.einsum('uviw,ixyz,vz,uxwy->', T2[pa,pa,hc,ha], V[c,a,a,a], eta1, lambda2, optimize=True)
    C0 += scale * +0.25000000 * np.einsum('uviw,ixyz,xw,uvyz->', T2[pa,pa,hc,ha], V[c,a,a,a], gamma1, lambda2, optimize=True)
    # C0 += scale * +0.25000000 * np.einsum('uviw,ixyz,uvxwyz->', T2[pa,pa,hc,ha], V[c,a,a,a], lambda3, optimize=True)
    C0 += scale * +0.50000000 * np.einsum('uaij,ijva,uv->', T2[pa,pv,hc,hc], V[c,c,a,v], eta1, optimize=True)
    C0 += scale * +1.00000000 * np.einsum('uaiv,iwxa,ux,wv->', T2[pa,pv,hc,ha], V[c,a,a,v], eta1, gamma1, optimize=True)
    C0 += scale * +1.00000000 * np.einsum('uaiv,iwxa,uwvx->', T2[pa,pv,hc,ha], V[c,a,a,v], lambda2, optimize=True)
    C0 += scale * +0.50000000 * np.einsum('uavw,xyza,uz,yw,xv->', T2[pa,pv,ha,ha], V[a,a,a,v], eta1, gamma1, gamma1, optimize=True)
    C0 += scale * +0.25000000 * np.einsum('uavw,xyza,uz,xyvw->', T2[pa,pv,ha,ha], V[a,a,a,v], eta1, lambda2, optimize=True)
    C0 += scale * +1.00000000 * np.einsum('uavw,xyza,yw,uxvz->', T2[pa,pv,ha,ha], V[a,a,a,v], gamma1, lambda2, optimize=True)
    # C0 += scale * -0.25000000 * np.einsum('uavw,xyza,uxyvwz->', T2[pa,pv,ha,ha], V[a,a,a,v], lambda3, optimize=True)
    C0 += scale * +0.25000000 * np.einsum('abij,ijab->', T2[pv,pv,hc,hc], V[c,c,v,v], optimize=True)
    C0 += scale * +0.50000000 * np.einsum('abiu,ivab,vu->', T2[pv,pv,hc,ha], V[c,a,v,v], gamma1, optimize=True)
    C0 += scale * +0.25000000 * np.einsum('abuv,wxab,xv,wu->', T2[pv,pv,ha,ha], V[a,a,v,v], gamma1, gamma1, optimize=True)
    C0 += scale * +0.12500000 * np.einsum('abuv,wxab,wxuv->', T2[pv,pv,ha,ha], V[a,a,v,v], lambda2, optimize=True)

    t1 = time.time()
    print("H_T_C0 took {:.4f} seconds to run.".format(t1-t0))

    return C0
