import time
import numpy as np

def h1_t1_c0(O, Hbar, t, gamma1, eta1, lambdas, orbspace, verbose=False, scale=1.0):
    # 3 lines
    t0 = time.time()
    c = orbspace['core']
    a = orbspace['active']
    v = orbspace['virt']
    hc = orbspace['hole_core']
    ha = orbspace['hole_active']
    pa = orbspace['particle_active']
    pv = orbspace['particle_virt']

    t1, t2 = t # Unpack cluster operator
    h1, h2 = Hbar # Unpack Hamiltonian
    lambda2 = lambdas['2'] # 2-cumulant
    # lambda3 = lambdas['3'] # 3-cumulant
    O += scale * +1.00000000 * np.einsum('uv,iv,ui->', eta1, h1[c,a], t1[pa,hc], optimize=True)
    O += scale * +1.00000000 * np.einsum('uv,ua,av->', gamma1, h1[a,v], t1[pv,ha], optimize=True)
    O += scale * +1.00000000 * np.einsum('ia,ai->', h1[c,v], t1[pv,hc], optimize=True)

    t1 = time.time()
    if verbose: print("h1_t1_c0 took {:.4f} seconds to run.".format(t1-t0))

    return O
def h1_t2_c0(O, Hbar, t, gamma1, eta1, lambdas, orbspace, verbose=False, scale=1.0):
    # 2 lines
    t0 = time.time()
    c = orbspace['core']
    a = orbspace['active']
    v = orbspace['virt']
    hc = orbspace['hole_core']
    ha = orbspace['hole_active']
    pa = orbspace['particle_active']
    pv = orbspace['particle_virt']

    t1, t2 = t # Unpack cluster operator
    h1, h2 = Hbar # Unpack Hamiltonian
    lambda2 = lambdas['2'] # 2-cumulant
    # lambda3 = lambdas['3'] # 3-cumulant
    O += scale * -0.50000000 * np.einsum('iu,vwux,vwix->', h1[c,a], lambda2, t2[pa,pa,hc,ha], optimize=True)
    O += scale * -0.50000000 * np.einsum('ua,uvwx,vawx->', h1[a,v], lambda2, t2[pa,pv,ha,ha], optimize=True)

    t1 = time.time()
    if verbose: print("h1_t2_c0 took {:.4f} seconds to run.".format(t1-t0))

    return O
def h2_t1_c0(O, Hbar, t, gamma1, eta1, lambdas, orbspace, verbose=False, scale=1.0):
    # 2 lines
    t0 = time.time()
    c = orbspace['core']
    a = orbspace['active']
    v = orbspace['virt']
    hc = orbspace['hole_core']
    ha = orbspace['hole_active']
    pa = orbspace['particle_active']
    pv = orbspace['particle_virt']

    t1, t2 = t # Unpack cluster operator
    h1, h2 = Hbar # Unpack Hamiltonian
    lambda2 = lambdas['2'] # 2-cumulant
    # lambda3 = lambdas['3'] # 3-cumulant
    O += scale * +0.50000000 * np.einsum('iuvw,uxvw,xi->', h2[c,a,a,a], lambda2, t1[pa,hc], optimize=True)
    O += scale * +0.50000000 * np.einsum('uvwa,uvwx,ax->', h2[a,a,a,v], lambda2, t1[pv,ha], optimize=True)

    t1 = time.time()
    if verbose: print("h2_t1_c0 took {:.4f} seconds to run.".format(t1-t0))

    return O
def h2_t2_c0(O, Hbar, t, gamma1, eta1, lambdas, orbspace, verbose=False, scale=1.0):
    # 17 lines
    t0 = time.time()
    c = orbspace['core']
    a = orbspace['active']
    v = orbspace['virt']
    hc = orbspace['hole_core']
    ha = orbspace['hole_active']
    pa = orbspace['particle_active']
    pv = orbspace['particle_virt']

    t1, t2 = t # Unpack cluster operator
    h1, h2 = Hbar # Unpack Hamiltonian
    lambda2 = lambdas['2'] # 2-cumulant
    lambda3 = lambdas['3'] # 3-cumulant
    O += scale * +0.50000000 * np.einsum('uv,wx,yz,iyvx,uwiz->', eta1, eta1, gamma1, h2[c,a,a,a], t2[pa,pa,hc,ha], optimize=True)
    O += scale * +0.25000000 * np.einsum('uv,wx,ijvx,uwij->', eta1, eta1, h2[c,c,a,a], t2[pa,pa,hc,hc], optimize=True)
    O += scale * +0.50000000 * np.einsum('uv,wx,yz,wyva,uaxz->', eta1, gamma1, gamma1, h2[a,a,a,v], t2[pa,pv,ha,ha], optimize=True)
    O += scale * +1.00000000 * np.einsum('uv,wx,iwva,uaix->', eta1, gamma1, h2[c,a,a,v], t2[pa,pv,hc,ha], optimize=True)
    O += scale * +0.50000000 * np.einsum('uv,ijva,uaij->', eta1, h2[c,c,a,v], t2[pa,pv,hc,hc], optimize=True)
    O += scale * +1.00000000 * np.einsum('uv,iwvx,wyxz,uyiz->', eta1, h2[c,a,a,a], lambda2, t2[pa,pa,hc,ha], optimize=True)
    O += scale * +0.25000000 * np.einsum('uv,wxva,wxyz,uayz->', eta1, h2[a,a,a,v], lambda2, t2[pa,pv,ha,ha], optimize=True)
    O += scale * +0.25000000 * np.einsum('uv,wx,uwab,abvx->', gamma1, gamma1, h2[a,a,v,v], t2[pv,pv,ha,ha], optimize=True)
    O += scale * +0.25000000 * np.einsum('uv,iuwx,yzwx,yziv->', gamma1, h2[c,a,a,a], lambda2, t2[pa,pa,hc,ha], optimize=True)
    O += scale * +0.50000000 * np.einsum('uv,iuab,abiv->', gamma1, h2[c,a,v,v], t2[pv,pv,hc,ha], optimize=True)
    O += scale * +1.00000000 * np.einsum('uv,uwxa,wyxz,yavz->', gamma1, h2[a,a,a,v], lambda2, t2[pa,pv,ha,ha], optimize=True)
    O += scale * +0.12500000 * np.einsum('ijuv,wxuv,wxij->', h2[c,c,a,a], lambda2, t2[pa,pa,hc,hc], optimize=True)
    O += scale * +0.25000000 * np.einsum('ijab,abij->', h2[c,c,v,v], t2[pv,pv,hc,hc], optimize=True)
    O += scale * +0.25000000 * np.einsum('iuvw,uxyvwz,xyiz->', h2[c,a,a,a], lambda3, t2[pa,pa,hc,ha], optimize=True)
    O += scale * +1.00000000 * np.einsum('iuva,uwvx,waix->', h2[c,a,a,v], lambda2, t2[pa,pv,hc,ha], optimize=True)
    O += scale * -0.25000000 * np.einsum('uvwa,uvxwyz,xayz->', h2[a,a,a,v], lambda3, t2[pa,pv,ha,ha], optimize=True)
    O += scale * +0.12500000 * np.einsum('uvab,uvwx,abwx->', h2[a,a,v,v], lambda2, t2[pv,pv,ha,ha], optimize=True)

    t1 = time.time()
    if verbose: print("h2_t2_c0 took {:.4f} seconds to run.".format(t1-t0))

    return O
def h1_t1_c1(O, Hbar, t, gamma1, eta1, lambdas, orbspace, verbose=False, scale=1.0):
    # 24 lines
    t0 = time.time()
    c = orbspace['core']
    a = orbspace['active']
    v = orbspace['virt']
    hc = orbspace['hole_core']
    ha = orbspace['hole_active']
    pa = orbspace['particle_active']
    pv = orbspace['particle_virt']

    t1, t2 = t # Unpack cluster operator
    h1, h2 = Hbar # Unpack Hamiltonian
    lambda2 = lambdas['2'] # 2-cumulant
    # lambda3 = lambdas['3'] # 3-cumulant
    O[a,a] += scale * -1.00000000 * np.einsum('iu,vi->vu', h1[c,a], t1[pa,hc], optimize=True)
    O[a,a] += scale * +1.00000000 * np.einsum('ua,av->uv', h1[a,v], t1[pv,ha], optimize=True)
    O[c,a] += scale * +1.00000000 * np.einsum('ia,au->iu', h1[c,v], t1[pv,ha], optimize=True)
    O[v,a] += scale * -1.00000000 * np.einsum('uv,uw,av->aw', eta1, h1[a,a], t1[pv,ha], optimize=True)
    O[v,a] += scale * -1.00000000 * np.einsum('uv,uw,av->aw', gamma1, h1[a,a], t1[pv,ha], optimize=True)
    O[v,a] += scale * -1.00000000 * np.einsum('iu,ai->au', h1[c,a], t1[pv,hc], optimize=True)
    O[v,a] += scale * +1.00000000 * np.einsum('ab,bu->au', h1[v,v], t1[pv,ha], optimize=True)
    O[a,c] += scale * +1.00000000 * np.einsum('uv,wv,ui->wi', eta1, h1[a,a], t1[pa,hc], optimize=True)
    O[a,c] += scale * +1.00000000 * np.einsum('uv,wv,ui->wi', gamma1, h1[a,a], t1[pa,hc], optimize=True)
    O[a,c] += scale * -1.00000000 * np.einsum('ij,ui->uj', h1[c,c], t1[pa,hc], optimize=True)
    O[a,c] += scale * +1.00000000 * np.einsum('ua,ai->ui', h1[a,v], t1[pv,hc], optimize=True)
    O[c,c] += scale * +1.00000000 * np.einsum('uv,iv,uj->ij', eta1, h1[c,a], t1[pa,hc], optimize=True)
    O[c,c] += scale * +1.00000000 * np.einsum('uv,iv,uj->ij', gamma1, h1[c,a], t1[pa,hc], optimize=True)
    O[c,c] += scale * +1.00000000 * np.einsum('ia,aj->ij', h1[c,v], t1[pv,hc], optimize=True)
    O[v,c] += scale * -1.00000000 * np.einsum('uv,ui,av->ai', eta1, h1[a,c], t1[pv,ha], optimize=True)
    O[v,c] += scale * +1.00000000 * np.einsum('uv,av,ui->ai', eta1, h1[v,a], t1[pa,hc], optimize=True)
    O[v,c] += scale * -1.00000000 * np.einsum('uv,ui,av->ai', gamma1, h1[a,c], t1[pv,ha], optimize=True)
    O[v,c] += scale * +1.00000000 * np.einsum('uv,av,ui->ai', gamma1, h1[v,a], t1[pa,hc], optimize=True)
    O[v,c] += scale * -1.00000000 * np.einsum('ij,ai->aj', h1[c,c], t1[pv,hc], optimize=True)
    O[v,c] += scale * +1.00000000 * np.einsum('ab,bi->ai', h1[v,v], t1[pv,hc], optimize=True)
    O[a,v] += scale * -1.00000000 * np.einsum('ia,ui->ua', h1[c,v], t1[pa,hc], optimize=True)
    O[v,v] += scale * -1.00000000 * np.einsum('uv,ua,bv->ba', eta1, h1[a,v], t1[pv,ha], optimize=True)
    O[v,v] += scale * -1.00000000 * np.einsum('uv,ua,bv->ba', gamma1, h1[a,v], t1[pv,ha], optimize=True)
    O[v,v] += scale * -1.00000000 * np.einsum('ia,bi->ba', h1[c,v], t1[pv,hc], optimize=True)

    t1 = time.time()
    if verbose: print("h1_t1_c1 took {:.4f} seconds to run.".format(t1-t0))

    return O
def h1_t2_c1(O, Hbar, t, gamma1, eta1, lambdas, orbspace, verbose=False, scale=1.0):
    # 18 lines
    t0 = time.time()
    c = orbspace['core']
    a = orbspace['active']
    v = orbspace['virt']
    hc = orbspace['hole_core']
    ha = orbspace['hole_active']
    pa = orbspace['particle_active']
    pv = orbspace['particle_virt']

    t1, t2 = t # Unpack cluster operator
    h1, h2 = Hbar # Unpack Hamiltonian
    lambda2 = lambdas['2'] # 2-cumulant
    # lambda3 = lambdas['3'] # 3-cumulant
    O[a,a] += scale * -1.00000000 * np.einsum('uv,iv,wuix->wx', eta1, h1[c,a], t2[pa,pa,hc,ha], optimize=True)
    O[a,a] += scale * +1.00000000 * np.einsum('uv,ua,waxv->wx', gamma1, h1[a,v], t2[pa,pv,ha,ha], optimize=True)
    O[a,a] += scale * -1.00000000 * np.einsum('ia,uaiv->uv', h1[c,v], t2[pa,pv,hc,ha], optimize=True)
    O[v,a] += scale * +1.00000000 * np.einsum('uv,wx,ux,wayv->ay', eta1, gamma1, h1[a,a], t2[pa,pv,ha,ha], optimize=True)
    O[v,a] += scale * -1.00000000 * np.einsum('uv,wx,wv,uayx->ay', eta1, gamma1, h1[a,a], t2[pa,pv,ha,ha], optimize=True)
    O[v,a] += scale * +1.00000000 * np.einsum('uv,iv,uaiw->aw', eta1, h1[c,a], t2[pa,pv,hc,ha], optimize=True)
    O[v,a] += scale * +1.00000000 * np.einsum('uv,ua,bawv->bw', gamma1, h1[a,v], t2[pv,pv,ha,ha], optimize=True)
    O[v,a] += scale * -1.00000000 * np.einsum('ia,baiu->bu', h1[c,v], t2[pv,pv,hc,ha], optimize=True)
    O[a,c] += scale * -1.00000000 * np.einsum('uv,wx,ux,ywiv->yi', eta1, gamma1, h1[a,a], t2[pa,pa,hc,ha], optimize=True)
    O[a,c] += scale * +1.00000000 * np.einsum('uv,wx,wv,yuix->yi', eta1, gamma1, h1[a,a], t2[pa,pa,hc,ha], optimize=True)
    O[a,c] += scale * +1.00000000 * np.einsum('uv,iv,wuji->wj', eta1, h1[c,a], t2[pa,pa,hc,hc], optimize=True)
    O[a,c] += scale * +1.00000000 * np.einsum('uv,ua,waiv->wi', gamma1, h1[a,v], t2[pa,pv,hc,ha], optimize=True)
    O[a,c] += scale * +1.00000000 * np.einsum('ia,uaji->uj', h1[c,v], t2[pa,pv,hc,hc], optimize=True)
    O[v,c] += scale * +1.00000000 * np.einsum('uv,wx,ux,waiv->ai', eta1, gamma1, h1[a,a], t2[pa,pv,hc,ha], optimize=True)
    O[v,c] += scale * -1.00000000 * np.einsum('uv,wx,wv,uaix->ai', eta1, gamma1, h1[a,a], t2[pa,pv,hc,ha], optimize=True)
    O[v,c] += scale * -1.00000000 * np.einsum('uv,iv,uaji->aj', eta1, h1[c,a], t2[pa,pv,hc,hc], optimize=True)
    O[v,c] += scale * +1.00000000 * np.einsum('uv,ua,baiv->bi', gamma1, h1[a,v], t2[pv,pv,hc,ha], optimize=True)
    O[v,c] += scale * +1.00000000 * np.einsum('ia,baji->bj', h1[c,v], t2[pv,pv,hc,hc], optimize=True)

    t1 = time.time()
    if verbose: print("h1_t2_c1 took {:.4f} seconds to run.".format(t1-t0))

    return O
def h2_t1_c1(O, Hbar, t, gamma1, eta1, lambdas, orbspace, verbose=False, scale=1.0):
    # 27 lines
    t0 = time.time()
    c = orbspace['core']
    a = orbspace['active']
    v = orbspace['virt']
    hc = orbspace['hole_core']
    ha = orbspace['hole_active']
    pa = orbspace['particle_active']
    pv = orbspace['particle_virt']

    t1, t2 = t # Unpack cluster operator
    h1, h2 = Hbar # Unpack Hamiltonian
    lambda2 = lambdas['2'] # 2-cumulant
    # lambda3 = lambdas['3'] # 3-cumulant
    O[a,a] += scale * -1.00000000 * np.einsum('uv,iwxv,ui->wx', eta1, h2[c,a,a,a], t1[pa,hc], optimize=True)
    O[a,a] += scale * +1.00000000 * np.einsum('uv,wuxa,av->wx', gamma1, h2[a,a,a,v], t1[pv,ha], optimize=True)
    O[a,a] += scale * -1.00000000 * np.einsum('iuva,ai->uv', h2[c,a,a,v], t1[pv,hc], optimize=True)
    O[c,a] += scale * +1.00000000 * np.einsum('uv,ijwv,uj->iw', eta1, h2[c,c,a,a], t1[pa,hc], optimize=True)
    O[c,a] += scale * +1.00000000 * np.einsum('uv,iuwa,av->iw', gamma1, h2[c,a,a,v], t1[pv,ha], optimize=True)
    O[c,a] += scale * +1.00000000 * np.einsum('ijua,aj->iu', h2[c,c,a,v], t1[pv,hc], optimize=True)
    O[v,a] += scale * -1.00000000 * np.einsum('uv,iawv,ui->aw', eta1, h2[c,v,a,a], t1[pa,hc], optimize=True)
    O[v,a] += scale * -1.00000000 * np.einsum('uv,uawb,bv->aw', gamma1, h2[a,v,a,v], t1[pv,ha], optimize=True)
    O[v,a] += scale * -1.00000000 * np.einsum('iaub,bi->au', h2[c,v,a,v], t1[pv,hc], optimize=True)
    O[a,c] += scale * -1.00000000 * np.einsum('uv,iwjv,ui->wj', eta1, h2[c,a,c,a], t1[pa,hc], optimize=True)
    O[a,c] += scale * +1.00000000 * np.einsum('uv,wuia,av->wi', gamma1, h2[a,a,c,v], t1[pv,ha], optimize=True)
    O[a,c] += scale * -1.00000000 * np.einsum('iuja,ai->uj', h2[c,a,c,v], t1[pv,hc], optimize=True)
    O[c,c] += scale * +1.00000000 * np.einsum('uv,ijkv,uj->ik', eta1, h2[c,c,c,a], t1[pa,hc], optimize=True)
    O[c,c] += scale * +1.00000000 * np.einsum('uv,iuja,av->ij', gamma1, h2[c,a,c,v], t1[pv,ha], optimize=True)
    O[c,c] += scale * +1.00000000 * np.einsum('ijka,aj->ik', h2[c,c,c,v], t1[pv,hc], optimize=True)
    O[v,c] += scale * -1.00000000 * np.einsum('uv,iajv,ui->aj', eta1, h2[c,v,c,a], t1[pa,hc], optimize=True)
    O[v,c] += scale * -1.00000000 * np.einsum('uv,uaib,bv->ai', gamma1, h2[a,v,c,v], t1[pv,ha], optimize=True)
    O[v,c] += scale * -1.00000000 * np.einsum('iajb,bi->aj', h2[c,v,c,v], t1[pv,hc], optimize=True)
    O[a,v] += scale * +1.00000000 * np.einsum('uv,iwva,ui->wa', eta1, h2[c,a,a,v], t1[pa,hc], optimize=True)
    O[a,v] += scale * +1.00000000 * np.einsum('uv,wuab,bv->wa', gamma1, h2[a,a,v,v], t1[pv,ha], optimize=True)
    O[a,v] += scale * -1.00000000 * np.einsum('iuab,bi->ua', h2[c,a,v,v], t1[pv,hc], optimize=True)
    O[c,v] += scale * -1.00000000 * np.einsum('uv,ijva,uj->ia', eta1, h2[c,c,a,v], t1[pa,hc], optimize=True)
    O[c,v] += scale * +1.00000000 * np.einsum('uv,iuab,bv->ia', gamma1, h2[c,a,v,v], t1[pv,ha], optimize=True)
    O[c,v] += scale * +1.00000000 * np.einsum('ijab,bj->ia', h2[c,c,v,v], t1[pv,hc], optimize=True)
    O[v,v] += scale * +1.00000000 * np.einsum('uv,iavb,ui->ab', eta1, h2[c,v,a,v], t1[pa,hc], optimize=True)
    O[v,v] += scale * -1.00000000 * np.einsum('uv,uabc,cv->ab', gamma1, h2[a,v,v,v], t1[pv,ha], optimize=True)
    O[v,v] += scale * -1.00000000 * np.einsum('iabc,ci->ab', h2[c,v,v,v], t1[pv,hc], optimize=True)

    t1 = time.time()
    if verbose: print("h2_t1_c1 took {:.4f} seconds to run.".format(t1-t0))

    return O
def h2_t2_c1(O, Hbar, t, gamma1, eta1, lambdas, orbspace, verbose=False, scale=1.0):
    # 158 lines
    t0 = time.time()
    c = orbspace['core']
    a = orbspace['active']
    v = orbspace['virt']
    hc = orbspace['hole_core']
    ha = orbspace['hole_active']
    pa = orbspace['particle_active']
    pv = orbspace['particle_virt']

    t1, t2 = t # Unpack cluster operator
    h1, h2 = Hbar # Unpack Hamiltonian
    lambda2 = lambdas['2'] # 2-cumulant
    # lambda3 = lambdas['3'] # 3-cumulant
    O[a,a] += scale * +0.50000000 * np.einsum('uv,wx,iyvx,uwiz->yz', eta1, eta1, h2[c,a,a,a], t2[pa,pa,hc,ha], optimize=True)
    O[a,a] += scale * -1.00000000 * np.einsum('uv,wx,iwyv,zuix->zy', eta1, gamma1, h2[c,a,a,a], t2[pa,pa,hc,ha], optimize=True)
    O[a,a] += scale * +1.00000000 * np.einsum('uv,wx,ywva,uazx->yz', eta1, gamma1, h2[a,a,a,v], t2[pa,pv,ha,ha], optimize=True)
    O[a,a] += scale * -0.50000000 * np.einsum('uv,ijwv,xuij->xw', eta1, h2[c,c,a,a], t2[pa,pa,hc,hc], optimize=True)
    O[a,a] += scale * +1.00000000 * np.einsum('uv,iwva,uaix->wx', eta1, h2[c,a,a,v], t2[pa,pv,hc,ha], optimize=True)
    O[a,a] += scale * -0.50000000 * np.einsum('uv,wx,uwya,zavx->zy', gamma1, gamma1, h2[a,a,a,v], t2[pa,pv,ha,ha], optimize=True)
    O[a,a] += scale * -1.00000000 * np.einsum('uv,iuwa,xaiv->xw', gamma1, h2[c,a,a,v], t2[pa,pv,hc,ha], optimize=True)
    O[a,a] += scale * +0.50000000 * np.einsum('uv,wuab,abxv->wx', gamma1, h2[a,a,v,v], t2[pv,pv,ha,ha], optimize=True)
    O[a,a] += scale * -0.50000000 * np.einsum('ijua,vaij->vu', h2[c,c,a,v], t2[pa,pv,hc,hc], optimize=True)
    O[a,a] += scale * +0.50000000 * np.einsum('iuvw,xywz,xyiz->uv', h2[c,a,a,a], lambda2, t2[pa,pa,hc,ha], optimize=True)
    O[a,a] += scale * +0.25000000 * np.einsum('iuvw,xyvw,xyiz->uz', h2[c,a,a,a], lambda2, t2[pa,pa,hc,ha], optimize=True)
    O[a,a] += scale * +0.50000000 * np.einsum('iuab,abiv->uv', h2[c,a,v,v], t2[pv,pv,hc,ha], optimize=True)
    O[a,a] += scale * -1.00000000 * np.einsum('iuvw,uxwy,zxiy->zv', h2[c,a,a,a], lambda2, t2[pa,pa,hc,ha], optimize=True)
    O[a,a] += scale * -0.50000000 * np.einsum('iuvw,uxvw,yxiz->yz', h2[c,a,a,a], lambda2, t2[pa,pa,hc,ha], optimize=True)
    O[a,a] += scale * -0.50000000 * np.einsum('uvwa,vxyz,xayz->uw', h2[a,a,a,v], lambda2, t2[pa,pv,ha,ha], optimize=True)
    O[a,a] += scale * +1.00000000 * np.einsum('uvwa,vxwy,xazy->uz', h2[a,a,a,v], lambda2, t2[pa,pv,ha,ha], optimize=True)
    O[a,a] += scale * -0.25000000 * np.einsum('uvwa,uvxy,zaxy->zw', h2[a,a,a,v], lambda2, t2[pa,pv,ha,ha], optimize=True)
    O[a,a] += scale * +0.50000000 * np.einsum('uvwa,uvwx,yazx->yz', h2[a,a,a,v], lambda2, t2[pa,pv,ha,ha], optimize=True)
    O[c,a] += scale * -0.50000000 * np.einsum('uv,wx,ijvx,uwjy->iy', eta1, eta1, h2[c,c,a,a], t2[pa,pa,hc,ha], optimize=True)
    O[c,a] += scale * +1.00000000 * np.einsum('uv,wx,iwva,uayx->iy', eta1, gamma1, h2[c,a,a,v], t2[pa,pv,ha,ha], optimize=True)
    O[c,a] += scale * -1.00000000 * np.einsum('uv,ijva,uajw->iw', eta1, h2[c,c,a,v], t2[pa,pv,hc,ha], optimize=True)
    O[c,a] += scale * +0.50000000 * np.einsum('uv,iuab,abwv->iw', gamma1, h2[c,a,v,v], t2[pv,pv,ha,ha], optimize=True)
    O[c,a] += scale * -0.50000000 * np.einsum('ijuv,wxvy,wxjy->iu', h2[c,c,a,a], lambda2, t2[pa,pa,hc,ha], optimize=True)
    O[c,a] += scale * -0.25000000 * np.einsum('ijuv,wxuv,wxjy->iy', h2[c,c,a,a], lambda2, t2[pa,pa,hc,ha], optimize=True)
    O[c,a] += scale * -0.50000000 * np.einsum('ijab,abju->iu', h2[c,c,v,v], t2[pv,pv,hc,ha], optimize=True)
    O[c,a] += scale * -0.50000000 * np.einsum('iuva,uwxy,waxy->iv', h2[c,a,a,v], lambda2, t2[pa,pv,ha,ha], optimize=True)
    O[c,a] += scale * +1.00000000 * np.einsum('iuva,uwvx,wayx->iy', h2[c,a,a,v], lambda2, t2[pa,pv,ha,ha], optimize=True)
    O[v,a] += scale * +0.50000000 * np.einsum('uv,wx,yz,uwrz,yavx->ar', eta1, eta1, gamma1, h2[a,a,a,a], t2[pa,pv,ha,ha], optimize=True)
    O[v,a] += scale * +0.50000000 * np.einsum('uv,wx,iavx,uwiy->ay', eta1, eta1, h2[c,v,a,a], t2[pa,pa,hc,ha], optimize=True)
    O[v,a] += scale * +0.50000000 * np.einsum('uv,wx,yz,wyrv,uaxz->ar', eta1, gamma1, gamma1, h2[a,a,a,a], t2[pa,pv,ha,ha], optimize=True)
    O[v,a] += scale * +1.00000000 * np.einsum('uv,wx,iwyv,uaix->ay', eta1, gamma1, h2[c,a,a,a], t2[pa,pv,hc,ha], optimize=True)
    O[v,a] += scale * -1.00000000 * np.einsum('uv,wx,wavb,ubyx->ay', eta1, gamma1, h2[a,v,a,v], t2[pa,pv,ha,ha], optimize=True)
    O[v,a] += scale * +0.50000000 * np.einsum('uv,ijwv,uaij->aw', eta1, h2[c,c,a,a], t2[pa,pv,hc,hc], optimize=True)
    O[v,a] += scale * +1.00000000 * np.einsum('uv,iavb,ubiw->aw', eta1, h2[c,v,a,v], t2[pa,pv,hc,ha], optimize=True)
    O[v,a] += scale * +1.00000000 * np.einsum('uv,uwxy,wzyr,zavr->ax', eta1, h2[a,a,a,a], lambda2, t2[pa,pv,ha,ha], optimize=True)
    O[v,a] += scale * -0.50000000 * np.einsum('uv,uwxy,wzxy,zarv->ar', eta1, h2[a,a,a,a], lambda2, t2[pa,pv,ha,ha], optimize=True)
    O[v,a] += scale * +0.25000000 * np.einsum('uv,wxyv,wxzr,uazr->ay', eta1, h2[a,a,a,a], lambda2, t2[pa,pv,ha,ha], optimize=True)
    O[v,a] += scale * +0.50000000 * np.einsum('uv,wxvy,wxyz,uarz->ar', eta1, h2[a,a,a,a], lambda2, t2[pa,pv,ha,ha], optimize=True)
    O[v,a] += scale * -0.50000000 * np.einsum('uv,wx,uwya,bavx->by', gamma1, gamma1, h2[a,a,a,v], t2[pv,pv,ha,ha], optimize=True)
    O[v,a] += scale * -1.00000000 * np.einsum('uv,iuwa,baiv->bw', gamma1, h2[c,a,a,v], t2[pv,pv,hc,ha], optimize=True)
    O[v,a] += scale * +1.00000000 * np.einsum('uv,uwxy,wzyr,zavr->ax', gamma1, h2[a,a,a,a], lambda2, t2[pa,pv,ha,ha], optimize=True)
    O[v,a] += scale * -0.50000000 * np.einsum('uv,uwxy,wzxy,zarv->ar', gamma1, h2[a,a,a,a], lambda2, t2[pa,pv,ha,ha], optimize=True)
    O[v,a] += scale * -0.50000000 * np.einsum('uv,uabc,bcwv->aw', gamma1, h2[a,v,v,v], t2[pv,pv,ha,ha], optimize=True)
    O[v,a] += scale * +0.25000000 * np.einsum('uv,wxyv,wxzr,uazr->ay', gamma1, h2[a,a,a,a], lambda2, t2[pa,pv,ha,ha], optimize=True)
    O[v,a] += scale * +0.50000000 * np.einsum('uv,wxvy,wxyz,uarz->ar', gamma1, h2[a,a,a,a], lambda2, t2[pa,pv,ha,ha], optimize=True)
    O[v,a] += scale * -0.50000000 * np.einsum('ijua,baij->bu', h2[c,c,a,v], t2[pv,pv,hc,hc], optimize=True)
    O[v,a] += scale * +1.00000000 * np.einsum('iuvw,uxwy,xaiy->av', h2[c,a,a,a], lambda2, t2[pa,pv,hc,ha], optimize=True)
    O[v,a] += scale * +0.50000000 * np.einsum('iuvw,uxvw,xaiy->ay', h2[c,a,a,a], lambda2, t2[pa,pv,hc,ha], optimize=True)
    O[v,a] += scale * +0.50000000 * np.einsum('iauv,wxvy,wxiy->au', h2[c,v,a,a], lambda2, t2[pa,pa,hc,ha], optimize=True)
    O[v,a] += scale * +0.25000000 * np.einsum('iauv,wxuv,wxiy->ay', h2[c,v,a,a], lambda2, t2[pa,pa,hc,ha], optimize=True)
    O[v,a] += scale * +0.50000000 * np.einsum('iabc,bciu->au', h2[c,v,v,v], t2[pv,pv,hc,ha], optimize=True)
    O[v,a] += scale * -0.25000000 * np.einsum('uvwa,uvxy,baxy->bw', h2[a,a,a,v], lambda2, t2[pv,pv,ha,ha], optimize=True)
    O[v,a] += scale * +0.50000000 * np.einsum('uvwa,uvwx,bayx->by', h2[a,a,a,v], lambda2, t2[pv,pv,ha,ha], optimize=True)
    O[v,a] += scale * +0.50000000 * np.einsum('uavb,uwxy,wbxy->av', h2[a,v,a,v], lambda2, t2[pa,pv,ha,ha], optimize=True)
    O[v,a] += scale * -1.00000000 * np.einsum('uavb,uwvx,wbyx->ay', h2[a,v,a,v], lambda2, t2[pa,pv,ha,ha], optimize=True)
    O[a,c] += scale * +0.50000000 * np.einsum('uv,wx,yz,ryvx,uwiz->ri', eta1, eta1, gamma1, h2[a,a,a,a], t2[pa,pa,hc,ha], optimize=True)
    O[a,c] += scale * -0.50000000 * np.einsum('uv,wx,iyvx,uwji->yj', eta1, eta1, h2[c,a,a,a], t2[pa,pa,hc,hc], optimize=True)
    O[a,c] += scale * +0.50000000 * np.einsum('uv,wx,yz,ruxz,wyiv->ri', eta1, gamma1, gamma1, h2[a,a,a,a], t2[pa,pa,hc,ha], optimize=True)
    O[a,c] += scale * -1.00000000 * np.einsum('uv,wx,iwjv,yuix->yj', eta1, gamma1, h2[c,a,c,a], t2[pa,pa,hc,ha], optimize=True)
    O[a,c] += scale * +1.00000000 * np.einsum('uv,wx,ywva,uaix->yi', eta1, gamma1, h2[a,a,a,v], t2[pa,pv,hc,ha], optimize=True)
    O[a,c] += scale * -0.50000000 * np.einsum('uv,ijkv,wuij->wk', eta1, h2[c,c,c,a], t2[pa,pa,hc,hc], optimize=True)
    O[a,c] += scale * -1.00000000 * np.einsum('uv,iwva,uaji->wj', eta1, h2[c,a,a,v], t2[pa,pv,hc,hc], optimize=True)
    O[a,c] += scale * +0.25000000 * np.einsum('uv,wuxy,zrxy,zriv->wi', eta1, h2[a,a,a,a], lambda2, t2[pa,pa,hc,ha], optimize=True)
    O[a,c] += scale * +1.00000000 * np.einsum('uv,wxvy,xzyr,uzir->wi', eta1, h2[a,a,a,a], lambda2, t2[pa,pa,hc,ha], optimize=True)
    O[a,c] += scale * +0.50000000 * np.einsum('uv,uwxy,wzxy,rziv->ri', eta1, h2[a,a,a,a], lambda2, t2[pa,pa,hc,ha], optimize=True)
    O[a,c] += scale * -0.50000000 * np.einsum('uv,wxvy,wxyz,ruiz->ri', eta1, h2[a,a,a,a], lambda2, t2[pa,pa,hc,ha], optimize=True)
    O[a,c] += scale * -0.50000000 * np.einsum('uv,wx,uwia,yavx->yi', gamma1, gamma1, h2[a,a,c,v], t2[pa,pv,ha,ha], optimize=True)
    O[a,c] += scale * -1.00000000 * np.einsum('uv,iuja,waiv->wj', gamma1, h2[c,a,c,v], t2[pa,pv,hc,ha], optimize=True)
    O[a,c] += scale * +0.25000000 * np.einsum('uv,wuxy,zrxy,zriv->wi', gamma1, h2[a,a,a,a], lambda2, t2[pa,pa,hc,ha], optimize=True)
    O[a,c] += scale * +0.50000000 * np.einsum('uv,wuab,abiv->wi', gamma1, h2[a,a,v,v], t2[pv,pv,hc,ha], optimize=True)
    O[a,c] += scale * +1.00000000 * np.einsum('uv,wxvy,xzyr,uzir->wi', gamma1, h2[a,a,a,a], lambda2, t2[pa,pa,hc,ha], optimize=True)
    O[a,c] += scale * +0.50000000 * np.einsum('uv,uwxy,wzxy,rziv->ri', gamma1, h2[a,a,a,a], lambda2, t2[pa,pa,hc,ha], optimize=True)
    O[a,c] += scale * -0.50000000 * np.einsum('uv,wxvy,wxyz,ruiz->ri', gamma1, h2[a,a,a,a], lambda2, t2[pa,pa,hc,ha], optimize=True)
    O[a,c] += scale * -0.50000000 * np.einsum('ijka,uaij->uk', h2[c,c,c,v], t2[pa,pv,hc,hc], optimize=True)
    O[a,c] += scale * +0.50000000 * np.einsum('iujv,wxvy,wxiy->uj', h2[c,a,c,a], lambda2, t2[pa,pa,hc,ha], optimize=True)
    O[a,c] += scale * -0.25000000 * np.einsum('iuvw,xyvw,xyji->uj', h2[c,a,a,a], lambda2, t2[pa,pa,hc,hc], optimize=True)
    O[a,c] += scale * -0.50000000 * np.einsum('iuab,abji->uj', h2[c,a,v,v], t2[pv,pv,hc,hc], optimize=True)
    O[a,c] += scale * -1.00000000 * np.einsum('iujv,uwvx,ywix->yj', h2[c,a,c,a], lambda2, t2[pa,pa,hc,ha], optimize=True)
    O[a,c] += scale * +0.50000000 * np.einsum('iuvw,uxvw,yxji->yj', h2[c,a,a,a], lambda2, t2[pa,pa,hc,hc], optimize=True)
    O[a,c] += scale * -0.50000000 * np.einsum('uvia,vwxy,waxy->ui', h2[a,a,c,v], lambda2, t2[pa,pv,ha,ha], optimize=True)
    O[a,c] += scale * +1.00000000 * np.einsum('uvwa,vxwy,xaiy->ui', h2[a,a,a,v], lambda2, t2[pa,pv,hc,ha], optimize=True)
    O[a,c] += scale * -0.25000000 * np.einsum('uvia,uvwx,yawx->yi', h2[a,a,c,v], lambda2, t2[pa,pv,ha,ha], optimize=True)
    O[a,c] += scale * +0.50000000 * np.einsum('uvwa,uvwx,yaix->yi', h2[a,a,a,v], lambda2, t2[pa,pv,hc,ha], optimize=True)
    O[c,c] += scale * +0.50000000 * np.einsum('uv,wx,yz,iyvx,uwjz->ij', eta1, eta1, gamma1, h2[c,a,a,a], t2[pa,pa,hc,ha], optimize=True)
    O[c,c] += scale * +0.50000000 * np.einsum('uv,wx,ijvx,uwkj->ik', eta1, eta1, h2[c,c,a,a], t2[pa,pa,hc,hc], optimize=True)
    O[c,c] += scale * +0.50000000 * np.einsum('uv,wx,yz,iuxz,wyjv->ij', eta1, gamma1, gamma1, h2[c,a,a,a], t2[pa,pa,hc,ha], optimize=True)
    O[c,c] += scale * +1.00000000 * np.einsum('uv,wx,iwva,uajx->ij', eta1, gamma1, h2[c,a,a,v], t2[pa,pv,hc,ha], optimize=True)
    O[c,c] += scale * +1.00000000 * np.einsum('uv,ijva,uakj->ik', eta1, h2[c,c,a,v], t2[pa,pv,hc,hc], optimize=True)
    O[c,c] += scale * +0.25000000 * np.einsum('uv,iuwx,yzwx,yzjv->ij', eta1, h2[c,a,a,a], lambda2, t2[pa,pa,hc,ha], optimize=True)
    O[c,c] += scale * +1.00000000 * np.einsum('uv,iwvx,wyxz,uyjz->ij', eta1, h2[c,a,a,a], lambda2, t2[pa,pa,hc,ha], optimize=True)
    O[c,c] += scale * +0.25000000 * np.einsum('uv,iuwx,yzwx,yzjv->ij', gamma1, h2[c,a,a,a], lambda2, t2[pa,pa,hc,ha], optimize=True)
    O[c,c] += scale * +0.50000000 * np.einsum('uv,iuab,abjv->ij', gamma1, h2[c,a,v,v], t2[pv,pv,hc,ha], optimize=True)
    O[c,c] += scale * +1.00000000 * np.einsum('uv,iwvx,wyxz,uyjz->ij', gamma1, h2[c,a,a,a], lambda2, t2[pa,pa,hc,ha], optimize=True)
    O[c,c] += scale * -0.50000000 * np.einsum('ijku,vwux,vwjx->ik', h2[c,c,c,a], lambda2, t2[pa,pa,hc,ha], optimize=True)
    O[c,c] += scale * +0.25000000 * np.einsum('ijuv,wxuv,wxkj->ik', h2[c,c,a,a], lambda2, t2[pa,pa,hc,hc], optimize=True)
    O[c,c] += scale * +0.50000000 * np.einsum('ijab,abkj->ik', h2[c,c,v,v], t2[pv,pv,hc,hc], optimize=True)
    O[c,c] += scale * -0.50000000 * np.einsum('iuja,uvwx,vawx->ij', h2[c,a,c,v], lambda2, t2[pa,pv,ha,ha], optimize=True)
    O[c,c] += scale * +1.00000000 * np.einsum('iuva,uwvx,wajx->ij', h2[c,a,a,v], lambda2, t2[pa,pv,hc,ha], optimize=True)
    O[v,c] += scale * +0.50000000 * np.einsum('uv,wx,yz,uwiz,yavx->ai', eta1, eta1, gamma1, h2[a,a,c,a], t2[pa,pv,ha,ha], optimize=True)
    O[v,c] += scale * -0.50000000 * np.einsum('uv,wx,yz,yavx,uwiz->ai', eta1, eta1, gamma1, h2[a,v,a,a], t2[pa,pa,hc,ha], optimize=True)
    O[v,c] += scale * -0.50000000 * np.einsum('uv,wx,iavx,uwji->aj', eta1, eta1, h2[c,v,a,a], t2[pa,pa,hc,hc], optimize=True)
    O[v,c] += scale * -0.50000000 * np.einsum('uv,wx,yz,uaxz,wyiv->ai', eta1, gamma1, gamma1, h2[a,v,a,a], t2[pa,pa,hc,ha], optimize=True)
    O[v,c] += scale * +0.50000000 * np.einsum('uv,wx,yz,wyiv,uaxz->ai', eta1, gamma1, gamma1, h2[a,a,c,a], t2[pa,pv,ha,ha], optimize=True)
    O[v,c] += scale * +1.00000000 * np.einsum('uv,wx,iwjv,uaix->aj', eta1, gamma1, h2[c,a,c,a], t2[pa,pv,hc,ha], optimize=True)
    O[v,c] += scale * -1.00000000 * np.einsum('uv,wx,wavb,ubix->ai', eta1, gamma1, h2[a,v,a,v], t2[pa,pv,hc,ha], optimize=True)
    O[v,c] += scale * +0.50000000 * np.einsum('uv,ijkv,uaij->ak', eta1, h2[c,c,c,a], t2[pa,pv,hc,hc], optimize=True)
    O[v,c] += scale * -1.00000000 * np.einsum('uv,iavb,ubji->aj', eta1, h2[c,v,a,v], t2[pa,pv,hc,hc], optimize=True)
    O[v,c] += scale * +1.00000000 * np.einsum('uv,uwix,wyxz,yavz->ai', eta1, h2[a,a,c,a], lambda2, t2[pa,pv,ha,ha], optimize=True)
    O[v,c] += scale * -0.50000000 * np.einsum('uv,uwxy,wzxy,zaiv->ai', eta1, h2[a,a,a,a], lambda2, t2[pa,pv,hc,ha], optimize=True)
    O[v,c] += scale * -0.25000000 * np.einsum('uv,uawx,yzwx,yziv->ai', eta1, h2[a,v,a,a], lambda2, t2[pa,pa,hc,ha], optimize=True)
    O[v,c] += scale * +0.25000000 * np.einsum('uv,wxiv,wxyz,uayz->ai', eta1, h2[a,a,c,a], lambda2, t2[pa,pv,ha,ha], optimize=True)
    O[v,c] += scale * +0.50000000 * np.einsum('uv,wxvy,wxyz,uaiz->ai', eta1, h2[a,a,a,a], lambda2, t2[pa,pv,hc,ha], optimize=True)
    O[v,c] += scale * -1.00000000 * np.einsum('uv,wavx,wyxz,uyiz->ai', eta1, h2[a,v,a,a], lambda2, t2[pa,pa,hc,ha], optimize=True)
    O[v,c] += scale * -0.50000000 * np.einsum('uv,wx,uwia,bavx->bi', gamma1, gamma1, h2[a,a,c,v], t2[pv,pv,ha,ha], optimize=True)
    O[v,c] += scale * -1.00000000 * np.einsum('uv,iuja,baiv->bj', gamma1, h2[c,a,c,v], t2[pv,pv,hc,ha], optimize=True)
    O[v,c] += scale * +1.00000000 * np.einsum('uv,uwix,wyxz,yavz->ai', gamma1, h2[a,a,c,a], lambda2, t2[pa,pv,ha,ha], optimize=True)
    O[v,c] += scale * -0.50000000 * np.einsum('uv,uwxy,wzxy,zaiv->ai', gamma1, h2[a,a,a,a], lambda2, t2[pa,pv,hc,ha], optimize=True)
    O[v,c] += scale * -0.25000000 * np.einsum('uv,uawx,yzwx,yziv->ai', gamma1, h2[a,v,a,a], lambda2, t2[pa,pa,hc,ha], optimize=True)
    O[v,c] += scale * -0.50000000 * np.einsum('uv,uabc,bciv->ai', gamma1, h2[a,v,v,v], t2[pv,pv,hc,ha], optimize=True)
    O[v,c] += scale * +0.25000000 * np.einsum('uv,wxiv,wxyz,uayz->ai', gamma1, h2[a,a,c,a], lambda2, t2[pa,pv,ha,ha], optimize=True)
    O[v,c] += scale * +0.50000000 * np.einsum('uv,wxvy,wxyz,uaiz->ai', gamma1, h2[a,a,a,a], lambda2, t2[pa,pv,hc,ha], optimize=True)
    O[v,c] += scale * -1.00000000 * np.einsum('uv,wavx,wyxz,uyiz->ai', gamma1, h2[a,v,a,a], lambda2, t2[pa,pa,hc,ha], optimize=True)
    O[v,c] += scale * -0.50000000 * np.einsum('ijka,baij->bk', h2[c,c,c,v], t2[pv,pv,hc,hc], optimize=True)
    O[v,c] += scale * +1.00000000 * np.einsum('iujv,uwvx,waix->aj', h2[c,a,c,a], lambda2, t2[pa,pv,hc,ha], optimize=True)
    O[v,c] += scale * -0.50000000 * np.einsum('iuvw,uxvw,xaji->aj', h2[c,a,a,a], lambda2, t2[pa,pv,hc,hc], optimize=True)
    O[v,c] += scale * +0.50000000 * np.einsum('iaju,vwux,vwix->aj', h2[c,v,c,a], lambda2, t2[pa,pa,hc,ha], optimize=True)
    O[v,c] += scale * -0.25000000 * np.einsum('iauv,wxuv,wxji->aj', h2[c,v,a,a], lambda2, t2[pa,pa,hc,hc], optimize=True)
    O[v,c] += scale * -0.50000000 * np.einsum('iabc,bcji->aj', h2[c,v,v,v], t2[pv,pv,hc,hc], optimize=True)
    O[v,c] += scale * -0.25000000 * np.einsum('uvia,uvwx,bawx->bi', h2[a,a,c,v], lambda2, t2[pv,pv,ha,ha], optimize=True)
    O[v,c] += scale * +0.50000000 * np.einsum('uvwa,uvwx,baix->bi', h2[a,a,a,v], lambda2, t2[pv,pv,hc,ha], optimize=True)
    O[v,c] += scale * +0.50000000 * np.einsum('uaib,uvwx,vbwx->ai', h2[a,v,c,v], lambda2, t2[pa,pv,ha,ha], optimize=True)
    O[v,c] += scale * -1.00000000 * np.einsum('uavb,uwvx,wbix->ai', h2[a,v,a,v], lambda2, t2[pa,pv,hc,ha], optimize=True)
    O[a,v] += scale * +1.00000000 * np.einsum('uv,wx,iwva,yuix->ya', eta1, gamma1, h2[c,a,a,v], t2[pa,pa,hc,ha], optimize=True)
    O[a,v] += scale * +0.50000000 * np.einsum('uv,ijva,wuij->wa', eta1, h2[c,c,a,v], t2[pa,pa,hc,hc], optimize=True)
    O[a,v] += scale * -0.50000000 * np.einsum('uv,wx,uwab,ybvx->ya', gamma1, gamma1, h2[a,a,v,v], t2[pa,pv,ha,ha], optimize=True)
    O[a,v] += scale * -1.00000000 * np.einsum('uv,iuab,wbiv->wa', gamma1, h2[c,a,v,v], t2[pa,pv,hc,ha], optimize=True)
    O[a,v] += scale * -0.50000000 * np.einsum('ijab,ubij->ua', h2[c,c,v,v], t2[pa,pv,hc,hc], optimize=True)
    O[a,v] += scale * -0.50000000 * np.einsum('iuva,wxvy,wxiy->ua', h2[c,a,a,v], lambda2, t2[pa,pa,hc,ha], optimize=True)
    O[a,v] += scale * +1.00000000 * np.einsum('iuva,uwvx,ywix->ya', h2[c,a,a,v], lambda2, t2[pa,pa,hc,ha], optimize=True)
    O[a,v] += scale * -0.50000000 * np.einsum('uvab,vwxy,wbxy->ua', h2[a,a,v,v], lambda2, t2[pa,pv,ha,ha], optimize=True)
    O[a,v] += scale * -0.25000000 * np.einsum('uvab,uvwx,ybwx->ya', h2[a,a,v,v], lambda2, t2[pa,pv,ha,ha], optimize=True)
    O[c,v] += scale * +0.50000000 * np.einsum('ijua,vwux,vwjx->ia', h2[c,c,a,v], lambda2, t2[pa,pa,hc,ha], optimize=True)
    O[c,v] += scale * -0.50000000 * np.einsum('iuab,uvwx,vbwx->ia', h2[c,a,v,v], lambda2, t2[pa,pv,ha,ha], optimize=True)
    O[v,v] += scale * -0.50000000 * np.einsum('uv,wx,yz,uwza,ybvx->ba', eta1, eta1, gamma1, h2[a,a,a,v], t2[pa,pv,ha,ha], optimize=True)
    O[v,v] += scale * -0.50000000 * np.einsum('uv,wx,yz,wyva,ubxz->ba', eta1, gamma1, gamma1, h2[a,a,a,v], t2[pa,pv,ha,ha], optimize=True)
    O[v,v] += scale * -1.00000000 * np.einsum('uv,wx,iwva,ubix->ba', eta1, gamma1, h2[c,a,a,v], t2[pa,pv,hc,ha], optimize=True)
    O[v,v] += scale * -0.50000000 * np.einsum('uv,ijva,ubij->ba', eta1, h2[c,c,a,v], t2[pa,pv,hc,hc], optimize=True)
    O[v,v] += scale * -1.00000000 * np.einsum('uv,uwxa,wyxz,ybvz->ba', eta1, h2[a,a,a,v], lambda2, t2[pa,pv,ha,ha], optimize=True)
    O[v,v] += scale * -0.25000000 * np.einsum('uv,wxva,wxyz,ubyz->ba', eta1, h2[a,a,a,v], lambda2, t2[pa,pv,ha,ha], optimize=True)
    O[v,v] += scale * -0.50000000 * np.einsum('uv,wx,uwab,cbvx->ca', gamma1, gamma1, h2[a,a,v,v], t2[pv,pv,ha,ha], optimize=True)
    O[v,v] += scale * -1.00000000 * np.einsum('uv,iuab,cbiv->ca', gamma1, h2[c,a,v,v], t2[pv,pv,hc,ha], optimize=True)
    O[v,v] += scale * -1.00000000 * np.einsum('uv,uwxa,wyxz,ybvz->ba', gamma1, h2[a,a,a,v], lambda2, t2[pa,pv,ha,ha], optimize=True)
    O[v,v] += scale * -0.25000000 * np.einsum('uv,wxva,wxyz,ubyz->ba', gamma1, h2[a,a,a,v], lambda2, t2[pa,pv,ha,ha], optimize=True)
    O[v,v] += scale * -0.50000000 * np.einsum('ijab,cbij->ca', h2[c,c,v,v], t2[pv,pv,hc,hc], optimize=True)
    O[v,v] += scale * -1.00000000 * np.einsum('iuva,uwvx,wbix->ba', h2[c,a,a,v], lambda2, t2[pa,pv,hc,ha], optimize=True)
    O[v,v] += scale * -0.50000000 * np.einsum('iaub,vwux,vwix->ab', h2[c,v,a,v], lambda2, t2[pa,pa,hc,ha], optimize=True)
    O[v,v] += scale * -0.25000000 * np.einsum('uvab,uvwx,cbwx->ca', h2[a,a,v,v], lambda2, t2[pv,pv,ha,ha], optimize=True)
    O[v,v] += scale * +0.50000000 * np.einsum('uabc,uvwx,vcwx->ab', h2[a,v,v,v], lambda2, t2[pa,pv,ha,ha], optimize=True)

    t1 = time.time()
    if verbose: print("h2_t2_c1 took {:.4f} seconds to run.".format(t1-t0))

    return O
def h1_t2_c2(O, Hbar, t, gamma1, eta1, lambdas, orbspace, verbose=False, scale=1.0):
    # 96 lines
    t0 = time.time()
    c = orbspace['core']
    a = orbspace['active']
    v = orbspace['virt']
    hc = orbspace['hole_core']
    ha = orbspace['hole_active']
    pa = orbspace['particle_active']
    pv = orbspace['particle_virt']

    t1, t2 = t # Unpack cluster operator
    h1, h2 = Hbar # Unpack Hamiltonian
    lambda2 = lambdas['2'] # 2-cumulant
    # lambda3 = lambdas['3'] # 3-cumulant
    O[a,a,a,a] += scale * -0.50000000 * np.einsum('iu,vwix->vwux', h1[c,a], t2[pa,pa,hc,ha], optimize=True)
    O[a,a,a,a] += scale * -0.50000000 * np.einsum('ua,vawx->uvwx', h1[a,v], t2[pa,pv,ha,ha], optimize=True)
    O[c,a,a,a] += scale * -0.50000000 * np.einsum('ia,uavw->iuvw', h1[c,v], t2[pa,pv,ha,ha], optimize=True)
    O[a,v,a,a] += scale * +0.50000000 * np.einsum('uv,wv,uaxy->waxy', eta1, h1[a,a], t2[pa,pv,ha,ha], optimize=True)
    O[a,v,a,a] += scale * +1.00000000 * np.einsum('uv,uw,xayv->xawy', eta1, h1[a,a], t2[pa,pv,ha,ha], optimize=True)
    O[a,v,a,a] += scale * +0.50000000 * np.einsum('uv,wv,uaxy->waxy', gamma1, h1[a,a], t2[pa,pv,ha,ha], optimize=True)
    O[a,v,a,a] += scale * +1.00000000 * np.einsum('uv,uw,xayv->xawy', gamma1, h1[a,a], t2[pa,pv,ha,ha], optimize=True)
    O[a,v,a,a] += scale * -1.00000000 * np.einsum('iu,vaiw->vauw', h1[c,a], t2[pa,pv,hc,ha], optimize=True)
    O[a,v,a,a] += scale * -0.50000000 * np.einsum('ua,bavw->ubvw', h1[a,v], t2[pv,pv,ha,ha], optimize=True)
    O[a,v,a,a] += scale * +0.50000000 * np.einsum('ab,ubvw->uavw', h1[v,v], t2[pa,pv,ha,ha], optimize=True)
    O[c,v,a,a] += scale * +0.50000000 * np.einsum('uv,iv,uawx->iawx', eta1, h1[c,a], t2[pa,pv,ha,ha], optimize=True)
    O[c,v,a,a] += scale * +0.50000000 * np.einsum('uv,iv,uawx->iawx', gamma1, h1[c,a], t2[pa,pv,ha,ha], optimize=True)
    O[c,v,a,a] += scale * -0.50000000 * np.einsum('ia,bauv->ibuv', h1[c,v], t2[pv,pv,ha,ha], optimize=True)
    O[v,v,a,a] += scale * +0.50000000 * np.einsum('uv,uw,abxv->abwx', eta1, h1[a,a], t2[pv,pv,ha,ha], optimize=True)
    O[v,v,a,a] += scale * +0.50000000 * np.einsum('uv,av,ubwx->abwx', eta1, h1[v,a], t2[pa,pv,ha,ha], optimize=True)
    O[v,v,a,a] += scale * +0.50000000 * np.einsum('uv,uw,abxv->abwx', gamma1, h1[a,a], t2[pv,pv,ha,ha], optimize=True)
    O[v,v,a,a] += scale * +0.50000000 * np.einsum('uv,av,ubwx->abwx', gamma1, h1[v,a], t2[pa,pv,ha,ha], optimize=True)
    O[v,v,a,a] += scale * -0.50000000 * np.einsum('iu,abiv->abuv', h1[c,a], t2[pv,pv,hc,ha], optimize=True)
    O[v,v,a,a] += scale * -0.50000000 * np.einsum('ab,cbuv->acuv', h1[v,v], t2[pv,pv,ha,ha], optimize=True)
    O[a,a,a,v] += scale * +0.50000000 * np.einsum('ia,uviw->uvwa', h1[c,v], t2[pa,pa,hc,ha], optimize=True)
    O[a,v,a,v] += scale * -1.00000000 * np.einsum('uv,ua,wbxv->wbxa', eta1, h1[a,v], t2[pa,pv,ha,ha], optimize=True)
    O[a,v,a,v] += scale * -1.00000000 * np.einsum('uv,ua,wbxv->wbxa', gamma1, h1[a,v], t2[pa,pv,ha,ha], optimize=True)
    O[a,v,a,v] += scale * +1.00000000 * np.einsum('ia,ubiv->ubva', h1[c,v], t2[pa,pv,hc,ha], optimize=True)
    O[v,v,a,v] += scale * -0.50000000 * np.einsum('uv,ua,bcwv->bcwa', eta1, h1[a,v], t2[pv,pv,ha,ha], optimize=True)
    O[v,v,a,v] += scale * -0.50000000 * np.einsum('uv,ua,bcwv->bcwa', gamma1, h1[a,v], t2[pv,pv,ha,ha], optimize=True)
    O[v,v,a,v] += scale * +0.50000000 * np.einsum('ia,bciu->bcua', h1[c,v], t2[pv,pv,hc,ha], optimize=True)
    O[a,a,c,a] += scale * -1.00000000 * np.einsum('uv,wv,xuiy->wxiy', eta1, h1[a,a], t2[pa,pa,hc,ha], optimize=True)
    O[a,a,c,a] += scale * -0.50000000 * np.einsum('uv,uw,xyiv->xyiw', eta1, h1[a,a], t2[pa,pa,hc,ha], optimize=True)
    O[a,a,c,a] += scale * -1.00000000 * np.einsum('uv,wv,xuiy->wxiy', gamma1, h1[a,a], t2[pa,pa,hc,ha], optimize=True)
    O[a,a,c,a] += scale * -0.50000000 * np.einsum('uv,uw,xyiv->xyiw', gamma1, h1[a,a], t2[pa,pa,hc,ha], optimize=True)
    O[a,a,c,a] += scale * -0.50000000 * np.einsum('ij,uviw->uvjw', h1[c,c], t2[pa,pa,hc,ha], optimize=True)
    O[a,a,c,a] += scale * -0.50000000 * np.einsum('iu,vwji->vwju', h1[c,a], t2[pa,pa,hc,hc], optimize=True)
    O[a,a,c,a] += scale * -1.00000000 * np.einsum('ua,vaiw->uviw', h1[a,v], t2[pa,pv,hc,ha], optimize=True)
    O[c,a,c,a] += scale * -1.00000000 * np.einsum('uv,iv,wujx->iwjx', eta1, h1[c,a], t2[pa,pa,hc,ha], optimize=True)
    O[c,a,c,a] += scale * -1.00000000 * np.einsum('uv,iv,wujx->iwjx', gamma1, h1[c,a], t2[pa,pa,hc,ha], optimize=True)
    O[c,a,c,a] += scale * -1.00000000 * np.einsum('ia,uajv->iujv', h1[c,v], t2[pa,pv,hc,ha], optimize=True)
    O[a,v,c,a] += scale * +1.00000000 * np.einsum('uv,wv,uaix->waix', eta1, h1[a,a], t2[pa,pv,hc,ha], optimize=True)
    O[a,v,c,a] += scale * +1.00000000 * np.einsum('uv,ui,waxv->waix', eta1, h1[a,c], t2[pa,pv,ha,ha], optimize=True)
    O[a,v,c,a] += scale * -1.00000000 * np.einsum('uv,uw,xaiv->xaiw', eta1, h1[a,a], t2[pa,pv,hc,ha], optimize=True)
    O[a,v,c,a] += scale * +1.00000000 * np.einsum('uv,av,wuix->waix', eta1, h1[v,a], t2[pa,pa,hc,ha], optimize=True)
    O[a,v,c,a] += scale * +1.00000000 * np.einsum('uv,wv,uaix->waix', gamma1, h1[a,a], t2[pa,pv,hc,ha], optimize=True)
    O[a,v,c,a] += scale * +1.00000000 * np.einsum('uv,ui,waxv->waix', gamma1, h1[a,c], t2[pa,pv,ha,ha], optimize=True)
    O[a,v,c,a] += scale * -1.00000000 * np.einsum('uv,uw,xaiv->xaiw', gamma1, h1[a,a], t2[pa,pv,hc,ha], optimize=True)
    O[a,v,c,a] += scale * +1.00000000 * np.einsum('uv,av,wuix->waix', gamma1, h1[v,a], t2[pa,pa,hc,ha], optimize=True)
    O[a,v,c,a] += scale * -1.00000000 * np.einsum('ij,uaiv->uajv', h1[c,c], t2[pa,pv,hc,ha], optimize=True)
    O[a,v,c,a] += scale * -1.00000000 * np.einsum('iu,vaji->vaju', h1[c,a], t2[pa,pv,hc,hc], optimize=True)
    O[a,v,c,a] += scale * -1.00000000 * np.einsum('ua,baiv->ubiv', h1[a,v], t2[pv,pv,hc,ha], optimize=True)
    O[a,v,c,a] += scale * +1.00000000 * np.einsum('ab,ubiv->uaiv', h1[v,v], t2[pa,pv,hc,ha], optimize=True)
    O[c,v,c,a] += scale * +1.00000000 * np.einsum('uv,iv,uajw->iajw', eta1, h1[c,a], t2[pa,pv,hc,ha], optimize=True)
    O[c,v,c,a] += scale * +1.00000000 * np.einsum('uv,iv,uajw->iajw', gamma1, h1[c,a], t2[pa,pv,hc,ha], optimize=True)
    O[c,v,c,a] += scale * -1.00000000 * np.einsum('ia,baju->ibju', h1[c,v], t2[pv,pv,hc,ha], optimize=True)
    O[v,v,c,a] += scale * +0.50000000 * np.einsum('uv,ui,abwv->abiw', eta1, h1[a,c], t2[pv,pv,ha,ha], optimize=True)
    O[v,v,c,a] += scale * -0.50000000 * np.einsum('uv,uw,abiv->abiw', eta1, h1[a,a], t2[pv,pv,hc,ha], optimize=True)
    O[v,v,c,a] += scale * +1.00000000 * np.einsum('uv,av,ubiw->abiw', eta1, h1[v,a], t2[pa,pv,hc,ha], optimize=True)
    O[v,v,c,a] += scale * +0.50000000 * np.einsum('uv,ui,abwv->abiw', gamma1, h1[a,c], t2[pv,pv,ha,ha], optimize=True)
    O[v,v,c,a] += scale * -0.50000000 * np.einsum('uv,uw,abiv->abiw', gamma1, h1[a,a], t2[pv,pv,hc,ha], optimize=True)
    O[v,v,c,a] += scale * +1.00000000 * np.einsum('uv,av,ubiw->abiw', gamma1, h1[v,a], t2[pa,pv,hc,ha], optimize=True)
    O[v,v,c,a] += scale * -0.50000000 * np.einsum('ij,abiu->abju', h1[c,c], t2[pv,pv,hc,ha], optimize=True)
    O[v,v,c,a] += scale * -0.50000000 * np.einsum('iu,abji->abju', h1[c,a], t2[pv,pv,hc,hc], optimize=True)
    O[v,v,c,a] += scale * -1.00000000 * np.einsum('ab,cbiu->aciu', h1[v,v], t2[pv,pv,hc,ha], optimize=True)
    O[a,a,c,c] += scale * -0.50000000 * np.einsum('uv,wv,xuij->wxij', eta1, h1[a,a], t2[pa,pa,hc,hc], optimize=True)
    O[a,a,c,c] += scale * +0.50000000 * np.einsum('uv,ui,wxjv->wxij', eta1, h1[a,c], t2[pa,pa,hc,ha], optimize=True)
    O[a,a,c,c] += scale * -0.50000000 * np.einsum('uv,wv,xuij->wxij', gamma1, h1[a,a], t2[pa,pa,hc,hc], optimize=True)
    O[a,a,c,c] += scale * +0.50000000 * np.einsum('uv,ui,wxjv->wxij', gamma1, h1[a,c], t2[pa,pa,hc,ha], optimize=True)
    O[a,a,c,c] += scale * +0.50000000 * np.einsum('ij,uvki->uvjk', h1[c,c], t2[pa,pa,hc,hc], optimize=True)
    O[a,a,c,c] += scale * -0.50000000 * np.einsum('ua,vaij->uvij', h1[a,v], t2[pa,pv,hc,hc], optimize=True)
    O[c,a,c,c] += scale * -0.50000000 * np.einsum('uv,iv,wujk->iwjk', eta1, h1[c,a], t2[pa,pa,hc,hc], optimize=True)
    O[c,a,c,c] += scale * -0.50000000 * np.einsum('uv,iv,wujk->iwjk', gamma1, h1[c,a], t2[pa,pa,hc,hc], optimize=True)
    O[c,a,c,c] += scale * -0.50000000 * np.einsum('ia,uajk->iujk', h1[c,v], t2[pa,pv,hc,hc], optimize=True)
    O[a,v,c,c] += scale * +0.50000000 * np.einsum('uv,wv,uaij->waij', eta1, h1[a,a], t2[pa,pv,hc,hc], optimize=True)
    O[a,v,c,c] += scale * +1.00000000 * np.einsum('uv,ui,wajv->waij', eta1, h1[a,c], t2[pa,pv,hc,ha], optimize=True)
    O[a,v,c,c] += scale * +0.50000000 * np.einsum('uv,av,wuij->waij', eta1, h1[v,a], t2[pa,pa,hc,hc], optimize=True)
    O[a,v,c,c] += scale * +0.50000000 * np.einsum('uv,wv,uaij->waij', gamma1, h1[a,a], t2[pa,pv,hc,hc], optimize=True)
    O[a,v,c,c] += scale * +1.00000000 * np.einsum('uv,ui,wajv->waij', gamma1, h1[a,c], t2[pa,pv,hc,ha], optimize=True)
    O[a,v,c,c] += scale * +0.50000000 * np.einsum('uv,av,wuij->waij', gamma1, h1[v,a], t2[pa,pa,hc,hc], optimize=True)
    O[a,v,c,c] += scale * +1.00000000 * np.einsum('ij,uaki->uajk', h1[c,c], t2[pa,pv,hc,hc], optimize=True)
    O[a,v,c,c] += scale * -0.50000000 * np.einsum('ua,baij->ubij', h1[a,v], t2[pv,pv,hc,hc], optimize=True)
    O[a,v,c,c] += scale * +0.50000000 * np.einsum('ab,ubij->uaij', h1[v,v], t2[pa,pv,hc,hc], optimize=True)
    O[c,v,c,c] += scale * +0.50000000 * np.einsum('uv,iv,uajk->iajk', eta1, h1[c,a], t2[pa,pv,hc,hc], optimize=True)
    O[c,v,c,c] += scale * +0.50000000 * np.einsum('uv,iv,uajk->iajk', gamma1, h1[c,a], t2[pa,pv,hc,hc], optimize=True)
    O[c,v,c,c] += scale * -0.50000000 * np.einsum('ia,bajk->ibjk', h1[c,v], t2[pv,pv,hc,hc], optimize=True)
    O[v,v,c,c] += scale * +0.50000000 * np.einsum('uv,ui,abjv->abij', eta1, h1[a,c], t2[pv,pv,hc,ha], optimize=True)
    O[v,v,c,c] += scale * +0.50000000 * np.einsum('uv,av,ubij->abij', eta1, h1[v,a], t2[pa,pv,hc,hc], optimize=True)
    O[v,v,c,c] += scale * +0.50000000 * np.einsum('uv,ui,abjv->abij', gamma1, h1[a,c], t2[pv,pv,hc,ha], optimize=True)
    O[v,v,c,c] += scale * +0.50000000 * np.einsum('uv,av,ubij->abij', gamma1, h1[v,a], t2[pa,pv,hc,hc], optimize=True)
    O[v,v,c,c] += scale * +0.50000000 * np.einsum('ij,abki->abjk', h1[c,c], t2[pv,pv,hc,hc], optimize=True)
    O[v,v,c,c] += scale * -0.50000000 * np.einsum('ab,cbij->acij', h1[v,v], t2[pv,pv,hc,hc], optimize=True)
    O[a,a,c,v] += scale * -0.50000000 * np.einsum('uv,ua,wxiv->wxia', eta1, h1[a,v], t2[pa,pa,hc,ha], optimize=True)
    O[a,a,c,v] += scale * -0.50000000 * np.einsum('uv,ua,wxiv->wxia', gamma1, h1[a,v], t2[pa,pa,hc,ha], optimize=True)
    O[a,a,c,v] += scale * -0.50000000 * np.einsum('ia,uvji->uvja', h1[c,v], t2[pa,pa,hc,hc], optimize=True)
    O[a,v,c,v] += scale * -1.00000000 * np.einsum('uv,ua,wbiv->wbia', eta1, h1[a,v], t2[pa,pv,hc,ha], optimize=True)
    O[a,v,c,v] += scale * -1.00000000 * np.einsum('uv,ua,wbiv->wbia', gamma1, h1[a,v], t2[pa,pv,hc,ha], optimize=True)
    O[a,v,c,v] += scale * -1.00000000 * np.einsum('ia,ubji->ubja', h1[c,v], t2[pa,pv,hc,hc], optimize=True)
    O[v,v,c,v] += scale * -0.50000000 * np.einsum('uv,ua,bciv->bcia', eta1, h1[a,v], t2[pv,pv,hc,ha], optimize=True)
    O[v,v,c,v] += scale * -0.50000000 * np.einsum('uv,ua,bciv->bcia', gamma1, h1[a,v], t2[pv,pv,hc,ha], optimize=True)
    O[v,v,c,v] += scale * -0.50000000 * np.einsum('ia,bcji->bcja', h1[c,v], t2[pv,pv,hc,hc], optimize=True)

    t1 = time.time()
    if verbose: print("h1_t2_c2 took {:.4f} seconds to run.".format(t1-t0))

    return O
def h2_t1_c2(O, Hbar, t, gamma1, eta1, lambdas, orbspace, verbose=False, scale=1.0):
    # 144 lines
    t0 = time.time()
    c = orbspace['core']
    a = orbspace['active']
    v = orbspace['virt']
    hc = orbspace['hole_core']
    ha = orbspace['hole_active']
    pa = orbspace['particle_active']
    pv = orbspace['particle_virt']

    t1, t2 = t # Unpack cluster operator
    h1, h2 = Hbar # Unpack Hamiltonian
    lambda2 = lambdas['2'] # 2-cumulant
    # lambda3 = lambdas['3'] # 3-cumulant
    O[a,a,a,a] += scale * +0.50000000 * np.einsum('iuvw,xi->uxvw', h2[c,a,a,a], t1[pa,hc], optimize=True)
    O[a,a,a,a] += scale * +0.50000000 * np.einsum('uvwa,ax->uvwx', h2[a,a,a,v], t1[pv,ha], optimize=True)
    O[c,a,a,a] += scale * -0.50000000 * np.einsum('ijuv,wj->iwuv', h2[c,c,a,a], t1[pa,hc], optimize=True)
    O[c,a,a,a] += scale * +1.00000000 * np.einsum('iuva,aw->iuvw', h2[c,a,a,v], t1[pv,ha], optimize=True)
    O[c,c,a,a] += scale * +0.50000000 * np.einsum('ijua,av->ijuv', h2[c,c,a,v], t1[pv,ha], optimize=True)
    O[a,v,a,a] += scale * -0.50000000 * np.einsum('uv,wuxy,av->waxy', eta1, h2[a,a,a,a], t1[pv,ha], optimize=True)
    O[a,v,a,a] += scale * -0.50000000 * np.einsum('uv,wuxy,av->waxy', gamma1, h2[a,a,a,a], t1[pv,ha], optimize=True)
    O[a,v,a,a] += scale * +0.50000000 * np.einsum('iuvw,ai->uavw', h2[c,a,a,a], t1[pv,hc], optimize=True)
    O[a,v,a,a] += scale * -0.50000000 * np.einsum('iauv,wi->wauv', h2[c,v,a,a], t1[pa,hc], optimize=True)
    O[a,v,a,a] += scale * +1.00000000 * np.einsum('uavb,bw->uavw', h2[a,v,a,v], t1[pv,ha], optimize=True)
    O[c,v,a,a] += scale * -0.50000000 * np.einsum('uv,iuwx,av->iawx', eta1, h2[c,a,a,a], t1[pv,ha], optimize=True)
    O[c,v,a,a] += scale * -0.50000000 * np.einsum('uv,iuwx,av->iawx', gamma1, h2[c,a,a,a], t1[pv,ha], optimize=True)
    O[c,v,a,a] += scale * -0.50000000 * np.einsum('ijuv,aj->iauv', h2[c,c,a,a], t1[pv,hc], optimize=True)
    O[c,v,a,a] += scale * +1.00000000 * np.einsum('iaub,bv->iauv', h2[c,v,a,v], t1[pv,ha], optimize=True)
    O[v,v,a,a] += scale * +0.50000000 * np.einsum('uv,uawx,bv->abwx', eta1, h2[a,v,a,a], t1[pv,ha], optimize=True)
    O[v,v,a,a] += scale * +0.50000000 * np.einsum('uv,uawx,bv->abwx', gamma1, h2[a,v,a,a], t1[pv,ha], optimize=True)
    O[v,v,a,a] += scale * +0.50000000 * np.einsum('iauv,bi->abuv', h2[c,v,a,a], t1[pv,hc], optimize=True)
    O[v,v,a,a] += scale * +0.50000000 * np.einsum('abuc,cv->abuv', h2[v,v,a,v], t1[pv,ha], optimize=True)
    O[a,a,a,v] += scale * +1.00000000 * np.einsum('iuva,wi->uwva', h2[c,a,a,v], t1[pa,hc], optimize=True)
    O[a,a,a,v] += scale * -0.50000000 * np.einsum('uvab,bw->uvwa', h2[a,a,v,v], t1[pv,ha], optimize=True)
    O[c,a,a,v] += scale * -1.00000000 * np.einsum('ijua,vj->ivua', h2[c,c,a,v], t1[pa,hc], optimize=True)
    O[c,a,a,v] += scale * -1.00000000 * np.einsum('iuab,bv->iuva', h2[c,a,v,v], t1[pv,ha], optimize=True)
    O[c,c,a,v] += scale * -0.50000000 * np.einsum('ijab,bu->ijua', h2[c,c,v,v], t1[pv,ha], optimize=True)
    O[a,v,a,v] += scale * -1.00000000 * np.einsum('uv,wuxa,bv->wbxa', eta1, h2[a,a,a,v], t1[pv,ha], optimize=True)
    O[a,v,a,v] += scale * -1.00000000 * np.einsum('uv,wuxa,bv->wbxa', gamma1, h2[a,a,a,v], t1[pv,ha], optimize=True)
    O[a,v,a,v] += scale * +1.00000000 * np.einsum('iuva,bi->ubva', h2[c,a,a,v], t1[pv,hc], optimize=True)
    O[a,v,a,v] += scale * -1.00000000 * np.einsum('iaub,vi->vaub', h2[c,v,a,v], t1[pa,hc], optimize=True)
    O[a,v,a,v] += scale * -1.00000000 * np.einsum('uabc,cv->uavb', h2[a,v,v,v], t1[pv,ha], optimize=True)
    O[c,v,a,v] += scale * -1.00000000 * np.einsum('uv,iuwa,bv->ibwa', eta1, h2[c,a,a,v], t1[pv,ha], optimize=True)
    O[c,v,a,v] += scale * -1.00000000 * np.einsum('uv,iuwa,bv->ibwa', gamma1, h2[c,a,a,v], t1[pv,ha], optimize=True)
    O[c,v,a,v] += scale * -1.00000000 * np.einsum('ijua,bj->ibua', h2[c,c,a,v], t1[pv,hc], optimize=True)
    O[c,v,a,v] += scale * -1.00000000 * np.einsum('iabc,cu->iaub', h2[c,v,v,v], t1[pv,ha], optimize=True)
    O[v,v,a,v] += scale * +1.00000000 * np.einsum('uv,uawb,cv->acwb', eta1, h2[a,v,a,v], t1[pv,ha], optimize=True)
    O[v,v,a,v] += scale * +1.00000000 * np.einsum('uv,uawb,cv->acwb', gamma1, h2[a,v,a,v], t1[pv,ha], optimize=True)
    O[v,v,a,v] += scale * +1.00000000 * np.einsum('iaub,ci->acub', h2[c,v,a,v], t1[pv,hc], optimize=True)
    O[v,v,a,v] += scale * -0.50000000 * np.einsum('abcd,du->abuc', h2[v,v,v,v], t1[pv,ha], optimize=True)
    O[a,a,c,a] += scale * -0.50000000 * np.einsum('uv,wxyv,ui->wxiy', eta1, h2[a,a,a,a], t1[pa,hc], optimize=True)
    O[a,a,c,a] += scale * -0.50000000 * np.einsum('uv,wxyv,ui->wxiy', gamma1, h2[a,a,a,a], t1[pa,hc], optimize=True)
    O[a,a,c,a] += scale * +1.00000000 * np.einsum('iujv,wi->uwjv', h2[c,a,c,a], t1[pa,hc], optimize=True)
    O[a,a,c,a] += scale * +0.50000000 * np.einsum('uvia,aw->uviw', h2[a,a,c,v], t1[pv,ha], optimize=True)
    O[a,a,c,a] += scale * -0.50000000 * np.einsum('uvwa,ai->uviw', h2[a,a,a,v], t1[pv,hc], optimize=True)
    O[c,a,c,a] += scale * -1.00000000 * np.einsum('uv,iwxv,uj->iwjx', eta1, h2[c,a,a,a], t1[pa,hc], optimize=True)
    O[c,a,c,a] += scale * -1.00000000 * np.einsum('uv,iwxv,uj->iwjx', gamma1, h2[c,a,a,a], t1[pa,hc], optimize=True)
    O[c,a,c,a] += scale * -1.00000000 * np.einsum('ijku,vj->ivku', h2[c,c,c,a], t1[pa,hc], optimize=True)
    O[c,a,c,a] += scale * +1.00000000 * np.einsum('iuja,av->iujv', h2[c,a,c,v], t1[pv,ha], optimize=True)
    O[c,a,c,a] += scale * -1.00000000 * np.einsum('iuva,aj->iujv', h2[c,a,a,v], t1[pv,hc], optimize=True)
    O[c,c,c,a] += scale * -0.50000000 * np.einsum('uv,ijwv,uk->ijkw', eta1, h2[c,c,a,a], t1[pa,hc], optimize=True)
    O[c,c,c,a] += scale * -0.50000000 * np.einsum('uv,ijwv,uk->ijkw', gamma1, h2[c,c,a,a], t1[pa,hc], optimize=True)
    O[c,c,c,a] += scale * +0.50000000 * np.einsum('ijka,au->ijku', h2[c,c,c,v], t1[pv,ha], optimize=True)
    O[c,c,c,a] += scale * -0.50000000 * np.einsum('ijua,ak->ijku', h2[c,c,a,v], t1[pv,hc], optimize=True)
    O[a,v,c,a] += scale * -1.00000000 * np.einsum('uv,wuix,av->waix', eta1, h2[a,a,c,a], t1[pv,ha], optimize=True)
    O[a,v,c,a] += scale * -1.00000000 * np.einsum('uv,waxv,ui->waix', eta1, h2[a,v,a,a], t1[pa,hc], optimize=True)
    O[a,v,c,a] += scale * -1.00000000 * np.einsum('uv,wuix,av->waix', gamma1, h2[a,a,c,a], t1[pv,ha], optimize=True)
    O[a,v,c,a] += scale * -1.00000000 * np.einsum('uv,waxv,ui->waix', gamma1, h2[a,v,a,a], t1[pa,hc], optimize=True)
    O[a,v,c,a] += scale * +1.00000000 * np.einsum('iujv,ai->uajv', h2[c,a,c,a], t1[pv,hc], optimize=True)
    O[a,v,c,a] += scale * -1.00000000 * np.einsum('iaju,vi->vaju', h2[c,v,c,a], t1[pa,hc], optimize=True)
    O[a,v,c,a] += scale * +1.00000000 * np.einsum('uaib,bv->uaiv', h2[a,v,c,v], t1[pv,ha], optimize=True)
    O[a,v,c,a] += scale * -1.00000000 * np.einsum('uavb,bi->uaiv', h2[a,v,a,v], t1[pv,hc], optimize=True)
    O[c,v,c,a] += scale * -1.00000000 * np.einsum('uv,iujw,av->iajw', eta1, h2[c,a,c,a], t1[pv,ha], optimize=True)
    O[c,v,c,a] += scale * -1.00000000 * np.einsum('uv,iawv,uj->iajw', eta1, h2[c,v,a,a], t1[pa,hc], optimize=True)
    O[c,v,c,a] += scale * -1.00000000 * np.einsum('uv,iujw,av->iajw', gamma1, h2[c,a,c,a], t1[pv,ha], optimize=True)
    O[c,v,c,a] += scale * -1.00000000 * np.einsum('uv,iawv,uj->iajw', gamma1, h2[c,v,a,a], t1[pa,hc], optimize=True)
    O[c,v,c,a] += scale * -1.00000000 * np.einsum('ijku,aj->iaku', h2[c,c,c,a], t1[pv,hc], optimize=True)
    O[c,v,c,a] += scale * +1.00000000 * np.einsum('iajb,bu->iaju', h2[c,v,c,v], t1[pv,ha], optimize=True)
    O[c,v,c,a] += scale * -1.00000000 * np.einsum('iaub,bj->iaju', h2[c,v,a,v], t1[pv,hc], optimize=True)
    O[v,v,c,a] += scale * +1.00000000 * np.einsum('uv,uaiw,bv->abiw', eta1, h2[a,v,c,a], t1[pv,ha], optimize=True)
    O[v,v,c,a] += scale * -0.50000000 * np.einsum('uv,abwv,ui->abiw', eta1, h2[v,v,a,a], t1[pa,hc], optimize=True)
    O[v,v,c,a] += scale * +1.00000000 * np.einsum('uv,uaiw,bv->abiw', gamma1, h2[a,v,c,a], t1[pv,ha], optimize=True)
    O[v,v,c,a] += scale * -0.50000000 * np.einsum('uv,abwv,ui->abiw', gamma1, h2[v,v,a,a], t1[pa,hc], optimize=True)
    O[v,v,c,a] += scale * +1.00000000 * np.einsum('iaju,bi->abju', h2[c,v,c,a], t1[pv,hc], optimize=True)
    O[v,v,c,a] += scale * +0.50000000 * np.einsum('abic,cu->abiu', h2[v,v,c,v], t1[pv,ha], optimize=True)
    O[v,v,c,a] += scale * -0.50000000 * np.einsum('abuc,ci->abiu', h2[v,v,a,v], t1[pv,hc], optimize=True)
    O[a,a,c,c] += scale * +0.50000000 * np.einsum('uv,wxiv,uj->wxij', eta1, h2[a,a,c,a], t1[pa,hc], optimize=True)
    O[a,a,c,c] += scale * +0.50000000 * np.einsum('uv,wxiv,uj->wxij', gamma1, h2[a,a,c,a], t1[pa,hc], optimize=True)
    O[a,a,c,c] += scale * +0.50000000 * np.einsum('iujk,vi->uvjk', h2[c,a,c,c], t1[pa,hc], optimize=True)
    O[a,a,c,c] += scale * +0.50000000 * np.einsum('uvia,aj->uvij', h2[a,a,c,v], t1[pv,hc], optimize=True)
    O[c,a,c,c] += scale * +1.00000000 * np.einsum('uv,iwjv,uk->iwjk', eta1, h2[c,a,c,a], t1[pa,hc], optimize=True)
    O[c,a,c,c] += scale * +1.00000000 * np.einsum('uv,iwjv,uk->iwjk', gamma1, h2[c,a,c,a], t1[pa,hc], optimize=True)
    O[c,a,c,c] += scale * -0.50000000 * np.einsum('ijkl,uj->iukl', h2[c,c,c,c], t1[pa,hc], optimize=True)
    O[c,a,c,c] += scale * +1.00000000 * np.einsum('iuja,ak->iujk', h2[c,a,c,v], t1[pv,hc], optimize=True)
    O[c,c,c,c] += scale * +0.50000000 * np.einsum('uv,ijkv,ul->ijkl', eta1, h2[c,c,c,a], t1[pa,hc], optimize=True)
    O[c,c,c,c] += scale * +0.50000000 * np.einsum('uv,ijkv,ul->ijkl', gamma1, h2[c,c,c,a], t1[pa,hc], optimize=True)
    O[c,c,c,c] += scale * +0.50000000 * np.einsum('ijka,al->ijkl', h2[c,c,c,v], t1[pv,hc], optimize=True)
    O[a,v,c,c] += scale * -0.50000000 * np.einsum('uv,wuij,av->waij', eta1, h2[a,a,c,c], t1[pv,ha], optimize=True)
    O[a,v,c,c] += scale * +1.00000000 * np.einsum('uv,waiv,uj->waij', eta1, h2[a,v,c,a], t1[pa,hc], optimize=True)
    O[a,v,c,c] += scale * -0.50000000 * np.einsum('uv,wuij,av->waij', gamma1, h2[a,a,c,c], t1[pv,ha], optimize=True)
    O[a,v,c,c] += scale * +1.00000000 * np.einsum('uv,waiv,uj->waij', gamma1, h2[a,v,c,a], t1[pa,hc], optimize=True)
    O[a,v,c,c] += scale * +0.50000000 * np.einsum('iujk,ai->uajk', h2[c,a,c,c], t1[pv,hc], optimize=True)
    O[a,v,c,c] += scale * -0.50000000 * np.einsum('iajk,ui->uajk', h2[c,v,c,c], t1[pa,hc], optimize=True)
    O[a,v,c,c] += scale * +1.00000000 * np.einsum('uaib,bj->uaij', h2[a,v,c,v], t1[pv,hc], optimize=True)
    O[c,v,c,c] += scale * -0.50000000 * np.einsum('uv,iujk,av->iajk', eta1, h2[c,a,c,c], t1[pv,ha], optimize=True)
    O[c,v,c,c] += scale * +1.00000000 * np.einsum('uv,iajv,uk->iajk', eta1, h2[c,v,c,a], t1[pa,hc], optimize=True)
    O[c,v,c,c] += scale * -0.50000000 * np.einsum('uv,iujk,av->iajk', gamma1, h2[c,a,c,c], t1[pv,ha], optimize=True)
    O[c,v,c,c] += scale * +1.00000000 * np.einsum('uv,iajv,uk->iajk', gamma1, h2[c,v,c,a], t1[pa,hc], optimize=True)
    O[c,v,c,c] += scale * -0.50000000 * np.einsum('ijkl,aj->iakl', h2[c,c,c,c], t1[pv,hc], optimize=True)
    O[c,v,c,c] += scale * +1.00000000 * np.einsum('iajb,bk->iajk', h2[c,v,c,v], t1[pv,hc], optimize=True)
    O[v,v,c,c] += scale * +0.50000000 * np.einsum('uv,uaij,bv->abij', eta1, h2[a,v,c,c], t1[pv,ha], optimize=True)
    O[v,v,c,c] += scale * +0.50000000 * np.einsum('uv,abiv,uj->abij', eta1, h2[v,v,c,a], t1[pa,hc], optimize=True)
    O[v,v,c,c] += scale * +0.50000000 * np.einsum('uv,uaij,bv->abij', gamma1, h2[a,v,c,c], t1[pv,ha], optimize=True)
    O[v,v,c,c] += scale * +0.50000000 * np.einsum('uv,abiv,uj->abij', gamma1, h2[v,v,c,a], t1[pa,hc], optimize=True)
    O[v,v,c,c] += scale * +0.50000000 * np.einsum('iajk,bi->abjk', h2[c,v,c,c], t1[pv,hc], optimize=True)
    O[v,v,c,c] += scale * +0.50000000 * np.einsum('abic,cj->abij', h2[v,v,c,v], t1[pv,hc], optimize=True)
    O[a,a,c,v] += scale * +0.50000000 * np.einsum('uv,wxva,ui->wxia', eta1, h2[a,a,a,v], t1[pa,hc], optimize=True)
    O[a,a,c,v] += scale * +0.50000000 * np.einsum('uv,wxva,ui->wxia', gamma1, h2[a,a,a,v], t1[pa,hc], optimize=True)
    O[a,a,c,v] += scale * +1.00000000 * np.einsum('iuja,vi->uvja', h2[c,a,c,v], t1[pa,hc], optimize=True)
    O[a,a,c,v] += scale * -0.50000000 * np.einsum('uvab,bi->uvia', h2[a,a,v,v], t1[pv,hc], optimize=True)
    O[c,a,c,v] += scale * +1.00000000 * np.einsum('uv,iwva,uj->iwja', eta1, h2[c,a,a,v], t1[pa,hc], optimize=True)
    O[c,a,c,v] += scale * +1.00000000 * np.einsum('uv,iwva,uj->iwja', gamma1, h2[c,a,a,v], t1[pa,hc], optimize=True)
    O[c,a,c,v] += scale * -1.00000000 * np.einsum('ijka,uj->iuka', h2[c,c,c,v], t1[pa,hc], optimize=True)
    O[c,a,c,v] += scale * -1.00000000 * np.einsum('iuab,bj->iuja', h2[c,a,v,v], t1[pv,hc], optimize=True)
    O[c,c,c,v] += scale * +0.50000000 * np.einsum('uv,ijva,uk->ijka', eta1, h2[c,c,a,v], t1[pa,hc], optimize=True)
    O[c,c,c,v] += scale * +0.50000000 * np.einsum('uv,ijva,uk->ijka', gamma1, h2[c,c,a,v], t1[pa,hc], optimize=True)
    O[c,c,c,v] += scale * -0.50000000 * np.einsum('ijab,bk->ijka', h2[c,c,v,v], t1[pv,hc], optimize=True)
    O[a,v,c,v] += scale * -1.00000000 * np.einsum('uv,wuia,bv->wbia', eta1, h2[a,a,c,v], t1[pv,ha], optimize=True)
    O[a,v,c,v] += scale * +1.00000000 * np.einsum('uv,wavb,ui->waib', eta1, h2[a,v,a,v], t1[pa,hc], optimize=True)
    O[a,v,c,v] += scale * -1.00000000 * np.einsum('uv,wuia,bv->wbia', gamma1, h2[a,a,c,v], t1[pv,ha], optimize=True)
    O[a,v,c,v] += scale * +1.00000000 * np.einsum('uv,wavb,ui->waib', gamma1, h2[a,v,a,v], t1[pa,hc], optimize=True)
    O[a,v,c,v] += scale * +1.00000000 * np.einsum('iuja,bi->ubja', h2[c,a,c,v], t1[pv,hc], optimize=True)
    O[a,v,c,v] += scale * -1.00000000 * np.einsum('iajb,ui->uajb', h2[c,v,c,v], t1[pa,hc], optimize=True)
    O[a,v,c,v] += scale * -1.00000000 * np.einsum('uabc,ci->uaib', h2[a,v,v,v], t1[pv,hc], optimize=True)
    O[c,v,c,v] += scale * -1.00000000 * np.einsum('uv,iuja,bv->ibja', eta1, h2[c,a,c,v], t1[pv,ha], optimize=True)
    O[c,v,c,v] += scale * +1.00000000 * np.einsum('uv,iavb,uj->iajb', eta1, h2[c,v,a,v], t1[pa,hc], optimize=True)
    O[c,v,c,v] += scale * -1.00000000 * np.einsum('uv,iuja,bv->ibja', gamma1, h2[c,a,c,v], t1[pv,ha], optimize=True)
    O[c,v,c,v] += scale * +1.00000000 * np.einsum('uv,iavb,uj->iajb', gamma1, h2[c,v,a,v], t1[pa,hc], optimize=True)
    O[c,v,c,v] += scale * -1.00000000 * np.einsum('ijka,bj->ibka', h2[c,c,c,v], t1[pv,hc], optimize=True)
    O[c,v,c,v] += scale * -1.00000000 * np.einsum('iabc,cj->iajb', h2[c,v,v,v], t1[pv,hc], optimize=True)
    O[v,v,c,v] += scale * +1.00000000 * np.einsum('uv,uaib,cv->acib', eta1, h2[a,v,c,v], t1[pv,ha], optimize=True)
    O[v,v,c,v] += scale * +0.50000000 * np.einsum('uv,abvc,ui->abic', eta1, h2[v,v,a,v], t1[pa,hc], optimize=True)
    O[v,v,c,v] += scale * +1.00000000 * np.einsum('uv,uaib,cv->acib', gamma1, h2[a,v,c,v], t1[pv,ha], optimize=True)
    O[v,v,c,v] += scale * +0.50000000 * np.einsum('uv,abvc,ui->abic', gamma1, h2[v,v,a,v], t1[pa,hc], optimize=True)
    O[v,v,c,v] += scale * +1.00000000 * np.einsum('iajb,ci->acjb', h2[c,v,c,v], t1[pv,hc], optimize=True)
    O[v,v,c,v] += scale * -0.50000000 * np.einsum('abcd,di->abic', h2[v,v,v,v], t1[pv,hc], optimize=True)
    O[a,a,v,v] += scale * +0.50000000 * np.einsum('iuab,vi->uvab', h2[c,a,v,v], t1[pa,hc], optimize=True)
    O[c,a,v,v] += scale * -0.50000000 * np.einsum('ijab,uj->iuab', h2[c,c,v,v], t1[pa,hc], optimize=True)
    O[a,v,v,v] += scale * -0.50000000 * np.einsum('uv,wuab,cv->wcab', eta1, h2[a,a,v,v], t1[pv,ha], optimize=True)
    O[a,v,v,v] += scale * -0.50000000 * np.einsum('uv,wuab,cv->wcab', gamma1, h2[a,a,v,v], t1[pv,ha], optimize=True)
    O[a,v,v,v] += scale * +0.50000000 * np.einsum('iuab,ci->ucab', h2[c,a,v,v], t1[pv,hc], optimize=True)
    O[a,v,v,v] += scale * -0.50000000 * np.einsum('iabc,ui->uabc', h2[c,v,v,v], t1[pa,hc], optimize=True)
    O[c,v,v,v] += scale * -0.50000000 * np.einsum('uv,iuab,cv->icab', eta1, h2[c,a,v,v], t1[pv,ha], optimize=True)
    O[c,v,v,v] += scale * -0.50000000 * np.einsum('uv,iuab,cv->icab', gamma1, h2[c,a,v,v], t1[pv,ha], optimize=True)
    O[c,v,v,v] += scale * -0.50000000 * np.einsum('ijab,cj->icab', h2[c,c,v,v], t1[pv,hc], optimize=True)
    O[v,v,v,v] += scale * +0.50000000 * np.einsum('uv,uabc,dv->adbc', eta1, h2[a,v,v,v], t1[pv,ha], optimize=True)
    O[v,v,v,v] += scale * +0.50000000 * np.einsum('uv,uabc,dv->adbc', gamma1, h2[a,v,v,v], t1[pv,ha], optimize=True)
    O[v,v,v,v] += scale * +0.50000000 * np.einsum('iabc,di->adbc', h2[c,v,v,v], t1[pv,hc], optimize=True)

    t1 = time.time()
    if verbose: print("h2_t1_c2 took {:.4f} seconds to run.".format(t1-t0))

    return O
def h2_t2_c2(O, Hbar, t, gamma1, eta1, lambdas, orbspace, verbose=False, scale=1.0):
    # 282 lines
    t0 = time.time()
    c = orbspace['core']
    a = orbspace['active']
    v = orbspace['virt']
    hc = orbspace['hole_core']
    ha = orbspace['hole_active']
    pa = orbspace['particle_active']
    pv = orbspace['particle_virt']

    t1, t2 = t # Unpack cluster operator
    h1, h2 = Hbar # Unpack Hamiltonian
    lambda2 = lambdas['2'] # 2-cumulant
    # lambda3 = lambdas['3'] # 3-cumulant
    O[a,a,a,a] += scale * +1.00000000 * np.einsum('uv,iwxv,yuiz->wyxz', eta1, h2[c,a,a,a], t2[pa,pa,hc,ha], optimize=True)
    O[a,a,a,a] += scale * +0.25000000 * np.einsum('uv,wxva,uayz->wxyz', eta1, h2[a,a,a,v], t2[pa,pv,ha,ha], optimize=True)
    O[a,a,a,a] += scale * +0.25000000 * np.einsum('uv,iuwx,yziv->yzwx', gamma1, h2[c,a,a,a], t2[pa,pa,hc,ha], optimize=True)
    O[a,a,a,a] += scale * +1.00000000 * np.einsum('uv,wuxa,yazv->wyxz', gamma1, h2[a,a,a,v], t2[pa,pv,ha,ha], optimize=True)
    O[a,a,a,a] += scale * +0.12500000 * np.einsum('ijuv,wxij->wxuv', h2[c,c,a,a], t2[pa,pa,hc,hc], optimize=True)
    O[a,a,a,a] += scale * +1.00000000 * np.einsum('iuva,waix->uwvx', h2[c,a,a,v], t2[pa,pv,hc,ha], optimize=True)
    O[a,a,a,a] += scale * +0.12500000 * np.einsum('uvab,abwx->uvwx', h2[a,a,v,v], t2[pv,pv,ha,ha], optimize=True)
    O[c,a,a,a] += scale * -1.00000000 * np.einsum('uv,ijwv,xujy->ixwy', eta1, h2[c,c,a,a], t2[pa,pa,hc,ha], optimize=True)
    O[c,a,a,a] += scale * +0.50000000 * np.einsum('uv,iwva,uaxy->iwxy', eta1, h2[c,a,a,v], t2[pa,pv,ha,ha], optimize=True)
    O[c,a,a,a] += scale * +1.00000000 * np.einsum('uv,iuwa,xayv->ixwy', gamma1, h2[c,a,a,v], t2[pa,pv,ha,ha], optimize=True)
    O[c,a,a,a] += scale * -1.00000000 * np.einsum('ijua,vajw->ivuw', h2[c,c,a,v], t2[pa,pv,hc,ha], optimize=True)
    O[c,a,a,a] += scale * +0.25000000 * np.einsum('iuab,abvw->iuvw', h2[c,a,v,v], t2[pv,pv,ha,ha], optimize=True)
    O[c,c,a,a] += scale * +0.25000000 * np.einsum('uv,ijva,uawx->ijwx', eta1, h2[c,c,a,v], t2[pa,pv,ha,ha], optimize=True)
    O[c,c,a,a] += scale * +0.12500000 * np.einsum('ijab,abuv->ijuv', h2[c,c,v,v], t2[pv,pv,ha,ha], optimize=True)
    O[a,v,a,a] += scale * -0.25000000 * np.einsum('uv,wx,uwyz,ravx->rayz', eta1, eta1, h2[a,a,a,a], t2[pa,pv,ha,ha], optimize=True)
    O[a,v,a,a] += scale * +1.00000000 * np.einsum('uv,wx,yuzx,warv->yazr', eta1, gamma1, h2[a,a,a,a], t2[pa,pv,ha,ha], optimize=True)
    O[a,v,a,a] += scale * -1.00000000 * np.einsum('uv,wx,ywzv,uarx->yazr', eta1, gamma1, h2[a,a,a,a], t2[pa,pv,ha,ha], optimize=True)
    O[a,v,a,a] += scale * -1.00000000 * np.einsum('uv,iwxv,uaiy->waxy', eta1, h2[c,a,a,a], t2[pa,pv,hc,ha], optimize=True)
    O[a,v,a,a] += scale * -1.00000000 * np.einsum('uv,iawv,xuiy->xawy', eta1, h2[c,v,a,a], t2[pa,pa,hc,ha], optimize=True)
    O[a,v,a,a] += scale * +0.50000000 * np.einsum('uv,wavb,ubxy->waxy', eta1, h2[a,v,a,v], t2[pa,pv,ha,ha], optimize=True)
    O[a,v,a,a] += scale * +0.25000000 * np.einsum('uv,wx,uwyz,ravx->rayz', gamma1, gamma1, h2[a,a,a,a], t2[pa,pv,ha,ha], optimize=True)
    O[a,v,a,a] += scale * +0.50000000 * np.einsum('uv,iuwx,yaiv->yawx', gamma1, h2[c,a,a,a], t2[pa,pv,hc,ha], optimize=True)
    O[a,v,a,a] += scale * +1.00000000 * np.einsum('uv,wuxa,bayv->wbxy', gamma1, h2[a,a,a,v], t2[pv,pv,ha,ha], optimize=True)
    O[a,v,a,a] += scale * +1.00000000 * np.einsum('uv,uawb,xbyv->xawy', gamma1, h2[a,v,a,v], t2[pa,pv,ha,ha], optimize=True)
    O[a,v,a,a] += scale * +0.25000000 * np.einsum('ijuv,waij->wauv', h2[c,c,a,a], t2[pa,pv,hc,hc], optimize=True)
    O[a,v,a,a] += scale * +1.00000000 * np.einsum('iuva,baiw->ubvw', h2[c,a,a,v], t2[pv,pv,hc,ha], optimize=True)
    O[a,v,a,a] += scale * -1.00000000 * np.einsum('iaub,vbiw->vauw', h2[c,v,a,v], t2[pa,pv,hc,ha], optimize=True)
    O[a,v,a,a] += scale * +0.25000000 * np.einsum('uabc,bcvw->uavw', h2[a,v,v,v], t2[pv,pv,ha,ha], optimize=True)
    O[c,v,a,a] += scale * +1.00000000 * np.einsum('uv,wx,iuyx,wazv->iayz', eta1, gamma1, h2[c,a,a,a], t2[pa,pv,ha,ha], optimize=True)
    O[c,v,a,a] += scale * -1.00000000 * np.einsum('uv,wx,iwyv,uazx->iayz', eta1, gamma1, h2[c,a,a,a], t2[pa,pv,ha,ha], optimize=True)
    O[c,v,a,a] += scale * +1.00000000 * np.einsum('uv,ijwv,uajx->iawx', eta1, h2[c,c,a,a], t2[pa,pv,hc,ha], optimize=True)
    O[c,v,a,a] += scale * +0.50000000 * np.einsum('uv,iavb,ubwx->iawx', eta1, h2[c,v,a,v], t2[pa,pv,ha,ha], optimize=True)
    O[c,v,a,a] += scale * +1.00000000 * np.einsum('uv,iuwa,baxv->ibwx', gamma1, h2[c,a,a,v], t2[pv,pv,ha,ha], optimize=True)
    O[c,v,a,a] += scale * -1.00000000 * np.einsum('ijua,bajv->ibuv', h2[c,c,a,v], t2[pv,pv,hc,ha], optimize=True)
    O[c,v,a,a] += scale * +0.25000000 * np.einsum('iabc,bcuv->iauv', h2[c,v,v,v], t2[pv,pv,ha,ha], optimize=True)
    O[v,v,a,a] += scale * -0.12500000 * np.einsum('uv,wx,uwyz,abvx->abyz', eta1, eta1, h2[a,a,a,a], t2[pv,pv,ha,ha], optimize=True)
    O[v,v,a,a] += scale * -1.00000000 * np.einsum('uv,wx,uayx,wbzv->abyz', eta1, gamma1, h2[a,v,a,a], t2[pa,pv,ha,ha], optimize=True)
    O[v,v,a,a] += scale * +1.00000000 * np.einsum('uv,wx,wayv,ubzx->abyz', eta1, gamma1, h2[a,v,a,a], t2[pa,pv,ha,ha], optimize=True)
    O[v,v,a,a] += scale * -1.00000000 * np.einsum('uv,iawv,ubix->abwx', eta1, h2[c,v,a,a], t2[pa,pv,hc,ha], optimize=True)
    O[v,v,a,a] += scale * +0.25000000 * np.einsum('uv,abvc,ucwx->abwx', eta1, h2[v,v,a,v], t2[pa,pv,ha,ha], optimize=True)
    O[v,v,a,a] += scale * +0.12500000 * np.einsum('uv,wx,uwyz,abvx->abyz', gamma1, gamma1, h2[a,a,a,a], t2[pv,pv,ha,ha], optimize=True)
    O[v,v,a,a] += scale * +0.25000000 * np.einsum('uv,iuwx,abiv->abwx', gamma1, h2[c,a,a,a], t2[pv,pv,hc,ha], optimize=True)
    O[v,v,a,a] += scale * -1.00000000 * np.einsum('uv,uawb,cbxv->acwx', gamma1, h2[a,v,a,v], t2[pv,pv,ha,ha], optimize=True)
    O[v,v,a,a] += scale * +0.12500000 * np.einsum('ijuv,abij->abuv', h2[c,c,a,a], t2[pv,pv,hc,hc], optimize=True)
    O[v,v,a,a] += scale * +1.00000000 * np.einsum('iaub,cbiv->acuv', h2[c,v,a,v], t2[pv,pv,hc,ha], optimize=True)
    O[v,v,a,a] += scale * +0.12500000 * np.einsum('abcd,cduv->abuv', h2[v,v,v,v], t2[pv,pv,ha,ha], optimize=True)
    O[a,a,a,v] += scale * +1.00000000 * np.einsum('uv,iwva,xuiy->wxya', eta1, h2[c,a,a,v], t2[pa,pa,hc,ha], optimize=True)
    O[a,a,a,v] += scale * +0.50000000 * np.einsum('uv,iuwa,xyiv->xywa', gamma1, h2[c,a,a,v], t2[pa,pa,hc,ha], optimize=True)
    O[a,a,a,v] += scale * -1.00000000 * np.einsum('uv,wuab,xbyv->wxya', gamma1, h2[a,a,v,v], t2[pa,pv,ha,ha], optimize=True)
    O[a,a,a,v] += scale * +0.25000000 * np.einsum('ijua,vwij->vwua', h2[c,c,a,v], t2[pa,pa,hc,hc], optimize=True)
    O[a,a,a,v] += scale * -1.00000000 * np.einsum('iuab,vbiw->uvwa', h2[c,a,v,v], t2[pa,pv,hc,ha], optimize=True)
    O[c,a,a,v] += scale * -1.00000000 * np.einsum('uv,ijva,wujx->iwxa', eta1, h2[c,c,a,v], t2[pa,pa,hc,ha], optimize=True)
    O[c,a,a,v] += scale * -1.00000000 * np.einsum('uv,iuab,wbxv->iwxa', gamma1, h2[c,a,v,v], t2[pa,pv,ha,ha], optimize=True)
    O[c,a,a,v] += scale * +1.00000000 * np.einsum('ijab,ubjv->iuva', h2[c,c,v,v], t2[pa,pv,hc,ha], optimize=True)
    O[a,v,a,v] += scale * -0.50000000 * np.einsum('uv,wx,uwya,zbvx->zbya', eta1, eta1, h2[a,a,a,v], t2[pa,pv,ha,ha], optimize=True)
    O[a,v,a,v] += scale * +1.00000000 * np.einsum('uv,wx,yuxa,wbzv->ybza', eta1, gamma1, h2[a,a,a,v], t2[pa,pv,ha,ha], optimize=True)
    O[a,v,a,v] += scale * -1.00000000 * np.einsum('uv,wx,ywva,ubzx->ybza', eta1, gamma1, h2[a,a,a,v], t2[pa,pv,ha,ha], optimize=True)
    O[a,v,a,v] += scale * -1.00000000 * np.einsum('uv,iwva,ubix->wbxa', eta1, h2[c,a,a,v], t2[pa,pv,hc,ha], optimize=True)
    O[a,v,a,v] += scale * -1.00000000 * np.einsum('uv,iavb,wuix->waxb', eta1, h2[c,v,a,v], t2[pa,pa,hc,ha], optimize=True)
    O[a,v,a,v] += scale * +0.50000000 * np.einsum('uv,wx,uwya,zbvx->zbya', gamma1, gamma1, h2[a,a,a,v], t2[pa,pv,ha,ha], optimize=True)
    O[a,v,a,v] += scale * +1.00000000 * np.einsum('uv,iuwa,xbiv->xbwa', gamma1, h2[c,a,a,v], t2[pa,pv,hc,ha], optimize=True)
    O[a,v,a,v] += scale * -1.00000000 * np.einsum('uv,wuab,cbxv->wcxa', gamma1, h2[a,a,v,v], t2[pv,pv,ha,ha], optimize=True)
    O[a,v,a,v] += scale * -1.00000000 * np.einsum('uv,uabc,wcxv->waxb', gamma1, h2[a,v,v,v], t2[pa,pv,ha,ha], optimize=True)
    O[a,v,a,v] += scale * +0.50000000 * np.einsum('ijua,vbij->vbua', h2[c,c,a,v], t2[pa,pv,hc,hc], optimize=True)
    O[a,v,a,v] += scale * -1.00000000 * np.einsum('iuab,cbiv->ucva', h2[c,a,v,v], t2[pv,pv,hc,ha], optimize=True)
    O[a,v,a,v] += scale * +1.00000000 * np.einsum('iabc,uciv->uavb', h2[c,v,v,v], t2[pa,pv,hc,ha], optimize=True)
    O[c,v,a,v] += scale * +1.00000000 * np.einsum('uv,wx,iuxa,wbyv->ibya', eta1, gamma1, h2[c,a,a,v], t2[pa,pv,ha,ha], optimize=True)
    O[c,v,a,v] += scale * -1.00000000 * np.einsum('uv,wx,iwva,ubyx->ibya', eta1, gamma1, h2[c,a,a,v], t2[pa,pv,ha,ha], optimize=True)
    O[c,v,a,v] += scale * +1.00000000 * np.einsum('uv,ijva,ubjw->ibwa', eta1, h2[c,c,a,v], t2[pa,pv,hc,ha], optimize=True)
    O[c,v,a,v] += scale * -1.00000000 * np.einsum('uv,iuab,cbwv->icwa', gamma1, h2[c,a,v,v], t2[pv,pv,ha,ha], optimize=True)
    O[c,v,a,v] += scale * +1.00000000 * np.einsum('ijab,cbju->icua', h2[c,c,v,v], t2[pv,pv,hc,ha], optimize=True)
    O[v,v,a,v] += scale * -0.25000000 * np.einsum('uv,wx,uwya,bcvx->bcya', eta1, eta1, h2[a,a,a,v], t2[pv,pv,ha,ha], optimize=True)
    O[v,v,a,v] += scale * -1.00000000 * np.einsum('uv,wx,uaxb,wcyv->acyb', eta1, gamma1, h2[a,v,a,v], t2[pa,pv,ha,ha], optimize=True)
    O[v,v,a,v] += scale * +1.00000000 * np.einsum('uv,wx,wavb,ucyx->acyb', eta1, gamma1, h2[a,v,a,v], t2[pa,pv,ha,ha], optimize=True)
    O[v,v,a,v] += scale * -1.00000000 * np.einsum('uv,iavb,uciw->acwb', eta1, h2[c,v,a,v], t2[pa,pv,hc,ha], optimize=True)
    O[v,v,a,v] += scale * +0.25000000 * np.einsum('uv,wx,uwya,bcvx->bcya', gamma1, gamma1, h2[a,a,a,v], t2[pv,pv,ha,ha], optimize=True)
    O[v,v,a,v] += scale * +0.50000000 * np.einsum('uv,iuwa,bciv->bcwa', gamma1, h2[c,a,a,v], t2[pv,pv,hc,ha], optimize=True)
    O[v,v,a,v] += scale * +1.00000000 * np.einsum('uv,uabc,dcwv->adwb', gamma1, h2[a,v,v,v], t2[pv,pv,ha,ha], optimize=True)
    O[v,v,a,v] += scale * +0.25000000 * np.einsum('ijua,bcij->bcua', h2[c,c,a,v], t2[pv,pv,hc,hc], optimize=True)
    O[v,v,a,v] += scale * -1.00000000 * np.einsum('iabc,dciu->adub', h2[c,v,v,v], t2[pv,pv,hc,ha], optimize=True)
    O[a,a,c,a] += scale * +0.25000000 * np.einsum('uv,wx,yzvx,uwir->yzir', eta1, eta1, h2[a,a,a,a], t2[pa,pa,hc,ha], optimize=True)
    O[a,a,c,a] += scale * +1.00000000 * np.einsum('uv,wx,yuzx,rwiv->yriz', eta1, gamma1, h2[a,a,a,a], t2[pa,pa,hc,ha], optimize=True)
    O[a,a,c,a] += scale * -1.00000000 * np.einsum('uv,wx,ywzv,ruix->yriz', eta1, gamma1, h2[a,a,a,a], t2[pa,pa,hc,ha], optimize=True)
    O[a,a,c,a] += scale * +1.00000000 * np.einsum('uv,iwjv,xuiy->wxjy', eta1, h2[c,a,c,a], t2[pa,pa,hc,ha], optimize=True)
    O[a,a,c,a] += scale * +1.00000000 * np.einsum('uv,iwxv,yuji->wyjx', eta1, h2[c,a,a,a], t2[pa,pa,hc,hc], optimize=True)
    O[a,a,c,a] += scale * +0.50000000 * np.einsum('uv,wxva,uaiy->wxiy', eta1, h2[a,a,a,v], t2[pa,pv,hc,ha], optimize=True)
    O[a,a,c,a] += scale * -0.25000000 * np.einsum('uv,wx,yzvx,uwir->yzir', gamma1, gamma1, h2[a,a,a,a], t2[pa,pa,hc,ha], optimize=True)
    O[a,a,c,a] += scale * +0.50000000 * np.einsum('uv,iujw,xyiv->xyjw', gamma1, h2[c,a,c,a], t2[pa,pa,hc,ha], optimize=True)
    O[a,a,c,a] += scale * +1.00000000 * np.einsum('uv,wuia,xayv->wxiy', gamma1, h2[a,a,c,v], t2[pa,pv,ha,ha], optimize=True)
    O[a,a,c,a] += scale * -1.00000000 * np.einsum('uv,wuxa,yaiv->wyix', gamma1, h2[a,a,a,v], t2[pa,pv,hc,ha], optimize=True)
    O[a,a,c,a] += scale * +0.25000000 * np.einsum('ijku,vwij->vwku', h2[c,c,c,a], t2[pa,pa,hc,hc], optimize=True)
    O[a,a,c,a] += scale * +1.00000000 * np.einsum('iuja,vaiw->uvjw', h2[c,a,c,v], t2[pa,pv,hc,ha], optimize=True)
    O[a,a,c,a] += scale * +1.00000000 * np.einsum('iuva,waji->uwjv', h2[c,a,a,v], t2[pa,pv,hc,hc], optimize=True)
    O[a,a,c,a] += scale * +0.25000000 * np.einsum('uvab,abiw->uviw', h2[a,a,v,v], t2[pv,pv,hc,ha], optimize=True)
    O[c,a,c,a] += scale * +0.50000000 * np.einsum('uv,wx,iyvx,uwjz->iyjz', eta1, eta1, h2[c,a,a,a], t2[pa,pa,hc,ha], optimize=True)
    O[c,a,c,a] += scale * +1.00000000 * np.einsum('uv,wx,iuyx,zwjv->izjy', eta1, gamma1, h2[c,a,a,a], t2[pa,pa,hc,ha], optimize=True)
    O[c,a,c,a] += scale * -1.00000000 * np.einsum('uv,wx,iwyv,zujx->izjy', eta1, gamma1, h2[c,a,a,a], t2[pa,pa,hc,ha], optimize=True)
    O[c,a,c,a] += scale * -1.00000000 * np.einsum('uv,ijkv,wujx->iwkx', eta1, h2[c,c,c,a], t2[pa,pa,hc,ha], optimize=True)
    O[c,a,c,a] += scale * -1.00000000 * np.einsum('uv,ijwv,xukj->ixkw', eta1, h2[c,c,a,a], t2[pa,pa,hc,hc], optimize=True)
    O[c,a,c,a] += scale * +1.00000000 * np.einsum('uv,iwva,uajx->iwjx', eta1, h2[c,a,a,v], t2[pa,pv,hc,ha], optimize=True)
    O[c,a,c,a] += scale * -0.50000000 * np.einsum('uv,wx,iyvx,uwjz->iyjz', gamma1, gamma1, h2[c,a,a,a], t2[pa,pa,hc,ha], optimize=True)
    O[c,a,c,a] += scale * +1.00000000 * np.einsum('uv,iuja,waxv->iwjx', gamma1, h2[c,a,c,v], t2[pa,pv,ha,ha], optimize=True)
    O[c,a,c,a] += scale * -1.00000000 * np.einsum('uv,iuwa,xajv->ixjw', gamma1, h2[c,a,a,v], t2[pa,pv,hc,ha], optimize=True)
    O[c,a,c,a] += scale * -1.00000000 * np.einsum('ijka,uajv->iukv', h2[c,c,c,v], t2[pa,pv,hc,ha], optimize=True)
    O[c,a,c,a] += scale * -1.00000000 * np.einsum('ijua,vakj->ivku', h2[c,c,a,v], t2[pa,pv,hc,hc], optimize=True)
    O[c,a,c,a] += scale * +0.50000000 * np.einsum('iuab,abjv->iujv', h2[c,a,v,v], t2[pv,pv,hc,ha], optimize=True)
    O[c,c,c,a] += scale * +0.25000000 * np.einsum('uv,wx,ijvx,uwky->ijky', eta1, eta1, h2[c,c,a,a], t2[pa,pa,hc,ha], optimize=True)
    O[c,c,c,a] += scale * +0.50000000 * np.einsum('uv,ijva,uakw->ijkw', eta1, h2[c,c,a,v], t2[pa,pv,hc,ha], optimize=True)
    O[c,c,c,a] += scale * -0.25000000 * np.einsum('uv,wx,ijvx,uwky->ijky', gamma1, gamma1, h2[c,c,a,a], t2[pa,pa,hc,ha], optimize=True)
    O[c,c,c,a] += scale * +0.25000000 * np.einsum('ijab,abku->ijku', h2[c,c,v,v], t2[pv,pv,hc,ha], optimize=True)
    O[a,v,c,a] += scale * +0.50000000 * np.einsum('uv,wx,yavx,uwiz->yaiz', eta1, eta1, h2[a,v,a,a], t2[pa,pa,hc,ha], optimize=True)
    O[a,v,c,a] += scale * -0.50000000 * np.einsum('uv,wx,uwiy,zavx->zaiy', eta1, eta1, h2[a,a,c,a], t2[pa,pv,ha,ha], optimize=True)
    O[a,v,c,a] += scale * +1.00000000 * np.einsum('uv,wx,yuix,wazv->yaiz', eta1, gamma1, h2[a,a,c,a], t2[pa,pv,ha,ha], optimize=True)
    O[a,v,c,a] += scale * -1.00000000 * np.einsum('uv,wx,yuzx,waiv->yaiz', eta1, gamma1, h2[a,a,a,a], t2[pa,pv,hc,ha], optimize=True)
    O[a,v,c,a] += scale * -1.00000000 * np.einsum('uv,wx,ywiv,uazx->yaiz', eta1, gamma1, h2[a,a,c,a], t2[pa,pv,ha,ha], optimize=True)
    O[a,v,c,a] += scale * +1.00000000 * np.einsum('uv,wx,ywzv,uaix->yaiz', eta1, gamma1, h2[a,a,a,a], t2[pa,pv,hc,ha], optimize=True)
    O[a,v,c,a] += scale * +1.00000000 * np.einsum('uv,wx,uayx,zwiv->zaiy', eta1, gamma1, h2[a,v,a,a], t2[pa,pa,hc,ha], optimize=True)
    O[a,v,c,a] += scale * -1.00000000 * np.einsum('uv,wx,wayv,zuix->zaiy', eta1, gamma1, h2[a,v,a,a], t2[pa,pa,hc,ha], optimize=True)
    O[a,v,c,a] += scale * -1.00000000 * np.einsum('uv,iwjv,uaix->wajx', eta1, h2[c,a,c,a], t2[pa,pv,hc,ha], optimize=True)
    O[a,v,c,a] += scale * -1.00000000 * np.einsum('uv,iwxv,uaji->wajx', eta1, h2[c,a,a,a], t2[pa,pv,hc,hc], optimize=True)
    O[a,v,c,a] += scale * -1.00000000 * np.einsum('uv,iajv,wuix->wajx', eta1, h2[c,v,c,a], t2[pa,pa,hc,ha], optimize=True)
    O[a,v,c,a] += scale * -1.00000000 * np.einsum('uv,iawv,xuji->xajw', eta1, h2[c,v,a,a], t2[pa,pa,hc,hc], optimize=True)
    O[a,v,c,a] += scale * +1.00000000 * np.einsum('uv,wavb,ubix->waix', eta1, h2[a,v,a,v], t2[pa,pv,hc,ha], optimize=True)
    O[a,v,c,a] += scale * -0.50000000 * np.einsum('uv,wx,yavx,uwiz->yaiz', gamma1, gamma1, h2[a,v,a,a], t2[pa,pa,hc,ha], optimize=True)
    O[a,v,c,a] += scale * +0.50000000 * np.einsum('uv,wx,uwiy,zavx->zaiy', gamma1, gamma1, h2[a,a,c,a], t2[pa,pv,ha,ha], optimize=True)
    O[a,v,c,a] += scale * +1.00000000 * np.einsum('uv,iujw,xaiv->xajw', gamma1, h2[c,a,c,a], t2[pa,pv,hc,ha], optimize=True)
    O[a,v,c,a] += scale * +1.00000000 * np.einsum('uv,wuia,baxv->wbix', gamma1, h2[a,a,c,v], t2[pv,pv,ha,ha], optimize=True)
    O[a,v,c,a] += scale * -1.00000000 * np.einsum('uv,wuxa,baiv->wbix', gamma1, h2[a,a,a,v], t2[pv,pv,hc,ha], optimize=True)
    O[a,v,c,a] += scale * +1.00000000 * np.einsum('uv,uaib,wbxv->waix', gamma1, h2[a,v,c,v], t2[pa,pv,ha,ha], optimize=True)
    O[a,v,c,a] += scale * -1.00000000 * np.einsum('uv,uawb,xbiv->xaiw', gamma1, h2[a,v,a,v], t2[pa,pv,hc,ha], optimize=True)
    O[a,v,c,a] += scale * +0.50000000 * np.einsum('ijku,vaij->vaku', h2[c,c,c,a], t2[pa,pv,hc,hc], optimize=True)
    O[a,v,c,a] += scale * +1.00000000 * np.einsum('iuja,baiv->ubjv', h2[c,a,c,v], t2[pv,pv,hc,ha], optimize=True)
    O[a,v,c,a] += scale * +1.00000000 * np.einsum('iuva,baji->ubjv', h2[c,a,a,v], t2[pv,pv,hc,hc], optimize=True)
    O[a,v,c,a] += scale * -1.00000000 * np.einsum('iajb,ubiv->uajv', h2[c,v,c,v], t2[pa,pv,hc,ha], optimize=True)
    O[a,v,c,a] += scale * -1.00000000 * np.einsum('iaub,vbji->vaju', h2[c,v,a,v], t2[pa,pv,hc,hc], optimize=True)
    O[a,v,c,a] += scale * +0.50000000 * np.einsum('uabc,bciv->uaiv', h2[a,v,v,v], t2[pv,pv,hc,ha], optimize=True)
    O[c,v,c,a] += scale * +0.50000000 * np.einsum('uv,wx,iavx,uwjy->iajy', eta1, eta1, h2[c,v,a,a], t2[pa,pa,hc,ha], optimize=True)
    O[c,v,c,a] += scale * +1.00000000 * np.einsum('uv,wx,iujx,wayv->iajy', eta1, gamma1, h2[c,a,c,a], t2[pa,pv,ha,ha], optimize=True)
    O[c,v,c,a] += scale * -1.00000000 * np.einsum('uv,wx,iuyx,wajv->iajy', eta1, gamma1, h2[c,a,a,a], t2[pa,pv,hc,ha], optimize=True)
    O[c,v,c,a] += scale * -1.00000000 * np.einsum('uv,wx,iwjv,uayx->iajy', eta1, gamma1, h2[c,a,c,a], t2[pa,pv,ha,ha], optimize=True)
    O[c,v,c,a] += scale * +1.00000000 * np.einsum('uv,wx,iwyv,uajx->iajy', eta1, gamma1, h2[c,a,a,a], t2[pa,pv,hc,ha], optimize=True)
    O[c,v,c,a] += scale * +1.00000000 * np.einsum('uv,ijkv,uajw->iakw', eta1, h2[c,c,c,a], t2[pa,pv,hc,ha], optimize=True)
    O[c,v,c,a] += scale * +1.00000000 * np.einsum('uv,ijwv,uakj->iakw', eta1, h2[c,c,a,a], t2[pa,pv,hc,hc], optimize=True)
    O[c,v,c,a] += scale * +1.00000000 * np.einsum('uv,iavb,ubjw->iajw', eta1, h2[c,v,a,v], t2[pa,pv,hc,ha], optimize=True)
    O[c,v,c,a] += scale * -0.50000000 * np.einsum('uv,wx,iavx,uwjy->iajy', gamma1, gamma1, h2[c,v,a,a], t2[pa,pa,hc,ha], optimize=True)
    O[c,v,c,a] += scale * +1.00000000 * np.einsum('uv,iuja,bawv->ibjw', gamma1, h2[c,a,c,v], t2[pv,pv,ha,ha], optimize=True)
    O[c,v,c,a] += scale * -1.00000000 * np.einsum('uv,iuwa,bajv->ibjw', gamma1, h2[c,a,a,v], t2[pv,pv,hc,ha], optimize=True)
    O[c,v,c,a] += scale * -1.00000000 * np.einsum('ijka,baju->ibku', h2[c,c,c,v], t2[pv,pv,hc,ha], optimize=True)
    O[c,v,c,a] += scale * -1.00000000 * np.einsum('ijua,bakj->ibku', h2[c,c,a,v], t2[pv,pv,hc,hc], optimize=True)
    O[c,v,c,a] += scale * +0.50000000 * np.einsum('iabc,bcju->iaju', h2[c,v,v,v], t2[pv,pv,hc,ha], optimize=True)
    O[v,v,c,a] += scale * -0.25000000 * np.einsum('uv,wx,uwiy,abvx->abiy', eta1, eta1, h2[a,a,c,a], t2[pv,pv,ha,ha], optimize=True)
    O[v,v,c,a] += scale * +0.25000000 * np.einsum('uv,wx,abvx,uwiy->abiy', eta1, eta1, h2[v,v,a,a], t2[pa,pa,hc,ha], optimize=True)
    O[v,v,c,a] += scale * -1.00000000 * np.einsum('uv,wx,uaix,wbyv->abiy', eta1, gamma1, h2[a,v,c,a], t2[pa,pv,ha,ha], optimize=True)
    O[v,v,c,a] += scale * +1.00000000 * np.einsum('uv,wx,uayx,wbiv->abiy', eta1, gamma1, h2[a,v,a,a], t2[pa,pv,hc,ha], optimize=True)
    O[v,v,c,a] += scale * +1.00000000 * np.einsum('uv,wx,waiv,ubyx->abiy', eta1, gamma1, h2[a,v,c,a], t2[pa,pv,ha,ha], optimize=True)
    O[v,v,c,a] += scale * -1.00000000 * np.einsum('uv,wx,wayv,ubix->abiy', eta1, gamma1, h2[a,v,a,a], t2[pa,pv,hc,ha], optimize=True)
    O[v,v,c,a] += scale * -1.00000000 * np.einsum('uv,iajv,ubiw->abjw', eta1, h2[c,v,c,a], t2[pa,pv,hc,ha], optimize=True)
    O[v,v,c,a] += scale * -1.00000000 * np.einsum('uv,iawv,ubji->abjw', eta1, h2[c,v,a,a], t2[pa,pv,hc,hc], optimize=True)
    O[v,v,c,a] += scale * +0.50000000 * np.einsum('uv,abvc,uciw->abiw', eta1, h2[v,v,a,v], t2[pa,pv,hc,ha], optimize=True)
    O[v,v,c,a] += scale * +0.25000000 * np.einsum('uv,wx,uwiy,abvx->abiy', gamma1, gamma1, h2[a,a,c,a], t2[pv,pv,ha,ha], optimize=True)
    O[v,v,c,a] += scale * -0.25000000 * np.einsum('uv,wx,abvx,uwiy->abiy', gamma1, gamma1, h2[v,v,a,a], t2[pa,pa,hc,ha], optimize=True)
    O[v,v,c,a] += scale * +0.50000000 * np.einsum('uv,iujw,abiv->abjw', gamma1, h2[c,a,c,a], t2[pv,pv,hc,ha], optimize=True)
    O[v,v,c,a] += scale * -1.00000000 * np.einsum('uv,uaib,cbwv->aciw', gamma1, h2[a,v,c,v], t2[pv,pv,ha,ha], optimize=True)
    O[v,v,c,a] += scale * +1.00000000 * np.einsum('uv,uawb,cbiv->aciw', gamma1, h2[a,v,a,v], t2[pv,pv,hc,ha], optimize=True)
    O[v,v,c,a] += scale * +0.25000000 * np.einsum('ijku,abij->abku', h2[c,c,c,a], t2[pv,pv,hc,hc], optimize=True)
    O[v,v,c,a] += scale * +1.00000000 * np.einsum('iajb,cbiu->acju', h2[c,v,c,v], t2[pv,pv,hc,ha], optimize=True)
    O[v,v,c,a] += scale * +1.00000000 * np.einsum('iaub,cbji->acju', h2[c,v,a,v], t2[pv,pv,hc,hc], optimize=True)
    O[v,v,c,a] += scale * +0.25000000 * np.einsum('abcd,cdiu->abiu', h2[v,v,v,v], t2[pv,pv,hc,ha], optimize=True)
    O[a,a,c,c] += scale * +0.12500000 * np.einsum('uv,wx,yzvx,uwij->yzij', eta1, eta1, h2[a,a,a,a], t2[pa,pa,hc,hc], optimize=True)
    O[a,a,c,c] += scale * -1.00000000 * np.einsum('uv,wx,yuix,zwjv->yzij', eta1, gamma1, h2[a,a,c,a], t2[pa,pa,hc,ha], optimize=True)
    O[a,a,c,c] += scale * +1.00000000 * np.einsum('uv,wx,ywiv,zujx->yzij', eta1, gamma1, h2[a,a,c,a], t2[pa,pa,hc,ha], optimize=True)
    O[a,a,c,c] += scale * -1.00000000 * np.einsum('uv,iwjv,xuki->wxjk', eta1, h2[c,a,c,a], t2[pa,pa,hc,hc], optimize=True)
    O[a,a,c,c] += scale * +0.25000000 * np.einsum('uv,wxva,uaij->wxij', eta1, h2[a,a,a,v], t2[pa,pv,hc,hc], optimize=True)
    O[a,a,c,c] += scale * -0.12500000 * np.einsum('uv,wx,yzvx,uwij->yzij', gamma1, gamma1, h2[a,a,a,a], t2[pa,pa,hc,hc], optimize=True)
    O[a,a,c,c] += scale * +0.25000000 * np.einsum('uv,iujk,wxiv->wxjk', gamma1, h2[c,a,c,c], t2[pa,pa,hc,ha], optimize=True)
    O[a,a,c,c] += scale * +1.00000000 * np.einsum('uv,wuia,xajv->wxij', gamma1, h2[a,a,c,v], t2[pa,pv,hc,ha], optimize=True)
    O[a,a,c,c] += scale * +0.12500000 * np.einsum('ijkl,uvij->uvkl', h2[c,c,c,c], t2[pa,pa,hc,hc], optimize=True)
    O[a,a,c,c] += scale * -1.00000000 * np.einsum('iuja,vaki->uvjk', h2[c,a,c,v], t2[pa,pv,hc,hc], optimize=True)
    O[a,a,c,c] += scale * +0.12500000 * np.einsum('uvab,abij->uvij', h2[a,a,v,v], t2[pv,pv,hc,hc], optimize=True)
    O[c,a,c,c] += scale * +0.25000000 * np.einsum('uv,wx,iyvx,uwjk->iyjk', eta1, eta1, h2[c,a,a,a], t2[pa,pa,hc,hc], optimize=True)
    O[c,a,c,c] += scale * -1.00000000 * np.einsum('uv,wx,iujx,ywkv->iyjk', eta1, gamma1, h2[c,a,c,a], t2[pa,pa,hc,ha], optimize=True)
    O[c,a,c,c] += scale * +1.00000000 * np.einsum('uv,wx,iwjv,yukx->iyjk', eta1, gamma1, h2[c,a,c,a], t2[pa,pa,hc,ha], optimize=True)
    O[c,a,c,c] += scale * +1.00000000 * np.einsum('uv,ijkv,wulj->iwkl', eta1, h2[c,c,c,a], t2[pa,pa,hc,hc], optimize=True)
    O[c,a,c,c] += scale * +0.50000000 * np.einsum('uv,iwva,uajk->iwjk', eta1, h2[c,a,a,v], t2[pa,pv,hc,hc], optimize=True)
    O[c,a,c,c] += scale * -0.25000000 * np.einsum('uv,wx,iyvx,uwjk->iyjk', gamma1, gamma1, h2[c,a,a,a], t2[pa,pa,hc,hc], optimize=True)
    O[c,a,c,c] += scale * +1.00000000 * np.einsum('uv,iuja,wakv->iwjk', gamma1, h2[c,a,c,v], t2[pa,pv,hc,ha], optimize=True)
    O[c,a,c,c] += scale * +1.00000000 * np.einsum('ijka,ualj->iukl', h2[c,c,c,v], t2[pa,pv,hc,hc], optimize=True)
    O[c,a,c,c] += scale * +0.25000000 * np.einsum('iuab,abjk->iujk', h2[c,a,v,v], t2[pv,pv,hc,hc], optimize=True)
    O[c,c,c,c] += scale * +0.12500000 * np.einsum('uv,wx,ijvx,uwkl->ijkl', eta1, eta1, h2[c,c,a,a], t2[pa,pa,hc,hc], optimize=True)
    O[c,c,c,c] += scale * +0.25000000 * np.einsum('uv,ijva,uakl->ijkl', eta1, h2[c,c,a,v], t2[pa,pv,hc,hc], optimize=True)
    O[c,c,c,c] += scale * -0.12500000 * np.einsum('uv,wx,ijvx,uwkl->ijkl', gamma1, gamma1, h2[c,c,a,a], t2[pa,pa,hc,hc], optimize=True)
    O[c,c,c,c] += scale * +0.12500000 * np.einsum('ijab,abkl->ijkl', h2[c,c,v,v], t2[pv,pv,hc,hc], optimize=True)
    O[a,v,c,c] += scale * +0.25000000 * np.einsum('uv,wx,yavx,uwij->yaij', eta1, eta1, h2[a,v,a,a], t2[pa,pa,hc,hc], optimize=True)
    O[a,v,c,c] += scale * -0.25000000 * np.einsum('uv,wx,uwij,yavx->yaij', eta1, eta1, h2[a,a,c,c], t2[pa,pv,ha,ha], optimize=True)
    O[a,v,c,c] += scale * +1.00000000 * np.einsum('uv,wx,yuix,wajv->yaij', eta1, gamma1, h2[a,a,c,a], t2[pa,pv,hc,ha], optimize=True)
    O[a,v,c,c] += scale * -1.00000000 * np.einsum('uv,wx,ywiv,uajx->yaij', eta1, gamma1, h2[a,a,c,a], t2[pa,pv,hc,ha], optimize=True)
    O[a,v,c,c] += scale * -1.00000000 * np.einsum('uv,wx,uaix,ywjv->yaij', eta1, gamma1, h2[a,v,c,a], t2[pa,pa,hc,ha], optimize=True)
    O[a,v,c,c] += scale * +1.00000000 * np.einsum('uv,wx,waiv,yujx->yaij', eta1, gamma1, h2[a,v,c,a], t2[pa,pa,hc,ha], optimize=True)
    O[a,v,c,c] += scale * +1.00000000 * np.einsum('uv,iwjv,uaki->wajk', eta1, h2[c,a,c,a], t2[pa,pv,hc,hc], optimize=True)
    O[a,v,c,c] += scale * +1.00000000 * np.einsum('uv,iajv,wuki->wajk', eta1, h2[c,v,c,a], t2[pa,pa,hc,hc], optimize=True)
    O[a,v,c,c] += scale * +0.50000000 * np.einsum('uv,wavb,ubij->waij', eta1, h2[a,v,a,v], t2[pa,pv,hc,hc], optimize=True)
    O[a,v,c,c] += scale * -0.25000000 * np.einsum('uv,wx,yavx,uwij->yaij', gamma1, gamma1, h2[a,v,a,a], t2[pa,pa,hc,hc], optimize=True)
    O[a,v,c,c] += scale * +0.25000000 * np.einsum('uv,wx,uwij,yavx->yaij', gamma1, gamma1, h2[a,a,c,c], t2[pa,pv,ha,ha], optimize=True)
    O[a,v,c,c] += scale * +0.50000000 * np.einsum('uv,iujk,waiv->wajk', gamma1, h2[c,a,c,c], t2[pa,pv,hc,ha], optimize=True)
    O[a,v,c,c] += scale * +1.00000000 * np.einsum('uv,wuia,bajv->wbij', gamma1, h2[a,a,c,v], t2[pv,pv,hc,ha], optimize=True)
    O[a,v,c,c] += scale * +1.00000000 * np.einsum('uv,uaib,wbjv->waij', gamma1, h2[a,v,c,v], t2[pa,pv,hc,ha], optimize=True)
    O[a,v,c,c] += scale * +0.25000000 * np.einsum('ijkl,uaij->uakl', h2[c,c,c,c], t2[pa,pv,hc,hc], optimize=True)
    O[a,v,c,c] += scale * -1.00000000 * np.einsum('iuja,baki->ubjk', h2[c,a,c,v], t2[pv,pv,hc,hc], optimize=True)
    O[a,v,c,c] += scale * +1.00000000 * np.einsum('iajb,ubki->uajk', h2[c,v,c,v], t2[pa,pv,hc,hc], optimize=True)
    O[a,v,c,c] += scale * +0.25000000 * np.einsum('uabc,bcij->uaij', h2[a,v,v,v], t2[pv,pv,hc,hc], optimize=True)
    O[c,v,c,c] += scale * +0.25000000 * np.einsum('uv,wx,iavx,uwjk->iajk', eta1, eta1, h2[c,v,a,a], t2[pa,pa,hc,hc], optimize=True)
    O[c,v,c,c] += scale * +1.00000000 * np.einsum('uv,wx,iujx,wakv->iajk', eta1, gamma1, h2[c,a,c,a], t2[pa,pv,hc,ha], optimize=True)
    O[c,v,c,c] += scale * -1.00000000 * np.einsum('uv,wx,iwjv,uakx->iajk', eta1, gamma1, h2[c,a,c,a], t2[pa,pv,hc,ha], optimize=True)
    O[c,v,c,c] += scale * -1.00000000 * np.einsum('uv,ijkv,ualj->iakl', eta1, h2[c,c,c,a], t2[pa,pv,hc,hc], optimize=True)
    O[c,v,c,c] += scale * +0.50000000 * np.einsum('uv,iavb,ubjk->iajk', eta1, h2[c,v,a,v], t2[pa,pv,hc,hc], optimize=True)
    O[c,v,c,c] += scale * -0.25000000 * np.einsum('uv,wx,iavx,uwjk->iajk', gamma1, gamma1, h2[c,v,a,a], t2[pa,pa,hc,hc], optimize=True)
    O[c,v,c,c] += scale * +1.00000000 * np.einsum('uv,iuja,bakv->ibjk', gamma1, h2[c,a,c,v], t2[pv,pv,hc,ha], optimize=True)
    O[c,v,c,c] += scale * +1.00000000 * np.einsum('ijka,balj->ibkl', h2[c,c,c,v], t2[pv,pv,hc,hc], optimize=True)
    O[c,v,c,c] += scale * +0.25000000 * np.einsum('iabc,bcjk->iajk', h2[c,v,v,v], t2[pv,pv,hc,hc], optimize=True)
    O[v,v,c,c] += scale * -0.12500000 * np.einsum('uv,wx,uwij,abvx->abij', eta1, eta1, h2[a,a,c,c], t2[pv,pv,ha,ha], optimize=True)
    O[v,v,c,c] += scale * +0.12500000 * np.einsum('uv,wx,abvx,uwij->abij', eta1, eta1, h2[v,v,a,a], t2[pa,pa,hc,hc], optimize=True)
    O[v,v,c,c] += scale * -1.00000000 * np.einsum('uv,wx,uaix,wbjv->abij', eta1, gamma1, h2[a,v,c,a], t2[pa,pv,hc,ha], optimize=True)
    O[v,v,c,c] += scale * +1.00000000 * np.einsum('uv,wx,waiv,ubjx->abij', eta1, gamma1, h2[a,v,c,a], t2[pa,pv,hc,ha], optimize=True)
    O[v,v,c,c] += scale * +1.00000000 * np.einsum('uv,iajv,ubki->abjk', eta1, h2[c,v,c,a], t2[pa,pv,hc,hc], optimize=True)
    O[v,v,c,c] += scale * +0.25000000 * np.einsum('uv,abvc,ucij->abij', eta1, h2[v,v,a,v], t2[pa,pv,hc,hc], optimize=True)
    O[v,v,c,c] += scale * +0.12500000 * np.einsum('uv,wx,uwij,abvx->abij', gamma1, gamma1, h2[a,a,c,c], t2[pv,pv,ha,ha], optimize=True)
    O[v,v,c,c] += scale * -0.12500000 * np.einsum('uv,wx,abvx,uwij->abij', gamma1, gamma1, h2[v,v,a,a], t2[pa,pa,hc,hc], optimize=True)
    O[v,v,c,c] += scale * +0.25000000 * np.einsum('uv,iujk,abiv->abjk', gamma1, h2[c,a,c,c], t2[pv,pv,hc,ha], optimize=True)
    O[v,v,c,c] += scale * -1.00000000 * np.einsum('uv,uaib,cbjv->acij', gamma1, h2[a,v,c,v], t2[pv,pv,hc,ha], optimize=True)
    O[v,v,c,c] += scale * +0.12500000 * np.einsum('ijkl,abij->abkl', h2[c,c,c,c], t2[pv,pv,hc,hc], optimize=True)
    O[v,v,c,c] += scale * -1.00000000 * np.einsum('iajb,cbki->acjk', h2[c,v,c,v], t2[pv,pv,hc,hc], optimize=True)
    O[v,v,c,c] += scale * +0.12500000 * np.einsum('abcd,cdij->abij', h2[v,v,v,v], t2[pv,pv,hc,hc], optimize=True)
    O[a,a,c,v] += scale * -1.00000000 * np.einsum('uv,wx,yuxa,zwiv->yzia', eta1, gamma1, h2[a,a,a,v], t2[pa,pa,hc,ha], optimize=True)
    O[a,a,c,v] += scale * +1.00000000 * np.einsum('uv,wx,ywva,zuix->yzia', eta1, gamma1, h2[a,a,a,v], t2[pa,pa,hc,ha], optimize=True)
    O[a,a,c,v] += scale * -1.00000000 * np.einsum('uv,iwva,xuji->wxja', eta1, h2[c,a,a,v], t2[pa,pa,hc,hc], optimize=True)
    O[a,a,c,v] += scale * +0.50000000 * np.einsum('uv,iuja,wxiv->wxja', gamma1, h2[c,a,c,v], t2[pa,pa,hc,ha], optimize=True)
    O[a,a,c,v] += scale * -1.00000000 * np.einsum('uv,wuab,xbiv->wxia', gamma1, h2[a,a,v,v], t2[pa,pv,hc,ha], optimize=True)
    O[a,a,c,v] += scale * +0.25000000 * np.einsum('ijka,uvij->uvka', h2[c,c,c,v], t2[pa,pa,hc,hc], optimize=True)
    O[a,a,c,v] += scale * +1.00000000 * np.einsum('iuab,vbji->uvja', h2[c,a,v,v], t2[pa,pv,hc,hc], optimize=True)
    O[c,a,c,v] += scale * -1.00000000 * np.einsum('uv,wx,iuxa,ywjv->iyja', eta1, gamma1, h2[c,a,a,v], t2[pa,pa,hc,ha], optimize=True)
    O[c,a,c,v] += scale * +1.00000000 * np.einsum('uv,wx,iwva,yujx->iyja', eta1, gamma1, h2[c,a,a,v], t2[pa,pa,hc,ha], optimize=True)
    O[c,a,c,v] += scale * +1.00000000 * np.einsum('uv,ijva,wukj->iwka', eta1, h2[c,c,a,v], t2[pa,pa,hc,hc], optimize=True)
    O[c,a,c,v] += scale * -1.00000000 * np.einsum('uv,iuab,wbjv->iwja', gamma1, h2[c,a,v,v], t2[pa,pv,hc,ha], optimize=True)
    O[c,a,c,v] += scale * -1.00000000 * np.einsum('ijab,ubkj->iuka', h2[c,c,v,v], t2[pa,pv,hc,hc], optimize=True)
    O[a,v,c,v] += scale * -0.50000000 * np.einsum('uv,wx,uwia,ybvx->ybia', eta1, eta1, h2[a,a,c,v], t2[pa,pv,ha,ha], optimize=True)
    O[a,v,c,v] += scale * +1.00000000 * np.einsum('uv,wx,yuxa,wbiv->ybia', eta1, gamma1, h2[a,a,a,v], t2[pa,pv,hc,ha], optimize=True)
    O[a,v,c,v] += scale * -1.00000000 * np.einsum('uv,wx,ywva,ubix->ybia', eta1, gamma1, h2[a,a,a,v], t2[pa,pv,hc,ha], optimize=True)
    O[a,v,c,v] += scale * -1.00000000 * np.einsum('uv,wx,uaxb,ywiv->yaib', eta1, gamma1, h2[a,v,a,v], t2[pa,pa,hc,ha], optimize=True)
    O[a,v,c,v] += scale * +1.00000000 * np.einsum('uv,wx,wavb,yuix->yaib', eta1, gamma1, h2[a,v,a,v], t2[pa,pa,hc,ha], optimize=True)
    O[a,v,c,v] += scale * +1.00000000 * np.einsum('uv,iwva,ubji->wbja', eta1, h2[c,a,a,v], t2[pa,pv,hc,hc], optimize=True)
    O[a,v,c,v] += scale * +1.00000000 * np.einsum('uv,iavb,wuji->wajb', eta1, h2[c,v,a,v], t2[pa,pa,hc,hc], optimize=True)
    O[a,v,c,v] += scale * +0.50000000 * np.einsum('uv,wx,uwia,ybvx->ybia', gamma1, gamma1, h2[a,a,c,v], t2[pa,pv,ha,ha], optimize=True)
    O[a,v,c,v] += scale * +1.00000000 * np.einsum('uv,iuja,wbiv->wbja', gamma1, h2[c,a,c,v], t2[pa,pv,hc,ha], optimize=True)
    O[a,v,c,v] += scale * -1.00000000 * np.einsum('uv,wuab,cbiv->wcia', gamma1, h2[a,a,v,v], t2[pv,pv,hc,ha], optimize=True)
    O[a,v,c,v] += scale * -1.00000000 * np.einsum('uv,uabc,wciv->waib', gamma1, h2[a,v,v,v], t2[pa,pv,hc,ha], optimize=True)
    O[a,v,c,v] += scale * +0.50000000 * np.einsum('ijka,ubij->ubka', h2[c,c,c,v], t2[pa,pv,hc,hc], optimize=True)
    O[a,v,c,v] += scale * +1.00000000 * np.einsum('iuab,cbji->ucja', h2[c,a,v,v], t2[pv,pv,hc,hc], optimize=True)
    O[a,v,c,v] += scale * -1.00000000 * np.einsum('iabc,ucji->uajb', h2[c,v,v,v], t2[pa,pv,hc,hc], optimize=True)
    O[c,v,c,v] += scale * +1.00000000 * np.einsum('uv,wx,iuxa,wbjv->ibja', eta1, gamma1, h2[c,a,a,v], t2[pa,pv,hc,ha], optimize=True)
    O[c,v,c,v] += scale * -1.00000000 * np.einsum('uv,wx,iwva,ubjx->ibja', eta1, gamma1, h2[c,a,a,v], t2[pa,pv,hc,ha], optimize=True)
    O[c,v,c,v] += scale * -1.00000000 * np.einsum('uv,ijva,ubkj->ibka', eta1, h2[c,c,a,v], t2[pa,pv,hc,hc], optimize=True)
    O[c,v,c,v] += scale * -1.00000000 * np.einsum('uv,iuab,cbjv->icja', gamma1, h2[c,a,v,v], t2[pv,pv,hc,ha], optimize=True)
    O[c,v,c,v] += scale * -1.00000000 * np.einsum('ijab,cbkj->icka', h2[c,c,v,v], t2[pv,pv,hc,hc], optimize=True)
    O[v,v,c,v] += scale * -0.25000000 * np.einsum('uv,wx,uwia,bcvx->bcia', eta1, eta1, h2[a,a,c,v], t2[pv,pv,ha,ha], optimize=True)
    O[v,v,c,v] += scale * -1.00000000 * np.einsum('uv,wx,uaxb,wciv->acib', eta1, gamma1, h2[a,v,a,v], t2[pa,pv,hc,ha], optimize=True)
    O[v,v,c,v] += scale * +1.00000000 * np.einsum('uv,wx,wavb,ucix->acib', eta1, gamma1, h2[a,v,a,v], t2[pa,pv,hc,ha], optimize=True)
    O[v,v,c,v] += scale * +1.00000000 * np.einsum('uv,iavb,ucji->acjb', eta1, h2[c,v,a,v], t2[pa,pv,hc,hc], optimize=True)
    O[v,v,c,v] += scale * +0.25000000 * np.einsum('uv,wx,uwia,bcvx->bcia', gamma1, gamma1, h2[a,a,c,v], t2[pv,pv,ha,ha], optimize=True)
    O[v,v,c,v] += scale * +0.50000000 * np.einsum('uv,iuja,bciv->bcja', gamma1, h2[c,a,c,v], t2[pv,pv,hc,ha], optimize=True)
    O[v,v,c,v] += scale * +1.00000000 * np.einsum('uv,uabc,dciv->adib', gamma1, h2[a,v,v,v], t2[pv,pv,hc,ha], optimize=True)
    O[v,v,c,v] += scale * +0.25000000 * np.einsum('ijka,bcij->bcka', h2[c,c,c,v], t2[pv,pv,hc,hc], optimize=True)
    O[v,v,c,v] += scale * +1.00000000 * np.einsum('iabc,dcji->adjb', h2[c,v,v,v], t2[pv,pv,hc,hc], optimize=True)
    O[a,a,v,v] += scale * +0.25000000 * np.einsum('uv,iuab,wxiv->wxab', gamma1, h2[c,a,v,v], t2[pa,pa,hc,ha], optimize=True)
    O[a,a,v,v] += scale * +0.12500000 * np.einsum('ijab,uvij->uvab', h2[c,c,v,v], t2[pa,pa,hc,hc], optimize=True)
    O[a,v,v,v] += scale * -0.25000000 * np.einsum('uv,wx,uwab,ycvx->ycab', eta1, eta1, h2[a,a,v,v], t2[pa,pv,ha,ha], optimize=True)
    O[a,v,v,v] += scale * +0.25000000 * np.einsum('uv,wx,uwab,ycvx->ycab', gamma1, gamma1, h2[a,a,v,v], t2[pa,pv,ha,ha], optimize=True)
    O[a,v,v,v] += scale * +0.50000000 * np.einsum('uv,iuab,wciv->wcab', gamma1, h2[c,a,v,v], t2[pa,pv,hc,ha], optimize=True)
    O[a,v,v,v] += scale * +0.25000000 * np.einsum('ijab,ucij->ucab', h2[c,c,v,v], t2[pa,pv,hc,hc], optimize=True)
    O[v,v,v,v] += scale * -0.12500000 * np.einsum('uv,wx,uwab,cdvx->cdab', eta1, eta1, h2[a,a,v,v], t2[pv,pv,ha,ha], optimize=True)
    O[v,v,v,v] += scale * +0.12500000 * np.einsum('uv,wx,uwab,cdvx->cdab', gamma1, gamma1, h2[a,a,v,v], t2[pv,pv,ha,ha], optimize=True)
    O[v,v,v,v] += scale * +0.25000000 * np.einsum('uv,iuab,cdiv->cdab', gamma1, h2[c,a,v,v], t2[pv,pv,hc,ha], optimize=True)
    O[v,v,v,v] += scale * +0.12500000 * np.einsum('ijab,cdij->cdab', h2[c,c,v,v], t2[pv,pv,hc,hc], optimize=True)

    if verbose: print("h2_t2_c2 took {:.4f} seconds to run.".format(t1-t0))

    return O

#
# Optimized Codes
#
def make_gamma1_full(gamma1, nbasis, c, a, v):
    gamma1_full = np.zeros((nbasis, nbasis))

    nC = gamma1_full[c, c].shape[0]
    nV = gamma1_full[v, v].shape[0]
    nA = gamma1_full[a, a].shape[0]

    gamma1_full[c, c] = np.eye(nC)
    gamma1_full[a, a] = gamma1
    return gamma1_full

def make_eta1_full(eta1, nbasis, c, a, v):
    eta1_full = np.zeros((nbasis, nbasis))

    nC = eta1_full[c, c].shape[0]
    nV = eta1_full[v, v].shape[0]
    nA = eta1_full[a, a].shape[0]

    eta1_full[v, v] = np.eye(nV)
    eta1_full[a, a] = eta1
    return eta1_full

#@profile
def update_zerobody(Hbar, t, gamma1, eta1, gamma1_full, eta1_full, lambdas, orbspace, verbose=False, scale=1.0):
    t0 = time.time()
    c = orbspace['core']
    a = orbspace['active']
    v = orbspace['virt']
    hc = orbspace['hole_core']
    ha = orbspace['hole_active']
    pa = orbspace['particle_active']
    pv = orbspace['particle_virt']

    t1, t2 = t # Unpack cluster operator
    h1, h2 = Hbar # Unpack Hamiltonian
    lambda2 = lambdas['2'] # 2-cumulant

    ### Dressed T1
    tau1 = np.einsum("ia,ab,ji->jb", t1, eta1_full[p, p], gamma1_full[h, h], optimize=True)
    ### C0 <- [H1, t1]
    C0 = scale * np.einsum("bj,jb->", h1[p, h], tau1, optimize=True)
    ### C0 <- [H1, t2] + [H2, t1]
    temp = np.einsum("ex,uvey->uvxy", h1[v, a], t2[ha, ha, pv, pa], optimize=True)
    temp -= np.einsum("vm,umxy->uvxy", h1[a, c], t2[ha, hc, pa, pa], optimize=True)
    temp += np.einsum("evxy,ue->uvxy", h2[v, a, a, a], t1[ha, pv], optimize=True)
    temp -= np.einsum("uvmy,mx->uvxy", h2[a, a, c, a], t1[hc, pa], optimize=True)
    C0 += scale * 0.5 * np.einsum("uvxy,xyuv->", temp, lambda2, optimize=True)
    ### C0 <- [H2, t2]
    C0 += scale * 0.25 * np.einsum("efmn,mnef->", h2[v, v, c, c], t2[hc, hc, pv, pv], optimize=True)
    #
    temp1 = np.einsum("ijux,ki,lj->klux", t2[:, :, pa, pa], gamma1_full[h, h], gamma1_full[h, h], optimize=True)
    temp2 = np.einsum("klux,uv,xy->klvy", temp1, eta1, eta1, optimize=True)
    C0 += scale * 0.25 * np.einsum("vykl,klvy->", h2[a, a, h, h], temp2, optimize=True)
    #
    temp1 = np.einsum("ijue,ki,lj->klue", t2[:, :, pa, pv], gamma1_full[h, h], gamma1_full[h, h], optimize=True)
    temp2 = np.einsum("klue,uv->klve", temp1, eta1, optimize=True)
    C0 += scale * 0.5 * np.einsum("vekl,klve->", h2[a, v, h, h], temp2, optimize=True)
    #
    temp2 = -np.einsum("fexu,yvef->yvxu", h2[v, v, a, a], t2[ha, ha, pv, pv], optimize=True)
    C0 += scale * 0.25 * np.einsum("yvxu,xy,uv->", temp2, gamma1, gamma1, optimize=True)
    #
    temp2 = -np.einsum("femu,mvef->vu", h2[v, v, c, a], t2[hc, ha, pv, pv], optimize=True)
    C0 += scale * 0.5 * np.einsum("vu,uv->", temp2, gamma1, optimize=True)
    #
    temp1 = np.einsum("uvkl,ki,lj->uvij", h2[a, a, h, h], gamma1_full[h, h], gamma1_full[h, h], optimize=True)
    temp2 = 0.125 * np.einsum("uvij,ijxy->uvxy", temp1, t2[:, :, pa, pa], optimize=True)

    temp1 = np.einsum("uvab,ac,bd->uvcd", t2[ha, ha, :, :], eta1_full[p, p], eta1_full[p, p], optimize=True)
    temp2 += 0.125 * np.einsum("uvcd,cdxy->uvxy", temp1, h2[p, p, a, a], optimize=True)

    temp1 = np.einsum("iuay,ji,ab->juby", t2[:, ha, :, pa], gamma1_full[h, h], eta1_full[p, p], optimize=True)
    temp2 += np.einsum("vbjx,juby->uvxy", h2[a, p, h, a], temp1, optimize=True)
    C0 += scale * np.einsum("uvxy,xyuv->", temp2, lambda2, optimize=True)
    #
    temp1 = np.einsum("uviz,iwxy->uvwxyz", h2[a, a, h, a], t2[:, ha, pa, pa], optimize=True)
    temp1 += np.einsum("waxy,uvaz->uvwxyz", h2[a, p, a, a], t2[ha, ha, :, pa], optimize=True)
    C0 += scale * 0.25 * np.einsum("uvwxyz,xyzuvw->", temp1, lambda3, optimize=True)

    t1 = time.time()
    if verbose: print("zerobody took {:.4f} seconds to run.".format(t1-t0))

    return C0

#@profile
def update_onebody(C1, F, V, T1, T2, gamma1, eta1, lambda2, lambda3, mf, verbose=False, scale=1.0):
    t0 = time.time()
    hc = mf.hc
    ha = mf.ha
    pa = mf.pa
    pv = mf.pv
    c = mf.core
    a = mf.active
    v = mf.virt
    p = mf.part
    h = mf.hole
    norb = F.shape[0]
    ### Make full Gamma and Eta arrays
    gamma1_full = make_gamma1_full(gamma1, norb, c, a, v)
    ### C1 <- [H1, T1]
    C1[h, :] += scale * np.einsum("ap,ia->ip", F[p, :], T1, optimize=True)
    C1[:, p] -= scale * np.einsum("pi,ia->pa", F[:, h], T1, optimize=True)
    ### C1 <- [H1, T2]
    C1[h, p] += scale * np.einsum("bj,ijab->ia", F[p, c], T2[:, hc, :, :], optimize=True)
    C1[h, p] += scale * np.einsum("bu,ivab,uv->ia", F[p, a], T2[:, ha, :, :], gamma1, optimize=True)
    C1[h, p] -= scale * np.einsum("vj,ijau,uv->ia", F[a, h], T2[:, :, :, pa], gamma1, optimize=True)
    ### C1 <- [H2, T1]
    C1 += scale * np.einsum("qapm,ma->qp", V[:, p, :, c], T1[hc, :], optimize=True)
    C1 += scale * np.einsum("qapv,ua,vu->qp", V[:, p, :, a], T1[ha, :], gamma1, optimize=True)
    C1 -= scale * np.einsum("qvpm,mu,uv->qp", V[:, a, :, c], T1[hc, pa], gamma1, optimize=True)
    ### C1 <- [H2, T2]
    temp = np.einsum("abrk,ijab->ijrk", V[p, p, :, h], T2, optimize=True)
    C1[h, :] += scale * 0.5 * np.einsum("ijrk,kj->ir", temp, gamma1_full[h, h], optimize=True)
    #
    temp = np.einsum("ijux,uv,xy->ijvy", T2[:, :, pa, pa], gamma1, gamma1)
    C1[h, :] += scale * 0.5 * np.einsum("ijvy,vyrj->ir", temp, V[a, a, :, h], optimize=True)
    #
    temp = np.einsum("ijub,kj,uv->ikvb", T2[:, :, pa, :], gamma1_full[h, h], gamma1, optimize=True)
    C1[h, :] -= scale * np.einsum("ikvb,vbrk->ir", temp, V[a, p, :, h], optimize=True)
    #
    temp = np.einsum("ijau,uv->ijav", T2[:, :, :, pa], gamma1, optimize=True)
    C1[:, p] -= scale * 0.5 * np.einsum("ijav,pvij->pa", temp, V[:, a, h, h], optimize=True)
    #
    temp = np.einsum("ijab,ki,lj->klab", T2, gamma1_full[h, h], gamma1_full[h, h], optimize=True)
    C1[:, p] -= scale * 0.5 * np.einsum("klab,pbkl->pa", temp, V[:, p, h, h], optimize=True)
    #
    temp = np.einsum("ijau,uv,kj->ikav", T2[:, :, :, pa], gamma1, gamma1_full[h, h], optimize=True)
    C1[:, p] += scale * np.einsum("ikav,pvik->pa", temp, V[:, a, h, h], optimize=True)
    #
    temp = np.einsum("ijxy,xyuv->ijuv", T2[:, :, pa, pa], lambda2, optimize=True)
    C1[h, :] += scale * 0.25 * np.einsum("ijuv,uvrj->ir", temp, V[a, a, :, h], optimize=True)
    #
    temp = np.einsum("uvab,xyuv->xyab", T2[ha, ha, :, :], lambda2, optimize=True)
    C1[:, p] -= scale * 0.25 * np.einsum("xyab,pbxy->pa", temp, V[:, p, a, a], optimize=True)
    #
    temp = np.einsum("iyav,uvxy->iuax", T2[:, ha, :, pa], lambda2, optimize=True)
    C1[h, :] += scale * np.einsum("iuax,axru->ir", temp, V[p, a, :, a], optimize=True)
    C1[:, p] -= scale * np.einsum("iuax,pxiu->pa", temp, V[:, a, h, a], optimize=True)
    #
    temp = np.einsum("avxy,xyuv->au", V[p, a, a, a], lambda2, optimize=True)
    C1[h, p] += scale * 0.5 * np.einsum("au,ujab->jb", temp, T2[ha, :, :, :], optimize=True)
    #
    temp = np.einsum("xyiv,uvxy->ui", V[a, a, h, a], lambda2, optimize=True)
    C1[h, p] -= scale * 0.5 * np.einsum("ui,ijub->jb", temp, T2[:, :, pa, :], optimize=True)
    #
    temp = np.einsum("uvey,xyuv->xe", T2[ha, ha, pv, pa], lambda2, optimize=True)
    C1 += scale * 0.5 * np.einsum("xe,eqxs->qs", temp, V[v, :, a, :], optimize=True)
    #
    temp = np.einsum("myuv,uvxy->mx", T2[hc, ha, pa, pa], lambda2, optimize=True)
    C1 -= scale * 0.5 * np.einsum("mx,xqms->qs", temp, V[a, :, c, :], optimize=True)
    t1 = time.time()
    if verbose: print("C1 took {:.4f} seconds to run.".format(t1-t0))
    return C1

#@profile
def update_twobody(C2, F, V, T1, T2, gamma1, eta1, lambda2, lambda3, mf, verbose=False, scale=1.0):
    t0 = time.time()
    hc = mf.hc
    ha = mf.ha
    pa = mf.pa
    pv = mf.pv
    c = mf.core
    a = mf.active
    v = mf.virt
    p = mf.part
    h = mf.hole
    norb = F.shape[0]
    ### Make full Gamma and Eta arrays
    gamma1_full = make_gamma1_full(gamma1, norb, c, a, v)
    ### C2 <- [H1, T2]
    C2[h, h, :, p] += scale * 0.5 * np.einsum("ap,ijab->ijpb", F[p, :], T2, optimize=True)
    C2[:, h, p, p] -= scale * 0.5 * np.einsum("qi,ijab->qjab", F[:, h], T2, optimize=True)
    ### C2 <- [H2, T1]
    C2[h, :, :, :] += scale * 0.5 * np.einsum("arpq,ia->irpq", V[p, :, :, :], T1, optimize=True)
    C2[:, :, p, :] -= scale * 0.5 * np.einsum("rsiq,ia->rsaq", V[:, :, h, :], T1, optimize=True)
    ### C2 <- [H2, T2]
    # particle-particle contractions
    C2[h, h, :, :] += scale * 0.125 * np.einsum("abrs,ijab->ijrs", V[p, p, :, :], T2, optimize=True)
    temp = np.einsum("ijbx,xy->ijby", T2[:, :, :, pa], gamma1, optimize=True)
    C2[h, h, :, :] -= scale * 0.25 * np.einsum("ijby,byrs->ijrs", temp, V[p, a, :, :], optimize=True)
    # hole-hole contractions
    C2[:, :, p, p] += scale * 0.125 * np.einsum("pqij,ijab->pqab", V[:, :, h, h], T2, optimize=True)
    temp = np.einsum("yjab,xy->xjab", T2[ha, :, :, :], eta1, optimize=True)
    C2[:, :, p, p] -= scale * 0.25 * np.einsum("xjab,pqxj->pqab", temp, V[:, :, a, h], optimize=True)
    # particle-hole contractions
    temp = np.einsum("ijab,ki->kjab", T2, gamma1_full[h, h], optimize=True)
    C2[:, h, :, p] += scale * np.einsum("kjab,aqks->qjsb", temp, V[p, :, h, :], optimize=True)
    temp = np.einsum("ijub,uv->ijvb", T2[:, :, pa, :], gamma1, optimize=True)
    C2[:, h, :, p] -= scale * np.einsum("ijvb,vqis->qjsb", temp, V[a, :, h, :], optimize=True)
    # antisymmetrize
    C2 -= np.transpose(C2, (1, 0, 2, 3))
    C2 -= np.transpose(C2, (0, 1, 3, 2))
    t1 = time.time()
    if verbose: print("C2 took {:.4f} seconds to run.".format(t1-t0))
    return C2
