import time
import numpy as np

def h1a_t1a_c0(O, h, t, gamma1, eta1, lambdas, orbspace, verbose=False, scale=1.0):
    # 5 lines
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

    
    
    O['0'] += scale * +1.00000000 * np.einsum('uv,iv,ui->', eta1['a'], h['a'][c,a], t['a'][pa,hc], optimize=True)
    O['0'] += scale * +1.00000000 * np.einsum('uv,ua,av->', gamma1['a'], h['a'][a,v], t['a'][pv,ha], optimize=True)
    O['0'] += scale * +1.00000000 * np.einsum('ia,ai->', h['a'][c,v], t['a'][pv,hc], optimize=True)

    t1 = time.time()
    if verbose: print("h1a_t1a_c0 took {:.4f} seconds to run.".format(t1-t0))

    return O
def h1a_t2a_c0(O, h, t, gamma1, eta1, lambdas, orbspace, verbose=False, scale=1.0):
    # 6 lines
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

    
    
    
    
    O['0'] += scale * -0.50000000 * np.einsum('iu,vwux,vwix->', h['a'][c,a], lambdas['aa'], t['aa'][pa,pa,hc,ha], optimize=True)
    O['0'] += scale * -0.50000000 * np.einsum('ua,uvwx,vawx->', h['a'][a,v], lambdas['aa'], t['aa'][pa,pv,ha,ha], optimize=True)

    t1 = time.time()
    if verbose: print("h1a_t2a_c0 took {:.4f} seconds to run.".format(t1-t0))

    return O
def h1a_t2b_c0(O, h, t, gamma1, eta1, lambdas, orbspace, verbose=False, scale=1.0):
    # 6 lines
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

    
    
    
    
    O['0'] += scale * -1.00000000 * np.einsum('iu,vUuV,vUiV->', h['a'][c,a], lambdas['ab'], t['ab'][pa,pA,hc,hA], optimize=True)
    O['0'] += scale * +1.00000000 * np.einsum('ua,uUvV,aUvV->', h['a'][a,v], lambdas['ab'], t['ab'][pv,pA,ha,hA], optimize=True)

    t1 = time.time()
    if verbose: print("h1a_t2b_c0 took {:.4f} seconds to run.".format(t1-t0))

    return O
def h1b_t1b_c0(O, h, t, gamma1, eta1, lambdas, orbspace, verbose=False, scale=1.0):
    # 5 lines
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

    
    
    O['0'] += scale * +1.00000000 * np.einsum('UV,IV,UI->', eta1['b'], h['b'][C,A], t['b'][pA,hC], optimize=True)
    O['0'] += scale * +1.00000000 * np.einsum('UV,UA,AV->', gamma1['b'], h['b'][A,V], t['b'][pV,hA], optimize=True)
    O['0'] += scale * +1.00000000 * np.einsum('IA,AI->', h['b'][C,V], t['b'][pV,hC], optimize=True)

    t1 = time.time()
    if verbose: print("h1b_t1b_c0 took {:.4f} seconds to run.".format(t1-t0))

    return O
def h1b_t2b_c0(O, h, t, gamma1, eta1, lambdas, orbspace, verbose=False, scale=1.0):
    # 6 lines
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

    
    
    
    
    O['0'] += scale * -1.00000000 * np.einsum('IU,uVvU,uVvI->', h['b'][C,A], lambdas['ab'], t['ab'][pa,pA,ha,hC], optimize=True)
    O['0'] += scale * +1.00000000 * np.einsum('UA,uUvV,uAvV->', h['b'][A,V], lambdas['ab'], t['ab'][pa,pV,ha,hA], optimize=True)

    t1 = time.time()
    if verbose: print("h1b_t2b_c0 took {:.4f} seconds to run.".format(t1-t0))

    return O
def h1b_t2c_c0(O, h, t, gamma1, eta1, lambdas, orbspace, verbose=False, scale=1.0):
    # 6 lines
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

    
    
    
    
    O['0'] += scale * -0.50000000 * np.einsum('IU,VWUX,VWIX->', h['b'][C,A], lambdas['bb'], t['bb'][pA,pA,hC,hA], optimize=True)
    O['0'] += scale * -0.50000000 * np.einsum('UA,UVWX,VAWX->', h['b'][A,V], lambdas['bb'], t['bb'][pA,pV,hA,hA], optimize=True)

    t1 = time.time()
    if verbose: print("h1b_t2c_c0 took {:.4f} seconds to run.".format(t1-t0))

    return O
def h2a_t1a_c0(O, h, t, gamma1, eta1, lambdas, orbspace, verbose=False, scale=1.0):
    # 6 lines
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

    
    
    
    
    O['0'] += scale * +0.50000000 * np.einsum('iuvw,uxvw,xi->', h['aa'][c,a,a,a], lambdas['aa'], t['a'][pa,hc], optimize=True)
    O['0'] += scale * +0.50000000 * np.einsum('uvwa,uvwx,ax->', h['aa'][a,a,a,v], lambdas['aa'], t['a'][pv,ha], optimize=True)

    t1 = time.time()
    if verbose: print("h2a_t1a_c0 took {:.4f} seconds to run.".format(t1-t0))

    return O
def h2a_t2a_c0(O, h, t, gamma1, eta1, lambdas, orbspace, verbose=False, scale=1.0):
    # 29 lines
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

    
    
    O['0'] += scale * +0.50000000 * np.einsum('uv,wx,yz,iyvx,uwiz->', eta1['a'], eta1['a'], gamma1['a'], h['aa'][c,a,a,a], t['aa'][pa,pa,hc,ha], optimize=True)
    O['0'] += scale * +0.25000000 * np.einsum('uv,wx,ijvx,uwij->', eta1['a'], eta1['a'], h['aa'][c,c,a,a], t['aa'][pa,pa,hc,hc], optimize=True)
    
    
    O['0'] += scale * +0.50000000 * np.einsum('uv,wx,yz,wyva,uaxz->', eta1['a'], gamma1['a'], gamma1['a'], h['aa'][a,a,a,v], t['aa'][pa,pv,ha,ha], optimize=True)
    O['0'] += scale * +1.00000000 * np.einsum('uv,wx,iwva,uaix->', eta1['a'], gamma1['a'], h['aa'][c,a,a,v], t['aa'][pa,pv,hc,ha], optimize=True)
    
    
    O['0'] += scale * +0.50000000 * np.einsum('uv,ijva,uaij->', eta1['a'], h['aa'][c,c,a,v], t['aa'][pa,pv,hc,hc], optimize=True)
    O['0'] += scale * +1.00000000 * np.einsum('uv,iwvx,wyxz,uyiz->', eta1['a'], h['aa'][c,a,a,a], lambdas['aa'], t['aa'][pa,pa,hc,ha], optimize=True)
    
    
    O['0'] += scale * +0.25000000 * np.einsum('uv,wxva,wxyz,uayz->', eta1['a'], h['aa'][a,a,a,v], lambdas['aa'], t['aa'][pa,pv,ha,ha], optimize=True)
    
    O['0'] += scale * +0.25000000 * np.einsum('uv,wx,uwab,abvx->', gamma1['a'], gamma1['a'], h['aa'][a,a,v,v], t['aa'][pv,pv,ha,ha], optimize=True)
    
    O['0'] += scale * +0.25000000 * np.einsum('uv,iuwx,yzwx,yziv->', gamma1['a'], h['aa'][c,a,a,a], lambdas['aa'], t['aa'][pa,pa,hc,ha], optimize=True)
    O['0'] += scale * +0.50000000 * np.einsum('uv,iuab,abiv->', gamma1['a'], h['aa'][c,a,v,v], t['aa'][pv,pv,hc,ha], optimize=True)
    
    O['0'] += scale * +1.00000000 * np.einsum('uv,uwxa,wyxz,yavz->', gamma1['a'], h['aa'][a,a,a,v], lambdas['aa'], t['aa'][pa,pv,ha,ha], optimize=True)
    
    O['0'] += scale * +0.12500000 * np.einsum('ijuv,wxuv,wxij->', h['aa'][c,c,a,a], lambdas['aa'], t['aa'][pa,pa,hc,hc], optimize=True)
    O['0'] += scale * +0.25000000 * np.einsum('ijab,abij->', h['aa'][c,c,v,v], t['aa'][pv,pv,hc,hc], optimize=True)
    
    O['0'] += scale * +1.00000000 * np.einsum('iuva,uwvx,waix->', h['aa'][c,a,a,v], lambdas['aa'], t['aa'][pa,pv,hc,ha], optimize=True)
    
    O['0'] += scale * +0.12500000 * np.einsum('uvab,uvwx,abwx->', h['aa'][a,a,v,v], lambdas['aa'], t['aa'][pv,pv,ha,ha], optimize=True)

    O['0'] += scale * +0.25000000 * np.einsum('iuvw,uxyvwz,xyiz->', h['aa'][c,a,a,a], lambdas['aaa'], t['aa'][pa,pa,hc,ha], optimize=True)
    O['0'] += scale * -0.25000000 * np.einsum('uvwa,uvxwyz,xayz->', h['aa'][a,a,a,v], lambdas['aaa'], t['aa'][pa,pv,ha,ha], optimize=True)

    t1 = time.time()
    if verbose: print("h2a_t2a_c0 took {:.4f} seconds to run.".format(t1-t0))

    return O
def h2a_t2b_c0(O, h, t, gamma1, eta1, lambdas, orbspace, verbose=False, scale=1.0):
    # 11 lines
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

    
    
    O['0'] += scale * +1.00000000 * np.einsum('uv,iwvx,wUxV,uUiV->', eta1['a'], h['aa'][c,a,a,a], lambdas['ab'], t['ab'][pa,pA,hc,hA], optimize=True)
    
    
    
    O['0'] += scale * -1.00000000 * np.einsum('uv,uwxa,wUxV,aUvV->', gamma1['a'], h['aa'][a,a,a,v], lambdas['ab'], t['ab'][pv,pA,ha,hA], optimize=True)
    
    
    O['0'] += scale * -1.00000000 * np.einsum('iuva,uUvV,aUiV->', h['aa'][c,a,a,v], lambdas['ab'], t['ab'][pv,pA,hc,hA], optimize=True)
    O['0'] += scale * +0.50000000 * np.einsum('iuvw,uxUvwV,xUiV->', h['aa'][c,a,a,a], lambdas['aab'], t['ab'][pa,pA,hc,hA], optimize=True)
    O['0'] += scale * +0.50000000 * np.einsum('uvwa,uvUwxV,aUxV->', h['aa'][a,a,a,v], lambdas['aab'], t['ab'][pv,pA,ha,hA], optimize=True)
    

    t1 = time.time()
    if verbose: print("h2a_t2b_c0 took {:.4f} seconds to run.".format(t1-t0))

    return O
def h2b_t1a_c0(O, h, t, gamma1, eta1, lambdas, orbspace, verbose=False, scale=1.0):
    # 6 lines
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

    
    
    
    
    O['0'] += scale * -1.00000000 * np.einsum('iUuV,vUuV,vi->', h['ab'][c,A,a,A], lambdas['ab'], t['a'][pa,hc], optimize=True)
    O['0'] += scale * +1.00000000 * np.einsum('uUaV,uUvV,av->', h['ab'][a,A,v,A], lambdas['ab'], t['a'][pv,ha], optimize=True)

    t1 = time.time()
    if verbose: print("h2b_t1a_c0 took {:.4f} seconds to run.".format(t1-t0))

    return O
def h2b_t1b_c0(O, h, t, gamma1, eta1, lambdas, orbspace, verbose=False, scale=1.0):
    # 6 lines
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

    
    
    
    
    O['0'] += scale * -1.00000000 * np.einsum('uIvU,uVvU,VI->', h['ab'][a,C,a,A], lambdas['ab'], t['b'][pA,hC], optimize=True)
    O['0'] += scale * +1.00000000 * np.einsum('uUvA,uUvV,AV->', h['ab'][a,A,a,V], lambdas['ab'], t['b'][pV,hA], optimize=True)

    t1 = time.time()
    if verbose: print("h2b_t1b_c0 took {:.4f} seconds to run.".format(t1-t0))

    return O
def h2b_t2a_c0(O, h, t, gamma1, eta1, lambdas, orbspace, verbose=False, scale=1.0):
    # 11 lines
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

    
    
    O['0'] += scale * +1.00000000 * np.einsum('uv,iUvV,wUxV,uwix->', eta1['a'], h['ab'][c,A,a,A], lambdas['ab'], t['aa'][pa,pa,hc,ha], optimize=True)
    
    
    
    O['0'] += scale * -1.00000000 * np.einsum('uv,uUaV,wUxV,wavx->', gamma1['a'], h['ab'][a,A,v,A], lambdas['ab'], t['aa'][pa,pv,ha,ha], optimize=True)
    
    
    O['0'] += scale * -1.00000000 * np.einsum('iUaV,uUvV,uaiv->', h['ab'][c,A,v,A], lambdas['ab'], t['aa'][pa,pv,hc,ha], optimize=True)
    O['0'] += scale * -0.50000000 * np.einsum('iUuV,vwUuxV,vwix->', h['ab'][c,A,a,A], lambdas['aab'], t['aa'][pa,pa,hc,ha], optimize=True)
    O['0'] += scale * -0.50000000 * np.einsum('uUaV,uvUwxV,vawx->', h['ab'][a,A,v,A], lambdas['aab'], t['aa'][pa,pv,ha,ha], optimize=True)
    

    t1 = time.time()
    if verbose: print("h2b_t2a_c0 took {:.4f} seconds to run.".format(t1-t0))

    return O
def h2b_t2b_c0(O, h, t, gamma1, eta1, lambdas, orbspace, verbose=False, scale=1.0):
    # 59 lines
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

    
    O['0'] += scale * +1.00000000 * np.einsum('uv,wx,wIvA,uAxI->', eta1['a'], gamma1['a'], h['ab'][a,C,a,V], t['ab'][pa,pV,ha,hC], optimize=True)
    
    O['0'] += scale * +1.00000000 * np.einsum('uv,UV,wx,wUvA,uAxV->', eta1['a'], gamma1['b'], gamma1['a'], h['ab'][a,A,a,V], t['ab'][pa,pV,ha,hA], optimize=True)
    O['0'] += scale * +1.00000000 * np.einsum('uv,UV,iUvA,uAiV->', eta1['a'], gamma1['b'], h['ab'][c,A,a,V], t['ab'][pa,pV,hc,hA], optimize=True)
    
    
    O['0'] += scale * +1.00000000 * np.einsum('uv,iIvA,uAiI->', eta1['a'], h['ab'][c,C,a,V], t['ab'][pa,pV,hc,hC], optimize=True)
    O['0'] += scale * +1.00000000 * np.einsum('uv,iUvV,UWVX,uWiX->', eta1['a'], h['ab'][c,A,a,A], lambdas['bb'], t['ab'][pa,pA,hc,hA], optimize=True)
    
    O['0'] += scale * -1.00000000 * np.einsum('uv,wIvU,wVxU,uVxI->', eta1['a'], h['ab'][a,C,a,A], lambdas['ab'], t['ab'][pa,pA,ha,hC], optimize=True)
    
    O['0'] += scale * +1.00000000 * np.einsum('uv,wUvA,wUxV,uAxV->', eta1['a'], h['ab'][a,A,a,V], lambdas['ab'], t['ab'][pa,pV,ha,hA], optimize=True)
    O['0'] += scale * +1.00000000 * np.einsum('UV,uv,wx,wIvV,uUxI->', eta1['b'], eta1['a'], gamma1['a'], h['ab'][a,C,a,A], t['ab'][pa,pA,ha,hC], optimize=True)
    
    
    O['0'] += scale * +1.00000000 * np.einsum('UV,uv,WX,iWvV,uUiX->', eta1['b'], eta1['a'], gamma1['b'], h['ab'][c,A,a,A], t['ab'][pa,pA,hc,hA], optimize=True)
    O['0'] += scale * +1.00000000 * np.einsum('UV,uv,iIvV,uUiI->', eta1['b'], eta1['a'], h['ab'][c,C,a,A], t['ab'][pa,pA,hc,hC], optimize=True)
    
    
    O['0'] += scale * +1.00000000 * np.einsum('UV,uv,uIaV,aUvI->', eta1['b'], gamma1['a'], h['ab'][a,C,v,A], t['ab'][pv,pA,ha,hC], optimize=True)
    
    
    O['0'] += scale * +1.00000000 * np.einsum('UV,WX,uv,uWaV,aUvX->', eta1['b'], gamma1['b'], gamma1['a'], h['ab'][a,A,v,A], t['ab'][pv,pA,ha,hA], optimize=True)
    O['0'] += scale * +1.00000000 * np.einsum('UV,WX,iWaV,aUiX->', eta1['b'], gamma1['b'], h['ab'][c,A,v,A], t['ab'][pv,pA,hc,hA], optimize=True)
    
    
    O['0'] += scale * +1.00000000 * np.einsum('UV,iIaV,aUiI->', eta1['b'], h['ab'][c,C,v,A], t['ab'][pv,pA,hc,hC], optimize=True)
    O['0'] += scale * -1.00000000 * np.einsum('UV,iWuV,vWuX,vUiX->', eta1['b'], h['ab'][c,A,a,A], lambdas['ab'], t['ab'][pa,pA,hc,hA], optimize=True)
    O['0'] += scale * +1.00000000 * np.einsum('UV,uIvV,uwvx,wUxI->', eta1['b'], h['ab'][a,C,a,A], lambdas['aa'], t['ab'][pa,pA,ha,hC], optimize=True)
    
    
    O['0'] += scale * +1.00000000 * np.einsum('UV,uWaV,uWvX,aUvX->', eta1['b'], h['ab'][a,A,v,A], lambdas['ab'], t['ab'][pv,pA,ha,hA], optimize=True)
    O['0'] += scale * +1.00000000 * np.einsum('uv,uIwU,xVwU,xVvI->', gamma1['a'], h['ab'][a,C,a,A], lambdas['ab'], t['ab'][pa,pA,ha,hC], optimize=True)
    O['0'] += scale * +1.00000000 * np.einsum('uv,uIaA,aAvI->', gamma1['a'], h['ab'][a,C,v,V], t['ab'][pv,pV,ha,hC], optimize=True)
    
    O['0'] += scale * -1.00000000 * np.einsum('uv,uUwA,xUwV,xAvV->', gamma1['a'], h['ab'][a,A,a,V], lambdas['ab'], t['ab'][pa,pV,ha,hA], optimize=True)
    O['0'] += scale * +1.00000000 * np.einsum('uv,uUaV,UWVX,aWvX->', gamma1['a'], h['ab'][a,A,v,A], lambdas['bb'], t['ab'][pv,pA,ha,hA], optimize=True)
    
    
    O['0'] += scale * +1.00000000 * np.einsum('UV,uv,uUaA,aAvV->', gamma1['b'], gamma1['a'], h['ab'][a,A,v,V], t['ab'][pv,pV,ha,hA], optimize=True)
    
    O['0'] += scale * +1.00000000 * np.einsum('UV,iUuW,vXuW,vXiV->', gamma1['b'], h['ab'][c,A,a,A], lambdas['ab'], t['ab'][pa,pA,hc,hA], optimize=True)
    O['0'] += scale * +1.00000000 * np.einsum('UV,iUaA,aAiV->', gamma1['b'], h['ab'][c,A,v,V], t['ab'][pv,pV,hc,hA], optimize=True)
    
    O['0'] += scale * +1.00000000 * np.einsum('UV,uUvA,uwvx,wAxV->', gamma1['b'], h['ab'][a,A,a,V], lambdas['aa'], t['ab'][pa,pV,ha,hA], optimize=True)
    O['0'] += scale * -1.00000000 * np.einsum('UV,uUaW,uXvW,aXvV->', gamma1['b'], h['ab'][a,A,v,A], lambdas['ab'], t['ab'][pv,pA,ha,hA], optimize=True)
    
    O['0'] += scale * +1.00000000 * np.einsum('iIuU,vVuU,vViI->', h['ab'][c,C,a,A], lambdas['ab'], t['ab'][pa,pA,hc,hC], optimize=True)
    O['0'] += scale * +1.00000000 * np.einsum('iIaA,aAiI->', h['ab'][c,C,v,V], t['ab'][pv,pV,hc,hC], optimize=True)
    
    O['0'] += scale * -1.00000000 * np.einsum('iUuA,vUuV,vAiV->', h['ab'][c,A,a,V], lambdas['ab'], t['ab'][pa,pV,hc,hA], optimize=True)
    O['0'] += scale * +1.00000000 * np.einsum('iUaV,UWVX,aWiX->', h['ab'][c,A,v,A], lambdas['bb'], t['ab'][pv,pA,hc,hA], optimize=True)
    
    O['0'] += scale * +1.00000000 * np.einsum('uIvA,uwvx,wAxI->', h['ab'][a,C,a,V], lambdas['aa'], t['ab'][pa,pV,ha,hC], optimize=True)
    O['0'] += scale * -1.00000000 * np.einsum('uIaU,uVvU,aVvI->', h['ab'][a,C,v,A], lambdas['ab'], t['ab'][pv,pA,ha,hC], optimize=True)
    
    
    O['0'] += scale * +1.00000000 * np.einsum('uUaA,uUvV,aAvV->', h['ab'][a,A,v,V], lambdas['ab'], t['ab'][pv,pV,ha,hA], optimize=True)

    O['0'] += scale * -1.00000000 * np.einsum('iUuV,vUWuVX,vWiX->', h['ab'][c,A,a,A], lambdas['abb'], t['ab'][pa,pA,hc,hA], optimize=True)
    O['0'] += scale * -1.00000000 * np.einsum('uIvU,uwVvxU,wVxI->', h['ab'][a,C,a,A], lambdas['aab'], t['ab'][pa,pA,ha,hC], optimize=True)
    O['0'] += scale * +1.00000000 * np.einsum('uUvA,uwUvxV,wAxV->', h['ab'][a,A,a,V], lambdas['aab'], t['ab'][pa,pV,ha,hA], optimize=True)
    O['0'] += scale * +1.00000000 * np.einsum('uUaV,uUWvVX,aWvX->', h['ab'][a,A,v,A], lambdas['abb'], t['ab'][pv,pA,ha,hA], optimize=True)
    t1 = time.time()
    if verbose: print("h2b_t2b_c0 took {:.4f} seconds to run.".format(t1-t0))

    return O
def h2b_t2c_c0(O, h, t, gamma1, eta1, lambdas, orbspace, verbose=False, scale=1.0):
    # 11 lines
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

    
    
    O['0'] += scale * +1.00000000 * np.einsum('UV,uIvV,uWvX,UWIX->', eta1['b'], h['ab'][a,C,a,A], lambdas['ab'], t['bb'][pA,pA,hC,hA], optimize=True)
    
    
    
    O['0'] += scale * -1.00000000 * np.einsum('UV,uUvA,uWvX,WAVX->', gamma1['b'], h['ab'][a,A,a,V], lambdas['ab'], t['bb'][pA,pV,hA,hA], optimize=True)
    
    
    O['0'] += scale * -1.00000000 * np.einsum('uIvA,uUvV,UAIV->', h['ab'][a,C,a,V], lambdas['ab'], t['bb'][pA,pV,hC,hA], optimize=True)
    O['0'] += scale * -0.50000000 * np.einsum('uIvU,uVWvUX,VWIX->', h['ab'][a,C,a,A], lambdas['abb'], t['bb'][pA,pA,hC,hA], optimize=True)
    O['0'] += scale * -0.50000000 * np.einsum('uUvA,uUVvWX,VAWX->', h['ab'][a,A,a,V], lambdas['abb'], t['bb'][pA,pV,hA,hA], optimize=True)
    

    t1 = time.time()
    if verbose: print("h2b_t2c_c0 took {:.4f} seconds to run.".format(t1-t0))

    return O
def h2c_t1b_c0(O, h, t, gamma1, eta1, lambdas, orbspace, verbose=False, scale=1.0):
    # 6 lines
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

    
    
    
    
    O['0'] += scale * +0.50000000 * np.einsum('IUVW,UXVW,XI->', h['bb'][C,A,A,A], lambdas['bb'], t['b'][pA,hC], optimize=True)
    O['0'] += scale * +0.50000000 * np.einsum('UVWA,UVWX,AX->', h['bb'][A,A,A,V], lambdas['bb'], t['b'][pV,hA], optimize=True)

    t1 = time.time()
    if verbose: print("h2c_t1b_c0 took {:.4f} seconds to run.".format(t1-t0))

    return O
def h2c_t2b_c0(O, h, t, gamma1, eta1, lambdas, orbspace, verbose=False, scale=1.0):
    # 11 lines
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

    
    
    O['0'] += scale * +1.00000000 * np.einsum('UV,IWVX,uWvX,uUvI->', eta1['b'], h['bb'][C,A,A,A], lambdas['ab'], t['ab'][pa,pA,ha,hC], optimize=True)
    
    
    
    O['0'] += scale * -1.00000000 * np.einsum('UV,UWXA,uWvX,uAvV->', gamma1['b'], h['bb'][A,A,A,V], lambdas['ab'], t['ab'][pa,pV,ha,hA], optimize=True)
    
    
    O['0'] += scale * -1.00000000 * np.einsum('IUVA,uUvV,uAvI->', h['bb'][C,A,A,V], lambdas['ab'], t['ab'][pa,pV,ha,hC], optimize=True)
    O['0'] += scale * +0.50000000 * np.einsum('IUVW,uUXvVW,uXvI->', h['bb'][C,A,A,A], lambdas['abb'], t['ab'][pa,pA,ha,hC], optimize=True)
    O['0'] += scale * +0.50000000 * np.einsum('UVWA,uUVvWX,uAvX->', h['bb'][A,A,A,V], lambdas['abb'], t['ab'][pa,pV,ha,hA], optimize=True)
    

    t1 = time.time()
    if verbose: print("h2c_t2b_c0 took {:.4f} seconds to run.".format(t1-t0))

    return O
def h2c_t2c_c0(O, h, t, gamma1, eta1, lambdas, orbspace, verbose=False, scale=1.0):
    # 29 lines
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

    
    
    O['0'] += scale * +0.50000000 * np.einsum('UV,WX,YZ,IYVX,UWIZ->', eta1['b'], eta1['b'], gamma1['b'], h['bb'][C,A,A,A], t['bb'][pA,pA,hC,hA], optimize=True)
    O['0'] += scale * +0.25000000 * np.einsum('UV,WX,IJVX,UWIJ->', eta1['b'], eta1['b'], h['bb'][C,C,A,A], t['bb'][pA,pA,hC,hC], optimize=True)
    
    
    O['0'] += scale * +0.50000000 * np.einsum('UV,WX,YZ,WYVA,UAXZ->', eta1['b'], gamma1['b'], gamma1['b'], h['bb'][A,A,A,V], t['bb'][pA,pV,hA,hA], optimize=True)
    O['0'] += scale * +1.00000000 * np.einsum('UV,WX,IWVA,UAIX->', eta1['b'], gamma1['b'], h['bb'][C,A,A,V], t['bb'][pA,pV,hC,hA], optimize=True)
    
    
    O['0'] += scale * +0.50000000 * np.einsum('UV,IJVA,UAIJ->', eta1['b'], h['bb'][C,C,A,V], t['bb'][pA,pV,hC,hC], optimize=True)
    O['0'] += scale * +1.00000000 * np.einsum('UV,IWVX,WYXZ,UYIZ->', eta1['b'], h['bb'][C,A,A,A], lambdas['bb'], t['bb'][pA,pA,hC,hA], optimize=True)
    
    
    O['0'] += scale * +0.25000000 * np.einsum('UV,WXVA,WXYZ,UAYZ->', eta1['b'], h['bb'][A,A,A,V], lambdas['bb'], t['bb'][pA,pV,hA,hA], optimize=True)
    
    O['0'] += scale * +0.25000000 * np.einsum('UV,WX,UWAB,ABVX->', gamma1['b'], gamma1['b'], h['bb'][A,A,V,V], t['bb'][pV,pV,hA,hA], optimize=True)
    
    O['0'] += scale * +0.25000000 * np.einsum('UV,IUWX,YZWX,YZIV->', gamma1['b'], h['bb'][C,A,A,A], lambdas['bb'], t['bb'][pA,pA,hC,hA], optimize=True)
    O['0'] += scale * +0.50000000 * np.einsum('UV,IUAB,ABIV->', gamma1['b'], h['bb'][C,A,V,V], t['bb'][pV,pV,hC,hA], optimize=True)
    
    O['0'] += scale * +1.00000000 * np.einsum('UV,UWXA,WYXZ,YAVZ->', gamma1['b'], h['bb'][A,A,A,V], lambdas['bb'], t['bb'][pA,pV,hA,hA], optimize=True)
    
    O['0'] += scale * +0.12500000 * np.einsum('IJUV,WXUV,WXIJ->', h['bb'][C,C,A,A], lambdas['bb'], t['bb'][pA,pA,hC,hC], optimize=True)
    O['0'] += scale * +0.25000000 * np.einsum('IJAB,ABIJ->', h['bb'][C,C,V,V], t['bb'][pV,pV,hC,hC], optimize=True)
    
    O['0'] += scale * +1.00000000 * np.einsum('IUVA,UWVX,WAIX->', h['bb'][C,A,A,V], lambdas['bb'], t['bb'][pA,pV,hC,hA], optimize=True)
    
    O['0'] += scale * +0.12500000 * np.einsum('UVAB,UVWX,ABWX->', h['bb'][A,A,V,V], lambdas['bb'], t['bb'][pV,pV,hA,hA], optimize=True)
    O['0'] += scale * +0.25000000 * np.einsum('IUVW,UXYVWZ,XYIZ->', h['bb'][C,A,A,A], lambdas['bbb'], t['bb'][pA,pA,hC,hA], optimize=True)
    O['0'] += scale * -0.25000000 * np.einsum('UVWA,UVXWYZ,XAYZ->', h['bb'][A,A,A,V], lambdas['bbb'], t['bb'][pA,pV,hA,hA], optimize=True)

    t1 = time.time()
    if verbose: print("h2c_t2c_c0 took {:.4f} seconds to run.".format(t1-t0))

    return O
def h1a_t1a_c1a(O, h, t, gamma1, eta1, lambdas, orbspace, verbose=False, scale=1.0):
    # 36 lines
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

    
    
    
    
    O['a'][a,a] += scale * -1.00000000 * np.einsum('iu,vi->vu', h['a'][c,a], t['a'][pa,hc], optimize=True)
    O['a'][a,a] += scale * +1.00000000 * np.einsum('ua,av->uv', h['a'][a,v], t['a'][pv,ha], optimize=True)
    
    
    O['a'][c,a] += scale * +1.00000000 * np.einsum('ia,au->iu', h['a'][c,v], t['a'][pv,ha], optimize=True)
    O['a'][v,a] += scale * -1.00000000 * np.einsum('uv,uw,av->aw', eta1['a'], h['a'][a,a], t['a'][pv,ha], optimize=True)
    
    O['a'][v,a] += scale * -1.00000000 * np.einsum('uv,uw,av->aw', gamma1['a'], h['a'][a,a], t['a'][pv,ha], optimize=True)
    
    O['a'][v,a] += scale * -1.00000000 * np.einsum('iu,ai->au', h['a'][c,a], t['a'][pv,hc], optimize=True)
    O['a'][v,a] += scale * +1.00000000 * np.einsum('ab,bu->au', h['a'][v,v], t['a'][pv,ha], optimize=True)
    O['a'][a,c] += scale * +1.00000000 * np.einsum('uv,wv,ui->wi', eta1['a'], h['a'][a,a], t['a'][pa,hc], optimize=True)
    
    O['a'][a,c] += scale * +1.00000000 * np.einsum('uv,wv,ui->wi', gamma1['a'], h['a'][a,a], t['a'][pa,hc], optimize=True)
    
    O['a'][a,c] += scale * -1.00000000 * np.einsum('ij,ui->uj', h['a'][c,c], t['a'][pa,hc], optimize=True)
    O['a'][a,c] += scale * +1.00000000 * np.einsum('ua,ai->ui', h['a'][a,v], t['a'][pv,hc], optimize=True)
    O['a'][c,c] += scale * +1.00000000 * np.einsum('uv,iv,uj->ij', eta1['a'], h['a'][c,a], t['a'][pa,hc], optimize=True)
    O['a'][c,c] += scale * +1.00000000 * np.einsum('uv,iv,uj->ij', gamma1['a'], h['a'][c,a], t['a'][pa,hc], optimize=True)
    O['a'][c,c] += scale * +1.00000000 * np.einsum('ia,aj->ij', h['a'][c,v], t['a'][pv,hc], optimize=True)
    O['a'][v,c] += scale * -1.00000000 * np.einsum('uv,ui,av->ai', eta1['a'], h['a'][a,c], t['a'][pv,ha], optimize=True)
    O['a'][v,c] += scale * +1.00000000 * np.einsum('uv,av,ui->ai', eta1['a'], h['a'][v,a], t['a'][pa,hc], optimize=True)
    O['a'][v,c] += scale * -1.00000000 * np.einsum('uv,ui,av->ai', gamma1['a'], h['a'][a,c], t['a'][pv,ha], optimize=True)
    O['a'][v,c] += scale * +1.00000000 * np.einsum('uv,av,ui->ai', gamma1['a'], h['a'][v,a], t['a'][pa,hc], optimize=True)
    O['a'][v,c] += scale * -1.00000000 * np.einsum('ij,ai->aj', h['a'][c,c], t['a'][pv,hc], optimize=True)
    O['a'][v,c] += scale * +1.00000000 * np.einsum('ab,bi->ai', h['a'][v,v], t['a'][pv,hc], optimize=True)
    
    
    O['a'][a,v] += scale * -1.00000000 * np.einsum('ia,ui->ua', h['a'][c,v], t['a'][pa,hc], optimize=True)
    O['a'][v,v] += scale * -1.00000000 * np.einsum('uv,ua,bv->ba', eta1['a'], h['a'][a,v], t['a'][pv,ha], optimize=True)
    O['a'][v,v] += scale * -1.00000000 * np.einsum('uv,ua,bv->ba', gamma1['a'], h['a'][a,v], t['a'][pv,ha], optimize=True)
    O['a'][v,v] += scale * -1.00000000 * np.einsum('ia,bi->ba', h['a'][c,v], t['a'][pv,hc], optimize=True)

    t1 = time.time()
    if verbose: print("h1a_t1a_c1a took {:.4f} seconds to run.".format(t1-t0))

    return O
def h1a_t2a_c1a(O, h, t, gamma1, eta1, lambdas, orbspace, verbose=False, scale=1.0):
    # 20 lines
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

    
    
    O['a'][a,a] += scale * -1.00000000 * np.einsum('uv,iv,wuix->wx', eta1['a'], h['a'][c,a], t['aa'][pa,pa,hc,ha], optimize=True)
    O['a'][a,a] += scale * +1.00000000 * np.einsum('uv,ua,waxv->wx', gamma1['a'], h['a'][a,v], t['aa'][pa,pv,ha,ha], optimize=True)
    O['a'][a,a] += scale * -1.00000000 * np.einsum('ia,uaiv->uv', h['a'][c,v], t['aa'][pa,pv,hc,ha], optimize=True)
    O['a'][v,a] += scale * +1.00000000 * np.einsum('uv,wx,ux,wayv->ay', eta1['a'], gamma1['a'], h['a'][a,a], t['aa'][pa,pv,ha,ha], optimize=True)
    O['a'][v,a] += scale * -1.00000000 * np.einsum('uv,wx,wv,uayx->ay', eta1['a'], gamma1['a'], h['a'][a,a], t['aa'][pa,pv,ha,ha], optimize=True)
    O['a'][v,a] += scale * +1.00000000 * np.einsum('uv,iv,uaiw->aw', eta1['a'], h['a'][c,a], t['aa'][pa,pv,hc,ha], optimize=True)
    O['a'][v,a] += scale * +1.00000000 * np.einsum('uv,ua,bawv->bw', gamma1['a'], h['a'][a,v], t['aa'][pv,pv,ha,ha], optimize=True)
    O['a'][v,a] += scale * -1.00000000 * np.einsum('ia,baiu->bu', h['a'][c,v], t['aa'][pv,pv,hc,ha], optimize=True)
    O['a'][a,c] += scale * -1.00000000 * np.einsum('uv,wx,ux,ywiv->yi', eta1['a'], gamma1['a'], h['a'][a,a], t['aa'][pa,pa,hc,ha], optimize=True)
    O['a'][a,c] += scale * +1.00000000 * np.einsum('uv,wx,wv,yuix->yi', eta1['a'], gamma1['a'], h['a'][a,a], t['aa'][pa,pa,hc,ha], optimize=True)
    O['a'][a,c] += scale * +1.00000000 * np.einsum('uv,iv,wuji->wj', eta1['a'], h['a'][c,a], t['aa'][pa,pa,hc,hc], optimize=True)
    O['a'][a,c] += scale * +1.00000000 * np.einsum('uv,ua,waiv->wi', gamma1['a'], h['a'][a,v], t['aa'][pa,pv,hc,ha], optimize=True)
    O['a'][a,c] += scale * +1.00000000 * np.einsum('ia,uaji->uj', h['a'][c,v], t['aa'][pa,pv,hc,hc], optimize=True)
    O['a'][v,c] += scale * +1.00000000 * np.einsum('uv,wx,ux,waiv->ai', eta1['a'], gamma1['a'], h['a'][a,a], t['aa'][pa,pv,hc,ha], optimize=True)
    O['a'][v,c] += scale * -1.00000000 * np.einsum('uv,wx,wv,uaix->ai', eta1['a'], gamma1['a'], h['a'][a,a], t['aa'][pa,pv,hc,ha], optimize=True)
    O['a'][v,c] += scale * -1.00000000 * np.einsum('uv,iv,uaji->aj', eta1['a'], h['a'][c,a], t['aa'][pa,pv,hc,hc], optimize=True)
    O['a'][v,c] += scale * +1.00000000 * np.einsum('uv,ua,baiv->bi', gamma1['a'], h['a'][a,v], t['aa'][pv,pv,hc,ha], optimize=True)
    O['a'][v,c] += scale * +1.00000000 * np.einsum('ia,baji->bj', h['a'][c,v], t['aa'][pv,pv,hc,hc], optimize=True)

    t1 = time.time()
    if verbose: print("h1a_t2a_c1a took {:.4f} seconds to run.".format(t1-t0))

    return O
def h1b_t2b_c1a(O, h, t, gamma1, eta1, lambdas, orbspace, verbose=False, scale=1.0):
    # 20 lines
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

    
    
    O['a'][a,a] += scale * +1.00000000 * np.einsum('UV,IV,uUvI->uv', eta1['b'], h['b'][C,A], t['ab'][pa,pA,ha,hC], optimize=True)
    O['a'][a,a] += scale * +1.00000000 * np.einsum('UV,UA,uAvV->uv', gamma1['b'], h['b'][A,V], t['ab'][pa,pV,ha,hA], optimize=True)
    O['a'][a,a] += scale * +1.00000000 * np.einsum('IA,uAvI->uv', h['b'][C,V], t['ab'][pa,pV,ha,hC], optimize=True)
    O['a'][v,a] += scale * -1.00000000 * np.einsum('UV,WX,UX,aWuV->au', eta1['b'], gamma1['b'], h['b'][A,A], t['ab'][pv,pA,ha,hA], optimize=True)
    O['a'][v,a] += scale * +1.00000000 * np.einsum('UV,WX,WV,aUuX->au', eta1['b'], gamma1['b'], h['b'][A,A], t['ab'][pv,pA,ha,hA], optimize=True)
    O['a'][v,a] += scale * +1.00000000 * np.einsum('UV,IV,aUuI->au', eta1['b'], h['b'][C,A], t['ab'][pv,pA,ha,hC], optimize=True)
    O['a'][v,a] += scale * +1.00000000 * np.einsum('UV,UA,aAuV->au', gamma1['b'], h['b'][A,V], t['ab'][pv,pV,ha,hA], optimize=True)
    O['a'][v,a] += scale * +1.00000000 * np.einsum('IA,aAuI->au', h['b'][C,V], t['ab'][pv,pV,ha,hC], optimize=True)
    O['a'][a,c] += scale * -1.00000000 * np.einsum('UV,WX,UX,uWiV->ui', eta1['b'], gamma1['b'], h['b'][A,A], t['ab'][pa,pA,hc,hA], optimize=True)
    O['a'][a,c] += scale * +1.00000000 * np.einsum('UV,WX,WV,uUiX->ui', eta1['b'], gamma1['b'], h['b'][A,A], t['ab'][pa,pA,hc,hA], optimize=True)
    O['a'][a,c] += scale * +1.00000000 * np.einsum('UV,IV,uUiI->ui', eta1['b'], h['b'][C,A], t['ab'][pa,pA,hc,hC], optimize=True)
    O['a'][a,c] += scale * +1.00000000 * np.einsum('UV,UA,uAiV->ui', gamma1['b'], h['b'][A,V], t['ab'][pa,pV,hc,hA], optimize=True)
    O['a'][a,c] += scale * +1.00000000 * np.einsum('IA,uAiI->ui', h['b'][C,V], t['ab'][pa,pV,hc,hC], optimize=True)
    O['a'][v,c] += scale * -1.00000000 * np.einsum('UV,WX,UX,aWiV->ai', eta1['b'], gamma1['b'], h['b'][A,A], t['ab'][pv,pA,hc,hA], optimize=True)
    O['a'][v,c] += scale * +1.00000000 * np.einsum('UV,WX,WV,aUiX->ai', eta1['b'], gamma1['b'], h['b'][A,A], t['ab'][pv,pA,hc,hA], optimize=True)
    O['a'][v,c] += scale * +1.00000000 * np.einsum('UV,IV,aUiI->ai', eta1['b'], h['b'][C,A], t['ab'][pv,pA,hc,hC], optimize=True)
    O['a'][v,c] += scale * +1.00000000 * np.einsum('UV,UA,aAiV->ai', gamma1['b'], h['b'][A,V], t['ab'][pv,pV,hc,hA], optimize=True)
    O['a'][v,c] += scale * +1.00000000 * np.einsum('IA,aAiI->ai', h['b'][C,V], t['ab'][pv,pV,hc,hC], optimize=True)

    t1 = time.time()
    if verbose: print("h1b_t2b_c1a took {:.4f} seconds to run.".format(t1-t0))

    return O
def h2a_t1a_c1a(O, h, t, gamma1, eta1, lambdas, orbspace, verbose=False, scale=1.0):
    # 45 lines
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

    
    
    O['a'][a,a] += scale * -1.00000000 * np.einsum('uv,iwxv,ui->wx', eta1['a'], h['aa'][c,a,a,a], t['a'][pa,hc], optimize=True)
    O['a'][a,a] += scale * +1.00000000 * np.einsum('uv,wuxa,av->wx', gamma1['a'], h['aa'][a,a,a,v], t['a'][pv,ha], optimize=True)
    O['a'][a,a] += scale * -1.00000000 * np.einsum('iuva,ai->uv', h['aa'][c,a,a,v], t['a'][pv,hc], optimize=True)
    
    
    O['a'][c,a] += scale * +1.00000000 * np.einsum('uv,ijwv,uj->iw', eta1['a'], h['aa'][c,c,a,a], t['a'][pa,hc], optimize=True)
    O['a'][c,a] += scale * +1.00000000 * np.einsum('uv,iuwa,av->iw', gamma1['a'], h['aa'][c,a,a,v], t['a'][pv,ha], optimize=True)
    O['a'][c,a] += scale * +1.00000000 * np.einsum('ijua,aj->iu', h['aa'][c,c,a,v], t['a'][pv,hc], optimize=True)
    
    
    O['a'][v,a] += scale * -1.00000000 * np.einsum('uv,iawv,ui->aw', eta1['a'], h['aa'][c,v,a,a], t['a'][pa,hc], optimize=True)
    O['a'][v,a] += scale * -1.00000000 * np.einsum('uv,uawb,bv->aw', gamma1['a'], h['aa'][a,v,a,v], t['a'][pv,ha], optimize=True)
    O['a'][v,a] += scale * -1.00000000 * np.einsum('iaub,bi->au', h['aa'][c,v,a,v], t['a'][pv,hc], optimize=True)
    
    
    O['a'][a,c] += scale * -1.00000000 * np.einsum('uv,iwjv,ui->wj', eta1['a'], h['aa'][c,a,c,a], t['a'][pa,hc], optimize=True)
    O['a'][a,c] += scale * +1.00000000 * np.einsum('uv,wuia,av->wi', gamma1['a'], h['aa'][a,a,c,v], t['a'][pv,ha], optimize=True)
    O['a'][a,c] += scale * -1.00000000 * np.einsum('iuja,ai->uj', h['aa'][c,a,c,v], t['a'][pv,hc], optimize=True)
    
    
    O['a'][c,c] += scale * +1.00000000 * np.einsum('uv,ijkv,uj->ik', eta1['a'], h['aa'][c,c,c,a], t['a'][pa,hc], optimize=True)
    O['a'][c,c] += scale * +1.00000000 * np.einsum('uv,iuja,av->ij', gamma1['a'], h['aa'][c,a,c,v], t['a'][pv,ha], optimize=True)
    O['a'][c,c] += scale * +1.00000000 * np.einsum('ijka,aj->ik', h['aa'][c,c,c,v], t['a'][pv,hc], optimize=True)
    
    
    O['a'][v,c] += scale * -1.00000000 * np.einsum('uv,iajv,ui->aj', eta1['a'], h['aa'][c,v,c,a], t['a'][pa,hc], optimize=True)
    O['a'][v,c] += scale * -1.00000000 * np.einsum('uv,uaib,bv->ai', gamma1['a'], h['aa'][a,v,c,v], t['a'][pv,ha], optimize=True)
    O['a'][v,c] += scale * -1.00000000 * np.einsum('iajb,bi->aj', h['aa'][c,v,c,v], t['a'][pv,hc], optimize=True)
    
    
    O['a'][a,v] += scale * +1.00000000 * np.einsum('uv,iwva,ui->wa', eta1['a'], h['aa'][c,a,a,v], t['a'][pa,hc], optimize=True)
    O['a'][a,v] += scale * +1.00000000 * np.einsum('uv,wuab,bv->wa', gamma1['a'], h['aa'][a,a,v,v], t['a'][pv,ha], optimize=True)
    O['a'][a,v] += scale * -1.00000000 * np.einsum('iuab,bi->ua', h['aa'][c,a,v,v], t['a'][pv,hc], optimize=True)
    
    
    O['a'][c,v] += scale * -1.00000000 * np.einsum('uv,ijva,uj->ia', eta1['a'], h['aa'][c,c,a,v], t['a'][pa,hc], optimize=True)
    O['a'][c,v] += scale * +1.00000000 * np.einsum('uv,iuab,bv->ia', gamma1['a'], h['aa'][c,a,v,v], t['a'][pv,ha], optimize=True)
    O['a'][c,v] += scale * +1.00000000 * np.einsum('ijab,bj->ia', h['aa'][c,c,v,v], t['a'][pv,hc], optimize=True)
    
    
    O['a'][v,v] += scale * +1.00000000 * np.einsum('uv,iavb,ui->ab', eta1['a'], h['aa'][c,v,a,v], t['a'][pa,hc], optimize=True)
    O['a'][v,v] += scale * -1.00000000 * np.einsum('uv,uabc,cv->ab', gamma1['a'], h['aa'][a,v,v,v], t['a'][pv,ha], optimize=True)
    O['a'][v,v] += scale * -1.00000000 * np.einsum('iabc,ci->ab', h['aa'][c,v,v,v], t['a'][pv,hc], optimize=True)

    t1 = time.time()
    if verbose: print("h2a_t1a_c1a took {:.4f} seconds to run.".format(t1-t0))

    return O
def h2a_t2a_c1a(O, h, t, gamma1, eta1, lambdas, orbspace, verbose=False, scale=1.0):
    # 234 lines
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

    
    
    O['a'][a,a] += scale * +0.50000000 * np.einsum('uv,wx,iyvx,uwiz->yz', eta1['a'], eta1['a'], h['aa'][c,a,a,a], t['aa'][pa,pa,hc,ha], optimize=True)
    
    
    O['a'][a,a] += scale * -1.00000000 * np.einsum('uv,wx,iwyv,zuix->zy', eta1['a'], gamma1['a'], h['aa'][c,a,a,a], t['aa'][pa,pa,hc,ha], optimize=True)
    O['a'][a,a] += scale * +1.00000000 * np.einsum('uv,wx,ywva,uazx->yz', eta1['a'], gamma1['a'], h['aa'][a,a,a,v], t['aa'][pa,pv,ha,ha], optimize=True)
    O['a'][a,a] += scale * -0.50000000 * np.einsum('uv,ijwv,xuij->xw', eta1['a'], h['aa'][c,c,a,a], t['aa'][pa,pa,hc,hc], optimize=True)
    O['a'][a,a] += scale * +1.00000000 * np.einsum('uv,iwva,uaix->wx', eta1['a'], h['aa'][c,a,a,v], t['aa'][pa,pv,hc,ha], optimize=True)
    
    
    
    
    
    
    
    
    O['a'][a,a] += scale * -0.50000000 * np.einsum('uv,wx,uwya,zavx->zy', gamma1['a'], gamma1['a'], h['aa'][a,a,a,v], t['aa'][pa,pv,ha,ha], optimize=True)
    O['a'][a,a] += scale * -1.00000000 * np.einsum('uv,iuwa,xaiv->xw', gamma1['a'], h['aa'][c,a,a,v], t['aa'][pa,pv,hc,ha], optimize=True)
    
    
    O['a'][a,a] += scale * +0.50000000 * np.einsum('uv,wuab,abxv->wx', gamma1['a'], h['aa'][a,a,v,v], t['aa'][pv,pv,ha,ha], optimize=True)
    
    
    
    
    
    
    O['a'][a,a] += scale * -0.50000000 * np.einsum('ijua,vaij->vu', h['aa'][c,c,a,v], t['aa'][pa,pv,hc,hc], optimize=True)
    O['a'][a,a] += scale * +0.50000000 * np.einsum('iuvw,xywz,xyiz->uv', h['aa'][c,a,a,a], lambdas['aa'], t['aa'][pa,pa,hc,ha], optimize=True)
    O['a'][a,a] += scale * +0.25000000 * np.einsum('iuvw,xyvw,xyiz->uz', h['aa'][c,a,a,a], lambdas['aa'], t['aa'][pa,pa,hc,ha], optimize=True)
    O['a'][a,a] += scale * +0.50000000 * np.einsum('iuab,abiv->uv', h['aa'][c,a,v,v], t['aa'][pv,pv,hc,ha], optimize=True)
    O['a'][a,a] += scale * -1.00000000 * np.einsum('iuvw,uxwy,zxiy->zv', h['aa'][c,a,a,a], lambdas['aa'], t['aa'][pa,pa,hc,ha], optimize=True)
    O['a'][a,a] += scale * -0.50000000 * np.einsum('iuvw,uxvw,yxiz->yz', h['aa'][c,a,a,a], lambdas['aa'], t['aa'][pa,pa,hc,ha], optimize=True)
    O['a'][a,a] += scale * -0.50000000 * np.einsum('uvwa,vxyz,xayz->uw', h['aa'][a,a,a,v], lambdas['aa'], t['aa'][pa,pv,ha,ha], optimize=True)
    O['a'][a,a] += scale * +1.00000000 * np.einsum('uvwa,vxwy,xazy->uz', h['aa'][a,a,a,v], lambdas['aa'], t['aa'][pa,pv,ha,ha], optimize=True)
    O['a'][a,a] += scale * -0.25000000 * np.einsum('uvwa,uvxy,zaxy->zw', h['aa'][a,a,a,v], lambdas['aa'], t['aa'][pa,pv,ha,ha], optimize=True)
    O['a'][a,a] += scale * +0.50000000 * np.einsum('uvwa,uvwx,yazx->yz', h['aa'][a,a,a,v], lambdas['aa'], t['aa'][pa,pv,ha,ha], optimize=True)
    
    O['a'][c,a] += scale * -0.50000000 * np.einsum('uv,wx,ijvx,uwjy->iy', eta1['a'], eta1['a'], h['aa'][c,c,a,a], t['aa'][pa,pa,hc,ha], optimize=True)
    
    O['a'][c,a] += scale * +1.00000000 * np.einsum('uv,wx,iwva,uayx->iy', eta1['a'], gamma1['a'], h['aa'][c,a,a,v], t['aa'][pa,pv,ha,ha], optimize=True)
    O['a'][c,a] += scale * -1.00000000 * np.einsum('uv,ijva,uajw->iw', eta1['a'], h['aa'][c,c,a,v], t['aa'][pa,pv,hc,ha], optimize=True)
    
    
    
    
    
    
    O['a'][c,a] += scale * +0.50000000 * np.einsum('uv,iuab,abwv->iw', gamma1['a'], h['aa'][c,a,v,v], t['aa'][pv,pv,ha,ha], optimize=True)
    
    
    O['a'][c,a] += scale * -0.50000000 * np.einsum('ijuv,wxvy,wxjy->iu', h['aa'][c,c,a,a], lambdas['aa'], t['aa'][pa,pa,hc,ha], optimize=True)
    O['a'][c,a] += scale * -0.25000000 * np.einsum('ijuv,wxuv,wxjy->iy', h['aa'][c,c,a,a], lambdas['aa'], t['aa'][pa,pa,hc,ha], optimize=True)
    O['a'][c,a] += scale * -0.50000000 * np.einsum('ijab,abju->iu', h['aa'][c,c,v,v], t['aa'][pv,pv,hc,ha], optimize=True)
    O['a'][c,a] += scale * -0.50000000 * np.einsum('iuva,uwxy,waxy->iv', h['aa'][c,a,a,v], lambdas['aa'], t['aa'][pa,pv,ha,ha], optimize=True)
    O['a'][c,a] += scale * +1.00000000 * np.einsum('iuva,uwvx,wayx->iy', h['aa'][c,a,a,v], lambdas['aa'], t['aa'][pa,pv,ha,ha], optimize=True)
    O['a'][v,a] += scale * +0.50000000 * np.einsum('uv,wx,yz,uwrz,yavx->ar', eta1['a'], eta1['a'], gamma1['a'], h['aa'][a,a,a,a], t['aa'][pa,pv,ha,ha], optimize=True)
    
    O['a'][v,a] += scale * +0.50000000 * np.einsum('uv,wx,iavx,uwiy->ay', eta1['a'], eta1['a'], h['aa'][c,v,a,a], t['aa'][pa,pa,hc,ha], optimize=True)
    
    O['a'][v,a] += scale * +0.50000000 * np.einsum('uv,wx,yz,wyrv,uaxz->ar', eta1['a'], gamma1['a'], gamma1['a'], h['aa'][a,a,a,a], t['aa'][pa,pv,ha,ha], optimize=True)
    O['a'][v,a] += scale * +1.00000000 * np.einsum('uv,wx,iwyv,uaix->ay', eta1['a'], gamma1['a'], h['aa'][c,a,a,a], t['aa'][pa,pv,hc,ha], optimize=True)
    O['a'][v,a] += scale * -1.00000000 * np.einsum('uv,wx,wavb,ubyx->ay', eta1['a'], gamma1['a'], h['aa'][a,v,a,v], t['aa'][pa,pv,ha,ha], optimize=True)
    O['a'][v,a] += scale * +0.50000000 * np.einsum('uv,ijwv,uaij->aw', eta1['a'], h['aa'][c,c,a,a], t['aa'][pa,pv,hc,hc], optimize=True)
    O['a'][v,a] += scale * +1.00000000 * np.einsum('uv,iavb,ubiw->aw', eta1['a'], h['aa'][c,v,a,v], t['aa'][pa,pv,hc,ha], optimize=True)
    O['a'][v,a] += scale * +1.00000000 * np.einsum('uv,uwxy,wzyr,zavr->ax', eta1['a'], h['aa'][a,a,a,a], lambdas['aa'], t['aa'][pa,pv,ha,ha], optimize=True)
    O['a'][v,a] += scale * -0.50000000 * np.einsum('uv,uwxy,wzxy,zarv->ar', eta1['a'], h['aa'][a,a,a,a], lambdas['aa'], t['aa'][pa,pv,ha,ha], optimize=True)
    
    
    O['a'][v,a] += scale * +0.25000000 * np.einsum('uv,wxyv,wxzr,uazr->ay', eta1['a'], h['aa'][a,a,a,a], lambdas['aa'], t['aa'][pa,pv,ha,ha], optimize=True)
    O['a'][v,a] += scale * +0.50000000 * np.einsum('uv,wxvy,wxyz,uarz->ar', eta1['a'], h['aa'][a,a,a,a], lambdas['aa'], t['aa'][pa,pv,ha,ha], optimize=True)
    
    
    O['a'][v,a] += scale * -0.50000000 * np.einsum('uv,wx,uwya,bavx->by', gamma1['a'], gamma1['a'], h['aa'][a,a,a,v], t['aa'][pv,pv,ha,ha], optimize=True)
    O['a'][v,a] += scale * -1.00000000 * np.einsum('uv,iuwa,baiv->bw', gamma1['a'], h['aa'][c,a,a,v], t['aa'][pv,pv,hc,ha], optimize=True)
    O['a'][v,a] += scale * +1.00000000 * np.einsum('uv,uwxy,wzyr,zavr->ax', gamma1['a'], h['aa'][a,a,a,a], lambdas['aa'], t['aa'][pa,pv,ha,ha], optimize=True)
    O['a'][v,a] += scale * -0.50000000 * np.einsum('uv,uwxy,wzxy,zarv->ar', gamma1['a'], h['aa'][a,a,a,a], lambdas['aa'], t['aa'][pa,pv,ha,ha], optimize=True)
    
    
    O['a'][v,a] += scale * -0.50000000 * np.einsum('uv,uabc,bcwv->aw', gamma1['a'], h['aa'][a,v,v,v], t['aa'][pv,pv,ha,ha], optimize=True)
    O['a'][v,a] += scale * +0.25000000 * np.einsum('uv,wxyv,wxzr,uazr->ay', gamma1['a'], h['aa'][a,a,a,a], lambdas['aa'], t['aa'][pa,pv,ha,ha], optimize=True)
    O['a'][v,a] += scale * +0.50000000 * np.einsum('uv,wxvy,wxyz,uarz->ar', gamma1['a'], h['aa'][a,a,a,a], lambdas['aa'], t['aa'][pa,pv,ha,ha], optimize=True)
    
    
    O['a'][v,a] += scale * -0.50000000 * np.einsum('ijua,baij->bu', h['aa'][c,c,a,v], t['aa'][pv,pv,hc,hc], optimize=True)
    O['a'][v,a] += scale * +1.00000000 * np.einsum('iuvw,uxwy,xaiy->av', h['aa'][c,a,a,a], lambdas['aa'], t['aa'][pa,pv,hc,ha], optimize=True)
    O['a'][v,a] += scale * +0.50000000 * np.einsum('iuvw,uxvw,xaiy->ay', h['aa'][c,a,a,a], lambdas['aa'], t['aa'][pa,pv,hc,ha], optimize=True)
    O['a'][v,a] += scale * +0.50000000 * np.einsum('iauv,wxvy,wxiy->au', h['aa'][c,v,a,a], lambdas['aa'], t['aa'][pa,pa,hc,ha], optimize=True)
    O['a'][v,a] += scale * +0.25000000 * np.einsum('iauv,wxuv,wxiy->ay', h['aa'][c,v,a,a], lambdas['aa'], t['aa'][pa,pa,hc,ha], optimize=True)
    O['a'][v,a] += scale * +0.50000000 * np.einsum('iabc,bciu->au', h['aa'][c,v,v,v], t['aa'][pv,pv,hc,ha], optimize=True)
    O['a'][v,a] += scale * -0.25000000 * np.einsum('uvwa,uvxy,baxy->bw', h['aa'][a,a,a,v], lambdas['aa'], t['aa'][pv,pv,ha,ha], optimize=True)
    O['a'][v,a] += scale * +0.50000000 * np.einsum('uvwa,uvwx,bayx->by', h['aa'][a,a,a,v], lambdas['aa'], t['aa'][pv,pv,ha,ha], optimize=True)
    O['a'][v,a] += scale * +0.50000000 * np.einsum('uavb,uwxy,wbxy->av', h['aa'][a,v,a,v], lambdas['aa'], t['aa'][pa,pv,ha,ha], optimize=True)
    O['a'][v,a] += scale * -1.00000000 * np.einsum('uavb,uwvx,wbyx->ay', h['aa'][a,v,a,v], lambdas['aa'], t['aa'][pa,pv,ha,ha], optimize=True)
    O['a'][a,c] += scale * +0.50000000 * np.einsum('uv,wx,yz,ryvx,uwiz->ri', eta1['a'], eta1['a'], gamma1['a'], h['aa'][a,a,a,a], t['aa'][pa,pa,hc,ha], optimize=True)
    
    O['a'][a,c] += scale * -0.50000000 * np.einsum('uv,wx,iyvx,uwji->yj', eta1['a'], eta1['a'], h['aa'][c,a,a,a], t['aa'][pa,pa,hc,hc], optimize=True)
    O['a'][a,c] += scale * +0.50000000 * np.einsum('uv,wx,yz,ruxz,wyiv->ri', eta1['a'], gamma1['a'], gamma1['a'], h['aa'][a,a,a,a], t['aa'][pa,pa,hc,ha], optimize=True)
    
    O['a'][a,c] += scale * -1.00000000 * np.einsum('uv,wx,iwjv,yuix->yj', eta1['a'], gamma1['a'], h['aa'][c,a,c,a], t['aa'][pa,pa,hc,ha], optimize=True)
    O['a'][a,c] += scale * +1.00000000 * np.einsum('uv,wx,ywva,uaix->yi', eta1['a'], gamma1['a'], h['aa'][a,a,a,v], t['aa'][pa,pv,hc,ha], optimize=True)
    O['a'][a,c] += scale * -0.50000000 * np.einsum('uv,ijkv,wuij->wk', eta1['a'], h['aa'][c,c,c,a], t['aa'][pa,pa,hc,hc], optimize=True)
    O['a'][a,c] += scale * -1.00000000 * np.einsum('uv,iwva,uaji->wj', eta1['a'], h['aa'][c,a,a,v], t['aa'][pa,pv,hc,hc], optimize=True)
    
    O['a'][a,c] += scale * +0.25000000 * np.einsum('uv,wuxy,zrxy,zriv->wi', eta1['a'], h['aa'][a,a,a,a], lambdas['aa'], t['aa'][pa,pa,hc,ha], optimize=True)
    
    O['a'][a,c] += scale * +1.00000000 * np.einsum('uv,wxvy,xzyr,uzir->wi', eta1['a'], h['aa'][a,a,a,a], lambdas['aa'], t['aa'][pa,pa,hc,ha], optimize=True)
    
    O['a'][a,c] += scale * +0.50000000 * np.einsum('uv,uwxy,wzxy,rziv->ri', eta1['a'], h['aa'][a,a,a,a], lambdas['aa'], t['aa'][pa,pa,hc,ha], optimize=True)
    
    O['a'][a,c] += scale * -0.50000000 * np.einsum('uv,wxvy,wxyz,ruiz->ri', eta1['a'], h['aa'][a,a,a,a], lambdas['aa'], t['aa'][pa,pa,hc,ha], optimize=True)
    O['a'][a,c] += scale * -0.50000000 * np.einsum('uv,wx,uwia,yavx->yi', gamma1['a'], gamma1['a'], h['aa'][a,a,c,v], t['aa'][pa,pv,ha,ha], optimize=True)
    O['a'][a,c] += scale * -1.00000000 * np.einsum('uv,iuja,waiv->wj', gamma1['a'], h['aa'][c,a,c,v], t['aa'][pa,pv,hc,ha], optimize=True)
    
    O['a'][a,c] += scale * +0.25000000 * np.einsum('uv,wuxy,zrxy,zriv->wi', gamma1['a'], h['aa'][a,a,a,a], lambdas['aa'], t['aa'][pa,pa,hc,ha], optimize=True)
    O['a'][a,c] += scale * +0.50000000 * np.einsum('uv,wuab,abiv->wi', gamma1['a'], h['aa'][a,a,v,v], t['aa'][pv,pv,hc,ha], optimize=True)
    
    O['a'][a,c] += scale * +1.00000000 * np.einsum('uv,wxvy,xzyr,uzir->wi', gamma1['a'], h['aa'][a,a,a,a], lambdas['aa'], t['aa'][pa,pa,hc,ha], optimize=True)
    
    O['a'][a,c] += scale * +0.50000000 * np.einsum('uv,uwxy,wzxy,rziv->ri', gamma1['a'], h['aa'][a,a,a,a], lambdas['aa'], t['aa'][pa,pa,hc,ha], optimize=True)
    
    O['a'][a,c] += scale * -0.50000000 * np.einsum('uv,wxvy,wxyz,ruiz->ri', gamma1['a'], h['aa'][a,a,a,a], lambdas['aa'], t['aa'][pa,pa,hc,ha], optimize=True)
    O['a'][a,c] += scale * -0.50000000 * np.einsum('ijka,uaij->uk', h['aa'][c,c,c,v], t['aa'][pa,pv,hc,hc], optimize=True)
    O['a'][a,c] += scale * +0.50000000 * np.einsum('iujv,wxvy,wxiy->uj', h['aa'][c,a,c,a], lambdas['aa'], t['aa'][pa,pa,hc,ha], optimize=True)
    O['a'][a,c] += scale * -0.25000000 * np.einsum('iuvw,xyvw,xyji->uj', h['aa'][c,a,a,a], lambdas['aa'], t['aa'][pa,pa,hc,hc], optimize=True)
    O['a'][a,c] += scale * -0.50000000 * np.einsum('iuab,abji->uj', h['aa'][c,a,v,v], t['aa'][pv,pv,hc,hc], optimize=True)
    O['a'][a,c] += scale * -1.00000000 * np.einsum('iujv,uwvx,ywix->yj', h['aa'][c,a,c,a], lambdas['aa'], t['aa'][pa,pa,hc,ha], optimize=True)
    O['a'][a,c] += scale * +0.50000000 * np.einsum('iuvw,uxvw,yxji->yj', h['aa'][c,a,a,a], lambdas['aa'], t['aa'][pa,pa,hc,hc], optimize=True)
    O['a'][a,c] += scale * -0.50000000 * np.einsum('uvia,vwxy,waxy->ui', h['aa'][a,a,c,v], lambdas['aa'], t['aa'][pa,pv,ha,ha], optimize=True)
    O['a'][a,c] += scale * +1.00000000 * np.einsum('uvwa,vxwy,xaiy->ui', h['aa'][a,a,a,v], lambdas['aa'], t['aa'][pa,pv,hc,ha], optimize=True)
    O['a'][a,c] += scale * -0.25000000 * np.einsum('uvia,uvwx,yawx->yi', h['aa'][a,a,c,v], lambdas['aa'], t['aa'][pa,pv,ha,ha], optimize=True)
    O['a'][a,c] += scale * +0.50000000 * np.einsum('uvwa,uvwx,yaix->yi', h['aa'][a,a,a,v], lambdas['aa'], t['aa'][pa,pv,hc,ha], optimize=True)
    O['a'][c,c] += scale * +0.50000000 * np.einsum('uv,wx,yz,iyvx,uwjz->ij', eta1['a'], eta1['a'], gamma1['a'], h['aa'][c,a,a,a], t['aa'][pa,pa,hc,ha], optimize=True)
    O['a'][c,c] += scale * +0.50000000 * np.einsum('uv,wx,ijvx,uwkj->ik', eta1['a'], eta1['a'], h['aa'][c,c,a,a], t['aa'][pa,pa,hc,hc], optimize=True)
    O['a'][c,c] += scale * +0.50000000 * np.einsum('uv,wx,yz,iuxz,wyjv->ij', eta1['a'], gamma1['a'], gamma1['a'], h['aa'][c,a,a,a], t['aa'][pa,pa,hc,ha], optimize=True)
    O['a'][c,c] += scale * +1.00000000 * np.einsum('uv,wx,iwva,uajx->ij', eta1['a'], gamma1['a'], h['aa'][c,a,a,v], t['aa'][pa,pv,hc,ha], optimize=True)
    O['a'][c,c] += scale * +1.00000000 * np.einsum('uv,ijva,uakj->ik', eta1['a'], h['aa'][c,c,a,v], t['aa'][pa,pv,hc,hc], optimize=True)
    
    O['a'][c,c] += scale * +0.25000000 * np.einsum('uv,iuwx,yzwx,yzjv->ij', eta1['a'], h['aa'][c,a,a,a], lambdas['aa'], t['aa'][pa,pa,hc,ha], optimize=True)
    
    O['a'][c,c] += scale * +1.00000000 * np.einsum('uv,iwvx,wyxz,uyjz->ij', eta1['a'], h['aa'][c,a,a,a], lambdas['aa'], t['aa'][pa,pa,hc,ha], optimize=True)
    
    O['a'][c,c] += scale * +0.25000000 * np.einsum('uv,iuwx,yzwx,yzjv->ij', gamma1['a'], h['aa'][c,a,a,a], lambdas['aa'], t['aa'][pa,pa,hc,ha], optimize=True)
    O['a'][c,c] += scale * +0.50000000 * np.einsum('uv,iuab,abjv->ij', gamma1['a'], h['aa'][c,a,v,v], t['aa'][pv,pv,hc,ha], optimize=True)
    
    O['a'][c,c] += scale * +1.00000000 * np.einsum('uv,iwvx,wyxz,uyjz->ij', gamma1['a'], h['aa'][c,a,a,a], lambdas['aa'], t['aa'][pa,pa,hc,ha], optimize=True)
    O['a'][c,c] += scale * -0.50000000 * np.einsum('ijku,vwux,vwjx->ik', h['aa'][c,c,c,a], lambdas['aa'], t['aa'][pa,pa,hc,ha], optimize=True)
    O['a'][c,c] += scale * +0.25000000 * np.einsum('ijuv,wxuv,wxkj->ik', h['aa'][c,c,a,a], lambdas['aa'], t['aa'][pa,pa,hc,hc], optimize=True)
    O['a'][c,c] += scale * +0.50000000 * np.einsum('ijab,abkj->ik', h['aa'][c,c,v,v], t['aa'][pv,pv,hc,hc], optimize=True)
    O['a'][c,c] += scale * -0.50000000 * np.einsum('iuja,uvwx,vawx->ij', h['aa'][c,a,c,v], lambdas['aa'], t['aa'][pa,pv,ha,ha], optimize=True)
    O['a'][c,c] += scale * +1.00000000 * np.einsum('iuva,uwvx,wajx->ij', h['aa'][c,a,a,v], lambdas['aa'], t['aa'][pa,pv,hc,ha], optimize=True)
    O['a'][v,c] += scale * +0.50000000 * np.einsum('uv,wx,yz,uwiz,yavx->ai', eta1['a'], eta1['a'], gamma1['a'], h['aa'][a,a,c,a], t['aa'][pa,pv,ha,ha], optimize=True)
    O['a'][v,c] += scale * -0.50000000 * np.einsum('uv,wx,yz,yavx,uwiz->ai', eta1['a'], eta1['a'], gamma1['a'], h['aa'][a,v,a,a], t['aa'][pa,pa,hc,ha], optimize=True)
    O['a'][v,c] += scale * -0.50000000 * np.einsum('uv,wx,iavx,uwji->aj', eta1['a'], eta1['a'], h['aa'][c,v,a,a], t['aa'][pa,pa,hc,hc], optimize=True)
    O['a'][v,c] += scale * -0.50000000 * np.einsum('uv,wx,yz,uaxz,wyiv->ai', eta1['a'], gamma1['a'], gamma1['a'], h['aa'][a,v,a,a], t['aa'][pa,pa,hc,ha], optimize=True)
    O['a'][v,c] += scale * +0.50000000 * np.einsum('uv,wx,yz,wyiv,uaxz->ai', eta1['a'], gamma1['a'], gamma1['a'], h['aa'][a,a,c,a], t['aa'][pa,pv,ha,ha], optimize=True)
    O['a'][v,c] += scale * +1.00000000 * np.einsum('uv,wx,iwjv,uaix->aj', eta1['a'], gamma1['a'], h['aa'][c,a,c,a], t['aa'][pa,pv,hc,ha], optimize=True)
    O['a'][v,c] += scale * -1.00000000 * np.einsum('uv,wx,wavb,ubix->ai', eta1['a'], gamma1['a'], h['aa'][a,v,a,v], t['aa'][pa,pv,hc,ha], optimize=True)
    O['a'][v,c] += scale * +0.50000000 * np.einsum('uv,ijkv,uaij->ak', eta1['a'], h['aa'][c,c,c,a], t['aa'][pa,pv,hc,hc], optimize=True)
    O['a'][v,c] += scale * -1.00000000 * np.einsum('uv,iavb,ubji->aj', eta1['a'], h['aa'][c,v,a,v], t['aa'][pa,pv,hc,hc], optimize=True)
    O['a'][v,c] += scale * +1.00000000 * np.einsum('uv,uwix,wyxz,yavz->ai', eta1['a'], h['aa'][a,a,c,a], lambdas['aa'], t['aa'][pa,pv,ha,ha], optimize=True)
    O['a'][v,c] += scale * -0.50000000 * np.einsum('uv,uwxy,wzxy,zaiv->ai', eta1['a'], h['aa'][a,a,a,a], lambdas['aa'], t['aa'][pa,pv,hc,ha], optimize=True)
    
    O['a'][v,c] += scale * -0.25000000 * np.einsum('uv,uawx,yzwx,yziv->ai', eta1['a'], h['aa'][a,v,a,a], lambdas['aa'], t['aa'][pa,pa,hc,ha], optimize=True)
    O['a'][v,c] += scale * +0.25000000 * np.einsum('uv,wxiv,wxyz,uayz->ai', eta1['a'], h['aa'][a,a,c,a], lambdas['aa'], t['aa'][pa,pv,ha,ha], optimize=True)
    O['a'][v,c] += scale * +0.50000000 * np.einsum('uv,wxvy,wxyz,uaiz->ai', eta1['a'], h['aa'][a,a,a,a], lambdas['aa'], t['aa'][pa,pv,hc,ha], optimize=True)
    
    O['a'][v,c] += scale * -1.00000000 * np.einsum('uv,wavx,wyxz,uyiz->ai', eta1['a'], h['aa'][a,v,a,a], lambdas['aa'], t['aa'][pa,pa,hc,ha], optimize=True)
    O['a'][v,c] += scale * -0.50000000 * np.einsum('uv,wx,uwia,bavx->bi', gamma1['a'], gamma1['a'], h['aa'][a,a,c,v], t['aa'][pv,pv,ha,ha], optimize=True)
    O['a'][v,c] += scale * -1.00000000 * np.einsum('uv,iuja,baiv->bj', gamma1['a'], h['aa'][c,a,c,v], t['aa'][pv,pv,hc,ha], optimize=True)
    O['a'][v,c] += scale * +1.00000000 * np.einsum('uv,uwix,wyxz,yavz->ai', gamma1['a'], h['aa'][a,a,c,a], lambdas['aa'], t['aa'][pa,pv,ha,ha], optimize=True)
    O['a'][v,c] += scale * -0.50000000 * np.einsum('uv,uwxy,wzxy,zaiv->ai', gamma1['a'], h['aa'][a,a,a,a], lambdas['aa'], t['aa'][pa,pv,hc,ha], optimize=True)
    
    O['a'][v,c] += scale * -0.25000000 * np.einsum('uv,uawx,yzwx,yziv->ai', gamma1['a'], h['aa'][a,v,a,a], lambdas['aa'], t['aa'][pa,pa,hc,ha], optimize=True)
    O['a'][v,c] += scale * -0.50000000 * np.einsum('uv,uabc,bciv->ai', gamma1['a'], h['aa'][a,v,v,v], t['aa'][pv,pv,hc,ha], optimize=True)
    O['a'][v,c] += scale * +0.25000000 * np.einsum('uv,wxiv,wxyz,uayz->ai', gamma1['a'], h['aa'][a,a,c,a], lambdas['aa'], t['aa'][pa,pv,ha,ha], optimize=True)
    O['a'][v,c] += scale * +0.50000000 * np.einsum('uv,wxvy,wxyz,uaiz->ai', gamma1['a'], h['aa'][a,a,a,a], lambdas['aa'], t['aa'][pa,pv,hc,ha], optimize=True)
    
    O['a'][v,c] += scale * -1.00000000 * np.einsum('uv,wavx,wyxz,uyiz->ai', gamma1['a'], h['aa'][a,v,a,a], lambdas['aa'], t['aa'][pa,pa,hc,ha], optimize=True)
    O['a'][v,c] += scale * -0.50000000 * np.einsum('ijka,baij->bk', h['aa'][c,c,c,v], t['aa'][pv,pv,hc,hc], optimize=True)
    O['a'][v,c] += scale * +1.00000000 * np.einsum('iujv,uwvx,waix->aj', h['aa'][c,a,c,a], lambdas['aa'], t['aa'][pa,pv,hc,ha], optimize=True)
    O['a'][v,c] += scale * -0.50000000 * np.einsum('iuvw,uxvw,xaji->aj', h['aa'][c,a,a,a], lambdas['aa'], t['aa'][pa,pv,hc,hc], optimize=True)
    O['a'][v,c] += scale * +0.50000000 * np.einsum('iaju,vwux,vwix->aj', h['aa'][c,v,c,a], lambdas['aa'], t['aa'][pa,pa,hc,ha], optimize=True)
    O['a'][v,c] += scale * -0.25000000 * np.einsum('iauv,wxuv,wxji->aj', h['aa'][c,v,a,a], lambdas['aa'], t['aa'][pa,pa,hc,hc], optimize=True)
    O['a'][v,c] += scale * -0.50000000 * np.einsum('iabc,bcji->aj', h['aa'][c,v,v,v], t['aa'][pv,pv,hc,hc], optimize=True)
    O['a'][v,c] += scale * -0.25000000 * np.einsum('uvia,uvwx,bawx->bi', h['aa'][a,a,c,v], lambdas['aa'], t['aa'][pv,pv,ha,ha], optimize=True)
    O['a'][v,c] += scale * +0.50000000 * np.einsum('uvwa,uvwx,baix->bi', h['aa'][a,a,a,v], lambdas['aa'], t['aa'][pv,pv,hc,ha], optimize=True)
    O['a'][v,c] += scale * +0.50000000 * np.einsum('uaib,uvwx,vbwx->ai', h['aa'][a,v,c,v], lambdas['aa'], t['aa'][pa,pv,ha,ha], optimize=True)
    O['a'][v,c] += scale * -1.00000000 * np.einsum('uavb,uwvx,wbix->ai', h['aa'][a,v,a,v], lambdas['aa'], t['aa'][pa,pv,hc,ha], optimize=True)
    
    
    O['a'][a,v] += scale * +1.00000000 * np.einsum('uv,wx,iwva,yuix->ya', eta1['a'], gamma1['a'], h['aa'][c,a,a,v], t['aa'][pa,pa,hc,ha], optimize=True)
    O['a'][a,v] += scale * +0.50000000 * np.einsum('uv,ijva,wuij->wa', eta1['a'], h['aa'][c,c,a,v], t['aa'][pa,pa,hc,hc], optimize=True)
    
    
    
    
    O['a'][a,v] += scale * -0.50000000 * np.einsum('uv,wx,uwab,ybvx->ya', gamma1['a'], gamma1['a'], h['aa'][a,a,v,v], t['aa'][pa,pv,ha,ha], optimize=True)
    O['a'][a,v] += scale * -1.00000000 * np.einsum('uv,iuab,wbiv->wa', gamma1['a'], h['aa'][c,a,v,v], t['aa'][pa,pv,hc,ha], optimize=True)
    
    
    
    
    O['a'][a,v] += scale * -0.50000000 * np.einsum('ijab,ubij->ua', h['aa'][c,c,v,v], t['aa'][pa,pv,hc,hc], optimize=True)
    O['a'][a,v] += scale * -0.50000000 * np.einsum('iuva,wxvy,wxiy->ua', h['aa'][c,a,a,v], lambdas['aa'], t['aa'][pa,pa,hc,ha], optimize=True)
    O['a'][a,v] += scale * +1.00000000 * np.einsum('iuva,uwvx,ywix->ya', h['aa'][c,a,a,v], lambdas['aa'], t['aa'][pa,pa,hc,ha], optimize=True)
    O['a'][a,v] += scale * -0.50000000 * np.einsum('uvab,vwxy,wbxy->ua', h['aa'][a,a,v,v], lambdas['aa'], t['aa'][pa,pv,ha,ha], optimize=True)
    O['a'][a,v] += scale * -0.25000000 * np.einsum('uvab,uvwx,ybwx->ya', h['aa'][a,a,v,v], lambdas['aa'], t['aa'][pa,pv,ha,ha], optimize=True)
    
    
    
    
    O['a'][c,v] += scale * +0.50000000 * np.einsum('ijua,vwux,vwjx->ia', h['aa'][c,c,a,v], lambdas['aa'], t['aa'][pa,pa,hc,ha], optimize=True)
    O['a'][c,v] += scale * -0.50000000 * np.einsum('iuab,uvwx,vbwx->ia', h['aa'][c,a,v,v], lambdas['aa'], t['aa'][pa,pv,ha,ha], optimize=True)
    O['a'][v,v] += scale * -0.50000000 * np.einsum('uv,wx,yz,uwza,ybvx->ba', eta1['a'], eta1['a'], gamma1['a'], h['aa'][a,a,a,v], t['aa'][pa,pv,ha,ha], optimize=True)
    O['a'][v,v] += scale * -0.50000000 * np.einsum('uv,wx,yz,wyva,ubxz->ba', eta1['a'], gamma1['a'], gamma1['a'], h['aa'][a,a,a,v], t['aa'][pa,pv,ha,ha], optimize=True)
    O['a'][v,v] += scale * -1.00000000 * np.einsum('uv,wx,iwva,ubix->ba', eta1['a'], gamma1['a'], h['aa'][c,a,a,v], t['aa'][pa,pv,hc,ha], optimize=True)
    O['a'][v,v] += scale * -0.50000000 * np.einsum('uv,ijva,ubij->ba', eta1['a'], h['aa'][c,c,a,v], t['aa'][pa,pv,hc,hc], optimize=True)
    O['a'][v,v] += scale * -1.00000000 * np.einsum('uv,uwxa,wyxz,ybvz->ba', eta1['a'], h['aa'][a,a,a,v], lambdas['aa'], t['aa'][pa,pv,ha,ha], optimize=True)
    
    O['a'][v,v] += scale * -0.25000000 * np.einsum('uv,wxva,wxyz,ubyz->ba', eta1['a'], h['aa'][a,a,a,v], lambdas['aa'], t['aa'][pa,pv,ha,ha], optimize=True)
    
    O['a'][v,v] += scale * -0.50000000 * np.einsum('uv,wx,uwab,cbvx->ca', gamma1['a'], gamma1['a'], h['aa'][a,a,v,v], t['aa'][pv,pv,ha,ha], optimize=True)
    O['a'][v,v] += scale * -1.00000000 * np.einsum('uv,iuab,cbiv->ca', gamma1['a'], h['aa'][c,a,v,v], t['aa'][pv,pv,hc,ha], optimize=True)
    O['a'][v,v] += scale * -1.00000000 * np.einsum('uv,uwxa,wyxz,ybvz->ba', gamma1['a'], h['aa'][a,a,a,v], lambdas['aa'], t['aa'][pa,pv,ha,ha], optimize=True)
    
    O['a'][v,v] += scale * -0.25000000 * np.einsum('uv,wxva,wxyz,ubyz->ba', gamma1['a'], h['aa'][a,a,a,v], lambdas['aa'], t['aa'][pa,pv,ha,ha], optimize=True)
    
    O['a'][v,v] += scale * -0.50000000 * np.einsum('ijab,cbij->ca', h['aa'][c,c,v,v], t['aa'][pv,pv,hc,hc], optimize=True)
    O['a'][v,v] += scale * -1.00000000 * np.einsum('iuva,uwvx,wbix->ba', h['aa'][c,a,a,v], lambdas['aa'], t['aa'][pa,pv,hc,ha], optimize=True)
    O['a'][v,v] += scale * -0.50000000 * np.einsum('iaub,vwux,vwix->ab', h['aa'][c,v,a,v], lambdas['aa'], t['aa'][pa,pa,hc,ha], optimize=True)
    O['a'][v,v] += scale * -0.25000000 * np.einsum('uvab,uvwx,cbwx->ca', h['aa'][a,a,v,v], lambdas['aa'], t['aa'][pv,pv,ha,ha], optimize=True)
    O['a'][v,v] += scale * +0.50000000 * np.einsum('uabc,uvwx,vcwx->ab', h['aa'][a,v,v,v], lambdas['aa'], t['aa'][pa,pv,ha,ha], optimize=True)

    t1 = time.time()
    if verbose: print("h2a_t2a_c1a took {:.4f} seconds to run.".format(t1-t0))

    return O
def h2a_t2b_c1a(O, h, t, gamma1, eta1, lambdas, orbspace, verbose=False, scale=1.0):
    # 90 lines
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

    
    
    
    
    
    
    
    
    O['a'][a,a] += scale * +1.00000000 * np.einsum('iuvw,xUwV,xUiV->uv', h['aa'][c,a,a,a], lambdas['ab'], t['ab'][pa,pA,hc,hA], optimize=True)
    O['a'][a,a] += scale * -1.00000000 * np.einsum('iuvw,uUwV,xUiV->xv', h['aa'][c,a,a,a], lambdas['ab'], t['ab'][pa,pA,hc,hA], optimize=True)
    O['a'][a,a] += scale * +1.00000000 * np.einsum('uvwa,vUxV,aUxV->uw', h['aa'][a,a,a,v], lambdas['ab'], t['ab'][pv,pA,ha,hA], optimize=True)
    O['a'][a,a] += scale * -1.00000000 * np.einsum('uvwa,vUwV,aUxV->ux', h['aa'][a,a,a,v], lambdas['ab'], t['ab'][pv,pA,ha,hA], optimize=True)
    
    
    
    
    
    
    O['a'][c,a] += scale * -1.00000000 * np.einsum('ijuv,wUvV,wUjV->iu', h['aa'][c,c,a,a], lambdas['ab'], t['ab'][pa,pA,hc,hA], optimize=True)
    O['a'][c,a] += scale * +1.00000000 * np.einsum('iuva,uUwV,aUwV->iv', h['aa'][c,a,a,v], lambdas['ab'], t['ab'][pv,pA,ha,hA], optimize=True)
    O['a'][c,a] += scale * -1.00000000 * np.einsum('iuva,uUvV,aUwV->iw', h['aa'][c,a,a,v], lambdas['ab'], t['ab'][pv,pA,ha,hA], optimize=True)
    O['a'][v,a] += scale * -1.00000000 * np.einsum('uv,uwxy,wUyV,aUvV->ax', eta1['a'], h['aa'][a,a,a,a], lambdas['ab'], t['ab'][pv,pA,ha,hA], optimize=True)
    
    
    
    O['a'][v,a] += scale * -1.00000000 * np.einsum('uv,uwxy,wUyV,aUvV->ax', gamma1['a'], h['aa'][a,a,a,a], lambdas['ab'], t['ab'][pv,pA,ha,hA], optimize=True)
    
    
    
    O['a'][v,a] += scale * -1.00000000 * np.einsum('iuvw,uUwV,aUiV->av', h['aa'][c,a,a,a], lambdas['ab'], t['ab'][pv,pA,hc,hA], optimize=True)
    O['a'][v,a] += scale * +1.00000000 * np.einsum('iauv,wUvV,wUiV->au', h['aa'][c,v,a,a], lambdas['ab'], t['ab'][pa,pA,hc,hA], optimize=True)
    O['a'][v,a] += scale * -1.00000000 * np.einsum('uavb,uUwV,bUwV->av', h['aa'][a,v,a,v], lambdas['ab'], t['ab'][pv,pA,ha,hA], optimize=True)
    O['a'][v,a] += scale * +1.00000000 * np.einsum('uavb,uUvV,bUwV->aw', h['aa'][a,v,a,v], lambdas['ab'], t['ab'][pv,pA,ha,hA], optimize=True)
    
    
    O['a'][a,c] += scale * +1.00000000 * np.einsum('uv,wxvy,xUyV,uUiV->wi', eta1['a'], h['aa'][a,a,a,a], lambdas['ab'], t['ab'][pa,pA,hc,hA], optimize=True)
    
    
    
    O['a'][a,c] += scale * +1.00000000 * np.einsum('uv,wxvy,xUyV,uUiV->wi', gamma1['a'], h['aa'][a,a,a,a], lambdas['ab'], t['ab'][pa,pA,hc,hA], optimize=True)
    
    O['a'][a,c] += scale * +1.00000000 * np.einsum('iujv,wUvV,wUiV->uj', h['aa'][c,a,c,a], lambdas['ab'], t['ab'][pa,pA,hc,hA], optimize=True)
    O['a'][a,c] += scale * -1.00000000 * np.einsum('iujv,uUvV,wUiV->wj', h['aa'][c,a,c,a], lambdas['ab'], t['ab'][pa,pA,hc,hA], optimize=True)
    O['a'][a,c] += scale * +1.00000000 * np.einsum('uvia,vUwV,aUwV->ui', h['aa'][a,a,c,v], lambdas['ab'], t['ab'][pv,pA,ha,hA], optimize=True)
    O['a'][a,c] += scale * -1.00000000 * np.einsum('uvwa,vUwV,aUiV->ui', h['aa'][a,a,a,v], lambdas['ab'], t['ab'][pv,pA,hc,hA], optimize=True)
    
    
    O['a'][c,c] += scale * +1.00000000 * np.einsum('uv,iwvx,wUxV,uUjV->ij', eta1['a'], h['aa'][c,a,a,a], lambdas['ab'], t['ab'][pa,pA,hc,hA], optimize=True)
    
    
    O['a'][c,c] += scale * +1.00000000 * np.einsum('uv,iwvx,wUxV,uUjV->ij', gamma1['a'], h['aa'][c,a,a,a], lambdas['ab'], t['ab'][pa,pA,hc,hA], optimize=True)
    O['a'][c,c] += scale * -1.00000000 * np.einsum('ijku,vUuV,vUjV->ik', h['aa'][c,c,c,a], lambdas['ab'], t['ab'][pa,pA,hc,hA], optimize=True)
    O['a'][c,c] += scale * +1.00000000 * np.einsum('iuja,uUvV,aUvV->ij', h['aa'][c,a,c,v], lambdas['ab'], t['ab'][pv,pA,ha,hA], optimize=True)
    O['a'][c,c] += scale * -1.00000000 * np.einsum('iuva,uUvV,aUjV->ij', h['aa'][c,a,a,v], lambdas['ab'], t['ab'][pv,pA,hc,hA], optimize=True)
    O['a'][v,c] += scale * -1.00000000 * np.einsum('uv,uwix,wUxV,aUvV->ai', eta1['a'], h['aa'][a,a,c,a], lambdas['ab'], t['ab'][pv,pA,ha,hA], optimize=True)
    
    
    O['a'][v,c] += scale * -1.00000000 * np.einsum('uv,wavx,wUxV,uUiV->ai', eta1['a'], h['aa'][a,v,a,a], lambdas['ab'], t['ab'][pa,pA,hc,hA], optimize=True)
    O['a'][v,c] += scale * -1.00000000 * np.einsum('uv,uwix,wUxV,aUvV->ai', gamma1['a'], h['aa'][a,a,c,a], lambdas['ab'], t['ab'][pv,pA,ha,hA], optimize=True)
    
    
    O['a'][v,c] += scale * -1.00000000 * np.einsum('uv,wavx,wUxV,uUiV->ai', gamma1['a'], h['aa'][a,v,a,a], lambdas['ab'], t['ab'][pa,pA,hc,hA], optimize=True)
    O['a'][v,c] += scale * -1.00000000 * np.einsum('iujv,uUvV,aUiV->aj', h['aa'][c,a,c,a], lambdas['ab'], t['ab'][pv,pA,hc,hA], optimize=True)
    O['a'][v,c] += scale * +1.00000000 * np.einsum('iaju,vUuV,vUiV->aj', h['aa'][c,v,c,a], lambdas['ab'], t['ab'][pa,pA,hc,hA], optimize=True)
    O['a'][v,c] += scale * -1.00000000 * np.einsum('uaib,uUvV,bUvV->ai', h['aa'][a,v,c,v], lambdas['ab'], t['ab'][pv,pA,ha,hA], optimize=True)
    O['a'][v,c] += scale * +1.00000000 * np.einsum('uavb,uUvV,bUiV->ai', h['aa'][a,v,a,v], lambdas['ab'], t['ab'][pv,pA,hc,hA], optimize=True)
    
    
    
    
    
    
    O['a'][a,v] += scale * -1.00000000 * np.einsum('iuva,wUvV,wUiV->ua', h['aa'][c,a,a,v], lambdas['ab'], t['ab'][pa,pA,hc,hA], optimize=True)
    O['a'][a,v] += scale * +1.00000000 * np.einsum('iuva,uUvV,wUiV->wa', h['aa'][c,a,a,v], lambdas['ab'], t['ab'][pa,pA,hc,hA], optimize=True)
    O['a'][a,v] += scale * +1.00000000 * np.einsum('uvab,vUwV,bUwV->ua', h['aa'][a,a,v,v], lambdas['ab'], t['ab'][pv,pA,ha,hA], optimize=True)
    
    
    
    
    O['a'][c,v] += scale * +1.00000000 * np.einsum('ijua,vUuV,vUjV->ia', h['aa'][c,c,a,v], lambdas['ab'], t['ab'][pa,pA,hc,hA], optimize=True)
    O['a'][c,v] += scale * +1.00000000 * np.einsum('iuab,uUvV,bUvV->ia', h['aa'][c,a,v,v], lambdas['ab'], t['ab'][pv,pA,ha,hA], optimize=True)
    O['a'][v,v] += scale * +1.00000000 * np.einsum('uv,uwxa,wUxV,bUvV->ba', eta1['a'], h['aa'][a,a,a,v], lambdas['ab'], t['ab'][pv,pA,ha,hA], optimize=True)
    
    
    O['a'][v,v] += scale * +1.00000000 * np.einsum('uv,uwxa,wUxV,bUvV->ba', gamma1['a'], h['aa'][a,a,a,v], lambdas['ab'], t['ab'][pv,pA,ha,hA], optimize=True)
    
    
    O['a'][v,v] += scale * +1.00000000 * np.einsum('iuva,uUvV,bUiV->ba', h['aa'][c,a,a,v], lambdas['ab'], t['ab'][pv,pA,hc,hA], optimize=True)
    O['a'][v,v] += scale * -1.00000000 * np.einsum('iaub,vUuV,vUiV->ab', h['aa'][c,v,a,v], lambdas['ab'], t['ab'][pa,pA,hc,hA], optimize=True)
    O['a'][v,v] += scale * -1.00000000 * np.einsum('uabc,uUvV,cUvV->ab', h['aa'][a,v,v,v], lambdas['ab'], t['ab'][pv,pA,ha,hA], optimize=True)

    t1 = time.time()
    if verbose: print("h2a_t2b_c1a took {:.4f} seconds to run.".format(t1-t0))

    return O
def h2b_t1b_c1a(O, h, t, gamma1, eta1, lambdas, orbspace, verbose=False, scale=1.0):
    # 45 lines
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

    
    
    O['a'][a,a] += scale * +1.00000000 * np.einsum('UV,uIvV,UI->uv', eta1['b'], h['ab'][a,C,a,A], t['b'][pA,hC], optimize=True)
    O['a'][a,a] += scale * +1.00000000 * np.einsum('UV,uUvA,AV->uv', gamma1['b'], h['ab'][a,A,a,V], t['b'][pV,hA], optimize=True)
    O['a'][a,a] += scale * +1.00000000 * np.einsum('uIvA,AI->uv', h['ab'][a,C,a,V], t['b'][pV,hC], optimize=True)
    
    
    O['a'][c,a] += scale * +1.00000000 * np.einsum('UV,iIuV,UI->iu', eta1['b'], h['ab'][c,C,a,A], t['b'][pA,hC], optimize=True)
    O['a'][c,a] += scale * +1.00000000 * np.einsum('UV,iUuA,AV->iu', gamma1['b'], h['ab'][c,A,a,V], t['b'][pV,hA], optimize=True)
    O['a'][c,a] += scale * +1.00000000 * np.einsum('iIuA,AI->iu', h['ab'][c,C,a,V], t['b'][pV,hC], optimize=True)
    
    
    O['a'][v,a] += scale * +1.00000000 * np.einsum('UV,aIuV,UI->au', eta1['b'], h['ab'][v,C,a,A], t['b'][pA,hC], optimize=True)
    O['a'][v,a] += scale * +1.00000000 * np.einsum('UV,aUuA,AV->au', gamma1['b'], h['ab'][v,A,a,V], t['b'][pV,hA], optimize=True)
    O['a'][v,a] += scale * +1.00000000 * np.einsum('aIuA,AI->au', h['ab'][v,C,a,V], t['b'][pV,hC], optimize=True)
    
    
    O['a'][a,c] += scale * +1.00000000 * np.einsum('UV,uIiV,UI->ui', eta1['b'], h['ab'][a,C,c,A], t['b'][pA,hC], optimize=True)
    O['a'][a,c] += scale * +1.00000000 * np.einsum('UV,uUiA,AV->ui', gamma1['b'], h['ab'][a,A,c,V], t['b'][pV,hA], optimize=True)
    O['a'][a,c] += scale * +1.00000000 * np.einsum('uIiA,AI->ui', h['ab'][a,C,c,V], t['b'][pV,hC], optimize=True)
    
    
    O['a'][c,c] += scale * +1.00000000 * np.einsum('UV,iIjV,UI->ij', eta1['b'], h['ab'][c,C,c,A], t['b'][pA,hC], optimize=True)
    O['a'][c,c] += scale * +1.00000000 * np.einsum('UV,iUjA,AV->ij', gamma1['b'], h['ab'][c,A,c,V], t['b'][pV,hA], optimize=True)
    O['a'][c,c] += scale * +1.00000000 * np.einsum('iIjA,AI->ij', h['ab'][c,C,c,V], t['b'][pV,hC], optimize=True)
    
    
    O['a'][v,c] += scale * +1.00000000 * np.einsum('UV,aIiV,UI->ai', eta1['b'], h['ab'][v,C,c,A], t['b'][pA,hC], optimize=True)
    O['a'][v,c] += scale * +1.00000000 * np.einsum('UV,aUiA,AV->ai', gamma1['b'], h['ab'][v,A,c,V], t['b'][pV,hA], optimize=True)
    O['a'][v,c] += scale * +1.00000000 * np.einsum('aIiA,AI->ai', h['ab'][v,C,c,V], t['b'][pV,hC], optimize=True)
    
    
    O['a'][a,v] += scale * +1.00000000 * np.einsum('UV,uIaV,UI->ua', eta1['b'], h['ab'][a,C,v,A], t['b'][pA,hC], optimize=True)
    O['a'][a,v] += scale * +1.00000000 * np.einsum('UV,uUaA,AV->ua', gamma1['b'], h['ab'][a,A,v,V], t['b'][pV,hA], optimize=True)
    O['a'][a,v] += scale * +1.00000000 * np.einsum('uIaA,AI->ua', h['ab'][a,C,v,V], t['b'][pV,hC], optimize=True)
    
    
    O['a'][c,v] += scale * +1.00000000 * np.einsum('UV,iIaV,UI->ia', eta1['b'], h['ab'][c,C,v,A], t['b'][pA,hC], optimize=True)
    O['a'][c,v] += scale * +1.00000000 * np.einsum('UV,iUaA,AV->ia', gamma1['b'], h['ab'][c,A,v,V], t['b'][pV,hA], optimize=True)
    O['a'][c,v] += scale * +1.00000000 * np.einsum('iIaA,AI->ia', h['ab'][c,C,v,V], t['b'][pV,hC], optimize=True)
    
    
    O['a'][v,v] += scale * +1.00000000 * np.einsum('UV,aIbV,UI->ab', eta1['b'], h['ab'][v,C,v,A], t['b'][pA,hC], optimize=True)
    O['a'][v,v] += scale * +1.00000000 * np.einsum('UV,aUbA,AV->ab', gamma1['b'], h['ab'][v,A,v,V], t['b'][pV,hA], optimize=True)
    O['a'][v,v] += scale * +1.00000000 * np.einsum('aIbA,AI->ab', h['ab'][v,C,v,V], t['b'][pV,hC], optimize=True)

    t1 = time.time()
    if verbose: print("h2b_t1b_c1a took {:.4f} seconds to run.".format(t1-t0))

    return O
def h2b_t2a_c1a(O, h, t, gamma1, eta1, lambdas, orbspace, verbose=False, scale=1.0):
    # 60 lines
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

    
    
    
    
    
    
    
    
    O['a'][a,a] += scale * -1.00000000 * np.einsum('iUuV,vUwV,xviw->xu', h['ab'][c,A,a,A], lambdas['ab'], t['aa'][pa,pa,hc,ha], optimize=True)
    O['a'][a,a] += scale * +1.00000000 * np.einsum('iUuV,vUuV,wvix->wx', h['ab'][c,A,a,A], lambdas['ab'], t['aa'][pa,pa,hc,ha], optimize=True)
    O['a'][a,a] += scale * -1.00000000 * np.einsum('uUaV,vUwV,vaxw->ux', h['ab'][a,A,v,A], lambdas['ab'], t['aa'][pa,pv,ha,ha], optimize=True)
    O['a'][a,a] += scale * +1.00000000 * np.einsum('uUaV,uUvV,waxv->wx', h['ab'][a,A,v,A], lambdas['ab'], t['aa'][pa,pv,ha,ha], optimize=True)
    
    
    O['a'][c,a] += scale * -1.00000000 * np.einsum('iUaV,uUvV,uawv->iw', h['ab'][c,A,v,A], lambdas['ab'], t['aa'][pa,pv,ha,ha], optimize=True)
    O['a'][v,a] += scale * +1.00000000 * np.einsum('uv,uUwV,xUyV,xavy->aw', eta1['a'], h['ab'][a,A,a,A], lambdas['ab'], t['aa'][pa,pv,ha,ha], optimize=True)
    O['a'][v,a] += scale * +1.00000000 * np.einsum('uv,uUwV,xUwV,xayv->ay', eta1['a'], h['ab'][a,A,a,A], lambdas['ab'], t['aa'][pa,pv,ha,ha], optimize=True)
    O['a'][v,a] += scale * -1.00000000 * np.einsum('uv,wUvV,wUxV,uayx->ay', eta1['a'], h['ab'][a,A,a,A], lambdas['ab'], t['aa'][pa,pv,ha,ha], optimize=True)
    
    O['a'][v,a] += scale * +1.00000000 * np.einsum('uv,uUwV,xUyV,xavy->aw', gamma1['a'], h['ab'][a,A,a,A], lambdas['ab'], t['aa'][pa,pv,ha,ha], optimize=True)
    O['a'][v,a] += scale * +1.00000000 * np.einsum('uv,uUwV,xUwV,xayv->ay', gamma1['a'], h['ab'][a,A,a,A], lambdas['ab'], t['aa'][pa,pv,ha,ha], optimize=True)
    O['a'][v,a] += scale * -1.00000000 * np.einsum('uv,wUvV,wUxV,uayx->ay', gamma1['a'], h['ab'][a,A,a,A], lambdas['ab'], t['aa'][pa,pv,ha,ha], optimize=True)
    
    O['a'][v,a] += scale * +1.00000000 * np.einsum('iUuV,vUwV,vaiw->au', h['ab'][c,A,a,A], lambdas['ab'], t['aa'][pa,pv,hc,ha], optimize=True)
    O['a'][v,a] += scale * -1.00000000 * np.einsum('iUuV,vUuV,vaiw->aw', h['ab'][c,A,a,A], lambdas['ab'], t['aa'][pa,pv,hc,ha], optimize=True)
    O['a'][v,a] += scale * +1.00000000 * np.einsum('uUaV,uUvV,bawv->bw', h['ab'][a,A,v,A], lambdas['ab'], t['aa'][pv,pv,ha,ha], optimize=True)
    O['a'][v,a] += scale * -1.00000000 * np.einsum('aUbV,uUvV,ubwv->aw', h['ab'][v,A,v,A], lambdas['ab'], t['aa'][pa,pv,ha,ha], optimize=True)
    O['a'][a,c] += scale * +1.00000000 * np.einsum('uv,wUvV,xUyV,uxiy->wi', eta1['a'], h['ab'][a,A,a,A], lambdas['ab'], t['aa'][pa,pa,hc,ha], optimize=True)
    
    O['a'][a,c] += scale * -1.00000000 * np.einsum('uv,uUwV,xUwV,yxiv->yi', eta1['a'], h['ab'][a,A,a,A], lambdas['ab'], t['aa'][pa,pa,hc,ha], optimize=True)
    O['a'][a,c] += scale * +1.00000000 * np.einsum('uv,wUvV,wUxV,yuix->yi', eta1['a'], h['ab'][a,A,a,A], lambdas['ab'], t['aa'][pa,pa,hc,ha], optimize=True)
    O['a'][a,c] += scale * +1.00000000 * np.einsum('uv,wUvV,xUyV,uxiy->wi', gamma1['a'], h['ab'][a,A,a,A], lambdas['ab'], t['aa'][pa,pa,hc,ha], optimize=True)
    
    O['a'][a,c] += scale * -1.00000000 * np.einsum('uv,uUwV,xUwV,yxiv->yi', gamma1['a'], h['ab'][a,A,a,A], lambdas['ab'], t['aa'][pa,pa,hc,ha], optimize=True)
    O['a'][a,c] += scale * +1.00000000 * np.einsum('uv,wUvV,wUxV,yuix->yi', gamma1['a'], h['ab'][a,A,a,A], lambdas['ab'], t['aa'][pa,pa,hc,ha], optimize=True)
    O['a'][a,c] += scale * -1.00000000 * np.einsum('iUjV,uUvV,wuiv->wj', h['ab'][c,A,c,A], lambdas['ab'], t['aa'][pa,pa,hc,ha], optimize=True)
    O['a'][a,c] += scale * -1.00000000 * np.einsum('iUuV,vUuV,wvji->wj', h['ab'][c,A,a,A], lambdas['ab'], t['aa'][pa,pa,hc,hc], optimize=True)
    O['a'][a,c] += scale * -1.00000000 * np.einsum('uUaV,vUwV,vaiw->ui', h['ab'][a,A,v,A], lambdas['ab'], t['aa'][pa,pv,hc,ha], optimize=True)
    O['a'][a,c] += scale * +1.00000000 * np.einsum('uUaV,uUvV,waiv->wi', h['ab'][a,A,v,A], lambdas['ab'], t['aa'][pa,pv,hc,ha], optimize=True)
    O['a'][c,c] += scale * +1.00000000 * np.einsum('uv,iUvV,wUxV,uwjx->ij', eta1['a'], h['ab'][c,A,a,A], lambdas['ab'], t['aa'][pa,pa,hc,ha], optimize=True)
    O['a'][c,c] += scale * +1.00000000 * np.einsum('uv,iUvV,wUxV,uwjx->ij', gamma1['a'], h['ab'][c,A,a,A], lambdas['ab'], t['aa'][pa,pa,hc,ha], optimize=True)
    O['a'][c,c] += scale * -1.00000000 * np.einsum('iUaV,uUvV,uajv->ij', h['ab'][c,A,v,A], lambdas['ab'], t['aa'][pa,pv,hc,ha], optimize=True)
    O['a'][v,c] += scale * +1.00000000 * np.einsum('uv,uUiV,wUxV,wavx->ai', eta1['a'], h['ab'][a,A,c,A], lambdas['ab'], t['aa'][pa,pv,ha,ha], optimize=True)
    O['a'][v,c] += scale * +1.00000000 * np.einsum('uv,uUwV,xUwV,xaiv->ai', eta1['a'], h['ab'][a,A,a,A], lambdas['ab'], t['aa'][pa,pv,hc,ha], optimize=True)
    O['a'][v,c] += scale * -1.00000000 * np.einsum('uv,wUvV,wUxV,uaix->ai', eta1['a'], h['ab'][a,A,a,A], lambdas['ab'], t['aa'][pa,pv,hc,ha], optimize=True)
    O['a'][v,c] += scale * +1.00000000 * np.einsum('uv,aUvV,wUxV,uwix->ai', eta1['a'], h['ab'][v,A,a,A], lambdas['ab'], t['aa'][pa,pa,hc,ha], optimize=True)
    O['a'][v,c] += scale * +1.00000000 * np.einsum('uv,uUiV,wUxV,wavx->ai', gamma1['a'], h['ab'][a,A,c,A], lambdas['ab'], t['aa'][pa,pv,ha,ha], optimize=True)
    O['a'][v,c] += scale * +1.00000000 * np.einsum('uv,uUwV,xUwV,xaiv->ai', gamma1['a'], h['ab'][a,A,a,A], lambdas['ab'], t['aa'][pa,pv,hc,ha], optimize=True)
    O['a'][v,c] += scale * -1.00000000 * np.einsum('uv,wUvV,wUxV,uaix->ai', gamma1['a'], h['ab'][a,A,a,A], lambdas['ab'], t['aa'][pa,pv,hc,ha], optimize=True)
    O['a'][v,c] += scale * +1.00000000 * np.einsum('uv,aUvV,wUxV,uwix->ai', gamma1['a'], h['ab'][v,A,a,A], lambdas['ab'], t['aa'][pa,pa,hc,ha], optimize=True)
    O['a'][v,c] += scale * +1.00000000 * np.einsum('iUjV,uUvV,uaiv->aj', h['ab'][c,A,c,A], lambdas['ab'], t['aa'][pa,pv,hc,ha], optimize=True)
    O['a'][v,c] += scale * +1.00000000 * np.einsum('iUuV,vUuV,vaji->aj', h['ab'][c,A,a,A], lambdas['ab'], t['aa'][pa,pv,hc,hc], optimize=True)
    O['a'][v,c] += scale * +1.00000000 * np.einsum('uUaV,uUvV,baiv->bi', h['ab'][a,A,v,A], lambdas['ab'], t['aa'][pv,pv,hc,ha], optimize=True)
    O['a'][v,c] += scale * -1.00000000 * np.einsum('aUbV,uUvV,ubiv->ai', h['ab'][v,A,v,A], lambdas['ab'], t['aa'][pa,pv,hc,ha], optimize=True)
    
    
    O['a'][a,v] += scale * -1.00000000 * np.einsum('iUaV,uUvV,wuiv->wa', h['ab'][c,A,v,A], lambdas['ab'], t['aa'][pa,pa,hc,ha], optimize=True)
    O['a'][v,v] += scale * +1.00000000 * np.einsum('uv,uUaV,wUxV,wbvx->ba', eta1['a'], h['ab'][a,A,v,A], lambdas['ab'], t['aa'][pa,pv,ha,ha], optimize=True)
    O['a'][v,v] += scale * +1.00000000 * np.einsum('uv,uUaV,wUxV,wbvx->ba', gamma1['a'], h['ab'][a,A,v,A], lambdas['ab'], t['aa'][pa,pv,ha,ha], optimize=True)
    O['a'][v,v] += scale * +1.00000000 * np.einsum('iUaV,uUvV,ubiv->ba', h['ab'][c,A,v,A], lambdas['ab'], t['aa'][pa,pv,hc,ha], optimize=True)

    t1 = time.time()
    if verbose: print("h2b_t2a_c1a took {:.4f} seconds to run.".format(t1-t0))

    return O
def h2b_t2b_c1a(O, h, t, gamma1, eta1, lambdas, orbspace, verbose=False, scale=1.0):
    # 294 lines
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

    O['a'][a,a] += scale * +1.00000000 * np.einsum('uv,UV,wUvA,uAxV->wx', eta1['a'], gamma1['b'], h['ab'][a,A,a,V], t['ab'][pa,pV,ha,hA], optimize=True)
    O['a'][a,a] += scale * +1.00000000 * np.einsum('uv,wIvA,uAxI->wx', eta1['a'], h['ab'][a,C,a,V], t['ab'][pa,pV,ha,hC], optimize=True)
    
    
    
    
    O['a'][a,a] += scale * +1.00000000 * np.einsum('UV,uv,wIvV,uUxI->wx', eta1['b'], eta1['a'], h['ab'][a,C,a,A], t['ab'][pa,pA,ha,hC], optimize=True)
    O['a'][a,a] += scale * -1.00000000 * np.einsum('UV,uv,uIwV,xUvI->xw', eta1['b'], gamma1['a'], h['ab'][a,C,a,A], t['ab'][pa,pA,ha,hC], optimize=True)
    
    
    O['a'][a,a] += scale * -1.00000000 * np.einsum('UV,WX,iWuV,vUiX->vu', eta1['b'], gamma1['b'], h['ab'][c,A,a,A], t['ab'][pa,pA,hc,hA], optimize=True)
    O['a'][a,a] += scale * +1.00000000 * np.einsum('UV,WX,uWaV,aUvX->uv', eta1['b'], gamma1['b'], h['ab'][a,A,v,A], t['ab'][pv,pA,ha,hA], optimize=True)
    O['a'][a,a] += scale * -1.00000000 * np.einsum('UV,iIuV,vUiI->vu', eta1['b'], h['ab'][c,C,a,A], t['ab'][pa,pA,hc,hC], optimize=True)
    O['a'][a,a] += scale * +1.00000000 * np.einsum('UV,uIaV,aUvI->uv', eta1['b'], h['ab'][a,C,v,A], t['ab'][pv,pA,ha,hC], optimize=True)
    
    
    
    
    
    
    
    
    
    O['a'][a,a] += scale * -1.00000000 * np.einsum('uv,uIwA,xAvI->xw', gamma1['a'], h['ab'][a,C,a,V], t['ab'][pa,pV,ha,hC], optimize=True)
    
    O['a'][a,a] += scale * -1.00000000 * np.einsum('UV,uv,uUwA,xAvV->xw', gamma1['b'], gamma1['a'], h['ab'][a,A,a,V], t['ab'][pa,pV,ha,hA], optimize=True)
    O['a'][a,a] += scale * -1.00000000 * np.einsum('UV,iUuA,vAiV->vu', gamma1['b'], h['ab'][c,A,a,V], t['ab'][pa,pV,hc,hA], optimize=True)
    
    
    O['a'][a,a] += scale * +1.00000000 * np.einsum('UV,uUaA,aAvV->uv', gamma1['b'], h['ab'][a,A,v,V], t['ab'][pv,pV,ha,hA], optimize=True)
    
    
    
    
    
    
    O['a'][a,a] += scale * -1.00000000 * np.einsum('iIuA,vAiI->vu', h['ab'][c,C,a,V], t['ab'][pa,pV,hc,hC], optimize=True)
    O['a'][a,a] += scale * -1.00000000 * np.einsum('iUuV,UWVX,vWiX->vu', h['ab'][c,A,a,A], lambdas['bb'], t['ab'][pa,pA,hc,hA], optimize=True)
    O['a'][a,a] += scale * -1.00000000 * np.einsum('uIvU,wVxU,wVxI->uv', h['ab'][a,C,a,A], lambdas['ab'], t['ab'][pa,pA,ha,hC], optimize=True)
    O['a'][a,a] += scale * +1.00000000 * np.einsum('uIvU,wVvU,wVxI->ux', h['ab'][a,C,a,A], lambdas['ab'], t['ab'][pa,pA,ha,hC], optimize=True)
    O['a'][a,a] += scale * +1.00000000 * np.einsum('uIaA,aAvI->uv', h['ab'][a,C,v,V], t['ab'][pv,pV,ha,hC], optimize=True)
    O['a'][a,a] += scale * +1.00000000 * np.einsum('uUvA,wUxV,wAxV->uv', h['ab'][a,A,a,V], lambdas['ab'], t['ab'][pa,pV,ha,hA], optimize=True)
    O['a'][a,a] += scale * -1.00000000 * np.einsum('uUvA,wUvV,wAxV->ux', h['ab'][a,A,a,V], lambdas['ab'], t['ab'][pa,pV,ha,hA], optimize=True)
    O['a'][a,a] += scale * +1.00000000 * np.einsum('uUaV,UWVX,aWvX->uv', h['ab'][a,A,v,A], lambdas['bb'], t['ab'][pv,pA,ha,hA], optimize=True)
    O['a'][a,a] += scale * +1.00000000 * np.einsum('uIvU,uVwU,xVwI->xv', h['ab'][a,C,a,A], lambdas['ab'], t['ab'][pa,pA,ha,hC], optimize=True)
    O['a'][a,a] += scale * -1.00000000 * np.einsum('uIvU,uVvU,wVxI->wx', h['ab'][a,C,a,A], lambdas['ab'], t['ab'][pa,pA,ha,hC], optimize=True)
    O['a'][a,a] += scale * -1.00000000 * np.einsum('uUvA,uUwV,xAwV->xv', h['ab'][a,A,a,V], lambdas['ab'], t['ab'][pa,pV,ha,hA], optimize=True)
    O['a'][a,a] += scale * +1.00000000 * np.einsum('uUvA,uUvV,wAxV->wx', h['ab'][a,A,a,V], lambdas['ab'], t['ab'][pa,pV,ha,hA], optimize=True)
    O['a'][c,a] += scale * +1.00000000 * np.einsum('uv,UV,iUvA,uAwV->iw', eta1['a'], gamma1['b'], h['ab'][c,A,a,V], t['ab'][pa,pV,ha,hA], optimize=True)
    O['a'][c,a] += scale * +1.00000000 * np.einsum('uv,iIvA,uAwI->iw', eta1['a'], h['ab'][c,C,a,V], t['ab'][pa,pV,ha,hC], optimize=True)
    
    
    O['a'][c,a] += scale * +1.00000000 * np.einsum('UV,uv,iIvV,uUwI->iw', eta1['b'], eta1['a'], h['ab'][c,C,a,A], t['ab'][pa,pA,ha,hC], optimize=True)
    
    O['a'][c,a] += scale * +1.00000000 * np.einsum('UV,WX,iWaV,aUuX->iu', eta1['b'], gamma1['b'], h['ab'][c,A,v,A], t['ab'][pv,pA,ha,hA], optimize=True)
    O['a'][c,a] += scale * +1.00000000 * np.einsum('UV,iIaV,aUuI->iu', eta1['b'], h['ab'][c,C,v,A], t['ab'][pv,pA,ha,hC], optimize=True)
    
    
    
    
    
    
    
    O['a'][c,a] += scale * +1.00000000 * np.einsum('UV,iUaA,aAuV->iu', gamma1['b'], h['ab'][c,A,v,V], t['ab'][pv,pV,ha,hA], optimize=True)
    
    
    O['a'][c,a] += scale * -1.00000000 * np.einsum('iIuU,vVwU,vVwI->iu', h['ab'][c,C,a,A], lambdas['ab'], t['ab'][pa,pA,ha,hC], optimize=True)
    O['a'][c,a] += scale * +1.00000000 * np.einsum('iIuU,vVuU,vVwI->iw', h['ab'][c,C,a,A], lambdas['ab'], t['ab'][pa,pA,ha,hC], optimize=True)
    O['a'][c,a] += scale * +1.00000000 * np.einsum('iIaA,aAuI->iu', h['ab'][c,C,v,V], t['ab'][pv,pV,ha,hC], optimize=True)
    O['a'][c,a] += scale * +1.00000000 * np.einsum('iUuA,vUwV,vAwV->iu', h['ab'][c,A,a,V], lambdas['ab'], t['ab'][pa,pV,ha,hA], optimize=True)
    O['a'][c,a] += scale * -1.00000000 * np.einsum('iUuA,vUuV,vAwV->iw', h['ab'][c,A,a,V], lambdas['ab'], t['ab'][pa,pV,ha,hA], optimize=True)
    O['a'][c,a] += scale * +1.00000000 * np.einsum('iUaV,UWVX,aWuX->iu', h['ab'][c,A,v,A], lambdas['bb'], t['ab'][pv,pA,ha,hA], optimize=True)
    O['a'][v,a] += scale * +1.00000000 * np.einsum('uv,UV,aUvA,uAwV->aw', eta1['a'], gamma1['b'], h['ab'][v,A,a,V], t['ab'][pa,pV,ha,hA], optimize=True)
    O['a'][v,a] += scale * -1.00000000 * np.einsum('uv,uUwV,UWVX,aWvX->aw', eta1['a'], h['ab'][a,A,a,A], lambdas['bb'], t['ab'][pv,pA,ha,hA], optimize=True)
    O['a'][v,a] += scale * +1.00000000 * np.einsum('uv,aIvA,uAwI->aw', eta1['a'], h['ab'][v,C,a,V], t['ab'][pa,pV,ha,hC], optimize=True)
    
    O['a'][v,a] += scale * -1.00000000 * np.einsum('UV,uv,WX,uUwX,aWvV->aw', eta1['b'], eta1['a'], gamma1['b'], h['ab'][a,A,a,A], t['ab'][pv,pA,ha,hA], optimize=True)
    
    O['a'][v,a] += scale * +1.00000000 * np.einsum('UV,uv,aIvV,uUwI->aw', eta1['b'], eta1['a'], h['ab'][v,C,a,A], t['ab'][pa,pA,ha,hC], optimize=True)
    O['a'][v,a] += scale * -1.00000000 * np.einsum('UV,uv,uIwV,aUvI->aw', eta1['b'], gamma1['a'], h['ab'][a,C,a,A], t['ab'][pv,pA,ha,hC], optimize=True)
    O['a'][v,a] += scale * -1.00000000 * np.einsum('UV,WX,uv,uWwV,aUvX->aw', eta1['b'], gamma1['b'], gamma1['a'], h['ab'][a,A,a,A], t['ab'][pv,pA,ha,hA], optimize=True)
    
    O['a'][v,a] += scale * -1.00000000 * np.einsum('UV,WX,iWuV,aUiX->au', eta1['b'], gamma1['b'], h['ab'][c,A,a,A], t['ab'][pv,pA,hc,hA], optimize=True)
    O['a'][v,a] += scale * +1.00000000 * np.einsum('UV,WX,aWbV,bUuX->au', eta1['b'], gamma1['b'], h['ab'][v,A,v,A], t['ab'][pv,pA,ha,hA], optimize=True)
    O['a'][v,a] += scale * -1.00000000 * np.einsum('UV,iIuV,aUiI->au', eta1['b'], h['ab'][c,C,a,A], t['ab'][pv,pA,hc,hC], optimize=True)
    O['a'][v,a] += scale * +1.00000000 * np.einsum('UV,uUvW,uXwW,aXwV->av', eta1['b'], h['ab'][a,A,a,A], lambdas['ab'], t['ab'][pv,pA,ha,hA], optimize=True)
    O['a'][v,a] += scale * -1.00000000 * np.einsum('UV,uUvW,uXvW,aXwV->aw', eta1['b'], h['ab'][a,A,a,A], lambdas['ab'], t['ab'][pv,pA,ha,hA], optimize=True)
    O['a'][v,a] += scale * -1.00000000 * np.einsum('UV,uWvV,uWwX,aUwX->av', eta1['b'], h['ab'][a,A,a,A], lambdas['ab'], t['ab'][pv,pA,ha,hA], optimize=True)
    O['a'][v,a] += scale * +1.00000000 * np.einsum('UV,uWvV,uWvX,aUwX->aw', eta1['b'], h['ab'][a,A,a,A], lambdas['ab'], t['ab'][pv,pA,ha,hA], optimize=True)
    O['a'][v,a] += scale * +1.00000000 * np.einsum('UV,aIbV,bUuI->au', eta1['b'], h['ab'][v,C,v,A], t['ab'][pv,pA,ha,hC], optimize=True)
    
    
    
    
    O['a'][v,a] += scale * -1.00000000 * np.einsum('uv,uIwA,aAvI->aw', gamma1['a'], h['ab'][a,C,a,V], t['ab'][pv,pV,ha,hC], optimize=True)
    O['a'][v,a] += scale * -1.00000000 * np.einsum('uv,uUwV,UWVX,aWvX->aw', gamma1['a'], h['ab'][a,A,a,A], lambdas['bb'], t['ab'][pv,pA,ha,hA], optimize=True)
    
    O['a'][v,a] += scale * -1.00000000 * np.einsum('UV,uv,uUwA,aAvV->aw', gamma1['b'], gamma1['a'], h['ab'][a,A,a,V], t['ab'][pv,pV,ha,hA], optimize=True)
    O['a'][v,a] += scale * -1.00000000 * np.einsum('UV,iUuA,aAiV->au', gamma1['b'], h['ab'][c,A,a,V], t['ab'][pv,pV,hc,hA], optimize=True)
    O['a'][v,a] += scale * +1.00000000 * np.einsum('UV,uUvW,uXwW,aXwV->av', gamma1['b'], h['ab'][a,A,a,A], lambdas['ab'], t['ab'][pv,pA,ha,hA], optimize=True)
    O['a'][v,a] += scale * -1.00000000 * np.einsum('UV,uUvW,uXvW,aXwV->aw', gamma1['b'], h['ab'][a,A,a,A], lambdas['ab'], t['ab'][pv,pA,ha,hA], optimize=True)
    O['a'][v,a] += scale * -1.00000000 * np.einsum('UV,uWvV,uWwX,aUwX->av', gamma1['b'], h['ab'][a,A,a,A], lambdas['ab'], t['ab'][pv,pA,ha,hA], optimize=True)
    O['a'][v,a] += scale * +1.00000000 * np.einsum('UV,uWvV,uWvX,aUwX->aw', gamma1['b'], h['ab'][a,A,a,A], lambdas['ab'], t['ab'][pv,pA,ha,hA], optimize=True)
    
    
    O['a'][v,a] += scale * +1.00000000 * np.einsum('UV,aUbA,bAuV->au', gamma1['b'], h['ab'][v,A,v,V], t['ab'][pv,pV,ha,hA], optimize=True)
    
    
    O['a'][v,a] += scale * -1.00000000 * np.einsum('iIuA,aAiI->au', h['ab'][c,C,a,V], t['ab'][pv,pV,hc,hC], optimize=True)
    O['a'][v,a] += scale * -1.00000000 * np.einsum('iUuV,UWVX,aWiX->au', h['ab'][c,A,a,A], lambdas['bb'], t['ab'][pv,pA,hc,hA], optimize=True)
    O['a'][v,a] += scale * +1.00000000 * np.einsum('uIvU,uVwU,aVwI->av', h['ab'][a,C,a,A], lambdas['ab'], t['ab'][pv,pA,ha,hC], optimize=True)
    O['a'][v,a] += scale * -1.00000000 * np.einsum('uIvU,uVvU,aVwI->aw', h['ab'][a,C,a,A], lambdas['ab'], t['ab'][pv,pA,ha,hC], optimize=True)
    O['a'][v,a] += scale * -1.00000000 * np.einsum('uUvA,uUwV,aAwV->av', h['ab'][a,A,a,V], lambdas['ab'], t['ab'][pv,pV,ha,hA], optimize=True)
    O['a'][v,a] += scale * +1.00000000 * np.einsum('uUvA,uUvV,aAwV->aw', h['ab'][a,A,a,V], lambdas['ab'], t['ab'][pv,pV,ha,hA], optimize=True)
    O['a'][v,a] += scale * -1.00000000 * np.einsum('aIuU,vVwU,vVwI->au', h['ab'][v,C,a,A], lambdas['ab'], t['ab'][pa,pA,ha,hC], optimize=True)
    O['a'][v,a] += scale * +1.00000000 * np.einsum('aIuU,vVuU,vVwI->aw', h['ab'][v,C,a,A], lambdas['ab'], t['ab'][pa,pA,ha,hC], optimize=True)
    O['a'][v,a] += scale * +1.00000000 * np.einsum('aIbA,bAuI->au', h['ab'][v,C,v,V], t['ab'][pv,pV,ha,hC], optimize=True)
    O['a'][v,a] += scale * +1.00000000 * np.einsum('aUuA,vUwV,vAwV->au', h['ab'][v,A,a,V], lambdas['ab'], t['ab'][pa,pV,ha,hA], optimize=True)
    O['a'][v,a] += scale * -1.00000000 * np.einsum('aUuA,vUuV,vAwV->aw', h['ab'][v,A,a,V], lambdas['ab'], t['ab'][pa,pV,ha,hA], optimize=True)
    O['a'][v,a] += scale * +1.00000000 * np.einsum('aUbV,UWVX,bWuX->au', h['ab'][v,A,v,A], lambdas['bb'], t['ab'][pv,pA,ha,hA], optimize=True)
    O['a'][a,c] += scale * +1.00000000 * np.einsum('uv,UV,wUvA,uAiV->wi', eta1['a'], gamma1['b'], h['ab'][a,A,a,V], t['ab'][pa,pV,hc,hA], optimize=True)
    O['a'][a,c] += scale * +1.00000000 * np.einsum('uv,wIvA,uAiI->wi', eta1['a'], h['ab'][a,C,a,V], t['ab'][pa,pV,hc,hC], optimize=True)
    O['a'][a,c] += scale * +1.00000000 * np.einsum('uv,wUvV,UWVX,uWiX->wi', eta1['a'], h['ab'][a,A,a,A], lambdas['bb'], t['ab'][pa,pA,hc,hA], optimize=True)
    
    O['a'][a,c] += scale * +1.00000000 * np.einsum('UV,uv,WX,wWvV,uUiX->wi', eta1['b'], eta1['a'], gamma1['b'], h['ab'][a,A,a,A], t['ab'][pa,pA,hc,hA], optimize=True)
    
    O['a'][a,c] += scale * +1.00000000 * np.einsum('UV,uv,wIvV,uUiI->wi', eta1['b'], eta1['a'], h['ab'][a,C,a,A], t['ab'][pa,pA,hc,hC], optimize=True)
    O['a'][a,c] += scale * -1.00000000 * np.einsum('UV,uv,uIiV,wUvI->wi', eta1['b'], gamma1['a'], h['ab'][a,C,c,A], t['ab'][pa,pA,ha,hC], optimize=True)
    O['a'][a,c] += scale * +1.00000000 * np.einsum('UV,WX,uv,wUvX,uWiV->wi', eta1['b'], gamma1['b'], gamma1['a'], h['ab'][a,A,a,A], t['ab'][pa,pA,hc,hA], optimize=True)
    
    O['a'][a,c] += scale * -1.00000000 * np.einsum('UV,WX,iWjV,uUiX->uj', eta1['b'], gamma1['b'], h['ab'][c,A,c,A], t['ab'][pa,pA,hc,hA], optimize=True)
    O['a'][a,c] += scale * +1.00000000 * np.einsum('UV,WX,uWaV,aUiX->ui', eta1['b'], gamma1['b'], h['ab'][a,A,v,A], t['ab'][pv,pA,hc,hA], optimize=True)
    O['a'][a,c] += scale * -1.00000000 * np.einsum('UV,iIjV,uUiI->uj', eta1['b'], h['ab'][c,C,c,A], t['ab'][pa,pA,hc,hC], optimize=True)
    O['a'][a,c] += scale * +1.00000000 * np.einsum('UV,uIaV,aUiI->ui', eta1['b'], h['ab'][a,C,v,A], t['ab'][pv,pA,hc,hC], optimize=True)
    
    O['a'][a,c] += scale * +1.00000000 * np.einsum('UV,uUvW,wXvW,wXiV->ui', eta1['b'], h['ab'][a,A,a,A], lambdas['ab'], t['ab'][pa,pA,hc,hA], optimize=True)
    
    O['a'][a,c] += scale * -1.00000000 * np.einsum('UV,uWvV,wWvX,wUiX->ui', eta1['b'], h['ab'][a,A,a,A], lambdas['ab'], t['ab'][pa,pA,hc,hA], optimize=True)
    
    O['a'][a,c] += scale * -1.00000000 * np.einsum('UV,uUvW,uXvW,wXiV->wi', eta1['b'], h['ab'][a,A,a,A], lambdas['ab'], t['ab'][pa,pA,hc,hA], optimize=True)
    
    O['a'][a,c] += scale * +1.00000000 * np.einsum('UV,uWvV,uWvX,wUiX->wi', eta1['b'], h['ab'][a,A,a,A], lambdas['ab'], t['ab'][pa,pA,hc,hA], optimize=True)
    O['a'][a,c] += scale * +1.00000000 * np.einsum('uv,wUvV,UWVX,uWiX->wi', gamma1['a'], h['ab'][a,A,a,A], lambdas['bb'], t['ab'][pa,pA,hc,hA], optimize=True)
    O['a'][a,c] += scale * -1.00000000 * np.einsum('uv,uIiA,wAvI->wi', gamma1['a'], h['ab'][a,C,c,V], t['ab'][pa,pV,ha,hC], optimize=True)
    
    O['a'][a,c] += scale * -1.00000000 * np.einsum('UV,uv,uUiA,wAvV->wi', gamma1['b'], gamma1['a'], h['ab'][a,A,c,V], t['ab'][pa,pV,ha,hA], optimize=True)
    O['a'][a,c] += scale * -1.00000000 * np.einsum('UV,iUjA,uAiV->uj', gamma1['b'], h['ab'][c,A,c,V], t['ab'][pa,pV,hc,hA], optimize=True)
    
    O['a'][a,c] += scale * +1.00000000 * np.einsum('UV,uUvW,wXvW,wXiV->ui', gamma1['b'], h['ab'][a,A,a,A], lambdas['ab'], t['ab'][pa,pA,hc,hA], optimize=True)
    O['a'][a,c] += scale * +1.00000000 * np.einsum('UV,uUaA,aAiV->ui', gamma1['b'], h['ab'][a,A,v,V], t['ab'][pv,pV,hc,hA], optimize=True)
    
    O['a'][a,c] += scale * -1.00000000 * np.einsum('UV,uWvV,wWvX,wUiX->ui', gamma1['b'], h['ab'][a,A,a,A], lambdas['ab'], t['ab'][pa,pA,hc,hA], optimize=True)
    
    O['a'][a,c] += scale * -1.00000000 * np.einsum('UV,uUvW,uXvW,wXiV->wi', gamma1['b'], h['ab'][a,A,a,A], lambdas['ab'], t['ab'][pa,pA,hc,hA], optimize=True)
    
    O['a'][a,c] += scale * +1.00000000 * np.einsum('UV,uWvV,uWvX,wUiX->wi', gamma1['b'], h['ab'][a,A,a,A], lambdas['ab'], t['ab'][pa,pA,hc,hA], optimize=True)
    O['a'][a,c] += scale * -1.00000000 * np.einsum('iIjA,uAiI->uj', h['ab'][c,C,c,V], t['ab'][pa,pV,hc,hC], optimize=True)
    O['a'][a,c] += scale * -1.00000000 * np.einsum('iUjV,UWVX,uWiX->uj', h['ab'][c,A,c,A], lambdas['bb'], t['ab'][pa,pA,hc,hA], optimize=True)
    O['a'][a,c] += scale * -1.00000000 * np.einsum('uIiU,vVwU,vVwI->ui', h['ab'][a,C,c,A], lambdas['ab'], t['ab'][pa,pA,ha,hC], optimize=True)
    O['a'][a,c] += scale * +1.00000000 * np.einsum('uIvU,wVvU,wViI->ui', h['ab'][a,C,a,A], lambdas['ab'], t['ab'][pa,pA,hc,hC], optimize=True)
    O['a'][a,c] += scale * +1.00000000 * np.einsum('uIaA,aAiI->ui', h['ab'][a,C,v,V], t['ab'][pv,pV,hc,hC], optimize=True)
    O['a'][a,c] += scale * +1.00000000 * np.einsum('uUiA,vUwV,vAwV->ui', h['ab'][a,A,c,V], lambdas['ab'], t['ab'][pa,pV,ha,hA], optimize=True)
    O['a'][a,c] += scale * -1.00000000 * np.einsum('uUvA,wUvV,wAiV->ui', h['ab'][a,A,a,V], lambdas['ab'], t['ab'][pa,pV,hc,hA], optimize=True)
    O['a'][a,c] += scale * +1.00000000 * np.einsum('uUaV,UWVX,aWiX->ui', h['ab'][a,A,v,A], lambdas['bb'], t['ab'][pv,pA,hc,hA], optimize=True)
    O['a'][a,c] += scale * +1.00000000 * np.einsum('uIiU,uVvU,wVvI->wi', h['ab'][a,C,c,A], lambdas['ab'], t['ab'][pa,pA,ha,hC], optimize=True)
    O['a'][a,c] += scale * -1.00000000 * np.einsum('uIvU,uVvU,wViI->wi', h['ab'][a,C,a,A], lambdas['ab'], t['ab'][pa,pA,hc,hC], optimize=True)
    O['a'][a,c] += scale * -1.00000000 * np.einsum('uUiA,uUvV,wAvV->wi', h['ab'][a,A,c,V], lambdas['ab'], t['ab'][pa,pV,ha,hA], optimize=True)
    O['a'][a,c] += scale * +1.00000000 * np.einsum('uUvA,uUvV,wAiV->wi', h['ab'][a,A,a,V], lambdas['ab'], t['ab'][pa,pV,hc,hA], optimize=True)
    O['a'][c,c] += scale * +1.00000000 * np.einsum('uv,UV,iUvA,uAjV->ij', eta1['a'], gamma1['b'], h['ab'][c,A,a,V], t['ab'][pa,pV,hc,hA], optimize=True)
    O['a'][c,c] += scale * +1.00000000 * np.einsum('uv,iIvA,uAjI->ij', eta1['a'], h['ab'][c,C,a,V], t['ab'][pa,pV,hc,hC], optimize=True)
    O['a'][c,c] += scale * +1.00000000 * np.einsum('uv,iUvV,UWVX,uWjX->ij', eta1['a'], h['ab'][c,A,a,A], lambdas['bb'], t['ab'][pa,pA,hc,hA], optimize=True)
    O['a'][c,c] += scale * +1.00000000 * np.einsum('UV,uv,WX,iWvV,uUjX->ij', eta1['b'], eta1['a'], gamma1['b'], h['ab'][c,A,a,A], t['ab'][pa,pA,hc,hA], optimize=True)
    O['a'][c,c] += scale * +1.00000000 * np.einsum('UV,uv,iIvV,uUjI->ij', eta1['b'], eta1['a'], h['ab'][c,C,a,A], t['ab'][pa,pA,hc,hC], optimize=True)
    O['a'][c,c] += scale * +1.00000000 * np.einsum('UV,WX,uv,iUvX,uWjV->ij', eta1['b'], gamma1['b'], gamma1['a'], h['ab'][c,A,a,A], t['ab'][pa,pA,hc,hA], optimize=True)
    O['a'][c,c] += scale * +1.00000000 * np.einsum('UV,WX,iWaV,aUjX->ij', eta1['b'], gamma1['b'], h['ab'][c,A,v,A], t['ab'][pv,pA,hc,hA], optimize=True)
    O['a'][c,c] += scale * +1.00000000 * np.einsum('UV,iIaV,aUjI->ij', eta1['b'], h['ab'][c,C,v,A], t['ab'][pv,pA,hc,hC], optimize=True)
    
    O['a'][c,c] += scale * +1.00000000 * np.einsum('UV,iUuW,vXuW,vXjV->ij', eta1['b'], h['ab'][c,A,a,A], lambdas['ab'], t['ab'][pa,pA,hc,hA], optimize=True)
    
    O['a'][c,c] += scale * -1.00000000 * np.einsum('UV,iWuV,vWuX,vUjX->ij', eta1['b'], h['ab'][c,A,a,A], lambdas['ab'], t['ab'][pa,pA,hc,hA], optimize=True)
    O['a'][c,c] += scale * +1.00000000 * np.einsum('uv,iUvV,UWVX,uWjX->ij', gamma1['a'], h['ab'][c,A,a,A], lambdas['bb'], t['ab'][pa,pA,hc,hA], optimize=True)
    
    O['a'][c,c] += scale * +1.00000000 * np.einsum('UV,iUuW,vXuW,vXjV->ij', gamma1['b'], h['ab'][c,A,a,A], lambdas['ab'], t['ab'][pa,pA,hc,hA], optimize=True)
    O['a'][c,c] += scale * +1.00000000 * np.einsum('UV,iUaA,aAjV->ij', gamma1['b'], h['ab'][c,A,v,V], t['ab'][pv,pV,hc,hA], optimize=True)
    
    O['a'][c,c] += scale * -1.00000000 * np.einsum('UV,iWuV,vWuX,vUjX->ij', gamma1['b'], h['ab'][c,A,a,A], lambdas['ab'], t['ab'][pa,pA,hc,hA], optimize=True)
    O['a'][c,c] += scale * -1.00000000 * np.einsum('iIjU,uVvU,uVvI->ij', h['ab'][c,C,c,A], lambdas['ab'], t['ab'][pa,pA,ha,hC], optimize=True)
    O['a'][c,c] += scale * +1.00000000 * np.einsum('iIuU,vVuU,vVjI->ij', h['ab'][c,C,a,A], lambdas['ab'], t['ab'][pa,pA,hc,hC], optimize=True)
    O['a'][c,c] += scale * +1.00000000 * np.einsum('iIaA,aAjI->ij', h['ab'][c,C,v,V], t['ab'][pv,pV,hc,hC], optimize=True)
    O['a'][c,c] += scale * +1.00000000 * np.einsum('iUjA,uUvV,uAvV->ij', h['ab'][c,A,c,V], lambdas['ab'], t['ab'][pa,pV,ha,hA], optimize=True)
    O['a'][c,c] += scale * -1.00000000 * np.einsum('iUuA,vUuV,vAjV->ij', h['ab'][c,A,a,V], lambdas['ab'], t['ab'][pa,pV,hc,hA], optimize=True)
    O['a'][c,c] += scale * +1.00000000 * np.einsum('iUaV,UWVX,aWjX->ij', h['ab'][c,A,v,A], lambdas['bb'], t['ab'][pv,pA,hc,hA], optimize=True)
    O['a'][v,c] += scale * +1.00000000 * np.einsum('uv,UV,aUvA,uAiV->ai', eta1['a'], gamma1['b'], h['ab'][v,A,a,V], t['ab'][pa,pV,hc,hA], optimize=True)
    O['a'][v,c] += scale * -1.00000000 * np.einsum('uv,uUiV,UWVX,aWvX->ai', eta1['a'], h['ab'][a,A,c,A], lambdas['bb'], t['ab'][pv,pA,ha,hA], optimize=True)
    O['a'][v,c] += scale * +1.00000000 * np.einsum('uv,aIvA,uAiI->ai', eta1['a'], h['ab'][v,C,a,V], t['ab'][pa,pV,hc,hC], optimize=True)
    O['a'][v,c] += scale * +1.00000000 * np.einsum('uv,aUvV,UWVX,uWiX->ai', eta1['a'], h['ab'][v,A,a,A], lambdas['bb'], t['ab'][pa,pA,hc,hA], optimize=True)
    O['a'][v,c] += scale * -1.00000000 * np.einsum('UV,uv,WX,uUiX,aWvV->ai', eta1['b'], eta1['a'], gamma1['b'], h['ab'][a,A,c,A], t['ab'][pv,pA,ha,hA], optimize=True)
    O['a'][v,c] += scale * +1.00000000 * np.einsum('UV,uv,WX,aWvV,uUiX->ai', eta1['b'], eta1['a'], gamma1['b'], h['ab'][v,A,a,A], t['ab'][pa,pA,hc,hA], optimize=True)
    O['a'][v,c] += scale * +1.00000000 * np.einsum('UV,uv,aIvV,uUiI->ai', eta1['b'], eta1['a'], h['ab'][v,C,a,A], t['ab'][pa,pA,hc,hC], optimize=True)
    O['a'][v,c] += scale * -1.00000000 * np.einsum('UV,uv,uIiV,aUvI->ai', eta1['b'], gamma1['a'], h['ab'][a,C,c,A], t['ab'][pv,pA,ha,hC], optimize=True)
    O['a'][v,c] += scale * -1.00000000 * np.einsum('UV,WX,uv,uWiV,aUvX->ai', eta1['b'], gamma1['b'], gamma1['a'], h['ab'][a,A,c,A], t['ab'][pv,pA,ha,hA], optimize=True)
    O['a'][v,c] += scale * +1.00000000 * np.einsum('UV,WX,uv,aUvX,uWiV->ai', eta1['b'], gamma1['b'], gamma1['a'], h['ab'][v,A,a,A], t['ab'][pa,pA,hc,hA], optimize=True)
    O['a'][v,c] += scale * -1.00000000 * np.einsum('UV,WX,iWjV,aUiX->aj', eta1['b'], gamma1['b'], h['ab'][c,A,c,A], t['ab'][pv,pA,hc,hA], optimize=True)
    O['a'][v,c] += scale * +1.00000000 * np.einsum('UV,WX,aWbV,bUiX->ai', eta1['b'], gamma1['b'], h['ab'][v,A,v,A], t['ab'][pv,pA,hc,hA], optimize=True)
    O['a'][v,c] += scale * -1.00000000 * np.einsum('UV,iIjV,aUiI->aj', eta1['b'], h['ab'][c,C,c,A], t['ab'][pv,pA,hc,hC], optimize=True)
    O['a'][v,c] += scale * +1.00000000 * np.einsum('UV,uUiW,uXvW,aXvV->ai', eta1['b'], h['ab'][a,A,c,A], lambdas['ab'], t['ab'][pv,pA,ha,hA], optimize=True)
    O['a'][v,c] += scale * -1.00000000 * np.einsum('UV,uUvW,uXvW,aXiV->ai', eta1['b'], h['ab'][a,A,a,A], lambdas['ab'], t['ab'][pv,pA,hc,hA], optimize=True)
    O['a'][v,c] += scale * -1.00000000 * np.einsum('UV,uWiV,uWvX,aUvX->ai', eta1['b'], h['ab'][a,A,c,A], lambdas['ab'], t['ab'][pv,pA,ha,hA], optimize=True)
    O['a'][v,c] += scale * +1.00000000 * np.einsum('UV,uWvV,uWvX,aUiX->ai', eta1['b'], h['ab'][a,A,a,A], lambdas['ab'], t['ab'][pv,pA,hc,hA], optimize=True)
    O['a'][v,c] += scale * +1.00000000 * np.einsum('UV,aIbV,bUiI->ai', eta1['b'], h['ab'][v,C,v,A], t['ab'][pv,pA,hc,hC], optimize=True)
    
    O['a'][v,c] += scale * +1.00000000 * np.einsum('UV,aUuW,vXuW,vXiV->ai', eta1['b'], h['ab'][v,A,a,A], lambdas['ab'], t['ab'][pa,pA,hc,hA], optimize=True)
    
    O['a'][v,c] += scale * -1.00000000 * np.einsum('UV,aWuV,vWuX,vUiX->ai', eta1['b'], h['ab'][v,A,a,A], lambdas['ab'], t['ab'][pa,pA,hc,hA], optimize=True)
    O['a'][v,c] += scale * -1.00000000 * np.einsum('uv,uIiA,aAvI->ai', gamma1['a'], h['ab'][a,C,c,V], t['ab'][pv,pV,ha,hC], optimize=True)
    O['a'][v,c] += scale * -1.00000000 * np.einsum('uv,uUiV,UWVX,aWvX->ai', gamma1['a'], h['ab'][a,A,c,A], lambdas['bb'], t['ab'][pv,pA,ha,hA], optimize=True)
    O['a'][v,c] += scale * +1.00000000 * np.einsum('uv,aUvV,UWVX,uWiX->ai', gamma1['a'], h['ab'][v,A,a,A], lambdas['bb'], t['ab'][pa,pA,hc,hA], optimize=True)
    O['a'][v,c] += scale * -1.00000000 * np.einsum('UV,uv,uUiA,aAvV->ai', gamma1['b'], gamma1['a'], h['ab'][a,A,c,V], t['ab'][pv,pV,ha,hA], optimize=True)
    O['a'][v,c] += scale * -1.00000000 * np.einsum('UV,iUjA,aAiV->aj', gamma1['b'], h['ab'][c,A,c,V], t['ab'][pv,pV,hc,hA], optimize=True)
    O['a'][v,c] += scale * +1.00000000 * np.einsum('UV,uUiW,uXvW,aXvV->ai', gamma1['b'], h['ab'][a,A,c,A], lambdas['ab'], t['ab'][pv,pA,ha,hA], optimize=True)
    O['a'][v,c] += scale * -1.00000000 * np.einsum('UV,uUvW,uXvW,aXiV->ai', gamma1['b'], h['ab'][a,A,a,A], lambdas['ab'], t['ab'][pv,pA,hc,hA], optimize=True)
    O['a'][v,c] += scale * -1.00000000 * np.einsum('UV,uWiV,uWvX,aUvX->ai', gamma1['b'], h['ab'][a,A,c,A], lambdas['ab'], t['ab'][pv,pA,ha,hA], optimize=True)
    O['a'][v,c] += scale * +1.00000000 * np.einsum('UV,uWvV,uWvX,aUiX->ai', gamma1['b'], h['ab'][a,A,a,A], lambdas['ab'], t['ab'][pv,pA,hc,hA], optimize=True)
    
    O['a'][v,c] += scale * +1.00000000 * np.einsum('UV,aUuW,vXuW,vXiV->ai', gamma1['b'], h['ab'][v,A,a,A], lambdas['ab'], t['ab'][pa,pA,hc,hA], optimize=True)
    O['a'][v,c] += scale * +1.00000000 * np.einsum('UV,aUbA,bAiV->ai', gamma1['b'], h['ab'][v,A,v,V], t['ab'][pv,pV,hc,hA], optimize=True)
    
    O['a'][v,c] += scale * -1.00000000 * np.einsum('UV,aWuV,vWuX,vUiX->ai', gamma1['b'], h['ab'][v,A,a,A], lambdas['ab'], t['ab'][pa,pA,hc,hA], optimize=True)
    O['a'][v,c] += scale * -1.00000000 * np.einsum('iIjA,aAiI->aj', h['ab'][c,C,c,V], t['ab'][pv,pV,hc,hC], optimize=True)
    O['a'][v,c] += scale * -1.00000000 * np.einsum('iUjV,UWVX,aWiX->aj', h['ab'][c,A,c,A], lambdas['bb'], t['ab'][pv,pA,hc,hA], optimize=True)
    O['a'][v,c] += scale * +1.00000000 * np.einsum('uIiU,uVvU,aVvI->ai', h['ab'][a,C,c,A], lambdas['ab'], t['ab'][pv,pA,ha,hC], optimize=True)
    O['a'][v,c] += scale * -1.00000000 * np.einsum('uIvU,uVvU,aViI->ai', h['ab'][a,C,a,A], lambdas['ab'], t['ab'][pv,pA,hc,hC], optimize=True)
    O['a'][v,c] += scale * -1.00000000 * np.einsum('uUiA,uUvV,aAvV->ai', h['ab'][a,A,c,V], lambdas['ab'], t['ab'][pv,pV,ha,hA], optimize=True)
    O['a'][v,c] += scale * +1.00000000 * np.einsum('uUvA,uUvV,aAiV->ai', h['ab'][a,A,a,V], lambdas['ab'], t['ab'][pv,pV,hc,hA], optimize=True)
    O['a'][v,c] += scale * -1.00000000 * np.einsum('aIiU,uVvU,uVvI->ai', h['ab'][v,C,c,A], lambdas['ab'], t['ab'][pa,pA,ha,hC], optimize=True)
    O['a'][v,c] += scale * +1.00000000 * np.einsum('aIuU,vVuU,vViI->ai', h['ab'][v,C,a,A], lambdas['ab'], t['ab'][pa,pA,hc,hC], optimize=True)
    O['a'][v,c] += scale * +1.00000000 * np.einsum('aIbA,bAiI->ai', h['ab'][v,C,v,V], t['ab'][pv,pV,hc,hC], optimize=True)
    O['a'][v,c] += scale * +1.00000000 * np.einsum('aUiA,uUvV,uAvV->ai', h['ab'][v,A,c,V], lambdas['ab'], t['ab'][pa,pV,ha,hA], optimize=True)
    O['a'][v,c] += scale * -1.00000000 * np.einsum('aUuA,vUuV,vAiV->ai', h['ab'][v,A,a,V], lambdas['ab'], t['ab'][pa,pV,hc,hA], optimize=True)
    O['a'][v,c] += scale * +1.00000000 * np.einsum('aUbV,UWVX,bWiX->ai', h['ab'][v,A,v,A], lambdas['bb'], t['ab'][pv,pA,hc,hA], optimize=True)
    
    
    O['a'][a,v] += scale * -1.00000000 * np.einsum('UV,uv,uIaV,wUvI->wa', eta1['b'], gamma1['a'], h['ab'][a,C,v,A], t['ab'][pa,pA,ha,hC], optimize=True)
    
    O['a'][a,v] += scale * -1.00000000 * np.einsum('UV,WX,iWaV,uUiX->ua', eta1['b'], gamma1['b'], h['ab'][c,A,v,A], t['ab'][pa,pA,hc,hA], optimize=True)
    O['a'][a,v] += scale * -1.00000000 * np.einsum('UV,iIaV,uUiI->ua', eta1['b'], h['ab'][c,C,v,A], t['ab'][pa,pA,hc,hC], optimize=True)
    
    
    
    
    O['a'][a,v] += scale * -1.00000000 * np.einsum('uv,uIaA,wAvI->wa', gamma1['a'], h['ab'][a,C,v,V], t['ab'][pa,pV,ha,hC], optimize=True)
    
    O['a'][a,v] += scale * -1.00000000 * np.einsum('UV,uv,uUaA,wAvV->wa', gamma1['b'], gamma1['a'], h['ab'][a,A,v,V], t['ab'][pa,pV,ha,hA], optimize=True)
    O['a'][a,v] += scale * -1.00000000 * np.einsum('UV,iUaA,uAiV->ua', gamma1['b'], h['ab'][c,A,v,V], t['ab'][pa,pV,hc,hA], optimize=True)
    
    
    
    
    O['a'][a,v] += scale * -1.00000000 * np.einsum('iIaA,uAiI->ua', h['ab'][c,C,v,V], t['ab'][pa,pV,hc,hC], optimize=True)
    O['a'][a,v] += scale * -1.00000000 * np.einsum('iUaV,UWVX,uWiX->ua', h['ab'][c,A,v,A], lambdas['bb'], t['ab'][pa,pA,hc,hA], optimize=True)
    O['a'][a,v] += scale * -1.00000000 * np.einsum('uIaU,vVwU,vVwI->ua', h['ab'][a,C,v,A], lambdas['ab'], t['ab'][pa,pA,ha,hC], optimize=True)
    O['a'][a,v] += scale * +1.00000000 * np.einsum('uUaA,vUwV,vAwV->ua', h['ab'][a,A,v,V], lambdas['ab'], t['ab'][pa,pV,ha,hA], optimize=True)
    O['a'][a,v] += scale * +1.00000000 * np.einsum('uIaU,uVvU,wVvI->wa', h['ab'][a,C,v,A], lambdas['ab'], t['ab'][pa,pA,ha,hC], optimize=True)
    O['a'][a,v] += scale * -1.00000000 * np.einsum('uUaA,uUvV,wAvV->wa', h['ab'][a,A,v,V], lambdas['ab'], t['ab'][pa,pV,ha,hA], optimize=True)
    
    
    
    
    O['a'][c,v] += scale * -1.00000000 * np.einsum('iIaU,uVvU,uVvI->ia', h['ab'][c,C,v,A], lambdas['ab'], t['ab'][pa,pA,ha,hC], optimize=True)
    O['a'][c,v] += scale * +1.00000000 * np.einsum('iUaA,uUvV,uAvV->ia', h['ab'][c,A,v,V], lambdas['ab'], t['ab'][pa,pV,ha,hA], optimize=True)
    O['a'][v,v] += scale * -1.00000000 * np.einsum('uv,uUaV,UWVX,bWvX->ba', eta1['a'], h['ab'][a,A,v,A], lambdas['bb'], t['ab'][pv,pA,ha,hA], optimize=True)
    O['a'][v,v] += scale * -1.00000000 * np.einsum('UV,uv,WX,uUaX,bWvV->ba', eta1['b'], eta1['a'], gamma1['b'], h['ab'][a,A,v,A], t['ab'][pv,pA,ha,hA], optimize=True)
    O['a'][v,v] += scale * -1.00000000 * np.einsum('UV,uv,uIaV,bUvI->ba', eta1['b'], gamma1['a'], h['ab'][a,C,v,A], t['ab'][pv,pA,ha,hC], optimize=True)
    O['a'][v,v] += scale * -1.00000000 * np.einsum('UV,WX,uv,uWaV,bUvX->ba', eta1['b'], gamma1['b'], gamma1['a'], h['ab'][a,A,v,A], t['ab'][pv,pA,ha,hA], optimize=True)
    O['a'][v,v] += scale * -1.00000000 * np.einsum('UV,WX,iWaV,bUiX->ba', eta1['b'], gamma1['b'], h['ab'][c,A,v,A], t['ab'][pv,pA,hc,hA], optimize=True)
    O['a'][v,v] += scale * -1.00000000 * np.einsum('UV,iIaV,bUiI->ba', eta1['b'], h['ab'][c,C,v,A], t['ab'][pv,pA,hc,hC], optimize=True)
    O['a'][v,v] += scale * +1.00000000 * np.einsum('UV,uUaW,uXvW,bXvV->ba', eta1['b'], h['ab'][a,A,v,A], lambdas['ab'], t['ab'][pv,pA,ha,hA], optimize=True)
    O['a'][v,v] += scale * -1.00000000 * np.einsum('UV,uWaV,uWvX,bUvX->ba', eta1['b'], h['ab'][a,A,v,A], lambdas['ab'], t['ab'][pv,pA,ha,hA], optimize=True)
    
    
    O['a'][v,v] += scale * -1.00000000 * np.einsum('uv,uIaA,bAvI->ba', gamma1['a'], h['ab'][a,C,v,V], t['ab'][pv,pV,ha,hC], optimize=True)
    O['a'][v,v] += scale * -1.00000000 * np.einsum('uv,uUaV,UWVX,bWvX->ba', gamma1['a'], h['ab'][a,A,v,A], lambdas['bb'], t['ab'][pv,pA,ha,hA], optimize=True)
    O['a'][v,v] += scale * -1.00000000 * np.einsum('UV,uv,uUaA,bAvV->ba', gamma1['b'], gamma1['a'], h['ab'][a,A,v,V], t['ab'][pv,pV,ha,hA], optimize=True)
    O['a'][v,v] += scale * -1.00000000 * np.einsum('UV,iUaA,bAiV->ba', gamma1['b'], h['ab'][c,A,v,V], t['ab'][pv,pV,hc,hA], optimize=True)
    O['a'][v,v] += scale * +1.00000000 * np.einsum('UV,uUaW,uXvW,bXvV->ba', gamma1['b'], h['ab'][a,A,v,A], lambdas['ab'], t['ab'][pv,pA,ha,hA], optimize=True)
    O['a'][v,v] += scale * -1.00000000 * np.einsum('UV,uWaV,uWvX,bUvX->ba', gamma1['b'], h['ab'][a,A,v,A], lambdas['ab'], t['ab'][pv,pA,ha,hA], optimize=True)
    
    
    O['a'][v,v] += scale * -1.00000000 * np.einsum('iIaA,bAiI->ba', h['ab'][c,C,v,V], t['ab'][pv,pV,hc,hC], optimize=True)
    O['a'][v,v] += scale * -1.00000000 * np.einsum('iUaV,UWVX,bWiX->ba', h['ab'][c,A,v,A], lambdas['bb'], t['ab'][pv,pA,hc,hA], optimize=True)
    O['a'][v,v] += scale * +1.00000000 * np.einsum('uIaU,uVvU,bVvI->ba', h['ab'][a,C,v,A], lambdas['ab'], t['ab'][pv,pA,ha,hC], optimize=True)
    O['a'][v,v] += scale * -1.00000000 * np.einsum('uUaA,uUvV,bAvV->ba', h['ab'][a,A,v,V], lambdas['ab'], t['ab'][pv,pV,ha,hA], optimize=True)
    O['a'][v,v] += scale * -1.00000000 * np.einsum('aIbU,uVvU,uVvI->ab', h['ab'][v,C,v,A], lambdas['ab'], t['ab'][pa,pA,ha,hC], optimize=True)
    O['a'][v,v] += scale * +1.00000000 * np.einsum('aUbA,uUvV,uAvV->ab', h['ab'][v,A,v,V], lambdas['ab'], t['ab'][pa,pV,ha,hA], optimize=True)

    t1 = time.time()
    if verbose: print("h2b_t2b_c1a took {:.4f} seconds to run.".format(t1-t0))

    return O
def h2b_t2c_c1a(O, h, t, gamma1, eta1, lambdas, orbspace, verbose=False, scale=1.0):
    # 54 lines
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

    
    
    
    
    O['a'][a,a] += scale * -0.50000000 * np.einsum('uIvU,VWUX,VWIX->uv', h['ab'][a,C,a,A], lambdas['bb'], t['bb'][pA,pA,hC,hA], optimize=True)
    O['a'][a,a] += scale * -0.50000000 * np.einsum('uUvA,UVWX,VAWX->uv', h['ab'][a,A,a,V], lambdas['bb'], t['bb'][pA,pV,hA,hA], optimize=True)
    
    
    
    
    O['a'][c,a] += scale * -0.50000000 * np.einsum('iIuU,VWUX,VWIX->iu', h['ab'][c,C,a,A], lambdas['bb'], t['bb'][pA,pA,hC,hA], optimize=True)
    O['a'][c,a] += scale * -0.50000000 * np.einsum('iUuA,UVWX,VAWX->iu', h['ab'][c,A,a,V], lambdas['bb'], t['bb'][pA,pV,hA,hA], optimize=True)
    
    
    
    
    O['a'][v,a] += scale * -0.50000000 * np.einsum('aIuU,VWUX,VWIX->au', h['ab'][v,C,a,A], lambdas['bb'], t['bb'][pA,pA,hC,hA], optimize=True)
    O['a'][v,a] += scale * -0.50000000 * np.einsum('aUuA,UVWX,VAWX->au', h['ab'][v,A,a,V], lambdas['bb'], t['bb'][pA,pV,hA,hA], optimize=True)
    
    
    
    
    O['a'][a,c] += scale * -0.50000000 * np.einsum('uIiU,VWUX,VWIX->ui', h['ab'][a,C,c,A], lambdas['bb'], t['bb'][pA,pA,hC,hA], optimize=True)
    O['a'][a,c] += scale * -0.50000000 * np.einsum('uUiA,UVWX,VAWX->ui', h['ab'][a,A,c,V], lambdas['bb'], t['bb'][pA,pV,hA,hA], optimize=True)
    
    
    
    
    O['a'][c,c] += scale * -0.50000000 * np.einsum('iIjU,VWUX,VWIX->ij', h['ab'][c,C,c,A], lambdas['bb'], t['bb'][pA,pA,hC,hA], optimize=True)
    O['a'][c,c] += scale * -0.50000000 * np.einsum('iUjA,UVWX,VAWX->ij', h['ab'][c,A,c,V], lambdas['bb'], t['bb'][pA,pV,hA,hA], optimize=True)
    
    
    
    
    O['a'][v,c] += scale * -0.50000000 * np.einsum('aIiU,VWUX,VWIX->ai', h['ab'][v,C,c,A], lambdas['bb'], t['bb'][pA,pA,hC,hA], optimize=True)
    O['a'][v,c] += scale * -0.50000000 * np.einsum('aUiA,UVWX,VAWX->ai', h['ab'][v,A,c,V], lambdas['bb'], t['bb'][pA,pV,hA,hA], optimize=True)
    
    
    
    
    O['a'][a,v] += scale * -0.50000000 * np.einsum('uIaU,VWUX,VWIX->ua', h['ab'][a,C,v,A], lambdas['bb'], t['bb'][pA,pA,hC,hA], optimize=True)
    O['a'][a,v] += scale * -0.50000000 * np.einsum('uUaA,UVWX,VAWX->ua', h['ab'][a,A,v,V], lambdas['bb'], t['bb'][pA,pV,hA,hA], optimize=True)
    
    
    
    
    O['a'][c,v] += scale * -0.50000000 * np.einsum('iIaU,VWUX,VWIX->ia', h['ab'][c,C,v,A], lambdas['bb'], t['bb'][pA,pA,hC,hA], optimize=True)
    O['a'][c,v] += scale * -0.50000000 * np.einsum('iUaA,UVWX,VAWX->ia', h['ab'][c,A,v,V], lambdas['bb'], t['bb'][pA,pV,hA,hA], optimize=True)
    
    
    
    
    O['a'][v,v] += scale * -0.50000000 * np.einsum('aIbU,VWUX,VWIX->ab', h['ab'][v,C,v,A], lambdas['bb'], t['bb'][pA,pA,hC,hA], optimize=True)
    O['a'][v,v] += scale * -0.50000000 * np.einsum('aUbA,UVWX,VAWX->ab', h['ab'][v,A,v,V], lambdas['bb'], t['bb'][pA,pV,hA,hA], optimize=True)

    t1 = time.time()
    if verbose: print("h2b_t2c_c1a took {:.4f} seconds to run.".format(t1-t0))

    return O
def h2c_t2b_c1a(O, h, t, gamma1, eta1, lambdas, orbspace, verbose=False, scale=1.0):
    # 24 lines
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

    
    
    
    
    O['a'][a,a] += scale * +0.50000000 * np.einsum('IUVW,UXVW,uXvI->uv', h['bb'][C,A,A,A], lambdas['bb'], t['ab'][pa,pA,ha,hC], optimize=True)
    O['a'][a,a] += scale * +0.50000000 * np.einsum('UVWA,UVWX,uAvX->uv', h['bb'][A,A,A,V], lambdas['bb'], t['ab'][pa,pV,ha,hA], optimize=True)
    O['a'][v,a] += scale * +0.50000000 * np.einsum('UV,UWXY,WZXY,aZuV->au', eta1['b'], h['bb'][A,A,A,A], lambdas['bb'], t['ab'][pv,pA,ha,hA], optimize=True)
    O['a'][v,a] += scale * -0.50000000 * np.einsum('UV,WXVY,WXYZ,aUuZ->au', eta1['b'], h['bb'][A,A,A,A], lambdas['bb'], t['ab'][pv,pA,ha,hA], optimize=True)
    O['a'][v,a] += scale * +0.50000000 * np.einsum('UV,UWXY,WZXY,aZuV->au', gamma1['b'], h['bb'][A,A,A,A], lambdas['bb'], t['ab'][pv,pA,ha,hA], optimize=True)
    O['a'][v,a] += scale * -0.50000000 * np.einsum('UV,WXVY,WXYZ,aUuZ->au', gamma1['b'], h['bb'][A,A,A,A], lambdas['bb'], t['ab'][pv,pA,ha,hA], optimize=True)
    O['a'][v,a] += scale * +0.50000000 * np.einsum('IUVW,UXVW,aXuI->au', h['bb'][C,A,A,A], lambdas['bb'], t['ab'][pv,pA,ha,hC], optimize=True)
    O['a'][v,a] += scale * +0.50000000 * np.einsum('UVWA,UVWX,aAuX->au', h['bb'][A,A,A,V], lambdas['bb'], t['ab'][pv,pV,ha,hA], optimize=True)
    O['a'][a,c] += scale * +0.50000000 * np.einsum('UV,UWXY,WZXY,uZiV->ui', eta1['b'], h['bb'][A,A,A,A], lambdas['bb'], t['ab'][pa,pA,hc,hA], optimize=True)
    O['a'][a,c] += scale * -0.50000000 * np.einsum('UV,WXVY,WXYZ,uUiZ->ui', eta1['b'], h['bb'][A,A,A,A], lambdas['bb'], t['ab'][pa,pA,hc,hA], optimize=True)
    O['a'][a,c] += scale * +0.50000000 * np.einsum('UV,UWXY,WZXY,uZiV->ui', gamma1['b'], h['bb'][A,A,A,A], lambdas['bb'], t['ab'][pa,pA,hc,hA], optimize=True)
    O['a'][a,c] += scale * -0.50000000 * np.einsum('UV,WXVY,WXYZ,uUiZ->ui', gamma1['b'], h['bb'][A,A,A,A], lambdas['bb'], t['ab'][pa,pA,hc,hA], optimize=True)
    O['a'][a,c] += scale * +0.50000000 * np.einsum('IUVW,UXVW,uXiI->ui', h['bb'][C,A,A,A], lambdas['bb'], t['ab'][pa,pA,hc,hC], optimize=True)
    O['a'][a,c] += scale * +0.50000000 * np.einsum('UVWA,UVWX,uAiX->ui', h['bb'][A,A,A,V], lambdas['bb'], t['ab'][pa,pV,hc,hA], optimize=True)
    O['a'][v,c] += scale * +0.50000000 * np.einsum('UV,UWXY,WZXY,aZiV->ai', eta1['b'], h['bb'][A,A,A,A], lambdas['bb'], t['ab'][pv,pA,hc,hA], optimize=True)
    O['a'][v,c] += scale * -0.50000000 * np.einsum('UV,WXVY,WXYZ,aUiZ->ai', eta1['b'], h['bb'][A,A,A,A], lambdas['bb'], t['ab'][pv,pA,hc,hA], optimize=True)
    O['a'][v,c] += scale * +0.50000000 * np.einsum('UV,UWXY,WZXY,aZiV->ai', gamma1['b'], h['bb'][A,A,A,A], lambdas['bb'], t['ab'][pv,pA,hc,hA], optimize=True)
    O['a'][v,c] += scale * -0.50000000 * np.einsum('UV,WXVY,WXYZ,aUiZ->ai', gamma1['b'], h['bb'][A,A,A,A], lambdas['bb'], t['ab'][pv,pA,hc,hA], optimize=True)
    O['a'][v,c] += scale * +0.50000000 * np.einsum('IUVW,UXVW,aXiI->ai', h['bb'][C,A,A,A], lambdas['bb'], t['ab'][pv,pA,hc,hC], optimize=True)
    O['a'][v,c] += scale * +0.50000000 * np.einsum('UVWA,UVWX,aAiX->ai', h['bb'][A,A,A,V], lambdas['bb'], t['ab'][pv,pV,hc,hA], optimize=True)

    t1 = time.time()
    if verbose: print("h2c_t2b_c1a took {:.4f} seconds to run.".format(t1-t0))

    return O
def h1a_t2b_c1b(O, h, t, gamma1, eta1, lambdas, orbspace, verbose=False, scale=1.0):
    # 20 lines
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

    
    
    O['b'][A,A] += scale * +1.00000000 * np.einsum('uv,iv,uUiV->UV', eta1['a'], h['a'][c,a], t['ab'][pa,pA,hc,hA], optimize=True)
    O['b'][A,A] += scale * +1.00000000 * np.einsum('uv,ua,aUvV->UV', gamma1['a'], h['a'][a,v], t['ab'][pv,pA,ha,hA], optimize=True)
    O['b'][A,A] += scale * +1.00000000 * np.einsum('ia,aUiV->UV', h['a'][c,v], t['ab'][pv,pA,hc,hA], optimize=True)
    O['b'][V,A] += scale * -1.00000000 * np.einsum('uv,wx,ux,wAvU->AU', eta1['a'], gamma1['a'], h['a'][a,a], t['ab'][pa,pV,ha,hA], optimize=True)
    O['b'][V,A] += scale * +1.00000000 * np.einsum('uv,wx,wv,uAxU->AU', eta1['a'], gamma1['a'], h['a'][a,a], t['ab'][pa,pV,ha,hA], optimize=True)
    O['b'][V,A] += scale * +1.00000000 * np.einsum('uv,iv,uAiU->AU', eta1['a'], h['a'][c,a], t['ab'][pa,pV,hc,hA], optimize=True)
    O['b'][V,A] += scale * +1.00000000 * np.einsum('uv,ua,aAvU->AU', gamma1['a'], h['a'][a,v], t['ab'][pv,pV,ha,hA], optimize=True)
    O['b'][V,A] += scale * +1.00000000 * np.einsum('ia,aAiU->AU', h['a'][c,v], t['ab'][pv,pV,hc,hA], optimize=True)
    O['b'][A,C] += scale * -1.00000000 * np.einsum('uv,wx,ux,wUvI->UI', eta1['a'], gamma1['a'], h['a'][a,a], t['ab'][pa,pA,ha,hC], optimize=True)
    O['b'][A,C] += scale * +1.00000000 * np.einsum('uv,wx,wv,uUxI->UI', eta1['a'], gamma1['a'], h['a'][a,a], t['ab'][pa,pA,ha,hC], optimize=True)
    O['b'][A,C] += scale * +1.00000000 * np.einsum('uv,iv,uUiI->UI', eta1['a'], h['a'][c,a], t['ab'][pa,pA,hc,hC], optimize=True)
    O['b'][A,C] += scale * +1.00000000 * np.einsum('uv,ua,aUvI->UI', gamma1['a'], h['a'][a,v], t['ab'][pv,pA,ha,hC], optimize=True)
    O['b'][A,C] += scale * +1.00000000 * np.einsum('ia,aUiI->UI', h['a'][c,v], t['ab'][pv,pA,hc,hC], optimize=True)
    O['b'][V,C] += scale * -1.00000000 * np.einsum('uv,wx,ux,wAvI->AI', eta1['a'], gamma1['a'], h['a'][a,a], t['ab'][pa,pV,ha,hC], optimize=True)
    O['b'][V,C] += scale * +1.00000000 * np.einsum('uv,wx,wv,uAxI->AI', eta1['a'], gamma1['a'], h['a'][a,a], t['ab'][pa,pV,ha,hC], optimize=True)
    O['b'][V,C] += scale * +1.00000000 * np.einsum('uv,iv,uAiI->AI', eta1['a'], h['a'][c,a], t['ab'][pa,pV,hc,hC], optimize=True)
    O['b'][V,C] += scale * +1.00000000 * np.einsum('uv,ua,aAvI->AI', gamma1['a'], h['a'][a,v], t['ab'][pv,pV,ha,hC], optimize=True)
    O['b'][V,C] += scale * +1.00000000 * np.einsum('ia,aAiI->AI', h['a'][c,v], t['ab'][pv,pV,hc,hC], optimize=True)

    t1 = time.time()
    if verbose: print("h1a_t2b_c1b took {:.4f} seconds to run.".format(t1-t0))

    return O
def h1b_t1b_c1b(O, h, t, gamma1, eta1, lambdas, orbspace, verbose=False, scale=1.0):
    # 36 lines
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

    
    
    
    
    O['b'][A,A] += scale * -1.00000000 * np.einsum('IU,VI->VU', h['b'][C,A], t['b'][pA,hC], optimize=True)
    O['b'][A,A] += scale * +1.00000000 * np.einsum('UA,AV->UV', h['b'][A,V], t['b'][pV,hA], optimize=True)
    
    
    O['b'][C,A] += scale * +1.00000000 * np.einsum('IA,AU->IU', h['b'][C,V], t['b'][pV,hA], optimize=True)
    O['b'][V,A] += scale * -1.00000000 * np.einsum('UV,UW,AV->AW', eta1['b'], h['b'][A,A], t['b'][pV,hA], optimize=True)
    
    O['b'][V,A] += scale * -1.00000000 * np.einsum('UV,UW,AV->AW', gamma1['b'], h['b'][A,A], t['b'][pV,hA], optimize=True)
    
    O['b'][V,A] += scale * -1.00000000 * np.einsum('IU,AI->AU', h['b'][C,A], t['b'][pV,hC], optimize=True)
    O['b'][V,A] += scale * +1.00000000 * np.einsum('AB,BU->AU', h['b'][V,V], t['b'][pV,hA], optimize=True)
    O['b'][A,C] += scale * +1.00000000 * np.einsum('UV,WV,UI->WI', eta1['b'], h['b'][A,A], t['b'][pA,hC], optimize=True)
    
    O['b'][A,C] += scale * +1.00000000 * np.einsum('UV,WV,UI->WI', gamma1['b'], h['b'][A,A], t['b'][pA,hC], optimize=True)
    
    O['b'][A,C] += scale * -1.00000000 * np.einsum('IJ,UI->UJ', h['b'][C,C], t['b'][pA,hC], optimize=True)
    O['b'][A,C] += scale * +1.00000000 * np.einsum('UA,AI->UI', h['b'][A,V], t['b'][pV,hC], optimize=True)
    O['b'][C,C] += scale * +1.00000000 * np.einsum('UV,IV,UJ->IJ', eta1['b'], h['b'][C,A], t['b'][pA,hC], optimize=True)
    O['b'][C,C] += scale * +1.00000000 * np.einsum('UV,IV,UJ->IJ', gamma1['b'], h['b'][C,A], t['b'][pA,hC], optimize=True)
    O['b'][C,C] += scale * +1.00000000 * np.einsum('IA,AJ->IJ', h['b'][C,V], t['b'][pV,hC], optimize=True)
    O['b'][V,C] += scale * -1.00000000 * np.einsum('UV,UI,AV->AI', eta1['b'], h['b'][A,C], t['b'][pV,hA], optimize=True)
    O['b'][V,C] += scale * +1.00000000 * np.einsum('UV,AV,UI->AI', eta1['b'], h['b'][V,A], t['b'][pA,hC], optimize=True)
    O['b'][V,C] += scale * -1.00000000 * np.einsum('UV,UI,AV->AI', gamma1['b'], h['b'][A,C], t['b'][pV,hA], optimize=True)
    O['b'][V,C] += scale * +1.00000000 * np.einsum('UV,AV,UI->AI', gamma1['b'], h['b'][V,A], t['b'][pA,hC], optimize=True)
    O['b'][V,C] += scale * -1.00000000 * np.einsum('IJ,AI->AJ', h['b'][C,C], t['b'][pV,hC], optimize=True)
    O['b'][V,C] += scale * +1.00000000 * np.einsum('AB,BI->AI', h['b'][V,V], t['b'][pV,hC], optimize=True)
    
    
    O['b'][A,V] += scale * -1.00000000 * np.einsum('IA,UI->UA', h['b'][C,V], t['b'][pA,hC], optimize=True)
    O['b'][V,V] += scale * -1.00000000 * np.einsum('UV,UA,BV->BA', eta1['b'], h['b'][A,V], t['b'][pV,hA], optimize=True)
    O['b'][V,V] += scale * -1.00000000 * np.einsum('UV,UA,BV->BA', gamma1['b'], h['b'][A,V], t['b'][pV,hA], optimize=True)
    O['b'][V,V] += scale * -1.00000000 * np.einsum('IA,BI->BA', h['b'][C,V], t['b'][pV,hC], optimize=True)

    t1 = time.time()
    if verbose: print("h1b_t1b_c1b took {:.4f} seconds to run.".format(t1-t0))

    return O
def h1b_t2c_c1b(O, h, t, gamma1, eta1, lambdas, orbspace, verbose=False, scale=1.0):
    # 20 lines
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

    
    
    O['b'][A,A] += scale * -1.00000000 * np.einsum('UV,IV,WUIX->WX', eta1['b'], h['b'][C,A], t['bb'][pA,pA,hC,hA], optimize=True)
    O['b'][A,A] += scale * +1.00000000 * np.einsum('UV,UA,WAXV->WX', gamma1['b'], h['b'][A,V], t['bb'][pA,pV,hA,hA], optimize=True)
    O['b'][A,A] += scale * -1.00000000 * np.einsum('IA,UAIV->UV', h['b'][C,V], t['bb'][pA,pV,hC,hA], optimize=True)
    O['b'][V,A] += scale * +1.00000000 * np.einsum('UV,WX,UX,WAYV->AY', eta1['b'], gamma1['b'], h['b'][A,A], t['bb'][pA,pV,hA,hA], optimize=True)
    O['b'][V,A] += scale * -1.00000000 * np.einsum('UV,WX,WV,UAYX->AY', eta1['b'], gamma1['b'], h['b'][A,A], t['bb'][pA,pV,hA,hA], optimize=True)
    O['b'][V,A] += scale * +1.00000000 * np.einsum('UV,IV,UAIW->AW', eta1['b'], h['b'][C,A], t['bb'][pA,pV,hC,hA], optimize=True)
    O['b'][V,A] += scale * +1.00000000 * np.einsum('UV,UA,BAWV->BW', gamma1['b'], h['b'][A,V], t['bb'][pV,pV,hA,hA], optimize=True)
    O['b'][V,A] += scale * -1.00000000 * np.einsum('IA,BAIU->BU', h['b'][C,V], t['bb'][pV,pV,hC,hA], optimize=True)
    O['b'][A,C] += scale * -1.00000000 * np.einsum('UV,WX,UX,YWIV->YI', eta1['b'], gamma1['b'], h['b'][A,A], t['bb'][pA,pA,hC,hA], optimize=True)
    O['b'][A,C] += scale * +1.00000000 * np.einsum('UV,WX,WV,YUIX->YI', eta1['b'], gamma1['b'], h['b'][A,A], t['bb'][pA,pA,hC,hA], optimize=True)
    O['b'][A,C] += scale * +1.00000000 * np.einsum('UV,IV,WUJI->WJ', eta1['b'], h['b'][C,A], t['bb'][pA,pA,hC,hC], optimize=True)
    O['b'][A,C] += scale * +1.00000000 * np.einsum('UV,UA,WAIV->WI', gamma1['b'], h['b'][A,V], t['bb'][pA,pV,hC,hA], optimize=True)
    O['b'][A,C] += scale * +1.00000000 * np.einsum('IA,UAJI->UJ', h['b'][C,V], t['bb'][pA,pV,hC,hC], optimize=True)
    O['b'][V,C] += scale * +1.00000000 * np.einsum('UV,WX,UX,WAIV->AI', eta1['b'], gamma1['b'], h['b'][A,A], t['bb'][pA,pV,hC,hA], optimize=True)
    O['b'][V,C] += scale * -1.00000000 * np.einsum('UV,WX,WV,UAIX->AI', eta1['b'], gamma1['b'], h['b'][A,A], t['bb'][pA,pV,hC,hA], optimize=True)
    O['b'][V,C] += scale * -1.00000000 * np.einsum('UV,IV,UAJI->AJ', eta1['b'], h['b'][C,A], t['bb'][pA,pV,hC,hC], optimize=True)
    O['b'][V,C] += scale * +1.00000000 * np.einsum('UV,UA,BAIV->BI', gamma1['b'], h['b'][A,V], t['bb'][pV,pV,hC,hA], optimize=True)
    O['b'][V,C] += scale * +1.00000000 * np.einsum('IA,BAJI->BJ', h['b'][C,V], t['bb'][pV,pV,hC,hC], optimize=True)

    t1 = time.time()
    if verbose: print("h1b_t2c_c1b took {:.4f} seconds to run.".format(t1-t0))

    return O
def h2a_t2b_c1b(O, h, t, gamma1, eta1, lambdas, orbspace, verbose=False, scale=1.0):
    # 24 lines
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

    
    
    
    
    O['b'][A,A] += scale * +0.50000000 * np.einsum('iuvw,uxvw,xUiV->UV', h['aa'][c,a,a,a], lambdas['aa'], t['ab'][pa,pA,hc,hA], optimize=True)
    O['b'][A,A] += scale * +0.50000000 * np.einsum('uvwa,uvwx,aUxV->UV', h['aa'][a,a,a,v], lambdas['aa'], t['ab'][pv,pA,ha,hA], optimize=True)
    O['b'][V,A] += scale * +0.50000000 * np.einsum('uv,uwxy,wzxy,zAvU->AU', eta1['a'], h['aa'][a,a,a,a], lambdas['aa'], t['ab'][pa,pV,ha,hA], optimize=True)
    O['b'][V,A] += scale * -0.50000000 * np.einsum('uv,wxvy,wxyz,uAzU->AU', eta1['a'], h['aa'][a,a,a,a], lambdas['aa'], t['ab'][pa,pV,ha,hA], optimize=True)
    O['b'][V,A] += scale * +0.50000000 * np.einsum('uv,uwxy,wzxy,zAvU->AU', gamma1['a'], h['aa'][a,a,a,a], lambdas['aa'], t['ab'][pa,pV,ha,hA], optimize=True)
    O['b'][V,A] += scale * -0.50000000 * np.einsum('uv,wxvy,wxyz,uAzU->AU', gamma1['a'], h['aa'][a,a,a,a], lambdas['aa'], t['ab'][pa,pV,ha,hA], optimize=True)
    O['b'][V,A] += scale * +0.50000000 * np.einsum('iuvw,uxvw,xAiU->AU', h['aa'][c,a,a,a], lambdas['aa'], t['ab'][pa,pV,hc,hA], optimize=True)
    O['b'][V,A] += scale * +0.50000000 * np.einsum('uvwa,uvwx,aAxU->AU', h['aa'][a,a,a,v], lambdas['aa'], t['ab'][pv,pV,ha,hA], optimize=True)
    O['b'][A,C] += scale * +0.50000000 * np.einsum('uv,uwxy,wzxy,zUvI->UI', eta1['a'], h['aa'][a,a,a,a], lambdas['aa'], t['ab'][pa,pA,ha,hC], optimize=True)
    O['b'][A,C] += scale * -0.50000000 * np.einsum('uv,wxvy,wxyz,uUzI->UI', eta1['a'], h['aa'][a,a,a,a], lambdas['aa'], t['ab'][pa,pA,ha,hC], optimize=True)
    O['b'][A,C] += scale * +0.50000000 * np.einsum('uv,uwxy,wzxy,zUvI->UI', gamma1['a'], h['aa'][a,a,a,a], lambdas['aa'], t['ab'][pa,pA,ha,hC], optimize=True)
    O['b'][A,C] += scale * -0.50000000 * np.einsum('uv,wxvy,wxyz,uUzI->UI', gamma1['a'], h['aa'][a,a,a,a], lambdas['aa'], t['ab'][pa,pA,ha,hC], optimize=True)
    O['b'][A,C] += scale * +0.50000000 * np.einsum('iuvw,uxvw,xUiI->UI', h['aa'][c,a,a,a], lambdas['aa'], t['ab'][pa,pA,hc,hC], optimize=True)
    O['b'][A,C] += scale * +0.50000000 * np.einsum('uvwa,uvwx,aUxI->UI', h['aa'][a,a,a,v], lambdas['aa'], t['ab'][pv,pA,ha,hC], optimize=True)
    O['b'][V,C] += scale * +0.50000000 * np.einsum('uv,uwxy,wzxy,zAvI->AI', eta1['a'], h['aa'][a,a,a,a], lambdas['aa'], t['ab'][pa,pV,ha,hC], optimize=True)
    O['b'][V,C] += scale * -0.50000000 * np.einsum('uv,wxvy,wxyz,uAzI->AI', eta1['a'], h['aa'][a,a,a,a], lambdas['aa'], t['ab'][pa,pV,ha,hC], optimize=True)
    O['b'][V,C] += scale * +0.50000000 * np.einsum('uv,uwxy,wzxy,zAvI->AI', gamma1['a'], h['aa'][a,a,a,a], lambdas['aa'], t['ab'][pa,pV,ha,hC], optimize=True)
    O['b'][V,C] += scale * -0.50000000 * np.einsum('uv,wxvy,wxyz,uAzI->AI', gamma1['a'], h['aa'][a,a,a,a], lambdas['aa'], t['ab'][pa,pV,ha,hC], optimize=True)
    O['b'][V,C] += scale * +0.50000000 * np.einsum('iuvw,uxvw,xAiI->AI', h['aa'][c,a,a,a], lambdas['aa'], t['ab'][pa,pV,hc,hC], optimize=True)
    O['b'][V,C] += scale * +0.50000000 * np.einsum('uvwa,uvwx,aAxI->AI', h['aa'][a,a,a,v], lambdas['aa'], t['ab'][pv,pV,ha,hC], optimize=True)

    t1 = time.time()
    if verbose: print("h2a_t2b_c1b took {:.4f} seconds to run.".format(t1-t0))

    return O
def h2b_t1a_c1b(O, h, t, gamma1, eta1, lambdas, orbspace, verbose=False, scale=1.0):
    # 45 lines
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

    
    
    O['b'][A,A] += scale * +1.00000000 * np.einsum('uv,iUvV,ui->UV', eta1['a'], h['ab'][c,A,a,A], t['a'][pa,hc], optimize=True)
    O['b'][A,A] += scale * +1.00000000 * np.einsum('uv,uUaV,av->UV', gamma1['a'], h['ab'][a,A,v,A], t['a'][pv,ha], optimize=True)
    O['b'][A,A] += scale * +1.00000000 * np.einsum('iUaV,ai->UV', h['ab'][c,A,v,A], t['a'][pv,hc], optimize=True)
    
    
    O['b'][C,A] += scale * +1.00000000 * np.einsum('uv,iIvU,ui->IU', eta1['a'], h['ab'][c,C,a,A], t['a'][pa,hc], optimize=True)
    O['b'][C,A] += scale * +1.00000000 * np.einsum('uv,uIaU,av->IU', gamma1['a'], h['ab'][a,C,v,A], t['a'][pv,ha], optimize=True)
    O['b'][C,A] += scale * +1.00000000 * np.einsum('iIaU,ai->IU', h['ab'][c,C,v,A], t['a'][pv,hc], optimize=True)
    
    
    O['b'][V,A] += scale * +1.00000000 * np.einsum('uv,iAvU,ui->AU', eta1['a'], h['ab'][c,V,a,A], t['a'][pa,hc], optimize=True)
    O['b'][V,A] += scale * +1.00000000 * np.einsum('uv,uAaU,av->AU', gamma1['a'], h['ab'][a,V,v,A], t['a'][pv,ha], optimize=True)
    O['b'][V,A] += scale * +1.00000000 * np.einsum('iAaU,ai->AU', h['ab'][c,V,v,A], t['a'][pv,hc], optimize=True)
    
    
    O['b'][A,C] += scale * +1.00000000 * np.einsum('uv,iUvI,ui->UI', eta1['a'], h['ab'][c,A,a,C], t['a'][pa,hc], optimize=True)
    O['b'][A,C] += scale * +1.00000000 * np.einsum('uv,uUaI,av->UI', gamma1['a'], h['ab'][a,A,v,C], t['a'][pv,ha], optimize=True)
    O['b'][A,C] += scale * +1.00000000 * np.einsum('iUaI,ai->UI', h['ab'][c,A,v,C], t['a'][pv,hc], optimize=True)
    
    
    O['b'][C,C] += scale * +1.00000000 * np.einsum('uv,iIvJ,ui->IJ', eta1['a'], h['ab'][c,C,a,C], t['a'][pa,hc], optimize=True)
    O['b'][C,C] += scale * +1.00000000 * np.einsum('uv,uIaJ,av->IJ', gamma1['a'], h['ab'][a,C,v,C], t['a'][pv,ha], optimize=True)
    O['b'][C,C] += scale * +1.00000000 * np.einsum('iIaJ,ai->IJ', h['ab'][c,C,v,C], t['a'][pv,hc], optimize=True)
    
    
    O['b'][V,C] += scale * +1.00000000 * np.einsum('uv,iAvI,ui->AI', eta1['a'], h['ab'][c,V,a,C], t['a'][pa,hc], optimize=True)
    O['b'][V,C] += scale * +1.00000000 * np.einsum('uv,uAaI,av->AI', gamma1['a'], h['ab'][a,V,v,C], t['a'][pv,ha], optimize=True)
    O['b'][V,C] += scale * +1.00000000 * np.einsum('iAaI,ai->AI', h['ab'][c,V,v,C], t['a'][pv,hc], optimize=True)
    
    
    O['b'][A,V] += scale * +1.00000000 * np.einsum('uv,iUvA,ui->UA', eta1['a'], h['ab'][c,A,a,V], t['a'][pa,hc], optimize=True)
    O['b'][A,V] += scale * +1.00000000 * np.einsum('uv,uUaA,av->UA', gamma1['a'], h['ab'][a,A,v,V], t['a'][pv,ha], optimize=True)
    O['b'][A,V] += scale * +1.00000000 * np.einsum('iUaA,ai->UA', h['ab'][c,A,v,V], t['a'][pv,hc], optimize=True)
    
    
    O['b'][C,V] += scale * +1.00000000 * np.einsum('uv,iIvA,ui->IA', eta1['a'], h['ab'][c,C,a,V], t['a'][pa,hc], optimize=True)
    O['b'][C,V] += scale * +1.00000000 * np.einsum('uv,uIaA,av->IA', gamma1['a'], h['ab'][a,C,v,V], t['a'][pv,ha], optimize=True)
    O['b'][C,V] += scale * +1.00000000 * np.einsum('iIaA,ai->IA', h['ab'][c,C,v,V], t['a'][pv,hc], optimize=True)
    
    
    O['b'][V,V] += scale * +1.00000000 * np.einsum('uv,iAvB,ui->AB', eta1['a'], h['ab'][c,V,a,V], t['a'][pa,hc], optimize=True)
    O['b'][V,V] += scale * +1.00000000 * np.einsum('uv,uAaB,av->AB', gamma1['a'], h['ab'][a,V,v,V], t['a'][pv,ha], optimize=True)
    O['b'][V,V] += scale * +1.00000000 * np.einsum('iAaB,ai->AB', h['ab'][c,V,v,V], t['a'][pv,hc], optimize=True)

    t1 = time.time()
    if verbose: print("h2b_t1a_c1b took {:.4f} seconds to run.".format(t1-t0))

    return O
def h2b_t2a_c1b(O, h, t, gamma1, eta1, lambdas, orbspace, verbose=False, scale=1.0):
    # 54 lines
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

    
    
    
    
    O['b'][A,A] += scale * -0.50000000 * np.einsum('iUuV,vwux,vwix->UV', h['ab'][c,A,a,A], lambdas['aa'], t['aa'][pa,pa,hc,ha], optimize=True)
    O['b'][A,A] += scale * -0.50000000 * np.einsum('uUaV,uvwx,vawx->UV', h['ab'][a,A,v,A], lambdas['aa'], t['aa'][pa,pv,ha,ha], optimize=True)
    
    
    
    
    O['b'][C,A] += scale * -0.50000000 * np.einsum('iIuU,vwux,vwix->IU', h['ab'][c,C,a,A], lambdas['aa'], t['aa'][pa,pa,hc,ha], optimize=True)
    O['b'][C,A] += scale * -0.50000000 * np.einsum('uIaU,uvwx,vawx->IU', h['ab'][a,C,v,A], lambdas['aa'], t['aa'][pa,pv,ha,ha], optimize=True)
    
    
    
    
    O['b'][V,A] += scale * -0.50000000 * np.einsum('iAuU,vwux,vwix->AU', h['ab'][c,V,a,A], lambdas['aa'], t['aa'][pa,pa,hc,ha], optimize=True)
    O['b'][V,A] += scale * -0.50000000 * np.einsum('uAaU,uvwx,vawx->AU', h['ab'][a,V,v,A], lambdas['aa'], t['aa'][pa,pv,ha,ha], optimize=True)
    
    
    
    
    O['b'][A,C] += scale * -0.50000000 * np.einsum('iUuI,vwux,vwix->UI', h['ab'][c,A,a,C], lambdas['aa'], t['aa'][pa,pa,hc,ha], optimize=True)
    O['b'][A,C] += scale * -0.50000000 * np.einsum('uUaI,uvwx,vawx->UI', h['ab'][a,A,v,C], lambdas['aa'], t['aa'][pa,pv,ha,ha], optimize=True)
    
    
    
    
    O['b'][C,C] += scale * -0.50000000 * np.einsum('iIuJ,vwux,vwix->IJ', h['ab'][c,C,a,C], lambdas['aa'], t['aa'][pa,pa,hc,ha], optimize=True)
    O['b'][C,C] += scale * -0.50000000 * np.einsum('uIaJ,uvwx,vawx->IJ', h['ab'][a,C,v,C], lambdas['aa'], t['aa'][pa,pv,ha,ha], optimize=True)
    
    
    
    
    O['b'][V,C] += scale * -0.50000000 * np.einsum('iAuI,vwux,vwix->AI', h['ab'][c,V,a,C], lambdas['aa'], t['aa'][pa,pa,hc,ha], optimize=True)
    O['b'][V,C] += scale * -0.50000000 * np.einsum('uAaI,uvwx,vawx->AI', h['ab'][a,V,v,C], lambdas['aa'], t['aa'][pa,pv,ha,ha], optimize=True)
    
    
    
    
    O['b'][A,V] += scale * -0.50000000 * np.einsum('iUuA,vwux,vwix->UA', h['ab'][c,A,a,V], lambdas['aa'], t['aa'][pa,pa,hc,ha], optimize=True)
    O['b'][A,V] += scale * -0.50000000 * np.einsum('uUaA,uvwx,vawx->UA', h['ab'][a,A,v,V], lambdas['aa'], t['aa'][pa,pv,ha,ha], optimize=True)
    
    
    
    
    O['b'][C,V] += scale * -0.50000000 * np.einsum('iIuA,vwux,vwix->IA', h['ab'][c,C,a,V], lambdas['aa'], t['aa'][pa,pa,hc,ha], optimize=True)
    O['b'][C,V] += scale * -0.50000000 * np.einsum('uIaA,uvwx,vawx->IA', h['ab'][a,C,v,V], lambdas['aa'], t['aa'][pa,pv,ha,ha], optimize=True)
    
    
    
    
    O['b'][V,V] += scale * -0.50000000 * np.einsum('iAuB,vwux,vwix->AB', h['ab'][c,V,a,V], lambdas['aa'], t['aa'][pa,pa,hc,ha], optimize=True)
    O['b'][V,V] += scale * -0.50000000 * np.einsum('uAaB,uvwx,vawx->AB', h['ab'][a,V,v,V], lambdas['aa'], t['aa'][pa,pv,ha,ha], optimize=True)

    t1 = time.time()
    if verbose: print("h2b_t2a_c1b took {:.4f} seconds to run.".format(t1-t0))

    return O
def h2b_t2b_c1b(O, h, t, gamma1, eta1, lambdas, orbspace, verbose=False, scale=1.0):
    # 294 lines
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

    O['b'][A,A] += scale * -1.00000000 * np.einsum('uv,wx,wIvU,uVxI->VU', eta1['a'], gamma1['a'], h['ab'][a,C,a,A], t['ab'][pa,pA,ha,hC], optimize=True)
    O['b'][A,A] += scale * +1.00000000 * np.einsum('uv,wx,wUvA,uAxV->UV', eta1['a'], gamma1['a'], h['ab'][a,A,a,V], t['ab'][pa,pV,ha,hA], optimize=True)
    
    
    O['b'][A,A] += scale * -1.00000000 * np.einsum('uv,UV,iUvW,uXiV->XW', eta1['a'], gamma1['b'], h['ab'][c,A,a,A], t['ab'][pa,pA,hc,hA], optimize=True)
    O['b'][A,A] += scale * -1.00000000 * np.einsum('uv,iIvU,uViI->VU', eta1['a'], h['ab'][c,C,a,A], t['ab'][pa,pA,hc,hC], optimize=True)
    O['b'][A,A] += scale * +1.00000000 * np.einsum('uv,iUvA,uAiV->UV', eta1['a'], h['ab'][c,A,a,V], t['ab'][pa,pV,hc,hA], optimize=True)
    
    
    
    
    
    
    
    
    
    
    O['b'][A,A] += scale * +1.00000000 * np.einsum('UV,uv,iWvV,uUiX->WX', eta1['b'], eta1['a'], h['ab'][c,A,a,A], t['ab'][pa,pA,hc,hA], optimize=True)
    O['b'][A,A] += scale * +1.00000000 * np.einsum('UV,uv,uWaV,aUvX->WX', eta1['b'], gamma1['a'], h['ab'][a,A,v,A], t['ab'][pv,pA,ha,hA], optimize=True)
    O['b'][A,A] += scale * +1.00000000 * np.einsum('UV,iWaV,aUiX->WX', eta1['b'], h['ab'][c,A,v,A], t['ab'][pv,pA,hc,hA], optimize=True)
    
    
    O['b'][A,A] += scale * -1.00000000 * np.einsum('uv,uIaU,aVvI->VU', gamma1['a'], h['ab'][a,C,v,A], t['ab'][pv,pA,ha,hC], optimize=True)
    
    
    O['b'][A,A] += scale * +1.00000000 * np.einsum('uv,uUaA,aAvV->UV', gamma1['a'], h['ab'][a,A,v,V], t['ab'][pv,pV,ha,hA], optimize=True)
    
    
    
    
    
    
    O['b'][A,A] += scale * -1.00000000 * np.einsum('UV,uv,uUaW,aXvV->XW', gamma1['b'], gamma1['a'], h['ab'][a,A,v,A], t['ab'][pv,pA,ha,hA], optimize=True)
    O['b'][A,A] += scale * -1.00000000 * np.einsum('UV,iUaW,aXiV->XW', gamma1['b'], h['ab'][c,A,v,A], t['ab'][pv,pA,hc,hA], optimize=True)
    
    
    O['b'][A,A] += scale * -1.00000000 * np.einsum('iIaU,aViI->VU', h['ab'][c,C,v,A], t['ab'][pv,pA,hc,hC], optimize=True)
    O['b'][A,A] += scale * -1.00000000 * np.einsum('iUuV,vWuX,vWiX->UV', h['ab'][c,A,a,A], lambdas['ab'], t['ab'][pa,pA,hc,hA], optimize=True)
    O['b'][A,A] += scale * +1.00000000 * np.einsum('iUuV,vWuV,vWiX->UX', h['ab'][c,A,a,A], lambdas['ab'], t['ab'][pa,pA,hc,hA], optimize=True)
    O['b'][A,A] += scale * +1.00000000 * np.einsum('iUaA,aAiV->UV', h['ab'][c,A,v,V], t['ab'][pv,pV,hc,hA], optimize=True)
    O['b'][A,A] += scale * +1.00000000 * np.einsum('iUuV,vUuW,vXiW->XV', h['ab'][c,A,a,A], lambdas['ab'], t['ab'][pa,pA,hc,hA], optimize=True)
    O['b'][A,A] += scale * -1.00000000 * np.einsum('iUuV,vUuV,vWiX->WX', h['ab'][c,A,a,A], lambdas['ab'], t['ab'][pa,pA,hc,hA], optimize=True)
    O['b'][A,A] += scale * -1.00000000 * np.einsum('uIvU,uwvx,wVxI->VU', h['ab'][a,C,a,A], lambdas['aa'], t['ab'][pa,pA,ha,hC], optimize=True)
    O['b'][A,A] += scale * +1.00000000 * np.einsum('uUvA,uwvx,wAxV->UV', h['ab'][a,A,a,V], lambdas['aa'], t['ab'][pa,pV,ha,hA], optimize=True)
    O['b'][A,A] += scale * +1.00000000 * np.einsum('uUaV,uWvX,aWvX->UV', h['ab'][a,A,v,A], lambdas['ab'], t['ab'][pv,pA,ha,hA], optimize=True)
    O['b'][A,A] += scale * -1.00000000 * np.einsum('uUaV,uWvV,aWvX->UX', h['ab'][a,A,v,A], lambdas['ab'], t['ab'][pv,pA,ha,hA], optimize=True)
    O['b'][A,A] += scale * -1.00000000 * np.einsum('uUaV,uUvW,aXvW->XV', h['ab'][a,A,v,A], lambdas['ab'], t['ab'][pv,pA,ha,hA], optimize=True)
    O['b'][A,A] += scale * +1.00000000 * np.einsum('uUaV,uUvV,aWvX->WX', h['ab'][a,A,v,A], lambdas['ab'], t['ab'][pv,pA,ha,hA], optimize=True)
    O['b'][C,A] += scale * +1.00000000 * np.einsum('uv,wx,wIvA,uAxU->IU', eta1['a'], gamma1['a'], h['ab'][a,C,a,V], t['ab'][pa,pV,ha,hA], optimize=True)
    
    O['b'][C,A] += scale * +1.00000000 * np.einsum('uv,iIvA,uAiU->IU', eta1['a'], h['ab'][c,C,a,V], t['ab'][pa,pV,hc,hA], optimize=True)
    
    
    
    
    
    O['b'][C,A] += scale * +1.00000000 * np.einsum('UV,uv,iIvV,uUiW->IW', eta1['b'], eta1['a'], h['ab'][c,C,a,A], t['ab'][pa,pA,hc,hA], optimize=True)
    O['b'][C,A] += scale * +1.00000000 * np.einsum('UV,uv,uIaV,aUvW->IW', eta1['b'], gamma1['a'], h['ab'][a,C,v,A], t['ab'][pv,pA,ha,hA], optimize=True)
    O['b'][C,A] += scale * +1.00000000 * np.einsum('UV,iIaV,aUiW->IW', eta1['b'], h['ab'][c,C,v,A], t['ab'][pv,pA,hc,hA], optimize=True)
    
    
    
    O['b'][C,A] += scale * +1.00000000 * np.einsum('uv,uIaA,aAvU->IU', gamma1['a'], h['ab'][a,C,v,V], t['ab'][pv,pV,ha,hA], optimize=True)
    
    
    
    O['b'][C,A] += scale * -1.00000000 * np.einsum('iIuU,vVuW,vViW->IU', h['ab'][c,C,a,A], lambdas['ab'], t['ab'][pa,pA,hc,hA], optimize=True)
    O['b'][C,A] += scale * +1.00000000 * np.einsum('iIuU,vVuU,vViW->IW', h['ab'][c,C,a,A], lambdas['ab'], t['ab'][pa,pA,hc,hA], optimize=True)
    O['b'][C,A] += scale * +1.00000000 * np.einsum('iIaA,aAiU->IU', h['ab'][c,C,v,V], t['ab'][pv,pV,hc,hA], optimize=True)
    O['b'][C,A] += scale * +1.00000000 * np.einsum('uIvA,uwvx,wAxU->IU', h['ab'][a,C,a,V], lambdas['aa'], t['ab'][pa,pV,ha,hA], optimize=True)
    O['b'][C,A] += scale * +1.00000000 * np.einsum('uIaU,uVvW,aVvW->IU', h['ab'][a,C,v,A], lambdas['ab'], t['ab'][pv,pA,ha,hA], optimize=True)
    O['b'][C,A] += scale * -1.00000000 * np.einsum('uIaU,uVvU,aVvW->IW', h['ab'][a,C,v,A], lambdas['ab'], t['ab'][pv,pA,ha,hA], optimize=True)
    O['b'][V,A] += scale * -1.00000000 * np.einsum('uv,wx,wIvU,uAxI->AU', eta1['a'], gamma1['a'], h['ab'][a,C,a,A], t['ab'][pa,pV,ha,hC], optimize=True)
    O['b'][V,A] += scale * +1.00000000 * np.einsum('uv,wx,wAvB,uBxU->AU', eta1['a'], gamma1['a'], h['ab'][a,V,a,V], t['ab'][pa,pV,ha,hA], optimize=True)
    
    O['b'][V,A] += scale * -1.00000000 * np.einsum('uv,UV,wx,wUvW,uAxV->AW', eta1['a'], gamma1['b'], gamma1['a'], h['ab'][a,A,a,A], t['ab'][pa,pV,ha,hA], optimize=True)
    O['b'][V,A] += scale * -1.00000000 * np.einsum('uv,UV,iUvW,uAiV->AW', eta1['a'], gamma1['b'], h['ab'][c,A,a,A], t['ab'][pa,pV,hc,hA], optimize=True)
    O['b'][V,A] += scale * -1.00000000 * np.einsum('uv,iIvU,uAiI->AU', eta1['a'], h['ab'][c,C,a,A], t['ab'][pa,pV,hc,hC], optimize=True)
    O['b'][V,A] += scale * +1.00000000 * np.einsum('uv,iAvB,uBiU->AU', eta1['a'], h['ab'][c,V,a,V], t['ab'][pa,pV,hc,hA], optimize=True)
    O['b'][V,A] += scale * +1.00000000 * np.einsum('uv,uUwV,xUwW,xAvW->AV', eta1['a'], h['ab'][a,A,a,A], lambdas['ab'], t['ab'][pa,pV,ha,hA], optimize=True)
    O['b'][V,A] += scale * -1.00000000 * np.einsum('uv,uUwV,xUwV,xAvW->AW', eta1['a'], h['ab'][a,A,a,A], lambdas['ab'], t['ab'][pa,pV,ha,hA], optimize=True)
    
    
    O['b'][V,A] += scale * -1.00000000 * np.einsum('uv,wUvV,wUxW,uAxW->AV', eta1['a'], h['ab'][a,A,a,A], lambdas['ab'], t['ab'][pa,pV,ha,hA], optimize=True)
    O['b'][V,A] += scale * +1.00000000 * np.einsum('uv,wUvV,wUxV,uAxW->AW', eta1['a'], h['ab'][a,A,a,A], lambdas['ab'], t['ab'][pa,pV,ha,hA], optimize=True)
    
    
    O['b'][V,A] += scale * -1.00000000 * np.einsum('UV,uv,wx,uUxW,wAvV->AW', eta1['b'], eta1['a'], gamma1['a'], h['ab'][a,A,a,A], t['ab'][pa,pV,ha,hA], optimize=True)
    
    O['b'][V,A] += scale * +1.00000000 * np.einsum('UV,uv,iAvV,uUiW->AW', eta1['b'], eta1['a'], h['ab'][c,V,a,A], t['ab'][pa,pA,hc,hA], optimize=True)
    O['b'][V,A] += scale * +1.00000000 * np.einsum('UV,uv,uAaV,aUvW->AW', eta1['b'], gamma1['a'], h['ab'][a,V,v,A], t['ab'][pv,pA,ha,hA], optimize=True)
    O['b'][V,A] += scale * +1.00000000 * np.einsum('UV,iAaV,aUiW->AW', eta1['b'], h['ab'][c,V,v,A], t['ab'][pv,pA,hc,hA], optimize=True)
    O['b'][V,A] += scale * -1.00000000 * np.einsum('UV,uUvW,uwvx,wAxV->AW', eta1['b'], h['ab'][a,A,a,A], lambdas['aa'], t['ab'][pa,pV,ha,hA], optimize=True)
    
    O['b'][V,A] += scale * -1.00000000 * np.einsum('uv,uIaU,aAvI->AU', gamma1['a'], h['ab'][a,C,v,A], t['ab'][pv,pV,ha,hC], optimize=True)
    O['b'][V,A] += scale * +1.00000000 * np.einsum('uv,uUwV,xUwW,xAvW->AV', gamma1['a'], h['ab'][a,A,a,A], lambdas['ab'], t['ab'][pa,pV,ha,hA], optimize=True)
    O['b'][V,A] += scale * -1.00000000 * np.einsum('uv,uUwV,xUwV,xAvW->AW', gamma1['a'], h['ab'][a,A,a,A], lambdas['ab'], t['ab'][pa,pV,ha,hA], optimize=True)
    
    
    O['b'][V,A] += scale * +1.00000000 * np.einsum('uv,uAaB,aBvU->AU', gamma1['a'], h['ab'][a,V,v,V], t['ab'][pv,pV,ha,hA], optimize=True)
    O['b'][V,A] += scale * -1.00000000 * np.einsum('uv,wUvV,wUxW,uAxW->AV', gamma1['a'], h['ab'][a,A,a,A], lambdas['ab'], t['ab'][pa,pV,ha,hA], optimize=True)
    O['b'][V,A] += scale * +1.00000000 * np.einsum('uv,wUvV,wUxV,uAxW->AW', gamma1['a'], h['ab'][a,A,a,A], lambdas['ab'], t['ab'][pa,pV,ha,hA], optimize=True)
    
    
    O['b'][V,A] += scale * -1.00000000 * np.einsum('UV,uv,uUaW,aAvV->AW', gamma1['b'], gamma1['a'], h['ab'][a,A,v,A], t['ab'][pv,pV,ha,hA], optimize=True)
    O['b'][V,A] += scale * -1.00000000 * np.einsum('UV,iUaW,aAiV->AW', gamma1['b'], h['ab'][c,A,v,A], t['ab'][pv,pV,hc,hA], optimize=True)
    O['b'][V,A] += scale * -1.00000000 * np.einsum('UV,uUvW,uwvx,wAxV->AW', gamma1['b'], h['ab'][a,A,a,A], lambdas['aa'], t['ab'][pa,pV,ha,hA], optimize=True)
    
    O['b'][V,A] += scale * -1.00000000 * np.einsum('iIaU,aAiI->AU', h['ab'][c,C,v,A], t['ab'][pv,pV,hc,hC], optimize=True)
    O['b'][V,A] += scale * +1.00000000 * np.einsum('iUuV,vUuW,vAiW->AV', h['ab'][c,A,a,A], lambdas['ab'], t['ab'][pa,pV,hc,hA], optimize=True)
    O['b'][V,A] += scale * -1.00000000 * np.einsum('iUuV,vUuV,vAiW->AW', h['ab'][c,A,a,A], lambdas['ab'], t['ab'][pa,pV,hc,hA], optimize=True)
    O['b'][V,A] += scale * -1.00000000 * np.einsum('iAuU,vVuW,vViW->AU', h['ab'][c,V,a,A], lambdas['ab'], t['ab'][pa,pA,hc,hA], optimize=True)
    O['b'][V,A] += scale * +1.00000000 * np.einsum('iAuU,vVuU,vViW->AW', h['ab'][c,V,a,A], lambdas['ab'], t['ab'][pa,pA,hc,hA], optimize=True)
    O['b'][V,A] += scale * +1.00000000 * np.einsum('iAaB,aBiU->AU', h['ab'][c,V,v,V], t['ab'][pv,pV,hc,hA], optimize=True)
    O['b'][V,A] += scale * -1.00000000 * np.einsum('uIvU,uwvx,wAxI->AU', h['ab'][a,C,a,A], lambdas['aa'], t['ab'][pa,pV,ha,hC], optimize=True)
    O['b'][V,A] += scale * -1.00000000 * np.einsum('uUaV,uUvW,aAvW->AV', h['ab'][a,A,v,A], lambdas['ab'], t['ab'][pv,pV,ha,hA], optimize=True)
    O['b'][V,A] += scale * +1.00000000 * np.einsum('uUaV,uUvV,aAvW->AW', h['ab'][a,A,v,A], lambdas['ab'], t['ab'][pv,pV,ha,hA], optimize=True)
    O['b'][V,A] += scale * +1.00000000 * np.einsum('uAvB,uwvx,wBxU->AU', h['ab'][a,V,a,V], lambdas['aa'], t['ab'][pa,pV,ha,hA], optimize=True)
    O['b'][V,A] += scale * +1.00000000 * np.einsum('uAaU,uVvW,aVvW->AU', h['ab'][a,V,v,A], lambdas['ab'], t['ab'][pv,pA,ha,hA], optimize=True)
    O['b'][V,A] += scale * -1.00000000 * np.einsum('uAaU,uVvU,aVvW->AW', h['ab'][a,V,v,A], lambdas['ab'], t['ab'][pv,pA,ha,hA], optimize=True)
    O['b'][A,C] += scale * -1.00000000 * np.einsum('uv,wx,wIvJ,uUxI->UJ', eta1['a'], gamma1['a'], h['ab'][a,C,a,C], t['ab'][pa,pA,ha,hC], optimize=True)
    O['b'][A,C] += scale * +1.00000000 * np.einsum('uv,wx,wUvA,uAxI->UI', eta1['a'], gamma1['a'], h['ab'][a,A,a,V], t['ab'][pa,pV,ha,hC], optimize=True)
    O['b'][A,C] += scale * +1.00000000 * np.einsum('uv,UV,wx,uWxV,wUvI->WI', eta1['a'], gamma1['b'], gamma1['a'], h['ab'][a,A,a,A], t['ab'][pa,pA,ha,hC], optimize=True)
    
    O['b'][A,C] += scale * -1.00000000 * np.einsum('uv,UV,iUvI,uWiV->WI', eta1['a'], gamma1['b'], h['ab'][c,A,a,C], t['ab'][pa,pA,hc,hA], optimize=True)
    O['b'][A,C] += scale * -1.00000000 * np.einsum('uv,iIvJ,uUiI->UJ', eta1['a'], h['ab'][c,C,a,C], t['ab'][pa,pA,hc,hC], optimize=True)
    O['b'][A,C] += scale * +1.00000000 * np.einsum('uv,iUvA,uAiI->UI', eta1['a'], h['ab'][c,A,a,V], t['ab'][pa,pV,hc,hC], optimize=True)
    
    O['b'][A,C] += scale * +1.00000000 * np.einsum('uv,uUwV,xWwV,xWvI->UI', eta1['a'], h['ab'][a,A,a,A], lambdas['ab'], t['ab'][pa,pA,ha,hC], optimize=True)
    
    O['b'][A,C] += scale * -1.00000000 * np.einsum('uv,uUwV,xUwV,xWvI->WI', eta1['a'], h['ab'][a,A,a,A], lambdas['ab'], t['ab'][pa,pA,ha,hC], optimize=True)
    
    O['b'][A,C] += scale * -1.00000000 * np.einsum('uv,wUvV,wWxV,uWxI->UI', eta1['a'], h['ab'][a,A,a,A], lambdas['ab'], t['ab'][pa,pA,ha,hC], optimize=True)
    
    O['b'][A,C] += scale * +1.00000000 * np.einsum('uv,wUvV,wUxV,uWxI->WI', eta1['a'], h['ab'][a,A,a,A], lambdas['ab'], t['ab'][pa,pA,ha,hC], optimize=True)
    
    O['b'][A,C] += scale * +1.00000000 * np.einsum('UV,uv,wx,wWvV,uUxI->WI', eta1['b'], eta1['a'], gamma1['a'], h['ab'][a,A,a,A], t['ab'][pa,pA,ha,hC], optimize=True)
    O['b'][A,C] += scale * +1.00000000 * np.einsum('UV,uv,iWvV,uUiI->WI', eta1['b'], eta1['a'], h['ab'][c,A,a,A], t['ab'][pa,pA,hc,hC], optimize=True)
    O['b'][A,C] += scale * +1.00000000 * np.einsum('UV,uv,uWaV,aUvI->WI', eta1['b'], gamma1['a'], h['ab'][a,A,v,A], t['ab'][pv,pA,ha,hC], optimize=True)
    O['b'][A,C] += scale * +1.00000000 * np.einsum('UV,iWaV,aUiI->WI', eta1['b'], h['ab'][c,A,v,A], t['ab'][pv,pA,hc,hC], optimize=True)
    O['b'][A,C] += scale * +1.00000000 * np.einsum('UV,uWvV,uwvx,wUxI->WI', eta1['b'], h['ab'][a,A,a,A], lambdas['aa'], t['ab'][pa,pA,ha,hC], optimize=True)
    
    O['b'][A,C] += scale * -1.00000000 * np.einsum('uv,uIaJ,aUvI->UJ', gamma1['a'], h['ab'][a,C,v,C], t['ab'][pv,pA,ha,hC], optimize=True)
    
    O['b'][A,C] += scale * +1.00000000 * np.einsum('uv,uUwV,xWwV,xWvI->UI', gamma1['a'], h['ab'][a,A,a,A], lambdas['ab'], t['ab'][pa,pA,ha,hC], optimize=True)
    O['b'][A,C] += scale * +1.00000000 * np.einsum('uv,uUaA,aAvI->UI', gamma1['a'], h['ab'][a,A,v,V], t['ab'][pv,pV,ha,hC], optimize=True)
    
    O['b'][A,C] += scale * -1.00000000 * np.einsum('uv,uUwV,xUwV,xWvI->WI', gamma1['a'], h['ab'][a,A,a,A], lambdas['ab'], t['ab'][pa,pA,ha,hC], optimize=True)
    
    O['b'][A,C] += scale * -1.00000000 * np.einsum('uv,wUvV,wWxV,uWxI->UI', gamma1['a'], h['ab'][a,A,a,A], lambdas['ab'], t['ab'][pa,pA,ha,hC], optimize=True)
    
    O['b'][A,C] += scale * +1.00000000 * np.einsum('uv,wUvV,wUxV,uWxI->WI', gamma1['a'], h['ab'][a,A,a,A], lambdas['ab'], t['ab'][pa,pA,ha,hC], optimize=True)
    O['b'][A,C] += scale * -1.00000000 * np.einsum('UV,uv,uUaI,aWvV->WI', gamma1['b'], gamma1['a'], h['ab'][a,A,v,C], t['ab'][pv,pA,ha,hA], optimize=True)
    O['b'][A,C] += scale * -1.00000000 * np.einsum('UV,iUaI,aWiV->WI', gamma1['b'], h['ab'][c,A,v,C], t['ab'][pv,pA,hc,hA], optimize=True)
    O['b'][A,C] += scale * +1.00000000 * np.einsum('UV,uWvV,uwvx,wUxI->WI', gamma1['b'], h['ab'][a,A,a,A], lambdas['aa'], t['ab'][pa,pA,ha,hC], optimize=True)
    
    O['b'][A,C] += scale * -1.00000000 * np.einsum('iIaJ,aUiI->UJ', h['ab'][c,C,v,C], t['ab'][pv,pA,hc,hC], optimize=True)
    O['b'][A,C] += scale * -1.00000000 * np.einsum('iUuI,vVuW,vViW->UI', h['ab'][c,A,a,C], lambdas['ab'], t['ab'][pa,pA,hc,hA], optimize=True)
    O['b'][A,C] += scale * +1.00000000 * np.einsum('iUuV,vWuV,vWiI->UI', h['ab'][c,A,a,A], lambdas['ab'], t['ab'][pa,pA,hc,hC], optimize=True)
    O['b'][A,C] += scale * +1.00000000 * np.einsum('iUaA,aAiI->UI', h['ab'][c,A,v,V], t['ab'][pv,pV,hc,hC], optimize=True)
    O['b'][A,C] += scale * +1.00000000 * np.einsum('iUuI,vUuV,vWiV->WI', h['ab'][c,A,a,C], lambdas['ab'], t['ab'][pa,pA,hc,hA], optimize=True)
    O['b'][A,C] += scale * -1.00000000 * np.einsum('iUuV,vUuV,vWiI->WI', h['ab'][c,A,a,A], lambdas['ab'], t['ab'][pa,pA,hc,hC], optimize=True)
    O['b'][A,C] += scale * -1.00000000 * np.einsum('uIvJ,uwvx,wUxI->UJ', h['ab'][a,C,a,C], lambdas['aa'], t['ab'][pa,pA,ha,hC], optimize=True)
    O['b'][A,C] += scale * +1.00000000 * np.einsum('uUvA,uwvx,wAxI->UI', h['ab'][a,A,a,V], lambdas['aa'], t['ab'][pa,pV,ha,hC], optimize=True)
    O['b'][A,C] += scale * +1.00000000 * np.einsum('uUaI,uVvW,aVvW->UI', h['ab'][a,A,v,C], lambdas['ab'], t['ab'][pv,pA,ha,hA], optimize=True)
    O['b'][A,C] += scale * -1.00000000 * np.einsum('uUaV,uWvV,aWvI->UI', h['ab'][a,A,v,A], lambdas['ab'], t['ab'][pv,pA,ha,hC], optimize=True)
    O['b'][A,C] += scale * -1.00000000 * np.einsum('uUaI,uUvV,aWvV->WI', h['ab'][a,A,v,C], lambdas['ab'], t['ab'][pv,pA,ha,hA], optimize=True)
    O['b'][A,C] += scale * +1.00000000 * np.einsum('uUaV,uUvV,aWvI->WI', h['ab'][a,A,v,A], lambdas['ab'], t['ab'][pv,pA,ha,hC], optimize=True)
    O['b'][C,C] += scale * +1.00000000 * np.einsum('uv,wx,wIvA,uAxJ->IJ', eta1['a'], gamma1['a'], h['ab'][a,C,a,V], t['ab'][pa,pV,ha,hC], optimize=True)
    O['b'][C,C] += scale * +1.00000000 * np.einsum('uv,UV,wx,uIxV,wUvJ->IJ', eta1['a'], gamma1['b'], gamma1['a'], h['ab'][a,C,a,A], t['ab'][pa,pA,ha,hC], optimize=True)
    O['b'][C,C] += scale * +1.00000000 * np.einsum('uv,iIvA,uAiJ->IJ', eta1['a'], h['ab'][c,C,a,V], t['ab'][pa,pV,hc,hC], optimize=True)
    
    O['b'][C,C] += scale * +1.00000000 * np.einsum('uv,uIwU,xVwU,xVvJ->IJ', eta1['a'], h['ab'][a,C,a,A], lambdas['ab'], t['ab'][pa,pA,ha,hC], optimize=True)
    
    O['b'][C,C] += scale * -1.00000000 * np.einsum('uv,wIvU,wVxU,uVxJ->IJ', eta1['a'], h['ab'][a,C,a,A], lambdas['ab'], t['ab'][pa,pA,ha,hC], optimize=True)
    O['b'][C,C] += scale * +1.00000000 * np.einsum('UV,uv,wx,wIvV,uUxJ->IJ', eta1['b'], eta1['a'], gamma1['a'], h['ab'][a,C,a,A], t['ab'][pa,pA,ha,hC], optimize=True)
    O['b'][C,C] += scale * +1.00000000 * np.einsum('UV,uv,iIvV,uUiJ->IJ', eta1['b'], eta1['a'], h['ab'][c,C,a,A], t['ab'][pa,pA,hc,hC], optimize=True)
    O['b'][C,C] += scale * +1.00000000 * np.einsum('UV,uv,uIaV,aUvJ->IJ', eta1['b'], gamma1['a'], h['ab'][a,C,v,A], t['ab'][pv,pA,ha,hC], optimize=True)
    O['b'][C,C] += scale * +1.00000000 * np.einsum('UV,iIaV,aUiJ->IJ', eta1['b'], h['ab'][c,C,v,A], t['ab'][pv,pA,hc,hC], optimize=True)
    O['b'][C,C] += scale * +1.00000000 * np.einsum('UV,uIvV,uwvx,wUxJ->IJ', eta1['b'], h['ab'][a,C,a,A], lambdas['aa'], t['ab'][pa,pA,ha,hC], optimize=True)
    
    O['b'][C,C] += scale * +1.00000000 * np.einsum('uv,uIwU,xVwU,xVvJ->IJ', gamma1['a'], h['ab'][a,C,a,A], lambdas['ab'], t['ab'][pa,pA,ha,hC], optimize=True)
    O['b'][C,C] += scale * +1.00000000 * np.einsum('uv,uIaA,aAvJ->IJ', gamma1['a'], h['ab'][a,C,v,V], t['ab'][pv,pV,ha,hC], optimize=True)
    
    O['b'][C,C] += scale * -1.00000000 * np.einsum('uv,wIvU,wVxU,uVxJ->IJ', gamma1['a'], h['ab'][a,C,a,A], lambdas['ab'], t['ab'][pa,pA,ha,hC], optimize=True)
    O['b'][C,C] += scale * +1.00000000 * np.einsum('UV,uIvV,uwvx,wUxJ->IJ', gamma1['b'], h['ab'][a,C,a,A], lambdas['aa'], t['ab'][pa,pA,ha,hC], optimize=True)
    O['b'][C,C] += scale * -1.00000000 * np.einsum('iIuJ,vUuV,vUiV->IJ', h['ab'][c,C,a,C], lambdas['ab'], t['ab'][pa,pA,hc,hA], optimize=True)
    O['b'][C,C] += scale * +1.00000000 * np.einsum('iIuU,vVuU,vViJ->IJ', h['ab'][c,C,a,A], lambdas['ab'], t['ab'][pa,pA,hc,hC], optimize=True)
    O['b'][C,C] += scale * +1.00000000 * np.einsum('iIaA,aAiJ->IJ', h['ab'][c,C,v,V], t['ab'][pv,pV,hc,hC], optimize=True)
    O['b'][C,C] += scale * +1.00000000 * np.einsum('uIvA,uwvx,wAxJ->IJ', h['ab'][a,C,a,V], lambdas['aa'], t['ab'][pa,pV,ha,hC], optimize=True)
    O['b'][C,C] += scale * +1.00000000 * np.einsum('uIaJ,uUvV,aUvV->IJ', h['ab'][a,C,v,C], lambdas['ab'], t['ab'][pv,pA,ha,hA], optimize=True)
    O['b'][C,C] += scale * -1.00000000 * np.einsum('uIaU,uVvU,aVvJ->IJ', h['ab'][a,C,v,A], lambdas['ab'], t['ab'][pv,pA,ha,hC], optimize=True)
    O['b'][V,C] += scale * -1.00000000 * np.einsum('uv,wx,wIvJ,uAxI->AJ', eta1['a'], gamma1['a'], h['ab'][a,C,a,C], t['ab'][pa,pV,ha,hC], optimize=True)
    O['b'][V,C] += scale * +1.00000000 * np.einsum('uv,wx,wAvB,uBxI->AI', eta1['a'], gamma1['a'], h['ab'][a,V,a,V], t['ab'][pa,pV,ha,hC], optimize=True)
    O['b'][V,C] += scale * +1.00000000 * np.einsum('uv,UV,wx,uAxV,wUvI->AI', eta1['a'], gamma1['b'], gamma1['a'], h['ab'][a,V,a,A], t['ab'][pa,pA,ha,hC], optimize=True)
    O['b'][V,C] += scale * -1.00000000 * np.einsum('uv,UV,wx,wUvI,uAxV->AI', eta1['a'], gamma1['b'], gamma1['a'], h['ab'][a,A,a,C], t['ab'][pa,pV,ha,hA], optimize=True)
    O['b'][V,C] += scale * -1.00000000 * np.einsum('uv,UV,iUvI,uAiV->AI', eta1['a'], gamma1['b'], h['ab'][c,A,a,C], t['ab'][pa,pV,hc,hA], optimize=True)
    O['b'][V,C] += scale * -1.00000000 * np.einsum('uv,iIvJ,uAiI->AJ', eta1['a'], h['ab'][c,C,a,C], t['ab'][pa,pV,hc,hC], optimize=True)
    O['b'][V,C] += scale * +1.00000000 * np.einsum('uv,iAvB,uBiI->AI', eta1['a'], h['ab'][c,V,a,V], t['ab'][pa,pV,hc,hC], optimize=True)
    O['b'][V,C] += scale * +1.00000000 * np.einsum('uv,uUwI,xUwV,xAvV->AI', eta1['a'], h['ab'][a,A,a,C], lambdas['ab'], t['ab'][pa,pV,ha,hA], optimize=True)
    O['b'][V,C] += scale * -1.00000000 * np.einsum('uv,uUwV,xUwV,xAvI->AI', eta1['a'], h['ab'][a,A,a,A], lambdas['ab'], t['ab'][pa,pV,ha,hC], optimize=True)
    
    O['b'][V,C] += scale * +1.00000000 * np.einsum('uv,uAwU,xVwU,xVvI->AI', eta1['a'], h['ab'][a,V,a,A], lambdas['ab'], t['ab'][pa,pA,ha,hC], optimize=True)
    O['b'][V,C] += scale * -1.00000000 * np.einsum('uv,wUvI,wUxV,uAxV->AI', eta1['a'], h['ab'][a,A,a,C], lambdas['ab'], t['ab'][pa,pV,ha,hA], optimize=True)
    O['b'][V,C] += scale * +1.00000000 * np.einsum('uv,wUvV,wUxV,uAxI->AI', eta1['a'], h['ab'][a,A,a,A], lambdas['ab'], t['ab'][pa,pV,ha,hC], optimize=True)
    
    O['b'][V,C] += scale * -1.00000000 * np.einsum('uv,wAvU,wVxU,uVxI->AI', eta1['a'], h['ab'][a,V,a,A], lambdas['ab'], t['ab'][pa,pA,ha,hC], optimize=True)
    O['b'][V,C] += scale * -1.00000000 * np.einsum('UV,uv,wx,uUxI,wAvV->AI', eta1['b'], eta1['a'], gamma1['a'], h['ab'][a,A,a,C], t['ab'][pa,pV,ha,hA], optimize=True)
    O['b'][V,C] += scale * +1.00000000 * np.einsum('UV,uv,wx,wAvV,uUxI->AI', eta1['b'], eta1['a'], gamma1['a'], h['ab'][a,V,a,A], t['ab'][pa,pA,ha,hC], optimize=True)
    O['b'][V,C] += scale * +1.00000000 * np.einsum('UV,uv,iAvV,uUiI->AI', eta1['b'], eta1['a'], h['ab'][c,V,a,A], t['ab'][pa,pA,hc,hC], optimize=True)
    O['b'][V,C] += scale * +1.00000000 * np.einsum('UV,uv,uAaV,aUvI->AI', eta1['b'], gamma1['a'], h['ab'][a,V,v,A], t['ab'][pv,pA,ha,hC], optimize=True)
    O['b'][V,C] += scale * +1.00000000 * np.einsum('UV,iAaV,aUiI->AI', eta1['b'], h['ab'][c,V,v,A], t['ab'][pv,pA,hc,hC], optimize=True)
    O['b'][V,C] += scale * -1.00000000 * np.einsum('UV,uUvI,uwvx,wAxV->AI', eta1['b'], h['ab'][a,A,a,C], lambdas['aa'], t['ab'][pa,pV,ha,hA], optimize=True)
    O['b'][V,C] += scale * +1.00000000 * np.einsum('UV,uAvV,uwvx,wUxI->AI', eta1['b'], h['ab'][a,V,a,A], lambdas['aa'], t['ab'][pa,pA,ha,hC], optimize=True)
    O['b'][V,C] += scale * -1.00000000 * np.einsum('uv,uIaJ,aAvI->AJ', gamma1['a'], h['ab'][a,C,v,C], t['ab'][pv,pV,ha,hC], optimize=True)
    O['b'][V,C] += scale * +1.00000000 * np.einsum('uv,uUwI,xUwV,xAvV->AI', gamma1['a'], h['ab'][a,A,a,C], lambdas['ab'], t['ab'][pa,pV,ha,hA], optimize=True)
    O['b'][V,C] += scale * -1.00000000 * np.einsum('uv,uUwV,xUwV,xAvI->AI', gamma1['a'], h['ab'][a,A,a,A], lambdas['ab'], t['ab'][pa,pV,ha,hC], optimize=True)
    
    O['b'][V,C] += scale * +1.00000000 * np.einsum('uv,uAwU,xVwU,xVvI->AI', gamma1['a'], h['ab'][a,V,a,A], lambdas['ab'], t['ab'][pa,pA,ha,hC], optimize=True)
    O['b'][V,C] += scale * +1.00000000 * np.einsum('uv,uAaB,aBvI->AI', gamma1['a'], h['ab'][a,V,v,V], t['ab'][pv,pV,ha,hC], optimize=True)
    O['b'][V,C] += scale * -1.00000000 * np.einsum('uv,wUvI,wUxV,uAxV->AI', gamma1['a'], h['ab'][a,A,a,C], lambdas['ab'], t['ab'][pa,pV,ha,hA], optimize=True)
    O['b'][V,C] += scale * +1.00000000 * np.einsum('uv,wUvV,wUxV,uAxI->AI', gamma1['a'], h['ab'][a,A,a,A], lambdas['ab'], t['ab'][pa,pV,ha,hC], optimize=True)
    
    O['b'][V,C] += scale * -1.00000000 * np.einsum('uv,wAvU,wVxU,uVxI->AI', gamma1['a'], h['ab'][a,V,a,A], lambdas['ab'], t['ab'][pa,pA,ha,hC], optimize=True)
    O['b'][V,C] += scale * -1.00000000 * np.einsum('UV,uv,uUaI,aAvV->AI', gamma1['b'], gamma1['a'], h['ab'][a,A,v,C], t['ab'][pv,pV,ha,hA], optimize=True)
    O['b'][V,C] += scale * -1.00000000 * np.einsum('UV,iUaI,aAiV->AI', gamma1['b'], h['ab'][c,A,v,C], t['ab'][pv,pV,hc,hA], optimize=True)
    O['b'][V,C] += scale * -1.00000000 * np.einsum('UV,uUvI,uwvx,wAxV->AI', gamma1['b'], h['ab'][a,A,a,C], lambdas['aa'], t['ab'][pa,pV,ha,hA], optimize=True)
    O['b'][V,C] += scale * +1.00000000 * np.einsum('UV,uAvV,uwvx,wUxI->AI', gamma1['b'], h['ab'][a,V,a,A], lambdas['aa'], t['ab'][pa,pA,ha,hC], optimize=True)
    O['b'][V,C] += scale * -1.00000000 * np.einsum('iIaJ,aAiI->AJ', h['ab'][c,C,v,C], t['ab'][pv,pV,hc,hC], optimize=True)
    O['b'][V,C] += scale * +1.00000000 * np.einsum('iUuI,vUuV,vAiV->AI', h['ab'][c,A,a,C], lambdas['ab'], t['ab'][pa,pV,hc,hA], optimize=True)
    O['b'][V,C] += scale * -1.00000000 * np.einsum('iUuV,vUuV,vAiI->AI', h['ab'][c,A,a,A], lambdas['ab'], t['ab'][pa,pV,hc,hC], optimize=True)
    O['b'][V,C] += scale * -1.00000000 * np.einsum('iAuI,vUuV,vUiV->AI', h['ab'][c,V,a,C], lambdas['ab'], t['ab'][pa,pA,hc,hA], optimize=True)
    O['b'][V,C] += scale * +1.00000000 * np.einsum('iAuU,vVuU,vViI->AI', h['ab'][c,V,a,A], lambdas['ab'], t['ab'][pa,pA,hc,hC], optimize=True)
    O['b'][V,C] += scale * +1.00000000 * np.einsum('iAaB,aBiI->AI', h['ab'][c,V,v,V], t['ab'][pv,pV,hc,hC], optimize=True)
    O['b'][V,C] += scale * -1.00000000 * np.einsum('uIvJ,uwvx,wAxI->AJ', h['ab'][a,C,a,C], lambdas['aa'], t['ab'][pa,pV,ha,hC], optimize=True)
    O['b'][V,C] += scale * -1.00000000 * np.einsum('uUaI,uUvV,aAvV->AI', h['ab'][a,A,v,C], lambdas['ab'], t['ab'][pv,pV,ha,hA], optimize=True)
    O['b'][V,C] += scale * +1.00000000 * np.einsum('uUaV,uUvV,aAvI->AI', h['ab'][a,A,v,A], lambdas['ab'], t['ab'][pv,pV,ha,hC], optimize=True)
    O['b'][V,C] += scale * +1.00000000 * np.einsum('uAvB,uwvx,wBxI->AI', h['ab'][a,V,a,V], lambdas['aa'], t['ab'][pa,pV,ha,hC], optimize=True)
    O['b'][V,C] += scale * +1.00000000 * np.einsum('uAaI,uUvV,aUvV->AI', h['ab'][a,V,v,C], lambdas['ab'], t['ab'][pv,pA,ha,hA], optimize=True)
    O['b'][V,C] += scale * -1.00000000 * np.einsum('uAaU,uVvU,aVvI->AI', h['ab'][a,V,v,A], lambdas['ab'], t['ab'][pv,pA,ha,hC], optimize=True)
    O['b'][A,V] += scale * -1.00000000 * np.einsum('uv,wx,wIvA,uUxI->UA', eta1['a'], gamma1['a'], h['ab'][a,C,a,V], t['ab'][pa,pA,ha,hC], optimize=True)
    
    O['b'][A,V] += scale * -1.00000000 * np.einsum('uv,UV,iUvA,uWiV->WA', eta1['a'], gamma1['b'], h['ab'][c,A,a,V], t['ab'][pa,pA,hc,hA], optimize=True)
    O['b'][A,V] += scale * -1.00000000 * np.einsum('uv,iIvA,uUiI->UA', eta1['a'], h['ab'][c,C,a,V], t['ab'][pa,pA,hc,hC], optimize=True)
    
    
    
    
    
    
    O['b'][A,V] += scale * -1.00000000 * np.einsum('uv,uIaA,aUvI->UA', gamma1['a'], h['ab'][a,C,v,V], t['ab'][pv,pA,ha,hC], optimize=True)
    
    
    
    
    O['b'][A,V] += scale * -1.00000000 * np.einsum('UV,uv,uUaA,aWvV->WA', gamma1['b'], gamma1['a'], h['ab'][a,A,v,V], t['ab'][pv,pA,ha,hA], optimize=True)
    O['b'][A,V] += scale * -1.00000000 * np.einsum('UV,iUaA,aWiV->WA', gamma1['b'], h['ab'][c,A,v,V], t['ab'][pv,pA,hc,hA], optimize=True)
    
    O['b'][A,V] += scale * -1.00000000 * np.einsum('iIaA,aUiI->UA', h['ab'][c,C,v,V], t['ab'][pv,pA,hc,hC], optimize=True)
    O['b'][A,V] += scale * -1.00000000 * np.einsum('iUuA,vVuW,vViW->UA', h['ab'][c,A,a,V], lambdas['ab'], t['ab'][pa,pA,hc,hA], optimize=True)
    O['b'][A,V] += scale * +1.00000000 * np.einsum('iUuA,vUuV,vWiV->WA', h['ab'][c,A,a,V], lambdas['ab'], t['ab'][pa,pA,hc,hA], optimize=True)
    O['b'][A,V] += scale * -1.00000000 * np.einsum('uIvA,uwvx,wUxI->UA', h['ab'][a,C,a,V], lambdas['aa'], t['ab'][pa,pA,ha,hC], optimize=True)
    O['b'][A,V] += scale * +1.00000000 * np.einsum('uUaA,uVvW,aVvW->UA', h['ab'][a,A,v,V], lambdas['ab'], t['ab'][pv,pA,ha,hA], optimize=True)
    O['b'][A,V] += scale * -1.00000000 * np.einsum('uUaA,uUvV,aWvV->WA', h['ab'][a,A,v,V], lambdas['ab'], t['ab'][pv,pA,ha,hA], optimize=True)
    
    
    
    
    O['b'][C,V] += scale * -1.00000000 * np.einsum('iIuA,vUuV,vUiV->IA', h['ab'][c,C,a,V], lambdas['ab'], t['ab'][pa,pA,hc,hA], optimize=True)
    O['b'][C,V] += scale * +1.00000000 * np.einsum('uIaA,uUvV,aUvV->IA', h['ab'][a,C,v,V], lambdas['ab'], t['ab'][pv,pA,ha,hA], optimize=True)
    O['b'][V,V] += scale * -1.00000000 * np.einsum('uv,wx,wIvA,uBxI->BA', eta1['a'], gamma1['a'], h['ab'][a,C,a,V], t['ab'][pa,pV,ha,hC], optimize=True)
    O['b'][V,V] += scale * -1.00000000 * np.einsum('uv,UV,wx,wUvA,uBxV->BA', eta1['a'], gamma1['b'], gamma1['a'], h['ab'][a,A,a,V], t['ab'][pa,pV,ha,hA], optimize=True)
    O['b'][V,V] += scale * -1.00000000 * np.einsum('uv,UV,iUvA,uBiV->BA', eta1['a'], gamma1['b'], h['ab'][c,A,a,V], t['ab'][pa,pV,hc,hA], optimize=True)
    O['b'][V,V] += scale * -1.00000000 * np.einsum('uv,iIvA,uBiI->BA', eta1['a'], h['ab'][c,C,a,V], t['ab'][pa,pV,hc,hC], optimize=True)
    O['b'][V,V] += scale * +1.00000000 * np.einsum('uv,uUwA,xUwV,xBvV->BA', eta1['a'], h['ab'][a,A,a,V], lambdas['ab'], t['ab'][pa,pV,ha,hA], optimize=True)
    
    O['b'][V,V] += scale * -1.00000000 * np.einsum('uv,wUvA,wUxV,uBxV->BA', eta1['a'], h['ab'][a,A,a,V], lambdas['ab'], t['ab'][pa,pV,ha,hA], optimize=True)
    
    O['b'][V,V] += scale * -1.00000000 * np.einsum('UV,uv,wx,uUxA,wBvV->BA', eta1['b'], eta1['a'], gamma1['a'], h['ab'][a,A,a,V], t['ab'][pa,pV,ha,hA], optimize=True)
    O['b'][V,V] += scale * -1.00000000 * np.einsum('UV,uUvA,uwvx,wBxV->BA', eta1['b'], h['ab'][a,A,a,V], lambdas['aa'], t['ab'][pa,pV,ha,hA], optimize=True)
    O['b'][V,V] += scale * -1.00000000 * np.einsum('uv,uIaA,aBvI->BA', gamma1['a'], h['ab'][a,C,v,V], t['ab'][pv,pV,ha,hC], optimize=True)
    O['b'][V,V] += scale * +1.00000000 * np.einsum('uv,uUwA,xUwV,xBvV->BA', gamma1['a'], h['ab'][a,A,a,V], lambdas['ab'], t['ab'][pa,pV,ha,hA], optimize=True)
    
    O['b'][V,V] += scale * -1.00000000 * np.einsum('uv,wUvA,wUxV,uBxV->BA', gamma1['a'], h['ab'][a,A,a,V], lambdas['ab'], t['ab'][pa,pV,ha,hA], optimize=True)
    
    O['b'][V,V] += scale * -1.00000000 * np.einsum('UV,uv,uUaA,aBvV->BA', gamma1['b'], gamma1['a'], h['ab'][a,A,v,V], t['ab'][pv,pV,ha,hA], optimize=True)
    O['b'][V,V] += scale * -1.00000000 * np.einsum('UV,iUaA,aBiV->BA', gamma1['b'], h['ab'][c,A,v,V], t['ab'][pv,pV,hc,hA], optimize=True)
    O['b'][V,V] += scale * -1.00000000 * np.einsum('UV,uUvA,uwvx,wBxV->BA', gamma1['b'], h['ab'][a,A,a,V], lambdas['aa'], t['ab'][pa,pV,ha,hA], optimize=True)
    O['b'][V,V] += scale * -1.00000000 * np.einsum('iIaA,aBiI->BA', h['ab'][c,C,v,V], t['ab'][pv,pV,hc,hC], optimize=True)
    O['b'][V,V] += scale * +1.00000000 * np.einsum('iUuA,vUuV,vBiV->BA', h['ab'][c,A,a,V], lambdas['ab'], t['ab'][pa,pV,hc,hA], optimize=True)
    O['b'][V,V] += scale * -1.00000000 * np.einsum('iAuB,vUuV,vUiV->AB', h['ab'][c,V,a,V], lambdas['ab'], t['ab'][pa,pA,hc,hA], optimize=True)
    O['b'][V,V] += scale * -1.00000000 * np.einsum('uIvA,uwvx,wBxI->BA', h['ab'][a,C,a,V], lambdas['aa'], t['ab'][pa,pV,ha,hC], optimize=True)
    O['b'][V,V] += scale * -1.00000000 * np.einsum('uUaA,uUvV,aBvV->BA', h['ab'][a,A,v,V], lambdas['ab'], t['ab'][pv,pV,ha,hA], optimize=True)
    O['b'][V,V] += scale * +1.00000000 * np.einsum('uAaB,uUvV,aUvV->AB', h['ab'][a,V,v,V], lambdas['ab'], t['ab'][pv,pA,ha,hA], optimize=True)

    t1 = time.time()
    if verbose: print("h2b_t2b_c1b took {:.4f} seconds to run.".format(t1-t0))

    return O
def h2b_t2c_c1b(O, h, t, gamma1, eta1, lambdas, orbspace, verbose=False, scale=1.0):
    # 60 lines
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

    
    
    
    
    
    
    
    
    O['b'][A,A] += scale * -1.00000000 * np.einsum('uIvU,uVvW,XVIW->XU', h['ab'][a,C,a,A], lambdas['ab'], t['bb'][pA,pA,hC,hA], optimize=True)
    O['b'][A,A] += scale * +1.00000000 * np.einsum('uIvU,uVvU,WVIX->WX', h['ab'][a,C,a,A], lambdas['ab'], t['bb'][pA,pA,hC,hA], optimize=True)
    O['b'][A,A] += scale * -1.00000000 * np.einsum('uUvA,uVvW,VAXW->UX', h['ab'][a,A,a,V], lambdas['ab'], t['bb'][pA,pV,hA,hA], optimize=True)
    O['b'][A,A] += scale * +1.00000000 * np.einsum('uUvA,uUvV,WAXV->WX', h['ab'][a,A,a,V], lambdas['ab'], t['bb'][pA,pV,hA,hA], optimize=True)
    
    
    O['b'][C,A] += scale * -1.00000000 * np.einsum('uIvA,uUvV,UAWV->IW', h['ab'][a,C,a,V], lambdas['ab'], t['bb'][pA,pV,hA,hA], optimize=True)
    O['b'][V,A] += scale * +1.00000000 * np.einsum('UV,uUvW,uXvY,XAVY->AW', eta1['b'], h['ab'][a,A,a,A], lambdas['ab'], t['bb'][pA,pV,hA,hA], optimize=True)
    O['b'][V,A] += scale * +1.00000000 * np.einsum('UV,uUvW,uXvW,XAYV->AY', eta1['b'], h['ab'][a,A,a,A], lambdas['ab'], t['bb'][pA,pV,hA,hA], optimize=True)
    O['b'][V,A] += scale * -1.00000000 * np.einsum('UV,uWvV,uWvX,UAYX->AY', eta1['b'], h['ab'][a,A,a,A], lambdas['ab'], t['bb'][pA,pV,hA,hA], optimize=True)
    
    O['b'][V,A] += scale * +1.00000000 * np.einsum('UV,uUvW,uXvY,XAVY->AW', gamma1['b'], h['ab'][a,A,a,A], lambdas['ab'], t['bb'][pA,pV,hA,hA], optimize=True)
    O['b'][V,A] += scale * +1.00000000 * np.einsum('UV,uUvW,uXvW,XAYV->AY', gamma1['b'], h['ab'][a,A,a,A], lambdas['ab'], t['bb'][pA,pV,hA,hA], optimize=True)
    O['b'][V,A] += scale * -1.00000000 * np.einsum('UV,uWvV,uWvX,UAYX->AY', gamma1['b'], h['ab'][a,A,a,A], lambdas['ab'], t['bb'][pA,pV,hA,hA], optimize=True)
    
    O['b'][V,A] += scale * +1.00000000 * np.einsum('uIvU,uVvW,VAIW->AU', h['ab'][a,C,a,A], lambdas['ab'], t['bb'][pA,pV,hC,hA], optimize=True)
    O['b'][V,A] += scale * -1.00000000 * np.einsum('uIvU,uVvU,VAIW->AW', h['ab'][a,C,a,A], lambdas['ab'], t['bb'][pA,pV,hC,hA], optimize=True)
    O['b'][V,A] += scale * +1.00000000 * np.einsum('uUvA,uUvV,BAWV->BW', h['ab'][a,A,a,V], lambdas['ab'], t['bb'][pV,pV,hA,hA], optimize=True)
    O['b'][V,A] += scale * -1.00000000 * np.einsum('uAvB,uUvV,UBWV->AW', h['ab'][a,V,a,V], lambdas['ab'], t['bb'][pA,pV,hA,hA], optimize=True)
    O['b'][A,C] += scale * +1.00000000 * np.einsum('UV,uWvV,uXvY,UXIY->WI', eta1['b'], h['ab'][a,A,a,A], lambdas['ab'], t['bb'][pA,pA,hC,hA], optimize=True)
    
    O['b'][A,C] += scale * -1.00000000 * np.einsum('UV,uUvW,uXvW,YXIV->YI', eta1['b'], h['ab'][a,A,a,A], lambdas['ab'], t['bb'][pA,pA,hC,hA], optimize=True)
    O['b'][A,C] += scale * +1.00000000 * np.einsum('UV,uWvV,uWvX,YUIX->YI', eta1['b'], h['ab'][a,A,a,A], lambdas['ab'], t['bb'][pA,pA,hC,hA], optimize=True)
    O['b'][A,C] += scale * +1.00000000 * np.einsum('UV,uWvV,uXvY,UXIY->WI', gamma1['b'], h['ab'][a,A,a,A], lambdas['ab'], t['bb'][pA,pA,hC,hA], optimize=True)
    
    O['b'][A,C] += scale * -1.00000000 * np.einsum('UV,uUvW,uXvW,YXIV->YI', gamma1['b'], h['ab'][a,A,a,A], lambdas['ab'], t['bb'][pA,pA,hC,hA], optimize=True)
    O['b'][A,C] += scale * +1.00000000 * np.einsum('UV,uWvV,uWvX,YUIX->YI', gamma1['b'], h['ab'][a,A,a,A], lambdas['ab'], t['bb'][pA,pA,hC,hA], optimize=True)
    O['b'][A,C] += scale * -1.00000000 * np.einsum('uIvJ,uUvV,WUIV->WJ', h['ab'][a,C,a,C], lambdas['ab'], t['bb'][pA,pA,hC,hA], optimize=True)
    O['b'][A,C] += scale * -1.00000000 * np.einsum('uIvU,uVvU,WVJI->WJ', h['ab'][a,C,a,A], lambdas['ab'], t['bb'][pA,pA,hC,hC], optimize=True)
    O['b'][A,C] += scale * -1.00000000 * np.einsum('uUvA,uVvW,VAIW->UI', h['ab'][a,A,a,V], lambdas['ab'], t['bb'][pA,pV,hC,hA], optimize=True)
    O['b'][A,C] += scale * +1.00000000 * np.einsum('uUvA,uUvV,WAIV->WI', h['ab'][a,A,a,V], lambdas['ab'], t['bb'][pA,pV,hC,hA], optimize=True)
    O['b'][C,C] += scale * +1.00000000 * np.einsum('UV,uIvV,uWvX,UWJX->IJ', eta1['b'], h['ab'][a,C,a,A], lambdas['ab'], t['bb'][pA,pA,hC,hA], optimize=True)
    O['b'][C,C] += scale * +1.00000000 * np.einsum('UV,uIvV,uWvX,UWJX->IJ', gamma1['b'], h['ab'][a,C,a,A], lambdas['ab'], t['bb'][pA,pA,hC,hA], optimize=True)
    O['b'][C,C] += scale * -1.00000000 * np.einsum('uIvA,uUvV,UAJV->IJ', h['ab'][a,C,a,V], lambdas['ab'], t['bb'][pA,pV,hC,hA], optimize=True)
    O['b'][V,C] += scale * +1.00000000 * np.einsum('UV,uUvI,uWvX,WAVX->AI', eta1['b'], h['ab'][a,A,a,C], lambdas['ab'], t['bb'][pA,pV,hA,hA], optimize=True)
    O['b'][V,C] += scale * +1.00000000 * np.einsum('UV,uUvW,uXvW,XAIV->AI', eta1['b'], h['ab'][a,A,a,A], lambdas['ab'], t['bb'][pA,pV,hC,hA], optimize=True)
    O['b'][V,C] += scale * -1.00000000 * np.einsum('UV,uWvV,uWvX,UAIX->AI', eta1['b'], h['ab'][a,A,a,A], lambdas['ab'], t['bb'][pA,pV,hC,hA], optimize=True)
    O['b'][V,C] += scale * +1.00000000 * np.einsum('UV,uAvV,uWvX,UWIX->AI', eta1['b'], h['ab'][a,V,a,A], lambdas['ab'], t['bb'][pA,pA,hC,hA], optimize=True)
    O['b'][V,C] += scale * +1.00000000 * np.einsum('UV,uUvI,uWvX,WAVX->AI', gamma1['b'], h['ab'][a,A,a,C], lambdas['ab'], t['bb'][pA,pV,hA,hA], optimize=True)
    O['b'][V,C] += scale * +1.00000000 * np.einsum('UV,uUvW,uXvW,XAIV->AI', gamma1['b'], h['ab'][a,A,a,A], lambdas['ab'], t['bb'][pA,pV,hC,hA], optimize=True)
    O['b'][V,C] += scale * -1.00000000 * np.einsum('UV,uWvV,uWvX,UAIX->AI', gamma1['b'], h['ab'][a,A,a,A], lambdas['ab'], t['bb'][pA,pV,hC,hA], optimize=True)
    O['b'][V,C] += scale * +1.00000000 * np.einsum('UV,uAvV,uWvX,UWIX->AI', gamma1['b'], h['ab'][a,V,a,A], lambdas['ab'], t['bb'][pA,pA,hC,hA], optimize=True)
    O['b'][V,C] += scale * +1.00000000 * np.einsum('uIvJ,uUvV,UAIV->AJ', h['ab'][a,C,a,C], lambdas['ab'], t['bb'][pA,pV,hC,hA], optimize=True)
    O['b'][V,C] += scale * +1.00000000 * np.einsum('uIvU,uVvU,VAJI->AJ', h['ab'][a,C,a,A], lambdas['ab'], t['bb'][pA,pV,hC,hC], optimize=True)
    O['b'][V,C] += scale * +1.00000000 * np.einsum('uUvA,uUvV,BAIV->BI', h['ab'][a,A,a,V], lambdas['ab'], t['bb'][pV,pV,hC,hA], optimize=True)
    O['b'][V,C] += scale * -1.00000000 * np.einsum('uAvB,uUvV,UBIV->AI', h['ab'][a,V,a,V], lambdas['ab'], t['bb'][pA,pV,hC,hA], optimize=True)
    
    
    O['b'][A,V] += scale * -1.00000000 * np.einsum('uIvA,uUvV,WUIV->WA', h['ab'][a,C,a,V], lambdas['ab'], t['bb'][pA,pA,hC,hA], optimize=True)
    O['b'][V,V] += scale * +1.00000000 * np.einsum('UV,uUvA,uWvX,WBVX->BA', eta1['b'], h['ab'][a,A,a,V], lambdas['ab'], t['bb'][pA,pV,hA,hA], optimize=True)
    O['b'][V,V] += scale * +1.00000000 * np.einsum('UV,uUvA,uWvX,WBVX->BA', gamma1['b'], h['ab'][a,A,a,V], lambdas['ab'], t['bb'][pA,pV,hA,hA], optimize=True)
    O['b'][V,V] += scale * +1.00000000 * np.einsum('uIvA,uUvV,UBIV->BA', h['ab'][a,C,a,V], lambdas['ab'], t['bb'][pA,pV,hC,hA], optimize=True)

    t1 = time.time()
    if verbose: print("h2b_t2c_c1b took {:.4f} seconds to run.".format(t1-t0))

    return O
def h2c_t1b_c1b(O, h, t, gamma1, eta1, lambdas, orbspace, verbose=False, scale=1.0):
    # 45 lines
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

    
    
    O['b'][A,A] += scale * -1.00000000 * np.einsum('UV,IWXV,UI->WX', eta1['b'], h['bb'][C,A,A,A], t['b'][pA,hC], optimize=True)
    O['b'][A,A] += scale * +1.00000000 * np.einsum('UV,WUXA,AV->WX', gamma1['b'], h['bb'][A,A,A,V], t['b'][pV,hA], optimize=True)
    O['b'][A,A] += scale * -1.00000000 * np.einsum('IUVA,AI->UV', h['bb'][C,A,A,V], t['b'][pV,hC], optimize=True)
    
    
    O['b'][C,A] += scale * +1.00000000 * np.einsum('UV,IJWV,UJ->IW', eta1['b'], h['bb'][C,C,A,A], t['b'][pA,hC], optimize=True)
    O['b'][C,A] += scale * +1.00000000 * np.einsum('UV,IUWA,AV->IW', gamma1['b'], h['bb'][C,A,A,V], t['b'][pV,hA], optimize=True)
    O['b'][C,A] += scale * +1.00000000 * np.einsum('IJUA,AJ->IU', h['bb'][C,C,A,V], t['b'][pV,hC], optimize=True)
    
    
    O['b'][V,A] += scale * -1.00000000 * np.einsum('UV,IAWV,UI->AW', eta1['b'], h['bb'][C,V,A,A], t['b'][pA,hC], optimize=True)
    O['b'][V,A] += scale * -1.00000000 * np.einsum('UV,UAWB,BV->AW', gamma1['b'], h['bb'][A,V,A,V], t['b'][pV,hA], optimize=True)
    O['b'][V,A] += scale * -1.00000000 * np.einsum('IAUB,BI->AU', h['bb'][C,V,A,V], t['b'][pV,hC], optimize=True)
    
    
    O['b'][A,C] += scale * -1.00000000 * np.einsum('UV,IWJV,UI->WJ', eta1['b'], h['bb'][C,A,C,A], t['b'][pA,hC], optimize=True)
    O['b'][A,C] += scale * +1.00000000 * np.einsum('UV,WUIA,AV->WI', gamma1['b'], h['bb'][A,A,C,V], t['b'][pV,hA], optimize=True)
    O['b'][A,C] += scale * -1.00000000 * np.einsum('IUJA,AI->UJ', h['bb'][C,A,C,V], t['b'][pV,hC], optimize=True)
    
    
    O['b'][C,C] += scale * +1.00000000 * np.einsum('UV,IJKV,UJ->IK', eta1['b'], h['bb'][C,C,C,A], t['b'][pA,hC], optimize=True)
    O['b'][C,C] += scale * +1.00000000 * np.einsum('UV,IUJA,AV->IJ', gamma1['b'], h['bb'][C,A,C,V], t['b'][pV,hA], optimize=True)
    O['b'][C,C] += scale * +1.00000000 * np.einsum('IJKA,AJ->IK', h['bb'][C,C,C,V], t['b'][pV,hC], optimize=True)
    
    
    O['b'][V,C] += scale * -1.00000000 * np.einsum('UV,IAJV,UI->AJ', eta1['b'], h['bb'][C,V,C,A], t['b'][pA,hC], optimize=True)
    O['b'][V,C] += scale * -1.00000000 * np.einsum('UV,UAIB,BV->AI', gamma1['b'], h['bb'][A,V,C,V], t['b'][pV,hA], optimize=True)
    O['b'][V,C] += scale * -1.00000000 * np.einsum('IAJB,BI->AJ', h['bb'][C,V,C,V], t['b'][pV,hC], optimize=True)
    
    
    O['b'][A,V] += scale * +1.00000000 * np.einsum('UV,IWVA,UI->WA', eta1['b'], h['bb'][C,A,A,V], t['b'][pA,hC], optimize=True)
    O['b'][A,V] += scale * +1.00000000 * np.einsum('UV,WUAB,BV->WA', gamma1['b'], h['bb'][A,A,V,V], t['b'][pV,hA], optimize=True)
    O['b'][A,V] += scale * -1.00000000 * np.einsum('IUAB,BI->UA', h['bb'][C,A,V,V], t['b'][pV,hC], optimize=True)
    
    
    O['b'][C,V] += scale * -1.00000000 * np.einsum('UV,IJVA,UJ->IA', eta1['b'], h['bb'][C,C,A,V], t['b'][pA,hC], optimize=True)
    O['b'][C,V] += scale * +1.00000000 * np.einsum('UV,IUAB,BV->IA', gamma1['b'], h['bb'][C,A,V,V], t['b'][pV,hA], optimize=True)
    O['b'][C,V] += scale * +1.00000000 * np.einsum('IJAB,BJ->IA', h['bb'][C,C,V,V], t['b'][pV,hC], optimize=True)
    
    
    O['b'][V,V] += scale * +1.00000000 * np.einsum('UV,IAVB,UI->AB', eta1['b'], h['bb'][C,V,A,V], t['b'][pA,hC], optimize=True)
    O['b'][V,V] += scale * -1.00000000 * np.einsum('UV,UABC,CV->AB', gamma1['b'], h['bb'][A,V,V,V], t['b'][pV,hA], optimize=True)
    O['b'][V,V] += scale * -1.00000000 * np.einsum('IABC,CI->AB', h['bb'][C,V,V,V], t['b'][pV,hC], optimize=True)

    t1 = time.time()
    if verbose: print("h2c_t1b_c1b took {:.4f} seconds to run.".format(t1-t0))

    return O
def h2c_t2b_c1b(O, h, t, gamma1, eta1, lambdas, orbspace, verbose=False, scale=1.0):
    # 90 lines
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

    
    
    
    
    
    
    
    
    O['b'][A,A] += scale * +1.00000000 * np.einsum('IUVW,uXvW,uXvI->UV', h['bb'][C,A,A,A], lambdas['ab'], t['ab'][pa,pA,ha,hC], optimize=True)
    O['b'][A,A] += scale * -1.00000000 * np.einsum('IUVW,uUvW,uXvI->XV', h['bb'][C,A,A,A], lambdas['ab'], t['ab'][pa,pA,ha,hC], optimize=True)
    O['b'][A,A] += scale * +1.00000000 * np.einsum('UVWA,uVvX,uAvX->UW', h['bb'][A,A,A,V], lambdas['ab'], t['ab'][pa,pV,ha,hA], optimize=True)
    O['b'][A,A] += scale * -1.00000000 * np.einsum('UVWA,uVvW,uAvX->UX', h['bb'][A,A,A,V], lambdas['ab'], t['ab'][pa,pV,ha,hA], optimize=True)
    
    
    
    
    
    
    O['b'][C,A] += scale * -1.00000000 * np.einsum('IJUV,uWvV,uWvJ->IU', h['bb'][C,C,A,A], lambdas['ab'], t['ab'][pa,pA,ha,hC], optimize=True)
    O['b'][C,A] += scale * +1.00000000 * np.einsum('IUVA,uUvW,uAvW->IV', h['bb'][C,A,A,V], lambdas['ab'], t['ab'][pa,pV,ha,hA], optimize=True)
    O['b'][C,A] += scale * -1.00000000 * np.einsum('IUVA,uUvV,uAvW->IW', h['bb'][C,A,A,V], lambdas['ab'], t['ab'][pa,pV,ha,hA], optimize=True)
    O['b'][V,A] += scale * -1.00000000 * np.einsum('UV,UWXY,uWvY,uAvV->AX', eta1['b'], h['bb'][A,A,A,A], lambdas['ab'], t['ab'][pa,pV,ha,hA], optimize=True)
    
    
    
    O['b'][V,A] += scale * -1.00000000 * np.einsum('UV,UWXY,uWvY,uAvV->AX', gamma1['b'], h['bb'][A,A,A,A], lambdas['ab'], t['ab'][pa,pV,ha,hA], optimize=True)
    
    
    
    O['b'][V,A] += scale * -1.00000000 * np.einsum('IUVW,uUvW,uAvI->AV', h['bb'][C,A,A,A], lambdas['ab'], t['ab'][pa,pV,ha,hC], optimize=True)
    O['b'][V,A] += scale * +1.00000000 * np.einsum('IAUV,uWvV,uWvI->AU', h['bb'][C,V,A,A], lambdas['ab'], t['ab'][pa,pA,ha,hC], optimize=True)
    O['b'][V,A] += scale * -1.00000000 * np.einsum('UAVB,uUvW,uBvW->AV', h['bb'][A,V,A,V], lambdas['ab'], t['ab'][pa,pV,ha,hA], optimize=True)
    O['b'][V,A] += scale * +1.00000000 * np.einsum('UAVB,uUvV,uBvW->AW', h['bb'][A,V,A,V], lambdas['ab'], t['ab'][pa,pV,ha,hA], optimize=True)
    
    
    O['b'][A,C] += scale * +1.00000000 * np.einsum('UV,WXVY,uXvY,uUvI->WI', eta1['b'], h['bb'][A,A,A,A], lambdas['ab'], t['ab'][pa,pA,ha,hC], optimize=True)
    
    
    
    O['b'][A,C] += scale * +1.00000000 * np.einsum('UV,WXVY,uXvY,uUvI->WI', gamma1['b'], h['bb'][A,A,A,A], lambdas['ab'], t['ab'][pa,pA,ha,hC], optimize=True)
    
    O['b'][A,C] += scale * +1.00000000 * np.einsum('IUJV,uWvV,uWvI->UJ', h['bb'][C,A,C,A], lambdas['ab'], t['ab'][pa,pA,ha,hC], optimize=True)
    O['b'][A,C] += scale * -1.00000000 * np.einsum('IUJV,uUvV,uWvI->WJ', h['bb'][C,A,C,A], lambdas['ab'], t['ab'][pa,pA,ha,hC], optimize=True)
    O['b'][A,C] += scale * +1.00000000 * np.einsum('UVIA,uVvW,uAvW->UI', h['bb'][A,A,C,V], lambdas['ab'], t['ab'][pa,pV,ha,hA], optimize=True)
    O['b'][A,C] += scale * -1.00000000 * np.einsum('UVWA,uVvW,uAvI->UI', h['bb'][A,A,A,V], lambdas['ab'], t['ab'][pa,pV,ha,hC], optimize=True)
    
    
    O['b'][C,C] += scale * +1.00000000 * np.einsum('UV,IWVX,uWvX,uUvJ->IJ', eta1['b'], h['bb'][C,A,A,A], lambdas['ab'], t['ab'][pa,pA,ha,hC], optimize=True)
    
    
    O['b'][C,C] += scale * +1.00000000 * np.einsum('UV,IWVX,uWvX,uUvJ->IJ', gamma1['b'], h['bb'][C,A,A,A], lambdas['ab'], t['ab'][pa,pA,ha,hC], optimize=True)
    O['b'][C,C] += scale * -1.00000000 * np.einsum('IJKU,uVvU,uVvJ->IK', h['bb'][C,C,C,A], lambdas['ab'], t['ab'][pa,pA,ha,hC], optimize=True)
    O['b'][C,C] += scale * +1.00000000 * np.einsum('IUJA,uUvV,uAvV->IJ', h['bb'][C,A,C,V], lambdas['ab'], t['ab'][pa,pV,ha,hA], optimize=True)
    O['b'][C,C] += scale * -1.00000000 * np.einsum('IUVA,uUvV,uAvJ->IJ', h['bb'][C,A,A,V], lambdas['ab'], t['ab'][pa,pV,ha,hC], optimize=True)
    O['b'][V,C] += scale * -1.00000000 * np.einsum('UV,UWIX,uWvX,uAvV->AI', eta1['b'], h['bb'][A,A,C,A], lambdas['ab'], t['ab'][pa,pV,ha,hA], optimize=True)
    
    
    O['b'][V,C] += scale * -1.00000000 * np.einsum('UV,WAVX,uWvX,uUvI->AI', eta1['b'], h['bb'][A,V,A,A], lambdas['ab'], t['ab'][pa,pA,ha,hC], optimize=True)
    O['b'][V,C] += scale * -1.00000000 * np.einsum('UV,UWIX,uWvX,uAvV->AI', gamma1['b'], h['bb'][A,A,C,A], lambdas['ab'], t['ab'][pa,pV,ha,hA], optimize=True)
    
    
    O['b'][V,C] += scale * -1.00000000 * np.einsum('UV,WAVX,uWvX,uUvI->AI', gamma1['b'], h['bb'][A,V,A,A], lambdas['ab'], t['ab'][pa,pA,ha,hC], optimize=True)
    O['b'][V,C] += scale * -1.00000000 * np.einsum('IUJV,uUvV,uAvI->AJ', h['bb'][C,A,C,A], lambdas['ab'], t['ab'][pa,pV,ha,hC], optimize=True)
    O['b'][V,C] += scale * +1.00000000 * np.einsum('IAJU,uVvU,uVvI->AJ', h['bb'][C,V,C,A], lambdas['ab'], t['ab'][pa,pA,ha,hC], optimize=True)
    O['b'][V,C] += scale * -1.00000000 * np.einsum('UAIB,uUvV,uBvV->AI', h['bb'][A,V,C,V], lambdas['ab'], t['ab'][pa,pV,ha,hA], optimize=True)
    O['b'][V,C] += scale * +1.00000000 * np.einsum('UAVB,uUvV,uBvI->AI', h['bb'][A,V,A,V], lambdas['ab'], t['ab'][pa,pV,ha,hC], optimize=True)
    
    
    
    
    
    
    O['b'][A,V] += scale * -1.00000000 * np.einsum('IUVA,uWvV,uWvI->UA', h['bb'][C,A,A,V], lambdas['ab'], t['ab'][pa,pA,ha,hC], optimize=True)
    O['b'][A,V] += scale * +1.00000000 * np.einsum('IUVA,uUvV,uWvI->WA', h['bb'][C,A,A,V], lambdas['ab'], t['ab'][pa,pA,ha,hC], optimize=True)
    O['b'][A,V] += scale * +1.00000000 * np.einsum('UVAB,uVvW,uBvW->UA', h['bb'][A,A,V,V], lambdas['ab'], t['ab'][pa,pV,ha,hA], optimize=True)
    
    
    
    
    O['b'][C,V] += scale * +1.00000000 * np.einsum('IJUA,uVvU,uVvJ->IA', h['bb'][C,C,A,V], lambdas['ab'], t['ab'][pa,pA,ha,hC], optimize=True)
    O['b'][C,V] += scale * +1.00000000 * np.einsum('IUAB,uUvV,uBvV->IA', h['bb'][C,A,V,V], lambdas['ab'], t['ab'][pa,pV,ha,hA], optimize=True)
    O['b'][V,V] += scale * +1.00000000 * np.einsum('UV,UWXA,uWvX,uBvV->BA', eta1['b'], h['bb'][A,A,A,V], lambdas['ab'], t['ab'][pa,pV,ha,hA], optimize=True)
    
    
    O['b'][V,V] += scale * +1.00000000 * np.einsum('UV,UWXA,uWvX,uBvV->BA', gamma1['b'], h['bb'][A,A,A,V], lambdas['ab'], t['ab'][pa,pV,ha,hA], optimize=True)
    
    
    O['b'][V,V] += scale * +1.00000000 * np.einsum('IUVA,uUvV,uBvI->BA', h['bb'][C,A,A,V], lambdas['ab'], t['ab'][pa,pV,ha,hC], optimize=True)
    O['b'][V,V] += scale * -1.00000000 * np.einsum('IAUB,uVvU,uVvI->AB', h['bb'][C,V,A,V], lambdas['ab'], t['ab'][pa,pA,ha,hC], optimize=True)
    O['b'][V,V] += scale * -1.00000000 * np.einsum('UABC,uUvV,uCvV->AB', h['bb'][A,V,V,V], lambdas['ab'], t['ab'][pa,pV,ha,hA], optimize=True)

    t1 = time.time()
    if verbose: print("h2c_t2b_c1b took {:.4f} seconds to run.".format(t1-t0))

    return O
def h2c_t2c_c1b(O, h, t, gamma1, eta1, lambdas, orbspace, verbose=False, scale=1.0):
    # 234 lines
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

    
    
    O['b'][A,A] += scale * +0.50000000 * np.einsum('UV,WX,IYVX,UWIZ->YZ', eta1['b'], eta1['b'], h['bb'][C,A,A,A], t['bb'][pA,pA,hC,hA], optimize=True)
    
    
    O['b'][A,A] += scale * -1.00000000 * np.einsum('UV,WX,IWYV,ZUIX->ZY', eta1['b'], gamma1['b'], h['bb'][C,A,A,A], t['bb'][pA,pA,hC,hA], optimize=True)
    O['b'][A,A] += scale * +1.00000000 * np.einsum('UV,WX,YWVA,UAZX->YZ', eta1['b'], gamma1['b'], h['bb'][A,A,A,V], t['bb'][pA,pV,hA,hA], optimize=True)
    O['b'][A,A] += scale * -0.50000000 * np.einsum('UV,IJWV,XUIJ->XW', eta1['b'], h['bb'][C,C,A,A], t['bb'][pA,pA,hC,hC], optimize=True)
    O['b'][A,A] += scale * +1.00000000 * np.einsum('UV,IWVA,UAIX->WX', eta1['b'], h['bb'][C,A,A,V], t['bb'][pA,pV,hC,hA], optimize=True)
    
    
    
    
    
    
    
    
    O['b'][A,A] += scale * -0.50000000 * np.einsum('UV,WX,UWYA,ZAVX->ZY', gamma1['b'], gamma1['b'], h['bb'][A,A,A,V], t['bb'][pA,pV,hA,hA], optimize=True)
    O['b'][A,A] += scale * -1.00000000 * np.einsum('UV,IUWA,XAIV->XW', gamma1['b'], h['bb'][C,A,A,V], t['bb'][pA,pV,hC,hA], optimize=True)
    
    
    O['b'][A,A] += scale * +0.50000000 * np.einsum('UV,WUAB,ABXV->WX', gamma1['b'], h['bb'][A,A,V,V], t['bb'][pV,pV,hA,hA], optimize=True)
    
    
    
    
    
    
    O['b'][A,A] += scale * -0.50000000 * np.einsum('IJUA,VAIJ->VU', h['bb'][C,C,A,V], t['bb'][pA,pV,hC,hC], optimize=True)
    O['b'][A,A] += scale * +0.50000000 * np.einsum('IUVW,XYWZ,XYIZ->UV', h['bb'][C,A,A,A], lambdas['bb'], t['bb'][pA,pA,hC,hA], optimize=True)
    O['b'][A,A] += scale * +0.25000000 * np.einsum('IUVW,XYVW,XYIZ->UZ', h['bb'][C,A,A,A], lambdas['bb'], t['bb'][pA,pA,hC,hA], optimize=True)
    O['b'][A,A] += scale * +0.50000000 * np.einsum('IUAB,ABIV->UV', h['bb'][C,A,V,V], t['bb'][pV,pV,hC,hA], optimize=True)
    O['b'][A,A] += scale * -1.00000000 * np.einsum('IUVW,UXWY,ZXIY->ZV', h['bb'][C,A,A,A], lambdas['bb'], t['bb'][pA,pA,hC,hA], optimize=True)
    O['b'][A,A] += scale * -0.50000000 * np.einsum('IUVW,UXVW,YXIZ->YZ', h['bb'][C,A,A,A], lambdas['bb'], t['bb'][pA,pA,hC,hA], optimize=True)
    O['b'][A,A] += scale * -0.50000000 * np.einsum('UVWA,VXYZ,XAYZ->UW', h['bb'][A,A,A,V], lambdas['bb'], t['bb'][pA,pV,hA,hA], optimize=True)
    O['b'][A,A] += scale * +1.00000000 * np.einsum('UVWA,VXWY,XAZY->UZ', h['bb'][A,A,A,V], lambdas['bb'], t['bb'][pA,pV,hA,hA], optimize=True)
    O['b'][A,A] += scale * -0.25000000 * np.einsum('UVWA,UVXY,ZAXY->ZW', h['bb'][A,A,A,V], lambdas['bb'], t['bb'][pA,pV,hA,hA], optimize=True)
    O['b'][A,A] += scale * +0.50000000 * np.einsum('UVWA,UVWX,YAZX->YZ', h['bb'][A,A,A,V], lambdas['bb'], t['bb'][pA,pV,hA,hA], optimize=True)
    
    O['b'][C,A] += scale * -0.50000000 * np.einsum('UV,WX,IJVX,UWJY->IY', eta1['b'], eta1['b'], h['bb'][C,C,A,A], t['bb'][pA,pA,hC,hA], optimize=True)
    
    O['b'][C,A] += scale * +1.00000000 * np.einsum('UV,WX,IWVA,UAYX->IY', eta1['b'], gamma1['b'], h['bb'][C,A,A,V], t['bb'][pA,pV,hA,hA], optimize=True)
    O['b'][C,A] += scale * -1.00000000 * np.einsum('UV,IJVA,UAJW->IW', eta1['b'], h['bb'][C,C,A,V], t['bb'][pA,pV,hC,hA], optimize=True)
    
    
    
    
    
    
    O['b'][C,A] += scale * +0.50000000 * np.einsum('UV,IUAB,ABWV->IW', gamma1['b'], h['bb'][C,A,V,V], t['bb'][pV,pV,hA,hA], optimize=True)
    
    
    O['b'][C,A] += scale * -0.50000000 * np.einsum('IJUV,WXVY,WXJY->IU', h['bb'][C,C,A,A], lambdas['bb'], t['bb'][pA,pA,hC,hA], optimize=True)
    O['b'][C,A] += scale * -0.25000000 * np.einsum('IJUV,WXUV,WXJY->IY', h['bb'][C,C,A,A], lambdas['bb'], t['bb'][pA,pA,hC,hA], optimize=True)
    O['b'][C,A] += scale * -0.50000000 * np.einsum('IJAB,ABJU->IU', h['bb'][C,C,V,V], t['bb'][pV,pV,hC,hA], optimize=True)
    O['b'][C,A] += scale * -0.50000000 * np.einsum('IUVA,UWXY,WAXY->IV', h['bb'][C,A,A,V], lambdas['bb'], t['bb'][pA,pV,hA,hA], optimize=True)
    O['b'][C,A] += scale * +1.00000000 * np.einsum('IUVA,UWVX,WAYX->IY', h['bb'][C,A,A,V], lambdas['bb'], t['bb'][pA,pV,hA,hA], optimize=True)
    O['b'][V,A] += scale * +0.50000000 * np.einsum('UV,WX,YZ,UWRZ,YAVX->AR', eta1['b'], eta1['b'], gamma1['b'], h['bb'][A,A,A,A], t['bb'][pA,pV,hA,hA], optimize=True)
    
    O['b'][V,A] += scale * +0.50000000 * np.einsum('UV,WX,IAVX,UWIY->AY', eta1['b'], eta1['b'], h['bb'][C,V,A,A], t['bb'][pA,pA,hC,hA], optimize=True)
    
    O['b'][V,A] += scale * +0.50000000 * np.einsum('UV,WX,YZ,WYRV,UAXZ->AR', eta1['b'], gamma1['b'], gamma1['b'], h['bb'][A,A,A,A], t['bb'][pA,pV,hA,hA], optimize=True)
    O['b'][V,A] += scale * +1.00000000 * np.einsum('UV,WX,IWYV,UAIX->AY', eta1['b'], gamma1['b'], h['bb'][C,A,A,A], t['bb'][pA,pV,hC,hA], optimize=True)
    O['b'][V,A] += scale * -1.00000000 * np.einsum('UV,WX,WAVB,UBYX->AY', eta1['b'], gamma1['b'], h['bb'][A,V,A,V], t['bb'][pA,pV,hA,hA], optimize=True)
    O['b'][V,A] += scale * +0.50000000 * np.einsum('UV,IJWV,UAIJ->AW', eta1['b'], h['bb'][C,C,A,A], t['bb'][pA,pV,hC,hC], optimize=True)
    O['b'][V,A] += scale * +1.00000000 * np.einsum('UV,IAVB,UBIW->AW', eta1['b'], h['bb'][C,V,A,V], t['bb'][pA,pV,hC,hA], optimize=True)
    O['b'][V,A] += scale * +1.00000000 * np.einsum('UV,UWXY,WZYR,ZAVR->AX', eta1['b'], h['bb'][A,A,A,A], lambdas['bb'], t['bb'][pA,pV,hA,hA], optimize=True)
    O['b'][V,A] += scale * -0.50000000 * np.einsum('UV,UWXY,WZXY,ZARV->AR', eta1['b'], h['bb'][A,A,A,A], lambdas['bb'], t['bb'][pA,pV,hA,hA], optimize=True)
    
    
    O['b'][V,A] += scale * +0.25000000 * np.einsum('UV,WXYV,WXZR,UAZR->AY', eta1['b'], h['bb'][A,A,A,A], lambdas['bb'], t['bb'][pA,pV,hA,hA], optimize=True)
    O['b'][V,A] += scale * +0.50000000 * np.einsum('UV,WXVY,WXYZ,UARZ->AR', eta1['b'], h['bb'][A,A,A,A], lambdas['bb'], t['bb'][pA,pV,hA,hA], optimize=True)
    
    
    O['b'][V,A] += scale * -0.50000000 * np.einsum('UV,WX,UWYA,BAVX->BY', gamma1['b'], gamma1['b'], h['bb'][A,A,A,V], t['bb'][pV,pV,hA,hA], optimize=True)
    O['b'][V,A] += scale * -1.00000000 * np.einsum('UV,IUWA,BAIV->BW', gamma1['b'], h['bb'][C,A,A,V], t['bb'][pV,pV,hC,hA], optimize=True)
    O['b'][V,A] += scale * +1.00000000 * np.einsum('UV,UWXY,WZYR,ZAVR->AX', gamma1['b'], h['bb'][A,A,A,A], lambdas['bb'], t['bb'][pA,pV,hA,hA], optimize=True)
    O['b'][V,A] += scale * -0.50000000 * np.einsum('UV,UWXY,WZXY,ZARV->AR', gamma1['b'], h['bb'][A,A,A,A], lambdas['bb'], t['bb'][pA,pV,hA,hA], optimize=True)
    
    
    O['b'][V,A] += scale * -0.50000000 * np.einsum('UV,UABC,BCWV->AW', gamma1['b'], h['bb'][A,V,V,V], t['bb'][pV,pV,hA,hA], optimize=True)
    O['b'][V,A] += scale * +0.25000000 * np.einsum('UV,WXYV,WXZR,UAZR->AY', gamma1['b'], h['bb'][A,A,A,A], lambdas['bb'], t['bb'][pA,pV,hA,hA], optimize=True)
    O['b'][V,A] += scale * +0.50000000 * np.einsum('UV,WXVY,WXYZ,UARZ->AR', gamma1['b'], h['bb'][A,A,A,A], lambdas['bb'], t['bb'][pA,pV,hA,hA], optimize=True)
    
    
    O['b'][V,A] += scale * -0.50000000 * np.einsum('IJUA,BAIJ->BU', h['bb'][C,C,A,V], t['bb'][pV,pV,hC,hC], optimize=True)
    O['b'][V,A] += scale * +1.00000000 * np.einsum('IUVW,UXWY,XAIY->AV', h['bb'][C,A,A,A], lambdas['bb'], t['bb'][pA,pV,hC,hA], optimize=True)
    O['b'][V,A] += scale * +0.50000000 * np.einsum('IUVW,UXVW,XAIY->AY', h['bb'][C,A,A,A], lambdas['bb'], t['bb'][pA,pV,hC,hA], optimize=True)
    O['b'][V,A] += scale * +0.50000000 * np.einsum('IAUV,WXVY,WXIY->AU', h['bb'][C,V,A,A], lambdas['bb'], t['bb'][pA,pA,hC,hA], optimize=True)
    O['b'][V,A] += scale * +0.25000000 * np.einsum('IAUV,WXUV,WXIY->AY', h['bb'][C,V,A,A], lambdas['bb'], t['bb'][pA,pA,hC,hA], optimize=True)
    O['b'][V,A] += scale * +0.50000000 * np.einsum('IABC,BCIU->AU', h['bb'][C,V,V,V], t['bb'][pV,pV,hC,hA], optimize=True)
    O['b'][V,A] += scale * -0.25000000 * np.einsum('UVWA,UVXY,BAXY->BW', h['bb'][A,A,A,V], lambdas['bb'], t['bb'][pV,pV,hA,hA], optimize=True)
    O['b'][V,A] += scale * +0.50000000 * np.einsum('UVWA,UVWX,BAYX->BY', h['bb'][A,A,A,V], lambdas['bb'], t['bb'][pV,pV,hA,hA], optimize=True)
    O['b'][V,A] += scale * +0.50000000 * np.einsum('UAVB,UWXY,WBXY->AV', h['bb'][A,V,A,V], lambdas['bb'], t['bb'][pA,pV,hA,hA], optimize=True)
    O['b'][V,A] += scale * -1.00000000 * np.einsum('UAVB,UWVX,WBYX->AY', h['bb'][A,V,A,V], lambdas['bb'], t['bb'][pA,pV,hA,hA], optimize=True)
    O['b'][A,C] += scale * +0.50000000 * np.einsum('UV,WX,YZ,RYVX,UWIZ->RI', eta1['b'], eta1['b'], gamma1['b'], h['bb'][A,A,A,A], t['bb'][pA,pA,hC,hA], optimize=True)
    
    O['b'][A,C] += scale * -0.50000000 * np.einsum('UV,WX,IYVX,UWJI->YJ', eta1['b'], eta1['b'], h['bb'][C,A,A,A], t['bb'][pA,pA,hC,hC], optimize=True)
    O['b'][A,C] += scale * +0.50000000 * np.einsum('UV,WX,YZ,RUXZ,WYIV->RI', eta1['b'], gamma1['b'], gamma1['b'], h['bb'][A,A,A,A], t['bb'][pA,pA,hC,hA], optimize=True)
    
    O['b'][A,C] += scale * -1.00000000 * np.einsum('UV,WX,IWJV,YUIX->YJ', eta1['b'], gamma1['b'], h['bb'][C,A,C,A], t['bb'][pA,pA,hC,hA], optimize=True)
    O['b'][A,C] += scale * +1.00000000 * np.einsum('UV,WX,YWVA,UAIX->YI', eta1['b'], gamma1['b'], h['bb'][A,A,A,V], t['bb'][pA,pV,hC,hA], optimize=True)
    O['b'][A,C] += scale * -0.50000000 * np.einsum('UV,IJKV,WUIJ->WK', eta1['b'], h['bb'][C,C,C,A], t['bb'][pA,pA,hC,hC], optimize=True)
    O['b'][A,C] += scale * -1.00000000 * np.einsum('UV,IWVA,UAJI->WJ', eta1['b'], h['bb'][C,A,A,V], t['bb'][pA,pV,hC,hC], optimize=True)
    
    O['b'][A,C] += scale * +0.25000000 * np.einsum('UV,WUXY,ZRXY,ZRIV->WI', eta1['b'], h['bb'][A,A,A,A], lambdas['bb'], t['bb'][pA,pA,hC,hA], optimize=True)
    
    O['b'][A,C] += scale * +1.00000000 * np.einsum('UV,WXVY,XZYR,UZIR->WI', eta1['b'], h['bb'][A,A,A,A], lambdas['bb'], t['bb'][pA,pA,hC,hA], optimize=True)
    
    O['b'][A,C] += scale * +0.50000000 * np.einsum('UV,UWXY,WZXY,RZIV->RI', eta1['b'], h['bb'][A,A,A,A], lambdas['bb'], t['bb'][pA,pA,hC,hA], optimize=True)
    
    O['b'][A,C] += scale * -0.50000000 * np.einsum('UV,WXVY,WXYZ,RUIZ->RI', eta1['b'], h['bb'][A,A,A,A], lambdas['bb'], t['bb'][pA,pA,hC,hA], optimize=True)
    O['b'][A,C] += scale * -0.50000000 * np.einsum('UV,WX,UWIA,YAVX->YI', gamma1['b'], gamma1['b'], h['bb'][A,A,C,V], t['bb'][pA,pV,hA,hA], optimize=True)
    O['b'][A,C] += scale * -1.00000000 * np.einsum('UV,IUJA,WAIV->WJ', gamma1['b'], h['bb'][C,A,C,V], t['bb'][pA,pV,hC,hA], optimize=True)
    
    O['b'][A,C] += scale * +0.25000000 * np.einsum('UV,WUXY,ZRXY,ZRIV->WI', gamma1['b'], h['bb'][A,A,A,A], lambdas['bb'], t['bb'][pA,pA,hC,hA], optimize=True)
    O['b'][A,C] += scale * +0.50000000 * np.einsum('UV,WUAB,ABIV->WI', gamma1['b'], h['bb'][A,A,V,V], t['bb'][pV,pV,hC,hA], optimize=True)
    
    O['b'][A,C] += scale * +1.00000000 * np.einsum('UV,WXVY,XZYR,UZIR->WI', gamma1['b'], h['bb'][A,A,A,A], lambdas['bb'], t['bb'][pA,pA,hC,hA], optimize=True)
    
    O['b'][A,C] += scale * +0.50000000 * np.einsum('UV,UWXY,WZXY,RZIV->RI', gamma1['b'], h['bb'][A,A,A,A], lambdas['bb'], t['bb'][pA,pA,hC,hA], optimize=True)
    
    O['b'][A,C] += scale * -0.50000000 * np.einsum('UV,WXVY,WXYZ,RUIZ->RI', gamma1['b'], h['bb'][A,A,A,A], lambdas['bb'], t['bb'][pA,pA,hC,hA], optimize=True)
    O['b'][A,C] += scale * -0.50000000 * np.einsum('IJKA,UAIJ->UK', h['bb'][C,C,C,V], t['bb'][pA,pV,hC,hC], optimize=True)
    O['b'][A,C] += scale * +0.50000000 * np.einsum('IUJV,WXVY,WXIY->UJ', h['bb'][C,A,C,A], lambdas['bb'], t['bb'][pA,pA,hC,hA], optimize=True)
    O['b'][A,C] += scale * -0.25000000 * np.einsum('IUVW,XYVW,XYJI->UJ', h['bb'][C,A,A,A], lambdas['bb'], t['bb'][pA,pA,hC,hC], optimize=True)
    O['b'][A,C] += scale * -0.50000000 * np.einsum('IUAB,ABJI->UJ', h['bb'][C,A,V,V], t['bb'][pV,pV,hC,hC], optimize=True)
    O['b'][A,C] += scale * -1.00000000 * np.einsum('IUJV,UWVX,YWIX->YJ', h['bb'][C,A,C,A], lambdas['bb'], t['bb'][pA,pA,hC,hA], optimize=True)
    O['b'][A,C] += scale * +0.50000000 * np.einsum('IUVW,UXVW,YXJI->YJ', h['bb'][C,A,A,A], lambdas['bb'], t['bb'][pA,pA,hC,hC], optimize=True)
    O['b'][A,C] += scale * -0.50000000 * np.einsum('UVIA,VWXY,WAXY->UI', h['bb'][A,A,C,V], lambdas['bb'], t['bb'][pA,pV,hA,hA], optimize=True)
    O['b'][A,C] += scale * +1.00000000 * np.einsum('UVWA,VXWY,XAIY->UI', h['bb'][A,A,A,V], lambdas['bb'], t['bb'][pA,pV,hC,hA], optimize=True)
    O['b'][A,C] += scale * -0.25000000 * np.einsum('UVIA,UVWX,YAWX->YI', h['bb'][A,A,C,V], lambdas['bb'], t['bb'][pA,pV,hA,hA], optimize=True)
    O['b'][A,C] += scale * +0.50000000 * np.einsum('UVWA,UVWX,YAIX->YI', h['bb'][A,A,A,V], lambdas['bb'], t['bb'][pA,pV,hC,hA], optimize=True)
    O['b'][C,C] += scale * +0.50000000 * np.einsum('UV,WX,YZ,IYVX,UWJZ->IJ', eta1['b'], eta1['b'], gamma1['b'], h['bb'][C,A,A,A], t['bb'][pA,pA,hC,hA], optimize=True)
    O['b'][C,C] += scale * +0.50000000 * np.einsum('UV,WX,IJVX,UWKJ->IK', eta1['b'], eta1['b'], h['bb'][C,C,A,A], t['bb'][pA,pA,hC,hC], optimize=True)
    O['b'][C,C] += scale * +0.50000000 * np.einsum('UV,WX,YZ,IUXZ,WYJV->IJ', eta1['b'], gamma1['b'], gamma1['b'], h['bb'][C,A,A,A], t['bb'][pA,pA,hC,hA], optimize=True)
    O['b'][C,C] += scale * +1.00000000 * np.einsum('UV,WX,IWVA,UAJX->IJ', eta1['b'], gamma1['b'], h['bb'][C,A,A,V], t['bb'][pA,pV,hC,hA], optimize=True)
    O['b'][C,C] += scale * +1.00000000 * np.einsum('UV,IJVA,UAKJ->IK', eta1['b'], h['bb'][C,C,A,V], t['bb'][pA,pV,hC,hC], optimize=True)
    
    O['b'][C,C] += scale * +0.25000000 * np.einsum('UV,IUWX,YZWX,YZJV->IJ', eta1['b'], h['bb'][C,A,A,A], lambdas['bb'], t['bb'][pA,pA,hC,hA], optimize=True)
    
    O['b'][C,C] += scale * +1.00000000 * np.einsum('UV,IWVX,WYXZ,UYJZ->IJ', eta1['b'], h['bb'][C,A,A,A], lambdas['bb'], t['bb'][pA,pA,hC,hA], optimize=True)
    
    O['b'][C,C] += scale * +0.25000000 * np.einsum('UV,IUWX,YZWX,YZJV->IJ', gamma1['b'], h['bb'][C,A,A,A], lambdas['bb'], t['bb'][pA,pA,hC,hA], optimize=True)
    O['b'][C,C] += scale * +0.50000000 * np.einsum('UV,IUAB,ABJV->IJ', gamma1['b'], h['bb'][C,A,V,V], t['bb'][pV,pV,hC,hA], optimize=True)
    
    O['b'][C,C] += scale * +1.00000000 * np.einsum('UV,IWVX,WYXZ,UYJZ->IJ', gamma1['b'], h['bb'][C,A,A,A], lambdas['bb'], t['bb'][pA,pA,hC,hA], optimize=True)
    O['b'][C,C] += scale * -0.50000000 * np.einsum('IJKU,VWUX,VWJX->IK', h['bb'][C,C,C,A], lambdas['bb'], t['bb'][pA,pA,hC,hA], optimize=True)
    O['b'][C,C] += scale * +0.25000000 * np.einsum('IJUV,WXUV,WXKJ->IK', h['bb'][C,C,A,A], lambdas['bb'], t['bb'][pA,pA,hC,hC], optimize=True)
    O['b'][C,C] += scale * +0.50000000 * np.einsum('IJAB,ABKJ->IK', h['bb'][C,C,V,V], t['bb'][pV,pV,hC,hC], optimize=True)
    O['b'][C,C] += scale * -0.50000000 * np.einsum('IUJA,UVWX,VAWX->IJ', h['bb'][C,A,C,V], lambdas['bb'], t['bb'][pA,pV,hA,hA], optimize=True)
    O['b'][C,C] += scale * +1.00000000 * np.einsum('IUVA,UWVX,WAJX->IJ', h['bb'][C,A,A,V], lambdas['bb'], t['bb'][pA,pV,hC,hA], optimize=True)
    O['b'][V,C] += scale * +0.50000000 * np.einsum('UV,WX,YZ,UWIZ,YAVX->AI', eta1['b'], eta1['b'], gamma1['b'], h['bb'][A,A,C,A], t['bb'][pA,pV,hA,hA], optimize=True)
    O['b'][V,C] += scale * -0.50000000 * np.einsum('UV,WX,YZ,YAVX,UWIZ->AI', eta1['b'], eta1['b'], gamma1['b'], h['bb'][A,V,A,A], t['bb'][pA,pA,hC,hA], optimize=True)
    O['b'][V,C] += scale * -0.50000000 * np.einsum('UV,WX,IAVX,UWJI->AJ', eta1['b'], eta1['b'], h['bb'][C,V,A,A], t['bb'][pA,pA,hC,hC], optimize=True)
    O['b'][V,C] += scale * -0.50000000 * np.einsum('UV,WX,YZ,UAXZ,WYIV->AI', eta1['b'], gamma1['b'], gamma1['b'], h['bb'][A,V,A,A], t['bb'][pA,pA,hC,hA], optimize=True)
    O['b'][V,C] += scale * +0.50000000 * np.einsum('UV,WX,YZ,WYIV,UAXZ->AI', eta1['b'], gamma1['b'], gamma1['b'], h['bb'][A,A,C,A], t['bb'][pA,pV,hA,hA], optimize=True)
    O['b'][V,C] += scale * +1.00000000 * np.einsum('UV,WX,IWJV,UAIX->AJ', eta1['b'], gamma1['b'], h['bb'][C,A,C,A], t['bb'][pA,pV,hC,hA], optimize=True)
    O['b'][V,C] += scale * -1.00000000 * np.einsum('UV,WX,WAVB,UBIX->AI', eta1['b'], gamma1['b'], h['bb'][A,V,A,V], t['bb'][pA,pV,hC,hA], optimize=True)
    O['b'][V,C] += scale * +0.50000000 * np.einsum('UV,IJKV,UAIJ->AK', eta1['b'], h['bb'][C,C,C,A], t['bb'][pA,pV,hC,hC], optimize=True)
    O['b'][V,C] += scale * -1.00000000 * np.einsum('UV,IAVB,UBJI->AJ', eta1['b'], h['bb'][C,V,A,V], t['bb'][pA,pV,hC,hC], optimize=True)
    O['b'][V,C] += scale * +1.00000000 * np.einsum('UV,UWIX,WYXZ,YAVZ->AI', eta1['b'], h['bb'][A,A,C,A], lambdas['bb'], t['bb'][pA,pV,hA,hA], optimize=True)
    O['b'][V,C] += scale * -0.50000000 * np.einsum('UV,UWXY,WZXY,ZAIV->AI', eta1['b'], h['bb'][A,A,A,A], lambdas['bb'], t['bb'][pA,pV,hC,hA], optimize=True)
    
    O['b'][V,C] += scale * -0.25000000 * np.einsum('UV,UAWX,YZWX,YZIV->AI', eta1['b'], h['bb'][A,V,A,A], lambdas['bb'], t['bb'][pA,pA,hC,hA], optimize=True)
    O['b'][V,C] += scale * +0.25000000 * np.einsum('UV,WXIV,WXYZ,UAYZ->AI', eta1['b'], h['bb'][A,A,C,A], lambdas['bb'], t['bb'][pA,pV,hA,hA], optimize=True)
    O['b'][V,C] += scale * +0.50000000 * np.einsum('UV,WXVY,WXYZ,UAIZ->AI', eta1['b'], h['bb'][A,A,A,A], lambdas['bb'], t['bb'][pA,pV,hC,hA], optimize=True)
    
    O['b'][V,C] += scale * -1.00000000 * np.einsum('UV,WAVX,WYXZ,UYIZ->AI', eta1['b'], h['bb'][A,V,A,A], lambdas['bb'], t['bb'][pA,pA,hC,hA], optimize=True)
    O['b'][V,C] += scale * -0.50000000 * np.einsum('UV,WX,UWIA,BAVX->BI', gamma1['b'], gamma1['b'], h['bb'][A,A,C,V], t['bb'][pV,pV,hA,hA], optimize=True)
    O['b'][V,C] += scale * -1.00000000 * np.einsum('UV,IUJA,BAIV->BJ', gamma1['b'], h['bb'][C,A,C,V], t['bb'][pV,pV,hC,hA], optimize=True)
    O['b'][V,C] += scale * +1.00000000 * np.einsum('UV,UWIX,WYXZ,YAVZ->AI', gamma1['b'], h['bb'][A,A,C,A], lambdas['bb'], t['bb'][pA,pV,hA,hA], optimize=True)
    O['b'][V,C] += scale * -0.50000000 * np.einsum('UV,UWXY,WZXY,ZAIV->AI', gamma1['b'], h['bb'][A,A,A,A], lambdas['bb'], t['bb'][pA,pV,hC,hA], optimize=True)
    
    O['b'][V,C] += scale * -0.25000000 * np.einsum('UV,UAWX,YZWX,YZIV->AI', gamma1['b'], h['bb'][A,V,A,A], lambdas['bb'], t['bb'][pA,pA,hC,hA], optimize=True)
    O['b'][V,C] += scale * -0.50000000 * np.einsum('UV,UABC,BCIV->AI', gamma1['b'], h['bb'][A,V,V,V], t['bb'][pV,pV,hC,hA], optimize=True)
    O['b'][V,C] += scale * +0.25000000 * np.einsum('UV,WXIV,WXYZ,UAYZ->AI', gamma1['b'], h['bb'][A,A,C,A], lambdas['bb'], t['bb'][pA,pV,hA,hA], optimize=True)
    O['b'][V,C] += scale * +0.50000000 * np.einsum('UV,WXVY,WXYZ,UAIZ->AI', gamma1['b'], h['bb'][A,A,A,A], lambdas['bb'], t['bb'][pA,pV,hC,hA], optimize=True)
    
    O['b'][V,C] += scale * -1.00000000 * np.einsum('UV,WAVX,WYXZ,UYIZ->AI', gamma1['b'], h['bb'][A,V,A,A], lambdas['bb'], t['bb'][pA,pA,hC,hA], optimize=True)
    O['b'][V,C] += scale * -0.50000000 * np.einsum('IJKA,BAIJ->BK', h['bb'][C,C,C,V], t['bb'][pV,pV,hC,hC], optimize=True)
    O['b'][V,C] += scale * +1.00000000 * np.einsum('IUJV,UWVX,WAIX->AJ', h['bb'][C,A,C,A], lambdas['bb'], t['bb'][pA,pV,hC,hA], optimize=True)
    O['b'][V,C] += scale * -0.50000000 * np.einsum('IUVW,UXVW,XAJI->AJ', h['bb'][C,A,A,A], lambdas['bb'], t['bb'][pA,pV,hC,hC], optimize=True)
    O['b'][V,C] += scale * +0.50000000 * np.einsum('IAJU,VWUX,VWIX->AJ', h['bb'][C,V,C,A], lambdas['bb'], t['bb'][pA,pA,hC,hA], optimize=True)
    O['b'][V,C] += scale * -0.25000000 * np.einsum('IAUV,WXUV,WXJI->AJ', h['bb'][C,V,A,A], lambdas['bb'], t['bb'][pA,pA,hC,hC], optimize=True)
    O['b'][V,C] += scale * -0.50000000 * np.einsum('IABC,BCJI->AJ', h['bb'][C,V,V,V], t['bb'][pV,pV,hC,hC], optimize=True)
    O['b'][V,C] += scale * -0.25000000 * np.einsum('UVIA,UVWX,BAWX->BI', h['bb'][A,A,C,V], lambdas['bb'], t['bb'][pV,pV,hA,hA], optimize=True)
    O['b'][V,C] += scale * +0.50000000 * np.einsum('UVWA,UVWX,BAIX->BI', h['bb'][A,A,A,V], lambdas['bb'], t['bb'][pV,pV,hC,hA], optimize=True)
    O['b'][V,C] += scale * +0.50000000 * np.einsum('UAIB,UVWX,VBWX->AI', h['bb'][A,V,C,V], lambdas['bb'], t['bb'][pA,pV,hA,hA], optimize=True)
    O['b'][V,C] += scale * -1.00000000 * np.einsum('UAVB,UWVX,WBIX->AI', h['bb'][A,V,A,V], lambdas['bb'], t['bb'][pA,pV,hC,hA], optimize=True)
    
    
    O['b'][A,V] += scale * +1.00000000 * np.einsum('UV,WX,IWVA,YUIX->YA', eta1['b'], gamma1['b'], h['bb'][C,A,A,V], t['bb'][pA,pA,hC,hA], optimize=True)
    O['b'][A,V] += scale * +0.50000000 * np.einsum('UV,IJVA,WUIJ->WA', eta1['b'], h['bb'][C,C,A,V], t['bb'][pA,pA,hC,hC], optimize=True)
    
    
    
    
    O['b'][A,V] += scale * -0.50000000 * np.einsum('UV,WX,UWAB,YBVX->YA', gamma1['b'], gamma1['b'], h['bb'][A,A,V,V], t['bb'][pA,pV,hA,hA], optimize=True)
    O['b'][A,V] += scale * -1.00000000 * np.einsum('UV,IUAB,WBIV->WA', gamma1['b'], h['bb'][C,A,V,V], t['bb'][pA,pV,hC,hA], optimize=True)
    
    
    
    
    O['b'][A,V] += scale * -0.50000000 * np.einsum('IJAB,UBIJ->UA', h['bb'][C,C,V,V], t['bb'][pA,pV,hC,hC], optimize=True)
    O['b'][A,V] += scale * -0.50000000 * np.einsum('IUVA,WXVY,WXIY->UA', h['bb'][C,A,A,V], lambdas['bb'], t['bb'][pA,pA,hC,hA], optimize=True)
    O['b'][A,V] += scale * +1.00000000 * np.einsum('IUVA,UWVX,YWIX->YA', h['bb'][C,A,A,V], lambdas['bb'], t['bb'][pA,pA,hC,hA], optimize=True)
    O['b'][A,V] += scale * -0.50000000 * np.einsum('UVAB,VWXY,WBXY->UA', h['bb'][A,A,V,V], lambdas['bb'], t['bb'][pA,pV,hA,hA], optimize=True)
    O['b'][A,V] += scale * -0.25000000 * np.einsum('UVAB,UVWX,YBWX->YA', h['bb'][A,A,V,V], lambdas['bb'], t['bb'][pA,pV,hA,hA], optimize=True)
    
    
    
    
    O['b'][C,V] += scale * +0.50000000 * np.einsum('IJUA,VWUX,VWJX->IA', h['bb'][C,C,A,V], lambdas['bb'], t['bb'][pA,pA,hC,hA], optimize=True)
    O['b'][C,V] += scale * -0.50000000 * np.einsum('IUAB,UVWX,VBWX->IA', h['bb'][C,A,V,V], lambdas['bb'], t['bb'][pA,pV,hA,hA], optimize=True)
    O['b'][V,V] += scale * -0.50000000 * np.einsum('UV,WX,YZ,UWZA,YBVX->BA', eta1['b'], eta1['b'], gamma1['b'], h['bb'][A,A,A,V], t['bb'][pA,pV,hA,hA], optimize=True)
    O['b'][V,V] += scale * -0.50000000 * np.einsum('UV,WX,YZ,WYVA,UBXZ->BA', eta1['b'], gamma1['b'], gamma1['b'], h['bb'][A,A,A,V], t['bb'][pA,pV,hA,hA], optimize=True)
    O['b'][V,V] += scale * -1.00000000 * np.einsum('UV,WX,IWVA,UBIX->BA', eta1['b'], gamma1['b'], h['bb'][C,A,A,V], t['bb'][pA,pV,hC,hA], optimize=True)
    O['b'][V,V] += scale * -0.50000000 * np.einsum('UV,IJVA,UBIJ->BA', eta1['b'], h['bb'][C,C,A,V], t['bb'][pA,pV,hC,hC], optimize=True)
    O['b'][V,V] += scale * -1.00000000 * np.einsum('UV,UWXA,WYXZ,YBVZ->BA', eta1['b'], h['bb'][A,A,A,V], lambdas['bb'], t['bb'][pA,pV,hA,hA], optimize=True)
    
    O['b'][V,V] += scale * -0.25000000 * np.einsum('UV,WXVA,WXYZ,UBYZ->BA', eta1['b'], h['bb'][A,A,A,V], lambdas['bb'], t['bb'][pA,pV,hA,hA], optimize=True)
    
    O['b'][V,V] += scale * -0.50000000 * np.einsum('UV,WX,UWAB,CBVX->CA', gamma1['b'], gamma1['b'], h['bb'][A,A,V,V], t['bb'][pV,pV,hA,hA], optimize=True)
    O['b'][V,V] += scale * -1.00000000 * np.einsum('UV,IUAB,CBIV->CA', gamma1['b'], h['bb'][C,A,V,V], t['bb'][pV,pV,hC,hA], optimize=True)
    O['b'][V,V] += scale * -1.00000000 * np.einsum('UV,UWXA,WYXZ,YBVZ->BA', gamma1['b'], h['bb'][A,A,A,V], lambdas['bb'], t['bb'][pA,pV,hA,hA], optimize=True)
    
    O['b'][V,V] += scale * -0.25000000 * np.einsum('UV,WXVA,WXYZ,UBYZ->BA', gamma1['b'], h['bb'][A,A,A,V], lambdas['bb'], t['bb'][pA,pV,hA,hA], optimize=True)
    
    O['b'][V,V] += scale * -0.50000000 * np.einsum('IJAB,CBIJ->CA', h['bb'][C,C,V,V], t['bb'][pV,pV,hC,hC], optimize=True)
    O['b'][V,V] += scale * -1.00000000 * np.einsum('IUVA,UWVX,WBIX->BA', h['bb'][C,A,A,V], lambdas['bb'], t['bb'][pA,pV,hC,hA], optimize=True)
    O['b'][V,V] += scale * -0.50000000 * np.einsum('IAUB,VWUX,VWIX->AB', h['bb'][C,V,A,V], lambdas['bb'], t['bb'][pA,pA,hC,hA], optimize=True)
    O['b'][V,V] += scale * -0.25000000 * np.einsum('UVAB,UVWX,CBWX->CA', h['bb'][A,A,V,V], lambdas['bb'], t['bb'][pV,pV,hA,hA], optimize=True)
    O['b'][V,V] += scale * +0.50000000 * np.einsum('UABC,UVWX,VCWX->AB', h['bb'][A,V,V,V], lambdas['bb'], t['bb'][pA,pV,hA,hA], optimize=True)

    t1 = time.time()
    if verbose: print("h2c_t2c_c1b took {:.4f} seconds to run.".format(t1-t0))

    return O
def h1a_t2a_c2a(O, h, t, gamma1, eta1, lambdas, orbspace, verbose=False, scale=1.0):
    # 108 lines
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

    
    
    
    
    O['aa'][a,a,a,a] += scale * -0.50000000 * np.einsum('iu,vwix->vwux', h['a'][c,a], t['aa'][pa,pa,hc,ha], optimize=True)
    O['aa'][a,a,a,a] += scale * -0.50000000 * np.einsum('ua,vawx->uvwx', h['a'][a,v], t['aa'][pa,pv,ha,ha], optimize=True)
    
    
    O['aa'][c,a,a,a] += scale * -0.50000000 * np.einsum('ia,uavw->iuvw', h['a'][c,v], t['aa'][pa,pv,ha,ha], optimize=True)
    O['aa'][a,v,a,a] += scale * +0.50000000 * np.einsum('uv,wv,uaxy->waxy', eta1['a'], h['a'][a,a], t['aa'][pa,pv,ha,ha], optimize=True)
    O['aa'][a,v,a,a] += scale * +1.00000000 * np.einsum('uv,uw,xayv->xawy', eta1['a'], h['a'][a,a], t['aa'][pa,pv,ha,ha], optimize=True)
    
    O['aa'][a,v,a,a] += scale * +0.50000000 * np.einsum('uv,wv,uaxy->waxy', gamma1['a'], h['a'][a,a], t['aa'][pa,pv,ha,ha], optimize=True)
    O['aa'][a,v,a,a] += scale * +1.00000000 * np.einsum('uv,uw,xayv->xawy', gamma1['a'], h['a'][a,a], t['aa'][pa,pv,ha,ha], optimize=True)
    
    O['aa'][a,v,a,a] += scale * -1.00000000 * np.einsum('iu,vaiw->vauw', h['a'][c,a], t['aa'][pa,pv,hc,ha], optimize=True)
    O['aa'][a,v,a,a] += scale * -0.50000000 * np.einsum('ua,bavw->ubvw', h['a'][a,v], t['aa'][pv,pv,ha,ha], optimize=True)
    O['aa'][a,v,a,a] += scale * +0.50000000 * np.einsum('ab,ubvw->uavw', h['a'][v,v], t['aa'][pa,pv,ha,ha], optimize=True)
    O['aa'][c,v,a,a] += scale * +0.50000000 * np.einsum('uv,iv,uawx->iawx', eta1['a'], h['a'][c,a], t['aa'][pa,pv,ha,ha], optimize=True)
    O['aa'][c,v,a,a] += scale * +0.50000000 * np.einsum('uv,iv,uawx->iawx', gamma1['a'], h['a'][c,a], t['aa'][pa,pv,ha,ha], optimize=True)
    O['aa'][c,v,a,a] += scale * -0.50000000 * np.einsum('ia,bauv->ibuv', h['a'][c,v], t['aa'][pv,pv,ha,ha], optimize=True)
    O['aa'][v,v,a,a] += scale * +0.50000000 * np.einsum('uv,uw,abxv->abwx', eta1['a'], h['a'][a,a], t['aa'][pv,pv,ha,ha], optimize=True)
    O['aa'][v,v,a,a] += scale * +0.50000000 * np.einsum('uv,av,ubwx->abwx', eta1['a'], h['a'][v,a], t['aa'][pa,pv,ha,ha], optimize=True)
    O['aa'][v,v,a,a] += scale * +0.50000000 * np.einsum('uv,uw,abxv->abwx', gamma1['a'], h['a'][a,a], t['aa'][pv,pv,ha,ha], optimize=True)
    O['aa'][v,v,a,a] += scale * +0.50000000 * np.einsum('uv,av,ubwx->abwx', gamma1['a'], h['a'][v,a], t['aa'][pa,pv,ha,ha], optimize=True)
    O['aa'][v,v,a,a] += scale * -0.50000000 * np.einsum('iu,abiv->abuv', h['a'][c,a], t['aa'][pv,pv,hc,ha], optimize=True)
    O['aa'][v,v,a,a] += scale * -0.50000000 * np.einsum('ab,cbuv->acuv', h['a'][v,v], t['aa'][pv,pv,ha,ha], optimize=True)
    
    
    O['aa'][a,a,a,v] += scale * +0.50000000 * np.einsum('ia,uviw->uvwa', h['a'][c,v], t['aa'][pa,pa,hc,ha], optimize=True)
    O['aa'][a,v,a,v] += scale * -1.00000000 * np.einsum('uv,ua,wbxv->wbxa', eta1['a'], h['a'][a,v], t['aa'][pa,pv,ha,ha], optimize=True)
    O['aa'][a,v,a,v] += scale * -1.00000000 * np.einsum('uv,ua,wbxv->wbxa', gamma1['a'], h['a'][a,v], t['aa'][pa,pv,ha,ha], optimize=True)
    O['aa'][a,v,a,v] += scale * +1.00000000 * np.einsum('ia,ubiv->ubva', h['a'][c,v], t['aa'][pa,pv,hc,ha], optimize=True)
    O['aa'][v,v,a,v] += scale * -0.50000000 * np.einsum('uv,ua,bcwv->bcwa', eta1['a'], h['a'][a,v], t['aa'][pv,pv,ha,ha], optimize=True)
    O['aa'][v,v,a,v] += scale * -0.50000000 * np.einsum('uv,ua,bcwv->bcwa', gamma1['a'], h['a'][a,v], t['aa'][pv,pv,ha,ha], optimize=True)
    O['aa'][v,v,a,v] += scale * +0.50000000 * np.einsum('ia,bciu->bcua', h['a'][c,v], t['aa'][pv,pv,hc,ha], optimize=True)
    O['aa'][a,a,c,a] += scale * -1.00000000 * np.einsum('uv,wv,xuiy->wxiy', eta1['a'], h['a'][a,a], t['aa'][pa,pa,hc,ha], optimize=True)
    
    O['aa'][a,a,c,a] += scale * -0.50000000 * np.einsum('uv,uw,xyiv->xyiw', eta1['a'], h['a'][a,a], t['aa'][pa,pa,hc,ha], optimize=True)
    O['aa'][a,a,c,a] += scale * -1.00000000 * np.einsum('uv,wv,xuiy->wxiy', gamma1['a'], h['a'][a,a], t['aa'][pa,pa,hc,ha], optimize=True)
    
    O['aa'][a,a,c,a] += scale * -0.50000000 * np.einsum('uv,uw,xyiv->xyiw', gamma1['a'], h['a'][a,a], t['aa'][pa,pa,hc,ha], optimize=True)
    O['aa'][a,a,c,a] += scale * -0.50000000 * np.einsum('ij,uviw->uvjw', h['a'][c,c], t['aa'][pa,pa,hc,ha], optimize=True)
    O['aa'][a,a,c,a] += scale * -0.50000000 * np.einsum('iu,vwji->vwju', h['a'][c,a], t['aa'][pa,pa,hc,hc], optimize=True)
    O['aa'][a,a,c,a] += scale * -1.00000000 * np.einsum('ua,vaiw->uviw', h['a'][a,v], t['aa'][pa,pv,hc,ha], optimize=True)
    O['aa'][c,a,c,a] += scale * -1.00000000 * np.einsum('uv,iv,wujx->iwjx', eta1['a'], h['a'][c,a], t['aa'][pa,pa,hc,ha], optimize=True)
    O['aa'][c,a,c,a] += scale * -1.00000000 * np.einsum('uv,iv,wujx->iwjx', gamma1['a'], h['a'][c,a], t['aa'][pa,pa,hc,ha], optimize=True)
    O['aa'][c,a,c,a] += scale * -1.00000000 * np.einsum('ia,uajv->iujv', h['a'][c,v], t['aa'][pa,pv,hc,ha], optimize=True)
    O['aa'][a,v,c,a] += scale * +1.00000000 * np.einsum('uv,wv,uaix->waix', eta1['a'], h['a'][a,a], t['aa'][pa,pv,hc,ha], optimize=True)
    O['aa'][a,v,c,a] += scale * +1.00000000 * np.einsum('uv,ui,waxv->waix', eta1['a'], h['a'][a,c], t['aa'][pa,pv,ha,ha], optimize=True)
    O['aa'][a,v,c,a] += scale * -1.00000000 * np.einsum('uv,uw,xaiv->xaiw', eta1['a'], h['a'][a,a], t['aa'][pa,pv,hc,ha], optimize=True)
    O['aa'][a,v,c,a] += scale * +1.00000000 * np.einsum('uv,av,wuix->waix', eta1['a'], h['a'][v,a], t['aa'][pa,pa,hc,ha], optimize=True)
    O['aa'][a,v,c,a] += scale * +1.00000000 * np.einsum('uv,wv,uaix->waix', gamma1['a'], h['a'][a,a], t['aa'][pa,pv,hc,ha], optimize=True)
    O['aa'][a,v,c,a] += scale * +1.00000000 * np.einsum('uv,ui,waxv->waix', gamma1['a'], h['a'][a,c], t['aa'][pa,pv,ha,ha], optimize=True)
    O['aa'][a,v,c,a] += scale * -1.00000000 * np.einsum('uv,uw,xaiv->xaiw', gamma1['a'], h['a'][a,a], t['aa'][pa,pv,hc,ha], optimize=True)
    O['aa'][a,v,c,a] += scale * +1.00000000 * np.einsum('uv,av,wuix->waix', gamma1['a'], h['a'][v,a], t['aa'][pa,pa,hc,ha], optimize=True)
    O['aa'][a,v,c,a] += scale * -1.00000000 * np.einsum('ij,uaiv->uajv', h['a'][c,c], t['aa'][pa,pv,hc,ha], optimize=True)
    O['aa'][a,v,c,a] += scale * -1.00000000 * np.einsum('iu,vaji->vaju', h['a'][c,a], t['aa'][pa,pv,hc,hc], optimize=True)
    O['aa'][a,v,c,a] += scale * -1.00000000 * np.einsum('ua,baiv->ubiv', h['a'][a,v], t['aa'][pv,pv,hc,ha], optimize=True)
    O['aa'][a,v,c,a] += scale * +1.00000000 * np.einsum('ab,ubiv->uaiv', h['a'][v,v], t['aa'][pa,pv,hc,ha], optimize=True)
    O['aa'][c,v,c,a] += scale * +1.00000000 * np.einsum('uv,iv,uajw->iajw', eta1['a'], h['a'][c,a], t['aa'][pa,pv,hc,ha], optimize=True)
    O['aa'][c,v,c,a] += scale * +1.00000000 * np.einsum('uv,iv,uajw->iajw', gamma1['a'], h['a'][c,a], t['aa'][pa,pv,hc,ha], optimize=True)
    O['aa'][c,v,c,a] += scale * -1.00000000 * np.einsum('ia,baju->ibju', h['a'][c,v], t['aa'][pv,pv,hc,ha], optimize=True)
    O['aa'][v,v,c,a] += scale * +0.50000000 * np.einsum('uv,ui,abwv->abiw', eta1['a'], h['a'][a,c], t['aa'][pv,pv,ha,ha], optimize=True)
    O['aa'][v,v,c,a] += scale * -0.50000000 * np.einsum('uv,uw,abiv->abiw', eta1['a'], h['a'][a,a], t['aa'][pv,pv,hc,ha], optimize=True)
    O['aa'][v,v,c,a] += scale * +1.00000000 * np.einsum('uv,av,ubiw->abiw', eta1['a'], h['a'][v,a], t['aa'][pa,pv,hc,ha], optimize=True)
    O['aa'][v,v,c,a] += scale * +0.50000000 * np.einsum('uv,ui,abwv->abiw', gamma1['a'], h['a'][a,c], t['aa'][pv,pv,ha,ha], optimize=True)
    O['aa'][v,v,c,a] += scale * -0.50000000 * np.einsum('uv,uw,abiv->abiw', gamma1['a'], h['a'][a,a], t['aa'][pv,pv,hc,ha], optimize=True)
    O['aa'][v,v,c,a] += scale * +1.00000000 * np.einsum('uv,av,ubiw->abiw', gamma1['a'], h['a'][v,a], t['aa'][pa,pv,hc,ha], optimize=True)
    O['aa'][v,v,c,a] += scale * -0.50000000 * np.einsum('ij,abiu->abju', h['a'][c,c], t['aa'][pv,pv,hc,ha], optimize=True)
    O['aa'][v,v,c,a] += scale * -0.50000000 * np.einsum('iu,abji->abju', h['a'][c,a], t['aa'][pv,pv,hc,hc], optimize=True)
    O['aa'][v,v,c,a] += scale * -1.00000000 * np.einsum('ab,cbiu->aciu', h['a'][v,v], t['aa'][pv,pv,hc,ha], optimize=True)
    O['aa'][a,a,c,c] += scale * -0.50000000 * np.einsum('uv,wv,xuij->wxij', eta1['a'], h['a'][a,a], t['aa'][pa,pa,hc,hc], optimize=True)
    O['aa'][a,a,c,c] += scale * +0.50000000 * np.einsum('uv,ui,wxjv->wxij', eta1['a'], h['a'][a,c], t['aa'][pa,pa,hc,ha], optimize=True)
    O['aa'][a,a,c,c] += scale * -0.50000000 * np.einsum('uv,wv,xuij->wxij', gamma1['a'], h['a'][a,a], t['aa'][pa,pa,hc,hc], optimize=True)
    O['aa'][a,a,c,c] += scale * +0.50000000 * np.einsum('uv,ui,wxjv->wxij', gamma1['a'], h['a'][a,c], t['aa'][pa,pa,hc,ha], optimize=True)
    O['aa'][a,a,c,c] += scale * +0.50000000 * np.einsum('ij,uvki->uvjk', h['a'][c,c], t['aa'][pa,pa,hc,hc], optimize=True)
    O['aa'][a,a,c,c] += scale * -0.50000000 * np.einsum('ua,vaij->uvij', h['a'][a,v], t['aa'][pa,pv,hc,hc], optimize=True)
    O['aa'][c,a,c,c] += scale * -0.50000000 * np.einsum('uv,iv,wujk->iwjk', eta1['a'], h['a'][c,a], t['aa'][pa,pa,hc,hc], optimize=True)
    O['aa'][c,a,c,c] += scale * -0.50000000 * np.einsum('uv,iv,wujk->iwjk', gamma1['a'], h['a'][c,a], t['aa'][pa,pa,hc,hc], optimize=True)
    O['aa'][c,a,c,c] += scale * -0.50000000 * np.einsum('ia,uajk->iujk', h['a'][c,v], t['aa'][pa,pv,hc,hc], optimize=True)
    O['aa'][a,v,c,c] += scale * +0.50000000 * np.einsum('uv,wv,uaij->waij', eta1['a'], h['a'][a,a], t['aa'][pa,pv,hc,hc], optimize=True)
    O['aa'][a,v,c,c] += scale * +1.00000000 * np.einsum('uv,ui,wajv->waij', eta1['a'], h['a'][a,c], t['aa'][pa,pv,hc,ha], optimize=True)
    O['aa'][a,v,c,c] += scale * +0.50000000 * np.einsum('uv,av,wuij->waij', eta1['a'], h['a'][v,a], t['aa'][pa,pa,hc,hc], optimize=True)
    O['aa'][a,v,c,c] += scale * +0.50000000 * np.einsum('uv,wv,uaij->waij', gamma1['a'], h['a'][a,a], t['aa'][pa,pv,hc,hc], optimize=True)
    O['aa'][a,v,c,c] += scale * +1.00000000 * np.einsum('uv,ui,wajv->waij', gamma1['a'], h['a'][a,c], t['aa'][pa,pv,hc,ha], optimize=True)
    O['aa'][a,v,c,c] += scale * +0.50000000 * np.einsum('uv,av,wuij->waij', gamma1['a'], h['a'][v,a], t['aa'][pa,pa,hc,hc], optimize=True)
    O['aa'][a,v,c,c] += scale * +1.00000000 * np.einsum('ij,uaki->uajk', h['a'][c,c], t['aa'][pa,pv,hc,hc], optimize=True)
    O['aa'][a,v,c,c] += scale * -0.50000000 * np.einsum('ua,baij->ubij', h['a'][a,v], t['aa'][pv,pv,hc,hc], optimize=True)
    O['aa'][a,v,c,c] += scale * +0.50000000 * np.einsum('ab,ubij->uaij', h['a'][v,v], t['aa'][pa,pv,hc,hc], optimize=True)
    O['aa'][c,v,c,c] += scale * +0.50000000 * np.einsum('uv,iv,uajk->iajk', eta1['a'], h['a'][c,a], t['aa'][pa,pv,hc,hc], optimize=True)
    O['aa'][c,v,c,c] += scale * +0.50000000 * np.einsum('uv,iv,uajk->iajk', gamma1['a'], h['a'][c,a], t['aa'][pa,pv,hc,hc], optimize=True)
    O['aa'][c,v,c,c] += scale * -0.50000000 * np.einsum('ia,bajk->ibjk', h['a'][c,v], t['aa'][pv,pv,hc,hc], optimize=True)
    O['aa'][v,v,c,c] += scale * +0.50000000 * np.einsum('uv,ui,abjv->abij', eta1['a'], h['a'][a,c], t['aa'][pv,pv,hc,ha], optimize=True)
    O['aa'][v,v,c,c] += scale * +0.50000000 * np.einsum('uv,av,ubij->abij', eta1['a'], h['a'][v,a], t['aa'][pa,pv,hc,hc], optimize=True)
    O['aa'][v,v,c,c] += scale * +0.50000000 * np.einsum('uv,ui,abjv->abij', gamma1['a'], h['a'][a,c], t['aa'][pv,pv,hc,ha], optimize=True)
    O['aa'][v,v,c,c] += scale * +0.50000000 * np.einsum('uv,av,ubij->abij', gamma1['a'], h['a'][v,a], t['aa'][pa,pv,hc,hc], optimize=True)
    O['aa'][v,v,c,c] += scale * +0.50000000 * np.einsum('ij,abki->abjk', h['a'][c,c], t['aa'][pv,pv,hc,hc], optimize=True)
    O['aa'][v,v,c,c] += scale * -0.50000000 * np.einsum('ab,cbij->acij', h['a'][v,v], t['aa'][pv,pv,hc,hc], optimize=True)
    O['aa'][a,a,c,v] += scale * -0.50000000 * np.einsum('uv,ua,wxiv->wxia', eta1['a'], h['a'][a,v], t['aa'][pa,pa,hc,ha], optimize=True)
    O['aa'][a,a,c,v] += scale * -0.50000000 * np.einsum('uv,ua,wxiv->wxia', gamma1['a'], h['a'][a,v], t['aa'][pa,pa,hc,ha], optimize=True)
    O['aa'][a,a,c,v] += scale * -0.50000000 * np.einsum('ia,uvji->uvja', h['a'][c,v], t['aa'][pa,pa,hc,hc], optimize=True)
    O['aa'][a,v,c,v] += scale * -1.00000000 * np.einsum('uv,ua,wbiv->wbia', eta1['a'], h['a'][a,v], t['aa'][pa,pv,hc,ha], optimize=True)
    O['aa'][a,v,c,v] += scale * -1.00000000 * np.einsum('uv,ua,wbiv->wbia', gamma1['a'], h['a'][a,v], t['aa'][pa,pv,hc,ha], optimize=True)
    O['aa'][a,v,c,v] += scale * -1.00000000 * np.einsum('ia,ubji->ubja', h['a'][c,v], t['aa'][pa,pv,hc,hc], optimize=True)
    O['aa'][v,v,c,v] += scale * -0.50000000 * np.einsum('uv,ua,bciv->bcia', eta1['a'], h['a'][a,v], t['aa'][pv,pv,hc,ha], optimize=True)
    O['aa'][v,v,c,v] += scale * -0.50000000 * np.einsum('uv,ua,bciv->bcia', gamma1['a'], h['a'][a,v], t['aa'][pv,pv,hc,ha], optimize=True)
    O['aa'][v,v,c,v] += scale * -0.50000000 * np.einsum('ia,bcji->bcja', h['a'][c,v], t['aa'][pv,pv,hc,hc], optimize=True)

    t1 = time.time()
    if verbose: print("h1a_t2a_c2a took {:.4f} seconds to run.".format(t1-t0))

    return O
def h2a_t1a_c2a(O, h, t, gamma1, eta1, lambdas, orbspace, verbose=False, scale=1.0):
    # 216 lines
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

    
    
    
    
    O['aa'][a,a,a,a] += scale * +0.50000000 * np.einsum('iuvw,xi->uxvw', h['aa'][c,a,a,a], t['a'][pa,hc], optimize=True)
    O['aa'][a,a,a,a] += scale * +0.50000000 * np.einsum('uvwa,ax->uvwx', h['aa'][a,a,a,v], t['a'][pv,ha], optimize=True)
    
    
    
    
    O['aa'][c,a,a,a] += scale * -0.50000000 * np.einsum('ijuv,wj->iwuv', h['aa'][c,c,a,a], t['a'][pa,hc], optimize=True)
    O['aa'][c,a,a,a] += scale * +1.00000000 * np.einsum('iuva,aw->iuvw', h['aa'][c,a,a,v], t['a'][pv,ha], optimize=True)
    
    
    O['aa'][c,c,a,a] += scale * +0.50000000 * np.einsum('ijua,av->ijuv', h['aa'][c,c,a,v], t['a'][pv,ha], optimize=True)
    O['aa'][a,v,a,a] += scale * -0.50000000 * np.einsum('uv,wuxy,av->waxy', eta1['a'], h['aa'][a,a,a,a], t['a'][pv,ha], optimize=True)
    
    
    O['aa'][a,v,a,a] += scale * -0.50000000 * np.einsum('uv,wuxy,av->waxy', gamma1['a'], h['aa'][a,a,a,a], t['a'][pv,ha], optimize=True)
    
    
    O['aa'][a,v,a,a] += scale * +0.50000000 * np.einsum('iuvw,ai->uavw', h['aa'][c,a,a,a], t['a'][pv,hc], optimize=True)
    O['aa'][a,v,a,a] += scale * -0.50000000 * np.einsum('iauv,wi->wauv', h['aa'][c,v,a,a], t['a'][pa,hc], optimize=True)
    O['aa'][a,v,a,a] += scale * +1.00000000 * np.einsum('uavb,bw->uavw', h['aa'][a,v,a,v], t['a'][pv,ha], optimize=True)
    O['aa'][c,v,a,a] += scale * -0.50000000 * np.einsum('uv,iuwx,av->iawx', eta1['a'], h['aa'][c,a,a,a], t['a'][pv,ha], optimize=True)
    
    O['aa'][c,v,a,a] += scale * -0.50000000 * np.einsum('uv,iuwx,av->iawx', gamma1['a'], h['aa'][c,a,a,a], t['a'][pv,ha], optimize=True)
    
    O['aa'][c,v,a,a] += scale * -0.50000000 * np.einsum('ijuv,aj->iauv', h['aa'][c,c,a,a], t['a'][pv,hc], optimize=True)
    O['aa'][c,v,a,a] += scale * +1.00000000 * np.einsum('iaub,bv->iauv', h['aa'][c,v,a,v], t['a'][pv,ha], optimize=True)
    O['aa'][v,v,a,a] += scale * +0.50000000 * np.einsum('uv,uawx,bv->abwx', eta1['a'], h['aa'][a,v,a,a], t['a'][pv,ha], optimize=True)
    
    O['aa'][v,v,a,a] += scale * +0.50000000 * np.einsum('uv,uawx,bv->abwx', gamma1['a'], h['aa'][a,v,a,a], t['a'][pv,ha], optimize=True)
    
    O['aa'][v,v,a,a] += scale * +0.50000000 * np.einsum('iauv,bi->abuv', h['aa'][c,v,a,a], t['a'][pv,hc], optimize=True)
    O['aa'][v,v,a,a] += scale * +0.50000000 * np.einsum('abuc,cv->abuv', h['aa'][v,v,a,v], t['a'][pv,ha], optimize=True)
    
    
    
    
    O['aa'][a,a,a,v] += scale * +1.00000000 * np.einsum('iuva,wi->uwva', h['aa'][c,a,a,v], t['a'][pa,hc], optimize=True)
    O['aa'][a,a,a,v] += scale * -0.50000000 * np.einsum('uvab,bw->uvwa', h['aa'][a,a,v,v], t['a'][pv,ha], optimize=True)
    
    
    
    
    O['aa'][c,a,a,v] += scale * -1.00000000 * np.einsum('ijua,vj->ivua', h['aa'][c,c,a,v], t['a'][pa,hc], optimize=True)
    O['aa'][c,a,a,v] += scale * -1.00000000 * np.einsum('iuab,bv->iuva', h['aa'][c,a,v,v], t['a'][pv,ha], optimize=True)
    
    
    O['aa'][c,c,a,v] += scale * -0.50000000 * np.einsum('ijab,bu->ijua', h['aa'][c,c,v,v], t['a'][pv,ha], optimize=True)
    O['aa'][a,v,a,v] += scale * -1.00000000 * np.einsum('uv,wuxa,bv->wbxa', eta1['a'], h['aa'][a,a,a,v], t['a'][pv,ha], optimize=True)
    
    
    O['aa'][a,v,a,v] += scale * -1.00000000 * np.einsum('uv,wuxa,bv->wbxa', gamma1['a'], h['aa'][a,a,a,v], t['a'][pv,ha], optimize=True)
    
    
    O['aa'][a,v,a,v] += scale * +1.00000000 * np.einsum('iuva,bi->ubva', h['aa'][c,a,a,v], t['a'][pv,hc], optimize=True)
    O['aa'][a,v,a,v] += scale * -1.00000000 * np.einsum('iaub,vi->vaub', h['aa'][c,v,a,v], t['a'][pa,hc], optimize=True)
    O['aa'][a,v,a,v] += scale * -1.00000000 * np.einsum('uabc,cv->uavb', h['aa'][a,v,v,v], t['a'][pv,ha], optimize=True)
    O['aa'][c,v,a,v] += scale * -1.00000000 * np.einsum('uv,iuwa,bv->ibwa', eta1['a'], h['aa'][c,a,a,v], t['a'][pv,ha], optimize=True)
    
    O['aa'][c,v,a,v] += scale * -1.00000000 * np.einsum('uv,iuwa,bv->ibwa', gamma1['a'], h['aa'][c,a,a,v], t['a'][pv,ha], optimize=True)
    
    O['aa'][c,v,a,v] += scale * -1.00000000 * np.einsum('ijua,bj->ibua', h['aa'][c,c,a,v], t['a'][pv,hc], optimize=True)
    O['aa'][c,v,a,v] += scale * -1.00000000 * np.einsum('iabc,cu->iaub', h['aa'][c,v,v,v], t['a'][pv,ha], optimize=True)
    O['aa'][v,v,a,v] += scale * +1.00000000 * np.einsum('uv,uawb,cv->acwb', eta1['a'], h['aa'][a,v,a,v], t['a'][pv,ha], optimize=True)
    
    O['aa'][v,v,a,v] += scale * +1.00000000 * np.einsum('uv,uawb,cv->acwb', gamma1['a'], h['aa'][a,v,a,v], t['a'][pv,ha], optimize=True)
    
    O['aa'][v,v,a,v] += scale * +1.00000000 * np.einsum('iaub,ci->acub', h['aa'][c,v,a,v], t['a'][pv,hc], optimize=True)
    O['aa'][v,v,a,v] += scale * -0.50000000 * np.einsum('abcd,du->abuc', h['aa'][v,v,v,v], t['a'][pv,ha], optimize=True)
    
    O['aa'][a,a,c,a] += scale * -0.50000000 * np.einsum('uv,wxyv,ui->wxiy', eta1['a'], h['aa'][a,a,a,a], t['a'][pa,hc], optimize=True)
    
    
    O['aa'][a,a,c,a] += scale * -0.50000000 * np.einsum('uv,wxyv,ui->wxiy', gamma1['a'], h['aa'][a,a,a,a], t['a'][pa,hc], optimize=True)
    
    O['aa'][a,a,c,a] += scale * +1.00000000 * np.einsum('iujv,wi->uwjv', h['aa'][c,a,c,a], t['a'][pa,hc], optimize=True)
    O['aa'][a,a,c,a] += scale * +0.50000000 * np.einsum('uvia,aw->uviw', h['aa'][a,a,c,v], t['a'][pv,ha], optimize=True)
    O['aa'][a,a,c,a] += scale * -0.50000000 * np.einsum('uvwa,ai->uviw', h['aa'][a,a,a,v], t['a'][pv,hc], optimize=True)
    
    O['aa'][c,a,c,a] += scale * -1.00000000 * np.einsum('uv,iwxv,uj->iwjx', eta1['a'], h['aa'][c,a,a,a], t['a'][pa,hc], optimize=True)
    
    
    O['aa'][c,a,c,a] += scale * -1.00000000 * np.einsum('uv,iwxv,uj->iwjx', gamma1['a'], h['aa'][c,a,a,a], t['a'][pa,hc], optimize=True)
    
    O['aa'][c,a,c,a] += scale * -1.00000000 * np.einsum('ijku,vj->ivku', h['aa'][c,c,c,a], t['a'][pa,hc], optimize=True)
    O['aa'][c,a,c,a] += scale * +1.00000000 * np.einsum('iuja,av->iujv', h['aa'][c,a,c,v], t['a'][pv,ha], optimize=True)
    O['aa'][c,a,c,a] += scale * -1.00000000 * np.einsum('iuva,aj->iujv', h['aa'][c,a,a,v], t['a'][pv,hc], optimize=True)
    
    O['aa'][c,c,c,a] += scale * -0.50000000 * np.einsum('uv,ijwv,uk->ijkw', eta1['a'], h['aa'][c,c,a,a], t['a'][pa,hc], optimize=True)
    
    O['aa'][c,c,c,a] += scale * -0.50000000 * np.einsum('uv,ijwv,uk->ijkw', gamma1['a'], h['aa'][c,c,a,a], t['a'][pa,hc], optimize=True)
    O['aa'][c,c,c,a] += scale * +0.50000000 * np.einsum('ijka,au->ijku', h['aa'][c,c,c,v], t['a'][pv,ha], optimize=True)
    O['aa'][c,c,c,a] += scale * -0.50000000 * np.einsum('ijua,ak->ijku', h['aa'][c,c,a,v], t['a'][pv,hc], optimize=True)
    O['aa'][a,v,c,a] += scale * -1.00000000 * np.einsum('uv,wuix,av->waix', eta1['a'], h['aa'][a,a,c,a], t['a'][pv,ha], optimize=True)
    
    O['aa'][a,v,c,a] += scale * -1.00000000 * np.einsum('uv,waxv,ui->waix', eta1['a'], h['aa'][a,v,a,a], t['a'][pa,hc], optimize=True)
    
    O['aa'][a,v,c,a] += scale * -1.00000000 * np.einsum('uv,wuix,av->waix', gamma1['a'], h['aa'][a,a,c,a], t['a'][pv,ha], optimize=True)
    
    O['aa'][a,v,c,a] += scale * -1.00000000 * np.einsum('uv,waxv,ui->waix', gamma1['a'], h['aa'][a,v,a,a], t['a'][pa,hc], optimize=True)
    
    O['aa'][a,v,c,a] += scale * +1.00000000 * np.einsum('iujv,ai->uajv', h['aa'][c,a,c,a], t['a'][pv,hc], optimize=True)
    O['aa'][a,v,c,a] += scale * -1.00000000 * np.einsum('iaju,vi->vaju', h['aa'][c,v,c,a], t['a'][pa,hc], optimize=True)
    O['aa'][a,v,c,a] += scale * +1.00000000 * np.einsum('uaib,bv->uaiv', h['aa'][a,v,c,v], t['a'][pv,ha], optimize=True)
    O['aa'][a,v,c,a] += scale * -1.00000000 * np.einsum('uavb,bi->uaiv', h['aa'][a,v,a,v], t['a'][pv,hc], optimize=True)
    O['aa'][c,v,c,a] += scale * -1.00000000 * np.einsum('uv,iujw,av->iajw', eta1['a'], h['aa'][c,a,c,a], t['a'][pv,ha], optimize=True)
    
    O['aa'][c,v,c,a] += scale * -1.00000000 * np.einsum('uv,iawv,uj->iajw', eta1['a'], h['aa'][c,v,a,a], t['a'][pa,hc], optimize=True)
    O['aa'][c,v,c,a] += scale * -1.00000000 * np.einsum('uv,iujw,av->iajw', gamma1['a'], h['aa'][c,a,c,a], t['a'][pv,ha], optimize=True)
    
    O['aa'][c,v,c,a] += scale * -1.00000000 * np.einsum('uv,iawv,uj->iajw', gamma1['a'], h['aa'][c,v,a,a], t['a'][pa,hc], optimize=True)
    O['aa'][c,v,c,a] += scale * -1.00000000 * np.einsum('ijku,aj->iaku', h['aa'][c,c,c,a], t['a'][pv,hc], optimize=True)
    O['aa'][c,v,c,a] += scale * +1.00000000 * np.einsum('iajb,bu->iaju', h['aa'][c,v,c,v], t['a'][pv,ha], optimize=True)
    O['aa'][c,v,c,a] += scale * -1.00000000 * np.einsum('iaub,bj->iaju', h['aa'][c,v,a,v], t['a'][pv,hc], optimize=True)
    O['aa'][v,v,c,a] += scale * +1.00000000 * np.einsum('uv,uaiw,bv->abiw', eta1['a'], h['aa'][a,v,c,a], t['a'][pv,ha], optimize=True)
    
    O['aa'][v,v,c,a] += scale * -0.50000000 * np.einsum('uv,abwv,ui->abiw', eta1['a'], h['aa'][v,v,a,a], t['a'][pa,hc], optimize=True)
    O['aa'][v,v,c,a] += scale * +1.00000000 * np.einsum('uv,uaiw,bv->abiw', gamma1['a'], h['aa'][a,v,c,a], t['a'][pv,ha], optimize=True)
    
    O['aa'][v,v,c,a] += scale * -0.50000000 * np.einsum('uv,abwv,ui->abiw', gamma1['a'], h['aa'][v,v,a,a], t['a'][pa,hc], optimize=True)
    O['aa'][v,v,c,a] += scale * +1.00000000 * np.einsum('iaju,bi->abju', h['aa'][c,v,c,a], t['a'][pv,hc], optimize=True)
    O['aa'][v,v,c,a] += scale * +0.50000000 * np.einsum('abic,cu->abiu', h['aa'][v,v,c,v], t['a'][pv,ha], optimize=True)
    O['aa'][v,v,c,a] += scale * -0.50000000 * np.einsum('abuc,ci->abiu', h['aa'][v,v,a,v], t['a'][pv,hc], optimize=True)
    O['aa'][a,a,c,c] += scale * +0.50000000 * np.einsum('uv,wxiv,uj->wxij', eta1['a'], h['aa'][a,a,c,a], t['a'][pa,hc], optimize=True)
    
    O['aa'][a,a,c,c] += scale * +0.50000000 * np.einsum('uv,wxiv,uj->wxij', gamma1['a'], h['aa'][a,a,c,a], t['a'][pa,hc], optimize=True)
    
    O['aa'][a,a,c,c] += scale * +0.50000000 * np.einsum('iujk,vi->uvjk', h['aa'][c,a,c,c], t['a'][pa,hc], optimize=True)
    O['aa'][a,a,c,c] += scale * +0.50000000 * np.einsum('uvia,aj->uvij', h['aa'][a,a,c,v], t['a'][pv,hc], optimize=True)
    O['aa'][c,a,c,c] += scale * +1.00000000 * np.einsum('uv,iwjv,uk->iwjk', eta1['a'], h['aa'][c,a,c,a], t['a'][pa,hc], optimize=True)
    
    O['aa'][c,a,c,c] += scale * +1.00000000 * np.einsum('uv,iwjv,uk->iwjk', gamma1['a'], h['aa'][c,a,c,a], t['a'][pa,hc], optimize=True)
    
    O['aa'][c,a,c,c] += scale * -0.50000000 * np.einsum('ijkl,uj->iukl', h['aa'][c,c,c,c], t['a'][pa,hc], optimize=True)
    O['aa'][c,a,c,c] += scale * +1.00000000 * np.einsum('iuja,ak->iujk', h['aa'][c,a,c,v], t['a'][pv,hc], optimize=True)
    O['aa'][c,c,c,c] += scale * +0.50000000 * np.einsum('uv,ijkv,ul->ijkl', eta1['a'], h['aa'][c,c,c,a], t['a'][pa,hc], optimize=True)
    O['aa'][c,c,c,c] += scale * +0.50000000 * np.einsum('uv,ijkv,ul->ijkl', gamma1['a'], h['aa'][c,c,c,a], t['a'][pa,hc], optimize=True)
    O['aa'][c,c,c,c] += scale * +0.50000000 * np.einsum('ijka,al->ijkl', h['aa'][c,c,c,v], t['a'][pv,hc], optimize=True)
    O['aa'][a,v,c,c] += scale * -0.50000000 * np.einsum('uv,wuij,av->waij', eta1['a'], h['aa'][a,a,c,c], t['a'][pv,ha], optimize=True)
    O['aa'][a,v,c,c] += scale * +1.00000000 * np.einsum('uv,waiv,uj->waij', eta1['a'], h['aa'][a,v,c,a], t['a'][pa,hc], optimize=True)
    
    O['aa'][a,v,c,c] += scale * -0.50000000 * np.einsum('uv,wuij,av->waij', gamma1['a'], h['aa'][a,a,c,c], t['a'][pv,ha], optimize=True)
    O['aa'][a,v,c,c] += scale * +1.00000000 * np.einsum('uv,waiv,uj->waij', gamma1['a'], h['aa'][a,v,c,a], t['a'][pa,hc], optimize=True)
    
    O['aa'][a,v,c,c] += scale * +0.50000000 * np.einsum('iujk,ai->uajk', h['aa'][c,a,c,c], t['a'][pv,hc], optimize=True)
    O['aa'][a,v,c,c] += scale * -0.50000000 * np.einsum('iajk,ui->uajk', h['aa'][c,v,c,c], t['a'][pa,hc], optimize=True)
    O['aa'][a,v,c,c] += scale * +1.00000000 * np.einsum('uaib,bj->uaij', h['aa'][a,v,c,v], t['a'][pv,hc], optimize=True)
    O['aa'][c,v,c,c] += scale * -0.50000000 * np.einsum('uv,iujk,av->iajk', eta1['a'], h['aa'][c,a,c,c], t['a'][pv,ha], optimize=True)
    O['aa'][c,v,c,c] += scale * +1.00000000 * np.einsum('uv,iajv,uk->iajk', eta1['a'], h['aa'][c,v,c,a], t['a'][pa,hc], optimize=True)
    O['aa'][c,v,c,c] += scale * -0.50000000 * np.einsum('uv,iujk,av->iajk', gamma1['a'], h['aa'][c,a,c,c], t['a'][pv,ha], optimize=True)
    O['aa'][c,v,c,c] += scale * +1.00000000 * np.einsum('uv,iajv,uk->iajk', gamma1['a'], h['aa'][c,v,c,a], t['a'][pa,hc], optimize=True)
    O['aa'][c,v,c,c] += scale * -0.50000000 * np.einsum('ijkl,aj->iakl', h['aa'][c,c,c,c], t['a'][pv,hc], optimize=True)
    O['aa'][c,v,c,c] += scale * +1.00000000 * np.einsum('iajb,bk->iajk', h['aa'][c,v,c,v], t['a'][pv,hc], optimize=True)
    O['aa'][v,v,c,c] += scale * +0.50000000 * np.einsum('uv,uaij,bv->abij', eta1['a'], h['aa'][a,v,c,c], t['a'][pv,ha], optimize=True)
    O['aa'][v,v,c,c] += scale * +0.50000000 * np.einsum('uv,abiv,uj->abij', eta1['a'], h['aa'][v,v,c,a], t['a'][pa,hc], optimize=True)
    O['aa'][v,v,c,c] += scale * +0.50000000 * np.einsum('uv,uaij,bv->abij', gamma1['a'], h['aa'][a,v,c,c], t['a'][pv,ha], optimize=True)
    O['aa'][v,v,c,c] += scale * +0.50000000 * np.einsum('uv,abiv,uj->abij', gamma1['a'], h['aa'][v,v,c,a], t['a'][pa,hc], optimize=True)
    O['aa'][v,v,c,c] += scale * +0.50000000 * np.einsum('iajk,bi->abjk', h['aa'][c,v,c,c], t['a'][pv,hc], optimize=True)
    O['aa'][v,v,c,c] += scale * +0.50000000 * np.einsum('abic,cj->abij', h['aa'][v,v,c,v], t['a'][pv,hc], optimize=True)
    O['aa'][a,a,c,v] += scale * +0.50000000 * np.einsum('uv,wxva,ui->wxia', eta1['a'], h['aa'][a,a,a,v], t['a'][pa,hc], optimize=True)
    
    O['aa'][a,a,c,v] += scale * +0.50000000 * np.einsum('uv,wxva,ui->wxia', gamma1['a'], h['aa'][a,a,a,v], t['a'][pa,hc], optimize=True)
    
    O['aa'][a,a,c,v] += scale * +1.00000000 * np.einsum('iuja,vi->uvja', h['aa'][c,a,c,v], t['a'][pa,hc], optimize=True)
    O['aa'][a,a,c,v] += scale * -0.50000000 * np.einsum('uvab,bi->uvia', h['aa'][a,a,v,v], t['a'][pv,hc], optimize=True)
    O['aa'][c,a,c,v] += scale * +1.00000000 * np.einsum('uv,iwva,uj->iwja', eta1['a'], h['aa'][c,a,a,v], t['a'][pa,hc], optimize=True)
    
    O['aa'][c,a,c,v] += scale * +1.00000000 * np.einsum('uv,iwva,uj->iwja', gamma1['a'], h['aa'][c,a,a,v], t['a'][pa,hc], optimize=True)
    
    O['aa'][c,a,c,v] += scale * -1.00000000 * np.einsum('ijka,uj->iuka', h['aa'][c,c,c,v], t['a'][pa,hc], optimize=True)
    O['aa'][c,a,c,v] += scale * -1.00000000 * np.einsum('iuab,bj->iuja', h['aa'][c,a,v,v], t['a'][pv,hc], optimize=True)
    O['aa'][c,c,c,v] += scale * +0.50000000 * np.einsum('uv,ijva,uk->ijka', eta1['a'], h['aa'][c,c,a,v], t['a'][pa,hc], optimize=True)
    O['aa'][c,c,c,v] += scale * +0.50000000 * np.einsum('uv,ijva,uk->ijka', gamma1['a'], h['aa'][c,c,a,v], t['a'][pa,hc], optimize=True)
    O['aa'][c,c,c,v] += scale * -0.50000000 * np.einsum('ijab,bk->ijka', h['aa'][c,c,v,v], t['a'][pv,hc], optimize=True)
    O['aa'][a,v,c,v] += scale * -1.00000000 * np.einsum('uv,wuia,bv->wbia', eta1['a'], h['aa'][a,a,c,v], t['a'][pv,ha], optimize=True)
    O['aa'][a,v,c,v] += scale * +1.00000000 * np.einsum('uv,wavb,ui->waib', eta1['a'], h['aa'][a,v,a,v], t['a'][pa,hc], optimize=True)
    
    O['aa'][a,v,c,v] += scale * -1.00000000 * np.einsum('uv,wuia,bv->wbia', gamma1['a'], h['aa'][a,a,c,v], t['a'][pv,ha], optimize=True)
    O['aa'][a,v,c,v] += scale * +1.00000000 * np.einsum('uv,wavb,ui->waib', gamma1['a'], h['aa'][a,v,a,v], t['a'][pa,hc], optimize=True)
    
    O['aa'][a,v,c,v] += scale * +1.00000000 * np.einsum('iuja,bi->ubja', h['aa'][c,a,c,v], t['a'][pv,hc], optimize=True)
    O['aa'][a,v,c,v] += scale * -1.00000000 * np.einsum('iajb,ui->uajb', h['aa'][c,v,c,v], t['a'][pa,hc], optimize=True)
    O['aa'][a,v,c,v] += scale * -1.00000000 * np.einsum('uabc,ci->uaib', h['aa'][a,v,v,v], t['a'][pv,hc], optimize=True)
    O['aa'][c,v,c,v] += scale * -1.00000000 * np.einsum('uv,iuja,bv->ibja', eta1['a'], h['aa'][c,a,c,v], t['a'][pv,ha], optimize=True)
    O['aa'][c,v,c,v] += scale * +1.00000000 * np.einsum('uv,iavb,uj->iajb', eta1['a'], h['aa'][c,v,a,v], t['a'][pa,hc], optimize=True)
    O['aa'][c,v,c,v] += scale * -1.00000000 * np.einsum('uv,iuja,bv->ibja', gamma1['a'], h['aa'][c,a,c,v], t['a'][pv,ha], optimize=True)
    O['aa'][c,v,c,v] += scale * +1.00000000 * np.einsum('uv,iavb,uj->iajb', gamma1['a'], h['aa'][c,v,a,v], t['a'][pa,hc], optimize=True)
    O['aa'][c,v,c,v] += scale * -1.00000000 * np.einsum('ijka,bj->ibka', h['aa'][c,c,c,v], t['a'][pv,hc], optimize=True)
    O['aa'][c,v,c,v] += scale * -1.00000000 * np.einsum('iabc,cj->iajb', h['aa'][c,v,v,v], t['a'][pv,hc], optimize=True)
    O['aa'][v,v,c,v] += scale * +1.00000000 * np.einsum('uv,uaib,cv->acib', eta1['a'], h['aa'][a,v,c,v], t['a'][pv,ha], optimize=True)
    O['aa'][v,v,c,v] += scale * +0.50000000 * np.einsum('uv,abvc,ui->abic', eta1['a'], h['aa'][v,v,a,v], t['a'][pa,hc], optimize=True)
    O['aa'][v,v,c,v] += scale * +1.00000000 * np.einsum('uv,uaib,cv->acib', gamma1['a'], h['aa'][a,v,c,v], t['a'][pv,ha], optimize=True)
    O['aa'][v,v,c,v] += scale * +0.50000000 * np.einsum('uv,abvc,ui->abic', gamma1['a'], h['aa'][v,v,a,v], t['a'][pa,hc], optimize=True)
    O['aa'][v,v,c,v] += scale * +1.00000000 * np.einsum('iajb,ci->acjb', h['aa'][c,v,c,v], t['a'][pv,hc], optimize=True)
    O['aa'][v,v,c,v] += scale * -0.50000000 * np.einsum('abcd,di->abic', h['aa'][v,v,v,v], t['a'][pv,hc], optimize=True)
    
    
    O['aa'][a,a,v,v] += scale * +0.50000000 * np.einsum('iuab,vi->uvab', h['aa'][c,a,v,v], t['a'][pa,hc], optimize=True)
    
    
    O['aa'][c,a,v,v] += scale * -0.50000000 * np.einsum('ijab,uj->iuab', h['aa'][c,c,v,v], t['a'][pa,hc], optimize=True)
    O['aa'][a,v,v,v] += scale * -0.50000000 * np.einsum('uv,wuab,cv->wcab', eta1['a'], h['aa'][a,a,v,v], t['a'][pv,ha], optimize=True)
    
    O['aa'][a,v,v,v] += scale * -0.50000000 * np.einsum('uv,wuab,cv->wcab', gamma1['a'], h['aa'][a,a,v,v], t['a'][pv,ha], optimize=True)
    
    O['aa'][a,v,v,v] += scale * +0.50000000 * np.einsum('iuab,ci->ucab', h['aa'][c,a,v,v], t['a'][pv,hc], optimize=True)
    O['aa'][a,v,v,v] += scale * -0.50000000 * np.einsum('iabc,ui->uabc', h['aa'][c,v,v,v], t['a'][pa,hc], optimize=True)
    O['aa'][c,v,v,v] += scale * -0.50000000 * np.einsum('uv,iuab,cv->icab', eta1['a'], h['aa'][c,a,v,v], t['a'][pv,ha], optimize=True)
    O['aa'][c,v,v,v] += scale * -0.50000000 * np.einsum('uv,iuab,cv->icab', gamma1['a'], h['aa'][c,a,v,v], t['a'][pv,ha], optimize=True)
    O['aa'][c,v,v,v] += scale * -0.50000000 * np.einsum('ijab,cj->icab', h['aa'][c,c,v,v], t['a'][pv,hc], optimize=True)
    O['aa'][v,v,v,v] += scale * +0.50000000 * np.einsum('uv,uabc,dv->adbc', eta1['a'], h['aa'][a,v,v,v], t['a'][pv,ha], optimize=True)
    O['aa'][v,v,v,v] += scale * +0.50000000 * np.einsum('uv,uabc,dv->adbc', gamma1['a'], h['aa'][a,v,v,v], t['a'][pv,ha], optimize=True)
    O['aa'][v,v,v,v] += scale * +0.50000000 * np.einsum('iabc,di->adbc', h['aa'][c,v,v,v], t['a'][pv,hc], optimize=True)

    t1 = time.time()
    if verbose: print("h2a_t1a_c2a took {:.4f} seconds to run.".format(t1-t0))

    return O
def h2a_t2a_c2a(O, h, t, gamma1, eta1, lambdas, orbspace, verbose=False, scale=1.0):
    # 324 lines
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

    
    
    
    
    O['aa'][a,a,a,a] += scale * +1.00000000 * np.einsum('uv,iwxv,yuiz->wyxz', eta1['a'], h['aa'][c,a,a,a], t['aa'][pa,pa,hc,ha], optimize=True)
    O['aa'][a,a,a,a] += scale * +0.25000000 * np.einsum('uv,wxva,uayz->wxyz', eta1['a'], h['aa'][a,a,a,v], t['aa'][pa,pv,ha,ha], optimize=True)
    
    
    O['aa'][a,a,a,a] += scale * +0.25000000 * np.einsum('uv,iuwx,yziv->yzwx', gamma1['a'], h['aa'][c,a,a,a], t['aa'][pa,pa,hc,ha], optimize=True)
    O['aa'][a,a,a,a] += scale * +1.00000000 * np.einsum('uv,wuxa,yazv->wyxz', gamma1['a'], h['aa'][a,a,a,v], t['aa'][pa,pv,ha,ha], optimize=True)
    O['aa'][a,a,a,a] += scale * +0.12500000 * np.einsum('ijuv,wxij->wxuv', h['aa'][c,c,a,a], t['aa'][pa,pa,hc,hc], optimize=True)
    O['aa'][a,a,a,a] += scale * +1.00000000 * np.einsum('iuva,waix->uwvx', h['aa'][c,a,a,v], t['aa'][pa,pv,hc,ha], optimize=True)
    O['aa'][a,a,a,a] += scale * +0.12500000 * np.einsum('uvab,abwx->uvwx', h['aa'][a,a,v,v], t['aa'][pv,pv,ha,ha], optimize=True)
    
    
    
    O['aa'][c,a,a,a] += scale * -1.00000000 * np.einsum('uv,ijwv,xujy->ixwy', eta1['a'], h['aa'][c,c,a,a], t['aa'][pa,pa,hc,ha], optimize=True)
    O['aa'][c,a,a,a] += scale * +0.50000000 * np.einsum('uv,iwva,uaxy->iwxy', eta1['a'], h['aa'][c,a,a,v], t['aa'][pa,pv,ha,ha], optimize=True)
    
    O['aa'][c,a,a,a] += scale * +1.00000000 * np.einsum('uv,iuwa,xayv->ixwy', gamma1['a'], h['aa'][c,a,a,v], t['aa'][pa,pv,ha,ha], optimize=True)
    O['aa'][c,a,a,a] += scale * -1.00000000 * np.einsum('ijua,vajw->ivuw', h['aa'][c,c,a,v], t['aa'][pa,pv,hc,ha], optimize=True)
    O['aa'][c,a,a,a] += scale * +0.25000000 * np.einsum('iuab,abvw->iuvw', h['aa'][c,a,v,v], t['aa'][pv,pv,ha,ha], optimize=True)
    
    O['aa'][c,c,a,a] += scale * +0.25000000 * np.einsum('uv,ijva,uawx->ijwx', eta1['a'], h['aa'][c,c,a,v], t['aa'][pa,pv,ha,ha], optimize=True)
    
    O['aa'][c,c,a,a] += scale * +0.12500000 * np.einsum('ijab,abuv->ijuv', h['aa'][c,c,v,v], t['aa'][pv,pv,ha,ha], optimize=True)
    
    O['aa'][a,v,a,a] += scale * -0.25000000 * np.einsum('uv,wx,uwyz,ravx->rayz', eta1['a'], eta1['a'], h['aa'][a,a,a,a], t['aa'][pa,pv,ha,ha], optimize=True)
    O['aa'][a,v,a,a] += scale * +1.00000000 * np.einsum('uv,wx,yuzx,warv->yazr', eta1['a'], gamma1['a'], h['aa'][a,a,a,a], t['aa'][pa,pv,ha,ha], optimize=True)
    O['aa'][a,v,a,a] += scale * -1.00000000 * np.einsum('uv,wx,ywzv,uarx->yazr', eta1['a'], gamma1['a'], h['aa'][a,a,a,a], t['aa'][pa,pv,ha,ha], optimize=True)
    
    
    O['aa'][a,v,a,a] += scale * -1.00000000 * np.einsum('uv,iwxv,uaiy->waxy', eta1['a'], h['aa'][c,a,a,a], t['aa'][pa,pv,hc,ha], optimize=True)
    O['aa'][a,v,a,a] += scale * -1.00000000 * np.einsum('uv,iawv,xuiy->xawy', eta1['a'], h['aa'][c,v,a,a], t['aa'][pa,pa,hc,ha], optimize=True)
    O['aa'][a,v,a,a] += scale * +0.50000000 * np.einsum('uv,wavb,ubxy->waxy', eta1['a'], h['aa'][a,v,a,v], t['aa'][pa,pv,ha,ha], optimize=True)
    
    O['aa'][a,v,a,a] += scale * +0.25000000 * np.einsum('uv,wx,uwyz,ravx->rayz', gamma1['a'], gamma1['a'], h['aa'][a,a,a,a], t['aa'][pa,pv,ha,ha], optimize=True)
    O['aa'][a,v,a,a] += scale * +0.50000000 * np.einsum('uv,iuwx,yaiv->yawx', gamma1['a'], h['aa'][c,a,a,a], t['aa'][pa,pv,hc,ha], optimize=True)
    O['aa'][a,v,a,a] += scale * +1.00000000 * np.einsum('uv,wuxa,bayv->wbxy', gamma1['a'], h['aa'][a,a,a,v], t['aa'][pv,pv,ha,ha], optimize=True)
    O['aa'][a,v,a,a] += scale * +1.00000000 * np.einsum('uv,uawb,xbyv->xawy', gamma1['a'], h['aa'][a,v,a,v], t['aa'][pa,pv,ha,ha], optimize=True)
    O['aa'][a,v,a,a] += scale * +0.25000000 * np.einsum('ijuv,waij->wauv', h['aa'][c,c,a,a], t['aa'][pa,pv,hc,hc], optimize=True)
    O['aa'][a,v,a,a] += scale * +1.00000000 * np.einsum('iuva,baiw->ubvw', h['aa'][c,a,a,v], t['aa'][pv,pv,hc,ha], optimize=True)
    O['aa'][a,v,a,a] += scale * -1.00000000 * np.einsum('iaub,vbiw->vauw', h['aa'][c,v,a,v], t['aa'][pa,pv,hc,ha], optimize=True)
    O['aa'][a,v,a,a] += scale * +0.25000000 * np.einsum('uabc,bcvw->uavw', h['aa'][a,v,v,v], t['aa'][pv,pv,ha,ha], optimize=True)
    
    O['aa'][c,v,a,a] += scale * +1.00000000 * np.einsum('uv,wx,iuyx,wazv->iayz', eta1['a'], gamma1['a'], h['aa'][c,a,a,a], t['aa'][pa,pv,ha,ha], optimize=True)
    O['aa'][c,v,a,a] += scale * -1.00000000 * np.einsum('uv,wx,iwyv,uazx->iayz', eta1['a'], gamma1['a'], h['aa'][c,a,a,a], t['aa'][pa,pv,ha,ha], optimize=True)
    O['aa'][c,v,a,a] += scale * +1.00000000 * np.einsum('uv,ijwv,uajx->iawx', eta1['a'], h['aa'][c,c,a,a], t['aa'][pa,pv,hc,ha], optimize=True)
    O['aa'][c,v,a,a] += scale * +0.50000000 * np.einsum('uv,iavb,ubwx->iawx', eta1['a'], h['aa'][c,v,a,v], t['aa'][pa,pv,ha,ha], optimize=True)
    
    O['aa'][c,v,a,a] += scale * +1.00000000 * np.einsum('uv,iuwa,baxv->ibwx', gamma1['a'], h['aa'][c,a,a,v], t['aa'][pv,pv,ha,ha], optimize=True)
    O['aa'][c,v,a,a] += scale * -1.00000000 * np.einsum('ijua,bajv->ibuv', h['aa'][c,c,a,v], t['aa'][pv,pv,hc,ha], optimize=True)
    O['aa'][c,v,a,a] += scale * +0.25000000 * np.einsum('iabc,bcuv->iauv', h['aa'][c,v,v,v], t['aa'][pv,pv,ha,ha], optimize=True)
    O['aa'][v,v,a,a] += scale * -0.12500000 * np.einsum('uv,wx,uwyz,abvx->abyz', eta1['a'], eta1['a'], h['aa'][a,a,a,a], t['aa'][pv,pv,ha,ha], optimize=True)
    
    O['aa'][v,v,a,a] += scale * -1.00000000 * np.einsum('uv,wx,uayx,wbzv->abyz', eta1['a'], gamma1['a'], h['aa'][a,v,a,a], t['aa'][pa,pv,ha,ha], optimize=True)
    O['aa'][v,v,a,a] += scale * +1.00000000 * np.einsum('uv,wx,wayv,ubzx->abyz', eta1['a'], gamma1['a'], h['aa'][a,v,a,a], t['aa'][pa,pv,ha,ha], optimize=True)
    O['aa'][v,v,a,a] += scale * -1.00000000 * np.einsum('uv,iawv,ubix->abwx', eta1['a'], h['aa'][c,v,a,a], t['aa'][pa,pv,hc,ha], optimize=True)
    O['aa'][v,v,a,a] += scale * +0.25000000 * np.einsum('uv,abvc,ucwx->abwx', eta1['a'], h['aa'][v,v,a,v], t['aa'][pa,pv,ha,ha], optimize=True)
    O['aa'][v,v,a,a] += scale * +0.12500000 * np.einsum('uv,wx,uwyz,abvx->abyz', gamma1['a'], gamma1['a'], h['aa'][a,a,a,a], t['aa'][pv,pv,ha,ha], optimize=True)
    
    O['aa'][v,v,a,a] += scale * +0.25000000 * np.einsum('uv,iuwx,abiv->abwx', gamma1['a'], h['aa'][c,a,a,a], t['aa'][pv,pv,hc,ha], optimize=True)
    O['aa'][v,v,a,a] += scale * -1.00000000 * np.einsum('uv,uawb,cbxv->acwx', gamma1['a'], h['aa'][a,v,a,v], t['aa'][pv,pv,ha,ha], optimize=True)
    O['aa'][v,v,a,a] += scale * +0.12500000 * np.einsum('ijuv,abij->abuv', h['aa'][c,c,a,a], t['aa'][pv,pv,hc,hc], optimize=True)
    O['aa'][v,v,a,a] += scale * +1.00000000 * np.einsum('iaub,cbiv->acuv', h['aa'][c,v,a,v], t['aa'][pv,pv,hc,ha], optimize=True)
    O['aa'][v,v,a,a] += scale * +0.12500000 * np.einsum('abcd,cduv->abuv', h['aa'][v,v,v,v], t['aa'][pv,pv,ha,ha], optimize=True)
    
    
    
    O['aa'][a,a,a,v] += scale * +1.00000000 * np.einsum('uv,iwva,xuiy->wxya', eta1['a'], h['aa'][c,a,a,v], t['aa'][pa,pa,hc,ha], optimize=True)
    
    O['aa'][a,a,a,v] += scale * +0.50000000 * np.einsum('uv,iuwa,xyiv->xywa', gamma1['a'], h['aa'][c,a,a,v], t['aa'][pa,pa,hc,ha], optimize=True)
    O['aa'][a,a,a,v] += scale * -1.00000000 * np.einsum('uv,wuab,xbyv->wxya', gamma1['a'], h['aa'][a,a,v,v], t['aa'][pa,pv,ha,ha], optimize=True)
    O['aa'][a,a,a,v] += scale * +0.25000000 * np.einsum('ijua,vwij->vwua', h['aa'][c,c,a,v], t['aa'][pa,pa,hc,hc], optimize=True)
    O['aa'][a,a,a,v] += scale * -1.00000000 * np.einsum('iuab,vbiw->uvwa', h['aa'][c,a,v,v], t['aa'][pa,pv,hc,ha], optimize=True)
    
    
    O['aa'][c,a,a,v] += scale * -1.00000000 * np.einsum('uv,ijva,wujx->iwxa', eta1['a'], h['aa'][c,c,a,v], t['aa'][pa,pa,hc,ha], optimize=True)
    O['aa'][c,a,a,v] += scale * -1.00000000 * np.einsum('uv,iuab,wbxv->iwxa', gamma1['a'], h['aa'][c,a,v,v], t['aa'][pa,pv,ha,ha], optimize=True)
    O['aa'][c,a,a,v] += scale * +1.00000000 * np.einsum('ijab,ubjv->iuva', h['aa'][c,c,v,v], t['aa'][pa,pv,hc,ha], optimize=True)
    O['aa'][a,v,a,v] += scale * -0.50000000 * np.einsum('uv,wx,uwya,zbvx->zbya', eta1['a'], eta1['a'], h['aa'][a,a,a,v], t['aa'][pa,pv,ha,ha], optimize=True)
    O['aa'][a,v,a,v] += scale * +1.00000000 * np.einsum('uv,wx,yuxa,wbzv->ybza', eta1['a'], gamma1['a'], h['aa'][a,a,a,v], t['aa'][pa,pv,ha,ha], optimize=True)
    O['aa'][a,v,a,v] += scale * -1.00000000 * np.einsum('uv,wx,ywva,ubzx->ybza', eta1['a'], gamma1['a'], h['aa'][a,a,a,v], t['aa'][pa,pv,ha,ha], optimize=True)
    
    
    O['aa'][a,v,a,v] += scale * -1.00000000 * np.einsum('uv,iwva,ubix->wbxa', eta1['a'], h['aa'][c,a,a,v], t['aa'][pa,pv,hc,ha], optimize=True)
    O['aa'][a,v,a,v] += scale * -1.00000000 * np.einsum('uv,iavb,wuix->waxb', eta1['a'], h['aa'][c,v,a,v], t['aa'][pa,pa,hc,ha], optimize=True)
    O['aa'][a,v,a,v] += scale * +0.50000000 * np.einsum('uv,wx,uwya,zbvx->zbya', gamma1['a'], gamma1['a'], h['aa'][a,a,a,v], t['aa'][pa,pv,ha,ha], optimize=True)
    O['aa'][a,v,a,v] += scale * +1.00000000 * np.einsum('uv,iuwa,xbiv->xbwa', gamma1['a'], h['aa'][c,a,a,v], t['aa'][pa,pv,hc,ha], optimize=True)
    O['aa'][a,v,a,v] += scale * -1.00000000 * np.einsum('uv,wuab,cbxv->wcxa', gamma1['a'], h['aa'][a,a,v,v], t['aa'][pv,pv,ha,ha], optimize=True)
    O['aa'][a,v,a,v] += scale * -1.00000000 * np.einsum('uv,uabc,wcxv->waxb', gamma1['a'], h['aa'][a,v,v,v], t['aa'][pa,pv,ha,ha], optimize=True)
    O['aa'][a,v,a,v] += scale * +0.50000000 * np.einsum('ijua,vbij->vbua', h['aa'][c,c,a,v], t['aa'][pa,pv,hc,hc], optimize=True)
    O['aa'][a,v,a,v] += scale * -1.00000000 * np.einsum('iuab,cbiv->ucva', h['aa'][c,a,v,v], t['aa'][pv,pv,hc,ha], optimize=True)
    O['aa'][a,v,a,v] += scale * +1.00000000 * np.einsum('iabc,uciv->uavb', h['aa'][c,v,v,v], t['aa'][pa,pv,hc,ha], optimize=True)
    O['aa'][c,v,a,v] += scale * +1.00000000 * np.einsum('uv,wx,iuxa,wbyv->ibya', eta1['a'], gamma1['a'], h['aa'][c,a,a,v], t['aa'][pa,pv,ha,ha], optimize=True)
    O['aa'][c,v,a,v] += scale * -1.00000000 * np.einsum('uv,wx,iwva,ubyx->ibya', eta1['a'], gamma1['a'], h['aa'][c,a,a,v], t['aa'][pa,pv,ha,ha], optimize=True)
    O['aa'][c,v,a,v] += scale * +1.00000000 * np.einsum('uv,ijva,ubjw->ibwa', eta1['a'], h['aa'][c,c,a,v], t['aa'][pa,pv,hc,ha], optimize=True)
    O['aa'][c,v,a,v] += scale * -1.00000000 * np.einsum('uv,iuab,cbwv->icwa', gamma1['a'], h['aa'][c,a,v,v], t['aa'][pv,pv,ha,ha], optimize=True)
    O['aa'][c,v,a,v] += scale * +1.00000000 * np.einsum('ijab,cbju->icua', h['aa'][c,c,v,v], t['aa'][pv,pv,hc,ha], optimize=True)
    O['aa'][v,v,a,v] += scale * -0.25000000 * np.einsum('uv,wx,uwya,bcvx->bcya', eta1['a'], eta1['a'], h['aa'][a,a,a,v], t['aa'][pv,pv,ha,ha], optimize=True)
    O['aa'][v,v,a,v] += scale * -1.00000000 * np.einsum('uv,wx,uaxb,wcyv->acyb', eta1['a'], gamma1['a'], h['aa'][a,v,a,v], t['aa'][pa,pv,ha,ha], optimize=True)
    O['aa'][v,v,a,v] += scale * +1.00000000 * np.einsum('uv,wx,wavb,ucyx->acyb', eta1['a'], gamma1['a'], h['aa'][a,v,a,v], t['aa'][pa,pv,ha,ha], optimize=True)
    O['aa'][v,v,a,v] += scale * -1.00000000 * np.einsum('uv,iavb,uciw->acwb', eta1['a'], h['aa'][c,v,a,v], t['aa'][pa,pv,hc,ha], optimize=True)
    O['aa'][v,v,a,v] += scale * +0.25000000 * np.einsum('uv,wx,uwya,bcvx->bcya', gamma1['a'], gamma1['a'], h['aa'][a,a,a,v], t['aa'][pv,pv,ha,ha], optimize=True)
    O['aa'][v,v,a,v] += scale * +0.50000000 * np.einsum('uv,iuwa,bciv->bcwa', gamma1['a'], h['aa'][c,a,a,v], t['aa'][pv,pv,hc,ha], optimize=True)
    O['aa'][v,v,a,v] += scale * +1.00000000 * np.einsum('uv,uabc,dcwv->adwb', gamma1['a'], h['aa'][a,v,v,v], t['aa'][pv,pv,ha,ha], optimize=True)
    O['aa'][v,v,a,v] += scale * +0.25000000 * np.einsum('ijua,bcij->bcua', h['aa'][c,c,a,v], t['aa'][pv,pv,hc,hc], optimize=True)
    O['aa'][v,v,a,v] += scale * -1.00000000 * np.einsum('iabc,dciu->adub', h['aa'][c,v,v,v], t['aa'][pv,pv,hc,ha], optimize=True)
    O['aa'][a,a,c,a] += scale * +0.25000000 * np.einsum('uv,wx,yzvx,uwir->yzir', eta1['a'], eta1['a'], h['aa'][a,a,a,a], t['aa'][pa,pa,hc,ha], optimize=True)
    
    
    O['aa'][a,a,c,a] += scale * +1.00000000 * np.einsum('uv,wx,yuzx,rwiv->yriz', eta1['a'], gamma1['a'], h['aa'][a,a,a,a], t['aa'][pa,pa,hc,ha], optimize=True)
    
    O['aa'][a,a,c,a] += scale * -1.00000000 * np.einsum('uv,wx,ywzv,ruix->yriz', eta1['a'], gamma1['a'], h['aa'][a,a,a,a], t['aa'][pa,pa,hc,ha], optimize=True)
    O['aa'][a,a,c,a] += scale * +1.00000000 * np.einsum('uv,iwjv,xuiy->wxjy', eta1['a'], h['aa'][c,a,c,a], t['aa'][pa,pa,hc,ha], optimize=True)
    O['aa'][a,a,c,a] += scale * +1.00000000 * np.einsum('uv,iwxv,yuji->wyjx', eta1['a'], h['aa'][c,a,a,a], t['aa'][pa,pa,hc,hc], optimize=True)
    O['aa'][a,a,c,a] += scale * +0.50000000 * np.einsum('uv,wxva,uaiy->wxiy', eta1['a'], h['aa'][a,a,a,v], t['aa'][pa,pv,hc,ha], optimize=True)
    O['aa'][a,a,c,a] += scale * -0.25000000 * np.einsum('uv,wx,yzvx,uwir->yzir', gamma1['a'], gamma1['a'], h['aa'][a,a,a,a], t['aa'][pa,pa,hc,ha], optimize=True)
    
    O['aa'][a,a,c,a] += scale * +0.50000000 * np.einsum('uv,iujw,xyiv->xyjw', gamma1['a'], h['aa'][c,a,c,a], t['aa'][pa,pa,hc,ha], optimize=True)
    O['aa'][a,a,c,a] += scale * +1.00000000 * np.einsum('uv,wuia,xayv->wxiy', gamma1['a'], h['aa'][a,a,c,v], t['aa'][pa,pv,ha,ha], optimize=True)
    O['aa'][a,a,c,a] += scale * -1.00000000 * np.einsum('uv,wuxa,yaiv->wyix', gamma1['a'], h['aa'][a,a,a,v], t['aa'][pa,pv,hc,ha], optimize=True)
    O['aa'][a,a,c,a] += scale * +0.25000000 * np.einsum('ijku,vwij->vwku', h['aa'][c,c,c,a], t['aa'][pa,pa,hc,hc], optimize=True)
    O['aa'][a,a,c,a] += scale * +1.00000000 * np.einsum('iuja,vaiw->uvjw', h['aa'][c,a,c,v], t['aa'][pa,pv,hc,ha], optimize=True)
    O['aa'][a,a,c,a] += scale * +1.00000000 * np.einsum('iuva,waji->uwjv', h['aa'][c,a,a,v], t['aa'][pa,pv,hc,hc], optimize=True)
    O['aa'][a,a,c,a] += scale * +0.25000000 * np.einsum('uvab,abiw->uviw', h['aa'][a,a,v,v], t['aa'][pv,pv,hc,ha], optimize=True)
    O['aa'][c,a,c,a] += scale * +0.50000000 * np.einsum('uv,wx,iyvx,uwjz->iyjz', eta1['a'], eta1['a'], h['aa'][c,a,a,a], t['aa'][pa,pa,hc,ha], optimize=True)
    
    O['aa'][c,a,c,a] += scale * +1.00000000 * np.einsum('uv,wx,iuyx,zwjv->izjy', eta1['a'], gamma1['a'], h['aa'][c,a,a,a], t['aa'][pa,pa,hc,ha], optimize=True)
    
    O['aa'][c,a,c,a] += scale * -1.00000000 * np.einsum('uv,wx,iwyv,zujx->izjy', eta1['a'], gamma1['a'], h['aa'][c,a,a,a], t['aa'][pa,pa,hc,ha], optimize=True)
    O['aa'][c,a,c,a] += scale * -1.00000000 * np.einsum('uv,ijkv,wujx->iwkx', eta1['a'], h['aa'][c,c,c,a], t['aa'][pa,pa,hc,ha], optimize=True)
    O['aa'][c,a,c,a] += scale * -1.00000000 * np.einsum('uv,ijwv,xukj->ixkw', eta1['a'], h['aa'][c,c,a,a], t['aa'][pa,pa,hc,hc], optimize=True)
    O['aa'][c,a,c,a] += scale * +1.00000000 * np.einsum('uv,iwva,uajx->iwjx', eta1['a'], h['aa'][c,a,a,v], t['aa'][pa,pv,hc,ha], optimize=True)
    O['aa'][c,a,c,a] += scale * -0.50000000 * np.einsum('uv,wx,iyvx,uwjz->iyjz', gamma1['a'], gamma1['a'], h['aa'][c,a,a,a], t['aa'][pa,pa,hc,ha], optimize=True)
    O['aa'][c,a,c,a] += scale * +1.00000000 * np.einsum('uv,iuja,waxv->iwjx', gamma1['a'], h['aa'][c,a,c,v], t['aa'][pa,pv,ha,ha], optimize=True)
    O['aa'][c,a,c,a] += scale * -1.00000000 * np.einsum('uv,iuwa,xajv->ixjw', gamma1['a'], h['aa'][c,a,a,v], t['aa'][pa,pv,hc,ha], optimize=True)
    O['aa'][c,a,c,a] += scale * -1.00000000 * np.einsum('ijka,uajv->iukv', h['aa'][c,c,c,v], t['aa'][pa,pv,hc,ha], optimize=True)
    O['aa'][c,a,c,a] += scale * -1.00000000 * np.einsum('ijua,vakj->ivku', h['aa'][c,c,a,v], t['aa'][pa,pv,hc,hc], optimize=True)
    O['aa'][c,a,c,a] += scale * +0.50000000 * np.einsum('iuab,abjv->iujv', h['aa'][c,a,v,v], t['aa'][pv,pv,hc,ha], optimize=True)
    O['aa'][c,c,c,a] += scale * +0.25000000 * np.einsum('uv,wx,ijvx,uwky->ijky', eta1['a'], eta1['a'], h['aa'][c,c,a,a], t['aa'][pa,pa,hc,ha], optimize=True)
    O['aa'][c,c,c,a] += scale * +0.50000000 * np.einsum('uv,ijva,uakw->ijkw', eta1['a'], h['aa'][c,c,a,v], t['aa'][pa,pv,hc,ha], optimize=True)
    O['aa'][c,c,c,a] += scale * -0.25000000 * np.einsum('uv,wx,ijvx,uwky->ijky', gamma1['a'], gamma1['a'], h['aa'][c,c,a,a], t['aa'][pa,pa,hc,ha], optimize=True)
    O['aa'][c,c,c,a] += scale * +0.25000000 * np.einsum('ijab,abku->ijku', h['aa'][c,c,v,v], t['aa'][pv,pv,hc,ha], optimize=True)
    O['aa'][a,v,c,a] += scale * +0.50000000 * np.einsum('uv,wx,yavx,uwiz->yaiz', eta1['a'], eta1['a'], h['aa'][a,v,a,a], t['aa'][pa,pa,hc,ha], optimize=True)
    O['aa'][a,v,c,a] += scale * -0.50000000 * np.einsum('uv,wx,uwiy,zavx->zaiy', eta1['a'], eta1['a'], h['aa'][a,a,c,a], t['aa'][pa,pv,ha,ha], optimize=True)
    O['aa'][a,v,c,a] += scale * +1.00000000 * np.einsum('uv,wx,yuix,wazv->yaiz', eta1['a'], gamma1['a'], h['aa'][a,a,c,a], t['aa'][pa,pv,ha,ha], optimize=True)
    O['aa'][a,v,c,a] += scale * -1.00000000 * np.einsum('uv,wx,yuzx,waiv->yaiz', eta1['a'], gamma1['a'], h['aa'][a,a,a,a], t['aa'][pa,pv,hc,ha], optimize=True)
    O['aa'][a,v,c,a] += scale * -1.00000000 * np.einsum('uv,wx,ywiv,uazx->yaiz', eta1['a'], gamma1['a'], h['aa'][a,a,c,a], t['aa'][pa,pv,ha,ha], optimize=True)
    O['aa'][a,v,c,a] += scale * +1.00000000 * np.einsum('uv,wx,ywzv,uaix->yaiz', eta1['a'], gamma1['a'], h['aa'][a,a,a,a], t['aa'][pa,pv,hc,ha], optimize=True)
    
    O['aa'][a,v,c,a] += scale * +1.00000000 * np.einsum('uv,wx,uayx,zwiv->zaiy', eta1['a'], gamma1['a'], h['aa'][a,v,a,a], t['aa'][pa,pa,hc,ha], optimize=True)
    
    O['aa'][a,v,c,a] += scale * -1.00000000 * np.einsum('uv,wx,wayv,zuix->zaiy', eta1['a'], gamma1['a'], h['aa'][a,v,a,a], t['aa'][pa,pa,hc,ha], optimize=True)
    O['aa'][a,v,c,a] += scale * -1.00000000 * np.einsum('uv,iwjv,uaix->wajx', eta1['a'], h['aa'][c,a,c,a], t['aa'][pa,pv,hc,ha], optimize=True)
    O['aa'][a,v,c,a] += scale * -1.00000000 * np.einsum('uv,iwxv,uaji->wajx', eta1['a'], h['aa'][c,a,a,a], t['aa'][pa,pv,hc,hc], optimize=True)
    O['aa'][a,v,c,a] += scale * -1.00000000 * np.einsum('uv,iajv,wuix->wajx', eta1['a'], h['aa'][c,v,c,a], t['aa'][pa,pa,hc,ha], optimize=True)
    O['aa'][a,v,c,a] += scale * -1.00000000 * np.einsum('uv,iawv,xuji->xajw', eta1['a'], h['aa'][c,v,a,a], t['aa'][pa,pa,hc,hc], optimize=True)
    O['aa'][a,v,c,a] += scale * +1.00000000 * np.einsum('uv,wavb,ubix->waix', eta1['a'], h['aa'][a,v,a,v], t['aa'][pa,pv,hc,ha], optimize=True)
    O['aa'][a,v,c,a] += scale * -0.50000000 * np.einsum('uv,wx,yavx,uwiz->yaiz', gamma1['a'], gamma1['a'], h['aa'][a,v,a,a], t['aa'][pa,pa,hc,ha], optimize=True)
    O['aa'][a,v,c,a] += scale * +0.50000000 * np.einsum('uv,wx,uwiy,zavx->zaiy', gamma1['a'], gamma1['a'], h['aa'][a,a,c,a], t['aa'][pa,pv,ha,ha], optimize=True)
    O['aa'][a,v,c,a] += scale * +1.00000000 * np.einsum('uv,iujw,xaiv->xajw', gamma1['a'], h['aa'][c,a,c,a], t['aa'][pa,pv,hc,ha], optimize=True)
    O['aa'][a,v,c,a] += scale * +1.00000000 * np.einsum('uv,wuia,baxv->wbix', gamma1['a'], h['aa'][a,a,c,v], t['aa'][pv,pv,ha,ha], optimize=True)
    O['aa'][a,v,c,a] += scale * -1.00000000 * np.einsum('uv,wuxa,baiv->wbix', gamma1['a'], h['aa'][a,a,a,v], t['aa'][pv,pv,hc,ha], optimize=True)
    O['aa'][a,v,c,a] += scale * +1.00000000 * np.einsum('uv,uaib,wbxv->waix', gamma1['a'], h['aa'][a,v,c,v], t['aa'][pa,pv,ha,ha], optimize=True)
    O['aa'][a,v,c,a] += scale * -1.00000000 * np.einsum('uv,uawb,xbiv->xaiw', gamma1['a'], h['aa'][a,v,a,v], t['aa'][pa,pv,hc,ha], optimize=True)
    O['aa'][a,v,c,a] += scale * +0.50000000 * np.einsum('ijku,vaij->vaku', h['aa'][c,c,c,a], t['aa'][pa,pv,hc,hc], optimize=True)
    O['aa'][a,v,c,a] += scale * +1.00000000 * np.einsum('iuja,baiv->ubjv', h['aa'][c,a,c,v], t['aa'][pv,pv,hc,ha], optimize=True)
    O['aa'][a,v,c,a] += scale * +1.00000000 * np.einsum('iuva,baji->ubjv', h['aa'][c,a,a,v], t['aa'][pv,pv,hc,hc], optimize=True)
    O['aa'][a,v,c,a] += scale * -1.00000000 * np.einsum('iajb,ubiv->uajv', h['aa'][c,v,c,v], t['aa'][pa,pv,hc,ha], optimize=True)
    O['aa'][a,v,c,a] += scale * -1.00000000 * np.einsum('iaub,vbji->vaju', h['aa'][c,v,a,v], t['aa'][pa,pv,hc,hc], optimize=True)
    O['aa'][a,v,c,a] += scale * +0.50000000 * np.einsum('uabc,bciv->uaiv', h['aa'][a,v,v,v], t['aa'][pv,pv,hc,ha], optimize=True)
    O['aa'][c,v,c,a] += scale * +0.50000000 * np.einsum('uv,wx,iavx,uwjy->iajy', eta1['a'], eta1['a'], h['aa'][c,v,a,a], t['aa'][pa,pa,hc,ha], optimize=True)
    O['aa'][c,v,c,a] += scale * +1.00000000 * np.einsum('uv,wx,iujx,wayv->iajy', eta1['a'], gamma1['a'], h['aa'][c,a,c,a], t['aa'][pa,pv,ha,ha], optimize=True)
    O['aa'][c,v,c,a] += scale * -1.00000000 * np.einsum('uv,wx,iuyx,wajv->iajy', eta1['a'], gamma1['a'], h['aa'][c,a,a,a], t['aa'][pa,pv,hc,ha], optimize=True)
    O['aa'][c,v,c,a] += scale * -1.00000000 * np.einsum('uv,wx,iwjv,uayx->iajy', eta1['a'], gamma1['a'], h['aa'][c,a,c,a], t['aa'][pa,pv,ha,ha], optimize=True)
    O['aa'][c,v,c,a] += scale * +1.00000000 * np.einsum('uv,wx,iwyv,uajx->iajy', eta1['a'], gamma1['a'], h['aa'][c,a,a,a], t['aa'][pa,pv,hc,ha], optimize=True)
    O['aa'][c,v,c,a] += scale * +1.00000000 * np.einsum('uv,ijkv,uajw->iakw', eta1['a'], h['aa'][c,c,c,a], t['aa'][pa,pv,hc,ha], optimize=True)
    O['aa'][c,v,c,a] += scale * +1.00000000 * np.einsum('uv,ijwv,uakj->iakw', eta1['a'], h['aa'][c,c,a,a], t['aa'][pa,pv,hc,hc], optimize=True)
    O['aa'][c,v,c,a] += scale * +1.00000000 * np.einsum('uv,iavb,ubjw->iajw', eta1['a'], h['aa'][c,v,a,v], t['aa'][pa,pv,hc,ha], optimize=True)
    O['aa'][c,v,c,a] += scale * -0.50000000 * np.einsum('uv,wx,iavx,uwjy->iajy', gamma1['a'], gamma1['a'], h['aa'][c,v,a,a], t['aa'][pa,pa,hc,ha], optimize=True)
    O['aa'][c,v,c,a] += scale * +1.00000000 * np.einsum('uv,iuja,bawv->ibjw', gamma1['a'], h['aa'][c,a,c,v], t['aa'][pv,pv,ha,ha], optimize=True)
    O['aa'][c,v,c,a] += scale * -1.00000000 * np.einsum('uv,iuwa,bajv->ibjw', gamma1['a'], h['aa'][c,a,a,v], t['aa'][pv,pv,hc,ha], optimize=True)
    O['aa'][c,v,c,a] += scale * -1.00000000 * np.einsum('ijka,baju->ibku', h['aa'][c,c,c,v], t['aa'][pv,pv,hc,ha], optimize=True)
    O['aa'][c,v,c,a] += scale * -1.00000000 * np.einsum('ijua,bakj->ibku', h['aa'][c,c,a,v], t['aa'][pv,pv,hc,hc], optimize=True)
    O['aa'][c,v,c,a] += scale * +0.50000000 * np.einsum('iabc,bcju->iaju', h['aa'][c,v,v,v], t['aa'][pv,pv,hc,ha], optimize=True)
    O['aa'][v,v,c,a] += scale * -0.25000000 * np.einsum('uv,wx,uwiy,abvx->abiy', eta1['a'], eta1['a'], h['aa'][a,a,c,a], t['aa'][pv,pv,ha,ha], optimize=True)
    O['aa'][v,v,c,a] += scale * +0.25000000 * np.einsum('uv,wx,abvx,uwiy->abiy', eta1['a'], eta1['a'], h['aa'][v,v,a,a], t['aa'][pa,pa,hc,ha], optimize=True)
    O['aa'][v,v,c,a] += scale * -1.00000000 * np.einsum('uv,wx,uaix,wbyv->abiy', eta1['a'], gamma1['a'], h['aa'][a,v,c,a], t['aa'][pa,pv,ha,ha], optimize=True)
    O['aa'][v,v,c,a] += scale * +1.00000000 * np.einsum('uv,wx,uayx,wbiv->abiy', eta1['a'], gamma1['a'], h['aa'][a,v,a,a], t['aa'][pa,pv,hc,ha], optimize=True)
    O['aa'][v,v,c,a] += scale * +1.00000000 * np.einsum('uv,wx,waiv,ubyx->abiy', eta1['a'], gamma1['a'], h['aa'][a,v,c,a], t['aa'][pa,pv,ha,ha], optimize=True)
    O['aa'][v,v,c,a] += scale * -1.00000000 * np.einsum('uv,wx,wayv,ubix->abiy', eta1['a'], gamma1['a'], h['aa'][a,v,a,a], t['aa'][pa,pv,hc,ha], optimize=True)
    O['aa'][v,v,c,a] += scale * -1.00000000 * np.einsum('uv,iajv,ubiw->abjw', eta1['a'], h['aa'][c,v,c,a], t['aa'][pa,pv,hc,ha], optimize=True)
    O['aa'][v,v,c,a] += scale * -1.00000000 * np.einsum('uv,iawv,ubji->abjw', eta1['a'], h['aa'][c,v,a,a], t['aa'][pa,pv,hc,hc], optimize=True)
    O['aa'][v,v,c,a] += scale * +0.50000000 * np.einsum('uv,abvc,uciw->abiw', eta1['a'], h['aa'][v,v,a,v], t['aa'][pa,pv,hc,ha], optimize=True)
    O['aa'][v,v,c,a] += scale * +0.25000000 * np.einsum('uv,wx,uwiy,abvx->abiy', gamma1['a'], gamma1['a'], h['aa'][a,a,c,a], t['aa'][pv,pv,ha,ha], optimize=True)
    O['aa'][v,v,c,a] += scale * -0.25000000 * np.einsum('uv,wx,abvx,uwiy->abiy', gamma1['a'], gamma1['a'], h['aa'][v,v,a,a], t['aa'][pa,pa,hc,ha], optimize=True)
    O['aa'][v,v,c,a] += scale * +0.50000000 * np.einsum('uv,iujw,abiv->abjw', gamma1['a'], h['aa'][c,a,c,a], t['aa'][pv,pv,hc,ha], optimize=True)
    O['aa'][v,v,c,a] += scale * -1.00000000 * np.einsum('uv,uaib,cbwv->aciw', gamma1['a'], h['aa'][a,v,c,v], t['aa'][pv,pv,ha,ha], optimize=True)
    O['aa'][v,v,c,a] += scale * +1.00000000 * np.einsum('uv,uawb,cbiv->aciw', gamma1['a'], h['aa'][a,v,a,v], t['aa'][pv,pv,hc,ha], optimize=True)
    O['aa'][v,v,c,a] += scale * +0.25000000 * np.einsum('ijku,abij->abku', h['aa'][c,c,c,a], t['aa'][pv,pv,hc,hc], optimize=True)
    O['aa'][v,v,c,a] += scale * +1.00000000 * np.einsum('iajb,cbiu->acju', h['aa'][c,v,c,v], t['aa'][pv,pv,hc,ha], optimize=True)
    O['aa'][v,v,c,a] += scale * +1.00000000 * np.einsum('iaub,cbji->acju', h['aa'][c,v,a,v], t['aa'][pv,pv,hc,hc], optimize=True)
    O['aa'][v,v,c,a] += scale * +0.25000000 * np.einsum('abcd,cdiu->abiu', h['aa'][v,v,v,v], t['aa'][pv,pv,hc,ha], optimize=True)
    O['aa'][a,a,c,c] += scale * +0.12500000 * np.einsum('uv,wx,yzvx,uwij->yzij', eta1['a'], eta1['a'], h['aa'][a,a,a,a], t['aa'][pa,pa,hc,hc], optimize=True)
    
    O['aa'][a,a,c,c] += scale * -1.00000000 * np.einsum('uv,wx,yuix,zwjv->yzij', eta1['a'], gamma1['a'], h['aa'][a,a,c,a], t['aa'][pa,pa,hc,ha], optimize=True)
    O['aa'][a,a,c,c] += scale * +1.00000000 * np.einsum('uv,wx,ywiv,zujx->yzij', eta1['a'], gamma1['a'], h['aa'][a,a,c,a], t['aa'][pa,pa,hc,ha], optimize=True)
    O['aa'][a,a,c,c] += scale * -1.00000000 * np.einsum('uv,iwjv,xuki->wxjk', eta1['a'], h['aa'][c,a,c,a], t['aa'][pa,pa,hc,hc], optimize=True)
    O['aa'][a,a,c,c] += scale * +0.25000000 * np.einsum('uv,wxva,uaij->wxij', eta1['a'], h['aa'][a,a,a,v], t['aa'][pa,pv,hc,hc], optimize=True)
    O['aa'][a,a,c,c] += scale * -0.12500000 * np.einsum('uv,wx,yzvx,uwij->yzij', gamma1['a'], gamma1['a'], h['aa'][a,a,a,a], t['aa'][pa,pa,hc,hc], optimize=True)
    
    O['aa'][a,a,c,c] += scale * +0.25000000 * np.einsum('uv,iujk,wxiv->wxjk', gamma1['a'], h['aa'][c,a,c,c], t['aa'][pa,pa,hc,ha], optimize=True)
    O['aa'][a,a,c,c] += scale * +1.00000000 * np.einsum('uv,wuia,xajv->wxij', gamma1['a'], h['aa'][a,a,c,v], t['aa'][pa,pv,hc,ha], optimize=True)
    O['aa'][a,a,c,c] += scale * +0.12500000 * np.einsum('ijkl,uvij->uvkl', h['aa'][c,c,c,c], t['aa'][pa,pa,hc,hc], optimize=True)
    O['aa'][a,a,c,c] += scale * -1.00000000 * np.einsum('iuja,vaki->uvjk', h['aa'][c,a,c,v], t['aa'][pa,pv,hc,hc], optimize=True)
    O['aa'][a,a,c,c] += scale * +0.12500000 * np.einsum('uvab,abij->uvij', h['aa'][a,a,v,v], t['aa'][pv,pv,hc,hc], optimize=True)
    O['aa'][c,a,c,c] += scale * +0.25000000 * np.einsum('uv,wx,iyvx,uwjk->iyjk', eta1['a'], eta1['a'], h['aa'][c,a,a,a], t['aa'][pa,pa,hc,hc], optimize=True)
    O['aa'][c,a,c,c] += scale * -1.00000000 * np.einsum('uv,wx,iujx,ywkv->iyjk', eta1['a'], gamma1['a'], h['aa'][c,a,c,a], t['aa'][pa,pa,hc,ha], optimize=True)
    O['aa'][c,a,c,c] += scale * +1.00000000 * np.einsum('uv,wx,iwjv,yukx->iyjk', eta1['a'], gamma1['a'], h['aa'][c,a,c,a], t['aa'][pa,pa,hc,ha], optimize=True)
    O['aa'][c,a,c,c] += scale * +1.00000000 * np.einsum('uv,ijkv,wulj->iwkl', eta1['a'], h['aa'][c,c,c,a], t['aa'][pa,pa,hc,hc], optimize=True)
    O['aa'][c,a,c,c] += scale * +0.50000000 * np.einsum('uv,iwva,uajk->iwjk', eta1['a'], h['aa'][c,a,a,v], t['aa'][pa,pv,hc,hc], optimize=True)
    O['aa'][c,a,c,c] += scale * -0.25000000 * np.einsum('uv,wx,iyvx,uwjk->iyjk', gamma1['a'], gamma1['a'], h['aa'][c,a,a,a], t['aa'][pa,pa,hc,hc], optimize=True)
    O['aa'][c,a,c,c] += scale * +1.00000000 * np.einsum('uv,iuja,wakv->iwjk', gamma1['a'], h['aa'][c,a,c,v], t['aa'][pa,pv,hc,ha], optimize=True)
    O['aa'][c,a,c,c] += scale * +1.00000000 * np.einsum('ijka,ualj->iukl', h['aa'][c,c,c,v], t['aa'][pa,pv,hc,hc], optimize=True)
    O['aa'][c,a,c,c] += scale * +0.25000000 * np.einsum('iuab,abjk->iujk', h['aa'][c,a,v,v], t['aa'][pv,pv,hc,hc], optimize=True)
    O['aa'][c,c,c,c] += scale * +0.12500000 * np.einsum('uv,wx,ijvx,uwkl->ijkl', eta1['a'], eta1['a'], h['aa'][c,c,a,a], t['aa'][pa,pa,hc,hc], optimize=True)
    O['aa'][c,c,c,c] += scale * +0.25000000 * np.einsum('uv,ijva,uakl->ijkl', eta1['a'], h['aa'][c,c,a,v], t['aa'][pa,pv,hc,hc], optimize=True)
    O['aa'][c,c,c,c] += scale * -0.12500000 * np.einsum('uv,wx,ijvx,uwkl->ijkl', gamma1['a'], gamma1['a'], h['aa'][c,c,a,a], t['aa'][pa,pa,hc,hc], optimize=True)
    O['aa'][c,c,c,c] += scale * +0.12500000 * np.einsum('ijab,abkl->ijkl', h['aa'][c,c,v,v], t['aa'][pv,pv,hc,hc], optimize=True)
    O['aa'][a,v,c,c] += scale * +0.25000000 * np.einsum('uv,wx,yavx,uwij->yaij', eta1['a'], eta1['a'], h['aa'][a,v,a,a], t['aa'][pa,pa,hc,hc], optimize=True)
    O['aa'][a,v,c,c] += scale * -0.25000000 * np.einsum('uv,wx,uwij,yavx->yaij', eta1['a'], eta1['a'], h['aa'][a,a,c,c], t['aa'][pa,pv,ha,ha], optimize=True)
    O['aa'][a,v,c,c] += scale * +1.00000000 * np.einsum('uv,wx,yuix,wajv->yaij', eta1['a'], gamma1['a'], h['aa'][a,a,c,a], t['aa'][pa,pv,hc,ha], optimize=True)
    O['aa'][a,v,c,c] += scale * -1.00000000 * np.einsum('uv,wx,ywiv,uajx->yaij', eta1['a'], gamma1['a'], h['aa'][a,a,c,a], t['aa'][pa,pv,hc,ha], optimize=True)
    O['aa'][a,v,c,c] += scale * -1.00000000 * np.einsum('uv,wx,uaix,ywjv->yaij', eta1['a'], gamma1['a'], h['aa'][a,v,c,a], t['aa'][pa,pa,hc,ha], optimize=True)
    O['aa'][a,v,c,c] += scale * +1.00000000 * np.einsum('uv,wx,waiv,yujx->yaij', eta1['a'], gamma1['a'], h['aa'][a,v,c,a], t['aa'][pa,pa,hc,ha], optimize=True)
    O['aa'][a,v,c,c] += scale * +1.00000000 * np.einsum('uv,iwjv,uaki->wajk', eta1['a'], h['aa'][c,a,c,a], t['aa'][pa,pv,hc,hc], optimize=True)
    O['aa'][a,v,c,c] += scale * +1.00000000 * np.einsum('uv,iajv,wuki->wajk', eta1['a'], h['aa'][c,v,c,a], t['aa'][pa,pa,hc,hc], optimize=True)
    O['aa'][a,v,c,c] += scale * +0.50000000 * np.einsum('uv,wavb,ubij->waij', eta1['a'], h['aa'][a,v,a,v], t['aa'][pa,pv,hc,hc], optimize=True)
    O['aa'][a,v,c,c] += scale * -0.25000000 * np.einsum('uv,wx,yavx,uwij->yaij', gamma1['a'], gamma1['a'], h['aa'][a,v,a,a], t['aa'][pa,pa,hc,hc], optimize=True)
    O['aa'][a,v,c,c] += scale * +0.25000000 * np.einsum('uv,wx,uwij,yavx->yaij', gamma1['a'], gamma1['a'], h['aa'][a,a,c,c], t['aa'][pa,pv,ha,ha], optimize=True)
    O['aa'][a,v,c,c] += scale * +0.50000000 * np.einsum('uv,iujk,waiv->wajk', gamma1['a'], h['aa'][c,a,c,c], t['aa'][pa,pv,hc,ha], optimize=True)
    O['aa'][a,v,c,c] += scale * +1.00000000 * np.einsum('uv,wuia,bajv->wbij', gamma1['a'], h['aa'][a,a,c,v], t['aa'][pv,pv,hc,ha], optimize=True)
    O['aa'][a,v,c,c] += scale * +1.00000000 * np.einsum('uv,uaib,wbjv->waij', gamma1['a'], h['aa'][a,v,c,v], t['aa'][pa,pv,hc,ha], optimize=True)
    O['aa'][a,v,c,c] += scale * +0.25000000 * np.einsum('ijkl,uaij->uakl', h['aa'][c,c,c,c], t['aa'][pa,pv,hc,hc], optimize=True)
    O['aa'][a,v,c,c] += scale * -1.00000000 * np.einsum('iuja,baki->ubjk', h['aa'][c,a,c,v], t['aa'][pv,pv,hc,hc], optimize=True)
    O['aa'][a,v,c,c] += scale * +1.00000000 * np.einsum('iajb,ubki->uajk', h['aa'][c,v,c,v], t['aa'][pa,pv,hc,hc], optimize=True)
    O['aa'][a,v,c,c] += scale * +0.25000000 * np.einsum('uabc,bcij->uaij', h['aa'][a,v,v,v], t['aa'][pv,pv,hc,hc], optimize=True)
    O['aa'][c,v,c,c] += scale * +0.25000000 * np.einsum('uv,wx,iavx,uwjk->iajk', eta1['a'], eta1['a'], h['aa'][c,v,a,a], t['aa'][pa,pa,hc,hc], optimize=True)
    O['aa'][c,v,c,c] += scale * +1.00000000 * np.einsum('uv,wx,iujx,wakv->iajk', eta1['a'], gamma1['a'], h['aa'][c,a,c,a], t['aa'][pa,pv,hc,ha], optimize=True)
    O['aa'][c,v,c,c] += scale * -1.00000000 * np.einsum('uv,wx,iwjv,uakx->iajk', eta1['a'], gamma1['a'], h['aa'][c,a,c,a], t['aa'][pa,pv,hc,ha], optimize=True)
    O['aa'][c,v,c,c] += scale * -1.00000000 * np.einsum('uv,ijkv,ualj->iakl', eta1['a'], h['aa'][c,c,c,a], t['aa'][pa,pv,hc,hc], optimize=True)
    O['aa'][c,v,c,c] += scale * +0.50000000 * np.einsum('uv,iavb,ubjk->iajk', eta1['a'], h['aa'][c,v,a,v], t['aa'][pa,pv,hc,hc], optimize=True)
    O['aa'][c,v,c,c] += scale * -0.25000000 * np.einsum('uv,wx,iavx,uwjk->iajk', gamma1['a'], gamma1['a'], h['aa'][c,v,a,a], t['aa'][pa,pa,hc,hc], optimize=True)
    O['aa'][c,v,c,c] += scale * +1.00000000 * np.einsum('uv,iuja,bakv->ibjk', gamma1['a'], h['aa'][c,a,c,v], t['aa'][pv,pv,hc,ha], optimize=True)
    O['aa'][c,v,c,c] += scale * +1.00000000 * np.einsum('ijka,balj->ibkl', h['aa'][c,c,c,v], t['aa'][pv,pv,hc,hc], optimize=True)
    O['aa'][c,v,c,c] += scale * +0.25000000 * np.einsum('iabc,bcjk->iajk', h['aa'][c,v,v,v], t['aa'][pv,pv,hc,hc], optimize=True)
    O['aa'][v,v,c,c] += scale * -0.12500000 * np.einsum('uv,wx,uwij,abvx->abij', eta1['a'], eta1['a'], h['aa'][a,a,c,c], t['aa'][pv,pv,ha,ha], optimize=True)
    O['aa'][v,v,c,c] += scale * +0.12500000 * np.einsum('uv,wx,abvx,uwij->abij', eta1['a'], eta1['a'], h['aa'][v,v,a,a], t['aa'][pa,pa,hc,hc], optimize=True)
    O['aa'][v,v,c,c] += scale * -1.00000000 * np.einsum('uv,wx,uaix,wbjv->abij', eta1['a'], gamma1['a'], h['aa'][a,v,c,a], t['aa'][pa,pv,hc,ha], optimize=True)
    O['aa'][v,v,c,c] += scale * +1.00000000 * np.einsum('uv,wx,waiv,ubjx->abij', eta1['a'], gamma1['a'], h['aa'][a,v,c,a], t['aa'][pa,pv,hc,ha], optimize=True)
    O['aa'][v,v,c,c] += scale * +1.00000000 * np.einsum('uv,iajv,ubki->abjk', eta1['a'], h['aa'][c,v,c,a], t['aa'][pa,pv,hc,hc], optimize=True)
    O['aa'][v,v,c,c] += scale * +0.25000000 * np.einsum('uv,abvc,ucij->abij', eta1['a'], h['aa'][v,v,a,v], t['aa'][pa,pv,hc,hc], optimize=True)
    O['aa'][v,v,c,c] += scale * +0.12500000 * np.einsum('uv,wx,uwij,abvx->abij', gamma1['a'], gamma1['a'], h['aa'][a,a,c,c], t['aa'][pv,pv,ha,ha], optimize=True)
    O['aa'][v,v,c,c] += scale * -0.12500000 * np.einsum('uv,wx,abvx,uwij->abij', gamma1['a'], gamma1['a'], h['aa'][v,v,a,a], t['aa'][pa,pa,hc,hc], optimize=True)
    O['aa'][v,v,c,c] += scale * +0.25000000 * np.einsum('uv,iujk,abiv->abjk', gamma1['a'], h['aa'][c,a,c,c], t['aa'][pv,pv,hc,ha], optimize=True)
    O['aa'][v,v,c,c] += scale * -1.00000000 * np.einsum('uv,uaib,cbjv->acij', gamma1['a'], h['aa'][a,v,c,v], t['aa'][pv,pv,hc,ha], optimize=True)
    O['aa'][v,v,c,c] += scale * +0.12500000 * np.einsum('ijkl,abij->abkl', h['aa'][c,c,c,c], t['aa'][pv,pv,hc,hc], optimize=True)
    O['aa'][v,v,c,c] += scale * -1.00000000 * np.einsum('iajb,cbki->acjk', h['aa'][c,v,c,v], t['aa'][pv,pv,hc,hc], optimize=True)
    O['aa'][v,v,c,c] += scale * +0.12500000 * np.einsum('abcd,cdij->abij', h['aa'][v,v,v,v], t['aa'][pv,pv,hc,hc], optimize=True)
    
    O['aa'][a,a,c,v] += scale * -1.00000000 * np.einsum('uv,wx,yuxa,zwiv->yzia', eta1['a'], gamma1['a'], h['aa'][a,a,a,v], t['aa'][pa,pa,hc,ha], optimize=True)
    O['aa'][a,a,c,v] += scale * +1.00000000 * np.einsum('uv,wx,ywva,zuix->yzia', eta1['a'], gamma1['a'], h['aa'][a,a,a,v], t['aa'][pa,pa,hc,ha], optimize=True)
    O['aa'][a,a,c,v] += scale * -1.00000000 * np.einsum('uv,iwva,xuji->wxja', eta1['a'], h['aa'][c,a,a,v], t['aa'][pa,pa,hc,hc], optimize=True)
    
    O['aa'][a,a,c,v] += scale * +0.50000000 * np.einsum('uv,iuja,wxiv->wxja', gamma1['a'], h['aa'][c,a,c,v], t['aa'][pa,pa,hc,ha], optimize=True)
    O['aa'][a,a,c,v] += scale * -1.00000000 * np.einsum('uv,wuab,xbiv->wxia', gamma1['a'], h['aa'][a,a,v,v], t['aa'][pa,pv,hc,ha], optimize=True)
    O['aa'][a,a,c,v] += scale * +0.25000000 * np.einsum('ijka,uvij->uvka', h['aa'][c,c,c,v], t['aa'][pa,pa,hc,hc], optimize=True)
    O['aa'][a,a,c,v] += scale * +1.00000000 * np.einsum('iuab,vbji->uvja', h['aa'][c,a,v,v], t['aa'][pa,pv,hc,hc], optimize=True)
    O['aa'][c,a,c,v] += scale * -1.00000000 * np.einsum('uv,wx,iuxa,ywjv->iyja', eta1['a'], gamma1['a'], h['aa'][c,a,a,v], t['aa'][pa,pa,hc,ha], optimize=True)
    O['aa'][c,a,c,v] += scale * +1.00000000 * np.einsum('uv,wx,iwva,yujx->iyja', eta1['a'], gamma1['a'], h['aa'][c,a,a,v], t['aa'][pa,pa,hc,ha], optimize=True)
    O['aa'][c,a,c,v] += scale * +1.00000000 * np.einsum('uv,ijva,wukj->iwka', eta1['a'], h['aa'][c,c,a,v], t['aa'][pa,pa,hc,hc], optimize=True)
    O['aa'][c,a,c,v] += scale * -1.00000000 * np.einsum('uv,iuab,wbjv->iwja', gamma1['a'], h['aa'][c,a,v,v], t['aa'][pa,pv,hc,ha], optimize=True)
    O['aa'][c,a,c,v] += scale * -1.00000000 * np.einsum('ijab,ubkj->iuka', h['aa'][c,c,v,v], t['aa'][pa,pv,hc,hc], optimize=True)
    O['aa'][a,v,c,v] += scale * -0.50000000 * np.einsum('uv,wx,uwia,ybvx->ybia', eta1['a'], eta1['a'], h['aa'][a,a,c,v], t['aa'][pa,pv,ha,ha], optimize=True)
    O['aa'][a,v,c,v] += scale * +1.00000000 * np.einsum('uv,wx,yuxa,wbiv->ybia', eta1['a'], gamma1['a'], h['aa'][a,a,a,v], t['aa'][pa,pv,hc,ha], optimize=True)
    O['aa'][a,v,c,v] += scale * -1.00000000 * np.einsum('uv,wx,ywva,ubix->ybia', eta1['a'], gamma1['a'], h['aa'][a,a,a,v], t['aa'][pa,pv,hc,ha], optimize=True)
    O['aa'][a,v,c,v] += scale * -1.00000000 * np.einsum('uv,wx,uaxb,ywiv->yaib', eta1['a'], gamma1['a'], h['aa'][a,v,a,v], t['aa'][pa,pa,hc,ha], optimize=True)
    O['aa'][a,v,c,v] += scale * +1.00000000 * np.einsum('uv,wx,wavb,yuix->yaib', eta1['a'], gamma1['a'], h['aa'][a,v,a,v], t['aa'][pa,pa,hc,ha], optimize=True)
    O['aa'][a,v,c,v] += scale * +1.00000000 * np.einsum('uv,iwva,ubji->wbja', eta1['a'], h['aa'][c,a,a,v], t['aa'][pa,pv,hc,hc], optimize=True)
    O['aa'][a,v,c,v] += scale * +1.00000000 * np.einsum('uv,iavb,wuji->wajb', eta1['a'], h['aa'][c,v,a,v], t['aa'][pa,pa,hc,hc], optimize=True)
    O['aa'][a,v,c,v] += scale * +0.50000000 * np.einsum('uv,wx,uwia,ybvx->ybia', gamma1['a'], gamma1['a'], h['aa'][a,a,c,v], t['aa'][pa,pv,ha,ha], optimize=True)
    O['aa'][a,v,c,v] += scale * +1.00000000 * np.einsum('uv,iuja,wbiv->wbja', gamma1['a'], h['aa'][c,a,c,v], t['aa'][pa,pv,hc,ha], optimize=True)
    O['aa'][a,v,c,v] += scale * -1.00000000 * np.einsum('uv,wuab,cbiv->wcia', gamma1['a'], h['aa'][a,a,v,v], t['aa'][pv,pv,hc,ha], optimize=True)
    O['aa'][a,v,c,v] += scale * -1.00000000 * np.einsum('uv,uabc,wciv->waib', gamma1['a'], h['aa'][a,v,v,v], t['aa'][pa,pv,hc,ha], optimize=True)
    O['aa'][a,v,c,v] += scale * +0.50000000 * np.einsum('ijka,ubij->ubka', h['aa'][c,c,c,v], t['aa'][pa,pv,hc,hc], optimize=True)
    O['aa'][a,v,c,v] += scale * +1.00000000 * np.einsum('iuab,cbji->ucja', h['aa'][c,a,v,v], t['aa'][pv,pv,hc,hc], optimize=True)
    O['aa'][a,v,c,v] += scale * -1.00000000 * np.einsum('iabc,ucji->uajb', h['aa'][c,v,v,v], t['aa'][pa,pv,hc,hc], optimize=True)
    O['aa'][c,v,c,v] += scale * +1.00000000 * np.einsum('uv,wx,iuxa,wbjv->ibja', eta1['a'], gamma1['a'], h['aa'][c,a,a,v], t['aa'][pa,pv,hc,ha], optimize=True)
    O['aa'][c,v,c,v] += scale * -1.00000000 * np.einsum('uv,wx,iwva,ubjx->ibja', eta1['a'], gamma1['a'], h['aa'][c,a,a,v], t['aa'][pa,pv,hc,ha], optimize=True)
    O['aa'][c,v,c,v] += scale * -1.00000000 * np.einsum('uv,ijva,ubkj->ibka', eta1['a'], h['aa'][c,c,a,v], t['aa'][pa,pv,hc,hc], optimize=True)
    O['aa'][c,v,c,v] += scale * -1.00000000 * np.einsum('uv,iuab,cbjv->icja', gamma1['a'], h['aa'][c,a,v,v], t['aa'][pv,pv,hc,ha], optimize=True)
    O['aa'][c,v,c,v] += scale * -1.00000000 * np.einsum('ijab,cbkj->icka', h['aa'][c,c,v,v], t['aa'][pv,pv,hc,hc], optimize=True)
    O['aa'][v,v,c,v] += scale * -0.25000000 * np.einsum('uv,wx,uwia,bcvx->bcia', eta1['a'], eta1['a'], h['aa'][a,a,c,v], t['aa'][pv,pv,ha,ha], optimize=True)
    O['aa'][v,v,c,v] += scale * -1.00000000 * np.einsum('uv,wx,uaxb,wciv->acib', eta1['a'], gamma1['a'], h['aa'][a,v,a,v], t['aa'][pa,pv,hc,ha], optimize=True)
    O['aa'][v,v,c,v] += scale * +1.00000000 * np.einsum('uv,wx,wavb,ucix->acib', eta1['a'], gamma1['a'], h['aa'][a,v,a,v], t['aa'][pa,pv,hc,ha], optimize=True)
    O['aa'][v,v,c,v] += scale * +1.00000000 * np.einsum('uv,iavb,ucji->acjb', eta1['a'], h['aa'][c,v,a,v], t['aa'][pa,pv,hc,hc], optimize=True)
    O['aa'][v,v,c,v] += scale * +0.25000000 * np.einsum('uv,wx,uwia,bcvx->bcia', gamma1['a'], gamma1['a'], h['aa'][a,a,c,v], t['aa'][pv,pv,ha,ha], optimize=True)
    O['aa'][v,v,c,v] += scale * +0.50000000 * np.einsum('uv,iuja,bciv->bcja', gamma1['a'], h['aa'][c,a,c,v], t['aa'][pv,pv,hc,ha], optimize=True)
    O['aa'][v,v,c,v] += scale * +1.00000000 * np.einsum('uv,uabc,dciv->adib', gamma1['a'], h['aa'][a,v,v,v], t['aa'][pv,pv,hc,ha], optimize=True)
    O['aa'][v,v,c,v] += scale * +0.25000000 * np.einsum('ijka,bcij->bcka', h['aa'][c,c,c,v], t['aa'][pv,pv,hc,hc], optimize=True)
    O['aa'][v,v,c,v] += scale * +1.00000000 * np.einsum('iabc,dcji->adjb', h['aa'][c,v,v,v], t['aa'][pv,pv,hc,hc], optimize=True)
    
    
    O['aa'][a,a,v,v] += scale * +0.25000000 * np.einsum('uv,iuab,wxiv->wxab', gamma1['a'], h['aa'][c,a,v,v], t['aa'][pa,pa,hc,ha], optimize=True)
    O['aa'][a,a,v,v] += scale * +0.12500000 * np.einsum('ijab,uvij->uvab', h['aa'][c,c,v,v], t['aa'][pa,pa,hc,hc], optimize=True)
    O['aa'][a,v,v,v] += scale * -0.25000000 * np.einsum('uv,wx,uwab,ycvx->ycab', eta1['a'], eta1['a'], h['aa'][a,a,v,v], t['aa'][pa,pv,ha,ha], optimize=True)
    O['aa'][a,v,v,v] += scale * +0.25000000 * np.einsum('uv,wx,uwab,ycvx->ycab', gamma1['a'], gamma1['a'], h['aa'][a,a,v,v], t['aa'][pa,pv,ha,ha], optimize=True)
    O['aa'][a,v,v,v] += scale * +0.50000000 * np.einsum('uv,iuab,wciv->wcab', gamma1['a'], h['aa'][c,a,v,v], t['aa'][pa,pv,hc,ha], optimize=True)
    O['aa'][a,v,v,v] += scale * +0.25000000 * np.einsum('ijab,ucij->ucab', h['aa'][c,c,v,v], t['aa'][pa,pv,hc,hc], optimize=True)
    O['aa'][v,v,v,v] += scale * -0.12500000 * np.einsum('uv,wx,uwab,cdvx->cdab', eta1['a'], eta1['a'], h['aa'][a,a,v,v], t['aa'][pv,pv,ha,ha], optimize=True)
    O['aa'][v,v,v,v] += scale * +0.12500000 * np.einsum('uv,wx,uwab,cdvx->cdab', gamma1['a'], gamma1['a'], h['aa'][a,a,v,v], t['aa'][pv,pv,ha,ha], optimize=True)
    O['aa'][v,v,v,v] += scale * +0.25000000 * np.einsum('uv,iuab,cdiv->cdab', gamma1['a'], h['aa'][c,a,v,v], t['aa'][pv,pv,hc,ha], optimize=True)
    O['aa'][v,v,v,v] += scale * +0.12500000 * np.einsum('ijab,cdij->cdab', h['aa'][c,c,v,v], t['aa'][pv,pv,hc,hc], optimize=True)

    t1 = time.time()
    if verbose: print("h2a_t2a_c2a took {:.4f} seconds to run.".format(t1-t0))

    return O
def h2b_t2b_c2a(O, h, t, gamma1, eta1, lambdas, orbspace, verbose=False, scale=1.0):
    # 180 lines
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

    
    
    O['aa'][a,a,a,a] += scale * +1.00000000 * np.einsum('UV,uIvV,wUxI->uwvx', eta1['b'], h['ab'][a,C,a,A], t['ab'][pa,pA,ha,hC], optimize=True)
    O['aa'][a,a,a,a] += scale * +1.00000000 * np.einsum('UV,uUvA,wAxV->uwvx', gamma1['b'], h['ab'][a,A,a,V], t['ab'][pa,pV,ha,hA], optimize=True)
    O['aa'][a,a,a,a] += scale * +1.00000000 * np.einsum('uIvA,wAxI->uwvx', h['ab'][a,C,a,V], t['ab'][pa,pV,ha,hC], optimize=True)
    
    
    O['aa'][c,a,a,a] += scale * +1.00000000 * np.einsum('UV,iIuV,vUwI->ivuw', eta1['b'], h['ab'][c,C,a,A], t['ab'][pa,pA,ha,hC], optimize=True)
    O['aa'][c,a,a,a] += scale * +1.00000000 * np.einsum('UV,iUuA,vAwV->ivuw', gamma1['b'], h['ab'][c,A,a,V], t['ab'][pa,pV,ha,hA], optimize=True)
    O['aa'][c,a,a,a] += scale * +1.00000000 * np.einsum('iIuA,vAwI->ivuw', h['ab'][c,C,a,V], t['ab'][pa,pV,ha,hC], optimize=True)
    O['aa'][a,v,a,a] += scale * -1.00000000 * np.einsum('UV,WX,uUvX,aWwV->uavw', eta1['b'], gamma1['b'], h['ab'][a,A,a,A], t['ab'][pv,pA,ha,hA], optimize=True)
    O['aa'][a,v,a,a] += scale * +1.00000000 * np.einsum('UV,WX,uWvV,aUwX->uavw', eta1['b'], gamma1['b'], h['ab'][a,A,a,A], t['ab'][pv,pA,ha,hA], optimize=True)
    
    
    O['aa'][a,v,a,a] += scale * +1.00000000 * np.einsum('UV,uIvV,aUwI->uavw', eta1['b'], h['ab'][a,C,a,A], t['ab'][pv,pA,ha,hC], optimize=True)
    O['aa'][a,v,a,a] += scale * -1.00000000 * np.einsum('UV,aIuV,vUwI->vauw', eta1['b'], h['ab'][v,C,a,A], t['ab'][pa,pA,ha,hC], optimize=True)
    O['aa'][a,v,a,a] += scale * +1.00000000 * np.einsum('UV,uUvA,aAwV->uavw', gamma1['b'], h['ab'][a,A,a,V], t['ab'][pv,pV,ha,hA], optimize=True)
    O['aa'][a,v,a,a] += scale * -1.00000000 * np.einsum('UV,aUuA,vAwV->vauw', gamma1['b'], h['ab'][v,A,a,V], t['ab'][pa,pV,ha,hA], optimize=True)
    O['aa'][a,v,a,a] += scale * +1.00000000 * np.einsum('uIvA,aAwI->uavw', h['ab'][a,C,a,V], t['ab'][pv,pV,ha,hC], optimize=True)
    O['aa'][a,v,a,a] += scale * -1.00000000 * np.einsum('aIuA,vAwI->vauw', h['ab'][v,C,a,V], t['ab'][pa,pV,ha,hC], optimize=True)
    O['aa'][c,v,a,a] += scale * -1.00000000 * np.einsum('UV,WX,iUuX,aWvV->iauv', eta1['b'], gamma1['b'], h['ab'][c,A,a,A], t['ab'][pv,pA,ha,hA], optimize=True)
    O['aa'][c,v,a,a] += scale * +1.00000000 * np.einsum('UV,WX,iWuV,aUvX->iauv', eta1['b'], gamma1['b'], h['ab'][c,A,a,A], t['ab'][pv,pA,ha,hA], optimize=True)
    O['aa'][c,v,a,a] += scale * +1.00000000 * np.einsum('UV,iIuV,aUvI->iauv', eta1['b'], h['ab'][c,C,a,A], t['ab'][pv,pA,ha,hC], optimize=True)
    O['aa'][c,v,a,a] += scale * +1.00000000 * np.einsum('UV,iUuA,aAvV->iauv', gamma1['b'], h['ab'][c,A,a,V], t['ab'][pv,pV,ha,hA], optimize=True)
    O['aa'][c,v,a,a] += scale * +1.00000000 * np.einsum('iIuA,aAvI->iauv', h['ab'][c,C,a,V], t['ab'][pv,pV,ha,hC], optimize=True)
    O['aa'][v,v,a,a] += scale * -1.00000000 * np.einsum('UV,WX,aUuX,bWvV->abuv', eta1['b'], gamma1['b'], h['ab'][v,A,a,A], t['ab'][pv,pA,ha,hA], optimize=True)
    O['aa'][v,v,a,a] += scale * +1.00000000 * np.einsum('UV,WX,aWuV,bUvX->abuv', eta1['b'], gamma1['b'], h['ab'][v,A,a,A], t['ab'][pv,pA,ha,hA], optimize=True)
    O['aa'][v,v,a,a] += scale * +1.00000000 * np.einsum('UV,aIuV,bUvI->abuv', eta1['b'], h['ab'][v,C,a,A], t['ab'][pv,pA,ha,hC], optimize=True)
    O['aa'][v,v,a,a] += scale * +1.00000000 * np.einsum('UV,aUuA,bAvV->abuv', gamma1['b'], h['ab'][v,A,a,V], t['ab'][pv,pV,ha,hA], optimize=True)
    O['aa'][v,v,a,a] += scale * +1.00000000 * np.einsum('aIuA,bAvI->abuv', h['ab'][v,C,a,V], t['ab'][pv,pV,ha,hC], optimize=True)
    
    
    O['aa'][a,a,a,v] += scale * -1.00000000 * np.einsum('UV,uIaV,vUwI->uvwa', eta1['b'], h['ab'][a,C,v,A], t['ab'][pa,pA,ha,hC], optimize=True)
    O['aa'][a,a,a,v] += scale * -1.00000000 * np.einsum('UV,uUaA,vAwV->uvwa', gamma1['b'], h['ab'][a,A,v,V], t['ab'][pa,pV,ha,hA], optimize=True)
    O['aa'][a,a,a,v] += scale * -1.00000000 * np.einsum('uIaA,vAwI->uvwa', h['ab'][a,C,v,V], t['ab'][pa,pV,ha,hC], optimize=True)
    
    
    O['aa'][c,a,a,v] += scale * -1.00000000 * np.einsum('UV,iIaV,uUvI->iuva', eta1['b'], h['ab'][c,C,v,A], t['ab'][pa,pA,ha,hC], optimize=True)
    O['aa'][c,a,a,v] += scale * -1.00000000 * np.einsum('UV,iUaA,uAvV->iuva', gamma1['b'], h['ab'][c,A,v,V], t['ab'][pa,pV,ha,hA], optimize=True)
    O['aa'][c,a,a,v] += scale * -1.00000000 * np.einsum('iIaA,uAvI->iuva', h['ab'][c,C,v,V], t['ab'][pa,pV,ha,hC], optimize=True)
    O['aa'][a,v,a,v] += scale * +1.00000000 * np.einsum('UV,WX,uUaX,bWvV->ubva', eta1['b'], gamma1['b'], h['ab'][a,A,v,A], t['ab'][pv,pA,ha,hA], optimize=True)
    O['aa'][a,v,a,v] += scale * -1.00000000 * np.einsum('UV,WX,uWaV,bUvX->ubva', eta1['b'], gamma1['b'], h['ab'][a,A,v,A], t['ab'][pv,pA,ha,hA], optimize=True)
    
    
    O['aa'][a,v,a,v] += scale * -1.00000000 * np.einsum('UV,uIaV,bUvI->ubva', eta1['b'], h['ab'][a,C,v,A], t['ab'][pv,pA,ha,hC], optimize=True)
    O['aa'][a,v,a,v] += scale * +1.00000000 * np.einsum('UV,aIbV,uUvI->uavb', eta1['b'], h['ab'][v,C,v,A], t['ab'][pa,pA,ha,hC], optimize=True)
    O['aa'][a,v,a,v] += scale * -1.00000000 * np.einsum('UV,uUaA,bAvV->ubva', gamma1['b'], h['ab'][a,A,v,V], t['ab'][pv,pV,ha,hA], optimize=True)
    O['aa'][a,v,a,v] += scale * +1.00000000 * np.einsum('UV,aUbA,uAvV->uavb', gamma1['b'], h['ab'][v,A,v,V], t['ab'][pa,pV,ha,hA], optimize=True)
    O['aa'][a,v,a,v] += scale * -1.00000000 * np.einsum('uIaA,bAvI->ubva', h['ab'][a,C,v,V], t['ab'][pv,pV,ha,hC], optimize=True)
    O['aa'][a,v,a,v] += scale * +1.00000000 * np.einsum('aIbA,uAvI->uavb', h['ab'][v,C,v,V], t['ab'][pa,pV,ha,hC], optimize=True)
    O['aa'][c,v,a,v] += scale * +1.00000000 * np.einsum('UV,WX,iUaX,bWuV->ibua', eta1['b'], gamma1['b'], h['ab'][c,A,v,A], t['ab'][pv,pA,ha,hA], optimize=True)
    O['aa'][c,v,a,v] += scale * -1.00000000 * np.einsum('UV,WX,iWaV,bUuX->ibua', eta1['b'], gamma1['b'], h['ab'][c,A,v,A], t['ab'][pv,pA,ha,hA], optimize=True)
    O['aa'][c,v,a,v] += scale * -1.00000000 * np.einsum('UV,iIaV,bUuI->ibua', eta1['b'], h['ab'][c,C,v,A], t['ab'][pv,pA,ha,hC], optimize=True)
    O['aa'][c,v,a,v] += scale * -1.00000000 * np.einsum('UV,iUaA,bAuV->ibua', gamma1['b'], h['ab'][c,A,v,V], t['ab'][pv,pV,ha,hA], optimize=True)
    O['aa'][c,v,a,v] += scale * -1.00000000 * np.einsum('iIaA,bAuI->ibua', h['ab'][c,C,v,V], t['ab'][pv,pV,ha,hC], optimize=True)
    O['aa'][v,v,a,v] += scale * +1.00000000 * np.einsum('UV,WX,aUbX,cWuV->acub', eta1['b'], gamma1['b'], h['ab'][v,A,v,A], t['ab'][pv,pA,ha,hA], optimize=True)
    O['aa'][v,v,a,v] += scale * -1.00000000 * np.einsum('UV,WX,aWbV,cUuX->acub', eta1['b'], gamma1['b'], h['ab'][v,A,v,A], t['ab'][pv,pA,ha,hA], optimize=True)
    O['aa'][v,v,a,v] += scale * -1.00000000 * np.einsum('UV,aIbV,cUuI->acub', eta1['b'], h['ab'][v,C,v,A], t['ab'][pv,pA,ha,hC], optimize=True)
    O['aa'][v,v,a,v] += scale * -1.00000000 * np.einsum('UV,aUbA,cAuV->acub', gamma1['b'], h['ab'][v,A,v,V], t['ab'][pv,pV,ha,hA], optimize=True)
    O['aa'][v,v,a,v] += scale * -1.00000000 * np.einsum('aIbA,cAuI->acub', h['ab'][v,C,v,V], t['ab'][pv,pV,ha,hC], optimize=True)
    
    O['aa'][a,a,c,a] += scale * +1.00000000 * np.einsum('UV,WX,uUvX,wWiV->uwiv', eta1['b'], gamma1['b'], h['ab'][a,A,a,A], t['ab'][pa,pA,hc,hA], optimize=True)
    
    O['aa'][a,a,c,a] += scale * -1.00000000 * np.einsum('UV,WX,uWvV,wUiX->uwiv', eta1['b'], gamma1['b'], h['ab'][a,A,a,A], t['ab'][pa,pA,hc,hA], optimize=True)
    O['aa'][a,a,c,a] += scale * +1.00000000 * np.einsum('UV,uIiV,vUwI->uviw', eta1['b'], h['ab'][a,C,c,A], t['ab'][pa,pA,ha,hC], optimize=True)
    O['aa'][a,a,c,a] += scale * -1.00000000 * np.einsum('UV,uIvV,wUiI->uwiv', eta1['b'], h['ab'][a,C,a,A], t['ab'][pa,pA,hc,hC], optimize=True)
    O['aa'][a,a,c,a] += scale * +1.00000000 * np.einsum('UV,uUiA,vAwV->uviw', gamma1['b'], h['ab'][a,A,c,V], t['ab'][pa,pV,ha,hA], optimize=True)
    O['aa'][a,a,c,a] += scale * -1.00000000 * np.einsum('UV,uUvA,wAiV->uwiv', gamma1['b'], h['ab'][a,A,a,V], t['ab'][pa,pV,hc,hA], optimize=True)
    O['aa'][a,a,c,a] += scale * +1.00000000 * np.einsum('uIiA,vAwI->uviw', h['ab'][a,C,c,V], t['ab'][pa,pV,ha,hC], optimize=True)
    O['aa'][a,a,c,a] += scale * -1.00000000 * np.einsum('uIvA,wAiI->uwiv', h['ab'][a,C,a,V], t['ab'][pa,pV,hc,hC], optimize=True)
    
    O['aa'][c,a,c,a] += scale * +1.00000000 * np.einsum('UV,WX,iUuX,vWjV->ivju', eta1['b'], gamma1['b'], h['ab'][c,A,a,A], t['ab'][pa,pA,hc,hA], optimize=True)
    
    O['aa'][c,a,c,a] += scale * -1.00000000 * np.einsum('UV,WX,iWuV,vUjX->ivju', eta1['b'], gamma1['b'], h['ab'][c,A,a,A], t['ab'][pa,pA,hc,hA], optimize=True)
    O['aa'][c,a,c,a] += scale * +1.00000000 * np.einsum('UV,iIjV,uUvI->iujv', eta1['b'], h['ab'][c,C,c,A], t['ab'][pa,pA,ha,hC], optimize=True)
    O['aa'][c,a,c,a] += scale * -1.00000000 * np.einsum('UV,iIuV,vUjI->ivju', eta1['b'], h['ab'][c,C,a,A], t['ab'][pa,pA,hc,hC], optimize=True)
    O['aa'][c,a,c,a] += scale * +1.00000000 * np.einsum('UV,iUjA,uAvV->iujv', gamma1['b'], h['ab'][c,A,c,V], t['ab'][pa,pV,ha,hA], optimize=True)
    O['aa'][c,a,c,a] += scale * -1.00000000 * np.einsum('UV,iUuA,vAjV->ivju', gamma1['b'], h['ab'][c,A,a,V], t['ab'][pa,pV,hc,hA], optimize=True)
    O['aa'][c,a,c,a] += scale * +1.00000000 * np.einsum('iIjA,uAvI->iujv', h['ab'][c,C,c,V], t['ab'][pa,pV,ha,hC], optimize=True)
    O['aa'][c,a,c,a] += scale * -1.00000000 * np.einsum('iIuA,vAjI->ivju', h['ab'][c,C,a,V], t['ab'][pa,pV,hc,hC], optimize=True)
    O['aa'][a,v,c,a] += scale * -1.00000000 * np.einsum('UV,WX,uUiX,aWvV->uaiv', eta1['b'], gamma1['b'], h['ab'][a,A,c,A], t['ab'][pv,pA,ha,hA], optimize=True)
    O['aa'][a,v,c,a] += scale * +1.00000000 * np.einsum('UV,WX,uUvX,aWiV->uaiv', eta1['b'], gamma1['b'], h['ab'][a,A,a,A], t['ab'][pv,pA,hc,hA], optimize=True)
    O['aa'][a,v,c,a] += scale * +1.00000000 * np.einsum('UV,WX,uWiV,aUvX->uaiv', eta1['b'], gamma1['b'], h['ab'][a,A,c,A], t['ab'][pv,pA,ha,hA], optimize=True)
    O['aa'][a,v,c,a] += scale * -1.00000000 * np.einsum('UV,WX,uWvV,aUiX->uaiv', eta1['b'], gamma1['b'], h['ab'][a,A,a,A], t['ab'][pv,pA,hc,hA], optimize=True)
    
    O['aa'][a,v,c,a] += scale * -1.00000000 * np.einsum('UV,WX,aUuX,vWiV->vaiu', eta1['b'], gamma1['b'], h['ab'][v,A,a,A], t['ab'][pa,pA,hc,hA], optimize=True)
    
    O['aa'][a,v,c,a] += scale * +1.00000000 * np.einsum('UV,WX,aWuV,vUiX->vaiu', eta1['b'], gamma1['b'], h['ab'][v,A,a,A], t['ab'][pa,pA,hc,hA], optimize=True)
    O['aa'][a,v,c,a] += scale * +1.00000000 * np.einsum('UV,uIiV,aUvI->uaiv', eta1['b'], h['ab'][a,C,c,A], t['ab'][pv,pA,ha,hC], optimize=True)
    O['aa'][a,v,c,a] += scale * -1.00000000 * np.einsum('UV,uIvV,aUiI->uaiv', eta1['b'], h['ab'][a,C,a,A], t['ab'][pv,pA,hc,hC], optimize=True)
    O['aa'][a,v,c,a] += scale * -1.00000000 * np.einsum('UV,aIiV,uUvI->uaiv', eta1['b'], h['ab'][v,C,c,A], t['ab'][pa,pA,ha,hC], optimize=True)
    O['aa'][a,v,c,a] += scale * +1.00000000 * np.einsum('UV,aIuV,vUiI->vaiu', eta1['b'], h['ab'][v,C,a,A], t['ab'][pa,pA,hc,hC], optimize=True)
    O['aa'][a,v,c,a] += scale * +1.00000000 * np.einsum('UV,uUiA,aAvV->uaiv', gamma1['b'], h['ab'][a,A,c,V], t['ab'][pv,pV,ha,hA], optimize=True)
    O['aa'][a,v,c,a] += scale * -1.00000000 * np.einsum('UV,uUvA,aAiV->uaiv', gamma1['b'], h['ab'][a,A,a,V], t['ab'][pv,pV,hc,hA], optimize=True)
    O['aa'][a,v,c,a] += scale * -1.00000000 * np.einsum('UV,aUiA,uAvV->uaiv', gamma1['b'], h['ab'][v,A,c,V], t['ab'][pa,pV,ha,hA], optimize=True)
    O['aa'][a,v,c,a] += scale * +1.00000000 * np.einsum('UV,aUuA,vAiV->vaiu', gamma1['b'], h['ab'][v,A,a,V], t['ab'][pa,pV,hc,hA], optimize=True)
    O['aa'][a,v,c,a] += scale * +1.00000000 * np.einsum('uIiA,aAvI->uaiv', h['ab'][a,C,c,V], t['ab'][pv,pV,ha,hC], optimize=True)
    O['aa'][a,v,c,a] += scale * -1.00000000 * np.einsum('uIvA,aAiI->uaiv', h['ab'][a,C,a,V], t['ab'][pv,pV,hc,hC], optimize=True)
    O['aa'][a,v,c,a] += scale * -1.00000000 * np.einsum('aIiA,uAvI->uaiv', h['ab'][v,C,c,V], t['ab'][pa,pV,ha,hC], optimize=True)
    O['aa'][a,v,c,a] += scale * +1.00000000 * np.einsum('aIuA,vAiI->vaiu', h['ab'][v,C,a,V], t['ab'][pa,pV,hc,hC], optimize=True)
    O['aa'][c,v,c,a] += scale * -1.00000000 * np.einsum('UV,WX,iUjX,aWuV->iaju', eta1['b'], gamma1['b'], h['ab'][c,A,c,A], t['ab'][pv,pA,ha,hA], optimize=True)
    O['aa'][c,v,c,a] += scale * +1.00000000 * np.einsum('UV,WX,iUuX,aWjV->iaju', eta1['b'], gamma1['b'], h['ab'][c,A,a,A], t['ab'][pv,pA,hc,hA], optimize=True)
    O['aa'][c,v,c,a] += scale * +1.00000000 * np.einsum('UV,WX,iWjV,aUuX->iaju', eta1['b'], gamma1['b'], h['ab'][c,A,c,A], t['ab'][pv,pA,ha,hA], optimize=True)
    O['aa'][c,v,c,a] += scale * -1.00000000 * np.einsum('UV,WX,iWuV,aUjX->iaju', eta1['b'], gamma1['b'], h['ab'][c,A,a,A], t['ab'][pv,pA,hc,hA], optimize=True)
    O['aa'][c,v,c,a] += scale * +1.00000000 * np.einsum('UV,iIjV,aUuI->iaju', eta1['b'], h['ab'][c,C,c,A], t['ab'][pv,pA,ha,hC], optimize=True)
    O['aa'][c,v,c,a] += scale * -1.00000000 * np.einsum('UV,iIuV,aUjI->iaju', eta1['b'], h['ab'][c,C,a,A], t['ab'][pv,pA,hc,hC], optimize=True)
    O['aa'][c,v,c,a] += scale * +1.00000000 * np.einsum('UV,iUjA,aAuV->iaju', gamma1['b'], h['ab'][c,A,c,V], t['ab'][pv,pV,ha,hA], optimize=True)
    O['aa'][c,v,c,a] += scale * -1.00000000 * np.einsum('UV,iUuA,aAjV->iaju', gamma1['b'], h['ab'][c,A,a,V], t['ab'][pv,pV,hc,hA], optimize=True)
    O['aa'][c,v,c,a] += scale * +1.00000000 * np.einsum('iIjA,aAuI->iaju', h['ab'][c,C,c,V], t['ab'][pv,pV,ha,hC], optimize=True)
    O['aa'][c,v,c,a] += scale * -1.00000000 * np.einsum('iIuA,aAjI->iaju', h['ab'][c,C,a,V], t['ab'][pv,pV,hc,hC], optimize=True)
    O['aa'][v,v,c,a] += scale * -1.00000000 * np.einsum('UV,WX,aUiX,bWuV->abiu', eta1['b'], gamma1['b'], h['ab'][v,A,c,A], t['ab'][pv,pA,ha,hA], optimize=True)
    O['aa'][v,v,c,a] += scale * +1.00000000 * np.einsum('UV,WX,aUuX,bWiV->abiu', eta1['b'], gamma1['b'], h['ab'][v,A,a,A], t['ab'][pv,pA,hc,hA], optimize=True)
    O['aa'][v,v,c,a] += scale * +1.00000000 * np.einsum('UV,WX,aWiV,bUuX->abiu', eta1['b'], gamma1['b'], h['ab'][v,A,c,A], t['ab'][pv,pA,ha,hA], optimize=True)
    O['aa'][v,v,c,a] += scale * -1.00000000 * np.einsum('UV,WX,aWuV,bUiX->abiu', eta1['b'], gamma1['b'], h['ab'][v,A,a,A], t['ab'][pv,pA,hc,hA], optimize=True)
    O['aa'][v,v,c,a] += scale * +1.00000000 * np.einsum('UV,aIiV,bUuI->abiu', eta1['b'], h['ab'][v,C,c,A], t['ab'][pv,pA,ha,hC], optimize=True)
    O['aa'][v,v,c,a] += scale * -1.00000000 * np.einsum('UV,aIuV,bUiI->abiu', eta1['b'], h['ab'][v,C,a,A], t['ab'][pv,pA,hc,hC], optimize=True)
    O['aa'][v,v,c,a] += scale * +1.00000000 * np.einsum('UV,aUiA,bAuV->abiu', gamma1['b'], h['ab'][v,A,c,V], t['ab'][pv,pV,ha,hA], optimize=True)
    O['aa'][v,v,c,a] += scale * -1.00000000 * np.einsum('UV,aUuA,bAiV->abiu', gamma1['b'], h['ab'][v,A,a,V], t['ab'][pv,pV,hc,hA], optimize=True)
    O['aa'][v,v,c,a] += scale * +1.00000000 * np.einsum('aIiA,bAuI->abiu', h['ab'][v,C,c,V], t['ab'][pv,pV,ha,hC], optimize=True)
    O['aa'][v,v,c,a] += scale * -1.00000000 * np.einsum('aIuA,bAiI->abiu', h['ab'][v,C,a,V], t['ab'][pv,pV,hc,hC], optimize=True)
    O['aa'][a,a,c,c] += scale * -1.00000000 * np.einsum('UV,WX,uUiX,vWjV->uvij', eta1['b'], gamma1['b'], h['ab'][a,A,c,A], t['ab'][pa,pA,hc,hA], optimize=True)
    O['aa'][a,a,c,c] += scale * +1.00000000 * np.einsum('UV,WX,uWiV,vUjX->uvij', eta1['b'], gamma1['b'], h['ab'][a,A,c,A], t['ab'][pa,pA,hc,hA], optimize=True)
    O['aa'][a,a,c,c] += scale * +1.00000000 * np.einsum('UV,uIiV,vUjI->uvij', eta1['b'], h['ab'][a,C,c,A], t['ab'][pa,pA,hc,hC], optimize=True)
    O['aa'][a,a,c,c] += scale * +1.00000000 * np.einsum('UV,uUiA,vAjV->uvij', gamma1['b'], h['ab'][a,A,c,V], t['ab'][pa,pV,hc,hA], optimize=True)
    O['aa'][a,a,c,c] += scale * +1.00000000 * np.einsum('uIiA,vAjI->uvij', h['ab'][a,C,c,V], t['ab'][pa,pV,hc,hC], optimize=True)
    O['aa'][c,a,c,c] += scale * -1.00000000 * np.einsum('UV,WX,iUjX,uWkV->iujk', eta1['b'], gamma1['b'], h['ab'][c,A,c,A], t['ab'][pa,pA,hc,hA], optimize=True)
    O['aa'][c,a,c,c] += scale * +1.00000000 * np.einsum('UV,WX,iWjV,uUkX->iujk', eta1['b'], gamma1['b'], h['ab'][c,A,c,A], t['ab'][pa,pA,hc,hA], optimize=True)
    O['aa'][c,a,c,c] += scale * +1.00000000 * np.einsum('UV,iIjV,uUkI->iujk', eta1['b'], h['ab'][c,C,c,A], t['ab'][pa,pA,hc,hC], optimize=True)
    O['aa'][c,a,c,c] += scale * +1.00000000 * np.einsum('UV,iUjA,uAkV->iujk', gamma1['b'], h['ab'][c,A,c,V], t['ab'][pa,pV,hc,hA], optimize=True)
    O['aa'][c,a,c,c] += scale * +1.00000000 * np.einsum('iIjA,uAkI->iujk', h['ab'][c,C,c,V], t['ab'][pa,pV,hc,hC], optimize=True)
    O['aa'][a,v,c,c] += scale * -1.00000000 * np.einsum('UV,WX,uUiX,aWjV->uaij', eta1['b'], gamma1['b'], h['ab'][a,A,c,A], t['ab'][pv,pA,hc,hA], optimize=True)
    O['aa'][a,v,c,c] += scale * +1.00000000 * np.einsum('UV,WX,uWiV,aUjX->uaij', eta1['b'], gamma1['b'], h['ab'][a,A,c,A], t['ab'][pv,pA,hc,hA], optimize=True)
    O['aa'][a,v,c,c] += scale * +1.00000000 * np.einsum('UV,WX,aUiX,uWjV->uaij', eta1['b'], gamma1['b'], h['ab'][v,A,c,A], t['ab'][pa,pA,hc,hA], optimize=True)
    O['aa'][a,v,c,c] += scale * -1.00000000 * np.einsum('UV,WX,aWiV,uUjX->uaij', eta1['b'], gamma1['b'], h['ab'][v,A,c,A], t['ab'][pa,pA,hc,hA], optimize=True)
    O['aa'][a,v,c,c] += scale * +1.00000000 * np.einsum('UV,uIiV,aUjI->uaij', eta1['b'], h['ab'][a,C,c,A], t['ab'][pv,pA,hc,hC], optimize=True)
    O['aa'][a,v,c,c] += scale * -1.00000000 * np.einsum('UV,aIiV,uUjI->uaij', eta1['b'], h['ab'][v,C,c,A], t['ab'][pa,pA,hc,hC], optimize=True)
    O['aa'][a,v,c,c] += scale * +1.00000000 * np.einsum('UV,uUiA,aAjV->uaij', gamma1['b'], h['ab'][a,A,c,V], t['ab'][pv,pV,hc,hA], optimize=True)
    O['aa'][a,v,c,c] += scale * -1.00000000 * np.einsum('UV,aUiA,uAjV->uaij', gamma1['b'], h['ab'][v,A,c,V], t['ab'][pa,pV,hc,hA], optimize=True)
    O['aa'][a,v,c,c] += scale * +1.00000000 * np.einsum('uIiA,aAjI->uaij', h['ab'][a,C,c,V], t['ab'][pv,pV,hc,hC], optimize=True)
    O['aa'][a,v,c,c] += scale * -1.00000000 * np.einsum('aIiA,uAjI->uaij', h['ab'][v,C,c,V], t['ab'][pa,pV,hc,hC], optimize=True)
    O['aa'][c,v,c,c] += scale * -1.00000000 * np.einsum('UV,WX,iUjX,aWkV->iajk', eta1['b'], gamma1['b'], h['ab'][c,A,c,A], t['ab'][pv,pA,hc,hA], optimize=True)
    O['aa'][c,v,c,c] += scale * +1.00000000 * np.einsum('UV,WX,iWjV,aUkX->iajk', eta1['b'], gamma1['b'], h['ab'][c,A,c,A], t['ab'][pv,pA,hc,hA], optimize=True)
    O['aa'][c,v,c,c] += scale * +1.00000000 * np.einsum('UV,iIjV,aUkI->iajk', eta1['b'], h['ab'][c,C,c,A], t['ab'][pv,pA,hc,hC], optimize=True)
    O['aa'][c,v,c,c] += scale * +1.00000000 * np.einsum('UV,iUjA,aAkV->iajk', gamma1['b'], h['ab'][c,A,c,V], t['ab'][pv,pV,hc,hA], optimize=True)
    O['aa'][c,v,c,c] += scale * +1.00000000 * np.einsum('iIjA,aAkI->iajk', h['ab'][c,C,c,V], t['ab'][pv,pV,hc,hC], optimize=True)
    O['aa'][v,v,c,c] += scale * -1.00000000 * np.einsum('UV,WX,aUiX,bWjV->abij', eta1['b'], gamma1['b'], h['ab'][v,A,c,A], t['ab'][pv,pA,hc,hA], optimize=True)
    O['aa'][v,v,c,c] += scale * +1.00000000 * np.einsum('UV,WX,aWiV,bUjX->abij', eta1['b'], gamma1['b'], h['ab'][v,A,c,A], t['ab'][pv,pA,hc,hA], optimize=True)
    O['aa'][v,v,c,c] += scale * +1.00000000 * np.einsum('UV,aIiV,bUjI->abij', eta1['b'], h['ab'][v,C,c,A], t['ab'][pv,pA,hc,hC], optimize=True)
    O['aa'][v,v,c,c] += scale * +1.00000000 * np.einsum('UV,aUiA,bAjV->abij', gamma1['b'], h['ab'][v,A,c,V], t['ab'][pv,pV,hc,hA], optimize=True)
    O['aa'][v,v,c,c] += scale * +1.00000000 * np.einsum('aIiA,bAjI->abij', h['ab'][v,C,c,V], t['ab'][pv,pV,hc,hC], optimize=True)
    O['aa'][a,a,c,v] += scale * +1.00000000 * np.einsum('UV,WX,uUaX,vWiV->uvia', eta1['b'], gamma1['b'], h['ab'][a,A,v,A], t['ab'][pa,pA,hc,hA], optimize=True)
    O['aa'][a,a,c,v] += scale * -1.00000000 * np.einsum('UV,WX,uWaV,vUiX->uvia', eta1['b'], gamma1['b'], h['ab'][a,A,v,A], t['ab'][pa,pA,hc,hA], optimize=True)
    O['aa'][a,a,c,v] += scale * -1.00000000 * np.einsum('UV,uIaV,vUiI->uvia', eta1['b'], h['ab'][a,C,v,A], t['ab'][pa,pA,hc,hC], optimize=True)
    O['aa'][a,a,c,v] += scale * -1.00000000 * np.einsum('UV,uUaA,vAiV->uvia', gamma1['b'], h['ab'][a,A,v,V], t['ab'][pa,pV,hc,hA], optimize=True)
    O['aa'][a,a,c,v] += scale * -1.00000000 * np.einsum('uIaA,vAiI->uvia', h['ab'][a,C,v,V], t['ab'][pa,pV,hc,hC], optimize=True)
    O['aa'][c,a,c,v] += scale * +1.00000000 * np.einsum('UV,WX,iUaX,uWjV->iuja', eta1['b'], gamma1['b'], h['ab'][c,A,v,A], t['ab'][pa,pA,hc,hA], optimize=True)
    O['aa'][c,a,c,v] += scale * -1.00000000 * np.einsum('UV,WX,iWaV,uUjX->iuja', eta1['b'], gamma1['b'], h['ab'][c,A,v,A], t['ab'][pa,pA,hc,hA], optimize=True)
    O['aa'][c,a,c,v] += scale * -1.00000000 * np.einsum('UV,iIaV,uUjI->iuja', eta1['b'], h['ab'][c,C,v,A], t['ab'][pa,pA,hc,hC], optimize=True)
    O['aa'][c,a,c,v] += scale * -1.00000000 * np.einsum('UV,iUaA,uAjV->iuja', gamma1['b'], h['ab'][c,A,v,V], t['ab'][pa,pV,hc,hA], optimize=True)
    O['aa'][c,a,c,v] += scale * -1.00000000 * np.einsum('iIaA,uAjI->iuja', h['ab'][c,C,v,V], t['ab'][pa,pV,hc,hC], optimize=True)
    O['aa'][a,v,c,v] += scale * +1.00000000 * np.einsum('UV,WX,uUaX,bWiV->ubia', eta1['b'], gamma1['b'], h['ab'][a,A,v,A], t['ab'][pv,pA,hc,hA], optimize=True)
    O['aa'][a,v,c,v] += scale * -1.00000000 * np.einsum('UV,WX,uWaV,bUiX->ubia', eta1['b'], gamma1['b'], h['ab'][a,A,v,A], t['ab'][pv,pA,hc,hA], optimize=True)
    O['aa'][a,v,c,v] += scale * -1.00000000 * np.einsum('UV,WX,aUbX,uWiV->uaib', eta1['b'], gamma1['b'], h['ab'][v,A,v,A], t['ab'][pa,pA,hc,hA], optimize=True)
    O['aa'][a,v,c,v] += scale * +1.00000000 * np.einsum('UV,WX,aWbV,uUiX->uaib', eta1['b'], gamma1['b'], h['ab'][v,A,v,A], t['ab'][pa,pA,hc,hA], optimize=True)
    O['aa'][a,v,c,v] += scale * -1.00000000 * np.einsum('UV,uIaV,bUiI->ubia', eta1['b'], h['ab'][a,C,v,A], t['ab'][pv,pA,hc,hC], optimize=True)
    O['aa'][a,v,c,v] += scale * +1.00000000 * np.einsum('UV,aIbV,uUiI->uaib', eta1['b'], h['ab'][v,C,v,A], t['ab'][pa,pA,hc,hC], optimize=True)
    O['aa'][a,v,c,v] += scale * -1.00000000 * np.einsum('UV,uUaA,bAiV->ubia', gamma1['b'], h['ab'][a,A,v,V], t['ab'][pv,pV,hc,hA], optimize=True)
    O['aa'][a,v,c,v] += scale * +1.00000000 * np.einsum('UV,aUbA,uAiV->uaib', gamma1['b'], h['ab'][v,A,v,V], t['ab'][pa,pV,hc,hA], optimize=True)
    O['aa'][a,v,c,v] += scale * -1.00000000 * np.einsum('uIaA,bAiI->ubia', h['ab'][a,C,v,V], t['ab'][pv,pV,hc,hC], optimize=True)
    O['aa'][a,v,c,v] += scale * +1.00000000 * np.einsum('aIbA,uAiI->uaib', h['ab'][v,C,v,V], t['ab'][pa,pV,hc,hC], optimize=True)
    O['aa'][c,v,c,v] += scale * +1.00000000 * np.einsum('UV,WX,iUaX,bWjV->ibja', eta1['b'], gamma1['b'], h['ab'][c,A,v,A], t['ab'][pv,pA,hc,hA], optimize=True)
    O['aa'][c,v,c,v] += scale * -1.00000000 * np.einsum('UV,WX,iWaV,bUjX->ibja', eta1['b'], gamma1['b'], h['ab'][c,A,v,A], t['ab'][pv,pA,hc,hA], optimize=True)
    O['aa'][c,v,c,v] += scale * -1.00000000 * np.einsum('UV,iIaV,bUjI->ibja', eta1['b'], h['ab'][c,C,v,A], t['ab'][pv,pA,hc,hC], optimize=True)
    O['aa'][c,v,c,v] += scale * -1.00000000 * np.einsum('UV,iUaA,bAjV->ibja', gamma1['b'], h['ab'][c,A,v,V], t['ab'][pv,pV,hc,hA], optimize=True)
    O['aa'][c,v,c,v] += scale * -1.00000000 * np.einsum('iIaA,bAjI->ibja', h['ab'][c,C,v,V], t['ab'][pv,pV,hc,hC], optimize=True)
    O['aa'][v,v,c,v] += scale * +1.00000000 * np.einsum('UV,WX,aUbX,cWiV->acib', eta1['b'], gamma1['b'], h['ab'][v,A,v,A], t['ab'][pv,pA,hc,hA], optimize=True)
    O['aa'][v,v,c,v] += scale * -1.00000000 * np.einsum('UV,WX,aWbV,cUiX->acib', eta1['b'], gamma1['b'], h['ab'][v,A,v,A], t['ab'][pv,pA,hc,hA], optimize=True)
    O['aa'][v,v,c,v] += scale * -1.00000000 * np.einsum('UV,aIbV,cUiI->acib', eta1['b'], h['ab'][v,C,v,A], t['ab'][pv,pA,hc,hC], optimize=True)
    O['aa'][v,v,c,v] += scale * -1.00000000 * np.einsum('UV,aUbA,cAiV->acib', gamma1['b'], h['ab'][v,A,v,V], t['ab'][pv,pV,hc,hA], optimize=True)
    O['aa'][v,v,c,v] += scale * -1.00000000 * np.einsum('aIbA,cAiI->acib', h['ab'][v,C,v,V], t['ab'][pv,pV,hc,hC], optimize=True)

    t1 = time.time()
    if verbose: print("h2b_t2b_c2a took {:.4f} seconds to run.".format(t1-t0))

    return O
def h1a_t2b_c2b(O, h, t, gamma1, eta1, lambdas, orbspace, verbose=False, scale=1.0):
    # 144 lines
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

    
    
    
    
    O['ab'][a,A,a,A] += scale * -1.00000000 * np.einsum('iu,vUiV->vUuV', h['a'][c,a], t['ab'][pa,pA,hc,hA], optimize=True)
    O['ab'][a,A,a,A] += scale * +1.00000000 * np.einsum('ua,aUvV->uUvV', h['a'][a,v], t['ab'][pv,pA,ha,hA], optimize=True)
    
    
    O['ab'][c,A,a,A] += scale * +1.00000000 * np.einsum('ia,aUuV->iUuV', h['a'][c,v], t['ab'][pv,pA,ha,hA], optimize=True)
    O['ab'][v,A,a,A] += scale * -1.00000000 * np.einsum('uv,uw,aUvV->aUwV', eta1['a'], h['a'][a,a], t['ab'][pv,pA,ha,hA], optimize=True)
    
    O['ab'][v,A,a,A] += scale * -1.00000000 * np.einsum('uv,uw,aUvV->aUwV', gamma1['a'], h['a'][a,a], t['ab'][pv,pA,ha,hA], optimize=True)
    
    O['ab'][v,A,a,A] += scale * -1.00000000 * np.einsum('iu,aUiV->aUuV', h['a'][c,a], t['ab'][pv,pA,hc,hA], optimize=True)
    O['ab'][v,A,a,A] += scale * +1.00000000 * np.einsum('ab,bUuV->aUuV', h['a'][v,v], t['ab'][pv,pA,ha,hA], optimize=True)
    O['ab'][a,V,a,A] += scale * +1.00000000 * np.einsum('uv,wv,uAxU->wAxU', eta1['a'], h['a'][a,a], t['ab'][pa,pV,ha,hA], optimize=True)
    O['ab'][a,V,a,A] += scale * -1.00000000 * np.einsum('uv,uw,xAvU->xAwU', eta1['a'], h['a'][a,a], t['ab'][pa,pV,ha,hA], optimize=True)
    O['ab'][a,V,a,A] += scale * +1.00000000 * np.einsum('uv,wv,uAxU->wAxU', gamma1['a'], h['a'][a,a], t['ab'][pa,pV,ha,hA], optimize=True)
    O['ab'][a,V,a,A] += scale * -1.00000000 * np.einsum('uv,uw,xAvU->xAwU', gamma1['a'], h['a'][a,a], t['ab'][pa,pV,ha,hA], optimize=True)
    O['ab'][a,V,a,A] += scale * -1.00000000 * np.einsum('iu,vAiU->vAuU', h['a'][c,a], t['ab'][pa,pV,hc,hA], optimize=True)
    O['ab'][a,V,a,A] += scale * +1.00000000 * np.einsum('ua,aAvU->uAvU', h['a'][a,v], t['ab'][pv,pV,ha,hA], optimize=True)
    O['ab'][c,V,a,A] += scale * +1.00000000 * np.einsum('uv,iv,uAwU->iAwU', eta1['a'], h['a'][c,a], t['ab'][pa,pV,ha,hA], optimize=True)
    O['ab'][c,V,a,A] += scale * +1.00000000 * np.einsum('uv,iv,uAwU->iAwU', gamma1['a'], h['a'][c,a], t['ab'][pa,pV,ha,hA], optimize=True)
    O['ab'][c,V,a,A] += scale * +1.00000000 * np.einsum('ia,aAuU->iAuU', h['a'][c,v], t['ab'][pv,pV,ha,hA], optimize=True)
    O['ab'][v,V,a,A] += scale * -1.00000000 * np.einsum('uv,uw,aAvU->aAwU', eta1['a'], h['a'][a,a], t['ab'][pv,pV,ha,hA], optimize=True)
    O['ab'][v,V,a,A] += scale * +1.00000000 * np.einsum('uv,av,uAwU->aAwU', eta1['a'], h['a'][v,a], t['ab'][pa,pV,ha,hA], optimize=True)
    O['ab'][v,V,a,A] += scale * -1.00000000 * np.einsum('uv,uw,aAvU->aAwU', gamma1['a'], h['a'][a,a], t['ab'][pv,pV,ha,hA], optimize=True)
    O['ab'][v,V,a,A] += scale * +1.00000000 * np.einsum('uv,av,uAwU->aAwU', gamma1['a'], h['a'][v,a], t['ab'][pa,pV,ha,hA], optimize=True)
    O['ab'][v,V,a,A] += scale * -1.00000000 * np.einsum('iu,aAiU->aAuU', h['a'][c,a], t['ab'][pv,pV,hc,hA], optimize=True)
    O['ab'][v,V,a,A] += scale * +1.00000000 * np.einsum('ab,bAuU->aAuU', h['a'][v,v], t['ab'][pv,pV,ha,hA], optimize=True)
    O['ab'][a,A,a,C] += scale * +1.00000000 * np.einsum('uv,wv,uUxI->wUxI', eta1['a'], h['a'][a,a], t['ab'][pa,pA,ha,hC], optimize=True)
    O['ab'][a,A,a,C] += scale * -1.00000000 * np.einsum('uv,uw,xUvI->xUwI', eta1['a'], h['a'][a,a], t['ab'][pa,pA,ha,hC], optimize=True)
    O['ab'][a,A,a,C] += scale * +1.00000000 * np.einsum('uv,wv,uUxI->wUxI', gamma1['a'], h['a'][a,a], t['ab'][pa,pA,ha,hC], optimize=True)
    O['ab'][a,A,a,C] += scale * -1.00000000 * np.einsum('uv,uw,xUvI->xUwI', gamma1['a'], h['a'][a,a], t['ab'][pa,pA,ha,hC], optimize=True)
    O['ab'][a,A,a,C] += scale * -1.00000000 * np.einsum('iu,vUiI->vUuI', h['a'][c,a], t['ab'][pa,pA,hc,hC], optimize=True)
    O['ab'][a,A,a,C] += scale * +1.00000000 * np.einsum('ua,aUvI->uUvI', h['a'][a,v], t['ab'][pv,pA,ha,hC], optimize=True)
    O['ab'][c,A,a,C] += scale * +1.00000000 * np.einsum('uv,iv,uUwI->iUwI', eta1['a'], h['a'][c,a], t['ab'][pa,pA,ha,hC], optimize=True)
    O['ab'][c,A,a,C] += scale * +1.00000000 * np.einsum('uv,iv,uUwI->iUwI', gamma1['a'], h['a'][c,a], t['ab'][pa,pA,ha,hC], optimize=True)
    O['ab'][c,A,a,C] += scale * +1.00000000 * np.einsum('ia,aUuI->iUuI', h['a'][c,v], t['ab'][pv,pA,ha,hC], optimize=True)
    O['ab'][v,A,a,C] += scale * -1.00000000 * np.einsum('uv,uw,aUvI->aUwI', eta1['a'], h['a'][a,a], t['ab'][pv,pA,ha,hC], optimize=True)
    O['ab'][v,A,a,C] += scale * +1.00000000 * np.einsum('uv,av,uUwI->aUwI', eta1['a'], h['a'][v,a], t['ab'][pa,pA,ha,hC], optimize=True)
    O['ab'][v,A,a,C] += scale * -1.00000000 * np.einsum('uv,uw,aUvI->aUwI', gamma1['a'], h['a'][a,a], t['ab'][pv,pA,ha,hC], optimize=True)
    O['ab'][v,A,a,C] += scale * +1.00000000 * np.einsum('uv,av,uUwI->aUwI', gamma1['a'], h['a'][v,a], t['ab'][pa,pA,ha,hC], optimize=True)
    O['ab'][v,A,a,C] += scale * -1.00000000 * np.einsum('iu,aUiI->aUuI', h['a'][c,a], t['ab'][pv,pA,hc,hC], optimize=True)
    O['ab'][v,A,a,C] += scale * +1.00000000 * np.einsum('ab,bUuI->aUuI', h['a'][v,v], t['ab'][pv,pA,ha,hC], optimize=True)
    O['ab'][a,V,a,C] += scale * +1.00000000 * np.einsum('uv,wv,uAxI->wAxI', eta1['a'], h['a'][a,a], t['ab'][pa,pV,ha,hC], optimize=True)
    O['ab'][a,V,a,C] += scale * -1.00000000 * np.einsum('uv,uw,xAvI->xAwI', eta1['a'], h['a'][a,a], t['ab'][pa,pV,ha,hC], optimize=True)
    O['ab'][a,V,a,C] += scale * +1.00000000 * np.einsum('uv,wv,uAxI->wAxI', gamma1['a'], h['a'][a,a], t['ab'][pa,pV,ha,hC], optimize=True)
    O['ab'][a,V,a,C] += scale * -1.00000000 * np.einsum('uv,uw,xAvI->xAwI', gamma1['a'], h['a'][a,a], t['ab'][pa,pV,ha,hC], optimize=True)
    O['ab'][a,V,a,C] += scale * -1.00000000 * np.einsum('iu,vAiI->vAuI', h['a'][c,a], t['ab'][pa,pV,hc,hC], optimize=True)
    O['ab'][a,V,a,C] += scale * +1.00000000 * np.einsum('ua,aAvI->uAvI', h['a'][a,v], t['ab'][pv,pV,ha,hC], optimize=True)
    O['ab'][c,V,a,C] += scale * +1.00000000 * np.einsum('uv,iv,uAwI->iAwI', eta1['a'], h['a'][c,a], t['ab'][pa,pV,ha,hC], optimize=True)
    O['ab'][c,V,a,C] += scale * +1.00000000 * np.einsum('uv,iv,uAwI->iAwI', gamma1['a'], h['a'][c,a], t['ab'][pa,pV,ha,hC], optimize=True)
    O['ab'][c,V,a,C] += scale * +1.00000000 * np.einsum('ia,aAuI->iAuI', h['a'][c,v], t['ab'][pv,pV,ha,hC], optimize=True)
    O['ab'][v,V,a,C] += scale * -1.00000000 * np.einsum('uv,uw,aAvI->aAwI', eta1['a'], h['a'][a,a], t['ab'][pv,pV,ha,hC], optimize=True)
    O['ab'][v,V,a,C] += scale * +1.00000000 * np.einsum('uv,av,uAwI->aAwI', eta1['a'], h['a'][v,a], t['ab'][pa,pV,ha,hC], optimize=True)
    O['ab'][v,V,a,C] += scale * -1.00000000 * np.einsum('uv,uw,aAvI->aAwI', gamma1['a'], h['a'][a,a], t['ab'][pv,pV,ha,hC], optimize=True)
    O['ab'][v,V,a,C] += scale * +1.00000000 * np.einsum('uv,av,uAwI->aAwI', gamma1['a'], h['a'][v,a], t['ab'][pa,pV,ha,hC], optimize=True)
    O['ab'][v,V,a,C] += scale * -1.00000000 * np.einsum('iu,aAiI->aAuI', h['a'][c,a], t['ab'][pv,pV,hc,hC], optimize=True)
    O['ab'][v,V,a,C] += scale * +1.00000000 * np.einsum('ab,bAuI->aAuI', h['a'][v,v], t['ab'][pv,pV,ha,hC], optimize=True)
    O['ab'][a,A,c,A] += scale * +1.00000000 * np.einsum('uv,wv,uUiV->wUiV', eta1['a'], h['a'][a,a], t['ab'][pa,pA,hc,hA], optimize=True)
    
    O['ab'][a,A,c,A] += scale * +1.00000000 * np.einsum('uv,wv,uUiV->wUiV', gamma1['a'], h['a'][a,a], t['ab'][pa,pA,hc,hA], optimize=True)
    
    O['ab'][a,A,c,A] += scale * -1.00000000 * np.einsum('ij,uUiV->uUjV', h['a'][c,c], t['ab'][pa,pA,hc,hA], optimize=True)
    O['ab'][a,A,c,A] += scale * +1.00000000 * np.einsum('ua,aUiV->uUiV', h['a'][a,v], t['ab'][pv,pA,hc,hA], optimize=True)
    O['ab'][c,A,c,A] += scale * +1.00000000 * np.einsum('uv,iv,uUjV->iUjV', eta1['a'], h['a'][c,a], t['ab'][pa,pA,hc,hA], optimize=True)
    O['ab'][c,A,c,A] += scale * +1.00000000 * np.einsum('uv,iv,uUjV->iUjV', gamma1['a'], h['a'][c,a], t['ab'][pa,pA,hc,hA], optimize=True)
    O['ab'][c,A,c,A] += scale * +1.00000000 * np.einsum('ia,aUjV->iUjV', h['a'][c,v], t['ab'][pv,pA,hc,hA], optimize=True)
    O['ab'][v,A,c,A] += scale * -1.00000000 * np.einsum('uv,ui,aUvV->aUiV', eta1['a'], h['a'][a,c], t['ab'][pv,pA,ha,hA], optimize=True)
    O['ab'][v,A,c,A] += scale * +1.00000000 * np.einsum('uv,av,uUiV->aUiV', eta1['a'], h['a'][v,a], t['ab'][pa,pA,hc,hA], optimize=True)
    O['ab'][v,A,c,A] += scale * -1.00000000 * np.einsum('uv,ui,aUvV->aUiV', gamma1['a'], h['a'][a,c], t['ab'][pv,pA,ha,hA], optimize=True)
    O['ab'][v,A,c,A] += scale * +1.00000000 * np.einsum('uv,av,uUiV->aUiV', gamma1['a'], h['a'][v,a], t['ab'][pa,pA,hc,hA], optimize=True)
    O['ab'][v,A,c,A] += scale * -1.00000000 * np.einsum('ij,aUiV->aUjV', h['a'][c,c], t['ab'][pv,pA,hc,hA], optimize=True)
    O['ab'][v,A,c,A] += scale * +1.00000000 * np.einsum('ab,bUiV->aUiV', h['a'][v,v], t['ab'][pv,pA,hc,hA], optimize=True)
    O['ab'][a,V,c,A] += scale * +1.00000000 * np.einsum('uv,wv,uAiU->wAiU', eta1['a'], h['a'][a,a], t['ab'][pa,pV,hc,hA], optimize=True)
    O['ab'][a,V,c,A] += scale * -1.00000000 * np.einsum('uv,ui,wAvU->wAiU', eta1['a'], h['a'][a,c], t['ab'][pa,pV,ha,hA], optimize=True)
    O['ab'][a,V,c,A] += scale * +1.00000000 * np.einsum('uv,wv,uAiU->wAiU', gamma1['a'], h['a'][a,a], t['ab'][pa,pV,hc,hA], optimize=True)
    O['ab'][a,V,c,A] += scale * -1.00000000 * np.einsum('uv,ui,wAvU->wAiU', gamma1['a'], h['a'][a,c], t['ab'][pa,pV,ha,hA], optimize=True)
    O['ab'][a,V,c,A] += scale * -1.00000000 * np.einsum('ij,uAiU->uAjU', h['a'][c,c], t['ab'][pa,pV,hc,hA], optimize=True)
    O['ab'][a,V,c,A] += scale * +1.00000000 * np.einsum('ua,aAiU->uAiU', h['a'][a,v], t['ab'][pv,pV,hc,hA], optimize=True)
    O['ab'][c,V,c,A] += scale * +1.00000000 * np.einsum('uv,iv,uAjU->iAjU', eta1['a'], h['a'][c,a], t['ab'][pa,pV,hc,hA], optimize=True)
    O['ab'][c,V,c,A] += scale * +1.00000000 * np.einsum('uv,iv,uAjU->iAjU', gamma1['a'], h['a'][c,a], t['ab'][pa,pV,hc,hA], optimize=True)
    O['ab'][c,V,c,A] += scale * +1.00000000 * np.einsum('ia,aAjU->iAjU', h['a'][c,v], t['ab'][pv,pV,hc,hA], optimize=True)
    O['ab'][v,V,c,A] += scale * -1.00000000 * np.einsum('uv,ui,aAvU->aAiU', eta1['a'], h['a'][a,c], t['ab'][pv,pV,ha,hA], optimize=True)
    O['ab'][v,V,c,A] += scale * +1.00000000 * np.einsum('uv,av,uAiU->aAiU', eta1['a'], h['a'][v,a], t['ab'][pa,pV,hc,hA], optimize=True)
    O['ab'][v,V,c,A] += scale * -1.00000000 * np.einsum('uv,ui,aAvU->aAiU', gamma1['a'], h['a'][a,c], t['ab'][pv,pV,ha,hA], optimize=True)
    O['ab'][v,V,c,A] += scale * +1.00000000 * np.einsum('uv,av,uAiU->aAiU', gamma1['a'], h['a'][v,a], t['ab'][pa,pV,hc,hA], optimize=True)
    O['ab'][v,V,c,A] += scale * -1.00000000 * np.einsum('ij,aAiU->aAjU', h['a'][c,c], t['ab'][pv,pV,hc,hA], optimize=True)
    O['ab'][v,V,c,A] += scale * +1.00000000 * np.einsum('ab,bAiU->aAiU', h['a'][v,v], t['ab'][pv,pV,hc,hA], optimize=True)
    O['ab'][a,A,c,C] += scale * +1.00000000 * np.einsum('uv,wv,uUiI->wUiI', eta1['a'], h['a'][a,a], t['ab'][pa,pA,hc,hC], optimize=True)
    O['ab'][a,A,c,C] += scale * -1.00000000 * np.einsum('uv,ui,wUvI->wUiI', eta1['a'], h['a'][a,c], t['ab'][pa,pA,ha,hC], optimize=True)
    O['ab'][a,A,c,C] += scale * +1.00000000 * np.einsum('uv,wv,uUiI->wUiI', gamma1['a'], h['a'][a,a], t['ab'][pa,pA,hc,hC], optimize=True)
    O['ab'][a,A,c,C] += scale * -1.00000000 * np.einsum('uv,ui,wUvI->wUiI', gamma1['a'], h['a'][a,c], t['ab'][pa,pA,ha,hC], optimize=True)
    O['ab'][a,A,c,C] += scale * -1.00000000 * np.einsum('ij,uUiI->uUjI', h['a'][c,c], t['ab'][pa,pA,hc,hC], optimize=True)
    O['ab'][a,A,c,C] += scale * +1.00000000 * np.einsum('ua,aUiI->uUiI', h['a'][a,v], t['ab'][pv,pA,hc,hC], optimize=True)
    O['ab'][c,A,c,C] += scale * +1.00000000 * np.einsum('uv,iv,uUjI->iUjI', eta1['a'], h['a'][c,a], t['ab'][pa,pA,hc,hC], optimize=True)
    O['ab'][c,A,c,C] += scale * +1.00000000 * np.einsum('uv,iv,uUjI->iUjI', gamma1['a'], h['a'][c,a], t['ab'][pa,pA,hc,hC], optimize=True)
    O['ab'][c,A,c,C] += scale * +1.00000000 * np.einsum('ia,aUjI->iUjI', h['a'][c,v], t['ab'][pv,pA,hc,hC], optimize=True)
    O['ab'][v,A,c,C] += scale * -1.00000000 * np.einsum('uv,ui,aUvI->aUiI', eta1['a'], h['a'][a,c], t['ab'][pv,pA,ha,hC], optimize=True)
    O['ab'][v,A,c,C] += scale * +1.00000000 * np.einsum('uv,av,uUiI->aUiI', eta1['a'], h['a'][v,a], t['ab'][pa,pA,hc,hC], optimize=True)
    O['ab'][v,A,c,C] += scale * -1.00000000 * np.einsum('uv,ui,aUvI->aUiI', gamma1['a'], h['a'][a,c], t['ab'][pv,pA,ha,hC], optimize=True)
    O['ab'][v,A,c,C] += scale * +1.00000000 * np.einsum('uv,av,uUiI->aUiI', gamma1['a'], h['a'][v,a], t['ab'][pa,pA,hc,hC], optimize=True)
    O['ab'][v,A,c,C] += scale * -1.00000000 * np.einsum('ij,aUiI->aUjI', h['a'][c,c], t['ab'][pv,pA,hc,hC], optimize=True)
    O['ab'][v,A,c,C] += scale * +1.00000000 * np.einsum('ab,bUiI->aUiI', h['a'][v,v], t['ab'][pv,pA,hc,hC], optimize=True)
    O['ab'][a,V,c,C] += scale * +1.00000000 * np.einsum('uv,wv,uAiI->wAiI', eta1['a'], h['a'][a,a], t['ab'][pa,pV,hc,hC], optimize=True)
    O['ab'][a,V,c,C] += scale * -1.00000000 * np.einsum('uv,ui,wAvI->wAiI', eta1['a'], h['a'][a,c], t['ab'][pa,pV,ha,hC], optimize=True)
    O['ab'][a,V,c,C] += scale * +1.00000000 * np.einsum('uv,wv,uAiI->wAiI', gamma1['a'], h['a'][a,a], t['ab'][pa,pV,hc,hC], optimize=True)
    O['ab'][a,V,c,C] += scale * -1.00000000 * np.einsum('uv,ui,wAvI->wAiI', gamma1['a'], h['a'][a,c], t['ab'][pa,pV,ha,hC], optimize=True)
    O['ab'][a,V,c,C] += scale * -1.00000000 * np.einsum('ij,uAiI->uAjI', h['a'][c,c], t['ab'][pa,pV,hc,hC], optimize=True)
    O['ab'][a,V,c,C] += scale * +1.00000000 * np.einsum('ua,aAiI->uAiI', h['a'][a,v], t['ab'][pv,pV,hc,hC], optimize=True)
    O['ab'][c,V,c,C] += scale * +1.00000000 * np.einsum('uv,iv,uAjI->iAjI', eta1['a'], h['a'][c,a], t['ab'][pa,pV,hc,hC], optimize=True)
    O['ab'][c,V,c,C] += scale * +1.00000000 * np.einsum('uv,iv,uAjI->iAjI', gamma1['a'], h['a'][c,a], t['ab'][pa,pV,hc,hC], optimize=True)
    O['ab'][c,V,c,C] += scale * +1.00000000 * np.einsum('ia,aAjI->iAjI', h['a'][c,v], t['ab'][pv,pV,hc,hC], optimize=True)
    O['ab'][v,V,c,C] += scale * -1.00000000 * np.einsum('uv,ui,aAvI->aAiI', eta1['a'], h['a'][a,c], t['ab'][pv,pV,ha,hC], optimize=True)
    O['ab'][v,V,c,C] += scale * +1.00000000 * np.einsum('uv,av,uAiI->aAiI', eta1['a'], h['a'][v,a], t['ab'][pa,pV,hc,hC], optimize=True)
    O['ab'][v,V,c,C] += scale * -1.00000000 * np.einsum('uv,ui,aAvI->aAiI', gamma1['a'], h['a'][a,c], t['ab'][pv,pV,ha,hC], optimize=True)
    O['ab'][v,V,c,C] += scale * +1.00000000 * np.einsum('uv,av,uAiI->aAiI', gamma1['a'], h['a'][v,a], t['ab'][pa,pV,hc,hC], optimize=True)
    O['ab'][v,V,c,C] += scale * -1.00000000 * np.einsum('ij,aAiI->aAjI', h['a'][c,c], t['ab'][pv,pV,hc,hC], optimize=True)
    O['ab'][v,V,c,C] += scale * +1.00000000 * np.einsum('ab,bAiI->aAiI', h['a'][v,v], t['ab'][pv,pV,hc,hC], optimize=True)
    
    
    O['ab'][a,A,v,A] += scale * -1.00000000 * np.einsum('ia,uUiV->uUaV', h['a'][c,v], t['ab'][pa,pA,hc,hA], optimize=True)
    O['ab'][v,A,v,A] += scale * -1.00000000 * np.einsum('uv,ua,bUvV->bUaV', eta1['a'], h['a'][a,v], t['ab'][pv,pA,ha,hA], optimize=True)
    O['ab'][v,A,v,A] += scale * -1.00000000 * np.einsum('uv,ua,bUvV->bUaV', gamma1['a'], h['a'][a,v], t['ab'][pv,pA,ha,hA], optimize=True)
    O['ab'][v,A,v,A] += scale * -1.00000000 * np.einsum('ia,bUiV->bUaV', h['a'][c,v], t['ab'][pv,pA,hc,hA], optimize=True)
    O['ab'][a,V,v,A] += scale * -1.00000000 * np.einsum('uv,ua,wAvU->wAaU', eta1['a'], h['a'][a,v], t['ab'][pa,pV,ha,hA], optimize=True)
    O['ab'][a,V,v,A] += scale * -1.00000000 * np.einsum('uv,ua,wAvU->wAaU', gamma1['a'], h['a'][a,v], t['ab'][pa,pV,ha,hA], optimize=True)
    O['ab'][a,V,v,A] += scale * -1.00000000 * np.einsum('ia,uAiU->uAaU', h['a'][c,v], t['ab'][pa,pV,hc,hA], optimize=True)
    O['ab'][v,V,v,A] += scale * -1.00000000 * np.einsum('uv,ua,bAvU->bAaU', eta1['a'], h['a'][a,v], t['ab'][pv,pV,ha,hA], optimize=True)
    O['ab'][v,V,v,A] += scale * -1.00000000 * np.einsum('uv,ua,bAvU->bAaU', gamma1['a'], h['a'][a,v], t['ab'][pv,pV,ha,hA], optimize=True)
    O['ab'][v,V,v,A] += scale * -1.00000000 * np.einsum('ia,bAiU->bAaU', h['a'][c,v], t['ab'][pv,pV,hc,hA], optimize=True)
    O['ab'][a,A,v,C] += scale * -1.00000000 * np.einsum('uv,ua,wUvI->wUaI', eta1['a'], h['a'][a,v], t['ab'][pa,pA,ha,hC], optimize=True)
    O['ab'][a,A,v,C] += scale * -1.00000000 * np.einsum('uv,ua,wUvI->wUaI', gamma1['a'], h['a'][a,v], t['ab'][pa,pA,ha,hC], optimize=True)
    O['ab'][a,A,v,C] += scale * -1.00000000 * np.einsum('ia,uUiI->uUaI', h['a'][c,v], t['ab'][pa,pA,hc,hC], optimize=True)
    O['ab'][v,A,v,C] += scale * -1.00000000 * np.einsum('uv,ua,bUvI->bUaI', eta1['a'], h['a'][a,v], t['ab'][pv,pA,ha,hC], optimize=True)
    O['ab'][v,A,v,C] += scale * -1.00000000 * np.einsum('uv,ua,bUvI->bUaI', gamma1['a'], h['a'][a,v], t['ab'][pv,pA,ha,hC], optimize=True)
    O['ab'][v,A,v,C] += scale * -1.00000000 * np.einsum('ia,bUiI->bUaI', h['a'][c,v], t['ab'][pv,pA,hc,hC], optimize=True)
    O['ab'][a,V,v,C] += scale * -1.00000000 * np.einsum('uv,ua,wAvI->wAaI', eta1['a'], h['a'][a,v], t['ab'][pa,pV,ha,hC], optimize=True)
    O['ab'][a,V,v,C] += scale * -1.00000000 * np.einsum('uv,ua,wAvI->wAaI', gamma1['a'], h['a'][a,v], t['ab'][pa,pV,ha,hC], optimize=True)
    O['ab'][a,V,v,C] += scale * -1.00000000 * np.einsum('ia,uAiI->uAaI', h['a'][c,v], t['ab'][pa,pV,hc,hC], optimize=True)
    O['ab'][v,V,v,C] += scale * -1.00000000 * np.einsum('uv,ua,bAvI->bAaI', eta1['a'], h['a'][a,v], t['ab'][pv,pV,ha,hC], optimize=True)
    O['ab'][v,V,v,C] += scale * -1.00000000 * np.einsum('uv,ua,bAvI->bAaI', gamma1['a'], h['a'][a,v], t['ab'][pv,pV,ha,hC], optimize=True)
    O['ab'][v,V,v,C] += scale * -1.00000000 * np.einsum('ia,bAiI->bAaI', h['a'][c,v], t['ab'][pv,pV,hc,hC], optimize=True)

    t1 = time.time()
    if verbose: print("h1a_t2b_c2b took {:.4f} seconds to run.".format(t1-t0))

    return O
def h1b_t2b_c2b(O, h, t, gamma1, eta1, lambdas, orbspace, verbose=False, scale=1.0):
    # 144 lines
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

    
    
    
    
    O['ab'][a,A,a,A] += scale * -1.00000000 * np.einsum('IU,uVvI->uVvU', h['b'][C,A], t['ab'][pa,pA,ha,hC], optimize=True)
    O['ab'][a,A,a,A] += scale * +1.00000000 * np.einsum('UA,uAvV->uUvV', h['b'][A,V], t['ab'][pa,pV,ha,hA], optimize=True)
    O['ab'][v,A,a,A] += scale * +1.00000000 * np.einsum('UV,WV,aUuX->aWuX', eta1['b'], h['b'][A,A], t['ab'][pv,pA,ha,hA], optimize=True)
    O['ab'][v,A,a,A] += scale * -1.00000000 * np.einsum('UV,UW,aXuV->aXuW', eta1['b'], h['b'][A,A], t['ab'][pv,pA,ha,hA], optimize=True)
    O['ab'][v,A,a,A] += scale * +1.00000000 * np.einsum('UV,WV,aUuX->aWuX', gamma1['b'], h['b'][A,A], t['ab'][pv,pA,ha,hA], optimize=True)
    O['ab'][v,A,a,A] += scale * -1.00000000 * np.einsum('UV,UW,aXuV->aXuW', gamma1['b'], h['b'][A,A], t['ab'][pv,pA,ha,hA], optimize=True)
    O['ab'][v,A,a,A] += scale * -1.00000000 * np.einsum('IU,aVuI->aVuU', h['b'][C,A], t['ab'][pv,pA,ha,hC], optimize=True)
    O['ab'][v,A,a,A] += scale * +1.00000000 * np.einsum('UA,aAuV->aUuV', h['b'][A,V], t['ab'][pv,pV,ha,hA], optimize=True)
    
    
    O['ab'][a,C,a,A] += scale * +1.00000000 * np.einsum('IA,uAvU->uIvU', h['b'][C,V], t['ab'][pa,pV,ha,hA], optimize=True)
    O['ab'][v,C,a,A] += scale * +1.00000000 * np.einsum('UV,IV,aUuW->aIuW', eta1['b'], h['b'][C,A], t['ab'][pv,pA,ha,hA], optimize=True)
    O['ab'][v,C,a,A] += scale * +1.00000000 * np.einsum('UV,IV,aUuW->aIuW', gamma1['b'], h['b'][C,A], t['ab'][pv,pA,ha,hA], optimize=True)
    O['ab'][v,C,a,A] += scale * +1.00000000 * np.einsum('IA,aAuU->aIuU', h['b'][C,V], t['ab'][pv,pV,ha,hA], optimize=True)
    O['ab'][a,V,a,A] += scale * -1.00000000 * np.einsum('UV,UW,uAvV->uAvW', eta1['b'], h['b'][A,A], t['ab'][pa,pV,ha,hA], optimize=True)
    
    O['ab'][a,V,a,A] += scale * -1.00000000 * np.einsum('UV,UW,uAvV->uAvW', gamma1['b'], h['b'][A,A], t['ab'][pa,pV,ha,hA], optimize=True)
    
    O['ab'][a,V,a,A] += scale * -1.00000000 * np.einsum('IU,uAvI->uAvU', h['b'][C,A], t['ab'][pa,pV,ha,hC], optimize=True)
    O['ab'][a,V,a,A] += scale * +1.00000000 * np.einsum('AB,uBvU->uAvU', h['b'][V,V], t['ab'][pa,pV,ha,hA], optimize=True)
    O['ab'][v,V,a,A] += scale * -1.00000000 * np.einsum('UV,UW,aAuV->aAuW', eta1['b'], h['b'][A,A], t['ab'][pv,pV,ha,hA], optimize=True)
    O['ab'][v,V,a,A] += scale * +1.00000000 * np.einsum('UV,AV,aUuW->aAuW', eta1['b'], h['b'][V,A], t['ab'][pv,pA,ha,hA], optimize=True)
    O['ab'][v,V,a,A] += scale * -1.00000000 * np.einsum('UV,UW,aAuV->aAuW', gamma1['b'], h['b'][A,A], t['ab'][pv,pV,ha,hA], optimize=True)
    O['ab'][v,V,a,A] += scale * +1.00000000 * np.einsum('UV,AV,aUuW->aAuW', gamma1['b'], h['b'][V,A], t['ab'][pv,pA,ha,hA], optimize=True)
    O['ab'][v,V,a,A] += scale * -1.00000000 * np.einsum('IU,aAuI->aAuU', h['b'][C,A], t['ab'][pv,pV,ha,hC], optimize=True)
    O['ab'][v,V,a,A] += scale * +1.00000000 * np.einsum('AB,aBuU->aAuU', h['b'][V,V], t['ab'][pv,pV,ha,hA], optimize=True)
    O['ab'][a,A,a,C] += scale * +1.00000000 * np.einsum('UV,WV,uUvI->uWvI', eta1['b'], h['b'][A,A], t['ab'][pa,pA,ha,hC], optimize=True)
    
    O['ab'][a,A,a,C] += scale * +1.00000000 * np.einsum('UV,WV,uUvI->uWvI', gamma1['b'], h['b'][A,A], t['ab'][pa,pA,ha,hC], optimize=True)
    
    O['ab'][a,A,a,C] += scale * -1.00000000 * np.einsum('IJ,uUvI->uUvJ', h['b'][C,C], t['ab'][pa,pA,ha,hC], optimize=True)
    O['ab'][a,A,a,C] += scale * +1.00000000 * np.einsum('UA,uAvI->uUvI', h['b'][A,V], t['ab'][pa,pV,ha,hC], optimize=True)
    O['ab'][v,A,a,C] += scale * +1.00000000 * np.einsum('UV,WV,aUuI->aWuI', eta1['b'], h['b'][A,A], t['ab'][pv,pA,ha,hC], optimize=True)
    O['ab'][v,A,a,C] += scale * -1.00000000 * np.einsum('UV,UI,aWuV->aWuI', eta1['b'], h['b'][A,C], t['ab'][pv,pA,ha,hA], optimize=True)
    O['ab'][v,A,a,C] += scale * +1.00000000 * np.einsum('UV,WV,aUuI->aWuI', gamma1['b'], h['b'][A,A], t['ab'][pv,pA,ha,hC], optimize=True)
    O['ab'][v,A,a,C] += scale * -1.00000000 * np.einsum('UV,UI,aWuV->aWuI', gamma1['b'], h['b'][A,C], t['ab'][pv,pA,ha,hA], optimize=True)
    O['ab'][v,A,a,C] += scale * -1.00000000 * np.einsum('IJ,aUuI->aUuJ', h['b'][C,C], t['ab'][pv,pA,ha,hC], optimize=True)
    O['ab'][v,A,a,C] += scale * +1.00000000 * np.einsum('UA,aAuI->aUuI', h['b'][A,V], t['ab'][pv,pV,ha,hC], optimize=True)
    O['ab'][a,C,a,C] += scale * +1.00000000 * np.einsum('UV,IV,uUvJ->uIvJ', eta1['b'], h['b'][C,A], t['ab'][pa,pA,ha,hC], optimize=True)
    O['ab'][a,C,a,C] += scale * +1.00000000 * np.einsum('UV,IV,uUvJ->uIvJ', gamma1['b'], h['b'][C,A], t['ab'][pa,pA,ha,hC], optimize=True)
    O['ab'][a,C,a,C] += scale * +1.00000000 * np.einsum('IA,uAvJ->uIvJ', h['b'][C,V], t['ab'][pa,pV,ha,hC], optimize=True)
    O['ab'][v,C,a,C] += scale * +1.00000000 * np.einsum('UV,IV,aUuJ->aIuJ', eta1['b'], h['b'][C,A], t['ab'][pv,pA,ha,hC], optimize=True)
    O['ab'][v,C,a,C] += scale * +1.00000000 * np.einsum('UV,IV,aUuJ->aIuJ', gamma1['b'], h['b'][C,A], t['ab'][pv,pA,ha,hC], optimize=True)
    O['ab'][v,C,a,C] += scale * +1.00000000 * np.einsum('IA,aAuJ->aIuJ', h['b'][C,V], t['ab'][pv,pV,ha,hC], optimize=True)
    O['ab'][a,V,a,C] += scale * -1.00000000 * np.einsum('UV,UI,uAvV->uAvI', eta1['b'], h['b'][A,C], t['ab'][pa,pV,ha,hA], optimize=True)
    O['ab'][a,V,a,C] += scale * +1.00000000 * np.einsum('UV,AV,uUvI->uAvI', eta1['b'], h['b'][V,A], t['ab'][pa,pA,ha,hC], optimize=True)
    O['ab'][a,V,a,C] += scale * -1.00000000 * np.einsum('UV,UI,uAvV->uAvI', gamma1['b'], h['b'][A,C], t['ab'][pa,pV,ha,hA], optimize=True)
    O['ab'][a,V,a,C] += scale * +1.00000000 * np.einsum('UV,AV,uUvI->uAvI', gamma1['b'], h['b'][V,A], t['ab'][pa,pA,ha,hC], optimize=True)
    O['ab'][a,V,a,C] += scale * -1.00000000 * np.einsum('IJ,uAvI->uAvJ', h['b'][C,C], t['ab'][pa,pV,ha,hC], optimize=True)
    O['ab'][a,V,a,C] += scale * +1.00000000 * np.einsum('AB,uBvI->uAvI', h['b'][V,V], t['ab'][pa,pV,ha,hC], optimize=True)
    O['ab'][v,V,a,C] += scale * -1.00000000 * np.einsum('UV,UI,aAuV->aAuI', eta1['b'], h['b'][A,C], t['ab'][pv,pV,ha,hA], optimize=True)
    O['ab'][v,V,a,C] += scale * +1.00000000 * np.einsum('UV,AV,aUuI->aAuI', eta1['b'], h['b'][V,A], t['ab'][pv,pA,ha,hC], optimize=True)
    O['ab'][v,V,a,C] += scale * -1.00000000 * np.einsum('UV,UI,aAuV->aAuI', gamma1['b'], h['b'][A,C], t['ab'][pv,pV,ha,hA], optimize=True)
    O['ab'][v,V,a,C] += scale * +1.00000000 * np.einsum('UV,AV,aUuI->aAuI', gamma1['b'], h['b'][V,A], t['ab'][pv,pA,ha,hC], optimize=True)
    O['ab'][v,V,a,C] += scale * -1.00000000 * np.einsum('IJ,aAuI->aAuJ', h['b'][C,C], t['ab'][pv,pV,ha,hC], optimize=True)
    O['ab'][v,V,a,C] += scale * +1.00000000 * np.einsum('AB,aBuI->aAuI', h['b'][V,V], t['ab'][pv,pV,ha,hC], optimize=True)
    
    
    O['ab'][a,A,a,V] += scale * -1.00000000 * np.einsum('IA,uUvI->uUvA', h['b'][C,V], t['ab'][pa,pA,ha,hC], optimize=True)
    O['ab'][v,A,a,V] += scale * -1.00000000 * np.einsum('UV,UA,aWuV->aWuA', eta1['b'], h['b'][A,V], t['ab'][pv,pA,ha,hA], optimize=True)
    O['ab'][v,A,a,V] += scale * -1.00000000 * np.einsum('UV,UA,aWuV->aWuA', gamma1['b'], h['b'][A,V], t['ab'][pv,pA,ha,hA], optimize=True)
    O['ab'][v,A,a,V] += scale * -1.00000000 * np.einsum('IA,aUuI->aUuA', h['b'][C,V], t['ab'][pv,pA,ha,hC], optimize=True)
    O['ab'][a,V,a,V] += scale * -1.00000000 * np.einsum('UV,UA,uBvV->uBvA', eta1['b'], h['b'][A,V], t['ab'][pa,pV,ha,hA], optimize=True)
    O['ab'][a,V,a,V] += scale * -1.00000000 * np.einsum('UV,UA,uBvV->uBvA', gamma1['b'], h['b'][A,V], t['ab'][pa,pV,ha,hA], optimize=True)
    O['ab'][a,V,a,V] += scale * -1.00000000 * np.einsum('IA,uBvI->uBvA', h['b'][C,V], t['ab'][pa,pV,ha,hC], optimize=True)
    O['ab'][v,V,a,V] += scale * -1.00000000 * np.einsum('UV,UA,aBuV->aBuA', eta1['b'], h['b'][A,V], t['ab'][pv,pV,ha,hA], optimize=True)
    O['ab'][v,V,a,V] += scale * -1.00000000 * np.einsum('UV,UA,aBuV->aBuA', gamma1['b'], h['b'][A,V], t['ab'][pv,pV,ha,hA], optimize=True)
    O['ab'][v,V,a,V] += scale * -1.00000000 * np.einsum('IA,aBuI->aBuA', h['b'][C,V], t['ab'][pv,pV,ha,hC], optimize=True)
    O['ab'][a,A,c,A] += scale * +1.00000000 * np.einsum('UV,WV,uUiX->uWiX', eta1['b'], h['b'][A,A], t['ab'][pa,pA,hc,hA], optimize=True)
    O['ab'][a,A,c,A] += scale * -1.00000000 * np.einsum('UV,UW,uXiV->uXiW', eta1['b'], h['b'][A,A], t['ab'][pa,pA,hc,hA], optimize=True)
    O['ab'][a,A,c,A] += scale * +1.00000000 * np.einsum('UV,WV,uUiX->uWiX', gamma1['b'], h['b'][A,A], t['ab'][pa,pA,hc,hA], optimize=True)
    O['ab'][a,A,c,A] += scale * -1.00000000 * np.einsum('UV,UW,uXiV->uXiW', gamma1['b'], h['b'][A,A], t['ab'][pa,pA,hc,hA], optimize=True)
    O['ab'][a,A,c,A] += scale * -1.00000000 * np.einsum('IU,uViI->uViU', h['b'][C,A], t['ab'][pa,pA,hc,hC], optimize=True)
    O['ab'][a,A,c,A] += scale * +1.00000000 * np.einsum('UA,uAiV->uUiV', h['b'][A,V], t['ab'][pa,pV,hc,hA], optimize=True)
    O['ab'][v,A,c,A] += scale * +1.00000000 * np.einsum('UV,WV,aUiX->aWiX', eta1['b'], h['b'][A,A], t['ab'][pv,pA,hc,hA], optimize=True)
    O['ab'][v,A,c,A] += scale * -1.00000000 * np.einsum('UV,UW,aXiV->aXiW', eta1['b'], h['b'][A,A], t['ab'][pv,pA,hc,hA], optimize=True)
    O['ab'][v,A,c,A] += scale * +1.00000000 * np.einsum('UV,WV,aUiX->aWiX', gamma1['b'], h['b'][A,A], t['ab'][pv,pA,hc,hA], optimize=True)
    O['ab'][v,A,c,A] += scale * -1.00000000 * np.einsum('UV,UW,aXiV->aXiW', gamma1['b'], h['b'][A,A], t['ab'][pv,pA,hc,hA], optimize=True)
    O['ab'][v,A,c,A] += scale * -1.00000000 * np.einsum('IU,aViI->aViU', h['b'][C,A], t['ab'][pv,pA,hc,hC], optimize=True)
    O['ab'][v,A,c,A] += scale * +1.00000000 * np.einsum('UA,aAiV->aUiV', h['b'][A,V], t['ab'][pv,pV,hc,hA], optimize=True)
    O['ab'][a,C,c,A] += scale * +1.00000000 * np.einsum('UV,IV,uUiW->uIiW', eta1['b'], h['b'][C,A], t['ab'][pa,pA,hc,hA], optimize=True)
    O['ab'][a,C,c,A] += scale * +1.00000000 * np.einsum('UV,IV,uUiW->uIiW', gamma1['b'], h['b'][C,A], t['ab'][pa,pA,hc,hA], optimize=True)
    O['ab'][a,C,c,A] += scale * +1.00000000 * np.einsum('IA,uAiU->uIiU', h['b'][C,V], t['ab'][pa,pV,hc,hA], optimize=True)
    O['ab'][v,C,c,A] += scale * +1.00000000 * np.einsum('UV,IV,aUiW->aIiW', eta1['b'], h['b'][C,A], t['ab'][pv,pA,hc,hA], optimize=True)
    O['ab'][v,C,c,A] += scale * +1.00000000 * np.einsum('UV,IV,aUiW->aIiW', gamma1['b'], h['b'][C,A], t['ab'][pv,pA,hc,hA], optimize=True)
    O['ab'][v,C,c,A] += scale * +1.00000000 * np.einsum('IA,aAiU->aIiU', h['b'][C,V], t['ab'][pv,pV,hc,hA], optimize=True)
    O['ab'][a,V,c,A] += scale * -1.00000000 * np.einsum('UV,UW,uAiV->uAiW', eta1['b'], h['b'][A,A], t['ab'][pa,pV,hc,hA], optimize=True)
    O['ab'][a,V,c,A] += scale * +1.00000000 * np.einsum('UV,AV,uUiW->uAiW', eta1['b'], h['b'][V,A], t['ab'][pa,pA,hc,hA], optimize=True)
    O['ab'][a,V,c,A] += scale * -1.00000000 * np.einsum('UV,UW,uAiV->uAiW', gamma1['b'], h['b'][A,A], t['ab'][pa,pV,hc,hA], optimize=True)
    O['ab'][a,V,c,A] += scale * +1.00000000 * np.einsum('UV,AV,uUiW->uAiW', gamma1['b'], h['b'][V,A], t['ab'][pa,pA,hc,hA], optimize=True)
    O['ab'][a,V,c,A] += scale * -1.00000000 * np.einsum('IU,uAiI->uAiU', h['b'][C,A], t['ab'][pa,pV,hc,hC], optimize=True)
    O['ab'][a,V,c,A] += scale * +1.00000000 * np.einsum('AB,uBiU->uAiU', h['b'][V,V], t['ab'][pa,pV,hc,hA], optimize=True)
    O['ab'][v,V,c,A] += scale * -1.00000000 * np.einsum('UV,UW,aAiV->aAiW', eta1['b'], h['b'][A,A], t['ab'][pv,pV,hc,hA], optimize=True)
    O['ab'][v,V,c,A] += scale * +1.00000000 * np.einsum('UV,AV,aUiW->aAiW', eta1['b'], h['b'][V,A], t['ab'][pv,pA,hc,hA], optimize=True)
    O['ab'][v,V,c,A] += scale * -1.00000000 * np.einsum('UV,UW,aAiV->aAiW', gamma1['b'], h['b'][A,A], t['ab'][pv,pV,hc,hA], optimize=True)
    O['ab'][v,V,c,A] += scale * +1.00000000 * np.einsum('UV,AV,aUiW->aAiW', gamma1['b'], h['b'][V,A], t['ab'][pv,pA,hc,hA], optimize=True)
    O['ab'][v,V,c,A] += scale * -1.00000000 * np.einsum('IU,aAiI->aAiU', h['b'][C,A], t['ab'][pv,pV,hc,hC], optimize=True)
    O['ab'][v,V,c,A] += scale * +1.00000000 * np.einsum('AB,aBiU->aAiU', h['b'][V,V], t['ab'][pv,pV,hc,hA], optimize=True)
    O['ab'][a,A,c,C] += scale * +1.00000000 * np.einsum('UV,WV,uUiI->uWiI', eta1['b'], h['b'][A,A], t['ab'][pa,pA,hc,hC], optimize=True)
    O['ab'][a,A,c,C] += scale * -1.00000000 * np.einsum('UV,UI,uWiV->uWiI', eta1['b'], h['b'][A,C], t['ab'][pa,pA,hc,hA], optimize=True)
    O['ab'][a,A,c,C] += scale * +1.00000000 * np.einsum('UV,WV,uUiI->uWiI', gamma1['b'], h['b'][A,A], t['ab'][pa,pA,hc,hC], optimize=True)
    O['ab'][a,A,c,C] += scale * -1.00000000 * np.einsum('UV,UI,uWiV->uWiI', gamma1['b'], h['b'][A,C], t['ab'][pa,pA,hc,hA], optimize=True)
    O['ab'][a,A,c,C] += scale * -1.00000000 * np.einsum('IJ,uUiI->uUiJ', h['b'][C,C], t['ab'][pa,pA,hc,hC], optimize=True)
    O['ab'][a,A,c,C] += scale * +1.00000000 * np.einsum('UA,uAiI->uUiI', h['b'][A,V], t['ab'][pa,pV,hc,hC], optimize=True)
    O['ab'][v,A,c,C] += scale * +1.00000000 * np.einsum('UV,WV,aUiI->aWiI', eta1['b'], h['b'][A,A], t['ab'][pv,pA,hc,hC], optimize=True)
    O['ab'][v,A,c,C] += scale * -1.00000000 * np.einsum('UV,UI,aWiV->aWiI', eta1['b'], h['b'][A,C], t['ab'][pv,pA,hc,hA], optimize=True)
    O['ab'][v,A,c,C] += scale * +1.00000000 * np.einsum('UV,WV,aUiI->aWiI', gamma1['b'], h['b'][A,A], t['ab'][pv,pA,hc,hC], optimize=True)
    O['ab'][v,A,c,C] += scale * -1.00000000 * np.einsum('UV,UI,aWiV->aWiI', gamma1['b'], h['b'][A,C], t['ab'][pv,pA,hc,hA], optimize=True)
    O['ab'][v,A,c,C] += scale * -1.00000000 * np.einsum('IJ,aUiI->aUiJ', h['b'][C,C], t['ab'][pv,pA,hc,hC], optimize=True)
    O['ab'][v,A,c,C] += scale * +1.00000000 * np.einsum('UA,aAiI->aUiI', h['b'][A,V], t['ab'][pv,pV,hc,hC], optimize=True)
    O['ab'][a,C,c,C] += scale * +1.00000000 * np.einsum('UV,IV,uUiJ->uIiJ', eta1['b'], h['b'][C,A], t['ab'][pa,pA,hc,hC], optimize=True)
    O['ab'][a,C,c,C] += scale * +1.00000000 * np.einsum('UV,IV,uUiJ->uIiJ', gamma1['b'], h['b'][C,A], t['ab'][pa,pA,hc,hC], optimize=True)
    O['ab'][a,C,c,C] += scale * +1.00000000 * np.einsum('IA,uAiJ->uIiJ', h['b'][C,V], t['ab'][pa,pV,hc,hC], optimize=True)
    O['ab'][v,C,c,C] += scale * +1.00000000 * np.einsum('UV,IV,aUiJ->aIiJ', eta1['b'], h['b'][C,A], t['ab'][pv,pA,hc,hC], optimize=True)
    O['ab'][v,C,c,C] += scale * +1.00000000 * np.einsum('UV,IV,aUiJ->aIiJ', gamma1['b'], h['b'][C,A], t['ab'][pv,pA,hc,hC], optimize=True)
    O['ab'][v,C,c,C] += scale * +1.00000000 * np.einsum('IA,aAiJ->aIiJ', h['b'][C,V], t['ab'][pv,pV,hc,hC], optimize=True)
    O['ab'][a,V,c,C] += scale * -1.00000000 * np.einsum('UV,UI,uAiV->uAiI', eta1['b'], h['b'][A,C], t['ab'][pa,pV,hc,hA], optimize=True)
    O['ab'][a,V,c,C] += scale * +1.00000000 * np.einsum('UV,AV,uUiI->uAiI', eta1['b'], h['b'][V,A], t['ab'][pa,pA,hc,hC], optimize=True)
    O['ab'][a,V,c,C] += scale * -1.00000000 * np.einsum('UV,UI,uAiV->uAiI', gamma1['b'], h['b'][A,C], t['ab'][pa,pV,hc,hA], optimize=True)
    O['ab'][a,V,c,C] += scale * +1.00000000 * np.einsum('UV,AV,uUiI->uAiI', gamma1['b'], h['b'][V,A], t['ab'][pa,pA,hc,hC], optimize=True)
    O['ab'][a,V,c,C] += scale * -1.00000000 * np.einsum('IJ,uAiI->uAiJ', h['b'][C,C], t['ab'][pa,pV,hc,hC], optimize=True)
    O['ab'][a,V,c,C] += scale * +1.00000000 * np.einsum('AB,uBiI->uAiI', h['b'][V,V], t['ab'][pa,pV,hc,hC], optimize=True)
    O['ab'][v,V,c,C] += scale * -1.00000000 * np.einsum('UV,UI,aAiV->aAiI', eta1['b'], h['b'][A,C], t['ab'][pv,pV,hc,hA], optimize=True)
    O['ab'][v,V,c,C] += scale * +1.00000000 * np.einsum('UV,AV,aUiI->aAiI', eta1['b'], h['b'][V,A], t['ab'][pv,pA,hc,hC], optimize=True)
    O['ab'][v,V,c,C] += scale * -1.00000000 * np.einsum('UV,UI,aAiV->aAiI', gamma1['b'], h['b'][A,C], t['ab'][pv,pV,hc,hA], optimize=True)
    O['ab'][v,V,c,C] += scale * +1.00000000 * np.einsum('UV,AV,aUiI->aAiI', gamma1['b'], h['b'][V,A], t['ab'][pv,pA,hc,hC], optimize=True)
    O['ab'][v,V,c,C] += scale * -1.00000000 * np.einsum('IJ,aAiI->aAiJ', h['b'][C,C], t['ab'][pv,pV,hc,hC], optimize=True)
    O['ab'][v,V,c,C] += scale * +1.00000000 * np.einsum('AB,aBiI->aAiI', h['b'][V,V], t['ab'][pv,pV,hc,hC], optimize=True)
    O['ab'][a,A,c,V] += scale * -1.00000000 * np.einsum('UV,UA,uWiV->uWiA', eta1['b'], h['b'][A,V], t['ab'][pa,pA,hc,hA], optimize=True)
    O['ab'][a,A,c,V] += scale * -1.00000000 * np.einsum('UV,UA,uWiV->uWiA', gamma1['b'], h['b'][A,V], t['ab'][pa,pA,hc,hA], optimize=True)
    O['ab'][a,A,c,V] += scale * -1.00000000 * np.einsum('IA,uUiI->uUiA', h['b'][C,V], t['ab'][pa,pA,hc,hC], optimize=True)
    O['ab'][v,A,c,V] += scale * -1.00000000 * np.einsum('UV,UA,aWiV->aWiA', eta1['b'], h['b'][A,V], t['ab'][pv,pA,hc,hA], optimize=True)
    O['ab'][v,A,c,V] += scale * -1.00000000 * np.einsum('UV,UA,aWiV->aWiA', gamma1['b'], h['b'][A,V], t['ab'][pv,pA,hc,hA], optimize=True)
    O['ab'][v,A,c,V] += scale * -1.00000000 * np.einsum('IA,aUiI->aUiA', h['b'][C,V], t['ab'][pv,pA,hc,hC], optimize=True)
    O['ab'][a,V,c,V] += scale * -1.00000000 * np.einsum('UV,UA,uBiV->uBiA', eta1['b'], h['b'][A,V], t['ab'][pa,pV,hc,hA], optimize=True)
    O['ab'][a,V,c,V] += scale * -1.00000000 * np.einsum('UV,UA,uBiV->uBiA', gamma1['b'], h['b'][A,V], t['ab'][pa,pV,hc,hA], optimize=True)
    O['ab'][a,V,c,V] += scale * -1.00000000 * np.einsum('IA,uBiI->uBiA', h['b'][C,V], t['ab'][pa,pV,hc,hC], optimize=True)
    O['ab'][v,V,c,V] += scale * -1.00000000 * np.einsum('UV,UA,aBiV->aBiA', eta1['b'], h['b'][A,V], t['ab'][pv,pV,hc,hA], optimize=True)
    O['ab'][v,V,c,V] += scale * -1.00000000 * np.einsum('UV,UA,aBiV->aBiA', gamma1['b'], h['b'][A,V], t['ab'][pv,pV,hc,hA], optimize=True)
    O['ab'][v,V,c,V] += scale * -1.00000000 * np.einsum('IA,aBiI->aBiA', h['b'][C,V], t['ab'][pv,pV,hc,hC], optimize=True)

    t1 = time.time()
    if verbose: print("h1b_t2b_c2b took {:.4f} seconds to run.".format(t1-t0))

    return O
def h2a_t2b_c2b(O, h, t, gamma1, eta1, lambdas, orbspace, verbose=False, scale=1.0):
    # 180 lines
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

    
    
    O['ab'][a,A,a,A] += scale * -1.00000000 * np.einsum('uv,iwxv,uUiV->wUxV', eta1['a'], h['aa'][c,a,a,a], t['ab'][pa,pA,hc,hA], optimize=True)
    O['ab'][a,A,a,A] += scale * +1.00000000 * np.einsum('uv,wuxa,aUvV->wUxV', gamma1['a'], h['aa'][a,a,a,v], t['ab'][pv,pA,ha,hA], optimize=True)
    O['ab'][a,A,a,A] += scale * -1.00000000 * np.einsum('iuva,aUiV->uUvV', h['aa'][c,a,a,v], t['ab'][pv,pA,hc,hA], optimize=True)
    
    
    O['ab'][c,A,a,A] += scale * +1.00000000 * np.einsum('uv,ijwv,uUjV->iUwV', eta1['a'], h['aa'][c,c,a,a], t['ab'][pa,pA,hc,hA], optimize=True)
    O['ab'][c,A,a,A] += scale * +1.00000000 * np.einsum('uv,iuwa,aUvV->iUwV', gamma1['a'], h['aa'][c,a,a,v], t['ab'][pv,pA,ha,hA], optimize=True)
    O['ab'][c,A,a,A] += scale * +1.00000000 * np.einsum('ijua,aUjV->iUuV', h['aa'][c,c,a,v], t['ab'][pv,pA,hc,hA], optimize=True)
    
    
    O['ab'][v,A,a,A] += scale * -1.00000000 * np.einsum('uv,iawv,uUiV->aUwV', eta1['a'], h['aa'][c,v,a,a], t['ab'][pa,pA,hc,hA], optimize=True)
    O['ab'][v,A,a,A] += scale * -1.00000000 * np.einsum('uv,uawb,bUvV->aUwV', gamma1['a'], h['aa'][a,v,a,v], t['ab'][pv,pA,ha,hA], optimize=True)
    O['ab'][v,A,a,A] += scale * -1.00000000 * np.einsum('iaub,bUiV->aUuV', h['aa'][c,v,a,v], t['ab'][pv,pA,hc,hA], optimize=True)
    O['ab'][a,V,a,A] += scale * -1.00000000 * np.einsum('uv,wx,yuzx,wAvU->yAzU', eta1['a'], gamma1['a'], h['aa'][a,a,a,a], t['ab'][pa,pV,ha,hA], optimize=True)
    O['ab'][a,V,a,A] += scale * +1.00000000 * np.einsum('uv,wx,ywzv,uAxU->yAzU', eta1['a'], gamma1['a'], h['aa'][a,a,a,a], t['ab'][pa,pV,ha,hA], optimize=True)
    O['ab'][a,V,a,A] += scale * -1.00000000 * np.einsum('uv,iwxv,uAiU->wAxU', eta1['a'], h['aa'][c,a,a,a], t['ab'][pa,pV,hc,hA], optimize=True)
    O['ab'][a,V,a,A] += scale * +1.00000000 * np.einsum('uv,wuxa,aAvU->wAxU', gamma1['a'], h['aa'][a,a,a,v], t['ab'][pv,pV,ha,hA], optimize=True)
    O['ab'][a,V,a,A] += scale * -1.00000000 * np.einsum('iuva,aAiU->uAvU', h['aa'][c,a,a,v], t['ab'][pv,pV,hc,hA], optimize=True)
    O['ab'][c,V,a,A] += scale * -1.00000000 * np.einsum('uv,wx,iuyx,wAvU->iAyU', eta1['a'], gamma1['a'], h['aa'][c,a,a,a], t['ab'][pa,pV,ha,hA], optimize=True)
    O['ab'][c,V,a,A] += scale * +1.00000000 * np.einsum('uv,wx,iwyv,uAxU->iAyU', eta1['a'], gamma1['a'], h['aa'][c,a,a,a], t['ab'][pa,pV,ha,hA], optimize=True)
    O['ab'][c,V,a,A] += scale * +1.00000000 * np.einsum('uv,ijwv,uAjU->iAwU', eta1['a'], h['aa'][c,c,a,a], t['ab'][pa,pV,hc,hA], optimize=True)
    O['ab'][c,V,a,A] += scale * +1.00000000 * np.einsum('uv,iuwa,aAvU->iAwU', gamma1['a'], h['aa'][c,a,a,v], t['ab'][pv,pV,ha,hA], optimize=True)
    O['ab'][c,V,a,A] += scale * +1.00000000 * np.einsum('ijua,aAjU->iAuU', h['aa'][c,c,a,v], t['ab'][pv,pV,hc,hA], optimize=True)
    O['ab'][v,V,a,A] += scale * +1.00000000 * np.einsum('uv,wx,uayx,wAvU->aAyU', eta1['a'], gamma1['a'], h['aa'][a,v,a,a], t['ab'][pa,pV,ha,hA], optimize=True)
    O['ab'][v,V,a,A] += scale * -1.00000000 * np.einsum('uv,wx,wayv,uAxU->aAyU', eta1['a'], gamma1['a'], h['aa'][a,v,a,a], t['ab'][pa,pV,ha,hA], optimize=True)
    O['ab'][v,V,a,A] += scale * -1.00000000 * np.einsum('uv,iawv,uAiU->aAwU', eta1['a'], h['aa'][c,v,a,a], t['ab'][pa,pV,hc,hA], optimize=True)
    O['ab'][v,V,a,A] += scale * -1.00000000 * np.einsum('uv,uawb,bAvU->aAwU', gamma1['a'], h['aa'][a,v,a,v], t['ab'][pv,pV,ha,hA], optimize=True)
    O['ab'][v,V,a,A] += scale * -1.00000000 * np.einsum('iaub,bAiU->aAuU', h['aa'][c,v,a,v], t['ab'][pv,pV,hc,hA], optimize=True)
    O['ab'][a,A,a,C] += scale * -1.00000000 * np.einsum('uv,wx,yuzx,wUvI->yUzI', eta1['a'], gamma1['a'], h['aa'][a,a,a,a], t['ab'][pa,pA,ha,hC], optimize=True)
    O['ab'][a,A,a,C] += scale * +1.00000000 * np.einsum('uv,wx,ywzv,uUxI->yUzI', eta1['a'], gamma1['a'], h['aa'][a,a,a,a], t['ab'][pa,pA,ha,hC], optimize=True)
    O['ab'][a,A,a,C] += scale * -1.00000000 * np.einsum('uv,iwxv,uUiI->wUxI', eta1['a'], h['aa'][c,a,a,a], t['ab'][pa,pA,hc,hC], optimize=True)
    O['ab'][a,A,a,C] += scale * +1.00000000 * np.einsum('uv,wuxa,aUvI->wUxI', gamma1['a'], h['aa'][a,a,a,v], t['ab'][pv,pA,ha,hC], optimize=True)
    O['ab'][a,A,a,C] += scale * -1.00000000 * np.einsum('iuva,aUiI->uUvI', h['aa'][c,a,a,v], t['ab'][pv,pA,hc,hC], optimize=True)
    O['ab'][c,A,a,C] += scale * -1.00000000 * np.einsum('uv,wx,iuyx,wUvI->iUyI', eta1['a'], gamma1['a'], h['aa'][c,a,a,a], t['ab'][pa,pA,ha,hC], optimize=True)
    O['ab'][c,A,a,C] += scale * +1.00000000 * np.einsum('uv,wx,iwyv,uUxI->iUyI', eta1['a'], gamma1['a'], h['aa'][c,a,a,a], t['ab'][pa,pA,ha,hC], optimize=True)
    O['ab'][c,A,a,C] += scale * +1.00000000 * np.einsum('uv,ijwv,uUjI->iUwI', eta1['a'], h['aa'][c,c,a,a], t['ab'][pa,pA,hc,hC], optimize=True)
    O['ab'][c,A,a,C] += scale * +1.00000000 * np.einsum('uv,iuwa,aUvI->iUwI', gamma1['a'], h['aa'][c,a,a,v], t['ab'][pv,pA,ha,hC], optimize=True)
    O['ab'][c,A,a,C] += scale * +1.00000000 * np.einsum('ijua,aUjI->iUuI', h['aa'][c,c,a,v], t['ab'][pv,pA,hc,hC], optimize=True)
    O['ab'][v,A,a,C] += scale * +1.00000000 * np.einsum('uv,wx,uayx,wUvI->aUyI', eta1['a'], gamma1['a'], h['aa'][a,v,a,a], t['ab'][pa,pA,ha,hC], optimize=True)
    O['ab'][v,A,a,C] += scale * -1.00000000 * np.einsum('uv,wx,wayv,uUxI->aUyI', eta1['a'], gamma1['a'], h['aa'][a,v,a,a], t['ab'][pa,pA,ha,hC], optimize=True)
    O['ab'][v,A,a,C] += scale * -1.00000000 * np.einsum('uv,iawv,uUiI->aUwI', eta1['a'], h['aa'][c,v,a,a], t['ab'][pa,pA,hc,hC], optimize=True)
    O['ab'][v,A,a,C] += scale * -1.00000000 * np.einsum('uv,uawb,bUvI->aUwI', gamma1['a'], h['aa'][a,v,a,v], t['ab'][pv,pA,ha,hC], optimize=True)
    O['ab'][v,A,a,C] += scale * -1.00000000 * np.einsum('iaub,bUiI->aUuI', h['aa'][c,v,a,v], t['ab'][pv,pA,hc,hC], optimize=True)
    O['ab'][a,V,a,C] += scale * -1.00000000 * np.einsum('uv,wx,yuzx,wAvI->yAzI', eta1['a'], gamma1['a'], h['aa'][a,a,a,a], t['ab'][pa,pV,ha,hC], optimize=True)
    O['ab'][a,V,a,C] += scale * +1.00000000 * np.einsum('uv,wx,ywzv,uAxI->yAzI', eta1['a'], gamma1['a'], h['aa'][a,a,a,a], t['ab'][pa,pV,ha,hC], optimize=True)
    O['ab'][a,V,a,C] += scale * -1.00000000 * np.einsum('uv,iwxv,uAiI->wAxI', eta1['a'], h['aa'][c,a,a,a], t['ab'][pa,pV,hc,hC], optimize=True)
    O['ab'][a,V,a,C] += scale * +1.00000000 * np.einsum('uv,wuxa,aAvI->wAxI', gamma1['a'], h['aa'][a,a,a,v], t['ab'][pv,pV,ha,hC], optimize=True)
    O['ab'][a,V,a,C] += scale * -1.00000000 * np.einsum('iuva,aAiI->uAvI', h['aa'][c,a,a,v], t['ab'][pv,pV,hc,hC], optimize=True)
    O['ab'][c,V,a,C] += scale * -1.00000000 * np.einsum('uv,wx,iuyx,wAvI->iAyI', eta1['a'], gamma1['a'], h['aa'][c,a,a,a], t['ab'][pa,pV,ha,hC], optimize=True)
    O['ab'][c,V,a,C] += scale * +1.00000000 * np.einsum('uv,wx,iwyv,uAxI->iAyI', eta1['a'], gamma1['a'], h['aa'][c,a,a,a], t['ab'][pa,pV,ha,hC], optimize=True)
    O['ab'][c,V,a,C] += scale * +1.00000000 * np.einsum('uv,ijwv,uAjI->iAwI', eta1['a'], h['aa'][c,c,a,a], t['ab'][pa,pV,hc,hC], optimize=True)
    O['ab'][c,V,a,C] += scale * +1.00000000 * np.einsum('uv,iuwa,aAvI->iAwI', gamma1['a'], h['aa'][c,a,a,v], t['ab'][pv,pV,ha,hC], optimize=True)
    O['ab'][c,V,a,C] += scale * +1.00000000 * np.einsum('ijua,aAjI->iAuI', h['aa'][c,c,a,v], t['ab'][pv,pV,hc,hC], optimize=True)
    O['ab'][v,V,a,C] += scale * +1.00000000 * np.einsum('uv,wx,uayx,wAvI->aAyI', eta1['a'], gamma1['a'], h['aa'][a,v,a,a], t['ab'][pa,pV,ha,hC], optimize=True)
    O['ab'][v,V,a,C] += scale * -1.00000000 * np.einsum('uv,wx,wayv,uAxI->aAyI', eta1['a'], gamma1['a'], h['aa'][a,v,a,a], t['ab'][pa,pV,ha,hC], optimize=True)
    O['ab'][v,V,a,C] += scale * -1.00000000 * np.einsum('uv,iawv,uAiI->aAwI', eta1['a'], h['aa'][c,v,a,a], t['ab'][pa,pV,hc,hC], optimize=True)
    O['ab'][v,V,a,C] += scale * -1.00000000 * np.einsum('uv,uawb,bAvI->aAwI', gamma1['a'], h['aa'][a,v,a,v], t['ab'][pv,pV,ha,hC], optimize=True)
    O['ab'][v,V,a,C] += scale * -1.00000000 * np.einsum('iaub,bAiI->aAuI', h['aa'][c,v,a,v], t['ab'][pv,pV,hc,hC], optimize=True)
    
    
    O['ab'][a,A,c,A] += scale * -1.00000000 * np.einsum('uv,iwjv,uUiV->wUjV', eta1['a'], h['aa'][c,a,c,a], t['ab'][pa,pA,hc,hA], optimize=True)
    O['ab'][a,A,c,A] += scale * +1.00000000 * np.einsum('uv,wuia,aUvV->wUiV', gamma1['a'], h['aa'][a,a,c,v], t['ab'][pv,pA,ha,hA], optimize=True)
    O['ab'][a,A,c,A] += scale * -1.00000000 * np.einsum('iuja,aUiV->uUjV', h['aa'][c,a,c,v], t['ab'][pv,pA,hc,hA], optimize=True)
    
    
    O['ab'][c,A,c,A] += scale * +1.00000000 * np.einsum('uv,ijkv,uUjV->iUkV', eta1['a'], h['aa'][c,c,c,a], t['ab'][pa,pA,hc,hA], optimize=True)
    O['ab'][c,A,c,A] += scale * +1.00000000 * np.einsum('uv,iuja,aUvV->iUjV', gamma1['a'], h['aa'][c,a,c,v], t['ab'][pv,pA,ha,hA], optimize=True)
    O['ab'][c,A,c,A] += scale * +1.00000000 * np.einsum('ijka,aUjV->iUkV', h['aa'][c,c,c,v], t['ab'][pv,pA,hc,hA], optimize=True)
    
    
    O['ab'][v,A,c,A] += scale * -1.00000000 * np.einsum('uv,iajv,uUiV->aUjV', eta1['a'], h['aa'][c,v,c,a], t['ab'][pa,pA,hc,hA], optimize=True)
    O['ab'][v,A,c,A] += scale * -1.00000000 * np.einsum('uv,uaib,bUvV->aUiV', gamma1['a'], h['aa'][a,v,c,v], t['ab'][pv,pA,ha,hA], optimize=True)
    O['ab'][v,A,c,A] += scale * -1.00000000 * np.einsum('iajb,bUiV->aUjV', h['aa'][c,v,c,v], t['ab'][pv,pA,hc,hA], optimize=True)
    O['ab'][a,V,c,A] += scale * -1.00000000 * np.einsum('uv,wx,yuix,wAvU->yAiU', eta1['a'], gamma1['a'], h['aa'][a,a,c,a], t['ab'][pa,pV,ha,hA], optimize=True)
    O['ab'][a,V,c,A] += scale * +1.00000000 * np.einsum('uv,wx,ywiv,uAxU->yAiU', eta1['a'], gamma1['a'], h['aa'][a,a,c,a], t['ab'][pa,pV,ha,hA], optimize=True)
    O['ab'][a,V,c,A] += scale * -1.00000000 * np.einsum('uv,iwjv,uAiU->wAjU', eta1['a'], h['aa'][c,a,c,a], t['ab'][pa,pV,hc,hA], optimize=True)
    O['ab'][a,V,c,A] += scale * +1.00000000 * np.einsum('uv,wuia,aAvU->wAiU', gamma1['a'], h['aa'][a,a,c,v], t['ab'][pv,pV,ha,hA], optimize=True)
    O['ab'][a,V,c,A] += scale * -1.00000000 * np.einsum('iuja,aAiU->uAjU', h['aa'][c,a,c,v], t['ab'][pv,pV,hc,hA], optimize=True)
    O['ab'][c,V,c,A] += scale * -1.00000000 * np.einsum('uv,wx,iujx,wAvU->iAjU', eta1['a'], gamma1['a'], h['aa'][c,a,c,a], t['ab'][pa,pV,ha,hA], optimize=True)
    O['ab'][c,V,c,A] += scale * +1.00000000 * np.einsum('uv,wx,iwjv,uAxU->iAjU', eta1['a'], gamma1['a'], h['aa'][c,a,c,a], t['ab'][pa,pV,ha,hA], optimize=True)
    O['ab'][c,V,c,A] += scale * +1.00000000 * np.einsum('uv,ijkv,uAjU->iAkU', eta1['a'], h['aa'][c,c,c,a], t['ab'][pa,pV,hc,hA], optimize=True)
    O['ab'][c,V,c,A] += scale * +1.00000000 * np.einsum('uv,iuja,aAvU->iAjU', gamma1['a'], h['aa'][c,a,c,v], t['ab'][pv,pV,ha,hA], optimize=True)
    O['ab'][c,V,c,A] += scale * +1.00000000 * np.einsum('ijka,aAjU->iAkU', h['aa'][c,c,c,v], t['ab'][pv,pV,hc,hA], optimize=True)
    O['ab'][v,V,c,A] += scale * +1.00000000 * np.einsum('uv,wx,uaix,wAvU->aAiU', eta1['a'], gamma1['a'], h['aa'][a,v,c,a], t['ab'][pa,pV,ha,hA], optimize=True)
    O['ab'][v,V,c,A] += scale * -1.00000000 * np.einsum('uv,wx,waiv,uAxU->aAiU', eta1['a'], gamma1['a'], h['aa'][a,v,c,a], t['ab'][pa,pV,ha,hA], optimize=True)
    O['ab'][v,V,c,A] += scale * -1.00000000 * np.einsum('uv,iajv,uAiU->aAjU', eta1['a'], h['aa'][c,v,c,a], t['ab'][pa,pV,hc,hA], optimize=True)
    O['ab'][v,V,c,A] += scale * -1.00000000 * np.einsum('uv,uaib,bAvU->aAiU', gamma1['a'], h['aa'][a,v,c,v], t['ab'][pv,pV,ha,hA], optimize=True)
    O['ab'][v,V,c,A] += scale * -1.00000000 * np.einsum('iajb,bAiU->aAjU', h['aa'][c,v,c,v], t['ab'][pv,pV,hc,hA], optimize=True)
    O['ab'][a,A,c,C] += scale * -1.00000000 * np.einsum('uv,wx,yuix,wUvI->yUiI', eta1['a'], gamma1['a'], h['aa'][a,a,c,a], t['ab'][pa,pA,ha,hC], optimize=True)
    O['ab'][a,A,c,C] += scale * +1.00000000 * np.einsum('uv,wx,ywiv,uUxI->yUiI', eta1['a'], gamma1['a'], h['aa'][a,a,c,a], t['ab'][pa,pA,ha,hC], optimize=True)
    O['ab'][a,A,c,C] += scale * -1.00000000 * np.einsum('uv,iwjv,uUiI->wUjI', eta1['a'], h['aa'][c,a,c,a], t['ab'][pa,pA,hc,hC], optimize=True)
    O['ab'][a,A,c,C] += scale * +1.00000000 * np.einsum('uv,wuia,aUvI->wUiI', gamma1['a'], h['aa'][a,a,c,v], t['ab'][pv,pA,ha,hC], optimize=True)
    O['ab'][a,A,c,C] += scale * -1.00000000 * np.einsum('iuja,aUiI->uUjI', h['aa'][c,a,c,v], t['ab'][pv,pA,hc,hC], optimize=True)
    O['ab'][c,A,c,C] += scale * -1.00000000 * np.einsum('uv,wx,iujx,wUvI->iUjI', eta1['a'], gamma1['a'], h['aa'][c,a,c,a], t['ab'][pa,pA,ha,hC], optimize=True)
    O['ab'][c,A,c,C] += scale * +1.00000000 * np.einsum('uv,wx,iwjv,uUxI->iUjI', eta1['a'], gamma1['a'], h['aa'][c,a,c,a], t['ab'][pa,pA,ha,hC], optimize=True)
    O['ab'][c,A,c,C] += scale * +1.00000000 * np.einsum('uv,ijkv,uUjI->iUkI', eta1['a'], h['aa'][c,c,c,a], t['ab'][pa,pA,hc,hC], optimize=True)
    O['ab'][c,A,c,C] += scale * +1.00000000 * np.einsum('uv,iuja,aUvI->iUjI', gamma1['a'], h['aa'][c,a,c,v], t['ab'][pv,pA,ha,hC], optimize=True)
    O['ab'][c,A,c,C] += scale * +1.00000000 * np.einsum('ijka,aUjI->iUkI', h['aa'][c,c,c,v], t['ab'][pv,pA,hc,hC], optimize=True)
    O['ab'][v,A,c,C] += scale * +1.00000000 * np.einsum('uv,wx,uaix,wUvI->aUiI', eta1['a'], gamma1['a'], h['aa'][a,v,c,a], t['ab'][pa,pA,ha,hC], optimize=True)
    O['ab'][v,A,c,C] += scale * -1.00000000 * np.einsum('uv,wx,waiv,uUxI->aUiI', eta1['a'], gamma1['a'], h['aa'][a,v,c,a], t['ab'][pa,pA,ha,hC], optimize=True)
    O['ab'][v,A,c,C] += scale * -1.00000000 * np.einsum('uv,iajv,uUiI->aUjI', eta1['a'], h['aa'][c,v,c,a], t['ab'][pa,pA,hc,hC], optimize=True)
    O['ab'][v,A,c,C] += scale * -1.00000000 * np.einsum('uv,uaib,bUvI->aUiI', gamma1['a'], h['aa'][a,v,c,v], t['ab'][pv,pA,ha,hC], optimize=True)
    O['ab'][v,A,c,C] += scale * -1.00000000 * np.einsum('iajb,bUiI->aUjI', h['aa'][c,v,c,v], t['ab'][pv,pA,hc,hC], optimize=True)
    O['ab'][a,V,c,C] += scale * -1.00000000 * np.einsum('uv,wx,yuix,wAvI->yAiI', eta1['a'], gamma1['a'], h['aa'][a,a,c,a], t['ab'][pa,pV,ha,hC], optimize=True)
    O['ab'][a,V,c,C] += scale * +1.00000000 * np.einsum('uv,wx,ywiv,uAxI->yAiI', eta1['a'], gamma1['a'], h['aa'][a,a,c,a], t['ab'][pa,pV,ha,hC], optimize=True)
    O['ab'][a,V,c,C] += scale * -1.00000000 * np.einsum('uv,iwjv,uAiI->wAjI', eta1['a'], h['aa'][c,a,c,a], t['ab'][pa,pV,hc,hC], optimize=True)
    O['ab'][a,V,c,C] += scale * +1.00000000 * np.einsum('uv,wuia,aAvI->wAiI', gamma1['a'], h['aa'][a,a,c,v], t['ab'][pv,pV,ha,hC], optimize=True)
    O['ab'][a,V,c,C] += scale * -1.00000000 * np.einsum('iuja,aAiI->uAjI', h['aa'][c,a,c,v], t['ab'][pv,pV,hc,hC], optimize=True)
    O['ab'][c,V,c,C] += scale * -1.00000000 * np.einsum('uv,wx,iujx,wAvI->iAjI', eta1['a'], gamma1['a'], h['aa'][c,a,c,a], t['ab'][pa,pV,ha,hC], optimize=True)
    O['ab'][c,V,c,C] += scale * +1.00000000 * np.einsum('uv,wx,iwjv,uAxI->iAjI', eta1['a'], gamma1['a'], h['aa'][c,a,c,a], t['ab'][pa,pV,ha,hC], optimize=True)
    O['ab'][c,V,c,C] += scale * +1.00000000 * np.einsum('uv,ijkv,uAjI->iAkI', eta1['a'], h['aa'][c,c,c,a], t['ab'][pa,pV,hc,hC], optimize=True)
    O['ab'][c,V,c,C] += scale * +1.00000000 * np.einsum('uv,iuja,aAvI->iAjI', gamma1['a'], h['aa'][c,a,c,v], t['ab'][pv,pV,ha,hC], optimize=True)
    O['ab'][c,V,c,C] += scale * +1.00000000 * np.einsum('ijka,aAjI->iAkI', h['aa'][c,c,c,v], t['ab'][pv,pV,hc,hC], optimize=True)
    O['ab'][v,V,c,C] += scale * +1.00000000 * np.einsum('uv,wx,uaix,wAvI->aAiI', eta1['a'], gamma1['a'], h['aa'][a,v,c,a], t['ab'][pa,pV,ha,hC], optimize=True)
    O['ab'][v,V,c,C] += scale * -1.00000000 * np.einsum('uv,wx,waiv,uAxI->aAiI', eta1['a'], gamma1['a'], h['aa'][a,v,c,a], t['ab'][pa,pV,ha,hC], optimize=True)
    O['ab'][v,V,c,C] += scale * -1.00000000 * np.einsum('uv,iajv,uAiI->aAjI', eta1['a'], h['aa'][c,v,c,a], t['ab'][pa,pV,hc,hC], optimize=True)
    O['ab'][v,V,c,C] += scale * -1.00000000 * np.einsum('uv,uaib,bAvI->aAiI', gamma1['a'], h['aa'][a,v,c,v], t['ab'][pv,pV,ha,hC], optimize=True)
    O['ab'][v,V,c,C] += scale * -1.00000000 * np.einsum('iajb,bAiI->aAjI', h['aa'][c,v,c,v], t['ab'][pv,pV,hc,hC], optimize=True)
    
    
    O['ab'][a,A,v,A] += scale * +1.00000000 * np.einsum('uv,iwva,uUiV->wUaV', eta1['a'], h['aa'][c,a,a,v], t['ab'][pa,pA,hc,hA], optimize=True)
    O['ab'][a,A,v,A] += scale * +1.00000000 * np.einsum('uv,wuab,bUvV->wUaV', gamma1['a'], h['aa'][a,a,v,v], t['ab'][pv,pA,ha,hA], optimize=True)
    O['ab'][a,A,v,A] += scale * -1.00000000 * np.einsum('iuab,bUiV->uUaV', h['aa'][c,a,v,v], t['ab'][pv,pA,hc,hA], optimize=True)
    
    
    O['ab'][c,A,v,A] += scale * -1.00000000 * np.einsum('uv,ijva,uUjV->iUaV', eta1['a'], h['aa'][c,c,a,v], t['ab'][pa,pA,hc,hA], optimize=True)
    O['ab'][c,A,v,A] += scale * +1.00000000 * np.einsum('uv,iuab,bUvV->iUaV', gamma1['a'], h['aa'][c,a,v,v], t['ab'][pv,pA,ha,hA], optimize=True)
    O['ab'][c,A,v,A] += scale * +1.00000000 * np.einsum('ijab,bUjV->iUaV', h['aa'][c,c,v,v], t['ab'][pv,pA,hc,hA], optimize=True)
    
    
    O['ab'][v,A,v,A] += scale * +1.00000000 * np.einsum('uv,iavb,uUiV->aUbV', eta1['a'], h['aa'][c,v,a,v], t['ab'][pa,pA,hc,hA], optimize=True)
    O['ab'][v,A,v,A] += scale * -1.00000000 * np.einsum('uv,uabc,cUvV->aUbV', gamma1['a'], h['aa'][a,v,v,v], t['ab'][pv,pA,ha,hA], optimize=True)
    O['ab'][v,A,v,A] += scale * -1.00000000 * np.einsum('iabc,cUiV->aUbV', h['aa'][c,v,v,v], t['ab'][pv,pA,hc,hA], optimize=True)
    O['ab'][a,V,v,A] += scale * +1.00000000 * np.einsum('uv,wx,yuxa,wAvU->yAaU', eta1['a'], gamma1['a'], h['aa'][a,a,a,v], t['ab'][pa,pV,ha,hA], optimize=True)
    O['ab'][a,V,v,A] += scale * -1.00000000 * np.einsum('uv,wx,ywva,uAxU->yAaU', eta1['a'], gamma1['a'], h['aa'][a,a,a,v], t['ab'][pa,pV,ha,hA], optimize=True)
    O['ab'][a,V,v,A] += scale * +1.00000000 * np.einsum('uv,iwva,uAiU->wAaU', eta1['a'], h['aa'][c,a,a,v], t['ab'][pa,pV,hc,hA], optimize=True)
    O['ab'][a,V,v,A] += scale * +1.00000000 * np.einsum('uv,wuab,bAvU->wAaU', gamma1['a'], h['aa'][a,a,v,v], t['ab'][pv,pV,ha,hA], optimize=True)
    O['ab'][a,V,v,A] += scale * -1.00000000 * np.einsum('iuab,bAiU->uAaU', h['aa'][c,a,v,v], t['ab'][pv,pV,hc,hA], optimize=True)
    O['ab'][c,V,v,A] += scale * +1.00000000 * np.einsum('uv,wx,iuxa,wAvU->iAaU', eta1['a'], gamma1['a'], h['aa'][c,a,a,v], t['ab'][pa,pV,ha,hA], optimize=True)
    O['ab'][c,V,v,A] += scale * -1.00000000 * np.einsum('uv,wx,iwva,uAxU->iAaU', eta1['a'], gamma1['a'], h['aa'][c,a,a,v], t['ab'][pa,pV,ha,hA], optimize=True)
    O['ab'][c,V,v,A] += scale * -1.00000000 * np.einsum('uv,ijva,uAjU->iAaU', eta1['a'], h['aa'][c,c,a,v], t['ab'][pa,pV,hc,hA], optimize=True)
    O['ab'][c,V,v,A] += scale * +1.00000000 * np.einsum('uv,iuab,bAvU->iAaU', gamma1['a'], h['aa'][c,a,v,v], t['ab'][pv,pV,ha,hA], optimize=True)
    O['ab'][c,V,v,A] += scale * +1.00000000 * np.einsum('ijab,bAjU->iAaU', h['aa'][c,c,v,v], t['ab'][pv,pV,hc,hA], optimize=True)
    O['ab'][v,V,v,A] += scale * -1.00000000 * np.einsum('uv,wx,uaxb,wAvU->aAbU', eta1['a'], gamma1['a'], h['aa'][a,v,a,v], t['ab'][pa,pV,ha,hA], optimize=True)
    O['ab'][v,V,v,A] += scale * +1.00000000 * np.einsum('uv,wx,wavb,uAxU->aAbU', eta1['a'], gamma1['a'], h['aa'][a,v,a,v], t['ab'][pa,pV,ha,hA], optimize=True)
    O['ab'][v,V,v,A] += scale * +1.00000000 * np.einsum('uv,iavb,uAiU->aAbU', eta1['a'], h['aa'][c,v,a,v], t['ab'][pa,pV,hc,hA], optimize=True)
    O['ab'][v,V,v,A] += scale * -1.00000000 * np.einsum('uv,uabc,cAvU->aAbU', gamma1['a'], h['aa'][a,v,v,v], t['ab'][pv,pV,ha,hA], optimize=True)
    O['ab'][v,V,v,A] += scale * -1.00000000 * np.einsum('iabc,cAiU->aAbU', h['aa'][c,v,v,v], t['ab'][pv,pV,hc,hA], optimize=True)
    O['ab'][a,A,v,C] += scale * +1.00000000 * np.einsum('uv,wx,yuxa,wUvI->yUaI', eta1['a'], gamma1['a'], h['aa'][a,a,a,v], t['ab'][pa,pA,ha,hC], optimize=True)
    O['ab'][a,A,v,C] += scale * -1.00000000 * np.einsum('uv,wx,ywva,uUxI->yUaI', eta1['a'], gamma1['a'], h['aa'][a,a,a,v], t['ab'][pa,pA,ha,hC], optimize=True)
    O['ab'][a,A,v,C] += scale * +1.00000000 * np.einsum('uv,iwva,uUiI->wUaI', eta1['a'], h['aa'][c,a,a,v], t['ab'][pa,pA,hc,hC], optimize=True)
    O['ab'][a,A,v,C] += scale * +1.00000000 * np.einsum('uv,wuab,bUvI->wUaI', gamma1['a'], h['aa'][a,a,v,v], t['ab'][pv,pA,ha,hC], optimize=True)
    O['ab'][a,A,v,C] += scale * -1.00000000 * np.einsum('iuab,bUiI->uUaI', h['aa'][c,a,v,v], t['ab'][pv,pA,hc,hC], optimize=True)
    O['ab'][c,A,v,C] += scale * +1.00000000 * np.einsum('uv,wx,iuxa,wUvI->iUaI', eta1['a'], gamma1['a'], h['aa'][c,a,a,v], t['ab'][pa,pA,ha,hC], optimize=True)
    O['ab'][c,A,v,C] += scale * -1.00000000 * np.einsum('uv,wx,iwva,uUxI->iUaI', eta1['a'], gamma1['a'], h['aa'][c,a,a,v], t['ab'][pa,pA,ha,hC], optimize=True)
    O['ab'][c,A,v,C] += scale * -1.00000000 * np.einsum('uv,ijva,uUjI->iUaI', eta1['a'], h['aa'][c,c,a,v], t['ab'][pa,pA,hc,hC], optimize=True)
    O['ab'][c,A,v,C] += scale * +1.00000000 * np.einsum('uv,iuab,bUvI->iUaI', gamma1['a'], h['aa'][c,a,v,v], t['ab'][pv,pA,ha,hC], optimize=True)
    O['ab'][c,A,v,C] += scale * +1.00000000 * np.einsum('ijab,bUjI->iUaI', h['aa'][c,c,v,v], t['ab'][pv,pA,hc,hC], optimize=True)
    O['ab'][v,A,v,C] += scale * -1.00000000 * np.einsum('uv,wx,uaxb,wUvI->aUbI', eta1['a'], gamma1['a'], h['aa'][a,v,a,v], t['ab'][pa,pA,ha,hC], optimize=True)
    O['ab'][v,A,v,C] += scale * +1.00000000 * np.einsum('uv,wx,wavb,uUxI->aUbI', eta1['a'], gamma1['a'], h['aa'][a,v,a,v], t['ab'][pa,pA,ha,hC], optimize=True)
    O['ab'][v,A,v,C] += scale * +1.00000000 * np.einsum('uv,iavb,uUiI->aUbI', eta1['a'], h['aa'][c,v,a,v], t['ab'][pa,pA,hc,hC], optimize=True)
    O['ab'][v,A,v,C] += scale * -1.00000000 * np.einsum('uv,uabc,cUvI->aUbI', gamma1['a'], h['aa'][a,v,v,v], t['ab'][pv,pA,ha,hC], optimize=True)
    O['ab'][v,A,v,C] += scale * -1.00000000 * np.einsum('iabc,cUiI->aUbI', h['aa'][c,v,v,v], t['ab'][pv,pA,hc,hC], optimize=True)
    O['ab'][a,V,v,C] += scale * +1.00000000 * np.einsum('uv,wx,yuxa,wAvI->yAaI', eta1['a'], gamma1['a'], h['aa'][a,a,a,v], t['ab'][pa,pV,ha,hC], optimize=True)
    O['ab'][a,V,v,C] += scale * -1.00000000 * np.einsum('uv,wx,ywva,uAxI->yAaI', eta1['a'], gamma1['a'], h['aa'][a,a,a,v], t['ab'][pa,pV,ha,hC], optimize=True)
    O['ab'][a,V,v,C] += scale * +1.00000000 * np.einsum('uv,iwva,uAiI->wAaI', eta1['a'], h['aa'][c,a,a,v], t['ab'][pa,pV,hc,hC], optimize=True)
    O['ab'][a,V,v,C] += scale * +1.00000000 * np.einsum('uv,wuab,bAvI->wAaI', gamma1['a'], h['aa'][a,a,v,v], t['ab'][pv,pV,ha,hC], optimize=True)
    O['ab'][a,V,v,C] += scale * -1.00000000 * np.einsum('iuab,bAiI->uAaI', h['aa'][c,a,v,v], t['ab'][pv,pV,hc,hC], optimize=True)
    O['ab'][c,V,v,C] += scale * +1.00000000 * np.einsum('uv,wx,iuxa,wAvI->iAaI', eta1['a'], gamma1['a'], h['aa'][c,a,a,v], t['ab'][pa,pV,ha,hC], optimize=True)
    O['ab'][c,V,v,C] += scale * -1.00000000 * np.einsum('uv,wx,iwva,uAxI->iAaI', eta1['a'], gamma1['a'], h['aa'][c,a,a,v], t['ab'][pa,pV,ha,hC], optimize=True)
    O['ab'][c,V,v,C] += scale * -1.00000000 * np.einsum('uv,ijva,uAjI->iAaI', eta1['a'], h['aa'][c,c,a,v], t['ab'][pa,pV,hc,hC], optimize=True)
    O['ab'][c,V,v,C] += scale * +1.00000000 * np.einsum('uv,iuab,bAvI->iAaI', gamma1['a'], h['aa'][c,a,v,v], t['ab'][pv,pV,ha,hC], optimize=True)
    O['ab'][c,V,v,C] += scale * +1.00000000 * np.einsum('ijab,bAjI->iAaI', h['aa'][c,c,v,v], t['ab'][pv,pV,hc,hC], optimize=True)
    O['ab'][v,V,v,C] += scale * -1.00000000 * np.einsum('uv,wx,uaxb,wAvI->aAbI', eta1['a'], gamma1['a'], h['aa'][a,v,a,v], t['ab'][pa,pV,ha,hC], optimize=True)
    O['ab'][v,V,v,C] += scale * +1.00000000 * np.einsum('uv,wx,wavb,uAxI->aAbI', eta1['a'], gamma1['a'], h['aa'][a,v,a,v], t['ab'][pa,pV,ha,hC], optimize=True)
    O['ab'][v,V,v,C] += scale * +1.00000000 * np.einsum('uv,iavb,uAiI->aAbI', eta1['a'], h['aa'][c,v,a,v], t['ab'][pa,pV,hc,hC], optimize=True)
    O['ab'][v,V,v,C] += scale * -1.00000000 * np.einsum('uv,uabc,cAvI->aAbI', gamma1['a'], h['aa'][a,v,v,v], t['ab'][pv,pV,ha,hC], optimize=True)
    O['ab'][v,V,v,C] += scale * -1.00000000 * np.einsum('iabc,cAiI->aAbI', h['aa'][c,v,v,v], t['ab'][pv,pV,hc,hC], optimize=True)

    t1 = time.time()
    if verbose: print("h2a_t2b_c2b took {:.4f} seconds to run.".format(t1-t0))

    return O
def h2b_t1a_c2b(O, h, t, gamma1, eta1, lambdas, orbspace, verbose=False, scale=1.0):
    # 324 lines
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

    
    
    
    
    O['ab'][a,A,a,A] += scale * -1.00000000 * np.einsum('iUuV,vi->vUuV', h['ab'][c,A,a,A], t['a'][pa,hc], optimize=True)
    O['ab'][a,A,a,A] += scale * +1.00000000 * np.einsum('uUaV,av->uUvV', h['ab'][a,A,v,A], t['a'][pv,ha], optimize=True)
    
    
    O['ab'][c,A,a,A] += scale * +1.00000000 * np.einsum('iUaV,au->iUuV', h['ab'][c,A,v,A], t['a'][pv,ha], optimize=True)
    O['ab'][v,A,a,A] += scale * -1.00000000 * np.einsum('uv,uUwV,av->aUwV', eta1['a'], h['ab'][a,A,a,A], t['a'][pv,ha], optimize=True)
    
    O['ab'][v,A,a,A] += scale * -1.00000000 * np.einsum('uv,uUwV,av->aUwV', gamma1['a'], h['ab'][a,A,a,A], t['a'][pv,ha], optimize=True)
    
    O['ab'][v,A,a,A] += scale * -1.00000000 * np.einsum('iUuV,ai->aUuV', h['ab'][c,A,a,A], t['a'][pv,hc], optimize=True)
    O['ab'][v,A,a,A] += scale * +1.00000000 * np.einsum('aUbV,bu->aUuV', h['ab'][v,A,v,A], t['a'][pv,ha], optimize=True)
    
    
    
    
    O['ab'][a,C,a,A] += scale * -1.00000000 * np.einsum('iIuU,vi->vIuU', h['ab'][c,C,a,A], t['a'][pa,hc], optimize=True)
    O['ab'][a,C,a,A] += scale * +1.00000000 * np.einsum('uIaU,av->uIvU', h['ab'][a,C,v,A], t['a'][pv,ha], optimize=True)
    
    
    O['ab'][c,C,a,A] += scale * +1.00000000 * np.einsum('iIaU,au->iIuU', h['ab'][c,C,v,A], t['a'][pv,ha], optimize=True)
    O['ab'][v,C,a,A] += scale * -1.00000000 * np.einsum('uv,uIwU,av->aIwU', eta1['a'], h['ab'][a,C,a,A], t['a'][pv,ha], optimize=True)
    
    O['ab'][v,C,a,A] += scale * -1.00000000 * np.einsum('uv,uIwU,av->aIwU', gamma1['a'], h['ab'][a,C,a,A], t['a'][pv,ha], optimize=True)
    
    O['ab'][v,C,a,A] += scale * -1.00000000 * np.einsum('iIuU,ai->aIuU', h['ab'][c,C,a,A], t['a'][pv,hc], optimize=True)
    O['ab'][v,C,a,A] += scale * +1.00000000 * np.einsum('aIbU,bu->aIuU', h['ab'][v,C,v,A], t['a'][pv,ha], optimize=True)
    
    
    
    
    O['ab'][a,V,a,A] += scale * -1.00000000 * np.einsum('iAuU,vi->vAuU', h['ab'][c,V,a,A], t['a'][pa,hc], optimize=True)
    O['ab'][a,V,a,A] += scale * +1.00000000 * np.einsum('uAaU,av->uAvU', h['ab'][a,V,v,A], t['a'][pv,ha], optimize=True)
    
    
    O['ab'][c,V,a,A] += scale * +1.00000000 * np.einsum('iAaU,au->iAuU', h['ab'][c,V,v,A], t['a'][pv,ha], optimize=True)
    O['ab'][v,V,a,A] += scale * -1.00000000 * np.einsum('uv,uAwU,av->aAwU', eta1['a'], h['ab'][a,V,a,A], t['a'][pv,ha], optimize=True)
    
    O['ab'][v,V,a,A] += scale * -1.00000000 * np.einsum('uv,uAwU,av->aAwU', gamma1['a'], h['ab'][a,V,a,A], t['a'][pv,ha], optimize=True)
    
    O['ab'][v,V,a,A] += scale * -1.00000000 * np.einsum('iAuU,ai->aAuU', h['ab'][c,V,a,A], t['a'][pv,hc], optimize=True)
    O['ab'][v,V,a,A] += scale * +1.00000000 * np.einsum('aAbU,bu->aAuU', h['ab'][v,V,v,A], t['a'][pv,ha], optimize=True)
    
    
    
    
    O['ab'][a,A,a,C] += scale * -1.00000000 * np.einsum('iUuI,vi->vUuI', h['ab'][c,A,a,C], t['a'][pa,hc], optimize=True)
    O['ab'][a,A,a,C] += scale * +1.00000000 * np.einsum('uUaI,av->uUvI', h['ab'][a,A,v,C], t['a'][pv,ha], optimize=True)
    
    
    O['ab'][c,A,a,C] += scale * +1.00000000 * np.einsum('iUaI,au->iUuI', h['ab'][c,A,v,C], t['a'][pv,ha], optimize=True)
    O['ab'][v,A,a,C] += scale * -1.00000000 * np.einsum('uv,uUwI,av->aUwI', eta1['a'], h['ab'][a,A,a,C], t['a'][pv,ha], optimize=True)
    
    O['ab'][v,A,a,C] += scale * -1.00000000 * np.einsum('uv,uUwI,av->aUwI', gamma1['a'], h['ab'][a,A,a,C], t['a'][pv,ha], optimize=True)
    
    O['ab'][v,A,a,C] += scale * -1.00000000 * np.einsum('iUuI,ai->aUuI', h['ab'][c,A,a,C], t['a'][pv,hc], optimize=True)
    O['ab'][v,A,a,C] += scale * +1.00000000 * np.einsum('aUbI,bu->aUuI', h['ab'][v,A,v,C], t['a'][pv,ha], optimize=True)
    
    
    
    
    O['ab'][a,C,a,C] += scale * -1.00000000 * np.einsum('iIuJ,vi->vIuJ', h['ab'][c,C,a,C], t['a'][pa,hc], optimize=True)
    O['ab'][a,C,a,C] += scale * +1.00000000 * np.einsum('uIaJ,av->uIvJ', h['ab'][a,C,v,C], t['a'][pv,ha], optimize=True)
    
    
    O['ab'][c,C,a,C] += scale * +1.00000000 * np.einsum('iIaJ,au->iIuJ', h['ab'][c,C,v,C], t['a'][pv,ha], optimize=True)
    O['ab'][v,C,a,C] += scale * -1.00000000 * np.einsum('uv,uIwJ,av->aIwJ', eta1['a'], h['ab'][a,C,a,C], t['a'][pv,ha], optimize=True)
    
    O['ab'][v,C,a,C] += scale * -1.00000000 * np.einsum('uv,uIwJ,av->aIwJ', gamma1['a'], h['ab'][a,C,a,C], t['a'][pv,ha], optimize=True)
    
    O['ab'][v,C,a,C] += scale * -1.00000000 * np.einsum('iIuJ,ai->aIuJ', h['ab'][c,C,a,C], t['a'][pv,hc], optimize=True)
    O['ab'][v,C,a,C] += scale * +1.00000000 * np.einsum('aIbJ,bu->aIuJ', h['ab'][v,C,v,C], t['a'][pv,ha], optimize=True)
    
    
    
    
    O['ab'][a,V,a,C] += scale * -1.00000000 * np.einsum('iAuI,vi->vAuI', h['ab'][c,V,a,C], t['a'][pa,hc], optimize=True)
    O['ab'][a,V,a,C] += scale * +1.00000000 * np.einsum('uAaI,av->uAvI', h['ab'][a,V,v,C], t['a'][pv,ha], optimize=True)
    
    
    O['ab'][c,V,a,C] += scale * +1.00000000 * np.einsum('iAaI,au->iAuI', h['ab'][c,V,v,C], t['a'][pv,ha], optimize=True)
    O['ab'][v,V,a,C] += scale * -1.00000000 * np.einsum('uv,uAwI,av->aAwI', eta1['a'], h['ab'][a,V,a,C], t['a'][pv,ha], optimize=True)
    
    O['ab'][v,V,a,C] += scale * -1.00000000 * np.einsum('uv,uAwI,av->aAwI', gamma1['a'], h['ab'][a,V,a,C], t['a'][pv,ha], optimize=True)
    
    O['ab'][v,V,a,C] += scale * -1.00000000 * np.einsum('iAuI,ai->aAuI', h['ab'][c,V,a,C], t['a'][pv,hc], optimize=True)
    O['ab'][v,V,a,C] += scale * +1.00000000 * np.einsum('aAbI,bu->aAuI', h['ab'][v,V,v,C], t['a'][pv,ha], optimize=True)
    
    
    
    
    O['ab'][a,A,a,V] += scale * -1.00000000 * np.einsum('iUuA,vi->vUuA', h['ab'][c,A,a,V], t['a'][pa,hc], optimize=True)
    O['ab'][a,A,a,V] += scale * +1.00000000 * np.einsum('uUaA,av->uUvA', h['ab'][a,A,v,V], t['a'][pv,ha], optimize=True)
    
    
    O['ab'][c,A,a,V] += scale * +1.00000000 * np.einsum('iUaA,au->iUuA', h['ab'][c,A,v,V], t['a'][pv,ha], optimize=True)
    O['ab'][v,A,a,V] += scale * -1.00000000 * np.einsum('uv,uUwA,av->aUwA', eta1['a'], h['ab'][a,A,a,V], t['a'][pv,ha], optimize=True)
    
    O['ab'][v,A,a,V] += scale * -1.00000000 * np.einsum('uv,uUwA,av->aUwA', gamma1['a'], h['ab'][a,A,a,V], t['a'][pv,ha], optimize=True)
    
    O['ab'][v,A,a,V] += scale * -1.00000000 * np.einsum('iUuA,ai->aUuA', h['ab'][c,A,a,V], t['a'][pv,hc], optimize=True)
    O['ab'][v,A,a,V] += scale * +1.00000000 * np.einsum('aUbA,bu->aUuA', h['ab'][v,A,v,V], t['a'][pv,ha], optimize=True)
    
    
    
    
    O['ab'][a,C,a,V] += scale * -1.00000000 * np.einsum('iIuA,vi->vIuA', h['ab'][c,C,a,V], t['a'][pa,hc], optimize=True)
    O['ab'][a,C,a,V] += scale * +1.00000000 * np.einsum('uIaA,av->uIvA', h['ab'][a,C,v,V], t['a'][pv,ha], optimize=True)
    
    
    O['ab'][c,C,a,V] += scale * +1.00000000 * np.einsum('iIaA,au->iIuA', h['ab'][c,C,v,V], t['a'][pv,ha], optimize=True)
    O['ab'][v,C,a,V] += scale * -1.00000000 * np.einsum('uv,uIwA,av->aIwA', eta1['a'], h['ab'][a,C,a,V], t['a'][pv,ha], optimize=True)
    
    O['ab'][v,C,a,V] += scale * -1.00000000 * np.einsum('uv,uIwA,av->aIwA', gamma1['a'], h['ab'][a,C,a,V], t['a'][pv,ha], optimize=True)
    
    O['ab'][v,C,a,V] += scale * -1.00000000 * np.einsum('iIuA,ai->aIuA', h['ab'][c,C,a,V], t['a'][pv,hc], optimize=True)
    O['ab'][v,C,a,V] += scale * +1.00000000 * np.einsum('aIbA,bu->aIuA', h['ab'][v,C,v,V], t['a'][pv,ha], optimize=True)
    
    
    
    
    O['ab'][a,V,a,V] += scale * -1.00000000 * np.einsum('iAuB,vi->vAuB', h['ab'][c,V,a,V], t['a'][pa,hc], optimize=True)
    O['ab'][a,V,a,V] += scale * +1.00000000 * np.einsum('uAaB,av->uAvB', h['ab'][a,V,v,V], t['a'][pv,ha], optimize=True)
    
    
    O['ab'][c,V,a,V] += scale * +1.00000000 * np.einsum('iAaB,au->iAuB', h['ab'][c,V,v,V], t['a'][pv,ha], optimize=True)
    O['ab'][v,V,a,V] += scale * -1.00000000 * np.einsum('uv,uAwB,av->aAwB', eta1['a'], h['ab'][a,V,a,V], t['a'][pv,ha], optimize=True)
    
    O['ab'][v,V,a,V] += scale * -1.00000000 * np.einsum('uv,uAwB,av->aAwB', gamma1['a'], h['ab'][a,V,a,V], t['a'][pv,ha], optimize=True)
    
    O['ab'][v,V,a,V] += scale * -1.00000000 * np.einsum('iAuB,ai->aAuB', h['ab'][c,V,a,V], t['a'][pv,hc], optimize=True)
    O['ab'][v,V,a,V] += scale * +1.00000000 * np.einsum('aAbB,bu->aAuB', h['ab'][v,V,v,V], t['a'][pv,ha], optimize=True)
    O['ab'][a,A,c,A] += scale * +1.00000000 * np.einsum('uv,wUvV,ui->wUiV', eta1['a'], h['ab'][a,A,a,A], t['a'][pa,hc], optimize=True)
    
    O['ab'][a,A,c,A] += scale * +1.00000000 * np.einsum('uv,wUvV,ui->wUiV', gamma1['a'], h['ab'][a,A,a,A], t['a'][pa,hc], optimize=True)
    
    O['ab'][a,A,c,A] += scale * -1.00000000 * np.einsum('iUjV,ui->uUjV', h['ab'][c,A,c,A], t['a'][pa,hc], optimize=True)
    O['ab'][a,A,c,A] += scale * +1.00000000 * np.einsum('uUaV,ai->uUiV', h['ab'][a,A,v,A], t['a'][pv,hc], optimize=True)
    O['ab'][c,A,c,A] += scale * +1.00000000 * np.einsum('uv,iUvV,uj->iUjV', eta1['a'], h['ab'][c,A,a,A], t['a'][pa,hc], optimize=True)
    O['ab'][c,A,c,A] += scale * +1.00000000 * np.einsum('uv,iUvV,uj->iUjV', gamma1['a'], h['ab'][c,A,a,A], t['a'][pa,hc], optimize=True)
    O['ab'][c,A,c,A] += scale * +1.00000000 * np.einsum('iUaV,aj->iUjV', h['ab'][c,A,v,A], t['a'][pv,hc], optimize=True)
    O['ab'][v,A,c,A] += scale * -1.00000000 * np.einsum('uv,uUiV,av->aUiV', eta1['a'], h['ab'][a,A,c,A], t['a'][pv,ha], optimize=True)
    O['ab'][v,A,c,A] += scale * +1.00000000 * np.einsum('uv,aUvV,ui->aUiV', eta1['a'], h['ab'][v,A,a,A], t['a'][pa,hc], optimize=True)
    O['ab'][v,A,c,A] += scale * -1.00000000 * np.einsum('uv,uUiV,av->aUiV', gamma1['a'], h['ab'][a,A,c,A], t['a'][pv,ha], optimize=True)
    O['ab'][v,A,c,A] += scale * +1.00000000 * np.einsum('uv,aUvV,ui->aUiV', gamma1['a'], h['ab'][v,A,a,A], t['a'][pa,hc], optimize=True)
    O['ab'][v,A,c,A] += scale * -1.00000000 * np.einsum('iUjV,ai->aUjV', h['ab'][c,A,c,A], t['a'][pv,hc], optimize=True)
    O['ab'][v,A,c,A] += scale * +1.00000000 * np.einsum('aUbV,bi->aUiV', h['ab'][v,A,v,A], t['a'][pv,hc], optimize=True)
    O['ab'][a,C,c,A] += scale * +1.00000000 * np.einsum('uv,wIvU,ui->wIiU', eta1['a'], h['ab'][a,C,a,A], t['a'][pa,hc], optimize=True)
    
    O['ab'][a,C,c,A] += scale * +1.00000000 * np.einsum('uv,wIvU,ui->wIiU', gamma1['a'], h['ab'][a,C,a,A], t['a'][pa,hc], optimize=True)
    
    O['ab'][a,C,c,A] += scale * -1.00000000 * np.einsum('iIjU,ui->uIjU', h['ab'][c,C,c,A], t['a'][pa,hc], optimize=True)
    O['ab'][a,C,c,A] += scale * +1.00000000 * np.einsum('uIaU,ai->uIiU', h['ab'][a,C,v,A], t['a'][pv,hc], optimize=True)
    O['ab'][c,C,c,A] += scale * +1.00000000 * np.einsum('uv,iIvU,uj->iIjU', eta1['a'], h['ab'][c,C,a,A], t['a'][pa,hc], optimize=True)
    O['ab'][c,C,c,A] += scale * +1.00000000 * np.einsum('uv,iIvU,uj->iIjU', gamma1['a'], h['ab'][c,C,a,A], t['a'][pa,hc], optimize=True)
    O['ab'][c,C,c,A] += scale * +1.00000000 * np.einsum('iIaU,aj->iIjU', h['ab'][c,C,v,A], t['a'][pv,hc], optimize=True)
    O['ab'][v,C,c,A] += scale * -1.00000000 * np.einsum('uv,uIiU,av->aIiU', eta1['a'], h['ab'][a,C,c,A], t['a'][pv,ha], optimize=True)
    O['ab'][v,C,c,A] += scale * +1.00000000 * np.einsum('uv,aIvU,ui->aIiU', eta1['a'], h['ab'][v,C,a,A], t['a'][pa,hc], optimize=True)
    O['ab'][v,C,c,A] += scale * -1.00000000 * np.einsum('uv,uIiU,av->aIiU', gamma1['a'], h['ab'][a,C,c,A], t['a'][pv,ha], optimize=True)
    O['ab'][v,C,c,A] += scale * +1.00000000 * np.einsum('uv,aIvU,ui->aIiU', gamma1['a'], h['ab'][v,C,a,A], t['a'][pa,hc], optimize=True)
    O['ab'][v,C,c,A] += scale * -1.00000000 * np.einsum('iIjU,ai->aIjU', h['ab'][c,C,c,A], t['a'][pv,hc], optimize=True)
    O['ab'][v,C,c,A] += scale * +1.00000000 * np.einsum('aIbU,bi->aIiU', h['ab'][v,C,v,A], t['a'][pv,hc], optimize=True)
    O['ab'][a,V,c,A] += scale * +1.00000000 * np.einsum('uv,wAvU,ui->wAiU', eta1['a'], h['ab'][a,V,a,A], t['a'][pa,hc], optimize=True)
    
    O['ab'][a,V,c,A] += scale * +1.00000000 * np.einsum('uv,wAvU,ui->wAiU', gamma1['a'], h['ab'][a,V,a,A], t['a'][pa,hc], optimize=True)
    
    O['ab'][a,V,c,A] += scale * -1.00000000 * np.einsum('iAjU,ui->uAjU', h['ab'][c,V,c,A], t['a'][pa,hc], optimize=True)
    O['ab'][a,V,c,A] += scale * +1.00000000 * np.einsum('uAaU,ai->uAiU', h['ab'][a,V,v,A], t['a'][pv,hc], optimize=True)
    O['ab'][c,V,c,A] += scale * +1.00000000 * np.einsum('uv,iAvU,uj->iAjU', eta1['a'], h['ab'][c,V,a,A], t['a'][pa,hc], optimize=True)
    O['ab'][c,V,c,A] += scale * +1.00000000 * np.einsum('uv,iAvU,uj->iAjU', gamma1['a'], h['ab'][c,V,a,A], t['a'][pa,hc], optimize=True)
    O['ab'][c,V,c,A] += scale * +1.00000000 * np.einsum('iAaU,aj->iAjU', h['ab'][c,V,v,A], t['a'][pv,hc], optimize=True)
    O['ab'][v,V,c,A] += scale * -1.00000000 * np.einsum('uv,uAiU,av->aAiU', eta1['a'], h['ab'][a,V,c,A], t['a'][pv,ha], optimize=True)
    O['ab'][v,V,c,A] += scale * +1.00000000 * np.einsum('uv,aAvU,ui->aAiU', eta1['a'], h['ab'][v,V,a,A], t['a'][pa,hc], optimize=True)
    O['ab'][v,V,c,A] += scale * -1.00000000 * np.einsum('uv,uAiU,av->aAiU', gamma1['a'], h['ab'][a,V,c,A], t['a'][pv,ha], optimize=True)
    O['ab'][v,V,c,A] += scale * +1.00000000 * np.einsum('uv,aAvU,ui->aAiU', gamma1['a'], h['ab'][v,V,a,A], t['a'][pa,hc], optimize=True)
    O['ab'][v,V,c,A] += scale * -1.00000000 * np.einsum('iAjU,ai->aAjU', h['ab'][c,V,c,A], t['a'][pv,hc], optimize=True)
    O['ab'][v,V,c,A] += scale * +1.00000000 * np.einsum('aAbU,bi->aAiU', h['ab'][v,V,v,A], t['a'][pv,hc], optimize=True)
    O['ab'][a,A,c,C] += scale * +1.00000000 * np.einsum('uv,wUvI,ui->wUiI', eta1['a'], h['ab'][a,A,a,C], t['a'][pa,hc], optimize=True)
    
    O['ab'][a,A,c,C] += scale * +1.00000000 * np.einsum('uv,wUvI,ui->wUiI', gamma1['a'], h['ab'][a,A,a,C], t['a'][pa,hc], optimize=True)
    
    O['ab'][a,A,c,C] += scale * -1.00000000 * np.einsum('iUjI,ui->uUjI', h['ab'][c,A,c,C], t['a'][pa,hc], optimize=True)
    O['ab'][a,A,c,C] += scale * +1.00000000 * np.einsum('uUaI,ai->uUiI', h['ab'][a,A,v,C], t['a'][pv,hc], optimize=True)
    O['ab'][c,A,c,C] += scale * +1.00000000 * np.einsum('uv,iUvI,uj->iUjI', eta1['a'], h['ab'][c,A,a,C], t['a'][pa,hc], optimize=True)
    O['ab'][c,A,c,C] += scale * +1.00000000 * np.einsum('uv,iUvI,uj->iUjI', gamma1['a'], h['ab'][c,A,a,C], t['a'][pa,hc], optimize=True)
    O['ab'][c,A,c,C] += scale * +1.00000000 * np.einsum('iUaI,aj->iUjI', h['ab'][c,A,v,C], t['a'][pv,hc], optimize=True)
    O['ab'][v,A,c,C] += scale * -1.00000000 * np.einsum('uv,uUiI,av->aUiI', eta1['a'], h['ab'][a,A,c,C], t['a'][pv,ha], optimize=True)
    O['ab'][v,A,c,C] += scale * +1.00000000 * np.einsum('uv,aUvI,ui->aUiI', eta1['a'], h['ab'][v,A,a,C], t['a'][pa,hc], optimize=True)
    O['ab'][v,A,c,C] += scale * -1.00000000 * np.einsum('uv,uUiI,av->aUiI', gamma1['a'], h['ab'][a,A,c,C], t['a'][pv,ha], optimize=True)
    O['ab'][v,A,c,C] += scale * +1.00000000 * np.einsum('uv,aUvI,ui->aUiI', gamma1['a'], h['ab'][v,A,a,C], t['a'][pa,hc], optimize=True)
    O['ab'][v,A,c,C] += scale * -1.00000000 * np.einsum('iUjI,ai->aUjI', h['ab'][c,A,c,C], t['a'][pv,hc], optimize=True)
    O['ab'][v,A,c,C] += scale * +1.00000000 * np.einsum('aUbI,bi->aUiI', h['ab'][v,A,v,C], t['a'][pv,hc], optimize=True)
    O['ab'][a,C,c,C] += scale * +1.00000000 * np.einsum('uv,wIvJ,ui->wIiJ', eta1['a'], h['ab'][a,C,a,C], t['a'][pa,hc], optimize=True)
    
    O['ab'][a,C,c,C] += scale * +1.00000000 * np.einsum('uv,wIvJ,ui->wIiJ', gamma1['a'], h['ab'][a,C,a,C], t['a'][pa,hc], optimize=True)
    
    O['ab'][a,C,c,C] += scale * -1.00000000 * np.einsum('iIjJ,ui->uIjJ', h['ab'][c,C,c,C], t['a'][pa,hc], optimize=True)
    O['ab'][a,C,c,C] += scale * +1.00000000 * np.einsum('uIaJ,ai->uIiJ', h['ab'][a,C,v,C], t['a'][pv,hc], optimize=True)
    O['ab'][c,C,c,C] += scale * +1.00000000 * np.einsum('uv,iIvJ,uj->iIjJ', eta1['a'], h['ab'][c,C,a,C], t['a'][pa,hc], optimize=True)
    O['ab'][c,C,c,C] += scale * +1.00000000 * np.einsum('uv,iIvJ,uj->iIjJ', gamma1['a'], h['ab'][c,C,a,C], t['a'][pa,hc], optimize=True)
    O['ab'][c,C,c,C] += scale * +1.00000000 * np.einsum('iIaJ,aj->iIjJ', h['ab'][c,C,v,C], t['a'][pv,hc], optimize=True)
    O['ab'][v,C,c,C] += scale * -1.00000000 * np.einsum('uv,uIiJ,av->aIiJ', eta1['a'], h['ab'][a,C,c,C], t['a'][pv,ha], optimize=True)
    O['ab'][v,C,c,C] += scale * +1.00000000 * np.einsum('uv,aIvJ,ui->aIiJ', eta1['a'], h['ab'][v,C,a,C], t['a'][pa,hc], optimize=True)
    O['ab'][v,C,c,C] += scale * -1.00000000 * np.einsum('uv,uIiJ,av->aIiJ', gamma1['a'], h['ab'][a,C,c,C], t['a'][pv,ha], optimize=True)
    O['ab'][v,C,c,C] += scale * +1.00000000 * np.einsum('uv,aIvJ,ui->aIiJ', gamma1['a'], h['ab'][v,C,a,C], t['a'][pa,hc], optimize=True)
    O['ab'][v,C,c,C] += scale * -1.00000000 * np.einsum('iIjJ,ai->aIjJ', h['ab'][c,C,c,C], t['a'][pv,hc], optimize=True)
    O['ab'][v,C,c,C] += scale * +1.00000000 * np.einsum('aIbJ,bi->aIiJ', h['ab'][v,C,v,C], t['a'][pv,hc], optimize=True)
    O['ab'][a,V,c,C] += scale * +1.00000000 * np.einsum('uv,wAvI,ui->wAiI', eta1['a'], h['ab'][a,V,a,C], t['a'][pa,hc], optimize=True)
    
    O['ab'][a,V,c,C] += scale * +1.00000000 * np.einsum('uv,wAvI,ui->wAiI', gamma1['a'], h['ab'][a,V,a,C], t['a'][pa,hc], optimize=True)
    
    O['ab'][a,V,c,C] += scale * -1.00000000 * np.einsum('iAjI,ui->uAjI', h['ab'][c,V,c,C], t['a'][pa,hc], optimize=True)
    O['ab'][a,V,c,C] += scale * +1.00000000 * np.einsum('uAaI,ai->uAiI', h['ab'][a,V,v,C], t['a'][pv,hc], optimize=True)
    O['ab'][c,V,c,C] += scale * +1.00000000 * np.einsum('uv,iAvI,uj->iAjI', eta1['a'], h['ab'][c,V,a,C], t['a'][pa,hc], optimize=True)
    O['ab'][c,V,c,C] += scale * +1.00000000 * np.einsum('uv,iAvI,uj->iAjI', gamma1['a'], h['ab'][c,V,a,C], t['a'][pa,hc], optimize=True)
    O['ab'][c,V,c,C] += scale * +1.00000000 * np.einsum('iAaI,aj->iAjI', h['ab'][c,V,v,C], t['a'][pv,hc], optimize=True)
    O['ab'][v,V,c,C] += scale * -1.00000000 * np.einsum('uv,uAiI,av->aAiI', eta1['a'], h['ab'][a,V,c,C], t['a'][pv,ha], optimize=True)
    O['ab'][v,V,c,C] += scale * +1.00000000 * np.einsum('uv,aAvI,ui->aAiI', eta1['a'], h['ab'][v,V,a,C], t['a'][pa,hc], optimize=True)
    O['ab'][v,V,c,C] += scale * -1.00000000 * np.einsum('uv,uAiI,av->aAiI', gamma1['a'], h['ab'][a,V,c,C], t['a'][pv,ha], optimize=True)
    O['ab'][v,V,c,C] += scale * +1.00000000 * np.einsum('uv,aAvI,ui->aAiI', gamma1['a'], h['ab'][v,V,a,C], t['a'][pa,hc], optimize=True)
    O['ab'][v,V,c,C] += scale * -1.00000000 * np.einsum('iAjI,ai->aAjI', h['ab'][c,V,c,C], t['a'][pv,hc], optimize=True)
    O['ab'][v,V,c,C] += scale * +1.00000000 * np.einsum('aAbI,bi->aAiI', h['ab'][v,V,v,C], t['a'][pv,hc], optimize=True)
    O['ab'][a,A,c,V] += scale * +1.00000000 * np.einsum('uv,wUvA,ui->wUiA', eta1['a'], h['ab'][a,A,a,V], t['a'][pa,hc], optimize=True)
    
    O['ab'][a,A,c,V] += scale * +1.00000000 * np.einsum('uv,wUvA,ui->wUiA', gamma1['a'], h['ab'][a,A,a,V], t['a'][pa,hc], optimize=True)
    
    O['ab'][a,A,c,V] += scale * -1.00000000 * np.einsum('iUjA,ui->uUjA', h['ab'][c,A,c,V], t['a'][pa,hc], optimize=True)
    O['ab'][a,A,c,V] += scale * +1.00000000 * np.einsum('uUaA,ai->uUiA', h['ab'][a,A,v,V], t['a'][pv,hc], optimize=True)
    O['ab'][c,A,c,V] += scale * +1.00000000 * np.einsum('uv,iUvA,uj->iUjA', eta1['a'], h['ab'][c,A,a,V], t['a'][pa,hc], optimize=True)
    O['ab'][c,A,c,V] += scale * +1.00000000 * np.einsum('uv,iUvA,uj->iUjA', gamma1['a'], h['ab'][c,A,a,V], t['a'][pa,hc], optimize=True)
    O['ab'][c,A,c,V] += scale * +1.00000000 * np.einsum('iUaA,aj->iUjA', h['ab'][c,A,v,V], t['a'][pv,hc], optimize=True)
    O['ab'][v,A,c,V] += scale * -1.00000000 * np.einsum('uv,uUiA,av->aUiA', eta1['a'], h['ab'][a,A,c,V], t['a'][pv,ha], optimize=True)
    O['ab'][v,A,c,V] += scale * +1.00000000 * np.einsum('uv,aUvA,ui->aUiA', eta1['a'], h['ab'][v,A,a,V], t['a'][pa,hc], optimize=True)
    O['ab'][v,A,c,V] += scale * -1.00000000 * np.einsum('uv,uUiA,av->aUiA', gamma1['a'], h['ab'][a,A,c,V], t['a'][pv,ha], optimize=True)
    O['ab'][v,A,c,V] += scale * +1.00000000 * np.einsum('uv,aUvA,ui->aUiA', gamma1['a'], h['ab'][v,A,a,V], t['a'][pa,hc], optimize=True)
    O['ab'][v,A,c,V] += scale * -1.00000000 * np.einsum('iUjA,ai->aUjA', h['ab'][c,A,c,V], t['a'][pv,hc], optimize=True)
    O['ab'][v,A,c,V] += scale * +1.00000000 * np.einsum('aUbA,bi->aUiA', h['ab'][v,A,v,V], t['a'][pv,hc], optimize=True)
    O['ab'][a,C,c,V] += scale * +1.00000000 * np.einsum('uv,wIvA,ui->wIiA', eta1['a'], h['ab'][a,C,a,V], t['a'][pa,hc], optimize=True)
    
    O['ab'][a,C,c,V] += scale * +1.00000000 * np.einsum('uv,wIvA,ui->wIiA', gamma1['a'], h['ab'][a,C,a,V], t['a'][pa,hc], optimize=True)
    
    O['ab'][a,C,c,V] += scale * -1.00000000 * np.einsum('iIjA,ui->uIjA', h['ab'][c,C,c,V], t['a'][pa,hc], optimize=True)
    O['ab'][a,C,c,V] += scale * +1.00000000 * np.einsum('uIaA,ai->uIiA', h['ab'][a,C,v,V], t['a'][pv,hc], optimize=True)
    O['ab'][c,C,c,V] += scale * +1.00000000 * np.einsum('uv,iIvA,uj->iIjA', eta1['a'], h['ab'][c,C,a,V], t['a'][pa,hc], optimize=True)
    O['ab'][c,C,c,V] += scale * +1.00000000 * np.einsum('uv,iIvA,uj->iIjA', gamma1['a'], h['ab'][c,C,a,V], t['a'][pa,hc], optimize=True)
    O['ab'][c,C,c,V] += scale * +1.00000000 * np.einsum('iIaA,aj->iIjA', h['ab'][c,C,v,V], t['a'][pv,hc], optimize=True)
    O['ab'][v,C,c,V] += scale * -1.00000000 * np.einsum('uv,uIiA,av->aIiA', eta1['a'], h['ab'][a,C,c,V], t['a'][pv,ha], optimize=True)
    O['ab'][v,C,c,V] += scale * +1.00000000 * np.einsum('uv,aIvA,ui->aIiA', eta1['a'], h['ab'][v,C,a,V], t['a'][pa,hc], optimize=True)
    O['ab'][v,C,c,V] += scale * -1.00000000 * np.einsum('uv,uIiA,av->aIiA', gamma1['a'], h['ab'][a,C,c,V], t['a'][pv,ha], optimize=True)
    O['ab'][v,C,c,V] += scale * +1.00000000 * np.einsum('uv,aIvA,ui->aIiA', gamma1['a'], h['ab'][v,C,a,V], t['a'][pa,hc], optimize=True)
    O['ab'][v,C,c,V] += scale * -1.00000000 * np.einsum('iIjA,ai->aIjA', h['ab'][c,C,c,V], t['a'][pv,hc], optimize=True)
    O['ab'][v,C,c,V] += scale * +1.00000000 * np.einsum('aIbA,bi->aIiA', h['ab'][v,C,v,V], t['a'][pv,hc], optimize=True)
    O['ab'][a,V,c,V] += scale * +1.00000000 * np.einsum('uv,wAvB,ui->wAiB', eta1['a'], h['ab'][a,V,a,V], t['a'][pa,hc], optimize=True)
    
    O['ab'][a,V,c,V] += scale * +1.00000000 * np.einsum('uv,wAvB,ui->wAiB', gamma1['a'], h['ab'][a,V,a,V], t['a'][pa,hc], optimize=True)
    
    O['ab'][a,V,c,V] += scale * -1.00000000 * np.einsum('iAjB,ui->uAjB', h['ab'][c,V,c,V], t['a'][pa,hc], optimize=True)
    O['ab'][a,V,c,V] += scale * +1.00000000 * np.einsum('uAaB,ai->uAiB', h['ab'][a,V,v,V], t['a'][pv,hc], optimize=True)
    O['ab'][c,V,c,V] += scale * +1.00000000 * np.einsum('uv,iAvB,uj->iAjB', eta1['a'], h['ab'][c,V,a,V], t['a'][pa,hc], optimize=True)
    O['ab'][c,V,c,V] += scale * +1.00000000 * np.einsum('uv,iAvB,uj->iAjB', gamma1['a'], h['ab'][c,V,a,V], t['a'][pa,hc], optimize=True)
    O['ab'][c,V,c,V] += scale * +1.00000000 * np.einsum('iAaB,aj->iAjB', h['ab'][c,V,v,V], t['a'][pv,hc], optimize=True)
    O['ab'][v,V,c,V] += scale * -1.00000000 * np.einsum('uv,uAiB,av->aAiB', eta1['a'], h['ab'][a,V,c,V], t['a'][pv,ha], optimize=True)
    O['ab'][v,V,c,V] += scale * +1.00000000 * np.einsum('uv,aAvB,ui->aAiB', eta1['a'], h['ab'][v,V,a,V], t['a'][pa,hc], optimize=True)
    O['ab'][v,V,c,V] += scale * -1.00000000 * np.einsum('uv,uAiB,av->aAiB', gamma1['a'], h['ab'][a,V,c,V], t['a'][pv,ha], optimize=True)
    O['ab'][v,V,c,V] += scale * +1.00000000 * np.einsum('uv,aAvB,ui->aAiB', gamma1['a'], h['ab'][v,V,a,V], t['a'][pa,hc], optimize=True)
    O['ab'][v,V,c,V] += scale * -1.00000000 * np.einsum('iAjB,ai->aAjB', h['ab'][c,V,c,V], t['a'][pv,hc], optimize=True)
    O['ab'][v,V,c,V] += scale * +1.00000000 * np.einsum('aAbB,bi->aAiB', h['ab'][v,V,v,V], t['a'][pv,hc], optimize=True)
    
    
    O['ab'][a,A,v,A] += scale * -1.00000000 * np.einsum('iUaV,ui->uUaV', h['ab'][c,A,v,A], t['a'][pa,hc], optimize=True)
    O['ab'][v,A,v,A] += scale * -1.00000000 * np.einsum('uv,uUaV,bv->bUaV', eta1['a'], h['ab'][a,A,v,A], t['a'][pv,ha], optimize=True)
    O['ab'][v,A,v,A] += scale * -1.00000000 * np.einsum('uv,uUaV,bv->bUaV', gamma1['a'], h['ab'][a,A,v,A], t['a'][pv,ha], optimize=True)
    O['ab'][v,A,v,A] += scale * -1.00000000 * np.einsum('iUaV,bi->bUaV', h['ab'][c,A,v,A], t['a'][pv,hc], optimize=True)
    
    
    O['ab'][a,C,v,A] += scale * -1.00000000 * np.einsum('iIaU,ui->uIaU', h['ab'][c,C,v,A], t['a'][pa,hc], optimize=True)
    O['ab'][v,C,v,A] += scale * -1.00000000 * np.einsum('uv,uIaU,bv->bIaU', eta1['a'], h['ab'][a,C,v,A], t['a'][pv,ha], optimize=True)
    O['ab'][v,C,v,A] += scale * -1.00000000 * np.einsum('uv,uIaU,bv->bIaU', gamma1['a'], h['ab'][a,C,v,A], t['a'][pv,ha], optimize=True)
    O['ab'][v,C,v,A] += scale * -1.00000000 * np.einsum('iIaU,bi->bIaU', h['ab'][c,C,v,A], t['a'][pv,hc], optimize=True)
    
    
    O['ab'][a,V,v,A] += scale * -1.00000000 * np.einsum('iAaU,ui->uAaU', h['ab'][c,V,v,A], t['a'][pa,hc], optimize=True)
    O['ab'][v,V,v,A] += scale * -1.00000000 * np.einsum('uv,uAaU,bv->bAaU', eta1['a'], h['ab'][a,V,v,A], t['a'][pv,ha], optimize=True)
    O['ab'][v,V,v,A] += scale * -1.00000000 * np.einsum('uv,uAaU,bv->bAaU', gamma1['a'], h['ab'][a,V,v,A], t['a'][pv,ha], optimize=True)
    O['ab'][v,V,v,A] += scale * -1.00000000 * np.einsum('iAaU,bi->bAaU', h['ab'][c,V,v,A], t['a'][pv,hc], optimize=True)
    
    
    O['ab'][a,A,v,C] += scale * -1.00000000 * np.einsum('iUaI,ui->uUaI', h['ab'][c,A,v,C], t['a'][pa,hc], optimize=True)
    O['ab'][v,A,v,C] += scale * -1.00000000 * np.einsum('uv,uUaI,bv->bUaI', eta1['a'], h['ab'][a,A,v,C], t['a'][pv,ha], optimize=True)
    O['ab'][v,A,v,C] += scale * -1.00000000 * np.einsum('uv,uUaI,bv->bUaI', gamma1['a'], h['ab'][a,A,v,C], t['a'][pv,ha], optimize=True)
    O['ab'][v,A,v,C] += scale * -1.00000000 * np.einsum('iUaI,bi->bUaI', h['ab'][c,A,v,C], t['a'][pv,hc], optimize=True)
    
    
    O['ab'][a,C,v,C] += scale * -1.00000000 * np.einsum('iIaJ,ui->uIaJ', h['ab'][c,C,v,C], t['a'][pa,hc], optimize=True)
    O['ab'][v,C,v,C] += scale * -1.00000000 * np.einsum('uv,uIaJ,bv->bIaJ', eta1['a'], h['ab'][a,C,v,C], t['a'][pv,ha], optimize=True)
    O['ab'][v,C,v,C] += scale * -1.00000000 * np.einsum('uv,uIaJ,bv->bIaJ', gamma1['a'], h['ab'][a,C,v,C], t['a'][pv,ha], optimize=True)
    O['ab'][v,C,v,C] += scale * -1.00000000 * np.einsum('iIaJ,bi->bIaJ', h['ab'][c,C,v,C], t['a'][pv,hc], optimize=True)
    
    
    O['ab'][a,V,v,C] += scale * -1.00000000 * np.einsum('iAaI,ui->uAaI', h['ab'][c,V,v,C], t['a'][pa,hc], optimize=True)
    O['ab'][v,V,v,C] += scale * -1.00000000 * np.einsum('uv,uAaI,bv->bAaI', eta1['a'], h['ab'][a,V,v,C], t['a'][pv,ha], optimize=True)
    O['ab'][v,V,v,C] += scale * -1.00000000 * np.einsum('uv,uAaI,bv->bAaI', gamma1['a'], h['ab'][a,V,v,C], t['a'][pv,ha], optimize=True)
    O['ab'][v,V,v,C] += scale * -1.00000000 * np.einsum('iAaI,bi->bAaI', h['ab'][c,V,v,C], t['a'][pv,hc], optimize=True)
    
    
    O['ab'][a,A,v,V] += scale * -1.00000000 * np.einsum('iUaA,ui->uUaA', h['ab'][c,A,v,V], t['a'][pa,hc], optimize=True)
    O['ab'][v,A,v,V] += scale * -1.00000000 * np.einsum('uv,uUaA,bv->bUaA', eta1['a'], h['ab'][a,A,v,V], t['a'][pv,ha], optimize=True)
    O['ab'][v,A,v,V] += scale * -1.00000000 * np.einsum('uv,uUaA,bv->bUaA', gamma1['a'], h['ab'][a,A,v,V], t['a'][pv,ha], optimize=True)
    O['ab'][v,A,v,V] += scale * -1.00000000 * np.einsum('iUaA,bi->bUaA', h['ab'][c,A,v,V], t['a'][pv,hc], optimize=True)
    
    
    O['ab'][a,C,v,V] += scale * -1.00000000 * np.einsum('iIaA,ui->uIaA', h['ab'][c,C,v,V], t['a'][pa,hc], optimize=True)
    O['ab'][v,C,v,V] += scale * -1.00000000 * np.einsum('uv,uIaA,bv->bIaA', eta1['a'], h['ab'][a,C,v,V], t['a'][pv,ha], optimize=True)
    O['ab'][v,C,v,V] += scale * -1.00000000 * np.einsum('uv,uIaA,bv->bIaA', gamma1['a'], h['ab'][a,C,v,V], t['a'][pv,ha], optimize=True)
    O['ab'][v,C,v,V] += scale * -1.00000000 * np.einsum('iIaA,bi->bIaA', h['ab'][c,C,v,V], t['a'][pv,hc], optimize=True)
    
    
    O['ab'][a,V,v,V] += scale * -1.00000000 * np.einsum('iAaB,ui->uAaB', h['ab'][c,V,v,V], t['a'][pa,hc], optimize=True)
    O['ab'][v,V,v,V] += scale * -1.00000000 * np.einsum('uv,uAaB,bv->bAaB', eta1['a'], h['ab'][a,V,v,V], t['a'][pv,ha], optimize=True)
    O['ab'][v,V,v,V] += scale * -1.00000000 * np.einsum('uv,uAaB,bv->bAaB', gamma1['a'], h['ab'][a,V,v,V], t['a'][pv,ha], optimize=True)
    O['ab'][v,V,v,V] += scale * -1.00000000 * np.einsum('iAaB,bi->bAaB', h['ab'][c,V,v,V], t['a'][pv,hc], optimize=True)

    t1 = time.time()
    if verbose: print("h2b_t1a_c2b took {:.4f} seconds to run.".format(t1-t0))

    return O
def h2b_t1b_c2b(O, h, t, gamma1, eta1, lambdas, orbspace, verbose=False, scale=1.0):
    # 324 lines
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

    
    
    
    
    O['ab'][a,A,a,A] += scale * -1.00000000 * np.einsum('uIvU,VI->uVvU', h['ab'][a,C,a,A], t['b'][pA,hC], optimize=True)
    O['ab'][a,A,a,A] += scale * +1.00000000 * np.einsum('uUvA,AV->uUvV', h['ab'][a,A,a,V], t['b'][pV,hA], optimize=True)
    
    
    
    
    O['ab'][c,A,a,A] += scale * -1.00000000 * np.einsum('iIuU,VI->iVuU', h['ab'][c,C,a,A], t['b'][pA,hC], optimize=True)
    O['ab'][c,A,a,A] += scale * +1.00000000 * np.einsum('iUuA,AV->iUuV', h['ab'][c,A,a,V], t['b'][pV,hA], optimize=True)
    
    
    
    
    O['ab'][v,A,a,A] += scale * -1.00000000 * np.einsum('aIuU,VI->aVuU', h['ab'][v,C,a,A], t['b'][pA,hC], optimize=True)
    O['ab'][v,A,a,A] += scale * +1.00000000 * np.einsum('aUuA,AV->aUuV', h['ab'][v,A,a,V], t['b'][pV,hA], optimize=True)
    
    
    O['ab'][a,C,a,A] += scale * +1.00000000 * np.einsum('uIvA,AU->uIvU', h['ab'][a,C,a,V], t['b'][pV,hA], optimize=True)
    
    
    O['ab'][c,C,a,A] += scale * +1.00000000 * np.einsum('iIuA,AU->iIuU', h['ab'][c,C,a,V], t['b'][pV,hA], optimize=True)
    
    
    O['ab'][v,C,a,A] += scale * +1.00000000 * np.einsum('aIuA,AU->aIuU', h['ab'][v,C,a,V], t['b'][pV,hA], optimize=True)
    O['ab'][a,V,a,A] += scale * -1.00000000 * np.einsum('UV,uUvW,AV->uAvW', eta1['b'], h['ab'][a,A,a,A], t['b'][pV,hA], optimize=True)
    
    O['ab'][a,V,a,A] += scale * -1.00000000 * np.einsum('UV,uUvW,AV->uAvW', gamma1['b'], h['ab'][a,A,a,A], t['b'][pV,hA], optimize=True)
    
    O['ab'][a,V,a,A] += scale * -1.00000000 * np.einsum('uIvU,AI->uAvU', h['ab'][a,C,a,A], t['b'][pV,hC], optimize=True)
    O['ab'][a,V,a,A] += scale * +1.00000000 * np.einsum('uAvB,BU->uAvU', h['ab'][a,V,a,V], t['b'][pV,hA], optimize=True)
    O['ab'][c,V,a,A] += scale * -1.00000000 * np.einsum('UV,iUuW,AV->iAuW', eta1['b'], h['ab'][c,A,a,A], t['b'][pV,hA], optimize=True)
    
    O['ab'][c,V,a,A] += scale * -1.00000000 * np.einsum('UV,iUuW,AV->iAuW', gamma1['b'], h['ab'][c,A,a,A], t['b'][pV,hA], optimize=True)
    
    O['ab'][c,V,a,A] += scale * -1.00000000 * np.einsum('iIuU,AI->iAuU', h['ab'][c,C,a,A], t['b'][pV,hC], optimize=True)
    O['ab'][c,V,a,A] += scale * +1.00000000 * np.einsum('iAuB,BU->iAuU', h['ab'][c,V,a,V], t['b'][pV,hA], optimize=True)
    O['ab'][v,V,a,A] += scale * -1.00000000 * np.einsum('UV,aUuW,AV->aAuW', eta1['b'], h['ab'][v,A,a,A], t['b'][pV,hA], optimize=True)
    
    O['ab'][v,V,a,A] += scale * -1.00000000 * np.einsum('UV,aUuW,AV->aAuW', gamma1['b'], h['ab'][v,A,a,A], t['b'][pV,hA], optimize=True)
    
    O['ab'][v,V,a,A] += scale * -1.00000000 * np.einsum('aIuU,AI->aAuU', h['ab'][v,C,a,A], t['b'][pV,hC], optimize=True)
    O['ab'][v,V,a,A] += scale * +1.00000000 * np.einsum('aAuB,BU->aAuU', h['ab'][v,V,a,V], t['b'][pV,hA], optimize=True)
    O['ab'][a,A,a,C] += scale * +1.00000000 * np.einsum('UV,uWvV,UI->uWvI', eta1['b'], h['ab'][a,A,a,A], t['b'][pA,hC], optimize=True)
    
    O['ab'][a,A,a,C] += scale * +1.00000000 * np.einsum('UV,uWvV,UI->uWvI', gamma1['b'], h['ab'][a,A,a,A], t['b'][pA,hC], optimize=True)
    
    O['ab'][a,A,a,C] += scale * -1.00000000 * np.einsum('uIvJ,UI->uUvJ', h['ab'][a,C,a,C], t['b'][pA,hC], optimize=True)
    O['ab'][a,A,a,C] += scale * +1.00000000 * np.einsum('uUvA,AI->uUvI', h['ab'][a,A,a,V], t['b'][pV,hC], optimize=True)
    O['ab'][c,A,a,C] += scale * +1.00000000 * np.einsum('UV,iWuV,UI->iWuI', eta1['b'], h['ab'][c,A,a,A], t['b'][pA,hC], optimize=True)
    
    O['ab'][c,A,a,C] += scale * +1.00000000 * np.einsum('UV,iWuV,UI->iWuI', gamma1['b'], h['ab'][c,A,a,A], t['b'][pA,hC], optimize=True)
    
    O['ab'][c,A,a,C] += scale * -1.00000000 * np.einsum('iIuJ,UI->iUuJ', h['ab'][c,C,a,C], t['b'][pA,hC], optimize=True)
    O['ab'][c,A,a,C] += scale * +1.00000000 * np.einsum('iUuA,AI->iUuI', h['ab'][c,A,a,V], t['b'][pV,hC], optimize=True)
    O['ab'][v,A,a,C] += scale * +1.00000000 * np.einsum('UV,aWuV,UI->aWuI', eta1['b'], h['ab'][v,A,a,A], t['b'][pA,hC], optimize=True)
    
    O['ab'][v,A,a,C] += scale * +1.00000000 * np.einsum('UV,aWuV,UI->aWuI', gamma1['b'], h['ab'][v,A,a,A], t['b'][pA,hC], optimize=True)
    
    O['ab'][v,A,a,C] += scale * -1.00000000 * np.einsum('aIuJ,UI->aUuJ', h['ab'][v,C,a,C], t['b'][pA,hC], optimize=True)
    O['ab'][v,A,a,C] += scale * +1.00000000 * np.einsum('aUuA,AI->aUuI', h['ab'][v,A,a,V], t['b'][pV,hC], optimize=True)
    O['ab'][a,C,a,C] += scale * +1.00000000 * np.einsum('UV,uIvV,UJ->uIvJ', eta1['b'], h['ab'][a,C,a,A], t['b'][pA,hC], optimize=True)
    O['ab'][a,C,a,C] += scale * +1.00000000 * np.einsum('UV,uIvV,UJ->uIvJ', gamma1['b'], h['ab'][a,C,a,A], t['b'][pA,hC], optimize=True)
    O['ab'][a,C,a,C] += scale * +1.00000000 * np.einsum('uIvA,AJ->uIvJ', h['ab'][a,C,a,V], t['b'][pV,hC], optimize=True)
    O['ab'][c,C,a,C] += scale * +1.00000000 * np.einsum('UV,iIuV,UJ->iIuJ', eta1['b'], h['ab'][c,C,a,A], t['b'][pA,hC], optimize=True)
    O['ab'][c,C,a,C] += scale * +1.00000000 * np.einsum('UV,iIuV,UJ->iIuJ', gamma1['b'], h['ab'][c,C,a,A], t['b'][pA,hC], optimize=True)
    O['ab'][c,C,a,C] += scale * +1.00000000 * np.einsum('iIuA,AJ->iIuJ', h['ab'][c,C,a,V], t['b'][pV,hC], optimize=True)
    O['ab'][v,C,a,C] += scale * +1.00000000 * np.einsum('UV,aIuV,UJ->aIuJ', eta1['b'], h['ab'][v,C,a,A], t['b'][pA,hC], optimize=True)
    O['ab'][v,C,a,C] += scale * +1.00000000 * np.einsum('UV,aIuV,UJ->aIuJ', gamma1['b'], h['ab'][v,C,a,A], t['b'][pA,hC], optimize=True)
    O['ab'][v,C,a,C] += scale * +1.00000000 * np.einsum('aIuA,AJ->aIuJ', h['ab'][v,C,a,V], t['b'][pV,hC], optimize=True)
    O['ab'][a,V,a,C] += scale * -1.00000000 * np.einsum('UV,uUvI,AV->uAvI', eta1['b'], h['ab'][a,A,a,C], t['b'][pV,hA], optimize=True)
    O['ab'][a,V,a,C] += scale * +1.00000000 * np.einsum('UV,uAvV,UI->uAvI', eta1['b'], h['ab'][a,V,a,A], t['b'][pA,hC], optimize=True)
    O['ab'][a,V,a,C] += scale * -1.00000000 * np.einsum('UV,uUvI,AV->uAvI', gamma1['b'], h['ab'][a,A,a,C], t['b'][pV,hA], optimize=True)
    O['ab'][a,V,a,C] += scale * +1.00000000 * np.einsum('UV,uAvV,UI->uAvI', gamma1['b'], h['ab'][a,V,a,A], t['b'][pA,hC], optimize=True)
    O['ab'][a,V,a,C] += scale * -1.00000000 * np.einsum('uIvJ,AI->uAvJ', h['ab'][a,C,a,C], t['b'][pV,hC], optimize=True)
    O['ab'][a,V,a,C] += scale * +1.00000000 * np.einsum('uAvB,BI->uAvI', h['ab'][a,V,a,V], t['b'][pV,hC], optimize=True)
    O['ab'][c,V,a,C] += scale * -1.00000000 * np.einsum('UV,iUuI,AV->iAuI', eta1['b'], h['ab'][c,A,a,C], t['b'][pV,hA], optimize=True)
    O['ab'][c,V,a,C] += scale * +1.00000000 * np.einsum('UV,iAuV,UI->iAuI', eta1['b'], h['ab'][c,V,a,A], t['b'][pA,hC], optimize=True)
    O['ab'][c,V,a,C] += scale * -1.00000000 * np.einsum('UV,iUuI,AV->iAuI', gamma1['b'], h['ab'][c,A,a,C], t['b'][pV,hA], optimize=True)
    O['ab'][c,V,a,C] += scale * +1.00000000 * np.einsum('UV,iAuV,UI->iAuI', gamma1['b'], h['ab'][c,V,a,A], t['b'][pA,hC], optimize=True)
    O['ab'][c,V,a,C] += scale * -1.00000000 * np.einsum('iIuJ,AI->iAuJ', h['ab'][c,C,a,C], t['b'][pV,hC], optimize=True)
    O['ab'][c,V,a,C] += scale * +1.00000000 * np.einsum('iAuB,BI->iAuI', h['ab'][c,V,a,V], t['b'][pV,hC], optimize=True)
    O['ab'][v,V,a,C] += scale * -1.00000000 * np.einsum('UV,aUuI,AV->aAuI', eta1['b'], h['ab'][v,A,a,C], t['b'][pV,hA], optimize=True)
    O['ab'][v,V,a,C] += scale * +1.00000000 * np.einsum('UV,aAuV,UI->aAuI', eta1['b'], h['ab'][v,V,a,A], t['b'][pA,hC], optimize=True)
    O['ab'][v,V,a,C] += scale * -1.00000000 * np.einsum('UV,aUuI,AV->aAuI', gamma1['b'], h['ab'][v,A,a,C], t['b'][pV,hA], optimize=True)
    O['ab'][v,V,a,C] += scale * +1.00000000 * np.einsum('UV,aAuV,UI->aAuI', gamma1['b'], h['ab'][v,V,a,A], t['b'][pA,hC], optimize=True)
    O['ab'][v,V,a,C] += scale * -1.00000000 * np.einsum('aIuJ,AI->aAuJ', h['ab'][v,C,a,C], t['b'][pV,hC], optimize=True)
    O['ab'][v,V,a,C] += scale * +1.00000000 * np.einsum('aAuB,BI->aAuI', h['ab'][v,V,a,V], t['b'][pV,hC], optimize=True)
    
    
    O['ab'][a,A,a,V] += scale * -1.00000000 * np.einsum('uIvA,UI->uUvA', h['ab'][a,C,a,V], t['b'][pA,hC], optimize=True)
    
    
    O['ab'][c,A,a,V] += scale * -1.00000000 * np.einsum('iIuA,UI->iUuA', h['ab'][c,C,a,V], t['b'][pA,hC], optimize=True)
    
    
    O['ab'][v,A,a,V] += scale * -1.00000000 * np.einsum('aIuA,UI->aUuA', h['ab'][v,C,a,V], t['b'][pA,hC], optimize=True)
    O['ab'][a,V,a,V] += scale * -1.00000000 * np.einsum('UV,uUvA,BV->uBvA', eta1['b'], h['ab'][a,A,a,V], t['b'][pV,hA], optimize=True)
    O['ab'][a,V,a,V] += scale * -1.00000000 * np.einsum('UV,uUvA,BV->uBvA', gamma1['b'], h['ab'][a,A,a,V], t['b'][pV,hA], optimize=True)
    O['ab'][a,V,a,V] += scale * -1.00000000 * np.einsum('uIvA,BI->uBvA', h['ab'][a,C,a,V], t['b'][pV,hC], optimize=True)
    O['ab'][c,V,a,V] += scale * -1.00000000 * np.einsum('UV,iUuA,BV->iBuA', eta1['b'], h['ab'][c,A,a,V], t['b'][pV,hA], optimize=True)
    O['ab'][c,V,a,V] += scale * -1.00000000 * np.einsum('UV,iUuA,BV->iBuA', gamma1['b'], h['ab'][c,A,a,V], t['b'][pV,hA], optimize=True)
    O['ab'][c,V,a,V] += scale * -1.00000000 * np.einsum('iIuA,BI->iBuA', h['ab'][c,C,a,V], t['b'][pV,hC], optimize=True)
    O['ab'][v,V,a,V] += scale * -1.00000000 * np.einsum('UV,aUuA,BV->aBuA', eta1['b'], h['ab'][v,A,a,V], t['b'][pV,hA], optimize=True)
    O['ab'][v,V,a,V] += scale * -1.00000000 * np.einsum('UV,aUuA,BV->aBuA', gamma1['b'], h['ab'][v,A,a,V], t['b'][pV,hA], optimize=True)
    O['ab'][v,V,a,V] += scale * -1.00000000 * np.einsum('aIuA,BI->aBuA', h['ab'][v,C,a,V], t['b'][pV,hC], optimize=True)
    
    
    
    
    O['ab'][a,A,c,A] += scale * -1.00000000 * np.einsum('uIiU,VI->uViU', h['ab'][a,C,c,A], t['b'][pA,hC], optimize=True)
    O['ab'][a,A,c,A] += scale * +1.00000000 * np.einsum('uUiA,AV->uUiV', h['ab'][a,A,c,V], t['b'][pV,hA], optimize=True)
    
    
    
    
    O['ab'][c,A,c,A] += scale * -1.00000000 * np.einsum('iIjU,VI->iVjU', h['ab'][c,C,c,A], t['b'][pA,hC], optimize=True)
    O['ab'][c,A,c,A] += scale * +1.00000000 * np.einsum('iUjA,AV->iUjV', h['ab'][c,A,c,V], t['b'][pV,hA], optimize=True)
    
    
    
    
    O['ab'][v,A,c,A] += scale * -1.00000000 * np.einsum('aIiU,VI->aViU', h['ab'][v,C,c,A], t['b'][pA,hC], optimize=True)
    O['ab'][v,A,c,A] += scale * +1.00000000 * np.einsum('aUiA,AV->aUiV', h['ab'][v,A,c,V], t['b'][pV,hA], optimize=True)
    
    
    O['ab'][a,C,c,A] += scale * +1.00000000 * np.einsum('uIiA,AU->uIiU', h['ab'][a,C,c,V], t['b'][pV,hA], optimize=True)
    
    
    O['ab'][c,C,c,A] += scale * +1.00000000 * np.einsum('iIjA,AU->iIjU', h['ab'][c,C,c,V], t['b'][pV,hA], optimize=True)
    
    
    O['ab'][v,C,c,A] += scale * +1.00000000 * np.einsum('aIiA,AU->aIiU', h['ab'][v,C,c,V], t['b'][pV,hA], optimize=True)
    O['ab'][a,V,c,A] += scale * -1.00000000 * np.einsum('UV,uUiW,AV->uAiW', eta1['b'], h['ab'][a,A,c,A], t['b'][pV,hA], optimize=True)
    
    O['ab'][a,V,c,A] += scale * -1.00000000 * np.einsum('UV,uUiW,AV->uAiW', gamma1['b'], h['ab'][a,A,c,A], t['b'][pV,hA], optimize=True)
    
    O['ab'][a,V,c,A] += scale * -1.00000000 * np.einsum('uIiU,AI->uAiU', h['ab'][a,C,c,A], t['b'][pV,hC], optimize=True)
    O['ab'][a,V,c,A] += scale * +1.00000000 * np.einsum('uAiB,BU->uAiU', h['ab'][a,V,c,V], t['b'][pV,hA], optimize=True)
    O['ab'][c,V,c,A] += scale * -1.00000000 * np.einsum('UV,iUjW,AV->iAjW', eta1['b'], h['ab'][c,A,c,A], t['b'][pV,hA], optimize=True)
    
    O['ab'][c,V,c,A] += scale * -1.00000000 * np.einsum('UV,iUjW,AV->iAjW', gamma1['b'], h['ab'][c,A,c,A], t['b'][pV,hA], optimize=True)
    
    O['ab'][c,V,c,A] += scale * -1.00000000 * np.einsum('iIjU,AI->iAjU', h['ab'][c,C,c,A], t['b'][pV,hC], optimize=True)
    O['ab'][c,V,c,A] += scale * +1.00000000 * np.einsum('iAjB,BU->iAjU', h['ab'][c,V,c,V], t['b'][pV,hA], optimize=True)
    O['ab'][v,V,c,A] += scale * -1.00000000 * np.einsum('UV,aUiW,AV->aAiW', eta1['b'], h['ab'][v,A,c,A], t['b'][pV,hA], optimize=True)
    
    O['ab'][v,V,c,A] += scale * -1.00000000 * np.einsum('UV,aUiW,AV->aAiW', gamma1['b'], h['ab'][v,A,c,A], t['b'][pV,hA], optimize=True)
    
    O['ab'][v,V,c,A] += scale * -1.00000000 * np.einsum('aIiU,AI->aAiU', h['ab'][v,C,c,A], t['b'][pV,hC], optimize=True)
    O['ab'][v,V,c,A] += scale * +1.00000000 * np.einsum('aAiB,BU->aAiU', h['ab'][v,V,c,V], t['b'][pV,hA], optimize=True)
    O['ab'][a,A,c,C] += scale * +1.00000000 * np.einsum('UV,uWiV,UI->uWiI', eta1['b'], h['ab'][a,A,c,A], t['b'][pA,hC], optimize=True)
    
    O['ab'][a,A,c,C] += scale * +1.00000000 * np.einsum('UV,uWiV,UI->uWiI', gamma1['b'], h['ab'][a,A,c,A], t['b'][pA,hC], optimize=True)
    
    O['ab'][a,A,c,C] += scale * -1.00000000 * np.einsum('uIiJ,UI->uUiJ', h['ab'][a,C,c,C], t['b'][pA,hC], optimize=True)
    O['ab'][a,A,c,C] += scale * +1.00000000 * np.einsum('uUiA,AI->uUiI', h['ab'][a,A,c,V], t['b'][pV,hC], optimize=True)
    O['ab'][c,A,c,C] += scale * +1.00000000 * np.einsum('UV,iWjV,UI->iWjI', eta1['b'], h['ab'][c,A,c,A], t['b'][pA,hC], optimize=True)
    
    O['ab'][c,A,c,C] += scale * +1.00000000 * np.einsum('UV,iWjV,UI->iWjI', gamma1['b'], h['ab'][c,A,c,A], t['b'][pA,hC], optimize=True)
    
    O['ab'][c,A,c,C] += scale * -1.00000000 * np.einsum('iIjJ,UI->iUjJ', h['ab'][c,C,c,C], t['b'][pA,hC], optimize=True)
    O['ab'][c,A,c,C] += scale * +1.00000000 * np.einsum('iUjA,AI->iUjI', h['ab'][c,A,c,V], t['b'][pV,hC], optimize=True)
    O['ab'][v,A,c,C] += scale * +1.00000000 * np.einsum('UV,aWiV,UI->aWiI', eta1['b'], h['ab'][v,A,c,A], t['b'][pA,hC], optimize=True)
    
    O['ab'][v,A,c,C] += scale * +1.00000000 * np.einsum('UV,aWiV,UI->aWiI', gamma1['b'], h['ab'][v,A,c,A], t['b'][pA,hC], optimize=True)
    
    O['ab'][v,A,c,C] += scale * -1.00000000 * np.einsum('aIiJ,UI->aUiJ', h['ab'][v,C,c,C], t['b'][pA,hC], optimize=True)
    O['ab'][v,A,c,C] += scale * +1.00000000 * np.einsum('aUiA,AI->aUiI', h['ab'][v,A,c,V], t['b'][pV,hC], optimize=True)
    O['ab'][a,C,c,C] += scale * +1.00000000 * np.einsum('UV,uIiV,UJ->uIiJ', eta1['b'], h['ab'][a,C,c,A], t['b'][pA,hC], optimize=True)
    O['ab'][a,C,c,C] += scale * +1.00000000 * np.einsum('UV,uIiV,UJ->uIiJ', gamma1['b'], h['ab'][a,C,c,A], t['b'][pA,hC], optimize=True)
    O['ab'][a,C,c,C] += scale * +1.00000000 * np.einsum('uIiA,AJ->uIiJ', h['ab'][a,C,c,V], t['b'][pV,hC], optimize=True)
    O['ab'][c,C,c,C] += scale * +1.00000000 * np.einsum('UV,iIjV,UJ->iIjJ', eta1['b'], h['ab'][c,C,c,A], t['b'][pA,hC], optimize=True)
    O['ab'][c,C,c,C] += scale * +1.00000000 * np.einsum('UV,iIjV,UJ->iIjJ', gamma1['b'], h['ab'][c,C,c,A], t['b'][pA,hC], optimize=True)
    O['ab'][c,C,c,C] += scale * +1.00000000 * np.einsum('iIjA,AJ->iIjJ', h['ab'][c,C,c,V], t['b'][pV,hC], optimize=True)
    O['ab'][v,C,c,C] += scale * +1.00000000 * np.einsum('UV,aIiV,UJ->aIiJ', eta1['b'], h['ab'][v,C,c,A], t['b'][pA,hC], optimize=True)
    O['ab'][v,C,c,C] += scale * +1.00000000 * np.einsum('UV,aIiV,UJ->aIiJ', gamma1['b'], h['ab'][v,C,c,A], t['b'][pA,hC], optimize=True)
    O['ab'][v,C,c,C] += scale * +1.00000000 * np.einsum('aIiA,AJ->aIiJ', h['ab'][v,C,c,V], t['b'][pV,hC], optimize=True)
    O['ab'][a,V,c,C] += scale * -1.00000000 * np.einsum('UV,uUiI,AV->uAiI', eta1['b'], h['ab'][a,A,c,C], t['b'][pV,hA], optimize=True)
    O['ab'][a,V,c,C] += scale * +1.00000000 * np.einsum('UV,uAiV,UI->uAiI', eta1['b'], h['ab'][a,V,c,A], t['b'][pA,hC], optimize=True)
    O['ab'][a,V,c,C] += scale * -1.00000000 * np.einsum('UV,uUiI,AV->uAiI', gamma1['b'], h['ab'][a,A,c,C], t['b'][pV,hA], optimize=True)
    O['ab'][a,V,c,C] += scale * +1.00000000 * np.einsum('UV,uAiV,UI->uAiI', gamma1['b'], h['ab'][a,V,c,A], t['b'][pA,hC], optimize=True)
    O['ab'][a,V,c,C] += scale * -1.00000000 * np.einsum('uIiJ,AI->uAiJ', h['ab'][a,C,c,C], t['b'][pV,hC], optimize=True)
    O['ab'][a,V,c,C] += scale * +1.00000000 * np.einsum('uAiB,BI->uAiI', h['ab'][a,V,c,V], t['b'][pV,hC], optimize=True)
    O['ab'][c,V,c,C] += scale * -1.00000000 * np.einsum('UV,iUjI,AV->iAjI', eta1['b'], h['ab'][c,A,c,C], t['b'][pV,hA], optimize=True)
    O['ab'][c,V,c,C] += scale * +1.00000000 * np.einsum('UV,iAjV,UI->iAjI', eta1['b'], h['ab'][c,V,c,A], t['b'][pA,hC], optimize=True)
    O['ab'][c,V,c,C] += scale * -1.00000000 * np.einsum('UV,iUjI,AV->iAjI', gamma1['b'], h['ab'][c,A,c,C], t['b'][pV,hA], optimize=True)
    O['ab'][c,V,c,C] += scale * +1.00000000 * np.einsum('UV,iAjV,UI->iAjI', gamma1['b'], h['ab'][c,V,c,A], t['b'][pA,hC], optimize=True)
    O['ab'][c,V,c,C] += scale * -1.00000000 * np.einsum('iIjJ,AI->iAjJ', h['ab'][c,C,c,C], t['b'][pV,hC], optimize=True)
    O['ab'][c,V,c,C] += scale * +1.00000000 * np.einsum('iAjB,BI->iAjI', h['ab'][c,V,c,V], t['b'][pV,hC], optimize=True)
    O['ab'][v,V,c,C] += scale * -1.00000000 * np.einsum('UV,aUiI,AV->aAiI', eta1['b'], h['ab'][v,A,c,C], t['b'][pV,hA], optimize=True)
    O['ab'][v,V,c,C] += scale * +1.00000000 * np.einsum('UV,aAiV,UI->aAiI', eta1['b'], h['ab'][v,V,c,A], t['b'][pA,hC], optimize=True)
    O['ab'][v,V,c,C] += scale * -1.00000000 * np.einsum('UV,aUiI,AV->aAiI', gamma1['b'], h['ab'][v,A,c,C], t['b'][pV,hA], optimize=True)
    O['ab'][v,V,c,C] += scale * +1.00000000 * np.einsum('UV,aAiV,UI->aAiI', gamma1['b'], h['ab'][v,V,c,A], t['b'][pA,hC], optimize=True)
    O['ab'][v,V,c,C] += scale * -1.00000000 * np.einsum('aIiJ,AI->aAiJ', h['ab'][v,C,c,C], t['b'][pV,hC], optimize=True)
    O['ab'][v,V,c,C] += scale * +1.00000000 * np.einsum('aAiB,BI->aAiI', h['ab'][v,V,c,V], t['b'][pV,hC], optimize=True)
    
    
    O['ab'][a,A,c,V] += scale * -1.00000000 * np.einsum('uIiA,UI->uUiA', h['ab'][a,C,c,V], t['b'][pA,hC], optimize=True)
    
    
    O['ab'][c,A,c,V] += scale * -1.00000000 * np.einsum('iIjA,UI->iUjA', h['ab'][c,C,c,V], t['b'][pA,hC], optimize=True)
    
    
    O['ab'][v,A,c,V] += scale * -1.00000000 * np.einsum('aIiA,UI->aUiA', h['ab'][v,C,c,V], t['b'][pA,hC], optimize=True)
    O['ab'][a,V,c,V] += scale * -1.00000000 * np.einsum('UV,uUiA,BV->uBiA', eta1['b'], h['ab'][a,A,c,V], t['b'][pV,hA], optimize=True)
    O['ab'][a,V,c,V] += scale * -1.00000000 * np.einsum('UV,uUiA,BV->uBiA', gamma1['b'], h['ab'][a,A,c,V], t['b'][pV,hA], optimize=True)
    O['ab'][a,V,c,V] += scale * -1.00000000 * np.einsum('uIiA,BI->uBiA', h['ab'][a,C,c,V], t['b'][pV,hC], optimize=True)
    O['ab'][c,V,c,V] += scale * -1.00000000 * np.einsum('UV,iUjA,BV->iBjA', eta1['b'], h['ab'][c,A,c,V], t['b'][pV,hA], optimize=True)
    O['ab'][c,V,c,V] += scale * -1.00000000 * np.einsum('UV,iUjA,BV->iBjA', gamma1['b'], h['ab'][c,A,c,V], t['b'][pV,hA], optimize=True)
    O['ab'][c,V,c,V] += scale * -1.00000000 * np.einsum('iIjA,BI->iBjA', h['ab'][c,C,c,V], t['b'][pV,hC], optimize=True)
    O['ab'][v,V,c,V] += scale * -1.00000000 * np.einsum('UV,aUiA,BV->aBiA', eta1['b'], h['ab'][v,A,c,V], t['b'][pV,hA], optimize=True)
    O['ab'][v,V,c,V] += scale * -1.00000000 * np.einsum('UV,aUiA,BV->aBiA', gamma1['b'], h['ab'][v,A,c,V], t['b'][pV,hA], optimize=True)
    O['ab'][v,V,c,V] += scale * -1.00000000 * np.einsum('aIiA,BI->aBiA', h['ab'][v,C,c,V], t['b'][pV,hC], optimize=True)
    
    
    
    
    O['ab'][a,A,v,A] += scale * -1.00000000 * np.einsum('uIaU,VI->uVaU', h['ab'][a,C,v,A], t['b'][pA,hC], optimize=True)
    O['ab'][a,A,v,A] += scale * +1.00000000 * np.einsum('uUaA,AV->uUaV', h['ab'][a,A,v,V], t['b'][pV,hA], optimize=True)
    
    
    
    
    O['ab'][c,A,v,A] += scale * -1.00000000 * np.einsum('iIaU,VI->iVaU', h['ab'][c,C,v,A], t['b'][pA,hC], optimize=True)
    O['ab'][c,A,v,A] += scale * +1.00000000 * np.einsum('iUaA,AV->iUaV', h['ab'][c,A,v,V], t['b'][pV,hA], optimize=True)
    
    
    
    
    O['ab'][v,A,v,A] += scale * -1.00000000 * np.einsum('aIbU,VI->aVbU', h['ab'][v,C,v,A], t['b'][pA,hC], optimize=True)
    O['ab'][v,A,v,A] += scale * +1.00000000 * np.einsum('aUbA,AV->aUbV', h['ab'][v,A,v,V], t['b'][pV,hA], optimize=True)
    
    
    O['ab'][a,C,v,A] += scale * +1.00000000 * np.einsum('uIaA,AU->uIaU', h['ab'][a,C,v,V], t['b'][pV,hA], optimize=True)
    
    
    O['ab'][c,C,v,A] += scale * +1.00000000 * np.einsum('iIaA,AU->iIaU', h['ab'][c,C,v,V], t['b'][pV,hA], optimize=True)
    
    
    O['ab'][v,C,v,A] += scale * +1.00000000 * np.einsum('aIbA,AU->aIbU', h['ab'][v,C,v,V], t['b'][pV,hA], optimize=True)
    O['ab'][a,V,v,A] += scale * -1.00000000 * np.einsum('UV,uUaW,AV->uAaW', eta1['b'], h['ab'][a,A,v,A], t['b'][pV,hA], optimize=True)
    
    O['ab'][a,V,v,A] += scale * -1.00000000 * np.einsum('UV,uUaW,AV->uAaW', gamma1['b'], h['ab'][a,A,v,A], t['b'][pV,hA], optimize=True)
    
    O['ab'][a,V,v,A] += scale * -1.00000000 * np.einsum('uIaU,AI->uAaU', h['ab'][a,C,v,A], t['b'][pV,hC], optimize=True)
    O['ab'][a,V,v,A] += scale * +1.00000000 * np.einsum('uAaB,BU->uAaU', h['ab'][a,V,v,V], t['b'][pV,hA], optimize=True)
    O['ab'][c,V,v,A] += scale * -1.00000000 * np.einsum('UV,iUaW,AV->iAaW', eta1['b'], h['ab'][c,A,v,A], t['b'][pV,hA], optimize=True)
    
    O['ab'][c,V,v,A] += scale * -1.00000000 * np.einsum('UV,iUaW,AV->iAaW', gamma1['b'], h['ab'][c,A,v,A], t['b'][pV,hA], optimize=True)
    
    O['ab'][c,V,v,A] += scale * -1.00000000 * np.einsum('iIaU,AI->iAaU', h['ab'][c,C,v,A], t['b'][pV,hC], optimize=True)
    O['ab'][c,V,v,A] += scale * +1.00000000 * np.einsum('iAaB,BU->iAaU', h['ab'][c,V,v,V], t['b'][pV,hA], optimize=True)
    O['ab'][v,V,v,A] += scale * -1.00000000 * np.einsum('UV,aUbW,AV->aAbW', eta1['b'], h['ab'][v,A,v,A], t['b'][pV,hA], optimize=True)
    
    O['ab'][v,V,v,A] += scale * -1.00000000 * np.einsum('UV,aUbW,AV->aAbW', gamma1['b'], h['ab'][v,A,v,A], t['b'][pV,hA], optimize=True)
    
    O['ab'][v,V,v,A] += scale * -1.00000000 * np.einsum('aIbU,AI->aAbU', h['ab'][v,C,v,A], t['b'][pV,hC], optimize=True)
    O['ab'][v,V,v,A] += scale * +1.00000000 * np.einsum('aAbB,BU->aAbU', h['ab'][v,V,v,V], t['b'][pV,hA], optimize=True)
    O['ab'][a,A,v,C] += scale * +1.00000000 * np.einsum('UV,uWaV,UI->uWaI', eta1['b'], h['ab'][a,A,v,A], t['b'][pA,hC], optimize=True)
    
    O['ab'][a,A,v,C] += scale * +1.00000000 * np.einsum('UV,uWaV,UI->uWaI', gamma1['b'], h['ab'][a,A,v,A], t['b'][pA,hC], optimize=True)
    
    O['ab'][a,A,v,C] += scale * -1.00000000 * np.einsum('uIaJ,UI->uUaJ', h['ab'][a,C,v,C], t['b'][pA,hC], optimize=True)
    O['ab'][a,A,v,C] += scale * +1.00000000 * np.einsum('uUaA,AI->uUaI', h['ab'][a,A,v,V], t['b'][pV,hC], optimize=True)
    O['ab'][c,A,v,C] += scale * +1.00000000 * np.einsum('UV,iWaV,UI->iWaI', eta1['b'], h['ab'][c,A,v,A], t['b'][pA,hC], optimize=True)
    
    O['ab'][c,A,v,C] += scale * +1.00000000 * np.einsum('UV,iWaV,UI->iWaI', gamma1['b'], h['ab'][c,A,v,A], t['b'][pA,hC], optimize=True)
    
    O['ab'][c,A,v,C] += scale * -1.00000000 * np.einsum('iIaJ,UI->iUaJ', h['ab'][c,C,v,C], t['b'][pA,hC], optimize=True)
    O['ab'][c,A,v,C] += scale * +1.00000000 * np.einsum('iUaA,AI->iUaI', h['ab'][c,A,v,V], t['b'][pV,hC], optimize=True)
    O['ab'][v,A,v,C] += scale * +1.00000000 * np.einsum('UV,aWbV,UI->aWbI', eta1['b'], h['ab'][v,A,v,A], t['b'][pA,hC], optimize=True)
    
    O['ab'][v,A,v,C] += scale * +1.00000000 * np.einsum('UV,aWbV,UI->aWbI', gamma1['b'], h['ab'][v,A,v,A], t['b'][pA,hC], optimize=True)
    
    O['ab'][v,A,v,C] += scale * -1.00000000 * np.einsum('aIbJ,UI->aUbJ', h['ab'][v,C,v,C], t['b'][pA,hC], optimize=True)
    O['ab'][v,A,v,C] += scale * +1.00000000 * np.einsum('aUbA,AI->aUbI', h['ab'][v,A,v,V], t['b'][pV,hC], optimize=True)
    O['ab'][a,C,v,C] += scale * +1.00000000 * np.einsum('UV,uIaV,UJ->uIaJ', eta1['b'], h['ab'][a,C,v,A], t['b'][pA,hC], optimize=True)
    O['ab'][a,C,v,C] += scale * +1.00000000 * np.einsum('UV,uIaV,UJ->uIaJ', gamma1['b'], h['ab'][a,C,v,A], t['b'][pA,hC], optimize=True)
    O['ab'][a,C,v,C] += scale * +1.00000000 * np.einsum('uIaA,AJ->uIaJ', h['ab'][a,C,v,V], t['b'][pV,hC], optimize=True)
    O['ab'][c,C,v,C] += scale * +1.00000000 * np.einsum('UV,iIaV,UJ->iIaJ', eta1['b'], h['ab'][c,C,v,A], t['b'][pA,hC], optimize=True)
    O['ab'][c,C,v,C] += scale * +1.00000000 * np.einsum('UV,iIaV,UJ->iIaJ', gamma1['b'], h['ab'][c,C,v,A], t['b'][pA,hC], optimize=True)
    O['ab'][c,C,v,C] += scale * +1.00000000 * np.einsum('iIaA,AJ->iIaJ', h['ab'][c,C,v,V], t['b'][pV,hC], optimize=True)
    O['ab'][v,C,v,C] += scale * +1.00000000 * np.einsum('UV,aIbV,UJ->aIbJ', eta1['b'], h['ab'][v,C,v,A], t['b'][pA,hC], optimize=True)
    O['ab'][v,C,v,C] += scale * +1.00000000 * np.einsum('UV,aIbV,UJ->aIbJ', gamma1['b'], h['ab'][v,C,v,A], t['b'][pA,hC], optimize=True)
    O['ab'][v,C,v,C] += scale * +1.00000000 * np.einsum('aIbA,AJ->aIbJ', h['ab'][v,C,v,V], t['b'][pV,hC], optimize=True)
    O['ab'][a,V,v,C] += scale * -1.00000000 * np.einsum('UV,uUaI,AV->uAaI', eta1['b'], h['ab'][a,A,v,C], t['b'][pV,hA], optimize=True)
    O['ab'][a,V,v,C] += scale * +1.00000000 * np.einsum('UV,uAaV,UI->uAaI', eta1['b'], h['ab'][a,V,v,A], t['b'][pA,hC], optimize=True)
    O['ab'][a,V,v,C] += scale * -1.00000000 * np.einsum('UV,uUaI,AV->uAaI', gamma1['b'], h['ab'][a,A,v,C], t['b'][pV,hA], optimize=True)
    O['ab'][a,V,v,C] += scale * +1.00000000 * np.einsum('UV,uAaV,UI->uAaI', gamma1['b'], h['ab'][a,V,v,A], t['b'][pA,hC], optimize=True)
    O['ab'][a,V,v,C] += scale * -1.00000000 * np.einsum('uIaJ,AI->uAaJ', h['ab'][a,C,v,C], t['b'][pV,hC], optimize=True)
    O['ab'][a,V,v,C] += scale * +1.00000000 * np.einsum('uAaB,BI->uAaI', h['ab'][a,V,v,V], t['b'][pV,hC], optimize=True)
    O['ab'][c,V,v,C] += scale * -1.00000000 * np.einsum('UV,iUaI,AV->iAaI', eta1['b'], h['ab'][c,A,v,C], t['b'][pV,hA], optimize=True)
    O['ab'][c,V,v,C] += scale * +1.00000000 * np.einsum('UV,iAaV,UI->iAaI', eta1['b'], h['ab'][c,V,v,A], t['b'][pA,hC], optimize=True)
    O['ab'][c,V,v,C] += scale * -1.00000000 * np.einsum('UV,iUaI,AV->iAaI', gamma1['b'], h['ab'][c,A,v,C], t['b'][pV,hA], optimize=True)
    O['ab'][c,V,v,C] += scale * +1.00000000 * np.einsum('UV,iAaV,UI->iAaI', gamma1['b'], h['ab'][c,V,v,A], t['b'][pA,hC], optimize=True)
    O['ab'][c,V,v,C] += scale * -1.00000000 * np.einsum('iIaJ,AI->iAaJ', h['ab'][c,C,v,C], t['b'][pV,hC], optimize=True)
    O['ab'][c,V,v,C] += scale * +1.00000000 * np.einsum('iAaB,BI->iAaI', h['ab'][c,V,v,V], t['b'][pV,hC], optimize=True)
    O['ab'][v,V,v,C] += scale * -1.00000000 * np.einsum('UV,aUbI,AV->aAbI', eta1['b'], h['ab'][v,A,v,C], t['b'][pV,hA], optimize=True)
    O['ab'][v,V,v,C] += scale * +1.00000000 * np.einsum('UV,aAbV,UI->aAbI', eta1['b'], h['ab'][v,V,v,A], t['b'][pA,hC], optimize=True)
    O['ab'][v,V,v,C] += scale * -1.00000000 * np.einsum('UV,aUbI,AV->aAbI', gamma1['b'], h['ab'][v,A,v,C], t['b'][pV,hA], optimize=True)
    O['ab'][v,V,v,C] += scale * +1.00000000 * np.einsum('UV,aAbV,UI->aAbI', gamma1['b'], h['ab'][v,V,v,A], t['b'][pA,hC], optimize=True)
    O['ab'][v,V,v,C] += scale * -1.00000000 * np.einsum('aIbJ,AI->aAbJ', h['ab'][v,C,v,C], t['b'][pV,hC], optimize=True)
    O['ab'][v,V,v,C] += scale * +1.00000000 * np.einsum('aAbB,BI->aAbI', h['ab'][v,V,v,V], t['b'][pV,hC], optimize=True)
    
    
    O['ab'][a,A,v,V] += scale * -1.00000000 * np.einsum('uIaA,UI->uUaA', h['ab'][a,C,v,V], t['b'][pA,hC], optimize=True)
    
    
    O['ab'][c,A,v,V] += scale * -1.00000000 * np.einsum('iIaA,UI->iUaA', h['ab'][c,C,v,V], t['b'][pA,hC], optimize=True)
    
    
    O['ab'][v,A,v,V] += scale * -1.00000000 * np.einsum('aIbA,UI->aUbA', h['ab'][v,C,v,V], t['b'][pA,hC], optimize=True)
    O['ab'][a,V,v,V] += scale * -1.00000000 * np.einsum('UV,uUaA,BV->uBaA', eta1['b'], h['ab'][a,A,v,V], t['b'][pV,hA], optimize=True)
    O['ab'][a,V,v,V] += scale * -1.00000000 * np.einsum('UV,uUaA,BV->uBaA', gamma1['b'], h['ab'][a,A,v,V], t['b'][pV,hA], optimize=True)
    O['ab'][a,V,v,V] += scale * -1.00000000 * np.einsum('uIaA,BI->uBaA', h['ab'][a,C,v,V], t['b'][pV,hC], optimize=True)
    O['ab'][c,V,v,V] += scale * -1.00000000 * np.einsum('UV,iUaA,BV->iBaA', eta1['b'], h['ab'][c,A,v,V], t['b'][pV,hA], optimize=True)
    O['ab'][c,V,v,V] += scale * -1.00000000 * np.einsum('UV,iUaA,BV->iBaA', gamma1['b'], h['ab'][c,A,v,V], t['b'][pV,hA], optimize=True)
    O['ab'][c,V,v,V] += scale * -1.00000000 * np.einsum('iIaA,BI->iBaA', h['ab'][c,C,v,V], t['b'][pV,hC], optimize=True)
    O['ab'][v,V,v,V] += scale * -1.00000000 * np.einsum('UV,aUbA,BV->aBbA', eta1['b'], h['ab'][v,A,v,V], t['b'][pV,hA], optimize=True)
    O['ab'][v,V,v,V] += scale * -1.00000000 * np.einsum('UV,aUbA,BV->aBbA', gamma1['b'], h['ab'][v,A,v,V], t['b'][pV,hA], optimize=True)
    O['ab'][v,V,v,V] += scale * -1.00000000 * np.einsum('aIbA,BI->aBbA', h['ab'][v,C,v,V], t['b'][pV,hC], optimize=True)

    t1 = time.time()
    if verbose: print("h2b_t1b_c2b took {:.4f} seconds to run.".format(t1-t0))

    return O
def h2b_t2a_c2b(O, h, t, gamma1, eta1, lambdas, orbspace, verbose=False, scale=1.0):
    # 180 lines
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

    
    
    O['ab'][a,A,a,A] += scale * -1.00000000 * np.einsum('uv,iUvV,wuix->wUxV', eta1['a'], h['ab'][c,A,a,A], t['aa'][pa,pa,hc,ha], optimize=True)
    O['ab'][a,A,a,A] += scale * +1.00000000 * np.einsum('uv,uUaV,waxv->wUxV', gamma1['a'], h['ab'][a,A,v,A], t['aa'][pa,pv,ha,ha], optimize=True)
    O['ab'][a,A,a,A] += scale * -1.00000000 * np.einsum('iUaV,uaiv->uUvV', h['ab'][c,A,v,A], t['aa'][pa,pv,hc,ha], optimize=True)
    O['ab'][v,A,a,A] += scale * +1.00000000 * np.einsum('uv,wx,uUxV,wayv->aUyV', eta1['a'], gamma1['a'], h['ab'][a,A,a,A], t['aa'][pa,pv,ha,ha], optimize=True)
    O['ab'][v,A,a,A] += scale * -1.00000000 * np.einsum('uv,wx,wUvV,uayx->aUyV', eta1['a'], gamma1['a'], h['ab'][a,A,a,A], t['aa'][pa,pv,ha,ha], optimize=True)
    O['ab'][v,A,a,A] += scale * +1.00000000 * np.einsum('uv,iUvV,uaiw->aUwV', eta1['a'], h['ab'][c,A,a,A], t['aa'][pa,pv,hc,ha], optimize=True)
    O['ab'][v,A,a,A] += scale * +1.00000000 * np.einsum('uv,uUaV,bawv->bUwV', gamma1['a'], h['ab'][a,A,v,A], t['aa'][pv,pv,ha,ha], optimize=True)
    O['ab'][v,A,a,A] += scale * -1.00000000 * np.einsum('iUaV,baiu->bUuV', h['ab'][c,A,v,A], t['aa'][pv,pv,hc,ha], optimize=True)
    
    
    O['ab'][a,C,a,A] += scale * -1.00000000 * np.einsum('uv,iIvU,wuix->wIxU', eta1['a'], h['ab'][c,C,a,A], t['aa'][pa,pa,hc,ha], optimize=True)
    O['ab'][a,C,a,A] += scale * +1.00000000 * np.einsum('uv,uIaU,waxv->wIxU', gamma1['a'], h['ab'][a,C,v,A], t['aa'][pa,pv,ha,ha], optimize=True)
    O['ab'][a,C,a,A] += scale * -1.00000000 * np.einsum('iIaU,uaiv->uIvU', h['ab'][c,C,v,A], t['aa'][pa,pv,hc,ha], optimize=True)
    O['ab'][v,C,a,A] += scale * +1.00000000 * np.einsum('uv,wx,uIxU,wayv->aIyU', eta1['a'], gamma1['a'], h['ab'][a,C,a,A], t['aa'][pa,pv,ha,ha], optimize=True)
    O['ab'][v,C,a,A] += scale * -1.00000000 * np.einsum('uv,wx,wIvU,uayx->aIyU', eta1['a'], gamma1['a'], h['ab'][a,C,a,A], t['aa'][pa,pv,ha,ha], optimize=True)
    O['ab'][v,C,a,A] += scale * +1.00000000 * np.einsum('uv,iIvU,uaiw->aIwU', eta1['a'], h['ab'][c,C,a,A], t['aa'][pa,pv,hc,ha], optimize=True)
    O['ab'][v,C,a,A] += scale * +1.00000000 * np.einsum('uv,uIaU,bawv->bIwU', gamma1['a'], h['ab'][a,C,v,A], t['aa'][pv,pv,ha,ha], optimize=True)
    O['ab'][v,C,a,A] += scale * -1.00000000 * np.einsum('iIaU,baiu->bIuU', h['ab'][c,C,v,A], t['aa'][pv,pv,hc,ha], optimize=True)
    
    
    O['ab'][a,V,a,A] += scale * -1.00000000 * np.einsum('uv,iAvU,wuix->wAxU', eta1['a'], h['ab'][c,V,a,A], t['aa'][pa,pa,hc,ha], optimize=True)
    O['ab'][a,V,a,A] += scale * +1.00000000 * np.einsum('uv,uAaU,waxv->wAxU', gamma1['a'], h['ab'][a,V,v,A], t['aa'][pa,pv,ha,ha], optimize=True)
    O['ab'][a,V,a,A] += scale * -1.00000000 * np.einsum('iAaU,uaiv->uAvU', h['ab'][c,V,v,A], t['aa'][pa,pv,hc,ha], optimize=True)
    O['ab'][v,V,a,A] += scale * +1.00000000 * np.einsum('uv,wx,uAxU,wayv->aAyU', eta1['a'], gamma1['a'], h['ab'][a,V,a,A], t['aa'][pa,pv,ha,ha], optimize=True)
    O['ab'][v,V,a,A] += scale * -1.00000000 * np.einsum('uv,wx,wAvU,uayx->aAyU', eta1['a'], gamma1['a'], h['ab'][a,V,a,A], t['aa'][pa,pv,ha,ha], optimize=True)
    O['ab'][v,V,a,A] += scale * +1.00000000 * np.einsum('uv,iAvU,uaiw->aAwU', eta1['a'], h['ab'][c,V,a,A], t['aa'][pa,pv,hc,ha], optimize=True)
    O['ab'][v,V,a,A] += scale * +1.00000000 * np.einsum('uv,uAaU,bawv->bAwU', gamma1['a'], h['ab'][a,V,v,A], t['aa'][pv,pv,ha,ha], optimize=True)
    O['ab'][v,V,a,A] += scale * -1.00000000 * np.einsum('iAaU,baiu->bAuU', h['ab'][c,V,v,A], t['aa'][pv,pv,hc,ha], optimize=True)
    
    
    O['ab'][a,A,a,C] += scale * -1.00000000 * np.einsum('uv,iUvI,wuix->wUxI', eta1['a'], h['ab'][c,A,a,C], t['aa'][pa,pa,hc,ha], optimize=True)
    O['ab'][a,A,a,C] += scale * +1.00000000 * np.einsum('uv,uUaI,waxv->wUxI', gamma1['a'], h['ab'][a,A,v,C], t['aa'][pa,pv,ha,ha], optimize=True)
    O['ab'][a,A,a,C] += scale * -1.00000000 * np.einsum('iUaI,uaiv->uUvI', h['ab'][c,A,v,C], t['aa'][pa,pv,hc,ha], optimize=True)
    O['ab'][v,A,a,C] += scale * +1.00000000 * np.einsum('uv,wx,uUxI,wayv->aUyI', eta1['a'], gamma1['a'], h['ab'][a,A,a,C], t['aa'][pa,pv,ha,ha], optimize=True)
    O['ab'][v,A,a,C] += scale * -1.00000000 * np.einsum('uv,wx,wUvI,uayx->aUyI', eta1['a'], gamma1['a'], h['ab'][a,A,a,C], t['aa'][pa,pv,ha,ha], optimize=True)
    O['ab'][v,A,a,C] += scale * +1.00000000 * np.einsum('uv,iUvI,uaiw->aUwI', eta1['a'], h['ab'][c,A,a,C], t['aa'][pa,pv,hc,ha], optimize=True)
    O['ab'][v,A,a,C] += scale * +1.00000000 * np.einsum('uv,uUaI,bawv->bUwI', gamma1['a'], h['ab'][a,A,v,C], t['aa'][pv,pv,ha,ha], optimize=True)
    O['ab'][v,A,a,C] += scale * -1.00000000 * np.einsum('iUaI,baiu->bUuI', h['ab'][c,A,v,C], t['aa'][pv,pv,hc,ha], optimize=True)
    
    
    O['ab'][a,C,a,C] += scale * -1.00000000 * np.einsum('uv,iIvJ,wuix->wIxJ', eta1['a'], h['ab'][c,C,a,C], t['aa'][pa,pa,hc,ha], optimize=True)
    O['ab'][a,C,a,C] += scale * +1.00000000 * np.einsum('uv,uIaJ,waxv->wIxJ', gamma1['a'], h['ab'][a,C,v,C], t['aa'][pa,pv,ha,ha], optimize=True)
    O['ab'][a,C,a,C] += scale * -1.00000000 * np.einsum('iIaJ,uaiv->uIvJ', h['ab'][c,C,v,C], t['aa'][pa,pv,hc,ha], optimize=True)
    O['ab'][v,C,a,C] += scale * +1.00000000 * np.einsum('uv,wx,uIxJ,wayv->aIyJ', eta1['a'], gamma1['a'], h['ab'][a,C,a,C], t['aa'][pa,pv,ha,ha], optimize=True)
    O['ab'][v,C,a,C] += scale * -1.00000000 * np.einsum('uv,wx,wIvJ,uayx->aIyJ', eta1['a'], gamma1['a'], h['ab'][a,C,a,C], t['aa'][pa,pv,ha,ha], optimize=True)
    O['ab'][v,C,a,C] += scale * +1.00000000 * np.einsum('uv,iIvJ,uaiw->aIwJ', eta1['a'], h['ab'][c,C,a,C], t['aa'][pa,pv,hc,ha], optimize=True)
    O['ab'][v,C,a,C] += scale * +1.00000000 * np.einsum('uv,uIaJ,bawv->bIwJ', gamma1['a'], h['ab'][a,C,v,C], t['aa'][pv,pv,ha,ha], optimize=True)
    O['ab'][v,C,a,C] += scale * -1.00000000 * np.einsum('iIaJ,baiu->bIuJ', h['ab'][c,C,v,C], t['aa'][pv,pv,hc,ha], optimize=True)
    
    
    O['ab'][a,V,a,C] += scale * -1.00000000 * np.einsum('uv,iAvI,wuix->wAxI', eta1['a'], h['ab'][c,V,a,C], t['aa'][pa,pa,hc,ha], optimize=True)
    O['ab'][a,V,a,C] += scale * +1.00000000 * np.einsum('uv,uAaI,waxv->wAxI', gamma1['a'], h['ab'][a,V,v,C], t['aa'][pa,pv,ha,ha], optimize=True)
    O['ab'][a,V,a,C] += scale * -1.00000000 * np.einsum('iAaI,uaiv->uAvI', h['ab'][c,V,v,C], t['aa'][pa,pv,hc,ha], optimize=True)
    O['ab'][v,V,a,C] += scale * +1.00000000 * np.einsum('uv,wx,uAxI,wayv->aAyI', eta1['a'], gamma1['a'], h['ab'][a,V,a,C], t['aa'][pa,pv,ha,ha], optimize=True)
    O['ab'][v,V,a,C] += scale * -1.00000000 * np.einsum('uv,wx,wAvI,uayx->aAyI', eta1['a'], gamma1['a'], h['ab'][a,V,a,C], t['aa'][pa,pv,ha,ha], optimize=True)
    O['ab'][v,V,a,C] += scale * +1.00000000 * np.einsum('uv,iAvI,uaiw->aAwI', eta1['a'], h['ab'][c,V,a,C], t['aa'][pa,pv,hc,ha], optimize=True)
    O['ab'][v,V,a,C] += scale * +1.00000000 * np.einsum('uv,uAaI,bawv->bAwI', gamma1['a'], h['ab'][a,V,v,C], t['aa'][pv,pv,ha,ha], optimize=True)
    O['ab'][v,V,a,C] += scale * -1.00000000 * np.einsum('iAaI,baiu->bAuI', h['ab'][c,V,v,C], t['aa'][pv,pv,hc,ha], optimize=True)
    
    
    O['ab'][a,A,a,V] += scale * -1.00000000 * np.einsum('uv,iUvA,wuix->wUxA', eta1['a'], h['ab'][c,A,a,V], t['aa'][pa,pa,hc,ha], optimize=True)
    O['ab'][a,A,a,V] += scale * +1.00000000 * np.einsum('uv,uUaA,waxv->wUxA', gamma1['a'], h['ab'][a,A,v,V], t['aa'][pa,pv,ha,ha], optimize=True)
    O['ab'][a,A,a,V] += scale * -1.00000000 * np.einsum('iUaA,uaiv->uUvA', h['ab'][c,A,v,V], t['aa'][pa,pv,hc,ha], optimize=True)
    O['ab'][v,A,a,V] += scale * +1.00000000 * np.einsum('uv,wx,uUxA,wayv->aUyA', eta1['a'], gamma1['a'], h['ab'][a,A,a,V], t['aa'][pa,pv,ha,ha], optimize=True)
    O['ab'][v,A,a,V] += scale * -1.00000000 * np.einsum('uv,wx,wUvA,uayx->aUyA', eta1['a'], gamma1['a'], h['ab'][a,A,a,V], t['aa'][pa,pv,ha,ha], optimize=True)
    O['ab'][v,A,a,V] += scale * +1.00000000 * np.einsum('uv,iUvA,uaiw->aUwA', eta1['a'], h['ab'][c,A,a,V], t['aa'][pa,pv,hc,ha], optimize=True)
    O['ab'][v,A,a,V] += scale * +1.00000000 * np.einsum('uv,uUaA,bawv->bUwA', gamma1['a'], h['ab'][a,A,v,V], t['aa'][pv,pv,ha,ha], optimize=True)
    O['ab'][v,A,a,V] += scale * -1.00000000 * np.einsum('iUaA,baiu->bUuA', h['ab'][c,A,v,V], t['aa'][pv,pv,hc,ha], optimize=True)
    
    
    O['ab'][a,C,a,V] += scale * -1.00000000 * np.einsum('uv,iIvA,wuix->wIxA', eta1['a'], h['ab'][c,C,a,V], t['aa'][pa,pa,hc,ha], optimize=True)
    O['ab'][a,C,a,V] += scale * +1.00000000 * np.einsum('uv,uIaA,waxv->wIxA', gamma1['a'], h['ab'][a,C,v,V], t['aa'][pa,pv,ha,ha], optimize=True)
    O['ab'][a,C,a,V] += scale * -1.00000000 * np.einsum('iIaA,uaiv->uIvA', h['ab'][c,C,v,V], t['aa'][pa,pv,hc,ha], optimize=True)
    O['ab'][v,C,a,V] += scale * +1.00000000 * np.einsum('uv,wx,uIxA,wayv->aIyA', eta1['a'], gamma1['a'], h['ab'][a,C,a,V], t['aa'][pa,pv,ha,ha], optimize=True)
    O['ab'][v,C,a,V] += scale * -1.00000000 * np.einsum('uv,wx,wIvA,uayx->aIyA', eta1['a'], gamma1['a'], h['ab'][a,C,a,V], t['aa'][pa,pv,ha,ha], optimize=True)
    O['ab'][v,C,a,V] += scale * +1.00000000 * np.einsum('uv,iIvA,uaiw->aIwA', eta1['a'], h['ab'][c,C,a,V], t['aa'][pa,pv,hc,ha], optimize=True)
    O['ab'][v,C,a,V] += scale * +1.00000000 * np.einsum('uv,uIaA,bawv->bIwA', gamma1['a'], h['ab'][a,C,v,V], t['aa'][pv,pv,ha,ha], optimize=True)
    O['ab'][v,C,a,V] += scale * -1.00000000 * np.einsum('iIaA,baiu->bIuA', h['ab'][c,C,v,V], t['aa'][pv,pv,hc,ha], optimize=True)
    
    
    O['ab'][a,V,a,V] += scale * -1.00000000 * np.einsum('uv,iAvB,wuix->wAxB', eta1['a'], h['ab'][c,V,a,V], t['aa'][pa,pa,hc,ha], optimize=True)
    O['ab'][a,V,a,V] += scale * +1.00000000 * np.einsum('uv,uAaB,waxv->wAxB', gamma1['a'], h['ab'][a,V,v,V], t['aa'][pa,pv,ha,ha], optimize=True)
    O['ab'][a,V,a,V] += scale * -1.00000000 * np.einsum('iAaB,uaiv->uAvB', h['ab'][c,V,v,V], t['aa'][pa,pv,hc,ha], optimize=True)
    O['ab'][v,V,a,V] += scale * +1.00000000 * np.einsum('uv,wx,uAxB,wayv->aAyB', eta1['a'], gamma1['a'], h['ab'][a,V,a,V], t['aa'][pa,pv,ha,ha], optimize=True)
    O['ab'][v,V,a,V] += scale * -1.00000000 * np.einsum('uv,wx,wAvB,uayx->aAyB', eta1['a'], gamma1['a'], h['ab'][a,V,a,V], t['aa'][pa,pv,ha,ha], optimize=True)
    O['ab'][v,V,a,V] += scale * +1.00000000 * np.einsum('uv,iAvB,uaiw->aAwB', eta1['a'], h['ab'][c,V,a,V], t['aa'][pa,pv,hc,ha], optimize=True)
    O['ab'][v,V,a,V] += scale * +1.00000000 * np.einsum('uv,uAaB,bawv->bAwB', gamma1['a'], h['ab'][a,V,v,V], t['aa'][pv,pv,ha,ha], optimize=True)
    O['ab'][v,V,a,V] += scale * -1.00000000 * np.einsum('iAaB,baiu->bAuB', h['ab'][c,V,v,V], t['aa'][pv,pv,hc,ha], optimize=True)
    O['ab'][a,A,c,A] += scale * -1.00000000 * np.einsum('uv,wx,uUxV,ywiv->yUiV', eta1['a'], gamma1['a'], h['ab'][a,A,a,A], t['aa'][pa,pa,hc,ha], optimize=True)
    O['ab'][a,A,c,A] += scale * +1.00000000 * np.einsum('uv,wx,wUvV,yuix->yUiV', eta1['a'], gamma1['a'], h['ab'][a,A,a,A], t['aa'][pa,pa,hc,ha], optimize=True)
    O['ab'][a,A,c,A] += scale * +1.00000000 * np.einsum('uv,iUvV,wuji->wUjV', eta1['a'], h['ab'][c,A,a,A], t['aa'][pa,pa,hc,hc], optimize=True)
    O['ab'][a,A,c,A] += scale * +1.00000000 * np.einsum('uv,uUaV,waiv->wUiV', gamma1['a'], h['ab'][a,A,v,A], t['aa'][pa,pv,hc,ha], optimize=True)
    O['ab'][a,A,c,A] += scale * +1.00000000 * np.einsum('iUaV,uaji->uUjV', h['ab'][c,A,v,A], t['aa'][pa,pv,hc,hc], optimize=True)
    O['ab'][v,A,c,A] += scale * +1.00000000 * np.einsum('uv,wx,uUxV,waiv->aUiV', eta1['a'], gamma1['a'], h['ab'][a,A,a,A], t['aa'][pa,pv,hc,ha], optimize=True)
    O['ab'][v,A,c,A] += scale * -1.00000000 * np.einsum('uv,wx,wUvV,uaix->aUiV', eta1['a'], gamma1['a'], h['ab'][a,A,a,A], t['aa'][pa,pv,hc,ha], optimize=True)
    O['ab'][v,A,c,A] += scale * -1.00000000 * np.einsum('uv,iUvV,uaji->aUjV', eta1['a'], h['ab'][c,A,a,A], t['aa'][pa,pv,hc,hc], optimize=True)
    O['ab'][v,A,c,A] += scale * +1.00000000 * np.einsum('uv,uUaV,baiv->bUiV', gamma1['a'], h['ab'][a,A,v,A], t['aa'][pv,pv,hc,ha], optimize=True)
    O['ab'][v,A,c,A] += scale * +1.00000000 * np.einsum('iUaV,baji->bUjV', h['ab'][c,A,v,A], t['aa'][pv,pv,hc,hc], optimize=True)
    O['ab'][a,C,c,A] += scale * -1.00000000 * np.einsum('uv,wx,uIxU,ywiv->yIiU', eta1['a'], gamma1['a'], h['ab'][a,C,a,A], t['aa'][pa,pa,hc,ha], optimize=True)
    O['ab'][a,C,c,A] += scale * +1.00000000 * np.einsum('uv,wx,wIvU,yuix->yIiU', eta1['a'], gamma1['a'], h['ab'][a,C,a,A], t['aa'][pa,pa,hc,ha], optimize=True)
    O['ab'][a,C,c,A] += scale * +1.00000000 * np.einsum('uv,iIvU,wuji->wIjU', eta1['a'], h['ab'][c,C,a,A], t['aa'][pa,pa,hc,hc], optimize=True)
    O['ab'][a,C,c,A] += scale * +1.00000000 * np.einsum('uv,uIaU,waiv->wIiU', gamma1['a'], h['ab'][a,C,v,A], t['aa'][pa,pv,hc,ha], optimize=True)
    O['ab'][a,C,c,A] += scale * +1.00000000 * np.einsum('iIaU,uaji->uIjU', h['ab'][c,C,v,A], t['aa'][pa,pv,hc,hc], optimize=True)
    O['ab'][v,C,c,A] += scale * +1.00000000 * np.einsum('uv,wx,uIxU,waiv->aIiU', eta1['a'], gamma1['a'], h['ab'][a,C,a,A], t['aa'][pa,pv,hc,ha], optimize=True)
    O['ab'][v,C,c,A] += scale * -1.00000000 * np.einsum('uv,wx,wIvU,uaix->aIiU', eta1['a'], gamma1['a'], h['ab'][a,C,a,A], t['aa'][pa,pv,hc,ha], optimize=True)
    O['ab'][v,C,c,A] += scale * -1.00000000 * np.einsum('uv,iIvU,uaji->aIjU', eta1['a'], h['ab'][c,C,a,A], t['aa'][pa,pv,hc,hc], optimize=True)
    O['ab'][v,C,c,A] += scale * +1.00000000 * np.einsum('uv,uIaU,baiv->bIiU', gamma1['a'], h['ab'][a,C,v,A], t['aa'][pv,pv,hc,ha], optimize=True)
    O['ab'][v,C,c,A] += scale * +1.00000000 * np.einsum('iIaU,baji->bIjU', h['ab'][c,C,v,A], t['aa'][pv,pv,hc,hc], optimize=True)
    O['ab'][a,V,c,A] += scale * -1.00000000 * np.einsum('uv,wx,uAxU,ywiv->yAiU', eta1['a'], gamma1['a'], h['ab'][a,V,a,A], t['aa'][pa,pa,hc,ha], optimize=True)
    O['ab'][a,V,c,A] += scale * +1.00000000 * np.einsum('uv,wx,wAvU,yuix->yAiU', eta1['a'], gamma1['a'], h['ab'][a,V,a,A], t['aa'][pa,pa,hc,ha], optimize=True)
    O['ab'][a,V,c,A] += scale * +1.00000000 * np.einsum('uv,iAvU,wuji->wAjU', eta1['a'], h['ab'][c,V,a,A], t['aa'][pa,pa,hc,hc], optimize=True)
    O['ab'][a,V,c,A] += scale * +1.00000000 * np.einsum('uv,uAaU,waiv->wAiU', gamma1['a'], h['ab'][a,V,v,A], t['aa'][pa,pv,hc,ha], optimize=True)
    O['ab'][a,V,c,A] += scale * +1.00000000 * np.einsum('iAaU,uaji->uAjU', h['ab'][c,V,v,A], t['aa'][pa,pv,hc,hc], optimize=True)
    O['ab'][v,V,c,A] += scale * +1.00000000 * np.einsum('uv,wx,uAxU,waiv->aAiU', eta1['a'], gamma1['a'], h['ab'][a,V,a,A], t['aa'][pa,pv,hc,ha], optimize=True)
    O['ab'][v,V,c,A] += scale * -1.00000000 * np.einsum('uv,wx,wAvU,uaix->aAiU', eta1['a'], gamma1['a'], h['ab'][a,V,a,A], t['aa'][pa,pv,hc,ha], optimize=True)
    O['ab'][v,V,c,A] += scale * -1.00000000 * np.einsum('uv,iAvU,uaji->aAjU', eta1['a'], h['ab'][c,V,a,A], t['aa'][pa,pv,hc,hc], optimize=True)
    O['ab'][v,V,c,A] += scale * +1.00000000 * np.einsum('uv,uAaU,baiv->bAiU', gamma1['a'], h['ab'][a,V,v,A], t['aa'][pv,pv,hc,ha], optimize=True)
    O['ab'][v,V,c,A] += scale * +1.00000000 * np.einsum('iAaU,baji->bAjU', h['ab'][c,V,v,A], t['aa'][pv,pv,hc,hc], optimize=True)
    O['ab'][a,A,c,C] += scale * -1.00000000 * np.einsum('uv,wx,uUxI,ywiv->yUiI', eta1['a'], gamma1['a'], h['ab'][a,A,a,C], t['aa'][pa,pa,hc,ha], optimize=True)
    O['ab'][a,A,c,C] += scale * +1.00000000 * np.einsum('uv,wx,wUvI,yuix->yUiI', eta1['a'], gamma1['a'], h['ab'][a,A,a,C], t['aa'][pa,pa,hc,ha], optimize=True)
    O['ab'][a,A,c,C] += scale * +1.00000000 * np.einsum('uv,iUvI,wuji->wUjI', eta1['a'], h['ab'][c,A,a,C], t['aa'][pa,pa,hc,hc], optimize=True)
    O['ab'][a,A,c,C] += scale * +1.00000000 * np.einsum('uv,uUaI,waiv->wUiI', gamma1['a'], h['ab'][a,A,v,C], t['aa'][pa,pv,hc,ha], optimize=True)
    O['ab'][a,A,c,C] += scale * +1.00000000 * np.einsum('iUaI,uaji->uUjI', h['ab'][c,A,v,C], t['aa'][pa,pv,hc,hc], optimize=True)
    O['ab'][v,A,c,C] += scale * +1.00000000 * np.einsum('uv,wx,uUxI,waiv->aUiI', eta1['a'], gamma1['a'], h['ab'][a,A,a,C], t['aa'][pa,pv,hc,ha], optimize=True)
    O['ab'][v,A,c,C] += scale * -1.00000000 * np.einsum('uv,wx,wUvI,uaix->aUiI', eta1['a'], gamma1['a'], h['ab'][a,A,a,C], t['aa'][pa,pv,hc,ha], optimize=True)
    O['ab'][v,A,c,C] += scale * -1.00000000 * np.einsum('uv,iUvI,uaji->aUjI', eta1['a'], h['ab'][c,A,a,C], t['aa'][pa,pv,hc,hc], optimize=True)
    O['ab'][v,A,c,C] += scale * +1.00000000 * np.einsum('uv,uUaI,baiv->bUiI', gamma1['a'], h['ab'][a,A,v,C], t['aa'][pv,pv,hc,ha], optimize=True)
    O['ab'][v,A,c,C] += scale * +1.00000000 * np.einsum('iUaI,baji->bUjI', h['ab'][c,A,v,C], t['aa'][pv,pv,hc,hc], optimize=True)
    O['ab'][a,C,c,C] += scale * -1.00000000 * np.einsum('uv,wx,uIxJ,ywiv->yIiJ', eta1['a'], gamma1['a'], h['ab'][a,C,a,C], t['aa'][pa,pa,hc,ha], optimize=True)
    O['ab'][a,C,c,C] += scale * +1.00000000 * np.einsum('uv,wx,wIvJ,yuix->yIiJ', eta1['a'], gamma1['a'], h['ab'][a,C,a,C], t['aa'][pa,pa,hc,ha], optimize=True)
    O['ab'][a,C,c,C] += scale * +1.00000000 * np.einsum('uv,iIvJ,wuji->wIjJ', eta1['a'], h['ab'][c,C,a,C], t['aa'][pa,pa,hc,hc], optimize=True)
    O['ab'][a,C,c,C] += scale * +1.00000000 * np.einsum('uv,uIaJ,waiv->wIiJ', gamma1['a'], h['ab'][a,C,v,C], t['aa'][pa,pv,hc,ha], optimize=True)
    O['ab'][a,C,c,C] += scale * +1.00000000 * np.einsum('iIaJ,uaji->uIjJ', h['ab'][c,C,v,C], t['aa'][pa,pv,hc,hc], optimize=True)
    O['ab'][v,C,c,C] += scale * +1.00000000 * np.einsum('uv,wx,uIxJ,waiv->aIiJ', eta1['a'], gamma1['a'], h['ab'][a,C,a,C], t['aa'][pa,pv,hc,ha], optimize=True)
    O['ab'][v,C,c,C] += scale * -1.00000000 * np.einsum('uv,wx,wIvJ,uaix->aIiJ', eta1['a'], gamma1['a'], h['ab'][a,C,a,C], t['aa'][pa,pv,hc,ha], optimize=True)
    O['ab'][v,C,c,C] += scale * -1.00000000 * np.einsum('uv,iIvJ,uaji->aIjJ', eta1['a'], h['ab'][c,C,a,C], t['aa'][pa,pv,hc,hc], optimize=True)
    O['ab'][v,C,c,C] += scale * +1.00000000 * np.einsum('uv,uIaJ,baiv->bIiJ', gamma1['a'], h['ab'][a,C,v,C], t['aa'][pv,pv,hc,ha], optimize=True)
    O['ab'][v,C,c,C] += scale * +1.00000000 * np.einsum('iIaJ,baji->bIjJ', h['ab'][c,C,v,C], t['aa'][pv,pv,hc,hc], optimize=True)
    O['ab'][a,V,c,C] += scale * -1.00000000 * np.einsum('uv,wx,uAxI,ywiv->yAiI', eta1['a'], gamma1['a'], h['ab'][a,V,a,C], t['aa'][pa,pa,hc,ha], optimize=True)
    O['ab'][a,V,c,C] += scale * +1.00000000 * np.einsum('uv,wx,wAvI,yuix->yAiI', eta1['a'], gamma1['a'], h['ab'][a,V,a,C], t['aa'][pa,pa,hc,ha], optimize=True)
    O['ab'][a,V,c,C] += scale * +1.00000000 * np.einsum('uv,iAvI,wuji->wAjI', eta1['a'], h['ab'][c,V,a,C], t['aa'][pa,pa,hc,hc], optimize=True)
    O['ab'][a,V,c,C] += scale * +1.00000000 * np.einsum('uv,uAaI,waiv->wAiI', gamma1['a'], h['ab'][a,V,v,C], t['aa'][pa,pv,hc,ha], optimize=True)
    O['ab'][a,V,c,C] += scale * +1.00000000 * np.einsum('iAaI,uaji->uAjI', h['ab'][c,V,v,C], t['aa'][pa,pv,hc,hc], optimize=True)
    O['ab'][v,V,c,C] += scale * +1.00000000 * np.einsum('uv,wx,uAxI,waiv->aAiI', eta1['a'], gamma1['a'], h['ab'][a,V,a,C], t['aa'][pa,pv,hc,ha], optimize=True)
    O['ab'][v,V,c,C] += scale * -1.00000000 * np.einsum('uv,wx,wAvI,uaix->aAiI', eta1['a'], gamma1['a'], h['ab'][a,V,a,C], t['aa'][pa,pv,hc,ha], optimize=True)
    O['ab'][v,V,c,C] += scale * -1.00000000 * np.einsum('uv,iAvI,uaji->aAjI', eta1['a'], h['ab'][c,V,a,C], t['aa'][pa,pv,hc,hc], optimize=True)
    O['ab'][v,V,c,C] += scale * +1.00000000 * np.einsum('uv,uAaI,baiv->bAiI', gamma1['a'], h['ab'][a,V,v,C], t['aa'][pv,pv,hc,ha], optimize=True)
    O['ab'][v,V,c,C] += scale * +1.00000000 * np.einsum('iAaI,baji->bAjI', h['ab'][c,V,v,C], t['aa'][pv,pv,hc,hc], optimize=True)
    O['ab'][a,A,c,V] += scale * -1.00000000 * np.einsum('uv,wx,uUxA,ywiv->yUiA', eta1['a'], gamma1['a'], h['ab'][a,A,a,V], t['aa'][pa,pa,hc,ha], optimize=True)
    O['ab'][a,A,c,V] += scale * +1.00000000 * np.einsum('uv,wx,wUvA,yuix->yUiA', eta1['a'], gamma1['a'], h['ab'][a,A,a,V], t['aa'][pa,pa,hc,ha], optimize=True)
    O['ab'][a,A,c,V] += scale * +1.00000000 * np.einsum('uv,iUvA,wuji->wUjA', eta1['a'], h['ab'][c,A,a,V], t['aa'][pa,pa,hc,hc], optimize=True)
    O['ab'][a,A,c,V] += scale * +1.00000000 * np.einsum('uv,uUaA,waiv->wUiA', gamma1['a'], h['ab'][a,A,v,V], t['aa'][pa,pv,hc,ha], optimize=True)
    O['ab'][a,A,c,V] += scale * +1.00000000 * np.einsum('iUaA,uaji->uUjA', h['ab'][c,A,v,V], t['aa'][pa,pv,hc,hc], optimize=True)
    O['ab'][v,A,c,V] += scale * +1.00000000 * np.einsum('uv,wx,uUxA,waiv->aUiA', eta1['a'], gamma1['a'], h['ab'][a,A,a,V], t['aa'][pa,pv,hc,ha], optimize=True)
    O['ab'][v,A,c,V] += scale * -1.00000000 * np.einsum('uv,wx,wUvA,uaix->aUiA', eta1['a'], gamma1['a'], h['ab'][a,A,a,V], t['aa'][pa,pv,hc,ha], optimize=True)
    O['ab'][v,A,c,V] += scale * -1.00000000 * np.einsum('uv,iUvA,uaji->aUjA', eta1['a'], h['ab'][c,A,a,V], t['aa'][pa,pv,hc,hc], optimize=True)
    O['ab'][v,A,c,V] += scale * +1.00000000 * np.einsum('uv,uUaA,baiv->bUiA', gamma1['a'], h['ab'][a,A,v,V], t['aa'][pv,pv,hc,ha], optimize=True)
    O['ab'][v,A,c,V] += scale * +1.00000000 * np.einsum('iUaA,baji->bUjA', h['ab'][c,A,v,V], t['aa'][pv,pv,hc,hc], optimize=True)
    O['ab'][a,C,c,V] += scale * -1.00000000 * np.einsum('uv,wx,uIxA,ywiv->yIiA', eta1['a'], gamma1['a'], h['ab'][a,C,a,V], t['aa'][pa,pa,hc,ha], optimize=True)
    O['ab'][a,C,c,V] += scale * +1.00000000 * np.einsum('uv,wx,wIvA,yuix->yIiA', eta1['a'], gamma1['a'], h['ab'][a,C,a,V], t['aa'][pa,pa,hc,ha], optimize=True)
    O['ab'][a,C,c,V] += scale * +1.00000000 * np.einsum('uv,iIvA,wuji->wIjA', eta1['a'], h['ab'][c,C,a,V], t['aa'][pa,pa,hc,hc], optimize=True)
    O['ab'][a,C,c,V] += scale * +1.00000000 * np.einsum('uv,uIaA,waiv->wIiA', gamma1['a'], h['ab'][a,C,v,V], t['aa'][pa,pv,hc,ha], optimize=True)
    O['ab'][a,C,c,V] += scale * +1.00000000 * np.einsum('iIaA,uaji->uIjA', h['ab'][c,C,v,V], t['aa'][pa,pv,hc,hc], optimize=True)
    O['ab'][v,C,c,V] += scale * +1.00000000 * np.einsum('uv,wx,uIxA,waiv->aIiA', eta1['a'], gamma1['a'], h['ab'][a,C,a,V], t['aa'][pa,pv,hc,ha], optimize=True)
    O['ab'][v,C,c,V] += scale * -1.00000000 * np.einsum('uv,wx,wIvA,uaix->aIiA', eta1['a'], gamma1['a'], h['ab'][a,C,a,V], t['aa'][pa,pv,hc,ha], optimize=True)
    O['ab'][v,C,c,V] += scale * -1.00000000 * np.einsum('uv,iIvA,uaji->aIjA', eta1['a'], h['ab'][c,C,a,V], t['aa'][pa,pv,hc,hc], optimize=True)
    O['ab'][v,C,c,V] += scale * +1.00000000 * np.einsum('uv,uIaA,baiv->bIiA', gamma1['a'], h['ab'][a,C,v,V], t['aa'][pv,pv,hc,ha], optimize=True)
    O['ab'][v,C,c,V] += scale * +1.00000000 * np.einsum('iIaA,baji->bIjA', h['ab'][c,C,v,V], t['aa'][pv,pv,hc,hc], optimize=True)
    O['ab'][a,V,c,V] += scale * -1.00000000 * np.einsum('uv,wx,uAxB,ywiv->yAiB', eta1['a'], gamma1['a'], h['ab'][a,V,a,V], t['aa'][pa,pa,hc,ha], optimize=True)
    O['ab'][a,V,c,V] += scale * +1.00000000 * np.einsum('uv,wx,wAvB,yuix->yAiB', eta1['a'], gamma1['a'], h['ab'][a,V,a,V], t['aa'][pa,pa,hc,ha], optimize=True)
    O['ab'][a,V,c,V] += scale * +1.00000000 * np.einsum('uv,iAvB,wuji->wAjB', eta1['a'], h['ab'][c,V,a,V], t['aa'][pa,pa,hc,hc], optimize=True)
    O['ab'][a,V,c,V] += scale * +1.00000000 * np.einsum('uv,uAaB,waiv->wAiB', gamma1['a'], h['ab'][a,V,v,V], t['aa'][pa,pv,hc,ha], optimize=True)
    O['ab'][a,V,c,V] += scale * +1.00000000 * np.einsum('iAaB,uaji->uAjB', h['ab'][c,V,v,V], t['aa'][pa,pv,hc,hc], optimize=True)
    O['ab'][v,V,c,V] += scale * +1.00000000 * np.einsum('uv,wx,uAxB,waiv->aAiB', eta1['a'], gamma1['a'], h['ab'][a,V,a,V], t['aa'][pa,pv,hc,ha], optimize=True)
    O['ab'][v,V,c,V] += scale * -1.00000000 * np.einsum('uv,wx,wAvB,uaix->aAiB', eta1['a'], gamma1['a'], h['ab'][a,V,a,V], t['aa'][pa,pv,hc,ha], optimize=True)
    O['ab'][v,V,c,V] += scale * -1.00000000 * np.einsum('uv,iAvB,uaji->aAjB', eta1['a'], h['ab'][c,V,a,V], t['aa'][pa,pv,hc,hc], optimize=True)
    O['ab'][v,V,c,V] += scale * +1.00000000 * np.einsum('uv,uAaB,baiv->bAiB', gamma1['a'], h['ab'][a,V,v,V], t['aa'][pv,pv,hc,ha], optimize=True)
    O['ab'][v,V,c,V] += scale * +1.00000000 * np.einsum('iAaB,baji->bAjB', h['ab'][c,V,v,V], t['aa'][pv,pv,hc,hc], optimize=True)

    t1 = time.time()
    if verbose: print("h2b_t2a_c2b took {:.4f} seconds to run.".format(t1-t0))

    return O
def h2b_t2b_c2b(O, h, t, gamma1, eta1, lambdas, orbspace, verbose=False, scale=1.0):
    # 720 lines
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

    
    
    O['ab'][a,A,a,A] += scale * -1.00000000 * np.einsum('uv,wIvU,uVxI->wVxU', eta1['a'], h['ab'][a,C,a,A], t['ab'][pa,pA,ha,hC], optimize=True)
    O['ab'][a,A,a,A] += scale * +1.00000000 * np.einsum('uv,wUvA,uAxV->wUxV', eta1['a'], h['ab'][a,A,a,V], t['ab'][pa,pV,ha,hA], optimize=True)
    
    
    
    
    O['ab'][a,A,a,A] += scale * -1.00000000 * np.einsum('UV,iWuV,vUiX->vWuX', eta1['b'], h['ab'][c,A,a,A], t['ab'][pa,pA,hc,hA], optimize=True)
    O['ab'][a,A,a,A] += scale * +1.00000000 * np.einsum('UV,uWaV,aUvX->uWvX', eta1['b'], h['ab'][a,A,v,A], t['ab'][pv,pA,ha,hA], optimize=True)
    O['ab'][a,A,a,A] += scale * +1.00000000 * np.einsum('uv,uIwU,xVvI->xVwU', gamma1['a'], h['ab'][a,C,a,A], t['ab'][pa,pA,ha,hC], optimize=True)
    O['ab'][a,A,a,A] += scale * -1.00000000 * np.einsum('uv,uUwA,xAvV->xUwV', gamma1['a'], h['ab'][a,A,a,V], t['ab'][pa,pV,ha,hA], optimize=True)
    
    
    O['ab'][a,A,a,A] += scale * +1.00000000 * np.einsum('UV,iUuW,vXiV->vXuW', gamma1['b'], h['ab'][c,A,a,A], t['ab'][pa,pA,hc,hA], optimize=True)
    O['ab'][a,A,a,A] += scale * -1.00000000 * np.einsum('UV,uUaW,aXvV->uXvW', gamma1['b'], h['ab'][a,A,v,A], t['ab'][pv,pA,ha,hA], optimize=True)
    O['ab'][a,A,a,A] += scale * +1.00000000 * np.einsum('iIuU,vViI->vVuU', h['ab'][c,C,a,A], t['ab'][pa,pA,hc,hC], optimize=True)
    O['ab'][a,A,a,A] += scale * -1.00000000 * np.einsum('iUuA,vAiV->vUuV', h['ab'][c,A,a,V], t['ab'][pa,pV,hc,hA], optimize=True)
    O['ab'][a,A,a,A] += scale * -1.00000000 * np.einsum('uIaU,aVvI->uVvU', h['ab'][a,C,v,A], t['ab'][pv,pA,ha,hC], optimize=True)
    O['ab'][a,A,a,A] += scale * +1.00000000 * np.einsum('uUaA,aAvV->uUvV', h['ab'][a,A,v,V], t['ab'][pv,pV,ha,hA], optimize=True)
    
    O['ab'][c,A,a,A] += scale * -1.00000000 * np.einsum('uv,iIvU,uVwI->iVwU', eta1['a'], h['ab'][c,C,a,A], t['ab'][pa,pA,ha,hC], optimize=True)
    O['ab'][c,A,a,A] += scale * +1.00000000 * np.einsum('uv,iUvA,uAwV->iUwV', eta1['a'], h['ab'][c,A,a,V], t['ab'][pa,pV,ha,hA], optimize=True)
    
    
    O['ab'][c,A,a,A] += scale * +1.00000000 * np.einsum('UV,iWaV,aUuX->iWuX', eta1['b'], h['ab'][c,A,v,A], t['ab'][pv,pA,ha,hA], optimize=True)
    
    O['ab'][c,A,a,A] += scale * -1.00000000 * np.einsum('UV,iUaW,aXuV->iXuW', gamma1['b'], h['ab'][c,A,v,A], t['ab'][pv,pA,ha,hA], optimize=True)
    O['ab'][c,A,a,A] += scale * -1.00000000 * np.einsum('iIaU,aVuI->iVuU', h['ab'][c,C,v,A], t['ab'][pv,pA,ha,hC], optimize=True)
    O['ab'][c,A,a,A] += scale * +1.00000000 * np.einsum('iUaA,aAuV->iUuV', h['ab'][c,A,v,V], t['ab'][pv,pV,ha,hA], optimize=True)
    O['ab'][v,A,a,A] += scale * +1.00000000 * np.einsum('uv,UV,uWwV,aUvX->aWwX', eta1['a'], gamma1['b'], h['ab'][a,A,a,A], t['ab'][pv,pA,ha,hA], optimize=True)
    
    O['ab'][v,A,a,A] += scale * -1.00000000 * np.einsum('uv,aIvU,uVwI->aVwU', eta1['a'], h['ab'][v,C,a,A], t['ab'][pa,pA,ha,hC], optimize=True)
    O['ab'][v,A,a,A] += scale * +1.00000000 * np.einsum('uv,aUvA,uAwV->aUwV', eta1['a'], h['ab'][v,A,a,V], t['ab'][pa,pV,ha,hA], optimize=True)
    O['ab'][v,A,a,A] += scale * -1.00000000 * np.einsum('UV,uv,uUwW,aXvV->aXwW', eta1['b'], eta1['a'], h['ab'][a,A,a,A], t['ab'][pv,pA,ha,hA], optimize=True)
    
    O['ab'][v,A,a,A] += scale * -1.00000000 * np.einsum('UV,uv,uWwV,aUvX->aWwX', eta1['b'], gamma1['a'], h['ab'][a,A,a,A], t['ab'][pv,pA,ha,hA], optimize=True)
    
    O['ab'][v,A,a,A] += scale * -1.00000000 * np.einsum('UV,iWuV,aUiX->aWuX', eta1['b'], h['ab'][c,A,a,A], t['ab'][pv,pA,hc,hA], optimize=True)
    O['ab'][v,A,a,A] += scale * +1.00000000 * np.einsum('UV,aWbV,bUuX->aWuX', eta1['b'], h['ab'][v,A,v,A], t['ab'][pv,pA,ha,hA], optimize=True)
    O['ab'][v,A,a,A] += scale * +1.00000000 * np.einsum('uv,uIwU,aVvI->aVwU', gamma1['a'], h['ab'][a,C,a,A], t['ab'][pv,pA,ha,hC], optimize=True)
    O['ab'][v,A,a,A] += scale * -1.00000000 * np.einsum('uv,uUwA,aAvV->aUwV', gamma1['a'], h['ab'][a,A,a,V], t['ab'][pv,pV,ha,hA], optimize=True)
    O['ab'][v,A,a,A] += scale * +1.00000000 * np.einsum('UV,uv,uUwW,aXvV->aXwW', gamma1['b'], gamma1['a'], h['ab'][a,A,a,A], t['ab'][pv,pA,ha,hA], optimize=True)
    
    O['ab'][v,A,a,A] += scale * +1.00000000 * np.einsum('UV,iUuW,aXiV->aXuW', gamma1['b'], h['ab'][c,A,a,A], t['ab'][pv,pA,hc,hA], optimize=True)
    O['ab'][v,A,a,A] += scale * -1.00000000 * np.einsum('UV,aUbW,bXuV->aXuW', gamma1['b'], h['ab'][v,A,v,A], t['ab'][pv,pA,ha,hA], optimize=True)
    O['ab'][v,A,a,A] += scale * +1.00000000 * np.einsum('iIuU,aViI->aVuU', h['ab'][c,C,a,A], t['ab'][pv,pA,hc,hC], optimize=True)
    O['ab'][v,A,a,A] += scale * -1.00000000 * np.einsum('iUuA,aAiV->aUuV', h['ab'][c,A,a,V], t['ab'][pv,pV,hc,hA], optimize=True)
    O['ab'][v,A,a,A] += scale * -1.00000000 * np.einsum('aIbU,bVuI->aVuU', h['ab'][v,C,v,A], t['ab'][pv,pA,ha,hC], optimize=True)
    O['ab'][v,A,a,A] += scale * +1.00000000 * np.einsum('aUbA,bAuV->aUuV', h['ab'][v,A,v,V], t['ab'][pv,pV,ha,hA], optimize=True)
    
    O['ab'][a,C,a,A] += scale * +1.00000000 * np.einsum('uv,wIvA,uAxU->wIxU', eta1['a'], h['ab'][a,C,a,V], t['ab'][pa,pV,ha,hA], optimize=True)
    
    
    O['ab'][a,C,a,A] += scale * -1.00000000 * np.einsum('UV,iIuV,vUiW->vIuW', eta1['b'], h['ab'][c,C,a,A], t['ab'][pa,pA,hc,hA], optimize=True)
    O['ab'][a,C,a,A] += scale * +1.00000000 * np.einsum('UV,uIaV,aUvW->uIvW', eta1['b'], h['ab'][a,C,v,A], t['ab'][pv,pA,ha,hA], optimize=True)
    O['ab'][a,C,a,A] += scale * -1.00000000 * np.einsum('uv,uIwA,xAvU->xIwU', gamma1['a'], h['ab'][a,C,a,V], t['ab'][pa,pV,ha,hA], optimize=True)
    
    O['ab'][a,C,a,A] += scale * -1.00000000 * np.einsum('iIuA,vAiU->vIuU', h['ab'][c,C,a,V], t['ab'][pa,pV,hc,hA], optimize=True)
    O['ab'][a,C,a,A] += scale * +1.00000000 * np.einsum('uIaA,aAvU->uIvU', h['ab'][a,C,v,V], t['ab'][pv,pV,ha,hA], optimize=True)
    O['ab'][c,C,a,A] += scale * +1.00000000 * np.einsum('uv,iIvA,uAwU->iIwU', eta1['a'], h['ab'][c,C,a,V], t['ab'][pa,pV,ha,hA], optimize=True)
    
    O['ab'][c,C,a,A] += scale * +1.00000000 * np.einsum('UV,iIaV,aUuW->iIuW', eta1['b'], h['ab'][c,C,v,A], t['ab'][pv,pA,ha,hA], optimize=True)
    
    O['ab'][c,C,a,A] += scale * +1.00000000 * np.einsum('iIaA,aAuU->iIuU', h['ab'][c,C,v,V], t['ab'][pv,pV,ha,hA], optimize=True)
    O['ab'][v,C,a,A] += scale * +1.00000000 * np.einsum('uv,UV,uIwV,aUvW->aIwW', eta1['a'], gamma1['b'], h['ab'][a,C,a,A], t['ab'][pv,pA,ha,hA], optimize=True)
    O['ab'][v,C,a,A] += scale * +1.00000000 * np.einsum('uv,aIvA,uAwU->aIwU', eta1['a'], h['ab'][v,C,a,V], t['ab'][pa,pV,ha,hA], optimize=True)
    
    O['ab'][v,C,a,A] += scale * -1.00000000 * np.einsum('UV,uv,uIwV,aUvW->aIwW', eta1['b'], gamma1['a'], h['ab'][a,C,a,A], t['ab'][pv,pA,ha,hA], optimize=True)
    O['ab'][v,C,a,A] += scale * -1.00000000 * np.einsum('UV,iIuV,aUiW->aIuW', eta1['b'], h['ab'][c,C,a,A], t['ab'][pv,pA,hc,hA], optimize=True)
    O['ab'][v,C,a,A] += scale * +1.00000000 * np.einsum('UV,aIbV,bUuW->aIuW', eta1['b'], h['ab'][v,C,v,A], t['ab'][pv,pA,ha,hA], optimize=True)
    O['ab'][v,C,a,A] += scale * -1.00000000 * np.einsum('uv,uIwA,aAvU->aIwU', gamma1['a'], h['ab'][a,C,a,V], t['ab'][pv,pV,ha,hA], optimize=True)
    
    O['ab'][v,C,a,A] += scale * -1.00000000 * np.einsum('iIuA,aAiU->aIuU', h['ab'][c,C,a,V], t['ab'][pv,pV,hc,hA], optimize=True)
    O['ab'][v,C,a,A] += scale * +1.00000000 * np.einsum('aIbA,bAuU->aIuU', h['ab'][v,C,v,V], t['ab'][pv,pV,ha,hA], optimize=True)
    O['ab'][a,V,a,A] += scale * -1.00000000 * np.einsum('uv,UV,wUvW,uAxV->wAxW', eta1['a'], gamma1['b'], h['ab'][a,A,a,A], t['ab'][pa,pV,ha,hA], optimize=True)
    
    O['ab'][a,V,a,A] += scale * -1.00000000 * np.einsum('uv,wIvU,uAxI->wAxU', eta1['a'], h['ab'][a,C,a,A], t['ab'][pa,pV,ha,hC], optimize=True)
    O['ab'][a,V,a,A] += scale * +1.00000000 * np.einsum('uv,wAvB,uBxU->wAxU', eta1['a'], h['ab'][a,V,a,V], t['ab'][pa,pV,ha,hA], optimize=True)
    
    O['ab'][a,V,a,A] += scale * -1.00000000 * np.einsum('UV,uv,uUwW,xAvV->xAwW', eta1['b'], eta1['a'], h['ab'][a,A,a,A], t['ab'][pa,pV,ha,hA], optimize=True)
    O['ab'][a,V,a,A] += scale * +1.00000000 * np.einsum('UV,uv,wUvW,uAxV->wAxW', eta1['b'], gamma1['a'], h['ab'][a,A,a,A], t['ab'][pa,pV,ha,hA], optimize=True)
    
    O['ab'][a,V,a,A] += scale * -1.00000000 * np.einsum('UV,iAuV,vUiW->vAuW', eta1['b'], h['ab'][c,V,a,A], t['ab'][pa,pA,hc,hA], optimize=True)
    O['ab'][a,V,a,A] += scale * +1.00000000 * np.einsum('UV,uAaV,aUvW->uAvW', eta1['b'], h['ab'][a,V,v,A], t['ab'][pv,pA,ha,hA], optimize=True)
    O['ab'][a,V,a,A] += scale * +1.00000000 * np.einsum('uv,uIwU,xAvI->xAwU', gamma1['a'], h['ab'][a,C,a,A], t['ab'][pa,pV,ha,hC], optimize=True)
    O['ab'][a,V,a,A] += scale * -1.00000000 * np.einsum('uv,uAwB,xBvU->xAwU', gamma1['a'], h['ab'][a,V,a,V], t['ab'][pa,pV,ha,hA], optimize=True)
    
    O['ab'][a,V,a,A] += scale * +1.00000000 * np.einsum('UV,uv,uUwW,xAvV->xAwW', gamma1['b'], gamma1['a'], h['ab'][a,A,a,A], t['ab'][pa,pV,ha,hA], optimize=True)
    O['ab'][a,V,a,A] += scale * +1.00000000 * np.einsum('UV,iUuW,vAiV->vAuW', gamma1['b'], h['ab'][c,A,a,A], t['ab'][pa,pV,hc,hA], optimize=True)
    O['ab'][a,V,a,A] += scale * -1.00000000 * np.einsum('UV,uUaW,aAvV->uAvW', gamma1['b'], h['ab'][a,A,v,A], t['ab'][pv,pV,ha,hA], optimize=True)
    O['ab'][a,V,a,A] += scale * +1.00000000 * np.einsum('iIuU,vAiI->vAuU', h['ab'][c,C,a,A], t['ab'][pa,pV,hc,hC], optimize=True)
    O['ab'][a,V,a,A] += scale * -1.00000000 * np.einsum('iAuB,vBiU->vAuU', h['ab'][c,V,a,V], t['ab'][pa,pV,hc,hA], optimize=True)
    O['ab'][a,V,a,A] += scale * -1.00000000 * np.einsum('uIaU,aAvI->uAvU', h['ab'][a,C,v,A], t['ab'][pv,pV,ha,hC], optimize=True)
    O['ab'][a,V,a,A] += scale * +1.00000000 * np.einsum('uAaB,aBvU->uAvU', h['ab'][a,V,v,V], t['ab'][pv,pV,ha,hA], optimize=True)
    O['ab'][c,V,a,A] += scale * -1.00000000 * np.einsum('uv,UV,iUvW,uAwV->iAwW', eta1['a'], gamma1['b'], h['ab'][c,A,a,A], t['ab'][pa,pV,ha,hA], optimize=True)
    O['ab'][c,V,a,A] += scale * -1.00000000 * np.einsum('uv,iIvU,uAwI->iAwU', eta1['a'], h['ab'][c,C,a,A], t['ab'][pa,pV,ha,hC], optimize=True)
    O['ab'][c,V,a,A] += scale * +1.00000000 * np.einsum('uv,iAvB,uBwU->iAwU', eta1['a'], h['ab'][c,V,a,V], t['ab'][pa,pV,ha,hA], optimize=True)
    
    O['ab'][c,V,a,A] += scale * +1.00000000 * np.einsum('UV,uv,iUvW,uAwV->iAwW', eta1['b'], gamma1['a'], h['ab'][c,A,a,A], t['ab'][pa,pV,ha,hA], optimize=True)
    O['ab'][c,V,a,A] += scale * +1.00000000 * np.einsum('UV,iAaV,aUuW->iAuW', eta1['b'], h['ab'][c,V,v,A], t['ab'][pv,pA,ha,hA], optimize=True)
    
    O['ab'][c,V,a,A] += scale * -1.00000000 * np.einsum('UV,iUaW,aAuV->iAuW', gamma1['b'], h['ab'][c,A,v,A], t['ab'][pv,pV,ha,hA], optimize=True)
    O['ab'][c,V,a,A] += scale * -1.00000000 * np.einsum('iIaU,aAuI->iAuU', h['ab'][c,C,v,A], t['ab'][pv,pV,ha,hC], optimize=True)
    O['ab'][c,V,a,A] += scale * +1.00000000 * np.einsum('iAaB,aBuU->iAuU', h['ab'][c,V,v,V], t['ab'][pv,pV,ha,hA], optimize=True)
    O['ab'][v,V,a,A] += scale * +1.00000000 * np.einsum('uv,UV,uAwV,aUvW->aAwW', eta1['a'], gamma1['b'], h['ab'][a,V,a,A], t['ab'][pv,pA,ha,hA], optimize=True)
    O['ab'][v,V,a,A] += scale * -1.00000000 * np.einsum('uv,UV,aUvW,uAwV->aAwW', eta1['a'], gamma1['b'], h['ab'][v,A,a,A], t['ab'][pa,pV,ha,hA], optimize=True)
    O['ab'][v,V,a,A] += scale * -1.00000000 * np.einsum('uv,aIvU,uAwI->aAwU', eta1['a'], h['ab'][v,C,a,A], t['ab'][pa,pV,ha,hC], optimize=True)
    O['ab'][v,V,a,A] += scale * +1.00000000 * np.einsum('uv,aAvB,uBwU->aAwU', eta1['a'], h['ab'][v,V,a,V], t['ab'][pa,pV,ha,hA], optimize=True)
    O['ab'][v,V,a,A] += scale * -1.00000000 * np.einsum('UV,uv,uUwW,aAvV->aAwW', eta1['b'], eta1['a'], h['ab'][a,A,a,A], t['ab'][pv,pV,ha,hA], optimize=True)
    
    O['ab'][v,V,a,A] += scale * -1.00000000 * np.einsum('UV,uv,uAwV,aUvW->aAwW', eta1['b'], gamma1['a'], h['ab'][a,V,a,A], t['ab'][pv,pA,ha,hA], optimize=True)
    O['ab'][v,V,a,A] += scale * +1.00000000 * np.einsum('UV,uv,aUvW,uAwV->aAwW', eta1['b'], gamma1['a'], h['ab'][v,A,a,A], t['ab'][pa,pV,ha,hA], optimize=True)
    O['ab'][v,V,a,A] += scale * -1.00000000 * np.einsum('UV,iAuV,aUiW->aAuW', eta1['b'], h['ab'][c,V,a,A], t['ab'][pv,pA,hc,hA], optimize=True)
    O['ab'][v,V,a,A] += scale * +1.00000000 * np.einsum('UV,aAbV,bUuW->aAuW', eta1['b'], h['ab'][v,V,v,A], t['ab'][pv,pA,ha,hA], optimize=True)
    O['ab'][v,V,a,A] += scale * +1.00000000 * np.einsum('uv,uIwU,aAvI->aAwU', gamma1['a'], h['ab'][a,C,a,A], t['ab'][pv,pV,ha,hC], optimize=True)
    O['ab'][v,V,a,A] += scale * -1.00000000 * np.einsum('uv,uAwB,aBvU->aAwU', gamma1['a'], h['ab'][a,V,a,V], t['ab'][pv,pV,ha,hA], optimize=True)
    O['ab'][v,V,a,A] += scale * +1.00000000 * np.einsum('UV,uv,uUwW,aAvV->aAwW', gamma1['b'], gamma1['a'], h['ab'][a,A,a,A], t['ab'][pv,pV,ha,hA], optimize=True)
    
    O['ab'][v,V,a,A] += scale * +1.00000000 * np.einsum('UV,iUuW,aAiV->aAuW', gamma1['b'], h['ab'][c,A,a,A], t['ab'][pv,pV,hc,hA], optimize=True)
    O['ab'][v,V,a,A] += scale * -1.00000000 * np.einsum('UV,aUbW,bAuV->aAuW', gamma1['b'], h['ab'][v,A,v,A], t['ab'][pv,pV,ha,hA], optimize=True)
    O['ab'][v,V,a,A] += scale * +1.00000000 * np.einsum('iIuU,aAiI->aAuU', h['ab'][c,C,a,A], t['ab'][pv,pV,hc,hC], optimize=True)
    O['ab'][v,V,a,A] += scale * -1.00000000 * np.einsum('iAuB,aBiU->aAuU', h['ab'][c,V,a,V], t['ab'][pv,pV,hc,hA], optimize=True)
    O['ab'][v,V,a,A] += scale * -1.00000000 * np.einsum('aIbU,bAuI->aAuU', h['ab'][v,C,v,A], t['ab'][pv,pV,ha,hC], optimize=True)
    O['ab'][v,V,a,A] += scale * +1.00000000 * np.einsum('aAbB,bBuU->aAuU', h['ab'][v,V,v,V], t['ab'][pv,pV,ha,hA], optimize=True)
    
    O['ab'][a,A,a,C] += scale * +1.00000000 * np.einsum('uv,UV,uWwV,xUvI->xWwI', eta1['a'], gamma1['b'], h['ab'][a,A,a,A], t['ab'][pa,pA,ha,hC], optimize=True)
    O['ab'][a,A,a,C] += scale * -1.00000000 * np.einsum('uv,wIvJ,uUxI->wUxJ', eta1['a'], h['ab'][a,C,a,C], t['ab'][pa,pA,ha,hC], optimize=True)
    O['ab'][a,A,a,C] += scale * +1.00000000 * np.einsum('uv,wUvA,uAxI->wUxI', eta1['a'], h['ab'][a,A,a,V], t['ab'][pa,pV,ha,hC], optimize=True)
    O['ab'][a,A,a,C] += scale * +1.00000000 * np.einsum('UV,uv,wWvV,uUxI->wWxI', eta1['b'], eta1['a'], h['ab'][a,A,a,A], t['ab'][pa,pA,ha,hC], optimize=True)
    
    
    O['ab'][a,A,a,C] += scale * -1.00000000 * np.einsum('UV,uv,uWwV,xUvI->xWwI', eta1['b'], gamma1['a'], h['ab'][a,A,a,A], t['ab'][pa,pA,ha,hC], optimize=True)
    O['ab'][a,A,a,C] += scale * -1.00000000 * np.einsum('UV,iWuV,vUiI->vWuI', eta1['b'], h['ab'][c,A,a,A], t['ab'][pa,pA,hc,hC], optimize=True)
    O['ab'][a,A,a,C] += scale * +1.00000000 * np.einsum('UV,uWaV,aUvI->uWvI', eta1['b'], h['ab'][a,A,v,A], t['ab'][pv,pA,ha,hC], optimize=True)
    O['ab'][a,A,a,C] += scale * +1.00000000 * np.einsum('uv,uIwJ,xUvI->xUwJ', gamma1['a'], h['ab'][a,C,a,C], t['ab'][pa,pA,ha,hC], optimize=True)
    O['ab'][a,A,a,C] += scale * -1.00000000 * np.einsum('uv,uUwA,xAvI->xUwI', gamma1['a'], h['ab'][a,A,a,V], t['ab'][pa,pV,ha,hC], optimize=True)
    O['ab'][a,A,a,C] += scale * -1.00000000 * np.einsum('UV,uv,wWvV,uUxI->wWxI', gamma1['b'], gamma1['a'], h['ab'][a,A,a,A], t['ab'][pa,pA,ha,hC], optimize=True)
    
    O['ab'][a,A,a,C] += scale * +1.00000000 * np.einsum('UV,iUuI,vWiV->vWuI', gamma1['b'], h['ab'][c,A,a,C], t['ab'][pa,pA,hc,hA], optimize=True)
    O['ab'][a,A,a,C] += scale * -1.00000000 * np.einsum('UV,uUaI,aWvV->uWvI', gamma1['b'], h['ab'][a,A,v,C], t['ab'][pv,pA,ha,hA], optimize=True)
    O['ab'][a,A,a,C] += scale * +1.00000000 * np.einsum('iIuJ,vUiI->vUuJ', h['ab'][c,C,a,C], t['ab'][pa,pA,hc,hC], optimize=True)
    O['ab'][a,A,a,C] += scale * -1.00000000 * np.einsum('iUuA,vAiI->vUuI', h['ab'][c,A,a,V], t['ab'][pa,pV,hc,hC], optimize=True)
    O['ab'][a,A,a,C] += scale * -1.00000000 * np.einsum('uIaJ,aUvI->uUvJ', h['ab'][a,C,v,C], t['ab'][pv,pA,ha,hC], optimize=True)
    O['ab'][a,A,a,C] += scale * +1.00000000 * np.einsum('uUaA,aAvI->uUvI', h['ab'][a,A,v,V], t['ab'][pv,pV,ha,hC], optimize=True)
    
    O['ab'][c,A,a,C] += scale * -1.00000000 * np.einsum('uv,iIvJ,uUwI->iUwJ', eta1['a'], h['ab'][c,C,a,C], t['ab'][pa,pA,ha,hC], optimize=True)
    O['ab'][c,A,a,C] += scale * +1.00000000 * np.einsum('uv,iUvA,uAwI->iUwI', eta1['a'], h['ab'][c,A,a,V], t['ab'][pa,pV,ha,hC], optimize=True)
    O['ab'][c,A,a,C] += scale * +1.00000000 * np.einsum('UV,uv,iWvV,uUwI->iWwI', eta1['b'], eta1['a'], h['ab'][c,A,a,A], t['ab'][pa,pA,ha,hC], optimize=True)
    
    O['ab'][c,A,a,C] += scale * +1.00000000 * np.einsum('UV,iWaV,aUuI->iWuI', eta1['b'], h['ab'][c,A,v,A], t['ab'][pv,pA,ha,hC], optimize=True)
    O['ab'][c,A,a,C] += scale * -1.00000000 * np.einsum('UV,uv,iWvV,uUwI->iWwI', gamma1['b'], gamma1['a'], h['ab'][c,A,a,A], t['ab'][pa,pA,ha,hC], optimize=True)
    O['ab'][c,A,a,C] += scale * -1.00000000 * np.einsum('UV,iUaI,aWuV->iWuI', gamma1['b'], h['ab'][c,A,v,C], t['ab'][pv,pA,ha,hA], optimize=True)
    O['ab'][c,A,a,C] += scale * -1.00000000 * np.einsum('iIaJ,aUuI->iUuJ', h['ab'][c,C,v,C], t['ab'][pv,pA,ha,hC], optimize=True)
    O['ab'][c,A,a,C] += scale * +1.00000000 * np.einsum('iUaA,aAuI->iUuI', h['ab'][c,A,v,V], t['ab'][pv,pV,ha,hC], optimize=True)
    O['ab'][v,A,a,C] += scale * +1.00000000 * np.einsum('uv,UV,uWwV,aUvI->aWwI', eta1['a'], gamma1['b'], h['ab'][a,A,a,A], t['ab'][pv,pA,ha,hC], optimize=True)
    
    O['ab'][v,A,a,C] += scale * -1.00000000 * np.einsum('uv,aIvJ,uUwI->aUwJ', eta1['a'], h['ab'][v,C,a,C], t['ab'][pa,pA,ha,hC], optimize=True)
    O['ab'][v,A,a,C] += scale * +1.00000000 * np.einsum('uv,aUvA,uAwI->aUwI', eta1['a'], h['ab'][v,A,a,V], t['ab'][pa,pV,ha,hC], optimize=True)
    O['ab'][v,A,a,C] += scale * -1.00000000 * np.einsum('UV,uv,uUwI,aWvV->aWwI', eta1['b'], eta1['a'], h['ab'][a,A,a,C], t['ab'][pv,pA,ha,hA], optimize=True)
    O['ab'][v,A,a,C] += scale * +1.00000000 * np.einsum('UV,uv,aWvV,uUwI->aWwI', eta1['b'], eta1['a'], h['ab'][v,A,a,A], t['ab'][pa,pA,ha,hC], optimize=True)
    O['ab'][v,A,a,C] += scale * -1.00000000 * np.einsum('UV,uv,uWwV,aUvI->aWwI', eta1['b'], gamma1['a'], h['ab'][a,A,a,A], t['ab'][pv,pA,ha,hC], optimize=True)
    
    O['ab'][v,A,a,C] += scale * -1.00000000 * np.einsum('UV,iWuV,aUiI->aWuI', eta1['b'], h['ab'][c,A,a,A], t['ab'][pv,pA,hc,hC], optimize=True)
    O['ab'][v,A,a,C] += scale * +1.00000000 * np.einsum('UV,aWbV,bUuI->aWuI', eta1['b'], h['ab'][v,A,v,A], t['ab'][pv,pA,ha,hC], optimize=True)
    O['ab'][v,A,a,C] += scale * +1.00000000 * np.einsum('uv,uIwJ,aUvI->aUwJ', gamma1['a'], h['ab'][a,C,a,C], t['ab'][pv,pA,ha,hC], optimize=True)
    O['ab'][v,A,a,C] += scale * -1.00000000 * np.einsum('uv,uUwA,aAvI->aUwI', gamma1['a'], h['ab'][a,A,a,V], t['ab'][pv,pV,ha,hC], optimize=True)
    O['ab'][v,A,a,C] += scale * +1.00000000 * np.einsum('UV,uv,uUwI,aWvV->aWwI', gamma1['b'], gamma1['a'], h['ab'][a,A,a,C], t['ab'][pv,pA,ha,hA], optimize=True)
    O['ab'][v,A,a,C] += scale * -1.00000000 * np.einsum('UV,uv,aWvV,uUwI->aWwI', gamma1['b'], gamma1['a'], h['ab'][v,A,a,A], t['ab'][pa,pA,ha,hC], optimize=True)
    O['ab'][v,A,a,C] += scale * +1.00000000 * np.einsum('UV,iUuI,aWiV->aWuI', gamma1['b'], h['ab'][c,A,a,C], t['ab'][pv,pA,hc,hA], optimize=True)
    O['ab'][v,A,a,C] += scale * -1.00000000 * np.einsum('UV,aUbI,bWuV->aWuI', gamma1['b'], h['ab'][v,A,v,C], t['ab'][pv,pA,ha,hA], optimize=True)
    O['ab'][v,A,a,C] += scale * +1.00000000 * np.einsum('iIuJ,aUiI->aUuJ', h['ab'][c,C,a,C], t['ab'][pv,pA,hc,hC], optimize=True)
    O['ab'][v,A,a,C] += scale * -1.00000000 * np.einsum('iUuA,aAiI->aUuI', h['ab'][c,A,a,V], t['ab'][pv,pV,hc,hC], optimize=True)
    O['ab'][v,A,a,C] += scale * -1.00000000 * np.einsum('aIbJ,bUuI->aUuJ', h['ab'][v,C,v,C], t['ab'][pv,pA,ha,hC], optimize=True)
    O['ab'][v,A,a,C] += scale * +1.00000000 * np.einsum('aUbA,bAuI->aUuI', h['ab'][v,A,v,V], t['ab'][pv,pV,ha,hC], optimize=True)
    O['ab'][a,C,a,C] += scale * +1.00000000 * np.einsum('uv,UV,uIwV,xUvJ->xIwJ', eta1['a'], gamma1['b'], h['ab'][a,C,a,A], t['ab'][pa,pA,ha,hC], optimize=True)
    O['ab'][a,C,a,C] += scale * +1.00000000 * np.einsum('uv,wIvA,uAxJ->wIxJ', eta1['a'], h['ab'][a,C,a,V], t['ab'][pa,pV,ha,hC], optimize=True)
    O['ab'][a,C,a,C] += scale * +1.00000000 * np.einsum('UV,uv,wIvV,uUxJ->wIxJ', eta1['b'], eta1['a'], h['ab'][a,C,a,A], t['ab'][pa,pA,ha,hC], optimize=True)
    O['ab'][a,C,a,C] += scale * -1.00000000 * np.einsum('UV,uv,uIwV,xUvJ->xIwJ', eta1['b'], gamma1['a'], h['ab'][a,C,a,A], t['ab'][pa,pA,ha,hC], optimize=True)
    O['ab'][a,C,a,C] += scale * -1.00000000 * np.einsum('UV,iIuV,vUiJ->vIuJ', eta1['b'], h['ab'][c,C,a,A], t['ab'][pa,pA,hc,hC], optimize=True)
    O['ab'][a,C,a,C] += scale * +1.00000000 * np.einsum('UV,uIaV,aUvJ->uIvJ', eta1['b'], h['ab'][a,C,v,A], t['ab'][pv,pA,ha,hC], optimize=True)
    O['ab'][a,C,a,C] += scale * -1.00000000 * np.einsum('uv,uIwA,xAvJ->xIwJ', gamma1['a'], h['ab'][a,C,a,V], t['ab'][pa,pV,ha,hC], optimize=True)
    O['ab'][a,C,a,C] += scale * -1.00000000 * np.einsum('UV,uv,wIvV,uUxJ->wIxJ', gamma1['b'], gamma1['a'], h['ab'][a,C,a,A], t['ab'][pa,pA,ha,hC], optimize=True)
    O['ab'][a,C,a,C] += scale * -1.00000000 * np.einsum('iIuA,vAiJ->vIuJ', h['ab'][c,C,a,V], t['ab'][pa,pV,hc,hC], optimize=True)
    O['ab'][a,C,a,C] += scale * +1.00000000 * np.einsum('uIaA,aAvJ->uIvJ', h['ab'][a,C,v,V], t['ab'][pv,pV,ha,hC], optimize=True)
    O['ab'][c,C,a,C] += scale * +1.00000000 * np.einsum('uv,iIvA,uAwJ->iIwJ', eta1['a'], h['ab'][c,C,a,V], t['ab'][pa,pV,ha,hC], optimize=True)
    O['ab'][c,C,a,C] += scale * +1.00000000 * np.einsum('UV,uv,iIvV,uUwJ->iIwJ', eta1['b'], eta1['a'], h['ab'][c,C,a,A], t['ab'][pa,pA,ha,hC], optimize=True)
    O['ab'][c,C,a,C] += scale * +1.00000000 * np.einsum('UV,iIaV,aUuJ->iIuJ', eta1['b'], h['ab'][c,C,v,A], t['ab'][pv,pA,ha,hC], optimize=True)
    O['ab'][c,C,a,C] += scale * -1.00000000 * np.einsum('UV,uv,iIvV,uUwJ->iIwJ', gamma1['b'], gamma1['a'], h['ab'][c,C,a,A], t['ab'][pa,pA,ha,hC], optimize=True)
    O['ab'][c,C,a,C] += scale * +1.00000000 * np.einsum('iIaA,aAuJ->iIuJ', h['ab'][c,C,v,V], t['ab'][pv,pV,ha,hC], optimize=True)
    O['ab'][v,C,a,C] += scale * +1.00000000 * np.einsum('uv,UV,uIwV,aUvJ->aIwJ', eta1['a'], gamma1['b'], h['ab'][a,C,a,A], t['ab'][pv,pA,ha,hC], optimize=True)
    O['ab'][v,C,a,C] += scale * +1.00000000 * np.einsum('uv,aIvA,uAwJ->aIwJ', eta1['a'], h['ab'][v,C,a,V], t['ab'][pa,pV,ha,hC], optimize=True)
    O['ab'][v,C,a,C] += scale * +1.00000000 * np.einsum('UV,uv,aIvV,uUwJ->aIwJ', eta1['b'], eta1['a'], h['ab'][v,C,a,A], t['ab'][pa,pA,ha,hC], optimize=True)
    O['ab'][v,C,a,C] += scale * -1.00000000 * np.einsum('UV,uv,uIwV,aUvJ->aIwJ', eta1['b'], gamma1['a'], h['ab'][a,C,a,A], t['ab'][pv,pA,ha,hC], optimize=True)
    O['ab'][v,C,a,C] += scale * -1.00000000 * np.einsum('UV,iIuV,aUiJ->aIuJ', eta1['b'], h['ab'][c,C,a,A], t['ab'][pv,pA,hc,hC], optimize=True)
    O['ab'][v,C,a,C] += scale * +1.00000000 * np.einsum('UV,aIbV,bUuJ->aIuJ', eta1['b'], h['ab'][v,C,v,A], t['ab'][pv,pA,ha,hC], optimize=True)
    O['ab'][v,C,a,C] += scale * -1.00000000 * np.einsum('uv,uIwA,aAvJ->aIwJ', gamma1['a'], h['ab'][a,C,a,V], t['ab'][pv,pV,ha,hC], optimize=True)
    O['ab'][v,C,a,C] += scale * -1.00000000 * np.einsum('UV,uv,aIvV,uUwJ->aIwJ', gamma1['b'], gamma1['a'], h['ab'][v,C,a,A], t['ab'][pa,pA,ha,hC], optimize=True)
    O['ab'][v,C,a,C] += scale * -1.00000000 * np.einsum('iIuA,aAiJ->aIuJ', h['ab'][c,C,a,V], t['ab'][pv,pV,hc,hC], optimize=True)
    O['ab'][v,C,a,C] += scale * +1.00000000 * np.einsum('aIbA,bAuJ->aIuJ', h['ab'][v,C,v,V], t['ab'][pv,pV,ha,hC], optimize=True)
    O['ab'][a,V,a,C] += scale * -1.00000000 * np.einsum('uv,UV,wUvI,uAxV->wAxI', eta1['a'], gamma1['b'], h['ab'][a,A,a,C], t['ab'][pa,pV,ha,hA], optimize=True)
    O['ab'][a,V,a,C] += scale * +1.00000000 * np.einsum('uv,UV,uAwV,xUvI->xAwI', eta1['a'], gamma1['b'], h['ab'][a,V,a,A], t['ab'][pa,pA,ha,hC], optimize=True)
    O['ab'][a,V,a,C] += scale * -1.00000000 * np.einsum('uv,wIvJ,uAxI->wAxJ', eta1['a'], h['ab'][a,C,a,C], t['ab'][pa,pV,ha,hC], optimize=True)
    O['ab'][a,V,a,C] += scale * +1.00000000 * np.einsum('uv,wAvB,uBxI->wAxI', eta1['a'], h['ab'][a,V,a,V], t['ab'][pa,pV,ha,hC], optimize=True)
    O['ab'][a,V,a,C] += scale * +1.00000000 * np.einsum('UV,uv,wAvV,uUxI->wAxI', eta1['b'], eta1['a'], h['ab'][a,V,a,A], t['ab'][pa,pA,ha,hC], optimize=True)
    O['ab'][a,V,a,C] += scale * -1.00000000 * np.einsum('UV,uv,uUwI,xAvV->xAwI', eta1['b'], eta1['a'], h['ab'][a,A,a,C], t['ab'][pa,pV,ha,hA], optimize=True)
    O['ab'][a,V,a,C] += scale * +1.00000000 * np.einsum('UV,uv,wUvI,uAxV->wAxI', eta1['b'], gamma1['a'], h['ab'][a,A,a,C], t['ab'][pa,pV,ha,hA], optimize=True)
    O['ab'][a,V,a,C] += scale * -1.00000000 * np.einsum('UV,uv,uAwV,xUvI->xAwI', eta1['b'], gamma1['a'], h['ab'][a,V,a,A], t['ab'][pa,pA,ha,hC], optimize=True)
    O['ab'][a,V,a,C] += scale * -1.00000000 * np.einsum('UV,iAuV,vUiI->vAuI', eta1['b'], h['ab'][c,V,a,A], t['ab'][pa,pA,hc,hC], optimize=True)
    O['ab'][a,V,a,C] += scale * +1.00000000 * np.einsum('UV,uAaV,aUvI->uAvI', eta1['b'], h['ab'][a,V,v,A], t['ab'][pv,pA,ha,hC], optimize=True)
    O['ab'][a,V,a,C] += scale * +1.00000000 * np.einsum('uv,uIwJ,xAvI->xAwJ', gamma1['a'], h['ab'][a,C,a,C], t['ab'][pa,pV,ha,hC], optimize=True)
    O['ab'][a,V,a,C] += scale * -1.00000000 * np.einsum('uv,uAwB,xBvI->xAwI', gamma1['a'], h['ab'][a,V,a,V], t['ab'][pa,pV,ha,hC], optimize=True)
    O['ab'][a,V,a,C] += scale * -1.00000000 * np.einsum('UV,uv,wAvV,uUxI->wAxI', gamma1['b'], gamma1['a'], h['ab'][a,V,a,A], t['ab'][pa,pA,ha,hC], optimize=True)
    O['ab'][a,V,a,C] += scale * +1.00000000 * np.einsum('UV,uv,uUwI,xAvV->xAwI', gamma1['b'], gamma1['a'], h['ab'][a,A,a,C], t['ab'][pa,pV,ha,hA], optimize=True)
    O['ab'][a,V,a,C] += scale * +1.00000000 * np.einsum('UV,iUuI,vAiV->vAuI', gamma1['b'], h['ab'][c,A,a,C], t['ab'][pa,pV,hc,hA], optimize=True)
    O['ab'][a,V,a,C] += scale * -1.00000000 * np.einsum('UV,uUaI,aAvV->uAvI', gamma1['b'], h['ab'][a,A,v,C], t['ab'][pv,pV,ha,hA], optimize=True)
    O['ab'][a,V,a,C] += scale * +1.00000000 * np.einsum('iIuJ,vAiI->vAuJ', h['ab'][c,C,a,C], t['ab'][pa,pV,hc,hC], optimize=True)
    O['ab'][a,V,a,C] += scale * -1.00000000 * np.einsum('iAuB,vBiI->vAuI', h['ab'][c,V,a,V], t['ab'][pa,pV,hc,hC], optimize=True)
    O['ab'][a,V,a,C] += scale * -1.00000000 * np.einsum('uIaJ,aAvI->uAvJ', h['ab'][a,C,v,C], t['ab'][pv,pV,ha,hC], optimize=True)
    O['ab'][a,V,a,C] += scale * +1.00000000 * np.einsum('uAaB,aBvI->uAvI', h['ab'][a,V,v,V], t['ab'][pv,pV,ha,hC], optimize=True)
    O['ab'][c,V,a,C] += scale * -1.00000000 * np.einsum('uv,UV,iUvI,uAwV->iAwI', eta1['a'], gamma1['b'], h['ab'][c,A,a,C], t['ab'][pa,pV,ha,hA], optimize=True)
    O['ab'][c,V,a,C] += scale * -1.00000000 * np.einsum('uv,iIvJ,uAwI->iAwJ', eta1['a'], h['ab'][c,C,a,C], t['ab'][pa,pV,ha,hC], optimize=True)
    O['ab'][c,V,a,C] += scale * +1.00000000 * np.einsum('uv,iAvB,uBwI->iAwI', eta1['a'], h['ab'][c,V,a,V], t['ab'][pa,pV,ha,hC], optimize=True)
    O['ab'][c,V,a,C] += scale * +1.00000000 * np.einsum('UV,uv,iAvV,uUwI->iAwI', eta1['b'], eta1['a'], h['ab'][c,V,a,A], t['ab'][pa,pA,ha,hC], optimize=True)
    O['ab'][c,V,a,C] += scale * +1.00000000 * np.einsum('UV,uv,iUvI,uAwV->iAwI', eta1['b'], gamma1['a'], h['ab'][c,A,a,C], t['ab'][pa,pV,ha,hA], optimize=True)
    O['ab'][c,V,a,C] += scale * +1.00000000 * np.einsum('UV,iAaV,aUuI->iAuI', eta1['b'], h['ab'][c,V,v,A], t['ab'][pv,pA,ha,hC], optimize=True)
    O['ab'][c,V,a,C] += scale * -1.00000000 * np.einsum('UV,uv,iAvV,uUwI->iAwI', gamma1['b'], gamma1['a'], h['ab'][c,V,a,A], t['ab'][pa,pA,ha,hC], optimize=True)
    O['ab'][c,V,a,C] += scale * -1.00000000 * np.einsum('UV,iUaI,aAuV->iAuI', gamma1['b'], h['ab'][c,A,v,C], t['ab'][pv,pV,ha,hA], optimize=True)
    O['ab'][c,V,a,C] += scale * -1.00000000 * np.einsum('iIaJ,aAuI->iAuJ', h['ab'][c,C,v,C], t['ab'][pv,pV,ha,hC], optimize=True)
    O['ab'][c,V,a,C] += scale * +1.00000000 * np.einsum('iAaB,aBuI->iAuI', h['ab'][c,V,v,V], t['ab'][pv,pV,ha,hC], optimize=True)
    O['ab'][v,V,a,C] += scale * +1.00000000 * np.einsum('uv,UV,uAwV,aUvI->aAwI', eta1['a'], gamma1['b'], h['ab'][a,V,a,A], t['ab'][pv,pA,ha,hC], optimize=True)
    O['ab'][v,V,a,C] += scale * -1.00000000 * np.einsum('uv,UV,aUvI,uAwV->aAwI', eta1['a'], gamma1['b'], h['ab'][v,A,a,C], t['ab'][pa,pV,ha,hA], optimize=True)
    O['ab'][v,V,a,C] += scale * -1.00000000 * np.einsum('uv,aIvJ,uAwI->aAwJ', eta1['a'], h['ab'][v,C,a,C], t['ab'][pa,pV,ha,hC], optimize=True)
    O['ab'][v,V,a,C] += scale * +1.00000000 * np.einsum('uv,aAvB,uBwI->aAwI', eta1['a'], h['ab'][v,V,a,V], t['ab'][pa,pV,ha,hC], optimize=True)
    O['ab'][v,V,a,C] += scale * -1.00000000 * np.einsum('UV,uv,uUwI,aAvV->aAwI', eta1['b'], eta1['a'], h['ab'][a,A,a,C], t['ab'][pv,pV,ha,hA], optimize=True)
    O['ab'][v,V,a,C] += scale * +1.00000000 * np.einsum('UV,uv,aAvV,uUwI->aAwI', eta1['b'], eta1['a'], h['ab'][v,V,a,A], t['ab'][pa,pA,ha,hC], optimize=True)
    O['ab'][v,V,a,C] += scale * -1.00000000 * np.einsum('UV,uv,uAwV,aUvI->aAwI', eta1['b'], gamma1['a'], h['ab'][a,V,a,A], t['ab'][pv,pA,ha,hC], optimize=True)
    O['ab'][v,V,a,C] += scale * +1.00000000 * np.einsum('UV,uv,aUvI,uAwV->aAwI', eta1['b'], gamma1['a'], h['ab'][v,A,a,C], t['ab'][pa,pV,ha,hA], optimize=True)
    O['ab'][v,V,a,C] += scale * -1.00000000 * np.einsum('UV,iAuV,aUiI->aAuI', eta1['b'], h['ab'][c,V,a,A], t['ab'][pv,pA,hc,hC], optimize=True)
    O['ab'][v,V,a,C] += scale * +1.00000000 * np.einsum('UV,aAbV,bUuI->aAuI', eta1['b'], h['ab'][v,V,v,A], t['ab'][pv,pA,ha,hC], optimize=True)
    O['ab'][v,V,a,C] += scale * +1.00000000 * np.einsum('uv,uIwJ,aAvI->aAwJ', gamma1['a'], h['ab'][a,C,a,C], t['ab'][pv,pV,ha,hC], optimize=True)
    O['ab'][v,V,a,C] += scale * -1.00000000 * np.einsum('uv,uAwB,aBvI->aAwI', gamma1['a'], h['ab'][a,V,a,V], t['ab'][pv,pV,ha,hC], optimize=True)
    O['ab'][v,V,a,C] += scale * +1.00000000 * np.einsum('UV,uv,uUwI,aAvV->aAwI', gamma1['b'], gamma1['a'], h['ab'][a,A,a,C], t['ab'][pv,pV,ha,hA], optimize=True)
    O['ab'][v,V,a,C] += scale * -1.00000000 * np.einsum('UV,uv,aAvV,uUwI->aAwI', gamma1['b'], gamma1['a'], h['ab'][v,V,a,A], t['ab'][pa,pA,ha,hC], optimize=True)
    O['ab'][v,V,a,C] += scale * +1.00000000 * np.einsum('UV,iUuI,aAiV->aAuI', gamma1['b'], h['ab'][c,A,a,C], t['ab'][pv,pV,hc,hA], optimize=True)
    O['ab'][v,V,a,C] += scale * -1.00000000 * np.einsum('UV,aUbI,bAuV->aAuI', gamma1['b'], h['ab'][v,A,v,C], t['ab'][pv,pV,ha,hA], optimize=True)
    O['ab'][v,V,a,C] += scale * +1.00000000 * np.einsum('iIuJ,aAiI->aAuJ', h['ab'][c,C,a,C], t['ab'][pv,pV,hc,hC], optimize=True)
    O['ab'][v,V,a,C] += scale * -1.00000000 * np.einsum('iAuB,aBiI->aAuI', h['ab'][c,V,a,V], t['ab'][pv,pV,hc,hC], optimize=True)
    O['ab'][v,V,a,C] += scale * -1.00000000 * np.einsum('aIbJ,bAuI->aAuJ', h['ab'][v,C,v,C], t['ab'][pv,pV,ha,hC], optimize=True)
    O['ab'][v,V,a,C] += scale * +1.00000000 * np.einsum('aAbB,bBuI->aAuI', h['ab'][v,V,v,V], t['ab'][pv,pV,ha,hC], optimize=True)
    
    O['ab'][a,A,a,V] += scale * -1.00000000 * np.einsum('uv,wIvA,uUxI->wUxA', eta1['a'], h['ab'][a,C,a,V], t['ab'][pa,pA,ha,hC], optimize=True)
    
    
    O['ab'][a,A,a,V] += scale * +1.00000000 * np.einsum('uv,uIwA,xUvI->xUwA', gamma1['a'], h['ab'][a,C,a,V], t['ab'][pa,pA,ha,hC], optimize=True)
    
    O['ab'][a,A,a,V] += scale * +1.00000000 * np.einsum('UV,iUuA,vWiV->vWuA', gamma1['b'], h['ab'][c,A,a,V], t['ab'][pa,pA,hc,hA], optimize=True)
    O['ab'][a,A,a,V] += scale * -1.00000000 * np.einsum('UV,uUaA,aWvV->uWvA', gamma1['b'], h['ab'][a,A,v,V], t['ab'][pv,pA,ha,hA], optimize=True)
    O['ab'][a,A,a,V] += scale * +1.00000000 * np.einsum('iIuA,vUiI->vUuA', h['ab'][c,C,a,V], t['ab'][pa,pA,hc,hC], optimize=True)
    O['ab'][a,A,a,V] += scale * -1.00000000 * np.einsum('uIaA,aUvI->uUvA', h['ab'][a,C,v,V], t['ab'][pv,pA,ha,hC], optimize=True)
    
    O['ab'][c,A,a,V] += scale * -1.00000000 * np.einsum('uv,iIvA,uUwI->iUwA', eta1['a'], h['ab'][c,C,a,V], t['ab'][pa,pA,ha,hC], optimize=True)
    
    O['ab'][c,A,a,V] += scale * -1.00000000 * np.einsum('UV,iUaA,aWuV->iWuA', gamma1['b'], h['ab'][c,A,v,V], t['ab'][pv,pA,ha,hA], optimize=True)
    O['ab'][c,A,a,V] += scale * -1.00000000 * np.einsum('iIaA,aUuI->iUuA', h['ab'][c,C,v,V], t['ab'][pv,pA,ha,hC], optimize=True)
    
    O['ab'][v,A,a,V] += scale * -1.00000000 * np.einsum('uv,aIvA,uUwI->aUwA', eta1['a'], h['ab'][v,C,a,V], t['ab'][pa,pA,ha,hC], optimize=True)
    O['ab'][v,A,a,V] += scale * -1.00000000 * np.einsum('UV,uv,uUwA,aWvV->aWwA', eta1['b'], eta1['a'], h['ab'][a,A,a,V], t['ab'][pv,pA,ha,hA], optimize=True)
    
    O['ab'][v,A,a,V] += scale * +1.00000000 * np.einsum('uv,uIwA,aUvI->aUwA', gamma1['a'], h['ab'][a,C,a,V], t['ab'][pv,pA,ha,hC], optimize=True)
    O['ab'][v,A,a,V] += scale * +1.00000000 * np.einsum('UV,uv,uUwA,aWvV->aWwA', gamma1['b'], gamma1['a'], h['ab'][a,A,a,V], t['ab'][pv,pA,ha,hA], optimize=True)
    O['ab'][v,A,a,V] += scale * +1.00000000 * np.einsum('UV,iUuA,aWiV->aWuA', gamma1['b'], h['ab'][c,A,a,V], t['ab'][pv,pA,hc,hA], optimize=True)
    O['ab'][v,A,a,V] += scale * -1.00000000 * np.einsum('UV,aUbA,bWuV->aWuA', gamma1['b'], h['ab'][v,A,v,V], t['ab'][pv,pA,ha,hA], optimize=True)
    O['ab'][v,A,a,V] += scale * +1.00000000 * np.einsum('iIuA,aUiI->aUuA', h['ab'][c,C,a,V], t['ab'][pv,pA,hc,hC], optimize=True)
    O['ab'][v,A,a,V] += scale * -1.00000000 * np.einsum('aIbA,bUuI->aUuA', h['ab'][v,C,v,V], t['ab'][pv,pA,ha,hC], optimize=True)
    O['ab'][a,V,a,V] += scale * -1.00000000 * np.einsum('uv,UV,wUvA,uBxV->wBxA', eta1['a'], gamma1['b'], h['ab'][a,A,a,V], t['ab'][pa,pV,ha,hA], optimize=True)
    O['ab'][a,V,a,V] += scale * -1.00000000 * np.einsum('uv,wIvA,uBxI->wBxA', eta1['a'], h['ab'][a,C,a,V], t['ab'][pa,pV,ha,hC], optimize=True)
    O['ab'][a,V,a,V] += scale * -1.00000000 * np.einsum('UV,uv,uUwA,xBvV->xBwA', eta1['b'], eta1['a'], h['ab'][a,A,a,V], t['ab'][pa,pV,ha,hA], optimize=True)
    O['ab'][a,V,a,V] += scale * +1.00000000 * np.einsum('UV,uv,wUvA,uBxV->wBxA', eta1['b'], gamma1['a'], h['ab'][a,A,a,V], t['ab'][pa,pV,ha,hA], optimize=True)
    O['ab'][a,V,a,V] += scale * +1.00000000 * np.einsum('uv,uIwA,xBvI->xBwA', gamma1['a'], h['ab'][a,C,a,V], t['ab'][pa,pV,ha,hC], optimize=True)
    O['ab'][a,V,a,V] += scale * +1.00000000 * np.einsum('UV,uv,uUwA,xBvV->xBwA', gamma1['b'], gamma1['a'], h['ab'][a,A,a,V], t['ab'][pa,pV,ha,hA], optimize=True)
    O['ab'][a,V,a,V] += scale * +1.00000000 * np.einsum('UV,iUuA,vBiV->vBuA', gamma1['b'], h['ab'][c,A,a,V], t['ab'][pa,pV,hc,hA], optimize=True)
    O['ab'][a,V,a,V] += scale * -1.00000000 * np.einsum('UV,uUaA,aBvV->uBvA', gamma1['b'], h['ab'][a,A,v,V], t['ab'][pv,pV,ha,hA], optimize=True)
    O['ab'][a,V,a,V] += scale * +1.00000000 * np.einsum('iIuA,vBiI->vBuA', h['ab'][c,C,a,V], t['ab'][pa,pV,hc,hC], optimize=True)
    O['ab'][a,V,a,V] += scale * -1.00000000 * np.einsum('uIaA,aBvI->uBvA', h['ab'][a,C,v,V], t['ab'][pv,pV,ha,hC], optimize=True)
    O['ab'][c,V,a,V] += scale * -1.00000000 * np.einsum('uv,UV,iUvA,uBwV->iBwA', eta1['a'], gamma1['b'], h['ab'][c,A,a,V], t['ab'][pa,pV,ha,hA], optimize=True)
    O['ab'][c,V,a,V] += scale * -1.00000000 * np.einsum('uv,iIvA,uBwI->iBwA', eta1['a'], h['ab'][c,C,a,V], t['ab'][pa,pV,ha,hC], optimize=True)
    O['ab'][c,V,a,V] += scale * +1.00000000 * np.einsum('UV,uv,iUvA,uBwV->iBwA', eta1['b'], gamma1['a'], h['ab'][c,A,a,V], t['ab'][pa,pV,ha,hA], optimize=True)
    O['ab'][c,V,a,V] += scale * -1.00000000 * np.einsum('UV,iUaA,aBuV->iBuA', gamma1['b'], h['ab'][c,A,v,V], t['ab'][pv,pV,ha,hA], optimize=True)
    O['ab'][c,V,a,V] += scale * -1.00000000 * np.einsum('iIaA,aBuI->iBuA', h['ab'][c,C,v,V], t['ab'][pv,pV,ha,hC], optimize=True)
    O['ab'][v,V,a,V] += scale * -1.00000000 * np.einsum('uv,UV,aUvA,uBwV->aBwA', eta1['a'], gamma1['b'], h['ab'][v,A,a,V], t['ab'][pa,pV,ha,hA], optimize=True)
    O['ab'][v,V,a,V] += scale * -1.00000000 * np.einsum('uv,aIvA,uBwI->aBwA', eta1['a'], h['ab'][v,C,a,V], t['ab'][pa,pV,ha,hC], optimize=True)
    O['ab'][v,V,a,V] += scale * -1.00000000 * np.einsum('UV,uv,uUwA,aBvV->aBwA', eta1['b'], eta1['a'], h['ab'][a,A,a,V], t['ab'][pv,pV,ha,hA], optimize=True)
    O['ab'][v,V,a,V] += scale * +1.00000000 * np.einsum('UV,uv,aUvA,uBwV->aBwA', eta1['b'], gamma1['a'], h['ab'][v,A,a,V], t['ab'][pa,pV,ha,hA], optimize=True)
    O['ab'][v,V,a,V] += scale * +1.00000000 * np.einsum('uv,uIwA,aBvI->aBwA', gamma1['a'], h['ab'][a,C,a,V], t['ab'][pv,pV,ha,hC], optimize=True)
    O['ab'][v,V,a,V] += scale * +1.00000000 * np.einsum('UV,uv,uUwA,aBvV->aBwA', gamma1['b'], gamma1['a'], h['ab'][a,A,a,V], t['ab'][pv,pV,ha,hA], optimize=True)
    O['ab'][v,V,a,V] += scale * +1.00000000 * np.einsum('UV,iUuA,aBiV->aBuA', gamma1['b'], h['ab'][c,A,a,V], t['ab'][pv,pV,hc,hA], optimize=True)
    O['ab'][v,V,a,V] += scale * -1.00000000 * np.einsum('UV,aUbA,bBuV->aBuA', gamma1['b'], h['ab'][v,A,v,V], t['ab'][pv,pV,ha,hA], optimize=True)
    O['ab'][v,V,a,V] += scale * +1.00000000 * np.einsum('iIuA,aBiI->aBuA', h['ab'][c,C,a,V], t['ab'][pv,pV,hc,hC], optimize=True)
    O['ab'][v,V,a,V] += scale * -1.00000000 * np.einsum('aIbA,bBuI->aBuA', h['ab'][v,C,v,V], t['ab'][pv,pV,ha,hC], optimize=True)
    O['ab'][a,A,c,A] += scale * -1.00000000 * np.einsum('uv,UV,wUvW,uXiV->wXiW', eta1['a'], gamma1['b'], h['ab'][a,A,a,A], t['ab'][pa,pA,hc,hA], optimize=True)
    
    O['ab'][a,A,c,A] += scale * -1.00000000 * np.einsum('uv,wIvU,uViI->wViU', eta1['a'], h['ab'][a,C,a,A], t['ab'][pa,pA,hc,hC], optimize=True)
    O['ab'][a,A,c,A] += scale * +1.00000000 * np.einsum('uv,wUvA,uAiV->wUiV', eta1['a'], h['ab'][a,A,a,V], t['ab'][pa,pV,hc,hA], optimize=True)
    O['ab'][a,A,c,A] += scale * +1.00000000 * np.einsum('UV,uv,wWvV,uUiX->wWiX', eta1['b'], eta1['a'], h['ab'][a,A,a,A], t['ab'][pa,pA,hc,hA], optimize=True)
    
    O['ab'][a,A,c,A] += scale * +1.00000000 * np.einsum('UV,uv,wUvW,uXiV->wXiW', eta1['b'], gamma1['a'], h['ab'][a,A,a,A], t['ab'][pa,pA,hc,hA], optimize=True)
    
    O['ab'][a,A,c,A] += scale * -1.00000000 * np.einsum('UV,iWjV,uUiX->uWjX', eta1['b'], h['ab'][c,A,c,A], t['ab'][pa,pA,hc,hA], optimize=True)
    O['ab'][a,A,c,A] += scale * +1.00000000 * np.einsum('UV,uWaV,aUiX->uWiX', eta1['b'], h['ab'][a,A,v,A], t['ab'][pv,pA,hc,hA], optimize=True)
    O['ab'][a,A,c,A] += scale * +1.00000000 * np.einsum('uv,uIiU,wVvI->wViU', gamma1['a'], h['ab'][a,C,c,A], t['ab'][pa,pA,ha,hC], optimize=True)
    O['ab'][a,A,c,A] += scale * -1.00000000 * np.einsum('uv,uUiA,wAvV->wUiV', gamma1['a'], h['ab'][a,A,c,V], t['ab'][pa,pV,ha,hA], optimize=True)
    O['ab'][a,A,c,A] += scale * -1.00000000 * np.einsum('UV,uv,wWvV,uUiX->wWiX', gamma1['b'], gamma1['a'], h['ab'][a,A,a,A], t['ab'][pa,pA,hc,hA], optimize=True)
    
    O['ab'][a,A,c,A] += scale * +1.00000000 * np.einsum('UV,iUjW,uXiV->uXjW', gamma1['b'], h['ab'][c,A,c,A], t['ab'][pa,pA,hc,hA], optimize=True)
    O['ab'][a,A,c,A] += scale * -1.00000000 * np.einsum('UV,uUaW,aXiV->uXiW', gamma1['b'], h['ab'][a,A,v,A], t['ab'][pv,pA,hc,hA], optimize=True)
    O['ab'][a,A,c,A] += scale * +1.00000000 * np.einsum('iIjU,uViI->uVjU', h['ab'][c,C,c,A], t['ab'][pa,pA,hc,hC], optimize=True)
    O['ab'][a,A,c,A] += scale * -1.00000000 * np.einsum('iUjA,uAiV->uUjV', h['ab'][c,A,c,V], t['ab'][pa,pV,hc,hA], optimize=True)
    O['ab'][a,A,c,A] += scale * -1.00000000 * np.einsum('uIaU,aViI->uViU', h['ab'][a,C,v,A], t['ab'][pv,pA,hc,hC], optimize=True)
    O['ab'][a,A,c,A] += scale * +1.00000000 * np.einsum('uUaA,aAiV->uUiV', h['ab'][a,A,v,V], t['ab'][pv,pV,hc,hA], optimize=True)
    O['ab'][c,A,c,A] += scale * -1.00000000 * np.einsum('uv,UV,iUvW,uXjV->iXjW', eta1['a'], gamma1['b'], h['ab'][c,A,a,A], t['ab'][pa,pA,hc,hA], optimize=True)
    O['ab'][c,A,c,A] += scale * -1.00000000 * np.einsum('uv,iIvU,uVjI->iVjU', eta1['a'], h['ab'][c,C,a,A], t['ab'][pa,pA,hc,hC], optimize=True)
    O['ab'][c,A,c,A] += scale * +1.00000000 * np.einsum('uv,iUvA,uAjV->iUjV', eta1['a'], h['ab'][c,A,a,V], t['ab'][pa,pV,hc,hA], optimize=True)
    O['ab'][c,A,c,A] += scale * +1.00000000 * np.einsum('UV,uv,iWvV,uUjX->iWjX', eta1['b'], eta1['a'], h['ab'][c,A,a,A], t['ab'][pa,pA,hc,hA], optimize=True)
    O['ab'][c,A,c,A] += scale * +1.00000000 * np.einsum('UV,uv,iUvW,uXjV->iXjW', eta1['b'], gamma1['a'], h['ab'][c,A,a,A], t['ab'][pa,pA,hc,hA], optimize=True)
    O['ab'][c,A,c,A] += scale * +1.00000000 * np.einsum('UV,iWaV,aUjX->iWjX', eta1['b'], h['ab'][c,A,v,A], t['ab'][pv,pA,hc,hA], optimize=True)
    O['ab'][c,A,c,A] += scale * -1.00000000 * np.einsum('UV,uv,iWvV,uUjX->iWjX', gamma1['b'], gamma1['a'], h['ab'][c,A,a,A], t['ab'][pa,pA,hc,hA], optimize=True)
    O['ab'][c,A,c,A] += scale * -1.00000000 * np.einsum('UV,iUaW,aXjV->iXjW', gamma1['b'], h['ab'][c,A,v,A], t['ab'][pv,pA,hc,hA], optimize=True)
    O['ab'][c,A,c,A] += scale * -1.00000000 * np.einsum('iIaU,aVjI->iVjU', h['ab'][c,C,v,A], t['ab'][pv,pA,hc,hC], optimize=True)
    O['ab'][c,A,c,A] += scale * +1.00000000 * np.einsum('iUaA,aAjV->iUjV', h['ab'][c,A,v,V], t['ab'][pv,pV,hc,hA], optimize=True)
    O['ab'][v,A,c,A] += scale * +1.00000000 * np.einsum('uv,UV,uWiV,aUvX->aWiX', eta1['a'], gamma1['b'], h['ab'][a,A,c,A], t['ab'][pv,pA,ha,hA], optimize=True)
    O['ab'][v,A,c,A] += scale * -1.00000000 * np.einsum('uv,UV,aUvW,uXiV->aXiW', eta1['a'], gamma1['b'], h['ab'][v,A,a,A], t['ab'][pa,pA,hc,hA], optimize=True)
    O['ab'][v,A,c,A] += scale * -1.00000000 * np.einsum('uv,aIvU,uViI->aViU', eta1['a'], h['ab'][v,C,a,A], t['ab'][pa,pA,hc,hC], optimize=True)
    O['ab'][v,A,c,A] += scale * +1.00000000 * np.einsum('uv,aUvA,uAiV->aUiV', eta1['a'], h['ab'][v,A,a,V], t['ab'][pa,pV,hc,hA], optimize=True)
    O['ab'][v,A,c,A] += scale * -1.00000000 * np.einsum('UV,uv,uUiW,aXvV->aXiW', eta1['b'], eta1['a'], h['ab'][a,A,c,A], t['ab'][pv,pA,ha,hA], optimize=True)
    O['ab'][v,A,c,A] += scale * +1.00000000 * np.einsum('UV,uv,aWvV,uUiX->aWiX', eta1['b'], eta1['a'], h['ab'][v,A,a,A], t['ab'][pa,pA,hc,hA], optimize=True)
    O['ab'][v,A,c,A] += scale * -1.00000000 * np.einsum('UV,uv,uWiV,aUvX->aWiX', eta1['b'], gamma1['a'], h['ab'][a,A,c,A], t['ab'][pv,pA,ha,hA], optimize=True)
    O['ab'][v,A,c,A] += scale * +1.00000000 * np.einsum('UV,uv,aUvW,uXiV->aXiW', eta1['b'], gamma1['a'], h['ab'][v,A,a,A], t['ab'][pa,pA,hc,hA], optimize=True)
    O['ab'][v,A,c,A] += scale * -1.00000000 * np.einsum('UV,iWjV,aUiX->aWjX', eta1['b'], h['ab'][c,A,c,A], t['ab'][pv,pA,hc,hA], optimize=True)
    O['ab'][v,A,c,A] += scale * +1.00000000 * np.einsum('UV,aWbV,bUiX->aWiX', eta1['b'], h['ab'][v,A,v,A], t['ab'][pv,pA,hc,hA], optimize=True)
    O['ab'][v,A,c,A] += scale * +1.00000000 * np.einsum('uv,uIiU,aVvI->aViU', gamma1['a'], h['ab'][a,C,c,A], t['ab'][pv,pA,ha,hC], optimize=True)
    O['ab'][v,A,c,A] += scale * -1.00000000 * np.einsum('uv,uUiA,aAvV->aUiV', gamma1['a'], h['ab'][a,A,c,V], t['ab'][pv,pV,ha,hA], optimize=True)
    O['ab'][v,A,c,A] += scale * +1.00000000 * np.einsum('UV,uv,uUiW,aXvV->aXiW', gamma1['b'], gamma1['a'], h['ab'][a,A,c,A], t['ab'][pv,pA,ha,hA], optimize=True)
    O['ab'][v,A,c,A] += scale * -1.00000000 * np.einsum('UV,uv,aWvV,uUiX->aWiX', gamma1['b'], gamma1['a'], h['ab'][v,A,a,A], t['ab'][pa,pA,hc,hA], optimize=True)
    O['ab'][v,A,c,A] += scale * +1.00000000 * np.einsum('UV,iUjW,aXiV->aXjW', gamma1['b'], h['ab'][c,A,c,A], t['ab'][pv,pA,hc,hA], optimize=True)
    O['ab'][v,A,c,A] += scale * -1.00000000 * np.einsum('UV,aUbW,bXiV->aXiW', gamma1['b'], h['ab'][v,A,v,A], t['ab'][pv,pA,hc,hA], optimize=True)
    O['ab'][v,A,c,A] += scale * +1.00000000 * np.einsum('iIjU,aViI->aVjU', h['ab'][c,C,c,A], t['ab'][pv,pA,hc,hC], optimize=True)
    O['ab'][v,A,c,A] += scale * -1.00000000 * np.einsum('iUjA,aAiV->aUjV', h['ab'][c,A,c,V], t['ab'][pv,pV,hc,hA], optimize=True)
    O['ab'][v,A,c,A] += scale * -1.00000000 * np.einsum('aIbU,bViI->aViU', h['ab'][v,C,v,A], t['ab'][pv,pA,hc,hC], optimize=True)
    O['ab'][v,A,c,A] += scale * +1.00000000 * np.einsum('aUbA,bAiV->aUiV', h['ab'][v,A,v,V], t['ab'][pv,pV,hc,hA], optimize=True)
    
    O['ab'][a,C,c,A] += scale * +1.00000000 * np.einsum('uv,wIvA,uAiU->wIiU', eta1['a'], h['ab'][a,C,a,V], t['ab'][pa,pV,hc,hA], optimize=True)
    O['ab'][a,C,c,A] += scale * +1.00000000 * np.einsum('UV,uv,wIvV,uUiW->wIiW', eta1['b'], eta1['a'], h['ab'][a,C,a,A], t['ab'][pa,pA,hc,hA], optimize=True)
    
    O['ab'][a,C,c,A] += scale * -1.00000000 * np.einsum('UV,iIjV,uUiW->uIjW', eta1['b'], h['ab'][c,C,c,A], t['ab'][pa,pA,hc,hA], optimize=True)
    O['ab'][a,C,c,A] += scale * +1.00000000 * np.einsum('UV,uIaV,aUiW->uIiW', eta1['b'], h['ab'][a,C,v,A], t['ab'][pv,pA,hc,hA], optimize=True)
    O['ab'][a,C,c,A] += scale * -1.00000000 * np.einsum('uv,uIiA,wAvU->wIiU', gamma1['a'], h['ab'][a,C,c,V], t['ab'][pa,pV,ha,hA], optimize=True)
    O['ab'][a,C,c,A] += scale * -1.00000000 * np.einsum('UV,uv,wIvV,uUiW->wIiW', gamma1['b'], gamma1['a'], h['ab'][a,C,a,A], t['ab'][pa,pA,hc,hA], optimize=True)
    O['ab'][a,C,c,A] += scale * -1.00000000 * np.einsum('iIjA,uAiU->uIjU', h['ab'][c,C,c,V], t['ab'][pa,pV,hc,hA], optimize=True)
    O['ab'][a,C,c,A] += scale * +1.00000000 * np.einsum('uIaA,aAiU->uIiU', h['ab'][a,C,v,V], t['ab'][pv,pV,hc,hA], optimize=True)
    O['ab'][c,C,c,A] += scale * +1.00000000 * np.einsum('uv,iIvA,uAjU->iIjU', eta1['a'], h['ab'][c,C,a,V], t['ab'][pa,pV,hc,hA], optimize=True)
    O['ab'][c,C,c,A] += scale * +1.00000000 * np.einsum('UV,uv,iIvV,uUjW->iIjW', eta1['b'], eta1['a'], h['ab'][c,C,a,A], t['ab'][pa,pA,hc,hA], optimize=True)
    O['ab'][c,C,c,A] += scale * +1.00000000 * np.einsum('UV,iIaV,aUjW->iIjW', eta1['b'], h['ab'][c,C,v,A], t['ab'][pv,pA,hc,hA], optimize=True)
    O['ab'][c,C,c,A] += scale * -1.00000000 * np.einsum('UV,uv,iIvV,uUjW->iIjW', gamma1['b'], gamma1['a'], h['ab'][c,C,a,A], t['ab'][pa,pA,hc,hA], optimize=True)
    O['ab'][c,C,c,A] += scale * +1.00000000 * np.einsum('iIaA,aAjU->iIjU', h['ab'][c,C,v,V], t['ab'][pv,pV,hc,hA], optimize=True)
    O['ab'][v,C,c,A] += scale * +1.00000000 * np.einsum('uv,UV,uIiV,aUvW->aIiW', eta1['a'], gamma1['b'], h['ab'][a,C,c,A], t['ab'][pv,pA,ha,hA], optimize=True)
    O['ab'][v,C,c,A] += scale * +1.00000000 * np.einsum('uv,aIvA,uAiU->aIiU', eta1['a'], h['ab'][v,C,a,V], t['ab'][pa,pV,hc,hA], optimize=True)
    O['ab'][v,C,c,A] += scale * +1.00000000 * np.einsum('UV,uv,aIvV,uUiW->aIiW', eta1['b'], eta1['a'], h['ab'][v,C,a,A], t['ab'][pa,pA,hc,hA], optimize=True)
    O['ab'][v,C,c,A] += scale * -1.00000000 * np.einsum('UV,uv,uIiV,aUvW->aIiW', eta1['b'], gamma1['a'], h['ab'][a,C,c,A], t['ab'][pv,pA,ha,hA], optimize=True)
    O['ab'][v,C,c,A] += scale * -1.00000000 * np.einsum('UV,iIjV,aUiW->aIjW', eta1['b'], h['ab'][c,C,c,A], t['ab'][pv,pA,hc,hA], optimize=True)
    O['ab'][v,C,c,A] += scale * +1.00000000 * np.einsum('UV,aIbV,bUiW->aIiW', eta1['b'], h['ab'][v,C,v,A], t['ab'][pv,pA,hc,hA], optimize=True)
    O['ab'][v,C,c,A] += scale * -1.00000000 * np.einsum('uv,uIiA,aAvU->aIiU', gamma1['a'], h['ab'][a,C,c,V], t['ab'][pv,pV,ha,hA], optimize=True)
    O['ab'][v,C,c,A] += scale * -1.00000000 * np.einsum('UV,uv,aIvV,uUiW->aIiW', gamma1['b'], gamma1['a'], h['ab'][v,C,a,A], t['ab'][pa,pA,hc,hA], optimize=True)
    O['ab'][v,C,c,A] += scale * -1.00000000 * np.einsum('iIjA,aAiU->aIjU', h['ab'][c,C,c,V], t['ab'][pv,pV,hc,hA], optimize=True)
    O['ab'][v,C,c,A] += scale * +1.00000000 * np.einsum('aIbA,bAiU->aIiU', h['ab'][v,C,v,V], t['ab'][pv,pV,hc,hA], optimize=True)
    O['ab'][a,V,c,A] += scale * -1.00000000 * np.einsum('uv,UV,wUvW,uAiV->wAiW', eta1['a'], gamma1['b'], h['ab'][a,A,a,A], t['ab'][pa,pV,hc,hA], optimize=True)
    
    O['ab'][a,V,c,A] += scale * -1.00000000 * np.einsum('uv,wIvU,uAiI->wAiU', eta1['a'], h['ab'][a,C,a,A], t['ab'][pa,pV,hc,hC], optimize=True)
    O['ab'][a,V,c,A] += scale * +1.00000000 * np.einsum('uv,wAvB,uBiU->wAiU', eta1['a'], h['ab'][a,V,a,V], t['ab'][pa,pV,hc,hA], optimize=True)
    O['ab'][a,V,c,A] += scale * +1.00000000 * np.einsum('UV,uv,wAvV,uUiW->wAiW', eta1['b'], eta1['a'], h['ab'][a,V,a,A], t['ab'][pa,pA,hc,hA], optimize=True)
    O['ab'][a,V,c,A] += scale * -1.00000000 * np.einsum('UV,uv,uUiW,wAvV->wAiW', eta1['b'], eta1['a'], h['ab'][a,A,c,A], t['ab'][pa,pV,ha,hA], optimize=True)
    O['ab'][a,V,c,A] += scale * +1.00000000 * np.einsum('UV,uv,wUvW,uAiV->wAiW', eta1['b'], gamma1['a'], h['ab'][a,A,a,A], t['ab'][pa,pV,hc,hA], optimize=True)
    
    O['ab'][a,V,c,A] += scale * -1.00000000 * np.einsum('UV,iAjV,uUiW->uAjW', eta1['b'], h['ab'][c,V,c,A], t['ab'][pa,pA,hc,hA], optimize=True)
    O['ab'][a,V,c,A] += scale * +1.00000000 * np.einsum('UV,uAaV,aUiW->uAiW', eta1['b'], h['ab'][a,V,v,A], t['ab'][pv,pA,hc,hA], optimize=True)
    O['ab'][a,V,c,A] += scale * +1.00000000 * np.einsum('uv,uIiU,wAvI->wAiU', gamma1['a'], h['ab'][a,C,c,A], t['ab'][pa,pV,ha,hC], optimize=True)
    O['ab'][a,V,c,A] += scale * -1.00000000 * np.einsum('uv,uAiB,wBvU->wAiU', gamma1['a'], h['ab'][a,V,c,V], t['ab'][pa,pV,ha,hA], optimize=True)
    O['ab'][a,V,c,A] += scale * -1.00000000 * np.einsum('UV,uv,wAvV,uUiW->wAiW', gamma1['b'], gamma1['a'], h['ab'][a,V,a,A], t['ab'][pa,pA,hc,hA], optimize=True)
    O['ab'][a,V,c,A] += scale * +1.00000000 * np.einsum('UV,uv,uUiW,wAvV->wAiW', gamma1['b'], gamma1['a'], h['ab'][a,A,c,A], t['ab'][pa,pV,ha,hA], optimize=True)
    O['ab'][a,V,c,A] += scale * +1.00000000 * np.einsum('UV,iUjW,uAiV->uAjW', gamma1['b'], h['ab'][c,A,c,A], t['ab'][pa,pV,hc,hA], optimize=True)
    O['ab'][a,V,c,A] += scale * -1.00000000 * np.einsum('UV,uUaW,aAiV->uAiW', gamma1['b'], h['ab'][a,A,v,A], t['ab'][pv,pV,hc,hA], optimize=True)
    O['ab'][a,V,c,A] += scale * +1.00000000 * np.einsum('iIjU,uAiI->uAjU', h['ab'][c,C,c,A], t['ab'][pa,pV,hc,hC], optimize=True)
    O['ab'][a,V,c,A] += scale * -1.00000000 * np.einsum('iAjB,uBiU->uAjU', h['ab'][c,V,c,V], t['ab'][pa,pV,hc,hA], optimize=True)
    O['ab'][a,V,c,A] += scale * -1.00000000 * np.einsum('uIaU,aAiI->uAiU', h['ab'][a,C,v,A], t['ab'][pv,pV,hc,hC], optimize=True)
    O['ab'][a,V,c,A] += scale * +1.00000000 * np.einsum('uAaB,aBiU->uAiU', h['ab'][a,V,v,V], t['ab'][pv,pV,hc,hA], optimize=True)
    O['ab'][c,V,c,A] += scale * -1.00000000 * np.einsum('uv,UV,iUvW,uAjV->iAjW', eta1['a'], gamma1['b'], h['ab'][c,A,a,A], t['ab'][pa,pV,hc,hA], optimize=True)
    O['ab'][c,V,c,A] += scale * -1.00000000 * np.einsum('uv,iIvU,uAjI->iAjU', eta1['a'], h['ab'][c,C,a,A], t['ab'][pa,pV,hc,hC], optimize=True)
    O['ab'][c,V,c,A] += scale * +1.00000000 * np.einsum('uv,iAvB,uBjU->iAjU', eta1['a'], h['ab'][c,V,a,V], t['ab'][pa,pV,hc,hA], optimize=True)
    O['ab'][c,V,c,A] += scale * +1.00000000 * np.einsum('UV,uv,iAvV,uUjW->iAjW', eta1['b'], eta1['a'], h['ab'][c,V,a,A], t['ab'][pa,pA,hc,hA], optimize=True)
    O['ab'][c,V,c,A] += scale * +1.00000000 * np.einsum('UV,uv,iUvW,uAjV->iAjW', eta1['b'], gamma1['a'], h['ab'][c,A,a,A], t['ab'][pa,pV,hc,hA], optimize=True)
    O['ab'][c,V,c,A] += scale * +1.00000000 * np.einsum('UV,iAaV,aUjW->iAjW', eta1['b'], h['ab'][c,V,v,A], t['ab'][pv,pA,hc,hA], optimize=True)
    O['ab'][c,V,c,A] += scale * -1.00000000 * np.einsum('UV,uv,iAvV,uUjW->iAjW', gamma1['b'], gamma1['a'], h['ab'][c,V,a,A], t['ab'][pa,pA,hc,hA], optimize=True)
    O['ab'][c,V,c,A] += scale * -1.00000000 * np.einsum('UV,iUaW,aAjV->iAjW', gamma1['b'], h['ab'][c,A,v,A], t['ab'][pv,pV,hc,hA], optimize=True)
    O['ab'][c,V,c,A] += scale * -1.00000000 * np.einsum('iIaU,aAjI->iAjU', h['ab'][c,C,v,A], t['ab'][pv,pV,hc,hC], optimize=True)
    O['ab'][c,V,c,A] += scale * +1.00000000 * np.einsum('iAaB,aBjU->iAjU', h['ab'][c,V,v,V], t['ab'][pv,pV,hc,hA], optimize=True)
    O['ab'][v,V,c,A] += scale * +1.00000000 * np.einsum('uv,UV,uAiV,aUvW->aAiW', eta1['a'], gamma1['b'], h['ab'][a,V,c,A], t['ab'][pv,pA,ha,hA], optimize=True)
    O['ab'][v,V,c,A] += scale * -1.00000000 * np.einsum('uv,UV,aUvW,uAiV->aAiW', eta1['a'], gamma1['b'], h['ab'][v,A,a,A], t['ab'][pa,pV,hc,hA], optimize=True)
    O['ab'][v,V,c,A] += scale * -1.00000000 * np.einsum('uv,aIvU,uAiI->aAiU', eta1['a'], h['ab'][v,C,a,A], t['ab'][pa,pV,hc,hC], optimize=True)
    O['ab'][v,V,c,A] += scale * +1.00000000 * np.einsum('uv,aAvB,uBiU->aAiU', eta1['a'], h['ab'][v,V,a,V], t['ab'][pa,pV,hc,hA], optimize=True)
    O['ab'][v,V,c,A] += scale * -1.00000000 * np.einsum('UV,uv,uUiW,aAvV->aAiW', eta1['b'], eta1['a'], h['ab'][a,A,c,A], t['ab'][pv,pV,ha,hA], optimize=True)
    O['ab'][v,V,c,A] += scale * +1.00000000 * np.einsum('UV,uv,aAvV,uUiW->aAiW', eta1['b'], eta1['a'], h['ab'][v,V,a,A], t['ab'][pa,pA,hc,hA], optimize=True)
    O['ab'][v,V,c,A] += scale * -1.00000000 * np.einsum('UV,uv,uAiV,aUvW->aAiW', eta1['b'], gamma1['a'], h['ab'][a,V,c,A], t['ab'][pv,pA,ha,hA], optimize=True)
    O['ab'][v,V,c,A] += scale * +1.00000000 * np.einsum('UV,uv,aUvW,uAiV->aAiW', eta1['b'], gamma1['a'], h['ab'][v,A,a,A], t['ab'][pa,pV,hc,hA], optimize=True)
    O['ab'][v,V,c,A] += scale * -1.00000000 * np.einsum('UV,iAjV,aUiW->aAjW', eta1['b'], h['ab'][c,V,c,A], t['ab'][pv,pA,hc,hA], optimize=True)
    O['ab'][v,V,c,A] += scale * +1.00000000 * np.einsum('UV,aAbV,bUiW->aAiW', eta1['b'], h['ab'][v,V,v,A], t['ab'][pv,pA,hc,hA], optimize=True)
    O['ab'][v,V,c,A] += scale * +1.00000000 * np.einsum('uv,uIiU,aAvI->aAiU', gamma1['a'], h['ab'][a,C,c,A], t['ab'][pv,pV,ha,hC], optimize=True)
    O['ab'][v,V,c,A] += scale * -1.00000000 * np.einsum('uv,uAiB,aBvU->aAiU', gamma1['a'], h['ab'][a,V,c,V], t['ab'][pv,pV,ha,hA], optimize=True)
    O['ab'][v,V,c,A] += scale * +1.00000000 * np.einsum('UV,uv,uUiW,aAvV->aAiW', gamma1['b'], gamma1['a'], h['ab'][a,A,c,A], t['ab'][pv,pV,ha,hA], optimize=True)
    O['ab'][v,V,c,A] += scale * -1.00000000 * np.einsum('UV,uv,aAvV,uUiW->aAiW', gamma1['b'], gamma1['a'], h['ab'][v,V,a,A], t['ab'][pa,pA,hc,hA], optimize=True)
    O['ab'][v,V,c,A] += scale * +1.00000000 * np.einsum('UV,iUjW,aAiV->aAjW', gamma1['b'], h['ab'][c,A,c,A], t['ab'][pv,pV,hc,hA], optimize=True)
    O['ab'][v,V,c,A] += scale * -1.00000000 * np.einsum('UV,aUbW,bAiV->aAiW', gamma1['b'], h['ab'][v,A,v,A], t['ab'][pv,pV,hc,hA], optimize=True)
    O['ab'][v,V,c,A] += scale * +1.00000000 * np.einsum('iIjU,aAiI->aAjU', h['ab'][c,C,c,A], t['ab'][pv,pV,hc,hC], optimize=True)
    O['ab'][v,V,c,A] += scale * -1.00000000 * np.einsum('iAjB,aBiU->aAjU', h['ab'][c,V,c,V], t['ab'][pv,pV,hc,hA], optimize=True)
    O['ab'][v,V,c,A] += scale * -1.00000000 * np.einsum('aIbU,bAiI->aAiU', h['ab'][v,C,v,A], t['ab'][pv,pV,hc,hC], optimize=True)
    O['ab'][v,V,c,A] += scale * +1.00000000 * np.einsum('aAbB,bBiU->aAiU', h['ab'][v,V,v,V], t['ab'][pv,pV,hc,hA], optimize=True)
    O['ab'][a,A,c,C] += scale * -1.00000000 * np.einsum('uv,UV,wUvI,uWiV->wWiI', eta1['a'], gamma1['b'], h['ab'][a,A,a,C], t['ab'][pa,pA,hc,hA], optimize=True)
    O['ab'][a,A,c,C] += scale * +1.00000000 * np.einsum('uv,UV,uWiV,wUvI->wWiI', eta1['a'], gamma1['b'], h['ab'][a,A,c,A], t['ab'][pa,pA,ha,hC], optimize=True)
    O['ab'][a,A,c,C] += scale * -1.00000000 * np.einsum('uv,wIvJ,uUiI->wUiJ', eta1['a'], h['ab'][a,C,a,C], t['ab'][pa,pA,hc,hC], optimize=True)
    O['ab'][a,A,c,C] += scale * +1.00000000 * np.einsum('uv,wUvA,uAiI->wUiI', eta1['a'], h['ab'][a,A,a,V], t['ab'][pa,pV,hc,hC], optimize=True)
    O['ab'][a,A,c,C] += scale * +1.00000000 * np.einsum('UV,uv,wWvV,uUiI->wWiI', eta1['b'], eta1['a'], h['ab'][a,A,a,A], t['ab'][pa,pA,hc,hC], optimize=True)
    
    O['ab'][a,A,c,C] += scale * +1.00000000 * np.einsum('UV,uv,wUvI,uWiV->wWiI', eta1['b'], gamma1['a'], h['ab'][a,A,a,C], t['ab'][pa,pA,hc,hA], optimize=True)
    O['ab'][a,A,c,C] += scale * -1.00000000 * np.einsum('UV,uv,uWiV,wUvI->wWiI', eta1['b'], gamma1['a'], h['ab'][a,A,c,A], t['ab'][pa,pA,ha,hC], optimize=True)
    O['ab'][a,A,c,C] += scale * -1.00000000 * np.einsum('UV,iWjV,uUiI->uWjI', eta1['b'], h['ab'][c,A,c,A], t['ab'][pa,pA,hc,hC], optimize=True)
    O['ab'][a,A,c,C] += scale * +1.00000000 * np.einsum('UV,uWaV,aUiI->uWiI', eta1['b'], h['ab'][a,A,v,A], t['ab'][pv,pA,hc,hC], optimize=True)
    O['ab'][a,A,c,C] += scale * +1.00000000 * np.einsum('uv,uIiJ,wUvI->wUiJ', gamma1['a'], h['ab'][a,C,c,C], t['ab'][pa,pA,ha,hC], optimize=True)
    O['ab'][a,A,c,C] += scale * -1.00000000 * np.einsum('uv,uUiA,wAvI->wUiI', gamma1['a'], h['ab'][a,A,c,V], t['ab'][pa,pV,ha,hC], optimize=True)
    O['ab'][a,A,c,C] += scale * -1.00000000 * np.einsum('UV,uv,wWvV,uUiI->wWiI', gamma1['b'], gamma1['a'], h['ab'][a,A,a,A], t['ab'][pa,pA,hc,hC], optimize=True)
    
    O['ab'][a,A,c,C] += scale * +1.00000000 * np.einsum('UV,iUjI,uWiV->uWjI', gamma1['b'], h['ab'][c,A,c,C], t['ab'][pa,pA,hc,hA], optimize=True)
    O['ab'][a,A,c,C] += scale * -1.00000000 * np.einsum('UV,uUaI,aWiV->uWiI', gamma1['b'], h['ab'][a,A,v,C], t['ab'][pv,pA,hc,hA], optimize=True)
    O['ab'][a,A,c,C] += scale * +1.00000000 * np.einsum('iIjJ,uUiI->uUjJ', h['ab'][c,C,c,C], t['ab'][pa,pA,hc,hC], optimize=True)
    O['ab'][a,A,c,C] += scale * -1.00000000 * np.einsum('iUjA,uAiI->uUjI', h['ab'][c,A,c,V], t['ab'][pa,pV,hc,hC], optimize=True)
    O['ab'][a,A,c,C] += scale * -1.00000000 * np.einsum('uIaJ,aUiI->uUiJ', h['ab'][a,C,v,C], t['ab'][pv,pA,hc,hC], optimize=True)
    O['ab'][a,A,c,C] += scale * +1.00000000 * np.einsum('uUaA,aAiI->uUiI', h['ab'][a,A,v,V], t['ab'][pv,pV,hc,hC], optimize=True)
    O['ab'][c,A,c,C] += scale * -1.00000000 * np.einsum('uv,UV,iUvI,uWjV->iWjI', eta1['a'], gamma1['b'], h['ab'][c,A,a,C], t['ab'][pa,pA,hc,hA], optimize=True)
    O['ab'][c,A,c,C] += scale * -1.00000000 * np.einsum('uv,iIvJ,uUjI->iUjJ', eta1['a'], h['ab'][c,C,a,C], t['ab'][pa,pA,hc,hC], optimize=True)
    O['ab'][c,A,c,C] += scale * +1.00000000 * np.einsum('uv,iUvA,uAjI->iUjI', eta1['a'], h['ab'][c,A,a,V], t['ab'][pa,pV,hc,hC], optimize=True)
    O['ab'][c,A,c,C] += scale * +1.00000000 * np.einsum('UV,uv,iWvV,uUjI->iWjI', eta1['b'], eta1['a'], h['ab'][c,A,a,A], t['ab'][pa,pA,hc,hC], optimize=True)
    O['ab'][c,A,c,C] += scale * +1.00000000 * np.einsum('UV,uv,iUvI,uWjV->iWjI', eta1['b'], gamma1['a'], h['ab'][c,A,a,C], t['ab'][pa,pA,hc,hA], optimize=True)
    O['ab'][c,A,c,C] += scale * +1.00000000 * np.einsum('UV,iWaV,aUjI->iWjI', eta1['b'], h['ab'][c,A,v,A], t['ab'][pv,pA,hc,hC], optimize=True)
    O['ab'][c,A,c,C] += scale * -1.00000000 * np.einsum('UV,uv,iWvV,uUjI->iWjI', gamma1['b'], gamma1['a'], h['ab'][c,A,a,A], t['ab'][pa,pA,hc,hC], optimize=True)
    O['ab'][c,A,c,C] += scale * -1.00000000 * np.einsum('UV,iUaI,aWjV->iWjI', gamma1['b'], h['ab'][c,A,v,C], t['ab'][pv,pA,hc,hA], optimize=True)
    O['ab'][c,A,c,C] += scale * -1.00000000 * np.einsum('iIaJ,aUjI->iUjJ', h['ab'][c,C,v,C], t['ab'][pv,pA,hc,hC], optimize=True)
    O['ab'][c,A,c,C] += scale * +1.00000000 * np.einsum('iUaA,aAjI->iUjI', h['ab'][c,A,v,V], t['ab'][pv,pV,hc,hC], optimize=True)
    O['ab'][v,A,c,C] += scale * +1.00000000 * np.einsum('uv,UV,uWiV,aUvI->aWiI', eta1['a'], gamma1['b'], h['ab'][a,A,c,A], t['ab'][pv,pA,ha,hC], optimize=True)
    O['ab'][v,A,c,C] += scale * -1.00000000 * np.einsum('uv,UV,aUvI,uWiV->aWiI', eta1['a'], gamma1['b'], h['ab'][v,A,a,C], t['ab'][pa,pA,hc,hA], optimize=True)
    O['ab'][v,A,c,C] += scale * -1.00000000 * np.einsum('uv,aIvJ,uUiI->aUiJ', eta1['a'], h['ab'][v,C,a,C], t['ab'][pa,pA,hc,hC], optimize=True)
    O['ab'][v,A,c,C] += scale * +1.00000000 * np.einsum('uv,aUvA,uAiI->aUiI', eta1['a'], h['ab'][v,A,a,V], t['ab'][pa,pV,hc,hC], optimize=True)
    O['ab'][v,A,c,C] += scale * -1.00000000 * np.einsum('UV,uv,uUiI,aWvV->aWiI', eta1['b'], eta1['a'], h['ab'][a,A,c,C], t['ab'][pv,pA,ha,hA], optimize=True)
    O['ab'][v,A,c,C] += scale * +1.00000000 * np.einsum('UV,uv,aWvV,uUiI->aWiI', eta1['b'], eta1['a'], h['ab'][v,A,a,A], t['ab'][pa,pA,hc,hC], optimize=True)
    O['ab'][v,A,c,C] += scale * -1.00000000 * np.einsum('UV,uv,uWiV,aUvI->aWiI', eta1['b'], gamma1['a'], h['ab'][a,A,c,A], t['ab'][pv,pA,ha,hC], optimize=True)
    O['ab'][v,A,c,C] += scale * +1.00000000 * np.einsum('UV,uv,aUvI,uWiV->aWiI', eta1['b'], gamma1['a'], h['ab'][v,A,a,C], t['ab'][pa,pA,hc,hA], optimize=True)
    O['ab'][v,A,c,C] += scale * -1.00000000 * np.einsum('UV,iWjV,aUiI->aWjI', eta1['b'], h['ab'][c,A,c,A], t['ab'][pv,pA,hc,hC], optimize=True)
    O['ab'][v,A,c,C] += scale * +1.00000000 * np.einsum('UV,aWbV,bUiI->aWiI', eta1['b'], h['ab'][v,A,v,A], t['ab'][pv,pA,hc,hC], optimize=True)
    O['ab'][v,A,c,C] += scale * +1.00000000 * np.einsum('uv,uIiJ,aUvI->aUiJ', gamma1['a'], h['ab'][a,C,c,C], t['ab'][pv,pA,ha,hC], optimize=True)
    O['ab'][v,A,c,C] += scale * -1.00000000 * np.einsum('uv,uUiA,aAvI->aUiI', gamma1['a'], h['ab'][a,A,c,V], t['ab'][pv,pV,ha,hC], optimize=True)
    O['ab'][v,A,c,C] += scale * +1.00000000 * np.einsum('UV,uv,uUiI,aWvV->aWiI', gamma1['b'], gamma1['a'], h['ab'][a,A,c,C], t['ab'][pv,pA,ha,hA], optimize=True)
    O['ab'][v,A,c,C] += scale * -1.00000000 * np.einsum('UV,uv,aWvV,uUiI->aWiI', gamma1['b'], gamma1['a'], h['ab'][v,A,a,A], t['ab'][pa,pA,hc,hC], optimize=True)
    O['ab'][v,A,c,C] += scale * +1.00000000 * np.einsum('UV,iUjI,aWiV->aWjI', gamma1['b'], h['ab'][c,A,c,C], t['ab'][pv,pA,hc,hA], optimize=True)
    O['ab'][v,A,c,C] += scale * -1.00000000 * np.einsum('UV,aUbI,bWiV->aWiI', gamma1['b'], h['ab'][v,A,v,C], t['ab'][pv,pA,hc,hA], optimize=True)
    O['ab'][v,A,c,C] += scale * +1.00000000 * np.einsum('iIjJ,aUiI->aUjJ', h['ab'][c,C,c,C], t['ab'][pv,pA,hc,hC], optimize=True)
    O['ab'][v,A,c,C] += scale * -1.00000000 * np.einsum('iUjA,aAiI->aUjI', h['ab'][c,A,c,V], t['ab'][pv,pV,hc,hC], optimize=True)
    O['ab'][v,A,c,C] += scale * -1.00000000 * np.einsum('aIbJ,bUiI->aUiJ', h['ab'][v,C,v,C], t['ab'][pv,pA,hc,hC], optimize=True)
    O['ab'][v,A,c,C] += scale * +1.00000000 * np.einsum('aUbA,bAiI->aUiI', h['ab'][v,A,v,V], t['ab'][pv,pV,hc,hC], optimize=True)
    O['ab'][a,C,c,C] += scale * +1.00000000 * np.einsum('uv,UV,uIiV,wUvJ->wIiJ', eta1['a'], gamma1['b'], h['ab'][a,C,c,A], t['ab'][pa,pA,ha,hC], optimize=True)
    O['ab'][a,C,c,C] += scale * +1.00000000 * np.einsum('uv,wIvA,uAiJ->wIiJ', eta1['a'], h['ab'][a,C,a,V], t['ab'][pa,pV,hc,hC], optimize=True)
    O['ab'][a,C,c,C] += scale * +1.00000000 * np.einsum('UV,uv,wIvV,uUiJ->wIiJ', eta1['b'], eta1['a'], h['ab'][a,C,a,A], t['ab'][pa,pA,hc,hC], optimize=True)
    O['ab'][a,C,c,C] += scale * -1.00000000 * np.einsum('UV,uv,uIiV,wUvJ->wIiJ', eta1['b'], gamma1['a'], h['ab'][a,C,c,A], t['ab'][pa,pA,ha,hC], optimize=True)
    O['ab'][a,C,c,C] += scale * -1.00000000 * np.einsum('UV,iIjV,uUiJ->uIjJ', eta1['b'], h['ab'][c,C,c,A], t['ab'][pa,pA,hc,hC], optimize=True)
    O['ab'][a,C,c,C] += scale * +1.00000000 * np.einsum('UV,uIaV,aUiJ->uIiJ', eta1['b'], h['ab'][a,C,v,A], t['ab'][pv,pA,hc,hC], optimize=True)
    O['ab'][a,C,c,C] += scale * -1.00000000 * np.einsum('uv,uIiA,wAvJ->wIiJ', gamma1['a'], h['ab'][a,C,c,V], t['ab'][pa,pV,ha,hC], optimize=True)
    O['ab'][a,C,c,C] += scale * -1.00000000 * np.einsum('UV,uv,wIvV,uUiJ->wIiJ', gamma1['b'], gamma1['a'], h['ab'][a,C,a,A], t['ab'][pa,pA,hc,hC], optimize=True)
    O['ab'][a,C,c,C] += scale * -1.00000000 * np.einsum('iIjA,uAiJ->uIjJ', h['ab'][c,C,c,V], t['ab'][pa,pV,hc,hC], optimize=True)
    O['ab'][a,C,c,C] += scale * +1.00000000 * np.einsum('uIaA,aAiJ->uIiJ', h['ab'][a,C,v,V], t['ab'][pv,pV,hc,hC], optimize=True)
    O['ab'][c,C,c,C] += scale * +1.00000000 * np.einsum('uv,iIvA,uAjJ->iIjJ', eta1['a'], h['ab'][c,C,a,V], t['ab'][pa,pV,hc,hC], optimize=True)
    O['ab'][c,C,c,C] += scale * +1.00000000 * np.einsum('UV,uv,iIvV,uUjJ->iIjJ', eta1['b'], eta1['a'], h['ab'][c,C,a,A], t['ab'][pa,pA,hc,hC], optimize=True)
    O['ab'][c,C,c,C] += scale * +1.00000000 * np.einsum('UV,iIaV,aUjJ->iIjJ', eta1['b'], h['ab'][c,C,v,A], t['ab'][pv,pA,hc,hC], optimize=True)
    O['ab'][c,C,c,C] += scale * -1.00000000 * np.einsum('UV,uv,iIvV,uUjJ->iIjJ', gamma1['b'], gamma1['a'], h['ab'][c,C,a,A], t['ab'][pa,pA,hc,hC], optimize=True)
    O['ab'][c,C,c,C] += scale * +1.00000000 * np.einsum('iIaA,aAjJ->iIjJ', h['ab'][c,C,v,V], t['ab'][pv,pV,hc,hC], optimize=True)
    O['ab'][v,C,c,C] += scale * +1.00000000 * np.einsum('uv,UV,uIiV,aUvJ->aIiJ', eta1['a'], gamma1['b'], h['ab'][a,C,c,A], t['ab'][pv,pA,ha,hC], optimize=True)
    O['ab'][v,C,c,C] += scale * +1.00000000 * np.einsum('uv,aIvA,uAiJ->aIiJ', eta1['a'], h['ab'][v,C,a,V], t['ab'][pa,pV,hc,hC], optimize=True)
    O['ab'][v,C,c,C] += scale * +1.00000000 * np.einsum('UV,uv,aIvV,uUiJ->aIiJ', eta1['b'], eta1['a'], h['ab'][v,C,a,A], t['ab'][pa,pA,hc,hC], optimize=True)
    O['ab'][v,C,c,C] += scale * -1.00000000 * np.einsum('UV,uv,uIiV,aUvJ->aIiJ', eta1['b'], gamma1['a'], h['ab'][a,C,c,A], t['ab'][pv,pA,ha,hC], optimize=True)
    O['ab'][v,C,c,C] += scale * -1.00000000 * np.einsum('UV,iIjV,aUiJ->aIjJ', eta1['b'], h['ab'][c,C,c,A], t['ab'][pv,pA,hc,hC], optimize=True)
    O['ab'][v,C,c,C] += scale * +1.00000000 * np.einsum('UV,aIbV,bUiJ->aIiJ', eta1['b'], h['ab'][v,C,v,A], t['ab'][pv,pA,hc,hC], optimize=True)
    O['ab'][v,C,c,C] += scale * -1.00000000 * np.einsum('uv,uIiA,aAvJ->aIiJ', gamma1['a'], h['ab'][a,C,c,V], t['ab'][pv,pV,ha,hC], optimize=True)
    O['ab'][v,C,c,C] += scale * -1.00000000 * np.einsum('UV,uv,aIvV,uUiJ->aIiJ', gamma1['b'], gamma1['a'], h['ab'][v,C,a,A], t['ab'][pa,pA,hc,hC], optimize=True)
    O['ab'][v,C,c,C] += scale * -1.00000000 * np.einsum('iIjA,aAiJ->aIjJ', h['ab'][c,C,c,V], t['ab'][pv,pV,hc,hC], optimize=True)
    O['ab'][v,C,c,C] += scale * +1.00000000 * np.einsum('aIbA,bAiJ->aIiJ', h['ab'][v,C,v,V], t['ab'][pv,pV,hc,hC], optimize=True)
    O['ab'][a,V,c,C] += scale * -1.00000000 * np.einsum('uv,UV,wUvI,uAiV->wAiI', eta1['a'], gamma1['b'], h['ab'][a,A,a,C], t['ab'][pa,pV,hc,hA], optimize=True)
    O['ab'][a,V,c,C] += scale * +1.00000000 * np.einsum('uv,UV,uAiV,wUvI->wAiI', eta1['a'], gamma1['b'], h['ab'][a,V,c,A], t['ab'][pa,pA,ha,hC], optimize=True)
    O['ab'][a,V,c,C] += scale * -1.00000000 * np.einsum('uv,wIvJ,uAiI->wAiJ', eta1['a'], h['ab'][a,C,a,C], t['ab'][pa,pV,hc,hC], optimize=True)
    O['ab'][a,V,c,C] += scale * +1.00000000 * np.einsum('uv,wAvB,uBiI->wAiI', eta1['a'], h['ab'][a,V,a,V], t['ab'][pa,pV,hc,hC], optimize=True)
    O['ab'][a,V,c,C] += scale * +1.00000000 * np.einsum('UV,uv,wAvV,uUiI->wAiI', eta1['b'], eta1['a'], h['ab'][a,V,a,A], t['ab'][pa,pA,hc,hC], optimize=True)
    O['ab'][a,V,c,C] += scale * -1.00000000 * np.einsum('UV,uv,uUiI,wAvV->wAiI', eta1['b'], eta1['a'], h['ab'][a,A,c,C], t['ab'][pa,pV,ha,hA], optimize=True)
    O['ab'][a,V,c,C] += scale * +1.00000000 * np.einsum('UV,uv,wUvI,uAiV->wAiI', eta1['b'], gamma1['a'], h['ab'][a,A,a,C], t['ab'][pa,pV,hc,hA], optimize=True)
    O['ab'][a,V,c,C] += scale * -1.00000000 * np.einsum('UV,uv,uAiV,wUvI->wAiI', eta1['b'], gamma1['a'], h['ab'][a,V,c,A], t['ab'][pa,pA,ha,hC], optimize=True)
    O['ab'][a,V,c,C] += scale * -1.00000000 * np.einsum('UV,iAjV,uUiI->uAjI', eta1['b'], h['ab'][c,V,c,A], t['ab'][pa,pA,hc,hC], optimize=True)
    O['ab'][a,V,c,C] += scale * +1.00000000 * np.einsum('UV,uAaV,aUiI->uAiI', eta1['b'], h['ab'][a,V,v,A], t['ab'][pv,pA,hc,hC], optimize=True)
    O['ab'][a,V,c,C] += scale * +1.00000000 * np.einsum('uv,uIiJ,wAvI->wAiJ', gamma1['a'], h['ab'][a,C,c,C], t['ab'][pa,pV,ha,hC], optimize=True)
    O['ab'][a,V,c,C] += scale * -1.00000000 * np.einsum('uv,uAiB,wBvI->wAiI', gamma1['a'], h['ab'][a,V,c,V], t['ab'][pa,pV,ha,hC], optimize=True)
    O['ab'][a,V,c,C] += scale * -1.00000000 * np.einsum('UV,uv,wAvV,uUiI->wAiI', gamma1['b'], gamma1['a'], h['ab'][a,V,a,A], t['ab'][pa,pA,hc,hC], optimize=True)
    O['ab'][a,V,c,C] += scale * +1.00000000 * np.einsum('UV,uv,uUiI,wAvV->wAiI', gamma1['b'], gamma1['a'], h['ab'][a,A,c,C], t['ab'][pa,pV,ha,hA], optimize=True)
    O['ab'][a,V,c,C] += scale * +1.00000000 * np.einsum('UV,iUjI,uAiV->uAjI', gamma1['b'], h['ab'][c,A,c,C], t['ab'][pa,pV,hc,hA], optimize=True)
    O['ab'][a,V,c,C] += scale * -1.00000000 * np.einsum('UV,uUaI,aAiV->uAiI', gamma1['b'], h['ab'][a,A,v,C], t['ab'][pv,pV,hc,hA], optimize=True)
    O['ab'][a,V,c,C] += scale * +1.00000000 * np.einsum('iIjJ,uAiI->uAjJ', h['ab'][c,C,c,C], t['ab'][pa,pV,hc,hC], optimize=True)
    O['ab'][a,V,c,C] += scale * -1.00000000 * np.einsum('iAjB,uBiI->uAjI', h['ab'][c,V,c,V], t['ab'][pa,pV,hc,hC], optimize=True)
    O['ab'][a,V,c,C] += scale * -1.00000000 * np.einsum('uIaJ,aAiI->uAiJ', h['ab'][a,C,v,C], t['ab'][pv,pV,hc,hC], optimize=True)
    O['ab'][a,V,c,C] += scale * +1.00000000 * np.einsum('uAaB,aBiI->uAiI', h['ab'][a,V,v,V], t['ab'][pv,pV,hc,hC], optimize=True)
    O['ab'][c,V,c,C] += scale * -1.00000000 * np.einsum('uv,UV,iUvI,uAjV->iAjI', eta1['a'], gamma1['b'], h['ab'][c,A,a,C], t['ab'][pa,pV,hc,hA], optimize=True)
    O['ab'][c,V,c,C] += scale * -1.00000000 * np.einsum('uv,iIvJ,uAjI->iAjJ', eta1['a'], h['ab'][c,C,a,C], t['ab'][pa,pV,hc,hC], optimize=True)
    O['ab'][c,V,c,C] += scale * +1.00000000 * np.einsum('uv,iAvB,uBjI->iAjI', eta1['a'], h['ab'][c,V,a,V], t['ab'][pa,pV,hc,hC], optimize=True)
    O['ab'][c,V,c,C] += scale * +1.00000000 * np.einsum('UV,uv,iAvV,uUjI->iAjI', eta1['b'], eta1['a'], h['ab'][c,V,a,A], t['ab'][pa,pA,hc,hC], optimize=True)
    O['ab'][c,V,c,C] += scale * +1.00000000 * np.einsum('UV,uv,iUvI,uAjV->iAjI', eta1['b'], gamma1['a'], h['ab'][c,A,a,C], t['ab'][pa,pV,hc,hA], optimize=True)
    O['ab'][c,V,c,C] += scale * +1.00000000 * np.einsum('UV,iAaV,aUjI->iAjI', eta1['b'], h['ab'][c,V,v,A], t['ab'][pv,pA,hc,hC], optimize=True)
    O['ab'][c,V,c,C] += scale * -1.00000000 * np.einsum('UV,uv,iAvV,uUjI->iAjI', gamma1['b'], gamma1['a'], h['ab'][c,V,a,A], t['ab'][pa,pA,hc,hC], optimize=True)
    O['ab'][c,V,c,C] += scale * -1.00000000 * np.einsum('UV,iUaI,aAjV->iAjI', gamma1['b'], h['ab'][c,A,v,C], t['ab'][pv,pV,hc,hA], optimize=True)
    O['ab'][c,V,c,C] += scale * -1.00000000 * np.einsum('iIaJ,aAjI->iAjJ', h['ab'][c,C,v,C], t['ab'][pv,pV,hc,hC], optimize=True)
    O['ab'][c,V,c,C] += scale * +1.00000000 * np.einsum('iAaB,aBjI->iAjI', h['ab'][c,V,v,V], t['ab'][pv,pV,hc,hC], optimize=True)
    O['ab'][v,V,c,C] += scale * +1.00000000 * np.einsum('uv,UV,uAiV,aUvI->aAiI', eta1['a'], gamma1['b'], h['ab'][a,V,c,A], t['ab'][pv,pA,ha,hC], optimize=True)
    O['ab'][v,V,c,C] += scale * -1.00000000 * np.einsum('uv,UV,aUvI,uAiV->aAiI', eta1['a'], gamma1['b'], h['ab'][v,A,a,C], t['ab'][pa,pV,hc,hA], optimize=True)
    O['ab'][v,V,c,C] += scale * -1.00000000 * np.einsum('uv,aIvJ,uAiI->aAiJ', eta1['a'], h['ab'][v,C,a,C], t['ab'][pa,pV,hc,hC], optimize=True)
    O['ab'][v,V,c,C] += scale * +1.00000000 * np.einsum('uv,aAvB,uBiI->aAiI', eta1['a'], h['ab'][v,V,a,V], t['ab'][pa,pV,hc,hC], optimize=True)
    O['ab'][v,V,c,C] += scale * -1.00000000 * np.einsum('UV,uv,uUiI,aAvV->aAiI', eta1['b'], eta1['a'], h['ab'][a,A,c,C], t['ab'][pv,pV,ha,hA], optimize=True)
    O['ab'][v,V,c,C] += scale * +1.00000000 * np.einsum('UV,uv,aAvV,uUiI->aAiI', eta1['b'], eta1['a'], h['ab'][v,V,a,A], t['ab'][pa,pA,hc,hC], optimize=True)
    O['ab'][v,V,c,C] += scale * -1.00000000 * np.einsum('UV,uv,uAiV,aUvI->aAiI', eta1['b'], gamma1['a'], h['ab'][a,V,c,A], t['ab'][pv,pA,ha,hC], optimize=True)
    O['ab'][v,V,c,C] += scale * +1.00000000 * np.einsum('UV,uv,aUvI,uAiV->aAiI', eta1['b'], gamma1['a'], h['ab'][v,A,a,C], t['ab'][pa,pV,hc,hA], optimize=True)
    O['ab'][v,V,c,C] += scale * -1.00000000 * np.einsum('UV,iAjV,aUiI->aAjI', eta1['b'], h['ab'][c,V,c,A], t['ab'][pv,pA,hc,hC], optimize=True)
    O['ab'][v,V,c,C] += scale * +1.00000000 * np.einsum('UV,aAbV,bUiI->aAiI', eta1['b'], h['ab'][v,V,v,A], t['ab'][pv,pA,hc,hC], optimize=True)
    O['ab'][v,V,c,C] += scale * +1.00000000 * np.einsum('uv,uIiJ,aAvI->aAiJ', gamma1['a'], h['ab'][a,C,c,C], t['ab'][pv,pV,ha,hC], optimize=True)
    O['ab'][v,V,c,C] += scale * -1.00000000 * np.einsum('uv,uAiB,aBvI->aAiI', gamma1['a'], h['ab'][a,V,c,V], t['ab'][pv,pV,ha,hC], optimize=True)
    O['ab'][v,V,c,C] += scale * +1.00000000 * np.einsum('UV,uv,uUiI,aAvV->aAiI', gamma1['b'], gamma1['a'], h['ab'][a,A,c,C], t['ab'][pv,pV,ha,hA], optimize=True)
    O['ab'][v,V,c,C] += scale * -1.00000000 * np.einsum('UV,uv,aAvV,uUiI->aAiI', gamma1['b'], gamma1['a'], h['ab'][v,V,a,A], t['ab'][pa,pA,hc,hC], optimize=True)
    O['ab'][v,V,c,C] += scale * +1.00000000 * np.einsum('UV,iUjI,aAiV->aAjI', gamma1['b'], h['ab'][c,A,c,C], t['ab'][pv,pV,hc,hA], optimize=True)
    O['ab'][v,V,c,C] += scale * -1.00000000 * np.einsum('UV,aUbI,bAiV->aAiI', gamma1['b'], h['ab'][v,A,v,C], t['ab'][pv,pV,hc,hA], optimize=True)
    O['ab'][v,V,c,C] += scale * +1.00000000 * np.einsum('iIjJ,aAiI->aAjJ', h['ab'][c,C,c,C], t['ab'][pv,pV,hc,hC], optimize=True)
    O['ab'][v,V,c,C] += scale * -1.00000000 * np.einsum('iAjB,aBiI->aAjI', h['ab'][c,V,c,V], t['ab'][pv,pV,hc,hC], optimize=True)
    O['ab'][v,V,c,C] += scale * -1.00000000 * np.einsum('aIbJ,bAiI->aAiJ', h['ab'][v,C,v,C], t['ab'][pv,pV,hc,hC], optimize=True)
    O['ab'][v,V,c,C] += scale * +1.00000000 * np.einsum('aAbB,bBiI->aAiI', h['ab'][v,V,v,V], t['ab'][pv,pV,hc,hC], optimize=True)
    O['ab'][a,A,c,V] += scale * -1.00000000 * np.einsum('uv,UV,wUvA,uWiV->wWiA', eta1['a'], gamma1['b'], h['ab'][a,A,a,V], t['ab'][pa,pA,hc,hA], optimize=True)
    O['ab'][a,A,c,V] += scale * -1.00000000 * np.einsum('uv,wIvA,uUiI->wUiA', eta1['a'], h['ab'][a,C,a,V], t['ab'][pa,pA,hc,hC], optimize=True)
    
    O['ab'][a,A,c,V] += scale * +1.00000000 * np.einsum('UV,uv,wUvA,uWiV->wWiA', eta1['b'], gamma1['a'], h['ab'][a,A,a,V], t['ab'][pa,pA,hc,hA], optimize=True)
    O['ab'][a,A,c,V] += scale * +1.00000000 * np.einsum('uv,uIiA,wUvI->wUiA', gamma1['a'], h['ab'][a,C,c,V], t['ab'][pa,pA,ha,hC], optimize=True)
    
    O['ab'][a,A,c,V] += scale * +1.00000000 * np.einsum('UV,iUjA,uWiV->uWjA', gamma1['b'], h['ab'][c,A,c,V], t['ab'][pa,pA,hc,hA], optimize=True)
    O['ab'][a,A,c,V] += scale * -1.00000000 * np.einsum('UV,uUaA,aWiV->uWiA', gamma1['b'], h['ab'][a,A,v,V], t['ab'][pv,pA,hc,hA], optimize=True)
    O['ab'][a,A,c,V] += scale * +1.00000000 * np.einsum('iIjA,uUiI->uUjA', h['ab'][c,C,c,V], t['ab'][pa,pA,hc,hC], optimize=True)
    O['ab'][a,A,c,V] += scale * -1.00000000 * np.einsum('uIaA,aUiI->uUiA', h['ab'][a,C,v,V], t['ab'][pv,pA,hc,hC], optimize=True)
    O['ab'][c,A,c,V] += scale * -1.00000000 * np.einsum('uv,UV,iUvA,uWjV->iWjA', eta1['a'], gamma1['b'], h['ab'][c,A,a,V], t['ab'][pa,pA,hc,hA], optimize=True)
    O['ab'][c,A,c,V] += scale * -1.00000000 * np.einsum('uv,iIvA,uUjI->iUjA', eta1['a'], h['ab'][c,C,a,V], t['ab'][pa,pA,hc,hC], optimize=True)
    O['ab'][c,A,c,V] += scale * +1.00000000 * np.einsum('UV,uv,iUvA,uWjV->iWjA', eta1['b'], gamma1['a'], h['ab'][c,A,a,V], t['ab'][pa,pA,hc,hA], optimize=True)
    O['ab'][c,A,c,V] += scale * -1.00000000 * np.einsum('UV,iUaA,aWjV->iWjA', gamma1['b'], h['ab'][c,A,v,V], t['ab'][pv,pA,hc,hA], optimize=True)
    O['ab'][c,A,c,V] += scale * -1.00000000 * np.einsum('iIaA,aUjI->iUjA', h['ab'][c,C,v,V], t['ab'][pv,pA,hc,hC], optimize=True)
    O['ab'][v,A,c,V] += scale * -1.00000000 * np.einsum('uv,UV,aUvA,uWiV->aWiA', eta1['a'], gamma1['b'], h['ab'][v,A,a,V], t['ab'][pa,pA,hc,hA], optimize=True)
    O['ab'][v,A,c,V] += scale * -1.00000000 * np.einsum('uv,aIvA,uUiI->aUiA', eta1['a'], h['ab'][v,C,a,V], t['ab'][pa,pA,hc,hC], optimize=True)
    O['ab'][v,A,c,V] += scale * -1.00000000 * np.einsum('UV,uv,uUiA,aWvV->aWiA', eta1['b'], eta1['a'], h['ab'][a,A,c,V], t['ab'][pv,pA,ha,hA], optimize=True)
    O['ab'][v,A,c,V] += scale * +1.00000000 * np.einsum('UV,uv,aUvA,uWiV->aWiA', eta1['b'], gamma1['a'], h['ab'][v,A,a,V], t['ab'][pa,pA,hc,hA], optimize=True)
    O['ab'][v,A,c,V] += scale * +1.00000000 * np.einsum('uv,uIiA,aUvI->aUiA', gamma1['a'], h['ab'][a,C,c,V], t['ab'][pv,pA,ha,hC], optimize=True)
    O['ab'][v,A,c,V] += scale * +1.00000000 * np.einsum('UV,uv,uUiA,aWvV->aWiA', gamma1['b'], gamma1['a'], h['ab'][a,A,c,V], t['ab'][pv,pA,ha,hA], optimize=True)
    O['ab'][v,A,c,V] += scale * +1.00000000 * np.einsum('UV,iUjA,aWiV->aWjA', gamma1['b'], h['ab'][c,A,c,V], t['ab'][pv,pA,hc,hA], optimize=True)
    O['ab'][v,A,c,V] += scale * -1.00000000 * np.einsum('UV,aUbA,bWiV->aWiA', gamma1['b'], h['ab'][v,A,v,V], t['ab'][pv,pA,hc,hA], optimize=True)
    O['ab'][v,A,c,V] += scale * +1.00000000 * np.einsum('iIjA,aUiI->aUjA', h['ab'][c,C,c,V], t['ab'][pv,pA,hc,hC], optimize=True)
    O['ab'][v,A,c,V] += scale * -1.00000000 * np.einsum('aIbA,bUiI->aUiA', h['ab'][v,C,v,V], t['ab'][pv,pA,hc,hC], optimize=True)
    O['ab'][a,V,c,V] += scale * -1.00000000 * np.einsum('uv,UV,wUvA,uBiV->wBiA', eta1['a'], gamma1['b'], h['ab'][a,A,a,V], t['ab'][pa,pV,hc,hA], optimize=True)
    O['ab'][a,V,c,V] += scale * -1.00000000 * np.einsum('uv,wIvA,uBiI->wBiA', eta1['a'], h['ab'][a,C,a,V], t['ab'][pa,pV,hc,hC], optimize=True)
    O['ab'][a,V,c,V] += scale * -1.00000000 * np.einsum('UV,uv,uUiA,wBvV->wBiA', eta1['b'], eta1['a'], h['ab'][a,A,c,V], t['ab'][pa,pV,ha,hA], optimize=True)
    O['ab'][a,V,c,V] += scale * +1.00000000 * np.einsum('UV,uv,wUvA,uBiV->wBiA', eta1['b'], gamma1['a'], h['ab'][a,A,a,V], t['ab'][pa,pV,hc,hA], optimize=True)
    O['ab'][a,V,c,V] += scale * +1.00000000 * np.einsum('uv,uIiA,wBvI->wBiA', gamma1['a'], h['ab'][a,C,c,V], t['ab'][pa,pV,ha,hC], optimize=True)
    O['ab'][a,V,c,V] += scale * +1.00000000 * np.einsum('UV,uv,uUiA,wBvV->wBiA', gamma1['b'], gamma1['a'], h['ab'][a,A,c,V], t['ab'][pa,pV,ha,hA], optimize=True)
    O['ab'][a,V,c,V] += scale * +1.00000000 * np.einsum('UV,iUjA,uBiV->uBjA', gamma1['b'], h['ab'][c,A,c,V], t['ab'][pa,pV,hc,hA], optimize=True)
    O['ab'][a,V,c,V] += scale * -1.00000000 * np.einsum('UV,uUaA,aBiV->uBiA', gamma1['b'], h['ab'][a,A,v,V], t['ab'][pv,pV,hc,hA], optimize=True)
    O['ab'][a,V,c,V] += scale * +1.00000000 * np.einsum('iIjA,uBiI->uBjA', h['ab'][c,C,c,V], t['ab'][pa,pV,hc,hC], optimize=True)
    O['ab'][a,V,c,V] += scale * -1.00000000 * np.einsum('uIaA,aBiI->uBiA', h['ab'][a,C,v,V], t['ab'][pv,pV,hc,hC], optimize=True)
    O['ab'][c,V,c,V] += scale * -1.00000000 * np.einsum('uv,UV,iUvA,uBjV->iBjA', eta1['a'], gamma1['b'], h['ab'][c,A,a,V], t['ab'][pa,pV,hc,hA], optimize=True)
    O['ab'][c,V,c,V] += scale * -1.00000000 * np.einsum('uv,iIvA,uBjI->iBjA', eta1['a'], h['ab'][c,C,a,V], t['ab'][pa,pV,hc,hC], optimize=True)
    O['ab'][c,V,c,V] += scale * +1.00000000 * np.einsum('UV,uv,iUvA,uBjV->iBjA', eta1['b'], gamma1['a'], h['ab'][c,A,a,V], t['ab'][pa,pV,hc,hA], optimize=True)
    O['ab'][c,V,c,V] += scale * -1.00000000 * np.einsum('UV,iUaA,aBjV->iBjA', gamma1['b'], h['ab'][c,A,v,V], t['ab'][pv,pV,hc,hA], optimize=True)
    O['ab'][c,V,c,V] += scale * -1.00000000 * np.einsum('iIaA,aBjI->iBjA', h['ab'][c,C,v,V], t['ab'][pv,pV,hc,hC], optimize=True)
    O['ab'][v,V,c,V] += scale * -1.00000000 * np.einsum('uv,UV,aUvA,uBiV->aBiA', eta1['a'], gamma1['b'], h['ab'][v,A,a,V], t['ab'][pa,pV,hc,hA], optimize=True)
    O['ab'][v,V,c,V] += scale * -1.00000000 * np.einsum('uv,aIvA,uBiI->aBiA', eta1['a'], h['ab'][v,C,a,V], t['ab'][pa,pV,hc,hC], optimize=True)
    O['ab'][v,V,c,V] += scale * -1.00000000 * np.einsum('UV,uv,uUiA,aBvV->aBiA', eta1['b'], eta1['a'], h['ab'][a,A,c,V], t['ab'][pv,pV,ha,hA], optimize=True)
    O['ab'][v,V,c,V] += scale * +1.00000000 * np.einsum('UV,uv,aUvA,uBiV->aBiA', eta1['b'], gamma1['a'], h['ab'][v,A,a,V], t['ab'][pa,pV,hc,hA], optimize=True)
    O['ab'][v,V,c,V] += scale * +1.00000000 * np.einsum('uv,uIiA,aBvI->aBiA', gamma1['a'], h['ab'][a,C,c,V], t['ab'][pv,pV,ha,hC], optimize=True)
    O['ab'][v,V,c,V] += scale * +1.00000000 * np.einsum('UV,uv,uUiA,aBvV->aBiA', gamma1['b'], gamma1['a'], h['ab'][a,A,c,V], t['ab'][pv,pV,ha,hA], optimize=True)
    O['ab'][v,V,c,V] += scale * +1.00000000 * np.einsum('UV,iUjA,aBiV->aBjA', gamma1['b'], h['ab'][c,A,c,V], t['ab'][pv,pV,hc,hA], optimize=True)
    O['ab'][v,V,c,V] += scale * -1.00000000 * np.einsum('UV,aUbA,bBiV->aBiA', gamma1['b'], h['ab'][v,A,v,V], t['ab'][pv,pV,hc,hA], optimize=True)
    O['ab'][v,V,c,V] += scale * +1.00000000 * np.einsum('iIjA,aBiI->aBjA', h['ab'][c,C,c,V], t['ab'][pv,pV,hc,hC], optimize=True)
    O['ab'][v,V,c,V] += scale * -1.00000000 * np.einsum('aIbA,bBiI->aBiA', h['ab'][v,C,v,V], t['ab'][pv,pV,hc,hC], optimize=True)
    
    
    
    O['ab'][a,A,v,A] += scale * -1.00000000 * np.einsum('UV,iWaV,uUiX->uWaX', eta1['b'], h['ab'][c,A,v,A], t['ab'][pa,pA,hc,hA], optimize=True)
    O['ab'][a,A,v,A] += scale * +1.00000000 * np.einsum('uv,uIaU,wVvI->wVaU', gamma1['a'], h['ab'][a,C,v,A], t['ab'][pa,pA,ha,hC], optimize=True)
    O['ab'][a,A,v,A] += scale * -1.00000000 * np.einsum('uv,uUaA,wAvV->wUaV', gamma1['a'], h['ab'][a,A,v,V], t['ab'][pa,pV,ha,hA], optimize=True)
    
    O['ab'][a,A,v,A] += scale * +1.00000000 * np.einsum('UV,iUaW,uXiV->uXaW', gamma1['b'], h['ab'][c,A,v,A], t['ab'][pa,pA,hc,hA], optimize=True)
    O['ab'][a,A,v,A] += scale * +1.00000000 * np.einsum('iIaU,uViI->uVaU', h['ab'][c,C,v,A], t['ab'][pa,pA,hc,hC], optimize=True)
    O['ab'][a,A,v,A] += scale * -1.00000000 * np.einsum('iUaA,uAiV->uUaV', h['ab'][c,A,v,V], t['ab'][pa,pV,hc,hA], optimize=True)
    O['ab'][v,A,v,A] += scale * +1.00000000 * np.einsum('uv,UV,uWaV,bUvX->bWaX', eta1['a'], gamma1['b'], h['ab'][a,A,v,A], t['ab'][pv,pA,ha,hA], optimize=True)
    O['ab'][v,A,v,A] += scale * -1.00000000 * np.einsum('UV,uv,uUaW,bXvV->bXaW', eta1['b'], eta1['a'], h['ab'][a,A,v,A], t['ab'][pv,pA,ha,hA], optimize=True)
    O['ab'][v,A,v,A] += scale * -1.00000000 * np.einsum('UV,uv,uWaV,bUvX->bWaX', eta1['b'], gamma1['a'], h['ab'][a,A,v,A], t['ab'][pv,pA,ha,hA], optimize=True)
    O['ab'][v,A,v,A] += scale * -1.00000000 * np.einsum('UV,iWaV,bUiX->bWaX', eta1['b'], h['ab'][c,A,v,A], t['ab'][pv,pA,hc,hA], optimize=True)
    O['ab'][v,A,v,A] += scale * +1.00000000 * np.einsum('uv,uIaU,bVvI->bVaU', gamma1['a'], h['ab'][a,C,v,A], t['ab'][pv,pA,ha,hC], optimize=True)
    O['ab'][v,A,v,A] += scale * -1.00000000 * np.einsum('uv,uUaA,bAvV->bUaV', gamma1['a'], h['ab'][a,A,v,V], t['ab'][pv,pV,ha,hA], optimize=True)
    O['ab'][v,A,v,A] += scale * +1.00000000 * np.einsum('UV,uv,uUaW,bXvV->bXaW', gamma1['b'], gamma1['a'], h['ab'][a,A,v,A], t['ab'][pv,pA,ha,hA], optimize=True)
    O['ab'][v,A,v,A] += scale * +1.00000000 * np.einsum('UV,iUaW,bXiV->bXaW', gamma1['b'], h['ab'][c,A,v,A], t['ab'][pv,pA,hc,hA], optimize=True)
    O['ab'][v,A,v,A] += scale * +1.00000000 * np.einsum('iIaU,bViI->bVaU', h['ab'][c,C,v,A], t['ab'][pv,pA,hc,hC], optimize=True)
    O['ab'][v,A,v,A] += scale * -1.00000000 * np.einsum('iUaA,bAiV->bUaV', h['ab'][c,A,v,V], t['ab'][pv,pV,hc,hA], optimize=True)
    
    
    O['ab'][a,C,v,A] += scale * -1.00000000 * np.einsum('UV,iIaV,uUiW->uIaW', eta1['b'], h['ab'][c,C,v,A], t['ab'][pa,pA,hc,hA], optimize=True)
    O['ab'][a,C,v,A] += scale * -1.00000000 * np.einsum('uv,uIaA,wAvU->wIaU', gamma1['a'], h['ab'][a,C,v,V], t['ab'][pa,pV,ha,hA], optimize=True)
    O['ab'][a,C,v,A] += scale * -1.00000000 * np.einsum('iIaA,uAiU->uIaU', h['ab'][c,C,v,V], t['ab'][pa,pV,hc,hA], optimize=True)
    O['ab'][v,C,v,A] += scale * +1.00000000 * np.einsum('uv,UV,uIaV,bUvW->bIaW', eta1['a'], gamma1['b'], h['ab'][a,C,v,A], t['ab'][pv,pA,ha,hA], optimize=True)
    O['ab'][v,C,v,A] += scale * -1.00000000 * np.einsum('UV,uv,uIaV,bUvW->bIaW', eta1['b'], gamma1['a'], h['ab'][a,C,v,A], t['ab'][pv,pA,ha,hA], optimize=True)
    O['ab'][v,C,v,A] += scale * -1.00000000 * np.einsum('UV,iIaV,bUiW->bIaW', eta1['b'], h['ab'][c,C,v,A], t['ab'][pv,pA,hc,hA], optimize=True)
    O['ab'][v,C,v,A] += scale * -1.00000000 * np.einsum('uv,uIaA,bAvU->bIaU', gamma1['a'], h['ab'][a,C,v,V], t['ab'][pv,pV,ha,hA], optimize=True)
    O['ab'][v,C,v,A] += scale * -1.00000000 * np.einsum('iIaA,bAiU->bIaU', h['ab'][c,C,v,V], t['ab'][pv,pV,hc,hA], optimize=True)
    
    O['ab'][a,V,v,A] += scale * -1.00000000 * np.einsum('UV,uv,uUaW,wAvV->wAaW', eta1['b'], eta1['a'], h['ab'][a,A,v,A], t['ab'][pa,pV,ha,hA], optimize=True)
    
    O['ab'][a,V,v,A] += scale * -1.00000000 * np.einsum('UV,iAaV,uUiW->uAaW', eta1['b'], h['ab'][c,V,v,A], t['ab'][pa,pA,hc,hA], optimize=True)
    O['ab'][a,V,v,A] += scale * +1.00000000 * np.einsum('uv,uIaU,wAvI->wAaU', gamma1['a'], h['ab'][a,C,v,A], t['ab'][pa,pV,ha,hC], optimize=True)
    O['ab'][a,V,v,A] += scale * -1.00000000 * np.einsum('uv,uAaB,wBvU->wAaU', gamma1['a'], h['ab'][a,V,v,V], t['ab'][pa,pV,ha,hA], optimize=True)
    O['ab'][a,V,v,A] += scale * +1.00000000 * np.einsum('UV,uv,uUaW,wAvV->wAaW', gamma1['b'], gamma1['a'], h['ab'][a,A,v,A], t['ab'][pa,pV,ha,hA], optimize=True)
    O['ab'][a,V,v,A] += scale * +1.00000000 * np.einsum('UV,iUaW,uAiV->uAaW', gamma1['b'], h['ab'][c,A,v,A], t['ab'][pa,pV,hc,hA], optimize=True)
    O['ab'][a,V,v,A] += scale * +1.00000000 * np.einsum('iIaU,uAiI->uAaU', h['ab'][c,C,v,A], t['ab'][pa,pV,hc,hC], optimize=True)
    O['ab'][a,V,v,A] += scale * -1.00000000 * np.einsum('iAaB,uBiU->uAaU', h['ab'][c,V,v,V], t['ab'][pa,pV,hc,hA], optimize=True)
    O['ab'][v,V,v,A] += scale * +1.00000000 * np.einsum('uv,UV,uAaV,bUvW->bAaW', eta1['a'], gamma1['b'], h['ab'][a,V,v,A], t['ab'][pv,pA,ha,hA], optimize=True)
    O['ab'][v,V,v,A] += scale * -1.00000000 * np.einsum('UV,uv,uUaW,bAvV->bAaW', eta1['b'], eta1['a'], h['ab'][a,A,v,A], t['ab'][pv,pV,ha,hA], optimize=True)
    O['ab'][v,V,v,A] += scale * -1.00000000 * np.einsum('UV,uv,uAaV,bUvW->bAaW', eta1['b'], gamma1['a'], h['ab'][a,V,v,A], t['ab'][pv,pA,ha,hA], optimize=True)
    O['ab'][v,V,v,A] += scale * -1.00000000 * np.einsum('UV,iAaV,bUiW->bAaW', eta1['b'], h['ab'][c,V,v,A], t['ab'][pv,pA,hc,hA], optimize=True)
    O['ab'][v,V,v,A] += scale * +1.00000000 * np.einsum('uv,uIaU,bAvI->bAaU', gamma1['a'], h['ab'][a,C,v,A], t['ab'][pv,pV,ha,hC], optimize=True)
    O['ab'][v,V,v,A] += scale * -1.00000000 * np.einsum('uv,uAaB,bBvU->bAaU', gamma1['a'], h['ab'][a,V,v,V], t['ab'][pv,pV,ha,hA], optimize=True)
    O['ab'][v,V,v,A] += scale * +1.00000000 * np.einsum('UV,uv,uUaW,bAvV->bAaW', gamma1['b'], gamma1['a'], h['ab'][a,A,v,A], t['ab'][pv,pV,ha,hA], optimize=True)
    O['ab'][v,V,v,A] += scale * +1.00000000 * np.einsum('UV,iUaW,bAiV->bAaW', gamma1['b'], h['ab'][c,A,v,A], t['ab'][pv,pV,hc,hA], optimize=True)
    O['ab'][v,V,v,A] += scale * +1.00000000 * np.einsum('iIaU,bAiI->bAaU', h['ab'][c,C,v,A], t['ab'][pv,pV,hc,hC], optimize=True)
    O['ab'][v,V,v,A] += scale * -1.00000000 * np.einsum('iAaB,bBiU->bAaU', h['ab'][c,V,v,V], t['ab'][pv,pV,hc,hA], optimize=True)
    O['ab'][a,A,v,C] += scale * +1.00000000 * np.einsum('uv,UV,uWaV,wUvI->wWaI', eta1['a'], gamma1['b'], h['ab'][a,A,v,A], t['ab'][pa,pA,ha,hC], optimize=True)
    
    O['ab'][a,A,v,C] += scale * -1.00000000 * np.einsum('UV,uv,uWaV,wUvI->wWaI', eta1['b'], gamma1['a'], h['ab'][a,A,v,A], t['ab'][pa,pA,ha,hC], optimize=True)
    O['ab'][a,A,v,C] += scale * -1.00000000 * np.einsum('UV,iWaV,uUiI->uWaI', eta1['b'], h['ab'][c,A,v,A], t['ab'][pa,pA,hc,hC], optimize=True)
    O['ab'][a,A,v,C] += scale * +1.00000000 * np.einsum('uv,uIaJ,wUvI->wUaJ', gamma1['a'], h['ab'][a,C,v,C], t['ab'][pa,pA,ha,hC], optimize=True)
    O['ab'][a,A,v,C] += scale * -1.00000000 * np.einsum('uv,uUaA,wAvI->wUaI', gamma1['a'], h['ab'][a,A,v,V], t['ab'][pa,pV,ha,hC], optimize=True)
    
    O['ab'][a,A,v,C] += scale * +1.00000000 * np.einsum('UV,iUaI,uWiV->uWaI', gamma1['b'], h['ab'][c,A,v,C], t['ab'][pa,pA,hc,hA], optimize=True)
    O['ab'][a,A,v,C] += scale * +1.00000000 * np.einsum('iIaJ,uUiI->uUaJ', h['ab'][c,C,v,C], t['ab'][pa,pA,hc,hC], optimize=True)
    O['ab'][a,A,v,C] += scale * -1.00000000 * np.einsum('iUaA,uAiI->uUaI', h['ab'][c,A,v,V], t['ab'][pa,pV,hc,hC], optimize=True)
    O['ab'][v,A,v,C] += scale * +1.00000000 * np.einsum('uv,UV,uWaV,bUvI->bWaI', eta1['a'], gamma1['b'], h['ab'][a,A,v,A], t['ab'][pv,pA,ha,hC], optimize=True)
    O['ab'][v,A,v,C] += scale * -1.00000000 * np.einsum('UV,uv,uUaI,bWvV->bWaI', eta1['b'], eta1['a'], h['ab'][a,A,v,C], t['ab'][pv,pA,ha,hA], optimize=True)
    O['ab'][v,A,v,C] += scale * -1.00000000 * np.einsum('UV,uv,uWaV,bUvI->bWaI', eta1['b'], gamma1['a'], h['ab'][a,A,v,A], t['ab'][pv,pA,ha,hC], optimize=True)
    O['ab'][v,A,v,C] += scale * -1.00000000 * np.einsum('UV,iWaV,bUiI->bWaI', eta1['b'], h['ab'][c,A,v,A], t['ab'][pv,pA,hc,hC], optimize=True)
    O['ab'][v,A,v,C] += scale * +1.00000000 * np.einsum('uv,uIaJ,bUvI->bUaJ', gamma1['a'], h['ab'][a,C,v,C], t['ab'][pv,pA,ha,hC], optimize=True)
    O['ab'][v,A,v,C] += scale * -1.00000000 * np.einsum('uv,uUaA,bAvI->bUaI', gamma1['a'], h['ab'][a,A,v,V], t['ab'][pv,pV,ha,hC], optimize=True)
    O['ab'][v,A,v,C] += scale * +1.00000000 * np.einsum('UV,uv,uUaI,bWvV->bWaI', gamma1['b'], gamma1['a'], h['ab'][a,A,v,C], t['ab'][pv,pA,ha,hA], optimize=True)
    O['ab'][v,A,v,C] += scale * +1.00000000 * np.einsum('UV,iUaI,bWiV->bWaI', gamma1['b'], h['ab'][c,A,v,C], t['ab'][pv,pA,hc,hA], optimize=True)
    O['ab'][v,A,v,C] += scale * +1.00000000 * np.einsum('iIaJ,bUiI->bUaJ', h['ab'][c,C,v,C], t['ab'][pv,pA,hc,hC], optimize=True)
    O['ab'][v,A,v,C] += scale * -1.00000000 * np.einsum('iUaA,bAiI->bUaI', h['ab'][c,A,v,V], t['ab'][pv,pV,hc,hC], optimize=True)
    O['ab'][a,C,v,C] += scale * +1.00000000 * np.einsum('uv,UV,uIaV,wUvJ->wIaJ', eta1['a'], gamma1['b'], h['ab'][a,C,v,A], t['ab'][pa,pA,ha,hC], optimize=True)
    O['ab'][a,C,v,C] += scale * -1.00000000 * np.einsum('UV,uv,uIaV,wUvJ->wIaJ', eta1['b'], gamma1['a'], h['ab'][a,C,v,A], t['ab'][pa,pA,ha,hC], optimize=True)
    O['ab'][a,C,v,C] += scale * -1.00000000 * np.einsum('UV,iIaV,uUiJ->uIaJ', eta1['b'], h['ab'][c,C,v,A], t['ab'][pa,pA,hc,hC], optimize=True)
    O['ab'][a,C,v,C] += scale * -1.00000000 * np.einsum('uv,uIaA,wAvJ->wIaJ', gamma1['a'], h['ab'][a,C,v,V], t['ab'][pa,pV,ha,hC], optimize=True)
    O['ab'][a,C,v,C] += scale * -1.00000000 * np.einsum('iIaA,uAiJ->uIaJ', h['ab'][c,C,v,V], t['ab'][pa,pV,hc,hC], optimize=True)
    O['ab'][v,C,v,C] += scale * +1.00000000 * np.einsum('uv,UV,uIaV,bUvJ->bIaJ', eta1['a'], gamma1['b'], h['ab'][a,C,v,A], t['ab'][pv,pA,ha,hC], optimize=True)
    O['ab'][v,C,v,C] += scale * -1.00000000 * np.einsum('UV,uv,uIaV,bUvJ->bIaJ', eta1['b'], gamma1['a'], h['ab'][a,C,v,A], t['ab'][pv,pA,ha,hC], optimize=True)
    O['ab'][v,C,v,C] += scale * -1.00000000 * np.einsum('UV,iIaV,bUiJ->bIaJ', eta1['b'], h['ab'][c,C,v,A], t['ab'][pv,pA,hc,hC], optimize=True)
    O['ab'][v,C,v,C] += scale * -1.00000000 * np.einsum('uv,uIaA,bAvJ->bIaJ', gamma1['a'], h['ab'][a,C,v,V], t['ab'][pv,pV,ha,hC], optimize=True)
    O['ab'][v,C,v,C] += scale * -1.00000000 * np.einsum('iIaA,bAiJ->bIaJ', h['ab'][c,C,v,V], t['ab'][pv,pV,hc,hC], optimize=True)
    O['ab'][a,V,v,C] += scale * +1.00000000 * np.einsum('uv,UV,uAaV,wUvI->wAaI', eta1['a'], gamma1['b'], h['ab'][a,V,v,A], t['ab'][pa,pA,ha,hC], optimize=True)
    O['ab'][a,V,v,C] += scale * -1.00000000 * np.einsum('UV,uv,uUaI,wAvV->wAaI', eta1['b'], eta1['a'], h['ab'][a,A,v,C], t['ab'][pa,pV,ha,hA], optimize=True)
    O['ab'][a,V,v,C] += scale * -1.00000000 * np.einsum('UV,uv,uAaV,wUvI->wAaI', eta1['b'], gamma1['a'], h['ab'][a,V,v,A], t['ab'][pa,pA,ha,hC], optimize=True)
    O['ab'][a,V,v,C] += scale * -1.00000000 * np.einsum('UV,iAaV,uUiI->uAaI', eta1['b'], h['ab'][c,V,v,A], t['ab'][pa,pA,hc,hC], optimize=True)
    O['ab'][a,V,v,C] += scale * +1.00000000 * np.einsum('uv,uIaJ,wAvI->wAaJ', gamma1['a'], h['ab'][a,C,v,C], t['ab'][pa,pV,ha,hC], optimize=True)
    O['ab'][a,V,v,C] += scale * -1.00000000 * np.einsum('uv,uAaB,wBvI->wAaI', gamma1['a'], h['ab'][a,V,v,V], t['ab'][pa,pV,ha,hC], optimize=True)
    O['ab'][a,V,v,C] += scale * +1.00000000 * np.einsum('UV,uv,uUaI,wAvV->wAaI', gamma1['b'], gamma1['a'], h['ab'][a,A,v,C], t['ab'][pa,pV,ha,hA], optimize=True)
    O['ab'][a,V,v,C] += scale * +1.00000000 * np.einsum('UV,iUaI,uAiV->uAaI', gamma1['b'], h['ab'][c,A,v,C], t['ab'][pa,pV,hc,hA], optimize=True)
    O['ab'][a,V,v,C] += scale * +1.00000000 * np.einsum('iIaJ,uAiI->uAaJ', h['ab'][c,C,v,C], t['ab'][pa,pV,hc,hC], optimize=True)
    O['ab'][a,V,v,C] += scale * -1.00000000 * np.einsum('iAaB,uBiI->uAaI', h['ab'][c,V,v,V], t['ab'][pa,pV,hc,hC], optimize=True)
    O['ab'][v,V,v,C] += scale * +1.00000000 * np.einsum('uv,UV,uAaV,bUvI->bAaI', eta1['a'], gamma1['b'], h['ab'][a,V,v,A], t['ab'][pv,pA,ha,hC], optimize=True)
    O['ab'][v,V,v,C] += scale * -1.00000000 * np.einsum('UV,uv,uUaI,bAvV->bAaI', eta1['b'], eta1['a'], h['ab'][a,A,v,C], t['ab'][pv,pV,ha,hA], optimize=True)
    O['ab'][v,V,v,C] += scale * -1.00000000 * np.einsum('UV,uv,uAaV,bUvI->bAaI', eta1['b'], gamma1['a'], h['ab'][a,V,v,A], t['ab'][pv,pA,ha,hC], optimize=True)
    O['ab'][v,V,v,C] += scale * -1.00000000 * np.einsum('UV,iAaV,bUiI->bAaI', eta1['b'], h['ab'][c,V,v,A], t['ab'][pv,pA,hc,hC], optimize=True)
    O['ab'][v,V,v,C] += scale * +1.00000000 * np.einsum('uv,uIaJ,bAvI->bAaJ', gamma1['a'], h['ab'][a,C,v,C], t['ab'][pv,pV,ha,hC], optimize=True)
    O['ab'][v,V,v,C] += scale * -1.00000000 * np.einsum('uv,uAaB,bBvI->bAaI', gamma1['a'], h['ab'][a,V,v,V], t['ab'][pv,pV,ha,hC], optimize=True)
    O['ab'][v,V,v,C] += scale * +1.00000000 * np.einsum('UV,uv,uUaI,bAvV->bAaI', gamma1['b'], gamma1['a'], h['ab'][a,A,v,C], t['ab'][pv,pV,ha,hA], optimize=True)
    O['ab'][v,V,v,C] += scale * +1.00000000 * np.einsum('UV,iUaI,bAiV->bAaI', gamma1['b'], h['ab'][c,A,v,C], t['ab'][pv,pV,hc,hA], optimize=True)
    O['ab'][v,V,v,C] += scale * +1.00000000 * np.einsum('iIaJ,bAiI->bAaJ', h['ab'][c,C,v,C], t['ab'][pv,pV,hc,hC], optimize=True)
    O['ab'][v,V,v,C] += scale * -1.00000000 * np.einsum('iAaB,bBiI->bAaI', h['ab'][c,V,v,V], t['ab'][pv,pV,hc,hC], optimize=True)
    
    O['ab'][a,A,v,V] += scale * +1.00000000 * np.einsum('uv,uIaA,wUvI->wUaA', gamma1['a'], h['ab'][a,C,v,V], t['ab'][pa,pA,ha,hC], optimize=True)
    
    O['ab'][a,A,v,V] += scale * +1.00000000 * np.einsum('UV,iUaA,uWiV->uWaA', gamma1['b'], h['ab'][c,A,v,V], t['ab'][pa,pA,hc,hA], optimize=True)
    O['ab'][a,A,v,V] += scale * +1.00000000 * np.einsum('iIaA,uUiI->uUaA', h['ab'][c,C,v,V], t['ab'][pa,pA,hc,hC], optimize=True)
    O['ab'][v,A,v,V] += scale * -1.00000000 * np.einsum('UV,uv,uUaA,bWvV->bWaA', eta1['b'], eta1['a'], h['ab'][a,A,v,V], t['ab'][pv,pA,ha,hA], optimize=True)
    O['ab'][v,A,v,V] += scale * +1.00000000 * np.einsum('uv,uIaA,bUvI->bUaA', gamma1['a'], h['ab'][a,C,v,V], t['ab'][pv,pA,ha,hC], optimize=True)
    O['ab'][v,A,v,V] += scale * +1.00000000 * np.einsum('UV,uv,uUaA,bWvV->bWaA', gamma1['b'], gamma1['a'], h['ab'][a,A,v,V], t['ab'][pv,pA,ha,hA], optimize=True)
    O['ab'][v,A,v,V] += scale * +1.00000000 * np.einsum('UV,iUaA,bWiV->bWaA', gamma1['b'], h['ab'][c,A,v,V], t['ab'][pv,pA,hc,hA], optimize=True)
    O['ab'][v,A,v,V] += scale * +1.00000000 * np.einsum('iIaA,bUiI->bUaA', h['ab'][c,C,v,V], t['ab'][pv,pA,hc,hC], optimize=True)
    O['ab'][a,V,v,V] += scale * -1.00000000 * np.einsum('UV,uv,uUaA,wBvV->wBaA', eta1['b'], eta1['a'], h['ab'][a,A,v,V], t['ab'][pa,pV,ha,hA], optimize=True)
    O['ab'][a,V,v,V] += scale * +1.00000000 * np.einsum('uv,uIaA,wBvI->wBaA', gamma1['a'], h['ab'][a,C,v,V], t['ab'][pa,pV,ha,hC], optimize=True)
    O['ab'][a,V,v,V] += scale * +1.00000000 * np.einsum('UV,uv,uUaA,wBvV->wBaA', gamma1['b'], gamma1['a'], h['ab'][a,A,v,V], t['ab'][pa,pV,ha,hA], optimize=True)
    O['ab'][a,V,v,V] += scale * +1.00000000 * np.einsum('UV,iUaA,uBiV->uBaA', gamma1['b'], h['ab'][c,A,v,V], t['ab'][pa,pV,hc,hA], optimize=True)
    O['ab'][a,V,v,V] += scale * +1.00000000 * np.einsum('iIaA,uBiI->uBaA', h['ab'][c,C,v,V], t['ab'][pa,pV,hc,hC], optimize=True)
    O['ab'][v,V,v,V] += scale * -1.00000000 * np.einsum('UV,uv,uUaA,bBvV->bBaA', eta1['b'], eta1['a'], h['ab'][a,A,v,V], t['ab'][pv,pV,ha,hA], optimize=True)
    O['ab'][v,V,v,V] += scale * +1.00000000 * np.einsum('uv,uIaA,bBvI->bBaA', gamma1['a'], h['ab'][a,C,v,V], t['ab'][pv,pV,ha,hC], optimize=True)
    O['ab'][v,V,v,V] += scale * +1.00000000 * np.einsum('UV,uv,uUaA,bBvV->bBaA', gamma1['b'], gamma1['a'], h['ab'][a,A,v,V], t['ab'][pv,pV,ha,hA], optimize=True)
    O['ab'][v,V,v,V] += scale * +1.00000000 * np.einsum('UV,iUaA,bBiV->bBaA', gamma1['b'], h['ab'][c,A,v,V], t['ab'][pv,pV,hc,hA], optimize=True)
    O['ab'][v,V,v,V] += scale * +1.00000000 * np.einsum('iIaA,bBiI->bBaA', h['ab'][c,C,v,V], t['ab'][pv,pV,hc,hC], optimize=True)

    t1 = time.time()
    if verbose: print("h2b_t2b_c2b took {:.4f} seconds to run.".format(t1-t0))

    return O
def h2b_t2c_c2b(O, h, t, gamma1, eta1, lambdas, orbspace, verbose=False, scale=1.0):
    # 180 lines
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

    
    
    O['ab'][a,A,a,A] += scale * -1.00000000 * np.einsum('UV,uIvV,WUIX->uWvX', eta1['b'], h['ab'][a,C,a,A], t['bb'][pA,pA,hC,hA], optimize=True)
    O['ab'][a,A,a,A] += scale * +1.00000000 * np.einsum('UV,uUvA,WAXV->uWvX', gamma1['b'], h['ab'][a,A,a,V], t['bb'][pA,pV,hA,hA], optimize=True)
    O['ab'][a,A,a,A] += scale * -1.00000000 * np.einsum('uIvA,UAIV->uUvV', h['ab'][a,C,a,V], t['bb'][pA,pV,hC,hA], optimize=True)
    
    
    O['ab'][c,A,a,A] += scale * -1.00000000 * np.einsum('UV,iIuV,WUIX->iWuX', eta1['b'], h['ab'][c,C,a,A], t['bb'][pA,pA,hC,hA], optimize=True)
    O['ab'][c,A,a,A] += scale * +1.00000000 * np.einsum('UV,iUuA,WAXV->iWuX', gamma1['b'], h['ab'][c,A,a,V], t['bb'][pA,pV,hA,hA], optimize=True)
    O['ab'][c,A,a,A] += scale * -1.00000000 * np.einsum('iIuA,UAIV->iUuV', h['ab'][c,C,a,V], t['bb'][pA,pV,hC,hA], optimize=True)
    
    
    O['ab'][v,A,a,A] += scale * -1.00000000 * np.einsum('UV,aIuV,WUIX->aWuX', eta1['b'], h['ab'][v,C,a,A], t['bb'][pA,pA,hC,hA], optimize=True)
    O['ab'][v,A,a,A] += scale * +1.00000000 * np.einsum('UV,aUuA,WAXV->aWuX', gamma1['b'], h['ab'][v,A,a,V], t['bb'][pA,pV,hA,hA], optimize=True)
    O['ab'][v,A,a,A] += scale * -1.00000000 * np.einsum('aIuA,UAIV->aUuV', h['ab'][v,C,a,V], t['bb'][pA,pV,hC,hA], optimize=True)
    O['ab'][a,V,a,A] += scale * +1.00000000 * np.einsum('UV,WX,uUvX,WAYV->uAvY', eta1['b'], gamma1['b'], h['ab'][a,A,a,A], t['bb'][pA,pV,hA,hA], optimize=True)
    O['ab'][a,V,a,A] += scale * -1.00000000 * np.einsum('UV,WX,uWvV,UAYX->uAvY', eta1['b'], gamma1['b'], h['ab'][a,A,a,A], t['bb'][pA,pV,hA,hA], optimize=True)
    O['ab'][a,V,a,A] += scale * +1.00000000 * np.einsum('UV,uIvV,UAIW->uAvW', eta1['b'], h['ab'][a,C,a,A], t['bb'][pA,pV,hC,hA], optimize=True)
    O['ab'][a,V,a,A] += scale * +1.00000000 * np.einsum('UV,uUvA,BAWV->uBvW', gamma1['b'], h['ab'][a,A,a,V], t['bb'][pV,pV,hA,hA], optimize=True)
    O['ab'][a,V,a,A] += scale * -1.00000000 * np.einsum('uIvA,BAIU->uBvU', h['ab'][a,C,a,V], t['bb'][pV,pV,hC,hA], optimize=True)
    O['ab'][c,V,a,A] += scale * +1.00000000 * np.einsum('UV,WX,iUuX,WAYV->iAuY', eta1['b'], gamma1['b'], h['ab'][c,A,a,A], t['bb'][pA,pV,hA,hA], optimize=True)
    O['ab'][c,V,a,A] += scale * -1.00000000 * np.einsum('UV,WX,iWuV,UAYX->iAuY', eta1['b'], gamma1['b'], h['ab'][c,A,a,A], t['bb'][pA,pV,hA,hA], optimize=True)
    O['ab'][c,V,a,A] += scale * +1.00000000 * np.einsum('UV,iIuV,UAIW->iAuW', eta1['b'], h['ab'][c,C,a,A], t['bb'][pA,pV,hC,hA], optimize=True)
    O['ab'][c,V,a,A] += scale * +1.00000000 * np.einsum('UV,iUuA,BAWV->iBuW', gamma1['b'], h['ab'][c,A,a,V], t['bb'][pV,pV,hA,hA], optimize=True)
    O['ab'][c,V,a,A] += scale * -1.00000000 * np.einsum('iIuA,BAIU->iBuU', h['ab'][c,C,a,V], t['bb'][pV,pV,hC,hA], optimize=True)
    O['ab'][v,V,a,A] += scale * +1.00000000 * np.einsum('UV,WX,aUuX,WAYV->aAuY', eta1['b'], gamma1['b'], h['ab'][v,A,a,A], t['bb'][pA,pV,hA,hA], optimize=True)
    O['ab'][v,V,a,A] += scale * -1.00000000 * np.einsum('UV,WX,aWuV,UAYX->aAuY', eta1['b'], gamma1['b'], h['ab'][v,A,a,A], t['bb'][pA,pV,hA,hA], optimize=True)
    O['ab'][v,V,a,A] += scale * +1.00000000 * np.einsum('UV,aIuV,UAIW->aAuW', eta1['b'], h['ab'][v,C,a,A], t['bb'][pA,pV,hC,hA], optimize=True)
    O['ab'][v,V,a,A] += scale * +1.00000000 * np.einsum('UV,aUuA,BAWV->aBuW', gamma1['b'], h['ab'][v,A,a,V], t['bb'][pV,pV,hA,hA], optimize=True)
    O['ab'][v,V,a,A] += scale * -1.00000000 * np.einsum('aIuA,BAIU->aBuU', h['ab'][v,C,a,V], t['bb'][pV,pV,hC,hA], optimize=True)
    O['ab'][a,A,a,C] += scale * -1.00000000 * np.einsum('UV,WX,uUvX,YWIV->uYvI', eta1['b'], gamma1['b'], h['ab'][a,A,a,A], t['bb'][pA,pA,hC,hA], optimize=True)
    O['ab'][a,A,a,C] += scale * +1.00000000 * np.einsum('UV,WX,uWvV,YUIX->uYvI', eta1['b'], gamma1['b'], h['ab'][a,A,a,A], t['bb'][pA,pA,hC,hA], optimize=True)
    O['ab'][a,A,a,C] += scale * +1.00000000 * np.einsum('UV,uIvV,WUJI->uWvJ', eta1['b'], h['ab'][a,C,a,A], t['bb'][pA,pA,hC,hC], optimize=True)
    O['ab'][a,A,a,C] += scale * +1.00000000 * np.einsum('UV,uUvA,WAIV->uWvI', gamma1['b'], h['ab'][a,A,a,V], t['bb'][pA,pV,hC,hA], optimize=True)
    O['ab'][a,A,a,C] += scale * +1.00000000 * np.einsum('uIvA,UAJI->uUvJ', h['ab'][a,C,a,V], t['bb'][pA,pV,hC,hC], optimize=True)
    O['ab'][c,A,a,C] += scale * -1.00000000 * np.einsum('UV,WX,iUuX,YWIV->iYuI', eta1['b'], gamma1['b'], h['ab'][c,A,a,A], t['bb'][pA,pA,hC,hA], optimize=True)
    O['ab'][c,A,a,C] += scale * +1.00000000 * np.einsum('UV,WX,iWuV,YUIX->iYuI', eta1['b'], gamma1['b'], h['ab'][c,A,a,A], t['bb'][pA,pA,hC,hA], optimize=True)
    O['ab'][c,A,a,C] += scale * +1.00000000 * np.einsum('UV,iIuV,WUJI->iWuJ', eta1['b'], h['ab'][c,C,a,A], t['bb'][pA,pA,hC,hC], optimize=True)
    O['ab'][c,A,a,C] += scale * +1.00000000 * np.einsum('UV,iUuA,WAIV->iWuI', gamma1['b'], h['ab'][c,A,a,V], t['bb'][pA,pV,hC,hA], optimize=True)
    O['ab'][c,A,a,C] += scale * +1.00000000 * np.einsum('iIuA,UAJI->iUuJ', h['ab'][c,C,a,V], t['bb'][pA,pV,hC,hC], optimize=True)
    O['ab'][v,A,a,C] += scale * -1.00000000 * np.einsum('UV,WX,aUuX,YWIV->aYuI', eta1['b'], gamma1['b'], h['ab'][v,A,a,A], t['bb'][pA,pA,hC,hA], optimize=True)
    O['ab'][v,A,a,C] += scale * +1.00000000 * np.einsum('UV,WX,aWuV,YUIX->aYuI', eta1['b'], gamma1['b'], h['ab'][v,A,a,A], t['bb'][pA,pA,hC,hA], optimize=True)
    O['ab'][v,A,a,C] += scale * +1.00000000 * np.einsum('UV,aIuV,WUJI->aWuJ', eta1['b'], h['ab'][v,C,a,A], t['bb'][pA,pA,hC,hC], optimize=True)
    O['ab'][v,A,a,C] += scale * +1.00000000 * np.einsum('UV,aUuA,WAIV->aWuI', gamma1['b'], h['ab'][v,A,a,V], t['bb'][pA,pV,hC,hA], optimize=True)
    O['ab'][v,A,a,C] += scale * +1.00000000 * np.einsum('aIuA,UAJI->aUuJ', h['ab'][v,C,a,V], t['bb'][pA,pV,hC,hC], optimize=True)
    O['ab'][a,V,a,C] += scale * +1.00000000 * np.einsum('UV,WX,uUvX,WAIV->uAvI', eta1['b'], gamma1['b'], h['ab'][a,A,a,A], t['bb'][pA,pV,hC,hA], optimize=True)
    O['ab'][a,V,a,C] += scale * -1.00000000 * np.einsum('UV,WX,uWvV,UAIX->uAvI', eta1['b'], gamma1['b'], h['ab'][a,A,a,A], t['bb'][pA,pV,hC,hA], optimize=True)
    O['ab'][a,V,a,C] += scale * -1.00000000 * np.einsum('UV,uIvV,UAJI->uAvJ', eta1['b'], h['ab'][a,C,a,A], t['bb'][pA,pV,hC,hC], optimize=True)
    O['ab'][a,V,a,C] += scale * +1.00000000 * np.einsum('UV,uUvA,BAIV->uBvI', gamma1['b'], h['ab'][a,A,a,V], t['bb'][pV,pV,hC,hA], optimize=True)
    O['ab'][a,V,a,C] += scale * +1.00000000 * np.einsum('uIvA,BAJI->uBvJ', h['ab'][a,C,a,V], t['bb'][pV,pV,hC,hC], optimize=True)
    O['ab'][c,V,a,C] += scale * +1.00000000 * np.einsum('UV,WX,iUuX,WAIV->iAuI', eta1['b'], gamma1['b'], h['ab'][c,A,a,A], t['bb'][pA,pV,hC,hA], optimize=True)
    O['ab'][c,V,a,C] += scale * -1.00000000 * np.einsum('UV,WX,iWuV,UAIX->iAuI', eta1['b'], gamma1['b'], h['ab'][c,A,a,A], t['bb'][pA,pV,hC,hA], optimize=True)
    O['ab'][c,V,a,C] += scale * -1.00000000 * np.einsum('UV,iIuV,UAJI->iAuJ', eta1['b'], h['ab'][c,C,a,A], t['bb'][pA,pV,hC,hC], optimize=True)
    O['ab'][c,V,a,C] += scale * +1.00000000 * np.einsum('UV,iUuA,BAIV->iBuI', gamma1['b'], h['ab'][c,A,a,V], t['bb'][pV,pV,hC,hA], optimize=True)
    O['ab'][c,V,a,C] += scale * +1.00000000 * np.einsum('iIuA,BAJI->iBuJ', h['ab'][c,C,a,V], t['bb'][pV,pV,hC,hC], optimize=True)
    O['ab'][v,V,a,C] += scale * +1.00000000 * np.einsum('UV,WX,aUuX,WAIV->aAuI', eta1['b'], gamma1['b'], h['ab'][v,A,a,A], t['bb'][pA,pV,hC,hA], optimize=True)
    O['ab'][v,V,a,C] += scale * -1.00000000 * np.einsum('UV,WX,aWuV,UAIX->aAuI', eta1['b'], gamma1['b'], h['ab'][v,A,a,A], t['bb'][pA,pV,hC,hA], optimize=True)
    O['ab'][v,V,a,C] += scale * -1.00000000 * np.einsum('UV,aIuV,UAJI->aAuJ', eta1['b'], h['ab'][v,C,a,A], t['bb'][pA,pV,hC,hC], optimize=True)
    O['ab'][v,V,a,C] += scale * +1.00000000 * np.einsum('UV,aUuA,BAIV->aBuI', gamma1['b'], h['ab'][v,A,a,V], t['bb'][pV,pV,hC,hA], optimize=True)
    O['ab'][v,V,a,C] += scale * +1.00000000 * np.einsum('aIuA,BAJI->aBuJ', h['ab'][v,C,a,V], t['bb'][pV,pV,hC,hC], optimize=True)
    
    
    O['ab'][a,A,c,A] += scale * -1.00000000 * np.einsum('UV,uIiV,WUIX->uWiX', eta1['b'], h['ab'][a,C,c,A], t['bb'][pA,pA,hC,hA], optimize=True)
    O['ab'][a,A,c,A] += scale * +1.00000000 * np.einsum('UV,uUiA,WAXV->uWiX', gamma1['b'], h['ab'][a,A,c,V], t['bb'][pA,pV,hA,hA], optimize=True)
    O['ab'][a,A,c,A] += scale * -1.00000000 * np.einsum('uIiA,UAIV->uUiV', h['ab'][a,C,c,V], t['bb'][pA,pV,hC,hA], optimize=True)
    
    
    O['ab'][c,A,c,A] += scale * -1.00000000 * np.einsum('UV,iIjV,WUIX->iWjX', eta1['b'], h['ab'][c,C,c,A], t['bb'][pA,pA,hC,hA], optimize=True)
    O['ab'][c,A,c,A] += scale * +1.00000000 * np.einsum('UV,iUjA,WAXV->iWjX', gamma1['b'], h['ab'][c,A,c,V], t['bb'][pA,pV,hA,hA], optimize=True)
    O['ab'][c,A,c,A] += scale * -1.00000000 * np.einsum('iIjA,UAIV->iUjV', h['ab'][c,C,c,V], t['bb'][pA,pV,hC,hA], optimize=True)
    
    
    O['ab'][v,A,c,A] += scale * -1.00000000 * np.einsum('UV,aIiV,WUIX->aWiX', eta1['b'], h['ab'][v,C,c,A], t['bb'][pA,pA,hC,hA], optimize=True)
    O['ab'][v,A,c,A] += scale * +1.00000000 * np.einsum('UV,aUiA,WAXV->aWiX', gamma1['b'], h['ab'][v,A,c,V], t['bb'][pA,pV,hA,hA], optimize=True)
    O['ab'][v,A,c,A] += scale * -1.00000000 * np.einsum('aIiA,UAIV->aUiV', h['ab'][v,C,c,V], t['bb'][pA,pV,hC,hA], optimize=True)
    O['ab'][a,V,c,A] += scale * +1.00000000 * np.einsum('UV,WX,uUiX,WAYV->uAiY', eta1['b'], gamma1['b'], h['ab'][a,A,c,A], t['bb'][pA,pV,hA,hA], optimize=True)
    O['ab'][a,V,c,A] += scale * -1.00000000 * np.einsum('UV,WX,uWiV,UAYX->uAiY', eta1['b'], gamma1['b'], h['ab'][a,A,c,A], t['bb'][pA,pV,hA,hA], optimize=True)
    O['ab'][a,V,c,A] += scale * +1.00000000 * np.einsum('UV,uIiV,UAIW->uAiW', eta1['b'], h['ab'][a,C,c,A], t['bb'][pA,pV,hC,hA], optimize=True)
    O['ab'][a,V,c,A] += scale * +1.00000000 * np.einsum('UV,uUiA,BAWV->uBiW', gamma1['b'], h['ab'][a,A,c,V], t['bb'][pV,pV,hA,hA], optimize=True)
    O['ab'][a,V,c,A] += scale * -1.00000000 * np.einsum('uIiA,BAIU->uBiU', h['ab'][a,C,c,V], t['bb'][pV,pV,hC,hA], optimize=True)
    O['ab'][c,V,c,A] += scale * +1.00000000 * np.einsum('UV,WX,iUjX,WAYV->iAjY', eta1['b'], gamma1['b'], h['ab'][c,A,c,A], t['bb'][pA,pV,hA,hA], optimize=True)
    O['ab'][c,V,c,A] += scale * -1.00000000 * np.einsum('UV,WX,iWjV,UAYX->iAjY', eta1['b'], gamma1['b'], h['ab'][c,A,c,A], t['bb'][pA,pV,hA,hA], optimize=True)
    O['ab'][c,V,c,A] += scale * +1.00000000 * np.einsum('UV,iIjV,UAIW->iAjW', eta1['b'], h['ab'][c,C,c,A], t['bb'][pA,pV,hC,hA], optimize=True)
    O['ab'][c,V,c,A] += scale * +1.00000000 * np.einsum('UV,iUjA,BAWV->iBjW', gamma1['b'], h['ab'][c,A,c,V], t['bb'][pV,pV,hA,hA], optimize=True)
    O['ab'][c,V,c,A] += scale * -1.00000000 * np.einsum('iIjA,BAIU->iBjU', h['ab'][c,C,c,V], t['bb'][pV,pV,hC,hA], optimize=True)
    O['ab'][v,V,c,A] += scale * +1.00000000 * np.einsum('UV,WX,aUiX,WAYV->aAiY', eta1['b'], gamma1['b'], h['ab'][v,A,c,A], t['bb'][pA,pV,hA,hA], optimize=True)
    O['ab'][v,V,c,A] += scale * -1.00000000 * np.einsum('UV,WX,aWiV,UAYX->aAiY', eta1['b'], gamma1['b'], h['ab'][v,A,c,A], t['bb'][pA,pV,hA,hA], optimize=True)
    O['ab'][v,V,c,A] += scale * +1.00000000 * np.einsum('UV,aIiV,UAIW->aAiW', eta1['b'], h['ab'][v,C,c,A], t['bb'][pA,pV,hC,hA], optimize=True)
    O['ab'][v,V,c,A] += scale * +1.00000000 * np.einsum('UV,aUiA,BAWV->aBiW', gamma1['b'], h['ab'][v,A,c,V], t['bb'][pV,pV,hA,hA], optimize=True)
    O['ab'][v,V,c,A] += scale * -1.00000000 * np.einsum('aIiA,BAIU->aBiU', h['ab'][v,C,c,V], t['bb'][pV,pV,hC,hA], optimize=True)
    O['ab'][a,A,c,C] += scale * -1.00000000 * np.einsum('UV,WX,uUiX,YWIV->uYiI', eta1['b'], gamma1['b'], h['ab'][a,A,c,A], t['bb'][pA,pA,hC,hA], optimize=True)
    O['ab'][a,A,c,C] += scale * +1.00000000 * np.einsum('UV,WX,uWiV,YUIX->uYiI', eta1['b'], gamma1['b'], h['ab'][a,A,c,A], t['bb'][pA,pA,hC,hA], optimize=True)
    O['ab'][a,A,c,C] += scale * +1.00000000 * np.einsum('UV,uIiV,WUJI->uWiJ', eta1['b'], h['ab'][a,C,c,A], t['bb'][pA,pA,hC,hC], optimize=True)
    O['ab'][a,A,c,C] += scale * +1.00000000 * np.einsum('UV,uUiA,WAIV->uWiI', gamma1['b'], h['ab'][a,A,c,V], t['bb'][pA,pV,hC,hA], optimize=True)
    O['ab'][a,A,c,C] += scale * +1.00000000 * np.einsum('uIiA,UAJI->uUiJ', h['ab'][a,C,c,V], t['bb'][pA,pV,hC,hC], optimize=True)
    O['ab'][c,A,c,C] += scale * -1.00000000 * np.einsum('UV,WX,iUjX,YWIV->iYjI', eta1['b'], gamma1['b'], h['ab'][c,A,c,A], t['bb'][pA,pA,hC,hA], optimize=True)
    O['ab'][c,A,c,C] += scale * +1.00000000 * np.einsum('UV,WX,iWjV,YUIX->iYjI', eta1['b'], gamma1['b'], h['ab'][c,A,c,A], t['bb'][pA,pA,hC,hA], optimize=True)
    O['ab'][c,A,c,C] += scale * +1.00000000 * np.einsum('UV,iIjV,WUJI->iWjJ', eta1['b'], h['ab'][c,C,c,A], t['bb'][pA,pA,hC,hC], optimize=True)
    O['ab'][c,A,c,C] += scale * +1.00000000 * np.einsum('UV,iUjA,WAIV->iWjI', gamma1['b'], h['ab'][c,A,c,V], t['bb'][pA,pV,hC,hA], optimize=True)
    O['ab'][c,A,c,C] += scale * +1.00000000 * np.einsum('iIjA,UAJI->iUjJ', h['ab'][c,C,c,V], t['bb'][pA,pV,hC,hC], optimize=True)
    O['ab'][v,A,c,C] += scale * -1.00000000 * np.einsum('UV,WX,aUiX,YWIV->aYiI', eta1['b'], gamma1['b'], h['ab'][v,A,c,A], t['bb'][pA,pA,hC,hA], optimize=True)
    O['ab'][v,A,c,C] += scale * +1.00000000 * np.einsum('UV,WX,aWiV,YUIX->aYiI', eta1['b'], gamma1['b'], h['ab'][v,A,c,A], t['bb'][pA,pA,hC,hA], optimize=True)
    O['ab'][v,A,c,C] += scale * +1.00000000 * np.einsum('UV,aIiV,WUJI->aWiJ', eta1['b'], h['ab'][v,C,c,A], t['bb'][pA,pA,hC,hC], optimize=True)
    O['ab'][v,A,c,C] += scale * +1.00000000 * np.einsum('UV,aUiA,WAIV->aWiI', gamma1['b'], h['ab'][v,A,c,V], t['bb'][pA,pV,hC,hA], optimize=True)
    O['ab'][v,A,c,C] += scale * +1.00000000 * np.einsum('aIiA,UAJI->aUiJ', h['ab'][v,C,c,V], t['bb'][pA,pV,hC,hC], optimize=True)
    O['ab'][a,V,c,C] += scale * +1.00000000 * np.einsum('UV,WX,uUiX,WAIV->uAiI', eta1['b'], gamma1['b'], h['ab'][a,A,c,A], t['bb'][pA,pV,hC,hA], optimize=True)
    O['ab'][a,V,c,C] += scale * -1.00000000 * np.einsum('UV,WX,uWiV,UAIX->uAiI', eta1['b'], gamma1['b'], h['ab'][a,A,c,A], t['bb'][pA,pV,hC,hA], optimize=True)
    O['ab'][a,V,c,C] += scale * -1.00000000 * np.einsum('UV,uIiV,UAJI->uAiJ', eta1['b'], h['ab'][a,C,c,A], t['bb'][pA,pV,hC,hC], optimize=True)
    O['ab'][a,V,c,C] += scale * +1.00000000 * np.einsum('UV,uUiA,BAIV->uBiI', gamma1['b'], h['ab'][a,A,c,V], t['bb'][pV,pV,hC,hA], optimize=True)
    O['ab'][a,V,c,C] += scale * +1.00000000 * np.einsum('uIiA,BAJI->uBiJ', h['ab'][a,C,c,V], t['bb'][pV,pV,hC,hC], optimize=True)
    O['ab'][c,V,c,C] += scale * +1.00000000 * np.einsum('UV,WX,iUjX,WAIV->iAjI', eta1['b'], gamma1['b'], h['ab'][c,A,c,A], t['bb'][pA,pV,hC,hA], optimize=True)
    O['ab'][c,V,c,C] += scale * -1.00000000 * np.einsum('UV,WX,iWjV,UAIX->iAjI', eta1['b'], gamma1['b'], h['ab'][c,A,c,A], t['bb'][pA,pV,hC,hA], optimize=True)
    O['ab'][c,V,c,C] += scale * -1.00000000 * np.einsum('UV,iIjV,UAJI->iAjJ', eta1['b'], h['ab'][c,C,c,A], t['bb'][pA,pV,hC,hC], optimize=True)
    O['ab'][c,V,c,C] += scale * +1.00000000 * np.einsum('UV,iUjA,BAIV->iBjI', gamma1['b'], h['ab'][c,A,c,V], t['bb'][pV,pV,hC,hA], optimize=True)
    O['ab'][c,V,c,C] += scale * +1.00000000 * np.einsum('iIjA,BAJI->iBjJ', h['ab'][c,C,c,V], t['bb'][pV,pV,hC,hC], optimize=True)
    O['ab'][v,V,c,C] += scale * +1.00000000 * np.einsum('UV,WX,aUiX,WAIV->aAiI', eta1['b'], gamma1['b'], h['ab'][v,A,c,A], t['bb'][pA,pV,hC,hA], optimize=True)
    O['ab'][v,V,c,C] += scale * -1.00000000 * np.einsum('UV,WX,aWiV,UAIX->aAiI', eta1['b'], gamma1['b'], h['ab'][v,A,c,A], t['bb'][pA,pV,hC,hA], optimize=True)
    O['ab'][v,V,c,C] += scale * -1.00000000 * np.einsum('UV,aIiV,UAJI->aAiJ', eta1['b'], h['ab'][v,C,c,A], t['bb'][pA,pV,hC,hC], optimize=True)
    O['ab'][v,V,c,C] += scale * +1.00000000 * np.einsum('UV,aUiA,BAIV->aBiI', gamma1['b'], h['ab'][v,A,c,V], t['bb'][pV,pV,hC,hA], optimize=True)
    O['ab'][v,V,c,C] += scale * +1.00000000 * np.einsum('aIiA,BAJI->aBiJ', h['ab'][v,C,c,V], t['bb'][pV,pV,hC,hC], optimize=True)
    
    
    O['ab'][a,A,v,A] += scale * -1.00000000 * np.einsum('UV,uIaV,WUIX->uWaX', eta1['b'], h['ab'][a,C,v,A], t['bb'][pA,pA,hC,hA], optimize=True)
    O['ab'][a,A,v,A] += scale * +1.00000000 * np.einsum('UV,uUaA,WAXV->uWaX', gamma1['b'], h['ab'][a,A,v,V], t['bb'][pA,pV,hA,hA], optimize=True)
    O['ab'][a,A,v,A] += scale * -1.00000000 * np.einsum('uIaA,UAIV->uUaV', h['ab'][a,C,v,V], t['bb'][pA,pV,hC,hA], optimize=True)
    
    
    O['ab'][c,A,v,A] += scale * -1.00000000 * np.einsum('UV,iIaV,WUIX->iWaX', eta1['b'], h['ab'][c,C,v,A], t['bb'][pA,pA,hC,hA], optimize=True)
    O['ab'][c,A,v,A] += scale * +1.00000000 * np.einsum('UV,iUaA,WAXV->iWaX', gamma1['b'], h['ab'][c,A,v,V], t['bb'][pA,pV,hA,hA], optimize=True)
    O['ab'][c,A,v,A] += scale * -1.00000000 * np.einsum('iIaA,UAIV->iUaV', h['ab'][c,C,v,V], t['bb'][pA,pV,hC,hA], optimize=True)
    
    
    O['ab'][v,A,v,A] += scale * -1.00000000 * np.einsum('UV,aIbV,WUIX->aWbX', eta1['b'], h['ab'][v,C,v,A], t['bb'][pA,pA,hC,hA], optimize=True)
    O['ab'][v,A,v,A] += scale * +1.00000000 * np.einsum('UV,aUbA,WAXV->aWbX', gamma1['b'], h['ab'][v,A,v,V], t['bb'][pA,pV,hA,hA], optimize=True)
    O['ab'][v,A,v,A] += scale * -1.00000000 * np.einsum('aIbA,UAIV->aUbV', h['ab'][v,C,v,V], t['bb'][pA,pV,hC,hA], optimize=True)
    O['ab'][a,V,v,A] += scale * +1.00000000 * np.einsum('UV,WX,uUaX,WAYV->uAaY', eta1['b'], gamma1['b'], h['ab'][a,A,v,A], t['bb'][pA,pV,hA,hA], optimize=True)
    O['ab'][a,V,v,A] += scale * -1.00000000 * np.einsum('UV,WX,uWaV,UAYX->uAaY', eta1['b'], gamma1['b'], h['ab'][a,A,v,A], t['bb'][pA,pV,hA,hA], optimize=True)
    O['ab'][a,V,v,A] += scale * +1.00000000 * np.einsum('UV,uIaV,UAIW->uAaW', eta1['b'], h['ab'][a,C,v,A], t['bb'][pA,pV,hC,hA], optimize=True)
    O['ab'][a,V,v,A] += scale * +1.00000000 * np.einsum('UV,uUaA,BAWV->uBaW', gamma1['b'], h['ab'][a,A,v,V], t['bb'][pV,pV,hA,hA], optimize=True)
    O['ab'][a,V,v,A] += scale * -1.00000000 * np.einsum('uIaA,BAIU->uBaU', h['ab'][a,C,v,V], t['bb'][pV,pV,hC,hA], optimize=True)
    O['ab'][c,V,v,A] += scale * +1.00000000 * np.einsum('UV,WX,iUaX,WAYV->iAaY', eta1['b'], gamma1['b'], h['ab'][c,A,v,A], t['bb'][pA,pV,hA,hA], optimize=True)
    O['ab'][c,V,v,A] += scale * -1.00000000 * np.einsum('UV,WX,iWaV,UAYX->iAaY', eta1['b'], gamma1['b'], h['ab'][c,A,v,A], t['bb'][pA,pV,hA,hA], optimize=True)
    O['ab'][c,V,v,A] += scale * +1.00000000 * np.einsum('UV,iIaV,UAIW->iAaW', eta1['b'], h['ab'][c,C,v,A], t['bb'][pA,pV,hC,hA], optimize=True)
    O['ab'][c,V,v,A] += scale * +1.00000000 * np.einsum('UV,iUaA,BAWV->iBaW', gamma1['b'], h['ab'][c,A,v,V], t['bb'][pV,pV,hA,hA], optimize=True)
    O['ab'][c,V,v,A] += scale * -1.00000000 * np.einsum('iIaA,BAIU->iBaU', h['ab'][c,C,v,V], t['bb'][pV,pV,hC,hA], optimize=True)
    O['ab'][v,V,v,A] += scale * +1.00000000 * np.einsum('UV,WX,aUbX,WAYV->aAbY', eta1['b'], gamma1['b'], h['ab'][v,A,v,A], t['bb'][pA,pV,hA,hA], optimize=True)
    O['ab'][v,V,v,A] += scale * -1.00000000 * np.einsum('UV,WX,aWbV,UAYX->aAbY', eta1['b'], gamma1['b'], h['ab'][v,A,v,A], t['bb'][pA,pV,hA,hA], optimize=True)
    O['ab'][v,V,v,A] += scale * +1.00000000 * np.einsum('UV,aIbV,UAIW->aAbW', eta1['b'], h['ab'][v,C,v,A], t['bb'][pA,pV,hC,hA], optimize=True)
    O['ab'][v,V,v,A] += scale * +1.00000000 * np.einsum('UV,aUbA,BAWV->aBbW', gamma1['b'], h['ab'][v,A,v,V], t['bb'][pV,pV,hA,hA], optimize=True)
    O['ab'][v,V,v,A] += scale * -1.00000000 * np.einsum('aIbA,BAIU->aBbU', h['ab'][v,C,v,V], t['bb'][pV,pV,hC,hA], optimize=True)
    O['ab'][a,A,v,C] += scale * -1.00000000 * np.einsum('UV,WX,uUaX,YWIV->uYaI', eta1['b'], gamma1['b'], h['ab'][a,A,v,A], t['bb'][pA,pA,hC,hA], optimize=True)
    O['ab'][a,A,v,C] += scale * +1.00000000 * np.einsum('UV,WX,uWaV,YUIX->uYaI', eta1['b'], gamma1['b'], h['ab'][a,A,v,A], t['bb'][pA,pA,hC,hA], optimize=True)
    O['ab'][a,A,v,C] += scale * +1.00000000 * np.einsum('UV,uIaV,WUJI->uWaJ', eta1['b'], h['ab'][a,C,v,A], t['bb'][pA,pA,hC,hC], optimize=True)
    O['ab'][a,A,v,C] += scale * +1.00000000 * np.einsum('UV,uUaA,WAIV->uWaI', gamma1['b'], h['ab'][a,A,v,V], t['bb'][pA,pV,hC,hA], optimize=True)
    O['ab'][a,A,v,C] += scale * +1.00000000 * np.einsum('uIaA,UAJI->uUaJ', h['ab'][a,C,v,V], t['bb'][pA,pV,hC,hC], optimize=True)
    O['ab'][c,A,v,C] += scale * -1.00000000 * np.einsum('UV,WX,iUaX,YWIV->iYaI', eta1['b'], gamma1['b'], h['ab'][c,A,v,A], t['bb'][pA,pA,hC,hA], optimize=True)
    O['ab'][c,A,v,C] += scale * +1.00000000 * np.einsum('UV,WX,iWaV,YUIX->iYaI', eta1['b'], gamma1['b'], h['ab'][c,A,v,A], t['bb'][pA,pA,hC,hA], optimize=True)
    O['ab'][c,A,v,C] += scale * +1.00000000 * np.einsum('UV,iIaV,WUJI->iWaJ', eta1['b'], h['ab'][c,C,v,A], t['bb'][pA,pA,hC,hC], optimize=True)
    O['ab'][c,A,v,C] += scale * +1.00000000 * np.einsum('UV,iUaA,WAIV->iWaI', gamma1['b'], h['ab'][c,A,v,V], t['bb'][pA,pV,hC,hA], optimize=True)
    O['ab'][c,A,v,C] += scale * +1.00000000 * np.einsum('iIaA,UAJI->iUaJ', h['ab'][c,C,v,V], t['bb'][pA,pV,hC,hC], optimize=True)
    O['ab'][v,A,v,C] += scale * -1.00000000 * np.einsum('UV,WX,aUbX,YWIV->aYbI', eta1['b'], gamma1['b'], h['ab'][v,A,v,A], t['bb'][pA,pA,hC,hA], optimize=True)
    O['ab'][v,A,v,C] += scale * +1.00000000 * np.einsum('UV,WX,aWbV,YUIX->aYbI', eta1['b'], gamma1['b'], h['ab'][v,A,v,A], t['bb'][pA,pA,hC,hA], optimize=True)
    O['ab'][v,A,v,C] += scale * +1.00000000 * np.einsum('UV,aIbV,WUJI->aWbJ', eta1['b'], h['ab'][v,C,v,A], t['bb'][pA,pA,hC,hC], optimize=True)
    O['ab'][v,A,v,C] += scale * +1.00000000 * np.einsum('UV,aUbA,WAIV->aWbI', gamma1['b'], h['ab'][v,A,v,V], t['bb'][pA,pV,hC,hA], optimize=True)
    O['ab'][v,A,v,C] += scale * +1.00000000 * np.einsum('aIbA,UAJI->aUbJ', h['ab'][v,C,v,V], t['bb'][pA,pV,hC,hC], optimize=True)
    O['ab'][a,V,v,C] += scale * +1.00000000 * np.einsum('UV,WX,uUaX,WAIV->uAaI', eta1['b'], gamma1['b'], h['ab'][a,A,v,A], t['bb'][pA,pV,hC,hA], optimize=True)
    O['ab'][a,V,v,C] += scale * -1.00000000 * np.einsum('UV,WX,uWaV,UAIX->uAaI', eta1['b'], gamma1['b'], h['ab'][a,A,v,A], t['bb'][pA,pV,hC,hA], optimize=True)
    O['ab'][a,V,v,C] += scale * -1.00000000 * np.einsum('UV,uIaV,UAJI->uAaJ', eta1['b'], h['ab'][a,C,v,A], t['bb'][pA,pV,hC,hC], optimize=True)
    O['ab'][a,V,v,C] += scale * +1.00000000 * np.einsum('UV,uUaA,BAIV->uBaI', gamma1['b'], h['ab'][a,A,v,V], t['bb'][pV,pV,hC,hA], optimize=True)
    O['ab'][a,V,v,C] += scale * +1.00000000 * np.einsum('uIaA,BAJI->uBaJ', h['ab'][a,C,v,V], t['bb'][pV,pV,hC,hC], optimize=True)
    O['ab'][c,V,v,C] += scale * +1.00000000 * np.einsum('UV,WX,iUaX,WAIV->iAaI', eta1['b'], gamma1['b'], h['ab'][c,A,v,A], t['bb'][pA,pV,hC,hA], optimize=True)
    O['ab'][c,V,v,C] += scale * -1.00000000 * np.einsum('UV,WX,iWaV,UAIX->iAaI', eta1['b'], gamma1['b'], h['ab'][c,A,v,A], t['bb'][pA,pV,hC,hA], optimize=True)
    O['ab'][c,V,v,C] += scale * -1.00000000 * np.einsum('UV,iIaV,UAJI->iAaJ', eta1['b'], h['ab'][c,C,v,A], t['bb'][pA,pV,hC,hC], optimize=True)
    O['ab'][c,V,v,C] += scale * +1.00000000 * np.einsum('UV,iUaA,BAIV->iBaI', gamma1['b'], h['ab'][c,A,v,V], t['bb'][pV,pV,hC,hA], optimize=True)
    O['ab'][c,V,v,C] += scale * +1.00000000 * np.einsum('iIaA,BAJI->iBaJ', h['ab'][c,C,v,V], t['bb'][pV,pV,hC,hC], optimize=True)
    O['ab'][v,V,v,C] += scale * +1.00000000 * np.einsum('UV,WX,aUbX,WAIV->aAbI', eta1['b'], gamma1['b'], h['ab'][v,A,v,A], t['bb'][pA,pV,hC,hA], optimize=True)
    O['ab'][v,V,v,C] += scale * -1.00000000 * np.einsum('UV,WX,aWbV,UAIX->aAbI', eta1['b'], gamma1['b'], h['ab'][v,A,v,A], t['bb'][pA,pV,hC,hA], optimize=True)
    O['ab'][v,V,v,C] += scale * -1.00000000 * np.einsum('UV,aIbV,UAJI->aAbJ', eta1['b'], h['ab'][v,C,v,A], t['bb'][pA,pV,hC,hC], optimize=True)
    O['ab'][v,V,v,C] += scale * +1.00000000 * np.einsum('UV,aUbA,BAIV->aBbI', gamma1['b'], h['ab'][v,A,v,V], t['bb'][pV,pV,hC,hA], optimize=True)
    O['ab'][v,V,v,C] += scale * +1.00000000 * np.einsum('aIbA,BAJI->aBbJ', h['ab'][v,C,v,V], t['bb'][pV,pV,hC,hC], optimize=True)

    t1 = time.time()
    if verbose: print("h2b_t2c_c2b took {:.4f} seconds to run.".format(t1-t0))

    return O
def h2c_t2b_c2b(O, h, t, gamma1, eta1, lambdas, orbspace, verbose=False, scale=1.0):
    # 180 lines
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

    
    
    O['ab'][a,A,a,A] += scale * -1.00000000 * np.einsum('UV,IWXV,uUvI->uWvX', eta1['b'], h['bb'][C,A,A,A], t['ab'][pa,pA,ha,hC], optimize=True)
    O['ab'][a,A,a,A] += scale * +1.00000000 * np.einsum('UV,WUXA,uAvV->uWvX', gamma1['b'], h['bb'][A,A,A,V], t['ab'][pa,pV,ha,hA], optimize=True)
    O['ab'][a,A,a,A] += scale * -1.00000000 * np.einsum('IUVA,uAvI->uUvV', h['bb'][C,A,A,V], t['ab'][pa,pV,ha,hC], optimize=True)
    O['ab'][v,A,a,A] += scale * -1.00000000 * np.einsum('UV,WX,YUZX,aWuV->aYuZ', eta1['b'], gamma1['b'], h['bb'][A,A,A,A], t['ab'][pv,pA,ha,hA], optimize=True)
    O['ab'][v,A,a,A] += scale * +1.00000000 * np.einsum('UV,WX,YWZV,aUuX->aYuZ', eta1['b'], gamma1['b'], h['bb'][A,A,A,A], t['ab'][pv,pA,ha,hA], optimize=True)
    O['ab'][v,A,a,A] += scale * -1.00000000 * np.einsum('UV,IWXV,aUuI->aWuX', eta1['b'], h['bb'][C,A,A,A], t['ab'][pv,pA,ha,hC], optimize=True)
    O['ab'][v,A,a,A] += scale * +1.00000000 * np.einsum('UV,WUXA,aAuV->aWuX', gamma1['b'], h['bb'][A,A,A,V], t['ab'][pv,pV,ha,hA], optimize=True)
    O['ab'][v,A,a,A] += scale * -1.00000000 * np.einsum('IUVA,aAuI->aUuV', h['bb'][C,A,A,V], t['ab'][pv,pV,ha,hC], optimize=True)
    
    
    O['ab'][a,C,a,A] += scale * +1.00000000 * np.einsum('UV,IJWV,uUvJ->uIvW', eta1['b'], h['bb'][C,C,A,A], t['ab'][pa,pA,ha,hC], optimize=True)
    O['ab'][a,C,a,A] += scale * +1.00000000 * np.einsum('UV,IUWA,uAvV->uIvW', gamma1['b'], h['bb'][C,A,A,V], t['ab'][pa,pV,ha,hA], optimize=True)
    O['ab'][a,C,a,A] += scale * +1.00000000 * np.einsum('IJUA,uAvJ->uIvU', h['bb'][C,C,A,V], t['ab'][pa,pV,ha,hC], optimize=True)
    O['ab'][v,C,a,A] += scale * -1.00000000 * np.einsum('UV,WX,IUYX,aWuV->aIuY', eta1['b'], gamma1['b'], h['bb'][C,A,A,A], t['ab'][pv,pA,ha,hA], optimize=True)
    O['ab'][v,C,a,A] += scale * +1.00000000 * np.einsum('UV,WX,IWYV,aUuX->aIuY', eta1['b'], gamma1['b'], h['bb'][C,A,A,A], t['ab'][pv,pA,ha,hA], optimize=True)
    O['ab'][v,C,a,A] += scale * +1.00000000 * np.einsum('UV,IJWV,aUuJ->aIuW', eta1['b'], h['bb'][C,C,A,A], t['ab'][pv,pA,ha,hC], optimize=True)
    O['ab'][v,C,a,A] += scale * +1.00000000 * np.einsum('UV,IUWA,aAuV->aIuW', gamma1['b'], h['bb'][C,A,A,V], t['ab'][pv,pV,ha,hA], optimize=True)
    O['ab'][v,C,a,A] += scale * +1.00000000 * np.einsum('IJUA,aAuJ->aIuU', h['bb'][C,C,A,V], t['ab'][pv,pV,ha,hC], optimize=True)
    
    
    O['ab'][a,V,a,A] += scale * -1.00000000 * np.einsum('UV,IAWV,uUvI->uAvW', eta1['b'], h['bb'][C,V,A,A], t['ab'][pa,pA,ha,hC], optimize=True)
    O['ab'][a,V,a,A] += scale * -1.00000000 * np.einsum('UV,UAWB,uBvV->uAvW', gamma1['b'], h['bb'][A,V,A,V], t['ab'][pa,pV,ha,hA], optimize=True)
    O['ab'][a,V,a,A] += scale * -1.00000000 * np.einsum('IAUB,uBvI->uAvU', h['bb'][C,V,A,V], t['ab'][pa,pV,ha,hC], optimize=True)
    O['ab'][v,V,a,A] += scale * +1.00000000 * np.einsum('UV,WX,UAYX,aWuV->aAuY', eta1['b'], gamma1['b'], h['bb'][A,V,A,A], t['ab'][pv,pA,ha,hA], optimize=True)
    O['ab'][v,V,a,A] += scale * -1.00000000 * np.einsum('UV,WX,WAYV,aUuX->aAuY', eta1['b'], gamma1['b'], h['bb'][A,V,A,A], t['ab'][pv,pA,ha,hA], optimize=True)
    O['ab'][v,V,a,A] += scale * -1.00000000 * np.einsum('UV,IAWV,aUuI->aAuW', eta1['b'], h['bb'][C,V,A,A], t['ab'][pv,pA,ha,hC], optimize=True)
    O['ab'][v,V,a,A] += scale * -1.00000000 * np.einsum('UV,UAWB,aBuV->aAuW', gamma1['b'], h['bb'][A,V,A,V], t['ab'][pv,pV,ha,hA], optimize=True)
    O['ab'][v,V,a,A] += scale * -1.00000000 * np.einsum('IAUB,aBuI->aAuU', h['bb'][C,V,A,V], t['ab'][pv,pV,ha,hC], optimize=True)
    
    
    O['ab'][a,A,a,C] += scale * -1.00000000 * np.einsum('UV,IWJV,uUvI->uWvJ', eta1['b'], h['bb'][C,A,C,A], t['ab'][pa,pA,ha,hC], optimize=True)
    O['ab'][a,A,a,C] += scale * +1.00000000 * np.einsum('UV,WUIA,uAvV->uWvI', gamma1['b'], h['bb'][A,A,C,V], t['ab'][pa,pV,ha,hA], optimize=True)
    O['ab'][a,A,a,C] += scale * -1.00000000 * np.einsum('IUJA,uAvI->uUvJ', h['bb'][C,A,C,V], t['ab'][pa,pV,ha,hC], optimize=True)
    O['ab'][v,A,a,C] += scale * -1.00000000 * np.einsum('UV,WX,YUIX,aWuV->aYuI', eta1['b'], gamma1['b'], h['bb'][A,A,C,A], t['ab'][pv,pA,ha,hA], optimize=True)
    O['ab'][v,A,a,C] += scale * +1.00000000 * np.einsum('UV,WX,YWIV,aUuX->aYuI', eta1['b'], gamma1['b'], h['bb'][A,A,C,A], t['ab'][pv,pA,ha,hA], optimize=True)
    O['ab'][v,A,a,C] += scale * -1.00000000 * np.einsum('UV,IWJV,aUuI->aWuJ', eta1['b'], h['bb'][C,A,C,A], t['ab'][pv,pA,ha,hC], optimize=True)
    O['ab'][v,A,a,C] += scale * +1.00000000 * np.einsum('UV,WUIA,aAuV->aWuI', gamma1['b'], h['bb'][A,A,C,V], t['ab'][pv,pV,ha,hA], optimize=True)
    O['ab'][v,A,a,C] += scale * -1.00000000 * np.einsum('IUJA,aAuI->aUuJ', h['bb'][C,A,C,V], t['ab'][pv,pV,ha,hC], optimize=True)
    
    
    O['ab'][a,C,a,C] += scale * +1.00000000 * np.einsum('UV,IJKV,uUvJ->uIvK', eta1['b'], h['bb'][C,C,C,A], t['ab'][pa,pA,ha,hC], optimize=True)
    O['ab'][a,C,a,C] += scale * +1.00000000 * np.einsum('UV,IUJA,uAvV->uIvJ', gamma1['b'], h['bb'][C,A,C,V], t['ab'][pa,pV,ha,hA], optimize=True)
    O['ab'][a,C,a,C] += scale * +1.00000000 * np.einsum('IJKA,uAvJ->uIvK', h['bb'][C,C,C,V], t['ab'][pa,pV,ha,hC], optimize=True)
    O['ab'][v,C,a,C] += scale * -1.00000000 * np.einsum('UV,WX,IUJX,aWuV->aIuJ', eta1['b'], gamma1['b'], h['bb'][C,A,C,A], t['ab'][pv,pA,ha,hA], optimize=True)
    O['ab'][v,C,a,C] += scale * +1.00000000 * np.einsum('UV,WX,IWJV,aUuX->aIuJ', eta1['b'], gamma1['b'], h['bb'][C,A,C,A], t['ab'][pv,pA,ha,hA], optimize=True)
    O['ab'][v,C,a,C] += scale * +1.00000000 * np.einsum('UV,IJKV,aUuJ->aIuK', eta1['b'], h['bb'][C,C,C,A], t['ab'][pv,pA,ha,hC], optimize=True)
    O['ab'][v,C,a,C] += scale * +1.00000000 * np.einsum('UV,IUJA,aAuV->aIuJ', gamma1['b'], h['bb'][C,A,C,V], t['ab'][pv,pV,ha,hA], optimize=True)
    O['ab'][v,C,a,C] += scale * +1.00000000 * np.einsum('IJKA,aAuJ->aIuK', h['bb'][C,C,C,V], t['ab'][pv,pV,ha,hC], optimize=True)
    
    
    O['ab'][a,V,a,C] += scale * -1.00000000 * np.einsum('UV,IAJV,uUvI->uAvJ', eta1['b'], h['bb'][C,V,C,A], t['ab'][pa,pA,ha,hC], optimize=True)
    O['ab'][a,V,a,C] += scale * -1.00000000 * np.einsum('UV,UAIB,uBvV->uAvI', gamma1['b'], h['bb'][A,V,C,V], t['ab'][pa,pV,ha,hA], optimize=True)
    O['ab'][a,V,a,C] += scale * -1.00000000 * np.einsum('IAJB,uBvI->uAvJ', h['bb'][C,V,C,V], t['ab'][pa,pV,ha,hC], optimize=True)
    O['ab'][v,V,a,C] += scale * +1.00000000 * np.einsum('UV,WX,UAIX,aWuV->aAuI', eta1['b'], gamma1['b'], h['bb'][A,V,C,A], t['ab'][pv,pA,ha,hA], optimize=True)
    O['ab'][v,V,a,C] += scale * -1.00000000 * np.einsum('UV,WX,WAIV,aUuX->aAuI', eta1['b'], gamma1['b'], h['bb'][A,V,C,A], t['ab'][pv,pA,ha,hA], optimize=True)
    O['ab'][v,V,a,C] += scale * -1.00000000 * np.einsum('UV,IAJV,aUuI->aAuJ', eta1['b'], h['bb'][C,V,C,A], t['ab'][pv,pA,ha,hC], optimize=True)
    O['ab'][v,V,a,C] += scale * -1.00000000 * np.einsum('UV,UAIB,aBuV->aAuI', gamma1['b'], h['bb'][A,V,C,V], t['ab'][pv,pV,ha,hA], optimize=True)
    O['ab'][v,V,a,C] += scale * -1.00000000 * np.einsum('IAJB,aBuI->aAuJ', h['bb'][C,V,C,V], t['ab'][pv,pV,ha,hC], optimize=True)
    
    
    O['ab'][a,A,a,V] += scale * +1.00000000 * np.einsum('UV,IWVA,uUvI->uWvA', eta1['b'], h['bb'][C,A,A,V], t['ab'][pa,pA,ha,hC], optimize=True)
    O['ab'][a,A,a,V] += scale * +1.00000000 * np.einsum('UV,WUAB,uBvV->uWvA', gamma1['b'], h['bb'][A,A,V,V], t['ab'][pa,pV,ha,hA], optimize=True)
    O['ab'][a,A,a,V] += scale * -1.00000000 * np.einsum('IUAB,uBvI->uUvA', h['bb'][C,A,V,V], t['ab'][pa,pV,ha,hC], optimize=True)
    O['ab'][v,A,a,V] += scale * +1.00000000 * np.einsum('UV,WX,YUXA,aWuV->aYuA', eta1['b'], gamma1['b'], h['bb'][A,A,A,V], t['ab'][pv,pA,ha,hA], optimize=True)
    O['ab'][v,A,a,V] += scale * -1.00000000 * np.einsum('UV,WX,YWVA,aUuX->aYuA', eta1['b'], gamma1['b'], h['bb'][A,A,A,V], t['ab'][pv,pA,ha,hA], optimize=True)
    O['ab'][v,A,a,V] += scale * +1.00000000 * np.einsum('UV,IWVA,aUuI->aWuA', eta1['b'], h['bb'][C,A,A,V], t['ab'][pv,pA,ha,hC], optimize=True)
    O['ab'][v,A,a,V] += scale * +1.00000000 * np.einsum('UV,WUAB,aBuV->aWuA', gamma1['b'], h['bb'][A,A,V,V], t['ab'][pv,pV,ha,hA], optimize=True)
    O['ab'][v,A,a,V] += scale * -1.00000000 * np.einsum('IUAB,aBuI->aUuA', h['bb'][C,A,V,V], t['ab'][pv,pV,ha,hC], optimize=True)
    
    
    O['ab'][a,C,a,V] += scale * -1.00000000 * np.einsum('UV,IJVA,uUvJ->uIvA', eta1['b'], h['bb'][C,C,A,V], t['ab'][pa,pA,ha,hC], optimize=True)
    O['ab'][a,C,a,V] += scale * +1.00000000 * np.einsum('UV,IUAB,uBvV->uIvA', gamma1['b'], h['bb'][C,A,V,V], t['ab'][pa,pV,ha,hA], optimize=True)
    O['ab'][a,C,a,V] += scale * +1.00000000 * np.einsum('IJAB,uBvJ->uIvA', h['bb'][C,C,V,V], t['ab'][pa,pV,ha,hC], optimize=True)
    O['ab'][v,C,a,V] += scale * +1.00000000 * np.einsum('UV,WX,IUXA,aWuV->aIuA', eta1['b'], gamma1['b'], h['bb'][C,A,A,V], t['ab'][pv,pA,ha,hA], optimize=True)
    O['ab'][v,C,a,V] += scale * -1.00000000 * np.einsum('UV,WX,IWVA,aUuX->aIuA', eta1['b'], gamma1['b'], h['bb'][C,A,A,V], t['ab'][pv,pA,ha,hA], optimize=True)
    O['ab'][v,C,a,V] += scale * -1.00000000 * np.einsum('UV,IJVA,aUuJ->aIuA', eta1['b'], h['bb'][C,C,A,V], t['ab'][pv,pA,ha,hC], optimize=True)
    O['ab'][v,C,a,V] += scale * +1.00000000 * np.einsum('UV,IUAB,aBuV->aIuA', gamma1['b'], h['bb'][C,A,V,V], t['ab'][pv,pV,ha,hA], optimize=True)
    O['ab'][v,C,a,V] += scale * +1.00000000 * np.einsum('IJAB,aBuJ->aIuA', h['bb'][C,C,V,V], t['ab'][pv,pV,ha,hC], optimize=True)
    
    
    O['ab'][a,V,a,V] += scale * +1.00000000 * np.einsum('UV,IAVB,uUvI->uAvB', eta1['b'], h['bb'][C,V,A,V], t['ab'][pa,pA,ha,hC], optimize=True)
    O['ab'][a,V,a,V] += scale * -1.00000000 * np.einsum('UV,UABC,uCvV->uAvB', gamma1['b'], h['bb'][A,V,V,V], t['ab'][pa,pV,ha,hA], optimize=True)
    O['ab'][a,V,a,V] += scale * -1.00000000 * np.einsum('IABC,uCvI->uAvB', h['bb'][C,V,V,V], t['ab'][pa,pV,ha,hC], optimize=True)
    O['ab'][v,V,a,V] += scale * -1.00000000 * np.einsum('UV,WX,UAXB,aWuV->aAuB', eta1['b'], gamma1['b'], h['bb'][A,V,A,V], t['ab'][pv,pA,ha,hA], optimize=True)
    O['ab'][v,V,a,V] += scale * +1.00000000 * np.einsum('UV,WX,WAVB,aUuX->aAuB', eta1['b'], gamma1['b'], h['bb'][A,V,A,V], t['ab'][pv,pA,ha,hA], optimize=True)
    O['ab'][v,V,a,V] += scale * +1.00000000 * np.einsum('UV,IAVB,aUuI->aAuB', eta1['b'], h['bb'][C,V,A,V], t['ab'][pv,pA,ha,hC], optimize=True)
    O['ab'][v,V,a,V] += scale * -1.00000000 * np.einsum('UV,UABC,aCuV->aAuB', gamma1['b'], h['bb'][A,V,V,V], t['ab'][pv,pV,ha,hA], optimize=True)
    O['ab'][v,V,a,V] += scale * -1.00000000 * np.einsum('IABC,aCuI->aAuB', h['bb'][C,V,V,V], t['ab'][pv,pV,ha,hC], optimize=True)
    O['ab'][a,A,c,A] += scale * -1.00000000 * np.einsum('UV,WX,YUZX,uWiV->uYiZ', eta1['b'], gamma1['b'], h['bb'][A,A,A,A], t['ab'][pa,pA,hc,hA], optimize=True)
    O['ab'][a,A,c,A] += scale * +1.00000000 * np.einsum('UV,WX,YWZV,uUiX->uYiZ', eta1['b'], gamma1['b'], h['bb'][A,A,A,A], t['ab'][pa,pA,hc,hA], optimize=True)
    O['ab'][a,A,c,A] += scale * -1.00000000 * np.einsum('UV,IWXV,uUiI->uWiX', eta1['b'], h['bb'][C,A,A,A], t['ab'][pa,pA,hc,hC], optimize=True)
    O['ab'][a,A,c,A] += scale * +1.00000000 * np.einsum('UV,WUXA,uAiV->uWiX', gamma1['b'], h['bb'][A,A,A,V], t['ab'][pa,pV,hc,hA], optimize=True)
    O['ab'][a,A,c,A] += scale * -1.00000000 * np.einsum('IUVA,uAiI->uUiV', h['bb'][C,A,A,V], t['ab'][pa,pV,hc,hC], optimize=True)
    O['ab'][v,A,c,A] += scale * -1.00000000 * np.einsum('UV,WX,YUZX,aWiV->aYiZ', eta1['b'], gamma1['b'], h['bb'][A,A,A,A], t['ab'][pv,pA,hc,hA], optimize=True)
    O['ab'][v,A,c,A] += scale * +1.00000000 * np.einsum('UV,WX,YWZV,aUiX->aYiZ', eta1['b'], gamma1['b'], h['bb'][A,A,A,A], t['ab'][pv,pA,hc,hA], optimize=True)
    O['ab'][v,A,c,A] += scale * -1.00000000 * np.einsum('UV,IWXV,aUiI->aWiX', eta1['b'], h['bb'][C,A,A,A], t['ab'][pv,pA,hc,hC], optimize=True)
    O['ab'][v,A,c,A] += scale * +1.00000000 * np.einsum('UV,WUXA,aAiV->aWiX', gamma1['b'], h['bb'][A,A,A,V], t['ab'][pv,pV,hc,hA], optimize=True)
    O['ab'][v,A,c,A] += scale * -1.00000000 * np.einsum('IUVA,aAiI->aUiV', h['bb'][C,A,A,V], t['ab'][pv,pV,hc,hC], optimize=True)
    O['ab'][a,C,c,A] += scale * -1.00000000 * np.einsum('UV,WX,IUYX,uWiV->uIiY', eta1['b'], gamma1['b'], h['bb'][C,A,A,A], t['ab'][pa,pA,hc,hA], optimize=True)
    O['ab'][a,C,c,A] += scale * +1.00000000 * np.einsum('UV,WX,IWYV,uUiX->uIiY', eta1['b'], gamma1['b'], h['bb'][C,A,A,A], t['ab'][pa,pA,hc,hA], optimize=True)
    O['ab'][a,C,c,A] += scale * +1.00000000 * np.einsum('UV,IJWV,uUiJ->uIiW', eta1['b'], h['bb'][C,C,A,A], t['ab'][pa,pA,hc,hC], optimize=True)
    O['ab'][a,C,c,A] += scale * +1.00000000 * np.einsum('UV,IUWA,uAiV->uIiW', gamma1['b'], h['bb'][C,A,A,V], t['ab'][pa,pV,hc,hA], optimize=True)
    O['ab'][a,C,c,A] += scale * +1.00000000 * np.einsum('IJUA,uAiJ->uIiU', h['bb'][C,C,A,V], t['ab'][pa,pV,hc,hC], optimize=True)
    O['ab'][v,C,c,A] += scale * -1.00000000 * np.einsum('UV,WX,IUYX,aWiV->aIiY', eta1['b'], gamma1['b'], h['bb'][C,A,A,A], t['ab'][pv,pA,hc,hA], optimize=True)
    O['ab'][v,C,c,A] += scale * +1.00000000 * np.einsum('UV,WX,IWYV,aUiX->aIiY', eta1['b'], gamma1['b'], h['bb'][C,A,A,A], t['ab'][pv,pA,hc,hA], optimize=True)
    O['ab'][v,C,c,A] += scale * +1.00000000 * np.einsum('UV,IJWV,aUiJ->aIiW', eta1['b'], h['bb'][C,C,A,A], t['ab'][pv,pA,hc,hC], optimize=True)
    O['ab'][v,C,c,A] += scale * +1.00000000 * np.einsum('UV,IUWA,aAiV->aIiW', gamma1['b'], h['bb'][C,A,A,V], t['ab'][pv,pV,hc,hA], optimize=True)
    O['ab'][v,C,c,A] += scale * +1.00000000 * np.einsum('IJUA,aAiJ->aIiU', h['bb'][C,C,A,V], t['ab'][pv,pV,hc,hC], optimize=True)
    O['ab'][a,V,c,A] += scale * +1.00000000 * np.einsum('UV,WX,UAYX,uWiV->uAiY', eta1['b'], gamma1['b'], h['bb'][A,V,A,A], t['ab'][pa,pA,hc,hA], optimize=True)
    O['ab'][a,V,c,A] += scale * -1.00000000 * np.einsum('UV,WX,WAYV,uUiX->uAiY', eta1['b'], gamma1['b'], h['bb'][A,V,A,A], t['ab'][pa,pA,hc,hA], optimize=True)
    O['ab'][a,V,c,A] += scale * -1.00000000 * np.einsum('UV,IAWV,uUiI->uAiW', eta1['b'], h['bb'][C,V,A,A], t['ab'][pa,pA,hc,hC], optimize=True)
    O['ab'][a,V,c,A] += scale * -1.00000000 * np.einsum('UV,UAWB,uBiV->uAiW', gamma1['b'], h['bb'][A,V,A,V], t['ab'][pa,pV,hc,hA], optimize=True)
    O['ab'][a,V,c,A] += scale * -1.00000000 * np.einsum('IAUB,uBiI->uAiU', h['bb'][C,V,A,V], t['ab'][pa,pV,hc,hC], optimize=True)
    O['ab'][v,V,c,A] += scale * +1.00000000 * np.einsum('UV,WX,UAYX,aWiV->aAiY', eta1['b'], gamma1['b'], h['bb'][A,V,A,A], t['ab'][pv,pA,hc,hA], optimize=True)
    O['ab'][v,V,c,A] += scale * -1.00000000 * np.einsum('UV,WX,WAYV,aUiX->aAiY', eta1['b'], gamma1['b'], h['bb'][A,V,A,A], t['ab'][pv,pA,hc,hA], optimize=True)
    O['ab'][v,V,c,A] += scale * -1.00000000 * np.einsum('UV,IAWV,aUiI->aAiW', eta1['b'], h['bb'][C,V,A,A], t['ab'][pv,pA,hc,hC], optimize=True)
    O['ab'][v,V,c,A] += scale * -1.00000000 * np.einsum('UV,UAWB,aBiV->aAiW', gamma1['b'], h['bb'][A,V,A,V], t['ab'][pv,pV,hc,hA], optimize=True)
    O['ab'][v,V,c,A] += scale * -1.00000000 * np.einsum('IAUB,aBiI->aAiU', h['bb'][C,V,A,V], t['ab'][pv,pV,hc,hC], optimize=True)
    O['ab'][a,A,c,C] += scale * -1.00000000 * np.einsum('UV,WX,YUIX,uWiV->uYiI', eta1['b'], gamma1['b'], h['bb'][A,A,C,A], t['ab'][pa,pA,hc,hA], optimize=True)
    O['ab'][a,A,c,C] += scale * +1.00000000 * np.einsum('UV,WX,YWIV,uUiX->uYiI', eta1['b'], gamma1['b'], h['bb'][A,A,C,A], t['ab'][pa,pA,hc,hA], optimize=True)
    O['ab'][a,A,c,C] += scale * -1.00000000 * np.einsum('UV,IWJV,uUiI->uWiJ', eta1['b'], h['bb'][C,A,C,A], t['ab'][pa,pA,hc,hC], optimize=True)
    O['ab'][a,A,c,C] += scale * +1.00000000 * np.einsum('UV,WUIA,uAiV->uWiI', gamma1['b'], h['bb'][A,A,C,V], t['ab'][pa,pV,hc,hA], optimize=True)
    O['ab'][a,A,c,C] += scale * -1.00000000 * np.einsum('IUJA,uAiI->uUiJ', h['bb'][C,A,C,V], t['ab'][pa,pV,hc,hC], optimize=True)
    O['ab'][v,A,c,C] += scale * -1.00000000 * np.einsum('UV,WX,YUIX,aWiV->aYiI', eta1['b'], gamma1['b'], h['bb'][A,A,C,A], t['ab'][pv,pA,hc,hA], optimize=True)
    O['ab'][v,A,c,C] += scale * +1.00000000 * np.einsum('UV,WX,YWIV,aUiX->aYiI', eta1['b'], gamma1['b'], h['bb'][A,A,C,A], t['ab'][pv,pA,hc,hA], optimize=True)
    O['ab'][v,A,c,C] += scale * -1.00000000 * np.einsum('UV,IWJV,aUiI->aWiJ', eta1['b'], h['bb'][C,A,C,A], t['ab'][pv,pA,hc,hC], optimize=True)
    O['ab'][v,A,c,C] += scale * +1.00000000 * np.einsum('UV,WUIA,aAiV->aWiI', gamma1['b'], h['bb'][A,A,C,V], t['ab'][pv,pV,hc,hA], optimize=True)
    O['ab'][v,A,c,C] += scale * -1.00000000 * np.einsum('IUJA,aAiI->aUiJ', h['bb'][C,A,C,V], t['ab'][pv,pV,hc,hC], optimize=True)
    O['ab'][a,C,c,C] += scale * -1.00000000 * np.einsum('UV,WX,IUJX,uWiV->uIiJ', eta1['b'], gamma1['b'], h['bb'][C,A,C,A], t['ab'][pa,pA,hc,hA], optimize=True)
    O['ab'][a,C,c,C] += scale * +1.00000000 * np.einsum('UV,WX,IWJV,uUiX->uIiJ', eta1['b'], gamma1['b'], h['bb'][C,A,C,A], t['ab'][pa,pA,hc,hA], optimize=True)
    O['ab'][a,C,c,C] += scale * +1.00000000 * np.einsum('UV,IJKV,uUiJ->uIiK', eta1['b'], h['bb'][C,C,C,A], t['ab'][pa,pA,hc,hC], optimize=True)
    O['ab'][a,C,c,C] += scale * +1.00000000 * np.einsum('UV,IUJA,uAiV->uIiJ', gamma1['b'], h['bb'][C,A,C,V], t['ab'][pa,pV,hc,hA], optimize=True)
    O['ab'][a,C,c,C] += scale * +1.00000000 * np.einsum('IJKA,uAiJ->uIiK', h['bb'][C,C,C,V], t['ab'][pa,pV,hc,hC], optimize=True)
    O['ab'][v,C,c,C] += scale * -1.00000000 * np.einsum('UV,WX,IUJX,aWiV->aIiJ', eta1['b'], gamma1['b'], h['bb'][C,A,C,A], t['ab'][pv,pA,hc,hA], optimize=True)
    O['ab'][v,C,c,C] += scale * +1.00000000 * np.einsum('UV,WX,IWJV,aUiX->aIiJ', eta1['b'], gamma1['b'], h['bb'][C,A,C,A], t['ab'][pv,pA,hc,hA], optimize=True)
    O['ab'][v,C,c,C] += scale * +1.00000000 * np.einsum('UV,IJKV,aUiJ->aIiK', eta1['b'], h['bb'][C,C,C,A], t['ab'][pv,pA,hc,hC], optimize=True)
    O['ab'][v,C,c,C] += scale * +1.00000000 * np.einsum('UV,IUJA,aAiV->aIiJ', gamma1['b'], h['bb'][C,A,C,V], t['ab'][pv,pV,hc,hA], optimize=True)
    O['ab'][v,C,c,C] += scale * +1.00000000 * np.einsum('IJKA,aAiJ->aIiK', h['bb'][C,C,C,V], t['ab'][pv,pV,hc,hC], optimize=True)
    O['ab'][a,V,c,C] += scale * +1.00000000 * np.einsum('UV,WX,UAIX,uWiV->uAiI', eta1['b'], gamma1['b'], h['bb'][A,V,C,A], t['ab'][pa,pA,hc,hA], optimize=True)
    O['ab'][a,V,c,C] += scale * -1.00000000 * np.einsum('UV,WX,WAIV,uUiX->uAiI', eta1['b'], gamma1['b'], h['bb'][A,V,C,A], t['ab'][pa,pA,hc,hA], optimize=True)
    O['ab'][a,V,c,C] += scale * -1.00000000 * np.einsum('UV,IAJV,uUiI->uAiJ', eta1['b'], h['bb'][C,V,C,A], t['ab'][pa,pA,hc,hC], optimize=True)
    O['ab'][a,V,c,C] += scale * -1.00000000 * np.einsum('UV,UAIB,uBiV->uAiI', gamma1['b'], h['bb'][A,V,C,V], t['ab'][pa,pV,hc,hA], optimize=True)
    O['ab'][a,V,c,C] += scale * -1.00000000 * np.einsum('IAJB,uBiI->uAiJ', h['bb'][C,V,C,V], t['ab'][pa,pV,hc,hC], optimize=True)
    O['ab'][v,V,c,C] += scale * +1.00000000 * np.einsum('UV,WX,UAIX,aWiV->aAiI', eta1['b'], gamma1['b'], h['bb'][A,V,C,A], t['ab'][pv,pA,hc,hA], optimize=True)
    O['ab'][v,V,c,C] += scale * -1.00000000 * np.einsum('UV,WX,WAIV,aUiX->aAiI', eta1['b'], gamma1['b'], h['bb'][A,V,C,A], t['ab'][pv,pA,hc,hA], optimize=True)
    O['ab'][v,V,c,C] += scale * -1.00000000 * np.einsum('UV,IAJV,aUiI->aAiJ', eta1['b'], h['bb'][C,V,C,A], t['ab'][pv,pA,hc,hC], optimize=True)
    O['ab'][v,V,c,C] += scale * -1.00000000 * np.einsum('UV,UAIB,aBiV->aAiI', gamma1['b'], h['bb'][A,V,C,V], t['ab'][pv,pV,hc,hA], optimize=True)
    O['ab'][v,V,c,C] += scale * -1.00000000 * np.einsum('IAJB,aBiI->aAiJ', h['bb'][C,V,C,V], t['ab'][pv,pV,hc,hC], optimize=True)
    O['ab'][a,A,c,V] += scale * +1.00000000 * np.einsum('UV,WX,YUXA,uWiV->uYiA', eta1['b'], gamma1['b'], h['bb'][A,A,A,V], t['ab'][pa,pA,hc,hA], optimize=True)
    O['ab'][a,A,c,V] += scale * -1.00000000 * np.einsum('UV,WX,YWVA,uUiX->uYiA', eta1['b'], gamma1['b'], h['bb'][A,A,A,V], t['ab'][pa,pA,hc,hA], optimize=True)
    O['ab'][a,A,c,V] += scale * +1.00000000 * np.einsum('UV,IWVA,uUiI->uWiA', eta1['b'], h['bb'][C,A,A,V], t['ab'][pa,pA,hc,hC], optimize=True)
    O['ab'][a,A,c,V] += scale * +1.00000000 * np.einsum('UV,WUAB,uBiV->uWiA', gamma1['b'], h['bb'][A,A,V,V], t['ab'][pa,pV,hc,hA], optimize=True)
    O['ab'][a,A,c,V] += scale * -1.00000000 * np.einsum('IUAB,uBiI->uUiA', h['bb'][C,A,V,V], t['ab'][pa,pV,hc,hC], optimize=True)
    O['ab'][v,A,c,V] += scale * +1.00000000 * np.einsum('UV,WX,YUXA,aWiV->aYiA', eta1['b'], gamma1['b'], h['bb'][A,A,A,V], t['ab'][pv,pA,hc,hA], optimize=True)
    O['ab'][v,A,c,V] += scale * -1.00000000 * np.einsum('UV,WX,YWVA,aUiX->aYiA', eta1['b'], gamma1['b'], h['bb'][A,A,A,V], t['ab'][pv,pA,hc,hA], optimize=True)
    O['ab'][v,A,c,V] += scale * +1.00000000 * np.einsum('UV,IWVA,aUiI->aWiA', eta1['b'], h['bb'][C,A,A,V], t['ab'][pv,pA,hc,hC], optimize=True)
    O['ab'][v,A,c,V] += scale * +1.00000000 * np.einsum('UV,WUAB,aBiV->aWiA', gamma1['b'], h['bb'][A,A,V,V], t['ab'][pv,pV,hc,hA], optimize=True)
    O['ab'][v,A,c,V] += scale * -1.00000000 * np.einsum('IUAB,aBiI->aUiA', h['bb'][C,A,V,V], t['ab'][pv,pV,hc,hC], optimize=True)
    O['ab'][a,C,c,V] += scale * +1.00000000 * np.einsum('UV,WX,IUXA,uWiV->uIiA', eta1['b'], gamma1['b'], h['bb'][C,A,A,V], t['ab'][pa,pA,hc,hA], optimize=True)
    O['ab'][a,C,c,V] += scale * -1.00000000 * np.einsum('UV,WX,IWVA,uUiX->uIiA', eta1['b'], gamma1['b'], h['bb'][C,A,A,V], t['ab'][pa,pA,hc,hA], optimize=True)
    O['ab'][a,C,c,V] += scale * -1.00000000 * np.einsum('UV,IJVA,uUiJ->uIiA', eta1['b'], h['bb'][C,C,A,V], t['ab'][pa,pA,hc,hC], optimize=True)
    O['ab'][a,C,c,V] += scale * +1.00000000 * np.einsum('UV,IUAB,uBiV->uIiA', gamma1['b'], h['bb'][C,A,V,V], t['ab'][pa,pV,hc,hA], optimize=True)
    O['ab'][a,C,c,V] += scale * +1.00000000 * np.einsum('IJAB,uBiJ->uIiA', h['bb'][C,C,V,V], t['ab'][pa,pV,hc,hC], optimize=True)
    O['ab'][v,C,c,V] += scale * +1.00000000 * np.einsum('UV,WX,IUXA,aWiV->aIiA', eta1['b'], gamma1['b'], h['bb'][C,A,A,V], t['ab'][pv,pA,hc,hA], optimize=True)
    O['ab'][v,C,c,V] += scale * -1.00000000 * np.einsum('UV,WX,IWVA,aUiX->aIiA', eta1['b'], gamma1['b'], h['bb'][C,A,A,V], t['ab'][pv,pA,hc,hA], optimize=True)
    O['ab'][v,C,c,V] += scale * -1.00000000 * np.einsum('UV,IJVA,aUiJ->aIiA', eta1['b'], h['bb'][C,C,A,V], t['ab'][pv,pA,hc,hC], optimize=True)
    O['ab'][v,C,c,V] += scale * +1.00000000 * np.einsum('UV,IUAB,aBiV->aIiA', gamma1['b'], h['bb'][C,A,V,V], t['ab'][pv,pV,hc,hA], optimize=True)
    O['ab'][v,C,c,V] += scale * +1.00000000 * np.einsum('IJAB,aBiJ->aIiA', h['bb'][C,C,V,V], t['ab'][pv,pV,hc,hC], optimize=True)
    O['ab'][a,V,c,V] += scale * -1.00000000 * np.einsum('UV,WX,UAXB,uWiV->uAiB', eta1['b'], gamma1['b'], h['bb'][A,V,A,V], t['ab'][pa,pA,hc,hA], optimize=True)
    O['ab'][a,V,c,V] += scale * +1.00000000 * np.einsum('UV,WX,WAVB,uUiX->uAiB', eta1['b'], gamma1['b'], h['bb'][A,V,A,V], t['ab'][pa,pA,hc,hA], optimize=True)
    O['ab'][a,V,c,V] += scale * +1.00000000 * np.einsum('UV,IAVB,uUiI->uAiB', eta1['b'], h['bb'][C,V,A,V], t['ab'][pa,pA,hc,hC], optimize=True)
    O['ab'][a,V,c,V] += scale * -1.00000000 * np.einsum('UV,UABC,uCiV->uAiB', gamma1['b'], h['bb'][A,V,V,V], t['ab'][pa,pV,hc,hA], optimize=True)
    O['ab'][a,V,c,V] += scale * -1.00000000 * np.einsum('IABC,uCiI->uAiB', h['bb'][C,V,V,V], t['ab'][pa,pV,hc,hC], optimize=True)
    O['ab'][v,V,c,V] += scale * -1.00000000 * np.einsum('UV,WX,UAXB,aWiV->aAiB', eta1['b'], gamma1['b'], h['bb'][A,V,A,V], t['ab'][pv,pA,hc,hA], optimize=True)
    O['ab'][v,V,c,V] += scale * +1.00000000 * np.einsum('UV,WX,WAVB,aUiX->aAiB', eta1['b'], gamma1['b'], h['bb'][A,V,A,V], t['ab'][pv,pA,hc,hA], optimize=True)
    O['ab'][v,V,c,V] += scale * +1.00000000 * np.einsum('UV,IAVB,aUiI->aAiB', eta1['b'], h['bb'][C,V,A,V], t['ab'][pv,pA,hc,hC], optimize=True)
    O['ab'][v,V,c,V] += scale * -1.00000000 * np.einsum('UV,UABC,aCiV->aAiB', gamma1['b'], h['bb'][A,V,V,V], t['ab'][pv,pV,hc,hA], optimize=True)
    O['ab'][v,V,c,V] += scale * -1.00000000 * np.einsum('IABC,aCiI->aAiB', h['bb'][C,V,V,V], t['ab'][pv,pV,hc,hC], optimize=True)

    t1 = time.time()
    if verbose: print("h2c_t2b_c2b took {:.4f} seconds to run.".format(t1-t0))

    return O
def h1b_t2c_c2c(O, h, t, gamma1, eta1, lambdas, orbspace, verbose=False, scale=1.0):
    # 108 lines
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

    
    
    
    
    O['bb'][A,A,A,A] += scale * -0.50000000 * np.einsum('IU,VWIX->VWUX', h['b'][C,A], t['bb'][pA,pA,hC,hA], optimize=True)
    O['bb'][A,A,A,A] += scale * -0.50000000 * np.einsum('UA,VAWX->UVWX', h['b'][A,V], t['bb'][pA,pV,hA,hA], optimize=True)
    
    
    O['bb'][C,A,A,A] += scale * -0.50000000 * np.einsum('IA,UAVW->IUVW', h['b'][C,V], t['bb'][pA,pV,hA,hA], optimize=True)
    O['bb'][A,V,A,A] += scale * +0.50000000 * np.einsum('UV,WV,UAXY->WAXY', eta1['b'], h['b'][A,A], t['bb'][pA,pV,hA,hA], optimize=True)
    O['bb'][A,V,A,A] += scale * +1.00000000 * np.einsum('UV,UW,XAYV->XAWY', eta1['b'], h['b'][A,A], t['bb'][pA,pV,hA,hA], optimize=True)
    
    O['bb'][A,V,A,A] += scale * +0.50000000 * np.einsum('UV,WV,UAXY->WAXY', gamma1['b'], h['b'][A,A], t['bb'][pA,pV,hA,hA], optimize=True)
    O['bb'][A,V,A,A] += scale * +1.00000000 * np.einsum('UV,UW,XAYV->XAWY', gamma1['b'], h['b'][A,A], t['bb'][pA,pV,hA,hA], optimize=True)
    
    O['bb'][A,V,A,A] += scale * -1.00000000 * np.einsum('IU,VAIW->VAUW', h['b'][C,A], t['bb'][pA,pV,hC,hA], optimize=True)
    O['bb'][A,V,A,A] += scale * -0.50000000 * np.einsum('UA,BAVW->UBVW', h['b'][A,V], t['bb'][pV,pV,hA,hA], optimize=True)
    O['bb'][A,V,A,A] += scale * +0.50000000 * np.einsum('AB,UBVW->UAVW', h['b'][V,V], t['bb'][pA,pV,hA,hA], optimize=True)
    O['bb'][C,V,A,A] += scale * +0.50000000 * np.einsum('UV,IV,UAWX->IAWX', eta1['b'], h['b'][C,A], t['bb'][pA,pV,hA,hA], optimize=True)
    O['bb'][C,V,A,A] += scale * +0.50000000 * np.einsum('UV,IV,UAWX->IAWX', gamma1['b'], h['b'][C,A], t['bb'][pA,pV,hA,hA], optimize=True)
    O['bb'][C,V,A,A] += scale * -0.50000000 * np.einsum('IA,BAUV->IBUV', h['b'][C,V], t['bb'][pV,pV,hA,hA], optimize=True)
    O['bb'][V,V,A,A] += scale * +0.50000000 * np.einsum('UV,UW,ABXV->ABWX', eta1['b'], h['b'][A,A], t['bb'][pV,pV,hA,hA], optimize=True)
    O['bb'][V,V,A,A] += scale * +0.50000000 * np.einsum('UV,AV,UBWX->ABWX', eta1['b'], h['b'][V,A], t['bb'][pA,pV,hA,hA], optimize=True)
    O['bb'][V,V,A,A] += scale * +0.50000000 * np.einsum('UV,UW,ABXV->ABWX', gamma1['b'], h['b'][A,A], t['bb'][pV,pV,hA,hA], optimize=True)
    O['bb'][V,V,A,A] += scale * +0.50000000 * np.einsum('UV,AV,UBWX->ABWX', gamma1['b'], h['b'][V,A], t['bb'][pA,pV,hA,hA], optimize=True)
    O['bb'][V,V,A,A] += scale * -0.50000000 * np.einsum('IU,ABIV->ABUV', h['b'][C,A], t['bb'][pV,pV,hC,hA], optimize=True)
    O['bb'][V,V,A,A] += scale * -0.50000000 * np.einsum('AB,CBUV->ACUV', h['b'][V,V], t['bb'][pV,pV,hA,hA], optimize=True)
    
    
    O['bb'][A,A,A,V] += scale * +0.50000000 * np.einsum('IA,UVIW->UVWA', h['b'][C,V], t['bb'][pA,pA,hC,hA], optimize=True)
    O['bb'][A,V,A,V] += scale * -1.00000000 * np.einsum('UV,UA,WBXV->WBXA', eta1['b'], h['b'][A,V], t['bb'][pA,pV,hA,hA], optimize=True)
    O['bb'][A,V,A,V] += scale * -1.00000000 * np.einsum('UV,UA,WBXV->WBXA', gamma1['b'], h['b'][A,V], t['bb'][pA,pV,hA,hA], optimize=True)
    O['bb'][A,V,A,V] += scale * +1.00000000 * np.einsum('IA,UBIV->UBVA', h['b'][C,V], t['bb'][pA,pV,hC,hA], optimize=True)
    O['bb'][V,V,A,V] += scale * -0.50000000 * np.einsum('UV,UA,BCWV->BCWA', eta1['b'], h['b'][A,V], t['bb'][pV,pV,hA,hA], optimize=True)
    O['bb'][V,V,A,V] += scale * -0.50000000 * np.einsum('UV,UA,BCWV->BCWA', gamma1['b'], h['b'][A,V], t['bb'][pV,pV,hA,hA], optimize=True)
    O['bb'][V,V,A,V] += scale * +0.50000000 * np.einsum('IA,BCIU->BCUA', h['b'][C,V], t['bb'][pV,pV,hC,hA], optimize=True)
    O['bb'][A,A,C,A] += scale * -1.00000000 * np.einsum('UV,WV,XUIY->WXIY', eta1['b'], h['b'][A,A], t['bb'][pA,pA,hC,hA], optimize=True)
    
    O['bb'][A,A,C,A] += scale * -0.50000000 * np.einsum('UV,UW,XYIV->XYIW', eta1['b'], h['b'][A,A], t['bb'][pA,pA,hC,hA], optimize=True)
    O['bb'][A,A,C,A] += scale * -1.00000000 * np.einsum('UV,WV,XUIY->WXIY', gamma1['b'], h['b'][A,A], t['bb'][pA,pA,hC,hA], optimize=True)
    
    O['bb'][A,A,C,A] += scale * -0.50000000 * np.einsum('UV,UW,XYIV->XYIW', gamma1['b'], h['b'][A,A], t['bb'][pA,pA,hC,hA], optimize=True)
    O['bb'][A,A,C,A] += scale * -0.50000000 * np.einsum('IJ,UVIW->UVJW', h['b'][C,C], t['bb'][pA,pA,hC,hA], optimize=True)
    O['bb'][A,A,C,A] += scale * -0.50000000 * np.einsum('IU,VWJI->VWJU', h['b'][C,A], t['bb'][pA,pA,hC,hC], optimize=True)
    O['bb'][A,A,C,A] += scale * -1.00000000 * np.einsum('UA,VAIW->UVIW', h['b'][A,V], t['bb'][pA,pV,hC,hA], optimize=True)
    O['bb'][C,A,C,A] += scale * -1.00000000 * np.einsum('UV,IV,WUJX->IWJX', eta1['b'], h['b'][C,A], t['bb'][pA,pA,hC,hA], optimize=True)
    O['bb'][C,A,C,A] += scale * -1.00000000 * np.einsum('UV,IV,WUJX->IWJX', gamma1['b'], h['b'][C,A], t['bb'][pA,pA,hC,hA], optimize=True)
    O['bb'][C,A,C,A] += scale * -1.00000000 * np.einsum('IA,UAJV->IUJV', h['b'][C,V], t['bb'][pA,pV,hC,hA], optimize=True)
    O['bb'][A,V,C,A] += scale * +1.00000000 * np.einsum('UV,WV,UAIX->WAIX', eta1['b'], h['b'][A,A], t['bb'][pA,pV,hC,hA], optimize=True)
    O['bb'][A,V,C,A] += scale * +1.00000000 * np.einsum('UV,UI,WAXV->WAIX', eta1['b'], h['b'][A,C], t['bb'][pA,pV,hA,hA], optimize=True)
    O['bb'][A,V,C,A] += scale * -1.00000000 * np.einsum('UV,UW,XAIV->XAIW', eta1['b'], h['b'][A,A], t['bb'][pA,pV,hC,hA], optimize=True)
    O['bb'][A,V,C,A] += scale * +1.00000000 * np.einsum('UV,AV,WUIX->WAIX', eta1['b'], h['b'][V,A], t['bb'][pA,pA,hC,hA], optimize=True)
    O['bb'][A,V,C,A] += scale * +1.00000000 * np.einsum('UV,WV,UAIX->WAIX', gamma1['b'], h['b'][A,A], t['bb'][pA,pV,hC,hA], optimize=True)
    O['bb'][A,V,C,A] += scale * +1.00000000 * np.einsum('UV,UI,WAXV->WAIX', gamma1['b'], h['b'][A,C], t['bb'][pA,pV,hA,hA], optimize=True)
    O['bb'][A,V,C,A] += scale * -1.00000000 * np.einsum('UV,UW,XAIV->XAIW', gamma1['b'], h['b'][A,A], t['bb'][pA,pV,hC,hA], optimize=True)
    O['bb'][A,V,C,A] += scale * +1.00000000 * np.einsum('UV,AV,WUIX->WAIX', gamma1['b'], h['b'][V,A], t['bb'][pA,pA,hC,hA], optimize=True)
    O['bb'][A,V,C,A] += scale * -1.00000000 * np.einsum('IJ,UAIV->UAJV', h['b'][C,C], t['bb'][pA,pV,hC,hA], optimize=True)
    O['bb'][A,V,C,A] += scale * -1.00000000 * np.einsum('IU,VAJI->VAJU', h['b'][C,A], t['bb'][pA,pV,hC,hC], optimize=True)
    O['bb'][A,V,C,A] += scale * -1.00000000 * np.einsum('UA,BAIV->UBIV', h['b'][A,V], t['bb'][pV,pV,hC,hA], optimize=True)
    O['bb'][A,V,C,A] += scale * +1.00000000 * np.einsum('AB,UBIV->UAIV', h['b'][V,V], t['bb'][pA,pV,hC,hA], optimize=True)
    O['bb'][C,V,C,A] += scale * +1.00000000 * np.einsum('UV,IV,UAJW->IAJW', eta1['b'], h['b'][C,A], t['bb'][pA,pV,hC,hA], optimize=True)
    O['bb'][C,V,C,A] += scale * +1.00000000 * np.einsum('UV,IV,UAJW->IAJW', gamma1['b'], h['b'][C,A], t['bb'][pA,pV,hC,hA], optimize=True)
    O['bb'][C,V,C,A] += scale * -1.00000000 * np.einsum('IA,BAJU->IBJU', h['b'][C,V], t['bb'][pV,pV,hC,hA], optimize=True)
    O['bb'][V,V,C,A] += scale * +0.50000000 * np.einsum('UV,UI,ABWV->ABIW', eta1['b'], h['b'][A,C], t['bb'][pV,pV,hA,hA], optimize=True)
    O['bb'][V,V,C,A] += scale * -0.50000000 * np.einsum('UV,UW,ABIV->ABIW', eta1['b'], h['b'][A,A], t['bb'][pV,pV,hC,hA], optimize=True)
    O['bb'][V,V,C,A] += scale * +1.00000000 * np.einsum('UV,AV,UBIW->ABIW', eta1['b'], h['b'][V,A], t['bb'][pA,pV,hC,hA], optimize=True)
    O['bb'][V,V,C,A] += scale * +0.50000000 * np.einsum('UV,UI,ABWV->ABIW', gamma1['b'], h['b'][A,C], t['bb'][pV,pV,hA,hA], optimize=True)
    O['bb'][V,V,C,A] += scale * -0.50000000 * np.einsum('UV,UW,ABIV->ABIW', gamma1['b'], h['b'][A,A], t['bb'][pV,pV,hC,hA], optimize=True)
    O['bb'][V,V,C,A] += scale * +1.00000000 * np.einsum('UV,AV,UBIW->ABIW', gamma1['b'], h['b'][V,A], t['bb'][pA,pV,hC,hA], optimize=True)
    O['bb'][V,V,C,A] += scale * -0.50000000 * np.einsum('IJ,ABIU->ABJU', h['b'][C,C], t['bb'][pV,pV,hC,hA], optimize=True)
    O['bb'][V,V,C,A] += scale * -0.50000000 * np.einsum('IU,ABJI->ABJU', h['b'][C,A], t['bb'][pV,pV,hC,hC], optimize=True)
    O['bb'][V,V,C,A] += scale * -1.00000000 * np.einsum('AB,CBIU->ACIU', h['b'][V,V], t['bb'][pV,pV,hC,hA], optimize=True)
    O['bb'][A,A,C,C] += scale * -0.50000000 * np.einsum('UV,WV,XUIJ->WXIJ', eta1['b'], h['b'][A,A], t['bb'][pA,pA,hC,hC], optimize=True)
    O['bb'][A,A,C,C] += scale * +0.50000000 * np.einsum('UV,UI,WXJV->WXIJ', eta1['b'], h['b'][A,C], t['bb'][pA,pA,hC,hA], optimize=True)
    O['bb'][A,A,C,C] += scale * -0.50000000 * np.einsum('UV,WV,XUIJ->WXIJ', gamma1['b'], h['b'][A,A], t['bb'][pA,pA,hC,hC], optimize=True)
    O['bb'][A,A,C,C] += scale * +0.50000000 * np.einsum('UV,UI,WXJV->WXIJ', gamma1['b'], h['b'][A,C], t['bb'][pA,pA,hC,hA], optimize=True)
    O['bb'][A,A,C,C] += scale * +0.50000000 * np.einsum('IJ,UVKI->UVJK', h['b'][C,C], t['bb'][pA,pA,hC,hC], optimize=True)
    O['bb'][A,A,C,C] += scale * -0.50000000 * np.einsum('UA,VAIJ->UVIJ', h['b'][A,V], t['bb'][pA,pV,hC,hC], optimize=True)
    O['bb'][C,A,C,C] += scale * -0.50000000 * np.einsum('UV,IV,WUJK->IWJK', eta1['b'], h['b'][C,A], t['bb'][pA,pA,hC,hC], optimize=True)
    O['bb'][C,A,C,C] += scale * -0.50000000 * np.einsum('UV,IV,WUJK->IWJK', gamma1['b'], h['b'][C,A], t['bb'][pA,pA,hC,hC], optimize=True)
    O['bb'][C,A,C,C] += scale * -0.50000000 * np.einsum('IA,UAJK->IUJK', h['b'][C,V], t['bb'][pA,pV,hC,hC], optimize=True)
    O['bb'][A,V,C,C] += scale * +0.50000000 * np.einsum('UV,WV,UAIJ->WAIJ', eta1['b'], h['b'][A,A], t['bb'][pA,pV,hC,hC], optimize=True)
    O['bb'][A,V,C,C] += scale * +1.00000000 * np.einsum('UV,UI,WAJV->WAIJ', eta1['b'], h['b'][A,C], t['bb'][pA,pV,hC,hA], optimize=True)
    O['bb'][A,V,C,C] += scale * +0.50000000 * np.einsum('UV,AV,WUIJ->WAIJ', eta1['b'], h['b'][V,A], t['bb'][pA,pA,hC,hC], optimize=True)
    O['bb'][A,V,C,C] += scale * +0.50000000 * np.einsum('UV,WV,UAIJ->WAIJ', gamma1['b'], h['b'][A,A], t['bb'][pA,pV,hC,hC], optimize=True)
    O['bb'][A,V,C,C] += scale * +1.00000000 * np.einsum('UV,UI,WAJV->WAIJ', gamma1['b'], h['b'][A,C], t['bb'][pA,pV,hC,hA], optimize=True)
    O['bb'][A,V,C,C] += scale * +0.50000000 * np.einsum('UV,AV,WUIJ->WAIJ', gamma1['b'], h['b'][V,A], t['bb'][pA,pA,hC,hC], optimize=True)
    O['bb'][A,V,C,C] += scale * +1.00000000 * np.einsum('IJ,UAKI->UAJK', h['b'][C,C], t['bb'][pA,pV,hC,hC], optimize=True)
    O['bb'][A,V,C,C] += scale * -0.50000000 * np.einsum('UA,BAIJ->UBIJ', h['b'][A,V], t['bb'][pV,pV,hC,hC], optimize=True)
    O['bb'][A,V,C,C] += scale * +0.50000000 * np.einsum('AB,UBIJ->UAIJ', h['b'][V,V], t['bb'][pA,pV,hC,hC], optimize=True)
    O['bb'][C,V,C,C] += scale * +0.50000000 * np.einsum('UV,IV,UAJK->IAJK', eta1['b'], h['b'][C,A], t['bb'][pA,pV,hC,hC], optimize=True)
    O['bb'][C,V,C,C] += scale * +0.50000000 * np.einsum('UV,IV,UAJK->IAJK', gamma1['b'], h['b'][C,A], t['bb'][pA,pV,hC,hC], optimize=True)
    O['bb'][C,V,C,C] += scale * -0.50000000 * np.einsum('IA,BAJK->IBJK', h['b'][C,V], t['bb'][pV,pV,hC,hC], optimize=True)
    O['bb'][V,V,C,C] += scale * +0.50000000 * np.einsum('UV,UI,ABJV->ABIJ', eta1['b'], h['b'][A,C], t['bb'][pV,pV,hC,hA], optimize=True)
    O['bb'][V,V,C,C] += scale * +0.50000000 * np.einsum('UV,AV,UBIJ->ABIJ', eta1['b'], h['b'][V,A], t['bb'][pA,pV,hC,hC], optimize=True)
    O['bb'][V,V,C,C] += scale * +0.50000000 * np.einsum('UV,UI,ABJV->ABIJ', gamma1['b'], h['b'][A,C], t['bb'][pV,pV,hC,hA], optimize=True)
    O['bb'][V,V,C,C] += scale * +0.50000000 * np.einsum('UV,AV,UBIJ->ABIJ', gamma1['b'], h['b'][V,A], t['bb'][pA,pV,hC,hC], optimize=True)
    O['bb'][V,V,C,C] += scale * +0.50000000 * np.einsum('IJ,ABKI->ABJK', h['b'][C,C], t['bb'][pV,pV,hC,hC], optimize=True)
    O['bb'][V,V,C,C] += scale * -0.50000000 * np.einsum('AB,CBIJ->ACIJ', h['b'][V,V], t['bb'][pV,pV,hC,hC], optimize=True)
    O['bb'][A,A,C,V] += scale * -0.50000000 * np.einsum('UV,UA,WXIV->WXIA', eta1['b'], h['b'][A,V], t['bb'][pA,pA,hC,hA], optimize=True)
    O['bb'][A,A,C,V] += scale * -0.50000000 * np.einsum('UV,UA,WXIV->WXIA', gamma1['b'], h['b'][A,V], t['bb'][pA,pA,hC,hA], optimize=True)
    O['bb'][A,A,C,V] += scale * -0.50000000 * np.einsum('IA,UVJI->UVJA', h['b'][C,V], t['bb'][pA,pA,hC,hC], optimize=True)
    O['bb'][A,V,C,V] += scale * -1.00000000 * np.einsum('UV,UA,WBIV->WBIA', eta1['b'], h['b'][A,V], t['bb'][pA,pV,hC,hA], optimize=True)
    O['bb'][A,V,C,V] += scale * -1.00000000 * np.einsum('UV,UA,WBIV->WBIA', gamma1['b'], h['b'][A,V], t['bb'][pA,pV,hC,hA], optimize=True)
    O['bb'][A,V,C,V] += scale * -1.00000000 * np.einsum('IA,UBJI->UBJA', h['b'][C,V], t['bb'][pA,pV,hC,hC], optimize=True)
    O['bb'][V,V,C,V] += scale * -0.50000000 * np.einsum('UV,UA,BCIV->BCIA', eta1['b'], h['b'][A,V], t['bb'][pV,pV,hC,hA], optimize=True)
    O['bb'][V,V,C,V] += scale * -0.50000000 * np.einsum('UV,UA,BCIV->BCIA', gamma1['b'], h['b'][A,V], t['bb'][pV,pV,hC,hA], optimize=True)
    O['bb'][V,V,C,V] += scale * -0.50000000 * np.einsum('IA,BCJI->BCJA', h['b'][C,V], t['bb'][pV,pV,hC,hC], optimize=True)

    t1 = time.time()
    if verbose: print("h1b_t2c_c2c took {:.4f} seconds to run.".format(t1-t0))

    return O
def h2b_t2b_c2c(O, h, t, gamma1, eta1, lambdas, orbspace, verbose=False, scale=1.0):
    # 180 lines
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

    
    
    O['bb'][A,A,A,A] += scale * +1.00000000 * np.einsum('uv,iUvV,uWiX->UWVX', eta1['a'], h['ab'][c,A,a,A], t['ab'][pa,pA,hc,hA], optimize=True)
    O['bb'][A,A,A,A] += scale * +1.00000000 * np.einsum('uv,uUaV,aWvX->UWVX', gamma1['a'], h['ab'][a,A,v,A], t['ab'][pv,pA,ha,hA], optimize=True)
    O['bb'][A,A,A,A] += scale * +1.00000000 * np.einsum('iUaV,aWiX->UWVX', h['ab'][c,A,v,A], t['ab'][pv,pA,hc,hA], optimize=True)
    
    
    O['bb'][C,A,A,A] += scale * +1.00000000 * np.einsum('uv,iIvU,uViW->IVUW', eta1['a'], h['ab'][c,C,a,A], t['ab'][pa,pA,hc,hA], optimize=True)
    O['bb'][C,A,A,A] += scale * +1.00000000 * np.einsum('uv,uIaU,aVvW->IVUW', gamma1['a'], h['ab'][a,C,v,A], t['ab'][pv,pA,ha,hA], optimize=True)
    O['bb'][C,A,A,A] += scale * +1.00000000 * np.einsum('iIaU,aViW->IVUW', h['ab'][c,C,v,A], t['ab'][pv,pA,hc,hA], optimize=True)
    O['bb'][A,V,A,A] += scale * -1.00000000 * np.einsum('uv,wx,uUxV,wAvW->UAVW', eta1['a'], gamma1['a'], h['ab'][a,A,a,A], t['ab'][pa,pV,ha,hA], optimize=True)
    
    O['bb'][A,V,A,A] += scale * +1.00000000 * np.einsum('uv,wx,wUvV,uAxW->UAVW', eta1['a'], gamma1['a'], h['ab'][a,A,a,A], t['ab'][pa,pV,ha,hA], optimize=True)
    
    O['bb'][A,V,A,A] += scale * +1.00000000 * np.einsum('uv,iUvV,uAiW->UAVW', eta1['a'], h['ab'][c,A,a,A], t['ab'][pa,pV,hc,hA], optimize=True)
    O['bb'][A,V,A,A] += scale * -1.00000000 * np.einsum('uv,iAvU,uViW->VAUW', eta1['a'], h['ab'][c,V,a,A], t['ab'][pa,pA,hc,hA], optimize=True)
    O['bb'][A,V,A,A] += scale * +1.00000000 * np.einsum('uv,uUaV,aAvW->UAVW', gamma1['a'], h['ab'][a,A,v,A], t['ab'][pv,pV,ha,hA], optimize=True)
    O['bb'][A,V,A,A] += scale * -1.00000000 * np.einsum('uv,uAaU,aVvW->VAUW', gamma1['a'], h['ab'][a,V,v,A], t['ab'][pv,pA,ha,hA], optimize=True)
    O['bb'][A,V,A,A] += scale * +1.00000000 * np.einsum('iUaV,aAiW->UAVW', h['ab'][c,A,v,A], t['ab'][pv,pV,hc,hA], optimize=True)
    O['bb'][A,V,A,A] += scale * -1.00000000 * np.einsum('iAaU,aViW->VAUW', h['ab'][c,V,v,A], t['ab'][pv,pA,hc,hA], optimize=True)
    O['bb'][C,V,A,A] += scale * -1.00000000 * np.einsum('uv,wx,uIxU,wAvV->IAUV', eta1['a'], gamma1['a'], h['ab'][a,C,a,A], t['ab'][pa,pV,ha,hA], optimize=True)
    O['bb'][C,V,A,A] += scale * +1.00000000 * np.einsum('uv,wx,wIvU,uAxV->IAUV', eta1['a'], gamma1['a'], h['ab'][a,C,a,A], t['ab'][pa,pV,ha,hA], optimize=True)
    O['bb'][C,V,A,A] += scale * +1.00000000 * np.einsum('uv,iIvU,uAiV->IAUV', eta1['a'], h['ab'][c,C,a,A], t['ab'][pa,pV,hc,hA], optimize=True)
    O['bb'][C,V,A,A] += scale * +1.00000000 * np.einsum('uv,uIaU,aAvV->IAUV', gamma1['a'], h['ab'][a,C,v,A], t['ab'][pv,pV,ha,hA], optimize=True)
    O['bb'][C,V,A,A] += scale * +1.00000000 * np.einsum('iIaU,aAiV->IAUV', h['ab'][c,C,v,A], t['ab'][pv,pV,hc,hA], optimize=True)
    O['bb'][V,V,A,A] += scale * -1.00000000 * np.einsum('uv,wx,uAxU,wBvV->ABUV', eta1['a'], gamma1['a'], h['ab'][a,V,a,A], t['ab'][pa,pV,ha,hA], optimize=True)
    O['bb'][V,V,A,A] += scale * +1.00000000 * np.einsum('uv,wx,wAvU,uBxV->ABUV', eta1['a'], gamma1['a'], h['ab'][a,V,a,A], t['ab'][pa,pV,ha,hA], optimize=True)
    O['bb'][V,V,A,A] += scale * +1.00000000 * np.einsum('uv,iAvU,uBiV->ABUV', eta1['a'], h['ab'][c,V,a,A], t['ab'][pa,pV,hc,hA], optimize=True)
    O['bb'][V,V,A,A] += scale * +1.00000000 * np.einsum('uv,uAaU,aBvV->ABUV', gamma1['a'], h['ab'][a,V,v,A], t['ab'][pv,pV,ha,hA], optimize=True)
    O['bb'][V,V,A,A] += scale * +1.00000000 * np.einsum('iAaU,aBiV->ABUV', h['ab'][c,V,v,A], t['ab'][pv,pV,hc,hA], optimize=True)
    
    
    O['bb'][A,A,A,V] += scale * -1.00000000 * np.einsum('uv,iUvA,uViW->UVWA', eta1['a'], h['ab'][c,A,a,V], t['ab'][pa,pA,hc,hA], optimize=True)
    O['bb'][A,A,A,V] += scale * -1.00000000 * np.einsum('uv,uUaA,aVvW->UVWA', gamma1['a'], h['ab'][a,A,v,V], t['ab'][pv,pA,ha,hA], optimize=True)
    O['bb'][A,A,A,V] += scale * -1.00000000 * np.einsum('iUaA,aViW->UVWA', h['ab'][c,A,v,V], t['ab'][pv,pA,hc,hA], optimize=True)
    
    
    O['bb'][C,A,A,V] += scale * -1.00000000 * np.einsum('uv,iIvA,uUiV->IUVA', eta1['a'], h['ab'][c,C,a,V], t['ab'][pa,pA,hc,hA], optimize=True)
    O['bb'][C,A,A,V] += scale * -1.00000000 * np.einsum('uv,uIaA,aUvV->IUVA', gamma1['a'], h['ab'][a,C,v,V], t['ab'][pv,pA,ha,hA], optimize=True)
    O['bb'][C,A,A,V] += scale * -1.00000000 * np.einsum('iIaA,aUiV->IUVA', h['ab'][c,C,v,V], t['ab'][pv,pA,hc,hA], optimize=True)
    O['bb'][A,V,A,V] += scale * +1.00000000 * np.einsum('uv,wx,uUxA,wBvV->UBVA', eta1['a'], gamma1['a'], h['ab'][a,A,a,V], t['ab'][pa,pV,ha,hA], optimize=True)
    
    O['bb'][A,V,A,V] += scale * -1.00000000 * np.einsum('uv,wx,wUvA,uBxV->UBVA', eta1['a'], gamma1['a'], h['ab'][a,A,a,V], t['ab'][pa,pV,ha,hA], optimize=True)
    
    O['bb'][A,V,A,V] += scale * -1.00000000 * np.einsum('uv,iUvA,uBiV->UBVA', eta1['a'], h['ab'][c,A,a,V], t['ab'][pa,pV,hc,hA], optimize=True)
    O['bb'][A,V,A,V] += scale * +1.00000000 * np.einsum('uv,iAvB,uUiV->UAVB', eta1['a'], h['ab'][c,V,a,V], t['ab'][pa,pA,hc,hA], optimize=True)
    O['bb'][A,V,A,V] += scale * -1.00000000 * np.einsum('uv,uUaA,aBvV->UBVA', gamma1['a'], h['ab'][a,A,v,V], t['ab'][pv,pV,ha,hA], optimize=True)
    O['bb'][A,V,A,V] += scale * +1.00000000 * np.einsum('uv,uAaB,aUvV->UAVB', gamma1['a'], h['ab'][a,V,v,V], t['ab'][pv,pA,ha,hA], optimize=True)
    O['bb'][A,V,A,V] += scale * -1.00000000 * np.einsum('iUaA,aBiV->UBVA', h['ab'][c,A,v,V], t['ab'][pv,pV,hc,hA], optimize=True)
    O['bb'][A,V,A,V] += scale * +1.00000000 * np.einsum('iAaB,aUiV->UAVB', h['ab'][c,V,v,V], t['ab'][pv,pA,hc,hA], optimize=True)
    O['bb'][C,V,A,V] += scale * +1.00000000 * np.einsum('uv,wx,uIxA,wBvU->IBUA', eta1['a'], gamma1['a'], h['ab'][a,C,a,V], t['ab'][pa,pV,ha,hA], optimize=True)
    O['bb'][C,V,A,V] += scale * -1.00000000 * np.einsum('uv,wx,wIvA,uBxU->IBUA', eta1['a'], gamma1['a'], h['ab'][a,C,a,V], t['ab'][pa,pV,ha,hA], optimize=True)
    O['bb'][C,V,A,V] += scale * -1.00000000 * np.einsum('uv,iIvA,uBiU->IBUA', eta1['a'], h['ab'][c,C,a,V], t['ab'][pa,pV,hc,hA], optimize=True)
    O['bb'][C,V,A,V] += scale * -1.00000000 * np.einsum('uv,uIaA,aBvU->IBUA', gamma1['a'], h['ab'][a,C,v,V], t['ab'][pv,pV,ha,hA], optimize=True)
    O['bb'][C,V,A,V] += scale * -1.00000000 * np.einsum('iIaA,aBiU->IBUA', h['ab'][c,C,v,V], t['ab'][pv,pV,hc,hA], optimize=True)
    O['bb'][V,V,A,V] += scale * +1.00000000 * np.einsum('uv,wx,uAxB,wCvU->ACUB', eta1['a'], gamma1['a'], h['ab'][a,V,a,V], t['ab'][pa,pV,ha,hA], optimize=True)
    O['bb'][V,V,A,V] += scale * -1.00000000 * np.einsum('uv,wx,wAvB,uCxU->ACUB', eta1['a'], gamma1['a'], h['ab'][a,V,a,V], t['ab'][pa,pV,ha,hA], optimize=True)
    O['bb'][V,V,A,V] += scale * -1.00000000 * np.einsum('uv,iAvB,uCiU->ACUB', eta1['a'], h['ab'][c,V,a,V], t['ab'][pa,pV,hc,hA], optimize=True)
    O['bb'][V,V,A,V] += scale * -1.00000000 * np.einsum('uv,uAaB,aCvU->ACUB', gamma1['a'], h['ab'][a,V,v,V], t['ab'][pv,pV,ha,hA], optimize=True)
    O['bb'][V,V,A,V] += scale * -1.00000000 * np.einsum('iAaB,aCiU->ACUB', h['ab'][c,V,v,V], t['ab'][pv,pV,hc,hA], optimize=True)
    
    O['bb'][A,A,C,A] += scale * +1.00000000 * np.einsum('uv,wx,uUxV,wWvI->UWIV', eta1['a'], gamma1['a'], h['ab'][a,A,a,A], t['ab'][pa,pA,ha,hC], optimize=True)
    
    O['bb'][A,A,C,A] += scale * -1.00000000 * np.einsum('uv,wx,wUvV,uWxI->UWIV', eta1['a'], gamma1['a'], h['ab'][a,A,a,A], t['ab'][pa,pA,ha,hC], optimize=True)
    O['bb'][A,A,C,A] += scale * +1.00000000 * np.einsum('uv,iUvI,uViW->UVIW', eta1['a'], h['ab'][c,A,a,C], t['ab'][pa,pA,hc,hA], optimize=True)
    O['bb'][A,A,C,A] += scale * -1.00000000 * np.einsum('uv,iUvV,uWiI->UWIV', eta1['a'], h['ab'][c,A,a,A], t['ab'][pa,pA,hc,hC], optimize=True)
    O['bb'][A,A,C,A] += scale * +1.00000000 * np.einsum('uv,uUaI,aVvW->UVIW', gamma1['a'], h['ab'][a,A,v,C], t['ab'][pv,pA,ha,hA], optimize=True)
    O['bb'][A,A,C,A] += scale * -1.00000000 * np.einsum('uv,uUaV,aWvI->UWIV', gamma1['a'], h['ab'][a,A,v,A], t['ab'][pv,pA,ha,hC], optimize=True)
    O['bb'][A,A,C,A] += scale * +1.00000000 * np.einsum('iUaI,aViW->UVIW', h['ab'][c,A,v,C], t['ab'][pv,pA,hc,hA], optimize=True)
    O['bb'][A,A,C,A] += scale * -1.00000000 * np.einsum('iUaV,aWiI->UWIV', h['ab'][c,A,v,A], t['ab'][pv,pA,hc,hC], optimize=True)
    
    O['bb'][C,A,C,A] += scale * +1.00000000 * np.einsum('uv,wx,uIxU,wVvJ->IVJU', eta1['a'], gamma1['a'], h['ab'][a,C,a,A], t['ab'][pa,pA,ha,hC], optimize=True)
    
    O['bb'][C,A,C,A] += scale * -1.00000000 * np.einsum('uv,wx,wIvU,uVxJ->IVJU', eta1['a'], gamma1['a'], h['ab'][a,C,a,A], t['ab'][pa,pA,ha,hC], optimize=True)
    O['bb'][C,A,C,A] += scale * +1.00000000 * np.einsum('uv,iIvJ,uUiV->IUJV', eta1['a'], h['ab'][c,C,a,C], t['ab'][pa,pA,hc,hA], optimize=True)
    O['bb'][C,A,C,A] += scale * -1.00000000 * np.einsum('uv,iIvU,uViJ->IVJU', eta1['a'], h['ab'][c,C,a,A], t['ab'][pa,pA,hc,hC], optimize=True)
    O['bb'][C,A,C,A] += scale * +1.00000000 * np.einsum('uv,uIaJ,aUvV->IUJV', gamma1['a'], h['ab'][a,C,v,C], t['ab'][pv,pA,ha,hA], optimize=True)
    O['bb'][C,A,C,A] += scale * -1.00000000 * np.einsum('uv,uIaU,aVvJ->IVJU', gamma1['a'], h['ab'][a,C,v,A], t['ab'][pv,pA,ha,hC], optimize=True)
    O['bb'][C,A,C,A] += scale * +1.00000000 * np.einsum('iIaJ,aUiV->IUJV', h['ab'][c,C,v,C], t['ab'][pv,pA,hc,hA], optimize=True)
    O['bb'][C,A,C,A] += scale * -1.00000000 * np.einsum('iIaU,aViJ->IVJU', h['ab'][c,C,v,A], t['ab'][pv,pA,hc,hC], optimize=True)
    O['bb'][A,V,C,A] += scale * -1.00000000 * np.einsum('uv,wx,uUxI,wAvV->UAIV', eta1['a'], gamma1['a'], h['ab'][a,A,a,C], t['ab'][pa,pV,ha,hA], optimize=True)
    O['bb'][A,V,C,A] += scale * +1.00000000 * np.einsum('uv,wx,uUxV,wAvI->UAIV', eta1['a'], gamma1['a'], h['ab'][a,A,a,A], t['ab'][pa,pV,ha,hC], optimize=True)
    
    O['bb'][A,V,C,A] += scale * -1.00000000 * np.einsum('uv,wx,uAxU,wVvI->VAIU', eta1['a'], gamma1['a'], h['ab'][a,V,a,A], t['ab'][pa,pA,ha,hC], optimize=True)
    O['bb'][A,V,C,A] += scale * +1.00000000 * np.einsum('uv,wx,wUvI,uAxV->UAIV', eta1['a'], gamma1['a'], h['ab'][a,A,a,C], t['ab'][pa,pV,ha,hA], optimize=True)
    O['bb'][A,V,C,A] += scale * -1.00000000 * np.einsum('uv,wx,wUvV,uAxI->UAIV', eta1['a'], gamma1['a'], h['ab'][a,A,a,A], t['ab'][pa,pV,ha,hC], optimize=True)
    
    O['bb'][A,V,C,A] += scale * +1.00000000 * np.einsum('uv,wx,wAvU,uVxI->VAIU', eta1['a'], gamma1['a'], h['ab'][a,V,a,A], t['ab'][pa,pA,ha,hC], optimize=True)
    O['bb'][A,V,C,A] += scale * +1.00000000 * np.einsum('uv,iUvI,uAiV->UAIV', eta1['a'], h['ab'][c,A,a,C], t['ab'][pa,pV,hc,hA], optimize=True)
    O['bb'][A,V,C,A] += scale * -1.00000000 * np.einsum('uv,iUvV,uAiI->UAIV', eta1['a'], h['ab'][c,A,a,A], t['ab'][pa,pV,hc,hC], optimize=True)
    O['bb'][A,V,C,A] += scale * -1.00000000 * np.einsum('uv,iAvI,uUiV->UAIV', eta1['a'], h['ab'][c,V,a,C], t['ab'][pa,pA,hc,hA], optimize=True)
    O['bb'][A,V,C,A] += scale * +1.00000000 * np.einsum('uv,iAvU,uViI->VAIU', eta1['a'], h['ab'][c,V,a,A], t['ab'][pa,pA,hc,hC], optimize=True)
    O['bb'][A,V,C,A] += scale * +1.00000000 * np.einsum('uv,uUaI,aAvV->UAIV', gamma1['a'], h['ab'][a,A,v,C], t['ab'][pv,pV,ha,hA], optimize=True)
    O['bb'][A,V,C,A] += scale * -1.00000000 * np.einsum('uv,uUaV,aAvI->UAIV', gamma1['a'], h['ab'][a,A,v,A], t['ab'][pv,pV,ha,hC], optimize=True)
    O['bb'][A,V,C,A] += scale * -1.00000000 * np.einsum('uv,uAaI,aUvV->UAIV', gamma1['a'], h['ab'][a,V,v,C], t['ab'][pv,pA,ha,hA], optimize=True)
    O['bb'][A,V,C,A] += scale * +1.00000000 * np.einsum('uv,uAaU,aVvI->VAIU', gamma1['a'], h['ab'][a,V,v,A], t['ab'][pv,pA,ha,hC], optimize=True)
    O['bb'][A,V,C,A] += scale * +1.00000000 * np.einsum('iUaI,aAiV->UAIV', h['ab'][c,A,v,C], t['ab'][pv,pV,hc,hA], optimize=True)
    O['bb'][A,V,C,A] += scale * -1.00000000 * np.einsum('iUaV,aAiI->UAIV', h['ab'][c,A,v,A], t['ab'][pv,pV,hc,hC], optimize=True)
    O['bb'][A,V,C,A] += scale * -1.00000000 * np.einsum('iAaI,aUiV->UAIV', h['ab'][c,V,v,C], t['ab'][pv,pA,hc,hA], optimize=True)
    O['bb'][A,V,C,A] += scale * +1.00000000 * np.einsum('iAaU,aViI->VAIU', h['ab'][c,V,v,A], t['ab'][pv,pA,hc,hC], optimize=True)
    O['bb'][C,V,C,A] += scale * -1.00000000 * np.einsum('uv,wx,uIxJ,wAvU->IAJU', eta1['a'], gamma1['a'], h['ab'][a,C,a,C], t['ab'][pa,pV,ha,hA], optimize=True)
    O['bb'][C,V,C,A] += scale * +1.00000000 * np.einsum('uv,wx,uIxU,wAvJ->IAJU', eta1['a'], gamma1['a'], h['ab'][a,C,a,A], t['ab'][pa,pV,ha,hC], optimize=True)
    O['bb'][C,V,C,A] += scale * +1.00000000 * np.einsum('uv,wx,wIvJ,uAxU->IAJU', eta1['a'], gamma1['a'], h['ab'][a,C,a,C], t['ab'][pa,pV,ha,hA], optimize=True)
    O['bb'][C,V,C,A] += scale * -1.00000000 * np.einsum('uv,wx,wIvU,uAxJ->IAJU', eta1['a'], gamma1['a'], h['ab'][a,C,a,A], t['ab'][pa,pV,ha,hC], optimize=True)
    O['bb'][C,V,C,A] += scale * +1.00000000 * np.einsum('uv,iIvJ,uAiU->IAJU', eta1['a'], h['ab'][c,C,a,C], t['ab'][pa,pV,hc,hA], optimize=True)
    O['bb'][C,V,C,A] += scale * -1.00000000 * np.einsum('uv,iIvU,uAiJ->IAJU', eta1['a'], h['ab'][c,C,a,A], t['ab'][pa,pV,hc,hC], optimize=True)
    O['bb'][C,V,C,A] += scale * +1.00000000 * np.einsum('uv,uIaJ,aAvU->IAJU', gamma1['a'], h['ab'][a,C,v,C], t['ab'][pv,pV,ha,hA], optimize=True)
    O['bb'][C,V,C,A] += scale * -1.00000000 * np.einsum('uv,uIaU,aAvJ->IAJU', gamma1['a'], h['ab'][a,C,v,A], t['ab'][pv,pV,ha,hC], optimize=True)
    O['bb'][C,V,C,A] += scale * +1.00000000 * np.einsum('iIaJ,aAiU->IAJU', h['ab'][c,C,v,C], t['ab'][pv,pV,hc,hA], optimize=True)
    O['bb'][C,V,C,A] += scale * -1.00000000 * np.einsum('iIaU,aAiJ->IAJU', h['ab'][c,C,v,A], t['ab'][pv,pV,hc,hC], optimize=True)
    O['bb'][V,V,C,A] += scale * -1.00000000 * np.einsum('uv,wx,uAxI,wBvU->ABIU', eta1['a'], gamma1['a'], h['ab'][a,V,a,C], t['ab'][pa,pV,ha,hA], optimize=True)
    O['bb'][V,V,C,A] += scale * +1.00000000 * np.einsum('uv,wx,uAxU,wBvI->ABIU', eta1['a'], gamma1['a'], h['ab'][a,V,a,A], t['ab'][pa,pV,ha,hC], optimize=True)
    O['bb'][V,V,C,A] += scale * +1.00000000 * np.einsum('uv,wx,wAvI,uBxU->ABIU', eta1['a'], gamma1['a'], h['ab'][a,V,a,C], t['ab'][pa,pV,ha,hA], optimize=True)
    O['bb'][V,V,C,A] += scale * -1.00000000 * np.einsum('uv,wx,wAvU,uBxI->ABIU', eta1['a'], gamma1['a'], h['ab'][a,V,a,A], t['ab'][pa,pV,ha,hC], optimize=True)
    O['bb'][V,V,C,A] += scale * +1.00000000 * np.einsum('uv,iAvI,uBiU->ABIU', eta1['a'], h['ab'][c,V,a,C], t['ab'][pa,pV,hc,hA], optimize=True)
    O['bb'][V,V,C,A] += scale * -1.00000000 * np.einsum('uv,iAvU,uBiI->ABIU', eta1['a'], h['ab'][c,V,a,A], t['ab'][pa,pV,hc,hC], optimize=True)
    O['bb'][V,V,C,A] += scale * +1.00000000 * np.einsum('uv,uAaI,aBvU->ABIU', gamma1['a'], h['ab'][a,V,v,C], t['ab'][pv,pV,ha,hA], optimize=True)
    O['bb'][V,V,C,A] += scale * -1.00000000 * np.einsum('uv,uAaU,aBvI->ABIU', gamma1['a'], h['ab'][a,V,v,A], t['ab'][pv,pV,ha,hC], optimize=True)
    O['bb'][V,V,C,A] += scale * +1.00000000 * np.einsum('iAaI,aBiU->ABIU', h['ab'][c,V,v,C], t['ab'][pv,pV,hc,hA], optimize=True)
    O['bb'][V,V,C,A] += scale * -1.00000000 * np.einsum('iAaU,aBiI->ABIU', h['ab'][c,V,v,A], t['ab'][pv,pV,hc,hC], optimize=True)
    O['bb'][A,A,C,C] += scale * -1.00000000 * np.einsum('uv,wx,uUxI,wVvJ->UVIJ', eta1['a'], gamma1['a'], h['ab'][a,A,a,C], t['ab'][pa,pA,ha,hC], optimize=True)
    O['bb'][A,A,C,C] += scale * +1.00000000 * np.einsum('uv,wx,wUvI,uVxJ->UVIJ', eta1['a'], gamma1['a'], h['ab'][a,A,a,C], t['ab'][pa,pA,ha,hC], optimize=True)
    O['bb'][A,A,C,C] += scale * +1.00000000 * np.einsum('uv,iUvI,uViJ->UVIJ', eta1['a'], h['ab'][c,A,a,C], t['ab'][pa,pA,hc,hC], optimize=True)
    O['bb'][A,A,C,C] += scale * +1.00000000 * np.einsum('uv,uUaI,aVvJ->UVIJ', gamma1['a'], h['ab'][a,A,v,C], t['ab'][pv,pA,ha,hC], optimize=True)
    O['bb'][A,A,C,C] += scale * +1.00000000 * np.einsum('iUaI,aViJ->UVIJ', h['ab'][c,A,v,C], t['ab'][pv,pA,hc,hC], optimize=True)
    O['bb'][C,A,C,C] += scale * -1.00000000 * np.einsum('uv,wx,uIxJ,wUvK->IUJK', eta1['a'], gamma1['a'], h['ab'][a,C,a,C], t['ab'][pa,pA,ha,hC], optimize=True)
    O['bb'][C,A,C,C] += scale * +1.00000000 * np.einsum('uv,wx,wIvJ,uUxK->IUJK', eta1['a'], gamma1['a'], h['ab'][a,C,a,C], t['ab'][pa,pA,ha,hC], optimize=True)
    O['bb'][C,A,C,C] += scale * +1.00000000 * np.einsum('uv,iIvJ,uUiK->IUJK', eta1['a'], h['ab'][c,C,a,C], t['ab'][pa,pA,hc,hC], optimize=True)
    O['bb'][C,A,C,C] += scale * +1.00000000 * np.einsum('uv,uIaJ,aUvK->IUJK', gamma1['a'], h['ab'][a,C,v,C], t['ab'][pv,pA,ha,hC], optimize=True)
    O['bb'][C,A,C,C] += scale * +1.00000000 * np.einsum('iIaJ,aUiK->IUJK', h['ab'][c,C,v,C], t['ab'][pv,pA,hc,hC], optimize=True)
    O['bb'][A,V,C,C] += scale * -1.00000000 * np.einsum('uv,wx,uUxI,wAvJ->UAIJ', eta1['a'], gamma1['a'], h['ab'][a,A,a,C], t['ab'][pa,pV,ha,hC], optimize=True)
    O['bb'][A,V,C,C] += scale * +1.00000000 * np.einsum('uv,wx,uAxI,wUvJ->UAIJ', eta1['a'], gamma1['a'], h['ab'][a,V,a,C], t['ab'][pa,pA,ha,hC], optimize=True)
    O['bb'][A,V,C,C] += scale * +1.00000000 * np.einsum('uv,wx,wUvI,uAxJ->UAIJ', eta1['a'], gamma1['a'], h['ab'][a,A,a,C], t['ab'][pa,pV,ha,hC], optimize=True)
    O['bb'][A,V,C,C] += scale * -1.00000000 * np.einsum('uv,wx,wAvI,uUxJ->UAIJ', eta1['a'], gamma1['a'], h['ab'][a,V,a,C], t['ab'][pa,pA,ha,hC], optimize=True)
    O['bb'][A,V,C,C] += scale * +1.00000000 * np.einsum('uv,iUvI,uAiJ->UAIJ', eta1['a'], h['ab'][c,A,a,C], t['ab'][pa,pV,hc,hC], optimize=True)
    O['bb'][A,V,C,C] += scale * -1.00000000 * np.einsum('uv,iAvI,uUiJ->UAIJ', eta1['a'], h['ab'][c,V,a,C], t['ab'][pa,pA,hc,hC], optimize=True)
    O['bb'][A,V,C,C] += scale * +1.00000000 * np.einsum('uv,uUaI,aAvJ->UAIJ', gamma1['a'], h['ab'][a,A,v,C], t['ab'][pv,pV,ha,hC], optimize=True)
    O['bb'][A,V,C,C] += scale * -1.00000000 * np.einsum('uv,uAaI,aUvJ->UAIJ', gamma1['a'], h['ab'][a,V,v,C], t['ab'][pv,pA,ha,hC], optimize=True)
    O['bb'][A,V,C,C] += scale * +1.00000000 * np.einsum('iUaI,aAiJ->UAIJ', h['ab'][c,A,v,C], t['ab'][pv,pV,hc,hC], optimize=True)
    O['bb'][A,V,C,C] += scale * -1.00000000 * np.einsum('iAaI,aUiJ->UAIJ', h['ab'][c,V,v,C], t['ab'][pv,pA,hc,hC], optimize=True)
    O['bb'][C,V,C,C] += scale * -1.00000000 * np.einsum('uv,wx,uIxJ,wAvK->IAJK', eta1['a'], gamma1['a'], h['ab'][a,C,a,C], t['ab'][pa,pV,ha,hC], optimize=True)
    O['bb'][C,V,C,C] += scale * +1.00000000 * np.einsum('uv,wx,wIvJ,uAxK->IAJK', eta1['a'], gamma1['a'], h['ab'][a,C,a,C], t['ab'][pa,pV,ha,hC], optimize=True)
    O['bb'][C,V,C,C] += scale * +1.00000000 * np.einsum('uv,iIvJ,uAiK->IAJK', eta1['a'], h['ab'][c,C,a,C], t['ab'][pa,pV,hc,hC], optimize=True)
    O['bb'][C,V,C,C] += scale * +1.00000000 * np.einsum('uv,uIaJ,aAvK->IAJK', gamma1['a'], h['ab'][a,C,v,C], t['ab'][pv,pV,ha,hC], optimize=True)
    O['bb'][C,V,C,C] += scale * +1.00000000 * np.einsum('iIaJ,aAiK->IAJK', h['ab'][c,C,v,C], t['ab'][pv,pV,hc,hC], optimize=True)
    O['bb'][V,V,C,C] += scale * -1.00000000 * np.einsum('uv,wx,uAxI,wBvJ->ABIJ', eta1['a'], gamma1['a'], h['ab'][a,V,a,C], t['ab'][pa,pV,ha,hC], optimize=True)
    O['bb'][V,V,C,C] += scale * +1.00000000 * np.einsum('uv,wx,wAvI,uBxJ->ABIJ', eta1['a'], gamma1['a'], h['ab'][a,V,a,C], t['ab'][pa,pV,ha,hC], optimize=True)
    O['bb'][V,V,C,C] += scale * +1.00000000 * np.einsum('uv,iAvI,uBiJ->ABIJ', eta1['a'], h['ab'][c,V,a,C], t['ab'][pa,pV,hc,hC], optimize=True)
    O['bb'][V,V,C,C] += scale * +1.00000000 * np.einsum('uv,uAaI,aBvJ->ABIJ', gamma1['a'], h['ab'][a,V,v,C], t['ab'][pv,pV,ha,hC], optimize=True)
    O['bb'][V,V,C,C] += scale * +1.00000000 * np.einsum('iAaI,aBiJ->ABIJ', h['ab'][c,V,v,C], t['ab'][pv,pV,hc,hC], optimize=True)
    O['bb'][A,A,C,V] += scale * +1.00000000 * np.einsum('uv,wx,uUxA,wVvI->UVIA', eta1['a'], gamma1['a'], h['ab'][a,A,a,V], t['ab'][pa,pA,ha,hC], optimize=True)
    O['bb'][A,A,C,V] += scale * -1.00000000 * np.einsum('uv,wx,wUvA,uVxI->UVIA', eta1['a'], gamma1['a'], h['ab'][a,A,a,V], t['ab'][pa,pA,ha,hC], optimize=True)
    O['bb'][A,A,C,V] += scale * -1.00000000 * np.einsum('uv,iUvA,uViI->UVIA', eta1['a'], h['ab'][c,A,a,V], t['ab'][pa,pA,hc,hC], optimize=True)
    O['bb'][A,A,C,V] += scale * -1.00000000 * np.einsum('uv,uUaA,aVvI->UVIA', gamma1['a'], h['ab'][a,A,v,V], t['ab'][pv,pA,ha,hC], optimize=True)
    O['bb'][A,A,C,V] += scale * -1.00000000 * np.einsum('iUaA,aViI->UVIA', h['ab'][c,A,v,V], t['ab'][pv,pA,hc,hC], optimize=True)
    O['bb'][C,A,C,V] += scale * +1.00000000 * np.einsum('uv,wx,uIxA,wUvJ->IUJA', eta1['a'], gamma1['a'], h['ab'][a,C,a,V], t['ab'][pa,pA,ha,hC], optimize=True)
    O['bb'][C,A,C,V] += scale * -1.00000000 * np.einsum('uv,wx,wIvA,uUxJ->IUJA', eta1['a'], gamma1['a'], h['ab'][a,C,a,V], t['ab'][pa,pA,ha,hC], optimize=True)
    O['bb'][C,A,C,V] += scale * -1.00000000 * np.einsum('uv,iIvA,uUiJ->IUJA', eta1['a'], h['ab'][c,C,a,V], t['ab'][pa,pA,hc,hC], optimize=True)
    O['bb'][C,A,C,V] += scale * -1.00000000 * np.einsum('uv,uIaA,aUvJ->IUJA', gamma1['a'], h['ab'][a,C,v,V], t['ab'][pv,pA,ha,hC], optimize=True)
    O['bb'][C,A,C,V] += scale * -1.00000000 * np.einsum('iIaA,aUiJ->IUJA', h['ab'][c,C,v,V], t['ab'][pv,pA,hc,hC], optimize=True)
    O['bb'][A,V,C,V] += scale * +1.00000000 * np.einsum('uv,wx,uUxA,wBvI->UBIA', eta1['a'], gamma1['a'], h['ab'][a,A,a,V], t['ab'][pa,pV,ha,hC], optimize=True)
    O['bb'][A,V,C,V] += scale * -1.00000000 * np.einsum('uv,wx,uAxB,wUvI->UAIB', eta1['a'], gamma1['a'], h['ab'][a,V,a,V], t['ab'][pa,pA,ha,hC], optimize=True)
    O['bb'][A,V,C,V] += scale * -1.00000000 * np.einsum('uv,wx,wUvA,uBxI->UBIA', eta1['a'], gamma1['a'], h['ab'][a,A,a,V], t['ab'][pa,pV,ha,hC], optimize=True)
    O['bb'][A,V,C,V] += scale * +1.00000000 * np.einsum('uv,wx,wAvB,uUxI->UAIB', eta1['a'], gamma1['a'], h['ab'][a,V,a,V], t['ab'][pa,pA,ha,hC], optimize=True)
    O['bb'][A,V,C,V] += scale * -1.00000000 * np.einsum('uv,iUvA,uBiI->UBIA', eta1['a'], h['ab'][c,A,a,V], t['ab'][pa,pV,hc,hC], optimize=True)
    O['bb'][A,V,C,V] += scale * +1.00000000 * np.einsum('uv,iAvB,uUiI->UAIB', eta1['a'], h['ab'][c,V,a,V], t['ab'][pa,pA,hc,hC], optimize=True)
    O['bb'][A,V,C,V] += scale * -1.00000000 * np.einsum('uv,uUaA,aBvI->UBIA', gamma1['a'], h['ab'][a,A,v,V], t['ab'][pv,pV,ha,hC], optimize=True)
    O['bb'][A,V,C,V] += scale * +1.00000000 * np.einsum('uv,uAaB,aUvI->UAIB', gamma1['a'], h['ab'][a,V,v,V], t['ab'][pv,pA,ha,hC], optimize=True)
    O['bb'][A,V,C,V] += scale * -1.00000000 * np.einsum('iUaA,aBiI->UBIA', h['ab'][c,A,v,V], t['ab'][pv,pV,hc,hC], optimize=True)
    O['bb'][A,V,C,V] += scale * +1.00000000 * np.einsum('iAaB,aUiI->UAIB', h['ab'][c,V,v,V], t['ab'][pv,pA,hc,hC], optimize=True)
    O['bb'][C,V,C,V] += scale * +1.00000000 * np.einsum('uv,wx,uIxA,wBvJ->IBJA', eta1['a'], gamma1['a'], h['ab'][a,C,a,V], t['ab'][pa,pV,ha,hC], optimize=True)
    O['bb'][C,V,C,V] += scale * -1.00000000 * np.einsum('uv,wx,wIvA,uBxJ->IBJA', eta1['a'], gamma1['a'], h['ab'][a,C,a,V], t['ab'][pa,pV,ha,hC], optimize=True)
    O['bb'][C,V,C,V] += scale * -1.00000000 * np.einsum('uv,iIvA,uBiJ->IBJA', eta1['a'], h['ab'][c,C,a,V], t['ab'][pa,pV,hc,hC], optimize=True)
    O['bb'][C,V,C,V] += scale * -1.00000000 * np.einsum('uv,uIaA,aBvJ->IBJA', gamma1['a'], h['ab'][a,C,v,V], t['ab'][pv,pV,ha,hC], optimize=True)
    O['bb'][C,V,C,V] += scale * -1.00000000 * np.einsum('iIaA,aBiJ->IBJA', h['ab'][c,C,v,V], t['ab'][pv,pV,hc,hC], optimize=True)
    O['bb'][V,V,C,V] += scale * +1.00000000 * np.einsum('uv,wx,uAxB,wCvI->ACIB', eta1['a'], gamma1['a'], h['ab'][a,V,a,V], t['ab'][pa,pV,ha,hC], optimize=True)
    O['bb'][V,V,C,V] += scale * -1.00000000 * np.einsum('uv,wx,wAvB,uCxI->ACIB', eta1['a'], gamma1['a'], h['ab'][a,V,a,V], t['ab'][pa,pV,ha,hC], optimize=True)
    O['bb'][V,V,C,V] += scale * -1.00000000 * np.einsum('uv,iAvB,uCiI->ACIB', eta1['a'], h['ab'][c,V,a,V], t['ab'][pa,pV,hc,hC], optimize=True)
    O['bb'][V,V,C,V] += scale * -1.00000000 * np.einsum('uv,uAaB,aCvI->ACIB', gamma1['a'], h['ab'][a,V,v,V], t['ab'][pv,pV,ha,hC], optimize=True)
    O['bb'][V,V,C,V] += scale * -1.00000000 * np.einsum('iAaB,aCiI->ACIB', h['ab'][c,V,v,V], t['ab'][pv,pV,hc,hC], optimize=True)

    t1 = time.time()
    if verbose: print("h2b_t2b_c2c took {:.4f} seconds to run.".format(t1-t0))

    return O
def h2c_t1b_c2c(O, h, t, gamma1, eta1, lambdas, orbspace, verbose=False, scale=1.0):
    # 216 lines
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

    
    
    
    
    O['bb'][A,A,A,A] += scale * +0.50000000 * np.einsum('IUVW,XI->UXVW', h['bb'][C,A,A,A], t['b'][pA,hC], optimize=True)
    O['bb'][A,A,A,A] += scale * +0.50000000 * np.einsum('UVWA,AX->UVWX', h['bb'][A,A,A,V], t['b'][pV,hA], optimize=True)
    
    
    
    
    O['bb'][C,A,A,A] += scale * -0.50000000 * np.einsum('IJUV,WJ->IWUV', h['bb'][C,C,A,A], t['b'][pA,hC], optimize=True)
    O['bb'][C,A,A,A] += scale * +1.00000000 * np.einsum('IUVA,AW->IUVW', h['bb'][C,A,A,V], t['b'][pV,hA], optimize=True)
    
    
    O['bb'][C,C,A,A] += scale * +0.50000000 * np.einsum('IJUA,AV->IJUV', h['bb'][C,C,A,V], t['b'][pV,hA], optimize=True)
    O['bb'][A,V,A,A] += scale * -0.50000000 * np.einsum('UV,WUXY,AV->WAXY', eta1['b'], h['bb'][A,A,A,A], t['b'][pV,hA], optimize=True)
    
    
    O['bb'][A,V,A,A] += scale * -0.50000000 * np.einsum('UV,WUXY,AV->WAXY', gamma1['b'], h['bb'][A,A,A,A], t['b'][pV,hA], optimize=True)
    
    
    O['bb'][A,V,A,A] += scale * +0.50000000 * np.einsum('IUVW,AI->UAVW', h['bb'][C,A,A,A], t['b'][pV,hC], optimize=True)
    O['bb'][A,V,A,A] += scale * -0.50000000 * np.einsum('IAUV,WI->WAUV', h['bb'][C,V,A,A], t['b'][pA,hC], optimize=True)
    O['bb'][A,V,A,A] += scale * +1.00000000 * np.einsum('UAVB,BW->UAVW', h['bb'][A,V,A,V], t['b'][pV,hA], optimize=True)
    O['bb'][C,V,A,A] += scale * -0.50000000 * np.einsum('UV,IUWX,AV->IAWX', eta1['b'], h['bb'][C,A,A,A], t['b'][pV,hA], optimize=True)
    
    O['bb'][C,V,A,A] += scale * -0.50000000 * np.einsum('UV,IUWX,AV->IAWX', gamma1['b'], h['bb'][C,A,A,A], t['b'][pV,hA], optimize=True)
    
    O['bb'][C,V,A,A] += scale * -0.50000000 * np.einsum('IJUV,AJ->IAUV', h['bb'][C,C,A,A], t['b'][pV,hC], optimize=True)
    O['bb'][C,V,A,A] += scale * +1.00000000 * np.einsum('IAUB,BV->IAUV', h['bb'][C,V,A,V], t['b'][pV,hA], optimize=True)
    O['bb'][V,V,A,A] += scale * +0.50000000 * np.einsum('UV,UAWX,BV->ABWX', eta1['b'], h['bb'][A,V,A,A], t['b'][pV,hA], optimize=True)
    
    O['bb'][V,V,A,A] += scale * +0.50000000 * np.einsum('UV,UAWX,BV->ABWX', gamma1['b'], h['bb'][A,V,A,A], t['b'][pV,hA], optimize=True)
    
    O['bb'][V,V,A,A] += scale * +0.50000000 * np.einsum('IAUV,BI->ABUV', h['bb'][C,V,A,A], t['b'][pV,hC], optimize=True)
    O['bb'][V,V,A,A] += scale * +0.50000000 * np.einsum('ABUC,CV->ABUV', h['bb'][V,V,A,V], t['b'][pV,hA], optimize=True)
    
    
    
    
    O['bb'][A,A,A,V] += scale * +1.00000000 * np.einsum('IUVA,WI->UWVA', h['bb'][C,A,A,V], t['b'][pA,hC], optimize=True)
    O['bb'][A,A,A,V] += scale * -0.50000000 * np.einsum('UVAB,BW->UVWA', h['bb'][A,A,V,V], t['b'][pV,hA], optimize=True)
    
    
    
    
    O['bb'][C,A,A,V] += scale * -1.00000000 * np.einsum('IJUA,VJ->IVUA', h['bb'][C,C,A,V], t['b'][pA,hC], optimize=True)
    O['bb'][C,A,A,V] += scale * -1.00000000 * np.einsum('IUAB,BV->IUVA', h['bb'][C,A,V,V], t['b'][pV,hA], optimize=True)
    
    
    O['bb'][C,C,A,V] += scale * -0.50000000 * np.einsum('IJAB,BU->IJUA', h['bb'][C,C,V,V], t['b'][pV,hA], optimize=True)
    O['bb'][A,V,A,V] += scale * -1.00000000 * np.einsum('UV,WUXA,BV->WBXA', eta1['b'], h['bb'][A,A,A,V], t['b'][pV,hA], optimize=True)
    
    
    O['bb'][A,V,A,V] += scale * -1.00000000 * np.einsum('UV,WUXA,BV->WBXA', gamma1['b'], h['bb'][A,A,A,V], t['b'][pV,hA], optimize=True)
    
    
    O['bb'][A,V,A,V] += scale * +1.00000000 * np.einsum('IUVA,BI->UBVA', h['bb'][C,A,A,V], t['b'][pV,hC], optimize=True)
    O['bb'][A,V,A,V] += scale * -1.00000000 * np.einsum('IAUB,VI->VAUB', h['bb'][C,V,A,V], t['b'][pA,hC], optimize=True)
    O['bb'][A,V,A,V] += scale * -1.00000000 * np.einsum('UABC,CV->UAVB', h['bb'][A,V,V,V], t['b'][pV,hA], optimize=True)
    O['bb'][C,V,A,V] += scale * -1.00000000 * np.einsum('UV,IUWA,BV->IBWA', eta1['b'], h['bb'][C,A,A,V], t['b'][pV,hA], optimize=True)
    
    O['bb'][C,V,A,V] += scale * -1.00000000 * np.einsum('UV,IUWA,BV->IBWA', gamma1['b'], h['bb'][C,A,A,V], t['b'][pV,hA], optimize=True)
    
    O['bb'][C,V,A,V] += scale * -1.00000000 * np.einsum('IJUA,BJ->IBUA', h['bb'][C,C,A,V], t['b'][pV,hC], optimize=True)
    O['bb'][C,V,A,V] += scale * -1.00000000 * np.einsum('IABC,CU->IAUB', h['bb'][C,V,V,V], t['b'][pV,hA], optimize=True)
    O['bb'][V,V,A,V] += scale * +1.00000000 * np.einsum('UV,UAWB,CV->ACWB', eta1['b'], h['bb'][A,V,A,V], t['b'][pV,hA], optimize=True)
    
    O['bb'][V,V,A,V] += scale * +1.00000000 * np.einsum('UV,UAWB,CV->ACWB', gamma1['b'], h['bb'][A,V,A,V], t['b'][pV,hA], optimize=True)
    
    O['bb'][V,V,A,V] += scale * +1.00000000 * np.einsum('IAUB,CI->ACUB', h['bb'][C,V,A,V], t['b'][pV,hC], optimize=True)
    O['bb'][V,V,A,V] += scale * -0.50000000 * np.einsum('ABCD,DU->ABUC', h['bb'][V,V,V,V], t['b'][pV,hA], optimize=True)
    
    O['bb'][A,A,C,A] += scale * -0.50000000 * np.einsum('UV,WXYV,UI->WXIY', eta1['b'], h['bb'][A,A,A,A], t['b'][pA,hC], optimize=True)
    
    
    O['bb'][A,A,C,A] += scale * -0.50000000 * np.einsum('UV,WXYV,UI->WXIY', gamma1['b'], h['bb'][A,A,A,A], t['b'][pA,hC], optimize=True)
    
    O['bb'][A,A,C,A] += scale * +1.00000000 * np.einsum('IUJV,WI->UWJV', h['bb'][C,A,C,A], t['b'][pA,hC], optimize=True)
    O['bb'][A,A,C,A] += scale * +0.50000000 * np.einsum('UVIA,AW->UVIW', h['bb'][A,A,C,V], t['b'][pV,hA], optimize=True)
    O['bb'][A,A,C,A] += scale * -0.50000000 * np.einsum('UVWA,AI->UVIW', h['bb'][A,A,A,V], t['b'][pV,hC], optimize=True)
    
    O['bb'][C,A,C,A] += scale * -1.00000000 * np.einsum('UV,IWXV,UJ->IWJX', eta1['b'], h['bb'][C,A,A,A], t['b'][pA,hC], optimize=True)
    
    
    O['bb'][C,A,C,A] += scale * -1.00000000 * np.einsum('UV,IWXV,UJ->IWJX', gamma1['b'], h['bb'][C,A,A,A], t['b'][pA,hC], optimize=True)
    
    O['bb'][C,A,C,A] += scale * -1.00000000 * np.einsum('IJKU,VJ->IVKU', h['bb'][C,C,C,A], t['b'][pA,hC], optimize=True)
    O['bb'][C,A,C,A] += scale * +1.00000000 * np.einsum('IUJA,AV->IUJV', h['bb'][C,A,C,V], t['b'][pV,hA], optimize=True)
    O['bb'][C,A,C,A] += scale * -1.00000000 * np.einsum('IUVA,AJ->IUJV', h['bb'][C,A,A,V], t['b'][pV,hC], optimize=True)
    
    O['bb'][C,C,C,A] += scale * -0.50000000 * np.einsum('UV,IJWV,UK->IJKW', eta1['b'], h['bb'][C,C,A,A], t['b'][pA,hC], optimize=True)
    
    O['bb'][C,C,C,A] += scale * -0.50000000 * np.einsum('UV,IJWV,UK->IJKW', gamma1['b'], h['bb'][C,C,A,A], t['b'][pA,hC], optimize=True)
    O['bb'][C,C,C,A] += scale * +0.50000000 * np.einsum('IJKA,AU->IJKU', h['bb'][C,C,C,V], t['b'][pV,hA], optimize=True)
    O['bb'][C,C,C,A] += scale * -0.50000000 * np.einsum('IJUA,AK->IJKU', h['bb'][C,C,A,V], t['b'][pV,hC], optimize=True)
    O['bb'][A,V,C,A] += scale * -1.00000000 * np.einsum('UV,WUIX,AV->WAIX', eta1['b'], h['bb'][A,A,C,A], t['b'][pV,hA], optimize=True)
    
    O['bb'][A,V,C,A] += scale * -1.00000000 * np.einsum('UV,WAXV,UI->WAIX', eta1['b'], h['bb'][A,V,A,A], t['b'][pA,hC], optimize=True)
    
    O['bb'][A,V,C,A] += scale * -1.00000000 * np.einsum('UV,WUIX,AV->WAIX', gamma1['b'], h['bb'][A,A,C,A], t['b'][pV,hA], optimize=True)
    
    O['bb'][A,V,C,A] += scale * -1.00000000 * np.einsum('UV,WAXV,UI->WAIX', gamma1['b'], h['bb'][A,V,A,A], t['b'][pA,hC], optimize=True)
    
    O['bb'][A,V,C,A] += scale * +1.00000000 * np.einsum('IUJV,AI->UAJV', h['bb'][C,A,C,A], t['b'][pV,hC], optimize=True)
    O['bb'][A,V,C,A] += scale * -1.00000000 * np.einsum('IAJU,VI->VAJU', h['bb'][C,V,C,A], t['b'][pA,hC], optimize=True)
    O['bb'][A,V,C,A] += scale * +1.00000000 * np.einsum('UAIB,BV->UAIV', h['bb'][A,V,C,V], t['b'][pV,hA], optimize=True)
    O['bb'][A,V,C,A] += scale * -1.00000000 * np.einsum('UAVB,BI->UAIV', h['bb'][A,V,A,V], t['b'][pV,hC], optimize=True)
    O['bb'][C,V,C,A] += scale * -1.00000000 * np.einsum('UV,IUJW,AV->IAJW', eta1['b'], h['bb'][C,A,C,A], t['b'][pV,hA], optimize=True)
    
    O['bb'][C,V,C,A] += scale * -1.00000000 * np.einsum('UV,IAWV,UJ->IAJW', eta1['b'], h['bb'][C,V,A,A], t['b'][pA,hC], optimize=True)
    O['bb'][C,V,C,A] += scale * -1.00000000 * np.einsum('UV,IUJW,AV->IAJW', gamma1['b'], h['bb'][C,A,C,A], t['b'][pV,hA], optimize=True)
    
    O['bb'][C,V,C,A] += scale * -1.00000000 * np.einsum('UV,IAWV,UJ->IAJW', gamma1['b'], h['bb'][C,V,A,A], t['b'][pA,hC], optimize=True)
    O['bb'][C,V,C,A] += scale * -1.00000000 * np.einsum('IJKU,AJ->IAKU', h['bb'][C,C,C,A], t['b'][pV,hC], optimize=True)
    O['bb'][C,V,C,A] += scale * +1.00000000 * np.einsum('IAJB,BU->IAJU', h['bb'][C,V,C,V], t['b'][pV,hA], optimize=True)
    O['bb'][C,V,C,A] += scale * -1.00000000 * np.einsum('IAUB,BJ->IAJU', h['bb'][C,V,A,V], t['b'][pV,hC], optimize=True)
    O['bb'][V,V,C,A] += scale * +1.00000000 * np.einsum('UV,UAIW,BV->ABIW', eta1['b'], h['bb'][A,V,C,A], t['b'][pV,hA], optimize=True)
    
    O['bb'][V,V,C,A] += scale * -0.50000000 * np.einsum('UV,ABWV,UI->ABIW', eta1['b'], h['bb'][V,V,A,A], t['b'][pA,hC], optimize=True)
    O['bb'][V,V,C,A] += scale * +1.00000000 * np.einsum('UV,UAIW,BV->ABIW', gamma1['b'], h['bb'][A,V,C,A], t['b'][pV,hA], optimize=True)
    
    O['bb'][V,V,C,A] += scale * -0.50000000 * np.einsum('UV,ABWV,UI->ABIW', gamma1['b'], h['bb'][V,V,A,A], t['b'][pA,hC], optimize=True)
    O['bb'][V,V,C,A] += scale * +1.00000000 * np.einsum('IAJU,BI->ABJU', h['bb'][C,V,C,A], t['b'][pV,hC], optimize=True)
    O['bb'][V,V,C,A] += scale * +0.50000000 * np.einsum('ABIC,CU->ABIU', h['bb'][V,V,C,V], t['b'][pV,hA], optimize=True)
    O['bb'][V,V,C,A] += scale * -0.50000000 * np.einsum('ABUC,CI->ABIU', h['bb'][V,V,A,V], t['b'][pV,hC], optimize=True)
    O['bb'][A,A,C,C] += scale * +0.50000000 * np.einsum('UV,WXIV,UJ->WXIJ', eta1['b'], h['bb'][A,A,C,A], t['b'][pA,hC], optimize=True)
    
    O['bb'][A,A,C,C] += scale * +0.50000000 * np.einsum('UV,WXIV,UJ->WXIJ', gamma1['b'], h['bb'][A,A,C,A], t['b'][pA,hC], optimize=True)
    
    O['bb'][A,A,C,C] += scale * +0.50000000 * np.einsum('IUJK,VI->UVJK', h['bb'][C,A,C,C], t['b'][pA,hC], optimize=True)
    O['bb'][A,A,C,C] += scale * +0.50000000 * np.einsum('UVIA,AJ->UVIJ', h['bb'][A,A,C,V], t['b'][pV,hC], optimize=True)
    O['bb'][C,A,C,C] += scale * +1.00000000 * np.einsum('UV,IWJV,UK->IWJK', eta1['b'], h['bb'][C,A,C,A], t['b'][pA,hC], optimize=True)
    
    O['bb'][C,A,C,C] += scale * +1.00000000 * np.einsum('UV,IWJV,UK->IWJK', gamma1['b'], h['bb'][C,A,C,A], t['b'][pA,hC], optimize=True)
    
    O['bb'][C,A,C,C] += scale * -0.50000000 * np.einsum('IJKL,UJ->IUKL', h['bb'][C,C,C,C], t['b'][pA,hC], optimize=True)
    O['bb'][C,A,C,C] += scale * +1.00000000 * np.einsum('IUJA,AK->IUJK', h['bb'][C,A,C,V], t['b'][pV,hC], optimize=True)
    O['bb'][C,C,C,C] += scale * +0.50000000 * np.einsum('UV,IJKV,UL->IJKL', eta1['b'], h['bb'][C,C,C,A], t['b'][pA,hC], optimize=True)
    O['bb'][C,C,C,C] += scale * +0.50000000 * np.einsum('UV,IJKV,UL->IJKL', gamma1['b'], h['bb'][C,C,C,A], t['b'][pA,hC], optimize=True)
    O['bb'][C,C,C,C] += scale * +0.50000000 * np.einsum('IJKA,AL->IJKL', h['bb'][C,C,C,V], t['b'][pV,hC], optimize=True)
    O['bb'][A,V,C,C] += scale * -0.50000000 * np.einsum('UV,WUIJ,AV->WAIJ', eta1['b'], h['bb'][A,A,C,C], t['b'][pV,hA], optimize=True)
    O['bb'][A,V,C,C] += scale * +1.00000000 * np.einsum('UV,WAIV,UJ->WAIJ', eta1['b'], h['bb'][A,V,C,A], t['b'][pA,hC], optimize=True)
    
    O['bb'][A,V,C,C] += scale * -0.50000000 * np.einsum('UV,WUIJ,AV->WAIJ', gamma1['b'], h['bb'][A,A,C,C], t['b'][pV,hA], optimize=True)
    O['bb'][A,V,C,C] += scale * +1.00000000 * np.einsum('UV,WAIV,UJ->WAIJ', gamma1['b'], h['bb'][A,V,C,A], t['b'][pA,hC], optimize=True)
    
    O['bb'][A,V,C,C] += scale * +0.50000000 * np.einsum('IUJK,AI->UAJK', h['bb'][C,A,C,C], t['b'][pV,hC], optimize=True)
    O['bb'][A,V,C,C] += scale * -0.50000000 * np.einsum('IAJK,UI->UAJK', h['bb'][C,V,C,C], t['b'][pA,hC], optimize=True)
    O['bb'][A,V,C,C] += scale * +1.00000000 * np.einsum('UAIB,BJ->UAIJ', h['bb'][A,V,C,V], t['b'][pV,hC], optimize=True)
    O['bb'][C,V,C,C] += scale * -0.50000000 * np.einsum('UV,IUJK,AV->IAJK', eta1['b'], h['bb'][C,A,C,C], t['b'][pV,hA], optimize=True)
    O['bb'][C,V,C,C] += scale * +1.00000000 * np.einsum('UV,IAJV,UK->IAJK', eta1['b'], h['bb'][C,V,C,A], t['b'][pA,hC], optimize=True)
    O['bb'][C,V,C,C] += scale * -0.50000000 * np.einsum('UV,IUJK,AV->IAJK', gamma1['b'], h['bb'][C,A,C,C], t['b'][pV,hA], optimize=True)
    O['bb'][C,V,C,C] += scale * +1.00000000 * np.einsum('UV,IAJV,UK->IAJK', gamma1['b'], h['bb'][C,V,C,A], t['b'][pA,hC], optimize=True)
    O['bb'][C,V,C,C] += scale * -0.50000000 * np.einsum('IJKL,AJ->IAKL', h['bb'][C,C,C,C], t['b'][pV,hC], optimize=True)
    O['bb'][C,V,C,C] += scale * +1.00000000 * np.einsum('IAJB,BK->IAJK', h['bb'][C,V,C,V], t['b'][pV,hC], optimize=True)
    O['bb'][V,V,C,C] += scale * +0.50000000 * np.einsum('UV,UAIJ,BV->ABIJ', eta1['b'], h['bb'][A,V,C,C], t['b'][pV,hA], optimize=True)
    O['bb'][V,V,C,C] += scale * +0.50000000 * np.einsum('UV,ABIV,UJ->ABIJ', eta1['b'], h['bb'][V,V,C,A], t['b'][pA,hC], optimize=True)
    O['bb'][V,V,C,C] += scale * +0.50000000 * np.einsum('UV,UAIJ,BV->ABIJ', gamma1['b'], h['bb'][A,V,C,C], t['b'][pV,hA], optimize=True)
    O['bb'][V,V,C,C] += scale * +0.50000000 * np.einsum('UV,ABIV,UJ->ABIJ', gamma1['b'], h['bb'][V,V,C,A], t['b'][pA,hC], optimize=True)
    O['bb'][V,V,C,C] += scale * +0.50000000 * np.einsum('IAJK,BI->ABJK', h['bb'][C,V,C,C], t['b'][pV,hC], optimize=True)
    O['bb'][V,V,C,C] += scale * +0.50000000 * np.einsum('ABIC,CJ->ABIJ', h['bb'][V,V,C,V], t['b'][pV,hC], optimize=True)
    O['bb'][A,A,C,V] += scale * +0.50000000 * np.einsum('UV,WXVA,UI->WXIA', eta1['b'], h['bb'][A,A,A,V], t['b'][pA,hC], optimize=True)
    
    O['bb'][A,A,C,V] += scale * +0.50000000 * np.einsum('UV,WXVA,UI->WXIA', gamma1['b'], h['bb'][A,A,A,V], t['b'][pA,hC], optimize=True)
    
    O['bb'][A,A,C,V] += scale * +1.00000000 * np.einsum('IUJA,VI->UVJA', h['bb'][C,A,C,V], t['b'][pA,hC], optimize=True)
    O['bb'][A,A,C,V] += scale * -0.50000000 * np.einsum('UVAB,BI->UVIA', h['bb'][A,A,V,V], t['b'][pV,hC], optimize=True)
    O['bb'][C,A,C,V] += scale * +1.00000000 * np.einsum('UV,IWVA,UJ->IWJA', eta1['b'], h['bb'][C,A,A,V], t['b'][pA,hC], optimize=True)
    
    O['bb'][C,A,C,V] += scale * +1.00000000 * np.einsum('UV,IWVA,UJ->IWJA', gamma1['b'], h['bb'][C,A,A,V], t['b'][pA,hC], optimize=True)
    
    O['bb'][C,A,C,V] += scale * -1.00000000 * np.einsum('IJKA,UJ->IUKA', h['bb'][C,C,C,V], t['b'][pA,hC], optimize=True)
    O['bb'][C,A,C,V] += scale * -1.00000000 * np.einsum('IUAB,BJ->IUJA', h['bb'][C,A,V,V], t['b'][pV,hC], optimize=True)
    O['bb'][C,C,C,V] += scale * +0.50000000 * np.einsum('UV,IJVA,UK->IJKA', eta1['b'], h['bb'][C,C,A,V], t['b'][pA,hC], optimize=True)
    O['bb'][C,C,C,V] += scale * +0.50000000 * np.einsum('UV,IJVA,UK->IJKA', gamma1['b'], h['bb'][C,C,A,V], t['b'][pA,hC], optimize=True)
    O['bb'][C,C,C,V] += scale * -0.50000000 * np.einsum('IJAB,BK->IJKA', h['bb'][C,C,V,V], t['b'][pV,hC], optimize=True)
    O['bb'][A,V,C,V] += scale * -1.00000000 * np.einsum('UV,WUIA,BV->WBIA', eta1['b'], h['bb'][A,A,C,V], t['b'][pV,hA], optimize=True)
    O['bb'][A,V,C,V] += scale * +1.00000000 * np.einsum('UV,WAVB,UI->WAIB', eta1['b'], h['bb'][A,V,A,V], t['b'][pA,hC], optimize=True)
    
    O['bb'][A,V,C,V] += scale * -1.00000000 * np.einsum('UV,WUIA,BV->WBIA', gamma1['b'], h['bb'][A,A,C,V], t['b'][pV,hA], optimize=True)
    O['bb'][A,V,C,V] += scale * +1.00000000 * np.einsum('UV,WAVB,UI->WAIB', gamma1['b'], h['bb'][A,V,A,V], t['b'][pA,hC], optimize=True)
    
    O['bb'][A,V,C,V] += scale * +1.00000000 * np.einsum('IUJA,BI->UBJA', h['bb'][C,A,C,V], t['b'][pV,hC], optimize=True)
    O['bb'][A,V,C,V] += scale * -1.00000000 * np.einsum('IAJB,UI->UAJB', h['bb'][C,V,C,V], t['b'][pA,hC], optimize=True)
    O['bb'][A,V,C,V] += scale * -1.00000000 * np.einsum('UABC,CI->UAIB', h['bb'][A,V,V,V], t['b'][pV,hC], optimize=True)
    O['bb'][C,V,C,V] += scale * -1.00000000 * np.einsum('UV,IUJA,BV->IBJA', eta1['b'], h['bb'][C,A,C,V], t['b'][pV,hA], optimize=True)
    O['bb'][C,V,C,V] += scale * +1.00000000 * np.einsum('UV,IAVB,UJ->IAJB', eta1['b'], h['bb'][C,V,A,V], t['b'][pA,hC], optimize=True)
    O['bb'][C,V,C,V] += scale * -1.00000000 * np.einsum('UV,IUJA,BV->IBJA', gamma1['b'], h['bb'][C,A,C,V], t['b'][pV,hA], optimize=True)
    O['bb'][C,V,C,V] += scale * +1.00000000 * np.einsum('UV,IAVB,UJ->IAJB', gamma1['b'], h['bb'][C,V,A,V], t['b'][pA,hC], optimize=True)
    O['bb'][C,V,C,V] += scale * -1.00000000 * np.einsum('IJKA,BJ->IBKA', h['bb'][C,C,C,V], t['b'][pV,hC], optimize=True)
    O['bb'][C,V,C,V] += scale * -1.00000000 * np.einsum('IABC,CJ->IAJB', h['bb'][C,V,V,V], t['b'][pV,hC], optimize=True)
    O['bb'][V,V,C,V] += scale * +1.00000000 * np.einsum('UV,UAIB,CV->ACIB', eta1['b'], h['bb'][A,V,C,V], t['b'][pV,hA], optimize=True)
    O['bb'][V,V,C,V] += scale * +0.50000000 * np.einsum('UV,ABVC,UI->ABIC', eta1['b'], h['bb'][V,V,A,V], t['b'][pA,hC], optimize=True)
    O['bb'][V,V,C,V] += scale * +1.00000000 * np.einsum('UV,UAIB,CV->ACIB', gamma1['b'], h['bb'][A,V,C,V], t['b'][pV,hA], optimize=True)
    O['bb'][V,V,C,V] += scale * +0.50000000 * np.einsum('UV,ABVC,UI->ABIC', gamma1['b'], h['bb'][V,V,A,V], t['b'][pA,hC], optimize=True)
    O['bb'][V,V,C,V] += scale * +1.00000000 * np.einsum('IAJB,CI->ACJB', h['bb'][C,V,C,V], t['b'][pV,hC], optimize=True)
    O['bb'][V,V,C,V] += scale * -0.50000000 * np.einsum('ABCD,DI->ABIC', h['bb'][V,V,V,V], t['b'][pV,hC], optimize=True)
    
    
    O['bb'][A,A,V,V] += scale * +0.50000000 * np.einsum('IUAB,VI->UVAB', h['bb'][C,A,V,V], t['b'][pA,hC], optimize=True)
    
    
    O['bb'][C,A,V,V] += scale * -0.50000000 * np.einsum('IJAB,UJ->IUAB', h['bb'][C,C,V,V], t['b'][pA,hC], optimize=True)
    O['bb'][A,V,V,V] += scale * -0.50000000 * np.einsum('UV,WUAB,CV->WCAB', eta1['b'], h['bb'][A,A,V,V], t['b'][pV,hA], optimize=True)
    
    O['bb'][A,V,V,V] += scale * -0.50000000 * np.einsum('UV,WUAB,CV->WCAB', gamma1['b'], h['bb'][A,A,V,V], t['b'][pV,hA], optimize=True)
    
    O['bb'][A,V,V,V] += scale * +0.50000000 * np.einsum('IUAB,CI->UCAB', h['bb'][C,A,V,V], t['b'][pV,hC], optimize=True)
    O['bb'][A,V,V,V] += scale * -0.50000000 * np.einsum('IABC,UI->UABC', h['bb'][C,V,V,V], t['b'][pA,hC], optimize=True)
    O['bb'][C,V,V,V] += scale * -0.50000000 * np.einsum('UV,IUAB,CV->ICAB', eta1['b'], h['bb'][C,A,V,V], t['b'][pV,hA], optimize=True)
    O['bb'][C,V,V,V] += scale * -0.50000000 * np.einsum('UV,IUAB,CV->ICAB', gamma1['b'], h['bb'][C,A,V,V], t['b'][pV,hA], optimize=True)
    O['bb'][C,V,V,V] += scale * -0.50000000 * np.einsum('IJAB,CJ->ICAB', h['bb'][C,C,V,V], t['b'][pV,hC], optimize=True)
    O['bb'][V,V,V,V] += scale * +0.50000000 * np.einsum('UV,UABC,DV->ADBC', eta1['b'], h['bb'][A,V,V,V], t['b'][pV,hA], optimize=True)
    O['bb'][V,V,V,V] += scale * +0.50000000 * np.einsum('UV,UABC,DV->ADBC', gamma1['b'], h['bb'][A,V,V,V], t['b'][pV,hA], optimize=True)
    O['bb'][V,V,V,V] += scale * +0.50000000 * np.einsum('IABC,DI->ADBC', h['bb'][C,V,V,V], t['b'][pV,hC], optimize=True)

    t1 = time.time()
    if verbose: print("h2c_t1b_c2c took {:.4f} seconds to run.".format(t1-t0))

    return O
def h2c_t2c_c2c(O, h, t, gamma1, eta1, lambdas, orbspace, verbose=False, scale=1.0):
    # 324 lines
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

    
    
    
    
    O['bb'][A,A,A,A] += scale * +1.00000000 * np.einsum('UV,IWXV,YUIZ->WYXZ', eta1['b'], h['bb'][C,A,A,A], t['bb'][pA,pA,hC,hA], optimize=True)
    O['bb'][A,A,A,A] += scale * +0.25000000 * np.einsum('UV,WXVA,UAYZ->WXYZ', eta1['b'], h['bb'][A,A,A,V], t['bb'][pA,pV,hA,hA], optimize=True)
    
    
    O['bb'][A,A,A,A] += scale * +0.25000000 * np.einsum('UV,IUWX,YZIV->YZWX', gamma1['b'], h['bb'][C,A,A,A], t['bb'][pA,pA,hC,hA], optimize=True)
    O['bb'][A,A,A,A] += scale * +1.00000000 * np.einsum('UV,WUXA,YAZV->WYXZ', gamma1['b'], h['bb'][A,A,A,V], t['bb'][pA,pV,hA,hA], optimize=True)
    O['bb'][A,A,A,A] += scale * +0.12500000 * np.einsum('IJUV,WXIJ->WXUV', h['bb'][C,C,A,A], t['bb'][pA,pA,hC,hC], optimize=True)
    O['bb'][A,A,A,A] += scale * +1.00000000 * np.einsum('IUVA,WAIX->UWVX', h['bb'][C,A,A,V], t['bb'][pA,pV,hC,hA], optimize=True)
    O['bb'][A,A,A,A] += scale * +0.12500000 * np.einsum('UVAB,ABWX->UVWX', h['bb'][A,A,V,V], t['bb'][pV,pV,hA,hA], optimize=True)
    
    
    
    O['bb'][C,A,A,A] += scale * -1.00000000 * np.einsum('UV,IJWV,XUJY->IXWY', eta1['b'], h['bb'][C,C,A,A], t['bb'][pA,pA,hC,hA], optimize=True)
    O['bb'][C,A,A,A] += scale * +0.50000000 * np.einsum('UV,IWVA,UAXY->IWXY', eta1['b'], h['bb'][C,A,A,V], t['bb'][pA,pV,hA,hA], optimize=True)
    
    O['bb'][C,A,A,A] += scale * +1.00000000 * np.einsum('UV,IUWA,XAYV->IXWY', gamma1['b'], h['bb'][C,A,A,V], t['bb'][pA,pV,hA,hA], optimize=True)
    O['bb'][C,A,A,A] += scale * -1.00000000 * np.einsum('IJUA,VAJW->IVUW', h['bb'][C,C,A,V], t['bb'][pA,pV,hC,hA], optimize=True)
    O['bb'][C,A,A,A] += scale * +0.25000000 * np.einsum('IUAB,ABVW->IUVW', h['bb'][C,A,V,V], t['bb'][pV,pV,hA,hA], optimize=True)
    
    O['bb'][C,C,A,A] += scale * +0.25000000 * np.einsum('UV,IJVA,UAWX->IJWX', eta1['b'], h['bb'][C,C,A,V], t['bb'][pA,pV,hA,hA], optimize=True)
    
    O['bb'][C,C,A,A] += scale * +0.12500000 * np.einsum('IJAB,ABUV->IJUV', h['bb'][C,C,V,V], t['bb'][pV,pV,hA,hA], optimize=True)
    
    O['bb'][A,V,A,A] += scale * -0.25000000 * np.einsum('UV,WX,UWYZ,RAVX->RAYZ', eta1['b'], eta1['b'], h['bb'][A,A,A,A], t['bb'][pA,pV,hA,hA], optimize=True)
    O['bb'][A,V,A,A] += scale * +1.00000000 * np.einsum('UV,WX,YUZX,WARV->YAZR', eta1['b'], gamma1['b'], h['bb'][A,A,A,A], t['bb'][pA,pV,hA,hA], optimize=True)
    O['bb'][A,V,A,A] += scale * -1.00000000 * np.einsum('UV,WX,YWZV,UARX->YAZR', eta1['b'], gamma1['b'], h['bb'][A,A,A,A], t['bb'][pA,pV,hA,hA], optimize=True)
    
    
    O['bb'][A,V,A,A] += scale * -1.00000000 * np.einsum('UV,IWXV,UAIY->WAXY', eta1['b'], h['bb'][C,A,A,A], t['bb'][pA,pV,hC,hA], optimize=True)
    O['bb'][A,V,A,A] += scale * -1.00000000 * np.einsum('UV,IAWV,XUIY->XAWY', eta1['b'], h['bb'][C,V,A,A], t['bb'][pA,pA,hC,hA], optimize=True)
    O['bb'][A,V,A,A] += scale * +0.50000000 * np.einsum('UV,WAVB,UBXY->WAXY', eta1['b'], h['bb'][A,V,A,V], t['bb'][pA,pV,hA,hA], optimize=True)
    
    O['bb'][A,V,A,A] += scale * +0.25000000 * np.einsum('UV,WX,UWYZ,RAVX->RAYZ', gamma1['b'], gamma1['b'], h['bb'][A,A,A,A], t['bb'][pA,pV,hA,hA], optimize=True)
    O['bb'][A,V,A,A] += scale * +0.50000000 * np.einsum('UV,IUWX,YAIV->YAWX', gamma1['b'], h['bb'][C,A,A,A], t['bb'][pA,pV,hC,hA], optimize=True)
    O['bb'][A,V,A,A] += scale * +1.00000000 * np.einsum('UV,WUXA,BAYV->WBXY', gamma1['b'], h['bb'][A,A,A,V], t['bb'][pV,pV,hA,hA], optimize=True)
    O['bb'][A,V,A,A] += scale * +1.00000000 * np.einsum('UV,UAWB,XBYV->XAWY', gamma1['b'], h['bb'][A,V,A,V], t['bb'][pA,pV,hA,hA], optimize=True)
    O['bb'][A,V,A,A] += scale * +0.25000000 * np.einsum('IJUV,WAIJ->WAUV', h['bb'][C,C,A,A], t['bb'][pA,pV,hC,hC], optimize=True)
    O['bb'][A,V,A,A] += scale * +1.00000000 * np.einsum('IUVA,BAIW->UBVW', h['bb'][C,A,A,V], t['bb'][pV,pV,hC,hA], optimize=True)
    O['bb'][A,V,A,A] += scale * -1.00000000 * np.einsum('IAUB,VBIW->VAUW', h['bb'][C,V,A,V], t['bb'][pA,pV,hC,hA], optimize=True)
    O['bb'][A,V,A,A] += scale * +0.25000000 * np.einsum('UABC,BCVW->UAVW', h['bb'][A,V,V,V], t['bb'][pV,pV,hA,hA], optimize=True)
    
    O['bb'][C,V,A,A] += scale * +1.00000000 * np.einsum('UV,WX,IUYX,WAZV->IAYZ', eta1['b'], gamma1['b'], h['bb'][C,A,A,A], t['bb'][pA,pV,hA,hA], optimize=True)
    O['bb'][C,V,A,A] += scale * -1.00000000 * np.einsum('UV,WX,IWYV,UAZX->IAYZ', eta1['b'], gamma1['b'], h['bb'][C,A,A,A], t['bb'][pA,pV,hA,hA], optimize=True)
    O['bb'][C,V,A,A] += scale * +1.00000000 * np.einsum('UV,IJWV,UAJX->IAWX', eta1['b'], h['bb'][C,C,A,A], t['bb'][pA,pV,hC,hA], optimize=True)
    O['bb'][C,V,A,A] += scale * +0.50000000 * np.einsum('UV,IAVB,UBWX->IAWX', eta1['b'], h['bb'][C,V,A,V], t['bb'][pA,pV,hA,hA], optimize=True)
    
    O['bb'][C,V,A,A] += scale * +1.00000000 * np.einsum('UV,IUWA,BAXV->IBWX', gamma1['b'], h['bb'][C,A,A,V], t['bb'][pV,pV,hA,hA], optimize=True)
    O['bb'][C,V,A,A] += scale * -1.00000000 * np.einsum('IJUA,BAJV->IBUV', h['bb'][C,C,A,V], t['bb'][pV,pV,hC,hA], optimize=True)
    O['bb'][C,V,A,A] += scale * +0.25000000 * np.einsum('IABC,BCUV->IAUV', h['bb'][C,V,V,V], t['bb'][pV,pV,hA,hA], optimize=True)
    O['bb'][V,V,A,A] += scale * -0.12500000 * np.einsum('UV,WX,UWYZ,ABVX->ABYZ', eta1['b'], eta1['b'], h['bb'][A,A,A,A], t['bb'][pV,pV,hA,hA], optimize=True)
    
    O['bb'][V,V,A,A] += scale * -1.00000000 * np.einsum('UV,WX,UAYX,WBZV->ABYZ', eta1['b'], gamma1['b'], h['bb'][A,V,A,A], t['bb'][pA,pV,hA,hA], optimize=True)
    O['bb'][V,V,A,A] += scale * +1.00000000 * np.einsum('UV,WX,WAYV,UBZX->ABYZ', eta1['b'], gamma1['b'], h['bb'][A,V,A,A], t['bb'][pA,pV,hA,hA], optimize=True)
    O['bb'][V,V,A,A] += scale * -1.00000000 * np.einsum('UV,IAWV,UBIX->ABWX', eta1['b'], h['bb'][C,V,A,A], t['bb'][pA,pV,hC,hA], optimize=True)
    O['bb'][V,V,A,A] += scale * +0.25000000 * np.einsum('UV,ABVC,UCWX->ABWX', eta1['b'], h['bb'][V,V,A,V], t['bb'][pA,pV,hA,hA], optimize=True)
    O['bb'][V,V,A,A] += scale * +0.12500000 * np.einsum('UV,WX,UWYZ,ABVX->ABYZ', gamma1['b'], gamma1['b'], h['bb'][A,A,A,A], t['bb'][pV,pV,hA,hA], optimize=True)
    
    O['bb'][V,V,A,A] += scale * +0.25000000 * np.einsum('UV,IUWX,ABIV->ABWX', gamma1['b'], h['bb'][C,A,A,A], t['bb'][pV,pV,hC,hA], optimize=True)
    O['bb'][V,V,A,A] += scale * -1.00000000 * np.einsum('UV,UAWB,CBXV->ACWX', gamma1['b'], h['bb'][A,V,A,V], t['bb'][pV,pV,hA,hA], optimize=True)
    O['bb'][V,V,A,A] += scale * +0.12500000 * np.einsum('IJUV,ABIJ->ABUV', h['bb'][C,C,A,A], t['bb'][pV,pV,hC,hC], optimize=True)
    O['bb'][V,V,A,A] += scale * +1.00000000 * np.einsum('IAUB,CBIV->ACUV', h['bb'][C,V,A,V], t['bb'][pV,pV,hC,hA], optimize=True)
    O['bb'][V,V,A,A] += scale * +0.12500000 * np.einsum('ABCD,CDUV->ABUV', h['bb'][V,V,V,V], t['bb'][pV,pV,hA,hA], optimize=True)
    
    
    
    O['bb'][A,A,A,V] += scale * +1.00000000 * np.einsum('UV,IWVA,XUIY->WXYA', eta1['b'], h['bb'][C,A,A,V], t['bb'][pA,pA,hC,hA], optimize=True)
    
    O['bb'][A,A,A,V] += scale * +0.50000000 * np.einsum('UV,IUWA,XYIV->XYWA', gamma1['b'], h['bb'][C,A,A,V], t['bb'][pA,pA,hC,hA], optimize=True)
    O['bb'][A,A,A,V] += scale * -1.00000000 * np.einsum('UV,WUAB,XBYV->WXYA', gamma1['b'], h['bb'][A,A,V,V], t['bb'][pA,pV,hA,hA], optimize=True)
    O['bb'][A,A,A,V] += scale * +0.25000000 * np.einsum('IJUA,VWIJ->VWUA', h['bb'][C,C,A,V], t['bb'][pA,pA,hC,hC], optimize=True)
    O['bb'][A,A,A,V] += scale * -1.00000000 * np.einsum('IUAB,VBIW->UVWA', h['bb'][C,A,V,V], t['bb'][pA,pV,hC,hA], optimize=True)
    
    
    O['bb'][C,A,A,V] += scale * -1.00000000 * np.einsum('UV,IJVA,WUJX->IWXA', eta1['b'], h['bb'][C,C,A,V], t['bb'][pA,pA,hC,hA], optimize=True)
    O['bb'][C,A,A,V] += scale * -1.00000000 * np.einsum('UV,IUAB,WBXV->IWXA', gamma1['b'], h['bb'][C,A,V,V], t['bb'][pA,pV,hA,hA], optimize=True)
    O['bb'][C,A,A,V] += scale * +1.00000000 * np.einsum('IJAB,UBJV->IUVA', h['bb'][C,C,V,V], t['bb'][pA,pV,hC,hA], optimize=True)
    O['bb'][A,V,A,V] += scale * -0.50000000 * np.einsum('UV,WX,UWYA,ZBVX->ZBYA', eta1['b'], eta1['b'], h['bb'][A,A,A,V], t['bb'][pA,pV,hA,hA], optimize=True)
    O['bb'][A,V,A,V] += scale * +1.00000000 * np.einsum('UV,WX,YUXA,WBZV->YBZA', eta1['b'], gamma1['b'], h['bb'][A,A,A,V], t['bb'][pA,pV,hA,hA], optimize=True)
    O['bb'][A,V,A,V] += scale * -1.00000000 * np.einsum('UV,WX,YWVA,UBZX->YBZA', eta1['b'], gamma1['b'], h['bb'][A,A,A,V], t['bb'][pA,pV,hA,hA], optimize=True)
    
    
    O['bb'][A,V,A,V] += scale * -1.00000000 * np.einsum('UV,IWVA,UBIX->WBXA', eta1['b'], h['bb'][C,A,A,V], t['bb'][pA,pV,hC,hA], optimize=True)
    O['bb'][A,V,A,V] += scale * -1.00000000 * np.einsum('UV,IAVB,WUIX->WAXB', eta1['b'], h['bb'][C,V,A,V], t['bb'][pA,pA,hC,hA], optimize=True)
    O['bb'][A,V,A,V] += scale * +0.50000000 * np.einsum('UV,WX,UWYA,ZBVX->ZBYA', gamma1['b'], gamma1['b'], h['bb'][A,A,A,V], t['bb'][pA,pV,hA,hA], optimize=True)
    O['bb'][A,V,A,V] += scale * +1.00000000 * np.einsum('UV,IUWA,XBIV->XBWA', gamma1['b'], h['bb'][C,A,A,V], t['bb'][pA,pV,hC,hA], optimize=True)
    O['bb'][A,V,A,V] += scale * -1.00000000 * np.einsum('UV,WUAB,CBXV->WCXA', gamma1['b'], h['bb'][A,A,V,V], t['bb'][pV,pV,hA,hA], optimize=True)
    O['bb'][A,V,A,V] += scale * -1.00000000 * np.einsum('UV,UABC,WCXV->WAXB', gamma1['b'], h['bb'][A,V,V,V], t['bb'][pA,pV,hA,hA], optimize=True)
    O['bb'][A,V,A,V] += scale * +0.50000000 * np.einsum('IJUA,VBIJ->VBUA', h['bb'][C,C,A,V], t['bb'][pA,pV,hC,hC], optimize=True)
    O['bb'][A,V,A,V] += scale * -1.00000000 * np.einsum('IUAB,CBIV->UCVA', h['bb'][C,A,V,V], t['bb'][pV,pV,hC,hA], optimize=True)
    O['bb'][A,V,A,V] += scale * +1.00000000 * np.einsum('IABC,UCIV->UAVB', h['bb'][C,V,V,V], t['bb'][pA,pV,hC,hA], optimize=True)
    O['bb'][C,V,A,V] += scale * +1.00000000 * np.einsum('UV,WX,IUXA,WBYV->IBYA', eta1['b'], gamma1['b'], h['bb'][C,A,A,V], t['bb'][pA,pV,hA,hA], optimize=True)
    O['bb'][C,V,A,V] += scale * -1.00000000 * np.einsum('UV,WX,IWVA,UBYX->IBYA', eta1['b'], gamma1['b'], h['bb'][C,A,A,V], t['bb'][pA,pV,hA,hA], optimize=True)
    O['bb'][C,V,A,V] += scale * +1.00000000 * np.einsum('UV,IJVA,UBJW->IBWA', eta1['b'], h['bb'][C,C,A,V], t['bb'][pA,pV,hC,hA], optimize=True)
    O['bb'][C,V,A,V] += scale * -1.00000000 * np.einsum('UV,IUAB,CBWV->ICWA', gamma1['b'], h['bb'][C,A,V,V], t['bb'][pV,pV,hA,hA], optimize=True)
    O['bb'][C,V,A,V] += scale * +1.00000000 * np.einsum('IJAB,CBJU->ICUA', h['bb'][C,C,V,V], t['bb'][pV,pV,hC,hA], optimize=True)
    O['bb'][V,V,A,V] += scale * -0.25000000 * np.einsum('UV,WX,UWYA,BCVX->BCYA', eta1['b'], eta1['b'], h['bb'][A,A,A,V], t['bb'][pV,pV,hA,hA], optimize=True)
    O['bb'][V,V,A,V] += scale * -1.00000000 * np.einsum('UV,WX,UAXB,WCYV->ACYB', eta1['b'], gamma1['b'], h['bb'][A,V,A,V], t['bb'][pA,pV,hA,hA], optimize=True)
    O['bb'][V,V,A,V] += scale * +1.00000000 * np.einsum('UV,WX,WAVB,UCYX->ACYB', eta1['b'], gamma1['b'], h['bb'][A,V,A,V], t['bb'][pA,pV,hA,hA], optimize=True)
    O['bb'][V,V,A,V] += scale * -1.00000000 * np.einsum('UV,IAVB,UCIW->ACWB', eta1['b'], h['bb'][C,V,A,V], t['bb'][pA,pV,hC,hA], optimize=True)
    O['bb'][V,V,A,V] += scale * +0.25000000 * np.einsum('UV,WX,UWYA,BCVX->BCYA', gamma1['b'], gamma1['b'], h['bb'][A,A,A,V], t['bb'][pV,pV,hA,hA], optimize=True)
    O['bb'][V,V,A,V] += scale * +0.50000000 * np.einsum('UV,IUWA,BCIV->BCWA', gamma1['b'], h['bb'][C,A,A,V], t['bb'][pV,pV,hC,hA], optimize=True)
    O['bb'][V,V,A,V] += scale * +1.00000000 * np.einsum('UV,UABC,DCWV->ADWB', gamma1['b'], h['bb'][A,V,V,V], t['bb'][pV,pV,hA,hA], optimize=True)
    O['bb'][V,V,A,V] += scale * +0.25000000 * np.einsum('IJUA,BCIJ->BCUA', h['bb'][C,C,A,V], t['bb'][pV,pV,hC,hC], optimize=True)
    O['bb'][V,V,A,V] += scale * -1.00000000 * np.einsum('IABC,DCIU->ADUB', h['bb'][C,V,V,V], t['bb'][pV,pV,hC,hA], optimize=True)
    O['bb'][A,A,C,A] += scale * +0.25000000 * np.einsum('UV,WX,YZVX,UWIR->YZIR', eta1['b'], eta1['b'], h['bb'][A,A,A,A], t['bb'][pA,pA,hC,hA], optimize=True)
    
    
    O['bb'][A,A,C,A] += scale * +1.00000000 * np.einsum('UV,WX,YUZX,RWIV->YRIZ', eta1['b'], gamma1['b'], h['bb'][A,A,A,A], t['bb'][pA,pA,hC,hA], optimize=True)
    
    O['bb'][A,A,C,A] += scale * -1.00000000 * np.einsum('UV,WX,YWZV,RUIX->YRIZ', eta1['b'], gamma1['b'], h['bb'][A,A,A,A], t['bb'][pA,pA,hC,hA], optimize=True)
    O['bb'][A,A,C,A] += scale * +1.00000000 * np.einsum('UV,IWJV,XUIY->WXJY', eta1['b'], h['bb'][C,A,C,A], t['bb'][pA,pA,hC,hA], optimize=True)
    O['bb'][A,A,C,A] += scale * +1.00000000 * np.einsum('UV,IWXV,YUJI->WYJX', eta1['b'], h['bb'][C,A,A,A], t['bb'][pA,pA,hC,hC], optimize=True)
    O['bb'][A,A,C,A] += scale * +0.50000000 * np.einsum('UV,WXVA,UAIY->WXIY', eta1['b'], h['bb'][A,A,A,V], t['bb'][pA,pV,hC,hA], optimize=True)
    O['bb'][A,A,C,A] += scale * -0.25000000 * np.einsum('UV,WX,YZVX,UWIR->YZIR', gamma1['b'], gamma1['b'], h['bb'][A,A,A,A], t['bb'][pA,pA,hC,hA], optimize=True)
    
    O['bb'][A,A,C,A] += scale * +0.50000000 * np.einsum('UV,IUJW,XYIV->XYJW', gamma1['b'], h['bb'][C,A,C,A], t['bb'][pA,pA,hC,hA], optimize=True)
    O['bb'][A,A,C,A] += scale * +1.00000000 * np.einsum('UV,WUIA,XAYV->WXIY', gamma1['b'], h['bb'][A,A,C,V], t['bb'][pA,pV,hA,hA], optimize=True)
    O['bb'][A,A,C,A] += scale * -1.00000000 * np.einsum('UV,WUXA,YAIV->WYIX', gamma1['b'], h['bb'][A,A,A,V], t['bb'][pA,pV,hC,hA], optimize=True)
    O['bb'][A,A,C,A] += scale * +0.25000000 * np.einsum('IJKU,VWIJ->VWKU', h['bb'][C,C,C,A], t['bb'][pA,pA,hC,hC], optimize=True)
    O['bb'][A,A,C,A] += scale * +1.00000000 * np.einsum('IUJA,VAIW->UVJW', h['bb'][C,A,C,V], t['bb'][pA,pV,hC,hA], optimize=True)
    O['bb'][A,A,C,A] += scale * +1.00000000 * np.einsum('IUVA,WAJI->UWJV', h['bb'][C,A,A,V], t['bb'][pA,pV,hC,hC], optimize=True)
    O['bb'][A,A,C,A] += scale * +0.25000000 * np.einsum('UVAB,ABIW->UVIW', h['bb'][A,A,V,V], t['bb'][pV,pV,hC,hA], optimize=True)
    O['bb'][C,A,C,A] += scale * +0.50000000 * np.einsum('UV,WX,IYVX,UWJZ->IYJZ', eta1['b'], eta1['b'], h['bb'][C,A,A,A], t['bb'][pA,pA,hC,hA], optimize=True)
    
    O['bb'][C,A,C,A] += scale * +1.00000000 * np.einsum('UV,WX,IUYX,ZWJV->IZJY', eta1['b'], gamma1['b'], h['bb'][C,A,A,A], t['bb'][pA,pA,hC,hA], optimize=True)
    
    O['bb'][C,A,C,A] += scale * -1.00000000 * np.einsum('UV,WX,IWYV,ZUJX->IZJY', eta1['b'], gamma1['b'], h['bb'][C,A,A,A], t['bb'][pA,pA,hC,hA], optimize=True)
    O['bb'][C,A,C,A] += scale * -1.00000000 * np.einsum('UV,IJKV,WUJX->IWKX', eta1['b'], h['bb'][C,C,C,A], t['bb'][pA,pA,hC,hA], optimize=True)
    O['bb'][C,A,C,A] += scale * -1.00000000 * np.einsum('UV,IJWV,XUKJ->IXKW', eta1['b'], h['bb'][C,C,A,A], t['bb'][pA,pA,hC,hC], optimize=True)
    O['bb'][C,A,C,A] += scale * +1.00000000 * np.einsum('UV,IWVA,UAJX->IWJX', eta1['b'], h['bb'][C,A,A,V], t['bb'][pA,pV,hC,hA], optimize=True)
    O['bb'][C,A,C,A] += scale * -0.50000000 * np.einsum('UV,WX,IYVX,UWJZ->IYJZ', gamma1['b'], gamma1['b'], h['bb'][C,A,A,A], t['bb'][pA,pA,hC,hA], optimize=True)
    O['bb'][C,A,C,A] += scale * +1.00000000 * np.einsum('UV,IUJA,WAXV->IWJX', gamma1['b'], h['bb'][C,A,C,V], t['bb'][pA,pV,hA,hA], optimize=True)
    O['bb'][C,A,C,A] += scale * -1.00000000 * np.einsum('UV,IUWA,XAJV->IXJW', gamma1['b'], h['bb'][C,A,A,V], t['bb'][pA,pV,hC,hA], optimize=True)
    O['bb'][C,A,C,A] += scale * -1.00000000 * np.einsum('IJKA,UAJV->IUKV', h['bb'][C,C,C,V], t['bb'][pA,pV,hC,hA], optimize=True)
    O['bb'][C,A,C,A] += scale * -1.00000000 * np.einsum('IJUA,VAKJ->IVKU', h['bb'][C,C,A,V], t['bb'][pA,pV,hC,hC], optimize=True)
    O['bb'][C,A,C,A] += scale * +0.50000000 * np.einsum('IUAB,ABJV->IUJV', h['bb'][C,A,V,V], t['bb'][pV,pV,hC,hA], optimize=True)
    O['bb'][C,C,C,A] += scale * +0.25000000 * np.einsum('UV,WX,IJVX,UWKY->IJKY', eta1['b'], eta1['b'], h['bb'][C,C,A,A], t['bb'][pA,pA,hC,hA], optimize=True)
    O['bb'][C,C,C,A] += scale * +0.50000000 * np.einsum('UV,IJVA,UAKW->IJKW', eta1['b'], h['bb'][C,C,A,V], t['bb'][pA,pV,hC,hA], optimize=True)
    O['bb'][C,C,C,A] += scale * -0.25000000 * np.einsum('UV,WX,IJVX,UWKY->IJKY', gamma1['b'], gamma1['b'], h['bb'][C,C,A,A], t['bb'][pA,pA,hC,hA], optimize=True)
    O['bb'][C,C,C,A] += scale * +0.25000000 * np.einsum('IJAB,ABKU->IJKU', h['bb'][C,C,V,V], t['bb'][pV,pV,hC,hA], optimize=True)
    O['bb'][A,V,C,A] += scale * +0.50000000 * np.einsum('UV,WX,YAVX,UWIZ->YAIZ', eta1['b'], eta1['b'], h['bb'][A,V,A,A], t['bb'][pA,pA,hC,hA], optimize=True)
    O['bb'][A,V,C,A] += scale * -0.50000000 * np.einsum('UV,WX,UWIY,ZAVX->ZAIY', eta1['b'], eta1['b'], h['bb'][A,A,C,A], t['bb'][pA,pV,hA,hA], optimize=True)
    O['bb'][A,V,C,A] += scale * +1.00000000 * np.einsum('UV,WX,YUIX,WAZV->YAIZ', eta1['b'], gamma1['b'], h['bb'][A,A,C,A], t['bb'][pA,pV,hA,hA], optimize=True)
    O['bb'][A,V,C,A] += scale * -1.00000000 * np.einsum('UV,WX,YUZX,WAIV->YAIZ', eta1['b'], gamma1['b'], h['bb'][A,A,A,A], t['bb'][pA,pV,hC,hA], optimize=True)
    O['bb'][A,V,C,A] += scale * -1.00000000 * np.einsum('UV,WX,YWIV,UAZX->YAIZ', eta1['b'], gamma1['b'], h['bb'][A,A,C,A], t['bb'][pA,pV,hA,hA], optimize=True)
    O['bb'][A,V,C,A] += scale * +1.00000000 * np.einsum('UV,WX,YWZV,UAIX->YAIZ', eta1['b'], gamma1['b'], h['bb'][A,A,A,A], t['bb'][pA,pV,hC,hA], optimize=True)
    
    O['bb'][A,V,C,A] += scale * +1.00000000 * np.einsum('UV,WX,UAYX,ZWIV->ZAIY', eta1['b'], gamma1['b'], h['bb'][A,V,A,A], t['bb'][pA,pA,hC,hA], optimize=True)
    
    O['bb'][A,V,C,A] += scale * -1.00000000 * np.einsum('UV,WX,WAYV,ZUIX->ZAIY', eta1['b'], gamma1['b'], h['bb'][A,V,A,A], t['bb'][pA,pA,hC,hA], optimize=True)
    O['bb'][A,V,C,A] += scale * -1.00000000 * np.einsum('UV,IWJV,UAIX->WAJX', eta1['b'], h['bb'][C,A,C,A], t['bb'][pA,pV,hC,hA], optimize=True)
    O['bb'][A,V,C,A] += scale * -1.00000000 * np.einsum('UV,IWXV,UAJI->WAJX', eta1['b'], h['bb'][C,A,A,A], t['bb'][pA,pV,hC,hC], optimize=True)
    O['bb'][A,V,C,A] += scale * -1.00000000 * np.einsum('UV,IAJV,WUIX->WAJX', eta1['b'], h['bb'][C,V,C,A], t['bb'][pA,pA,hC,hA], optimize=True)
    O['bb'][A,V,C,A] += scale * -1.00000000 * np.einsum('UV,IAWV,XUJI->XAJW', eta1['b'], h['bb'][C,V,A,A], t['bb'][pA,pA,hC,hC], optimize=True)
    O['bb'][A,V,C,A] += scale * +1.00000000 * np.einsum('UV,WAVB,UBIX->WAIX', eta1['b'], h['bb'][A,V,A,V], t['bb'][pA,pV,hC,hA], optimize=True)
    O['bb'][A,V,C,A] += scale * -0.50000000 * np.einsum('UV,WX,YAVX,UWIZ->YAIZ', gamma1['b'], gamma1['b'], h['bb'][A,V,A,A], t['bb'][pA,pA,hC,hA], optimize=True)
    O['bb'][A,V,C,A] += scale * +0.50000000 * np.einsum('UV,WX,UWIY,ZAVX->ZAIY', gamma1['b'], gamma1['b'], h['bb'][A,A,C,A], t['bb'][pA,pV,hA,hA], optimize=True)
    O['bb'][A,V,C,A] += scale * +1.00000000 * np.einsum('UV,IUJW,XAIV->XAJW', gamma1['b'], h['bb'][C,A,C,A], t['bb'][pA,pV,hC,hA], optimize=True)
    O['bb'][A,V,C,A] += scale * +1.00000000 * np.einsum('UV,WUIA,BAXV->WBIX', gamma1['b'], h['bb'][A,A,C,V], t['bb'][pV,pV,hA,hA], optimize=True)
    O['bb'][A,V,C,A] += scale * -1.00000000 * np.einsum('UV,WUXA,BAIV->WBIX', gamma1['b'], h['bb'][A,A,A,V], t['bb'][pV,pV,hC,hA], optimize=True)
    O['bb'][A,V,C,A] += scale * +1.00000000 * np.einsum('UV,UAIB,WBXV->WAIX', gamma1['b'], h['bb'][A,V,C,V], t['bb'][pA,pV,hA,hA], optimize=True)
    O['bb'][A,V,C,A] += scale * -1.00000000 * np.einsum('UV,UAWB,XBIV->XAIW', gamma1['b'], h['bb'][A,V,A,V], t['bb'][pA,pV,hC,hA], optimize=True)
    O['bb'][A,V,C,A] += scale * +0.50000000 * np.einsum('IJKU,VAIJ->VAKU', h['bb'][C,C,C,A], t['bb'][pA,pV,hC,hC], optimize=True)
    O['bb'][A,V,C,A] += scale * +1.00000000 * np.einsum('IUJA,BAIV->UBJV', h['bb'][C,A,C,V], t['bb'][pV,pV,hC,hA], optimize=True)
    O['bb'][A,V,C,A] += scale * +1.00000000 * np.einsum('IUVA,BAJI->UBJV', h['bb'][C,A,A,V], t['bb'][pV,pV,hC,hC], optimize=True)
    O['bb'][A,V,C,A] += scale * -1.00000000 * np.einsum('IAJB,UBIV->UAJV', h['bb'][C,V,C,V], t['bb'][pA,pV,hC,hA], optimize=True)
    O['bb'][A,V,C,A] += scale * -1.00000000 * np.einsum('IAUB,VBJI->VAJU', h['bb'][C,V,A,V], t['bb'][pA,pV,hC,hC], optimize=True)
    O['bb'][A,V,C,A] += scale * +0.50000000 * np.einsum('UABC,BCIV->UAIV', h['bb'][A,V,V,V], t['bb'][pV,pV,hC,hA], optimize=True)
    O['bb'][C,V,C,A] += scale * +0.50000000 * np.einsum('UV,WX,IAVX,UWJY->IAJY', eta1['b'], eta1['b'], h['bb'][C,V,A,A], t['bb'][pA,pA,hC,hA], optimize=True)
    O['bb'][C,V,C,A] += scale * +1.00000000 * np.einsum('UV,WX,IUJX,WAYV->IAJY', eta1['b'], gamma1['b'], h['bb'][C,A,C,A], t['bb'][pA,pV,hA,hA], optimize=True)
    O['bb'][C,V,C,A] += scale * -1.00000000 * np.einsum('UV,WX,IUYX,WAJV->IAJY', eta1['b'], gamma1['b'], h['bb'][C,A,A,A], t['bb'][pA,pV,hC,hA], optimize=True)
    O['bb'][C,V,C,A] += scale * -1.00000000 * np.einsum('UV,WX,IWJV,UAYX->IAJY', eta1['b'], gamma1['b'], h['bb'][C,A,C,A], t['bb'][pA,pV,hA,hA], optimize=True)
    O['bb'][C,V,C,A] += scale * +1.00000000 * np.einsum('UV,WX,IWYV,UAJX->IAJY', eta1['b'], gamma1['b'], h['bb'][C,A,A,A], t['bb'][pA,pV,hC,hA], optimize=True)
    O['bb'][C,V,C,A] += scale * +1.00000000 * np.einsum('UV,IJKV,UAJW->IAKW', eta1['b'], h['bb'][C,C,C,A], t['bb'][pA,pV,hC,hA], optimize=True)
    O['bb'][C,V,C,A] += scale * +1.00000000 * np.einsum('UV,IJWV,UAKJ->IAKW', eta1['b'], h['bb'][C,C,A,A], t['bb'][pA,pV,hC,hC], optimize=True)
    O['bb'][C,V,C,A] += scale * +1.00000000 * np.einsum('UV,IAVB,UBJW->IAJW', eta1['b'], h['bb'][C,V,A,V], t['bb'][pA,pV,hC,hA], optimize=True)
    O['bb'][C,V,C,A] += scale * -0.50000000 * np.einsum('UV,WX,IAVX,UWJY->IAJY', gamma1['b'], gamma1['b'], h['bb'][C,V,A,A], t['bb'][pA,pA,hC,hA], optimize=True)
    O['bb'][C,V,C,A] += scale * +1.00000000 * np.einsum('UV,IUJA,BAWV->IBJW', gamma1['b'], h['bb'][C,A,C,V], t['bb'][pV,pV,hA,hA], optimize=True)
    O['bb'][C,V,C,A] += scale * -1.00000000 * np.einsum('UV,IUWA,BAJV->IBJW', gamma1['b'], h['bb'][C,A,A,V], t['bb'][pV,pV,hC,hA], optimize=True)
    O['bb'][C,V,C,A] += scale * -1.00000000 * np.einsum('IJKA,BAJU->IBKU', h['bb'][C,C,C,V], t['bb'][pV,pV,hC,hA], optimize=True)
    O['bb'][C,V,C,A] += scale * -1.00000000 * np.einsum('IJUA,BAKJ->IBKU', h['bb'][C,C,A,V], t['bb'][pV,pV,hC,hC], optimize=True)
    O['bb'][C,V,C,A] += scale * +0.50000000 * np.einsum('IABC,BCJU->IAJU', h['bb'][C,V,V,V], t['bb'][pV,pV,hC,hA], optimize=True)
    O['bb'][V,V,C,A] += scale * -0.25000000 * np.einsum('UV,WX,UWIY,ABVX->ABIY', eta1['b'], eta1['b'], h['bb'][A,A,C,A], t['bb'][pV,pV,hA,hA], optimize=True)
    O['bb'][V,V,C,A] += scale * +0.25000000 * np.einsum('UV,WX,ABVX,UWIY->ABIY', eta1['b'], eta1['b'], h['bb'][V,V,A,A], t['bb'][pA,pA,hC,hA], optimize=True)
    O['bb'][V,V,C,A] += scale * -1.00000000 * np.einsum('UV,WX,UAIX,WBYV->ABIY', eta1['b'], gamma1['b'], h['bb'][A,V,C,A], t['bb'][pA,pV,hA,hA], optimize=True)
    O['bb'][V,V,C,A] += scale * +1.00000000 * np.einsum('UV,WX,UAYX,WBIV->ABIY', eta1['b'], gamma1['b'], h['bb'][A,V,A,A], t['bb'][pA,pV,hC,hA], optimize=True)
    O['bb'][V,V,C,A] += scale * +1.00000000 * np.einsum('UV,WX,WAIV,UBYX->ABIY', eta1['b'], gamma1['b'], h['bb'][A,V,C,A], t['bb'][pA,pV,hA,hA], optimize=True)
    O['bb'][V,V,C,A] += scale * -1.00000000 * np.einsum('UV,WX,WAYV,UBIX->ABIY', eta1['b'], gamma1['b'], h['bb'][A,V,A,A], t['bb'][pA,pV,hC,hA], optimize=True)
    O['bb'][V,V,C,A] += scale * -1.00000000 * np.einsum('UV,IAJV,UBIW->ABJW', eta1['b'], h['bb'][C,V,C,A], t['bb'][pA,pV,hC,hA], optimize=True)
    O['bb'][V,V,C,A] += scale * -1.00000000 * np.einsum('UV,IAWV,UBJI->ABJW', eta1['b'], h['bb'][C,V,A,A], t['bb'][pA,pV,hC,hC], optimize=True)
    O['bb'][V,V,C,A] += scale * +0.50000000 * np.einsum('UV,ABVC,UCIW->ABIW', eta1['b'], h['bb'][V,V,A,V], t['bb'][pA,pV,hC,hA], optimize=True)
    O['bb'][V,V,C,A] += scale * +0.25000000 * np.einsum('UV,WX,UWIY,ABVX->ABIY', gamma1['b'], gamma1['b'], h['bb'][A,A,C,A], t['bb'][pV,pV,hA,hA], optimize=True)
    O['bb'][V,V,C,A] += scale * -0.25000000 * np.einsum('UV,WX,ABVX,UWIY->ABIY', gamma1['b'], gamma1['b'], h['bb'][V,V,A,A], t['bb'][pA,pA,hC,hA], optimize=True)
    O['bb'][V,V,C,A] += scale * +0.50000000 * np.einsum('UV,IUJW,ABIV->ABJW', gamma1['b'], h['bb'][C,A,C,A], t['bb'][pV,pV,hC,hA], optimize=True)
    O['bb'][V,V,C,A] += scale * -1.00000000 * np.einsum('UV,UAIB,CBWV->ACIW', gamma1['b'], h['bb'][A,V,C,V], t['bb'][pV,pV,hA,hA], optimize=True)
    O['bb'][V,V,C,A] += scale * +1.00000000 * np.einsum('UV,UAWB,CBIV->ACIW', gamma1['b'], h['bb'][A,V,A,V], t['bb'][pV,pV,hC,hA], optimize=True)
    O['bb'][V,V,C,A] += scale * +0.25000000 * np.einsum('IJKU,ABIJ->ABKU', h['bb'][C,C,C,A], t['bb'][pV,pV,hC,hC], optimize=True)
    O['bb'][V,V,C,A] += scale * +1.00000000 * np.einsum('IAJB,CBIU->ACJU', h['bb'][C,V,C,V], t['bb'][pV,pV,hC,hA], optimize=True)
    O['bb'][V,V,C,A] += scale * +1.00000000 * np.einsum('IAUB,CBJI->ACJU', h['bb'][C,V,A,V], t['bb'][pV,pV,hC,hC], optimize=True)
    O['bb'][V,V,C,A] += scale * +0.25000000 * np.einsum('ABCD,CDIU->ABIU', h['bb'][V,V,V,V], t['bb'][pV,pV,hC,hA], optimize=True)
    O['bb'][A,A,C,C] += scale * +0.12500000 * np.einsum('UV,WX,YZVX,UWIJ->YZIJ', eta1['b'], eta1['b'], h['bb'][A,A,A,A], t['bb'][pA,pA,hC,hC], optimize=True)
    
    O['bb'][A,A,C,C] += scale * -1.00000000 * np.einsum('UV,WX,YUIX,ZWJV->YZIJ', eta1['b'], gamma1['b'], h['bb'][A,A,C,A], t['bb'][pA,pA,hC,hA], optimize=True)
    O['bb'][A,A,C,C] += scale * +1.00000000 * np.einsum('UV,WX,YWIV,ZUJX->YZIJ', eta1['b'], gamma1['b'], h['bb'][A,A,C,A], t['bb'][pA,pA,hC,hA], optimize=True)
    O['bb'][A,A,C,C] += scale * -1.00000000 * np.einsum('UV,IWJV,XUKI->WXJK', eta1['b'], h['bb'][C,A,C,A], t['bb'][pA,pA,hC,hC], optimize=True)
    O['bb'][A,A,C,C] += scale * +0.25000000 * np.einsum('UV,WXVA,UAIJ->WXIJ', eta1['b'], h['bb'][A,A,A,V], t['bb'][pA,pV,hC,hC], optimize=True)
    O['bb'][A,A,C,C] += scale * -0.12500000 * np.einsum('UV,WX,YZVX,UWIJ->YZIJ', gamma1['b'], gamma1['b'], h['bb'][A,A,A,A], t['bb'][pA,pA,hC,hC], optimize=True)
    
    O['bb'][A,A,C,C] += scale * +0.25000000 * np.einsum('UV,IUJK,WXIV->WXJK', gamma1['b'], h['bb'][C,A,C,C], t['bb'][pA,pA,hC,hA], optimize=True)
    O['bb'][A,A,C,C] += scale * +1.00000000 * np.einsum('UV,WUIA,XAJV->WXIJ', gamma1['b'], h['bb'][A,A,C,V], t['bb'][pA,pV,hC,hA], optimize=True)
    O['bb'][A,A,C,C] += scale * +0.12500000 * np.einsum('IJKL,UVIJ->UVKL', h['bb'][C,C,C,C], t['bb'][pA,pA,hC,hC], optimize=True)
    O['bb'][A,A,C,C] += scale * -1.00000000 * np.einsum('IUJA,VAKI->UVJK', h['bb'][C,A,C,V], t['bb'][pA,pV,hC,hC], optimize=True)
    O['bb'][A,A,C,C] += scale * +0.12500000 * np.einsum('UVAB,ABIJ->UVIJ', h['bb'][A,A,V,V], t['bb'][pV,pV,hC,hC], optimize=True)
    O['bb'][C,A,C,C] += scale * +0.25000000 * np.einsum('UV,WX,IYVX,UWJK->IYJK', eta1['b'], eta1['b'], h['bb'][C,A,A,A], t['bb'][pA,pA,hC,hC], optimize=True)
    O['bb'][C,A,C,C] += scale * -1.00000000 * np.einsum('UV,WX,IUJX,YWKV->IYJK', eta1['b'], gamma1['b'], h['bb'][C,A,C,A], t['bb'][pA,pA,hC,hA], optimize=True)
    O['bb'][C,A,C,C] += scale * +1.00000000 * np.einsum('UV,WX,IWJV,YUKX->IYJK', eta1['b'], gamma1['b'], h['bb'][C,A,C,A], t['bb'][pA,pA,hC,hA], optimize=True)
    O['bb'][C,A,C,C] += scale * +1.00000000 * np.einsum('UV,IJKV,WULJ->IWKL', eta1['b'], h['bb'][C,C,C,A], t['bb'][pA,pA,hC,hC], optimize=True)
    O['bb'][C,A,C,C] += scale * +0.50000000 * np.einsum('UV,IWVA,UAJK->IWJK', eta1['b'], h['bb'][C,A,A,V], t['bb'][pA,pV,hC,hC], optimize=True)
    O['bb'][C,A,C,C] += scale * -0.25000000 * np.einsum('UV,WX,IYVX,UWJK->IYJK', gamma1['b'], gamma1['b'], h['bb'][C,A,A,A], t['bb'][pA,pA,hC,hC], optimize=True)
    O['bb'][C,A,C,C] += scale * +1.00000000 * np.einsum('UV,IUJA,WAKV->IWJK', gamma1['b'], h['bb'][C,A,C,V], t['bb'][pA,pV,hC,hA], optimize=True)
    O['bb'][C,A,C,C] += scale * +1.00000000 * np.einsum('IJKA,UALJ->IUKL', h['bb'][C,C,C,V], t['bb'][pA,pV,hC,hC], optimize=True)
    O['bb'][C,A,C,C] += scale * +0.25000000 * np.einsum('IUAB,ABJK->IUJK', h['bb'][C,A,V,V], t['bb'][pV,pV,hC,hC], optimize=True)
    O['bb'][C,C,C,C] += scale * +0.12500000 * np.einsum('UV,WX,IJVX,UWKL->IJKL', eta1['b'], eta1['b'], h['bb'][C,C,A,A], t['bb'][pA,pA,hC,hC], optimize=True)
    O['bb'][C,C,C,C] += scale * +0.25000000 * np.einsum('UV,IJVA,UAKL->IJKL', eta1['b'], h['bb'][C,C,A,V], t['bb'][pA,pV,hC,hC], optimize=True)
    O['bb'][C,C,C,C] += scale * -0.12500000 * np.einsum('UV,WX,IJVX,UWKL->IJKL', gamma1['b'], gamma1['b'], h['bb'][C,C,A,A], t['bb'][pA,pA,hC,hC], optimize=True)
    O['bb'][C,C,C,C] += scale * +0.12500000 * np.einsum('IJAB,ABKL->IJKL', h['bb'][C,C,V,V], t['bb'][pV,pV,hC,hC], optimize=True)
    O['bb'][A,V,C,C] += scale * +0.25000000 * np.einsum('UV,WX,YAVX,UWIJ->YAIJ', eta1['b'], eta1['b'], h['bb'][A,V,A,A], t['bb'][pA,pA,hC,hC], optimize=True)
    O['bb'][A,V,C,C] += scale * -0.25000000 * np.einsum('UV,WX,UWIJ,YAVX->YAIJ', eta1['b'], eta1['b'], h['bb'][A,A,C,C], t['bb'][pA,pV,hA,hA], optimize=True)
    O['bb'][A,V,C,C] += scale * +1.00000000 * np.einsum('UV,WX,YUIX,WAJV->YAIJ', eta1['b'], gamma1['b'], h['bb'][A,A,C,A], t['bb'][pA,pV,hC,hA], optimize=True)
    O['bb'][A,V,C,C] += scale * -1.00000000 * np.einsum('UV,WX,YWIV,UAJX->YAIJ', eta1['b'], gamma1['b'], h['bb'][A,A,C,A], t['bb'][pA,pV,hC,hA], optimize=True)
    O['bb'][A,V,C,C] += scale * -1.00000000 * np.einsum('UV,WX,UAIX,YWJV->YAIJ', eta1['b'], gamma1['b'], h['bb'][A,V,C,A], t['bb'][pA,pA,hC,hA], optimize=True)
    O['bb'][A,V,C,C] += scale * +1.00000000 * np.einsum('UV,WX,WAIV,YUJX->YAIJ', eta1['b'], gamma1['b'], h['bb'][A,V,C,A], t['bb'][pA,pA,hC,hA], optimize=True)
    O['bb'][A,V,C,C] += scale * +1.00000000 * np.einsum('UV,IWJV,UAKI->WAJK', eta1['b'], h['bb'][C,A,C,A], t['bb'][pA,pV,hC,hC], optimize=True)
    O['bb'][A,V,C,C] += scale * +1.00000000 * np.einsum('UV,IAJV,WUKI->WAJK', eta1['b'], h['bb'][C,V,C,A], t['bb'][pA,pA,hC,hC], optimize=True)
    O['bb'][A,V,C,C] += scale * +0.50000000 * np.einsum('UV,WAVB,UBIJ->WAIJ', eta1['b'], h['bb'][A,V,A,V], t['bb'][pA,pV,hC,hC], optimize=True)
    O['bb'][A,V,C,C] += scale * -0.25000000 * np.einsum('UV,WX,YAVX,UWIJ->YAIJ', gamma1['b'], gamma1['b'], h['bb'][A,V,A,A], t['bb'][pA,pA,hC,hC], optimize=True)
    O['bb'][A,V,C,C] += scale * +0.25000000 * np.einsum('UV,WX,UWIJ,YAVX->YAIJ', gamma1['b'], gamma1['b'], h['bb'][A,A,C,C], t['bb'][pA,pV,hA,hA], optimize=True)
    O['bb'][A,V,C,C] += scale * +0.50000000 * np.einsum('UV,IUJK,WAIV->WAJK', gamma1['b'], h['bb'][C,A,C,C], t['bb'][pA,pV,hC,hA], optimize=True)
    O['bb'][A,V,C,C] += scale * +1.00000000 * np.einsum('UV,WUIA,BAJV->WBIJ', gamma1['b'], h['bb'][A,A,C,V], t['bb'][pV,pV,hC,hA], optimize=True)
    O['bb'][A,V,C,C] += scale * +1.00000000 * np.einsum('UV,UAIB,WBJV->WAIJ', gamma1['b'], h['bb'][A,V,C,V], t['bb'][pA,pV,hC,hA], optimize=True)
    O['bb'][A,V,C,C] += scale * +0.25000000 * np.einsum('IJKL,UAIJ->UAKL', h['bb'][C,C,C,C], t['bb'][pA,pV,hC,hC], optimize=True)
    O['bb'][A,V,C,C] += scale * -1.00000000 * np.einsum('IUJA,BAKI->UBJK', h['bb'][C,A,C,V], t['bb'][pV,pV,hC,hC], optimize=True)
    O['bb'][A,V,C,C] += scale * +1.00000000 * np.einsum('IAJB,UBKI->UAJK', h['bb'][C,V,C,V], t['bb'][pA,pV,hC,hC], optimize=True)
    O['bb'][A,V,C,C] += scale * +0.25000000 * np.einsum('UABC,BCIJ->UAIJ', h['bb'][A,V,V,V], t['bb'][pV,pV,hC,hC], optimize=True)
    O['bb'][C,V,C,C] += scale * +0.25000000 * np.einsum('UV,WX,IAVX,UWJK->IAJK', eta1['b'], eta1['b'], h['bb'][C,V,A,A], t['bb'][pA,pA,hC,hC], optimize=True)
    O['bb'][C,V,C,C] += scale * +1.00000000 * np.einsum('UV,WX,IUJX,WAKV->IAJK', eta1['b'], gamma1['b'], h['bb'][C,A,C,A], t['bb'][pA,pV,hC,hA], optimize=True)
    O['bb'][C,V,C,C] += scale * -1.00000000 * np.einsum('UV,WX,IWJV,UAKX->IAJK', eta1['b'], gamma1['b'], h['bb'][C,A,C,A], t['bb'][pA,pV,hC,hA], optimize=True)
    O['bb'][C,V,C,C] += scale * -1.00000000 * np.einsum('UV,IJKV,UALJ->IAKL', eta1['b'], h['bb'][C,C,C,A], t['bb'][pA,pV,hC,hC], optimize=True)
    O['bb'][C,V,C,C] += scale * +0.50000000 * np.einsum('UV,IAVB,UBJK->IAJK', eta1['b'], h['bb'][C,V,A,V], t['bb'][pA,pV,hC,hC], optimize=True)
    O['bb'][C,V,C,C] += scale * -0.25000000 * np.einsum('UV,WX,IAVX,UWJK->IAJK', gamma1['b'], gamma1['b'], h['bb'][C,V,A,A], t['bb'][pA,pA,hC,hC], optimize=True)
    O['bb'][C,V,C,C] += scale * +1.00000000 * np.einsum('UV,IUJA,BAKV->IBJK', gamma1['b'], h['bb'][C,A,C,V], t['bb'][pV,pV,hC,hA], optimize=True)
    O['bb'][C,V,C,C] += scale * +1.00000000 * np.einsum('IJKA,BALJ->IBKL', h['bb'][C,C,C,V], t['bb'][pV,pV,hC,hC], optimize=True)
    O['bb'][C,V,C,C] += scale * +0.25000000 * np.einsum('IABC,BCJK->IAJK', h['bb'][C,V,V,V], t['bb'][pV,pV,hC,hC], optimize=True)
    O['bb'][V,V,C,C] += scale * -0.12500000 * np.einsum('UV,WX,UWIJ,ABVX->ABIJ', eta1['b'], eta1['b'], h['bb'][A,A,C,C], t['bb'][pV,pV,hA,hA], optimize=True)
    O['bb'][V,V,C,C] += scale * +0.12500000 * np.einsum('UV,WX,ABVX,UWIJ->ABIJ', eta1['b'], eta1['b'], h['bb'][V,V,A,A], t['bb'][pA,pA,hC,hC], optimize=True)
    O['bb'][V,V,C,C] += scale * -1.00000000 * np.einsum('UV,WX,UAIX,WBJV->ABIJ', eta1['b'], gamma1['b'], h['bb'][A,V,C,A], t['bb'][pA,pV,hC,hA], optimize=True)
    O['bb'][V,V,C,C] += scale * +1.00000000 * np.einsum('UV,WX,WAIV,UBJX->ABIJ', eta1['b'], gamma1['b'], h['bb'][A,V,C,A], t['bb'][pA,pV,hC,hA], optimize=True)
    O['bb'][V,V,C,C] += scale * +1.00000000 * np.einsum('UV,IAJV,UBKI->ABJK', eta1['b'], h['bb'][C,V,C,A], t['bb'][pA,pV,hC,hC], optimize=True)
    O['bb'][V,V,C,C] += scale * +0.25000000 * np.einsum('UV,ABVC,UCIJ->ABIJ', eta1['b'], h['bb'][V,V,A,V], t['bb'][pA,pV,hC,hC], optimize=True)
    O['bb'][V,V,C,C] += scale * +0.12500000 * np.einsum('UV,WX,UWIJ,ABVX->ABIJ', gamma1['b'], gamma1['b'], h['bb'][A,A,C,C], t['bb'][pV,pV,hA,hA], optimize=True)
    O['bb'][V,V,C,C] += scale * -0.12500000 * np.einsum('UV,WX,ABVX,UWIJ->ABIJ', gamma1['b'], gamma1['b'], h['bb'][V,V,A,A], t['bb'][pA,pA,hC,hC], optimize=True)
    O['bb'][V,V,C,C] += scale * +0.25000000 * np.einsum('UV,IUJK,ABIV->ABJK', gamma1['b'], h['bb'][C,A,C,C], t['bb'][pV,pV,hC,hA], optimize=True)
    O['bb'][V,V,C,C] += scale * -1.00000000 * np.einsum('UV,UAIB,CBJV->ACIJ', gamma1['b'], h['bb'][A,V,C,V], t['bb'][pV,pV,hC,hA], optimize=True)
    O['bb'][V,V,C,C] += scale * +0.12500000 * np.einsum('IJKL,ABIJ->ABKL', h['bb'][C,C,C,C], t['bb'][pV,pV,hC,hC], optimize=True)
    O['bb'][V,V,C,C] += scale * -1.00000000 * np.einsum('IAJB,CBKI->ACJK', h['bb'][C,V,C,V], t['bb'][pV,pV,hC,hC], optimize=True)
    O['bb'][V,V,C,C] += scale * +0.12500000 * np.einsum('ABCD,CDIJ->ABIJ', h['bb'][V,V,V,V], t['bb'][pV,pV,hC,hC], optimize=True)
    
    O['bb'][A,A,C,V] += scale * -1.00000000 * np.einsum('UV,WX,YUXA,ZWIV->YZIA', eta1['b'], gamma1['b'], h['bb'][A,A,A,V], t['bb'][pA,pA,hC,hA], optimize=True)
    O['bb'][A,A,C,V] += scale * +1.00000000 * np.einsum('UV,WX,YWVA,ZUIX->YZIA', eta1['b'], gamma1['b'], h['bb'][A,A,A,V], t['bb'][pA,pA,hC,hA], optimize=True)
    O['bb'][A,A,C,V] += scale * -1.00000000 * np.einsum('UV,IWVA,XUJI->WXJA', eta1['b'], h['bb'][C,A,A,V], t['bb'][pA,pA,hC,hC], optimize=True)
    
    O['bb'][A,A,C,V] += scale * +0.50000000 * np.einsum('UV,IUJA,WXIV->WXJA', gamma1['b'], h['bb'][C,A,C,V], t['bb'][pA,pA,hC,hA], optimize=True)
    O['bb'][A,A,C,V] += scale * -1.00000000 * np.einsum('UV,WUAB,XBIV->WXIA', gamma1['b'], h['bb'][A,A,V,V], t['bb'][pA,pV,hC,hA], optimize=True)
    O['bb'][A,A,C,V] += scale * +0.25000000 * np.einsum('IJKA,UVIJ->UVKA', h['bb'][C,C,C,V], t['bb'][pA,pA,hC,hC], optimize=True)
    O['bb'][A,A,C,V] += scale * +1.00000000 * np.einsum('IUAB,VBJI->UVJA', h['bb'][C,A,V,V], t['bb'][pA,pV,hC,hC], optimize=True)
    O['bb'][C,A,C,V] += scale * -1.00000000 * np.einsum('UV,WX,IUXA,YWJV->IYJA', eta1['b'], gamma1['b'], h['bb'][C,A,A,V], t['bb'][pA,pA,hC,hA], optimize=True)
    O['bb'][C,A,C,V] += scale * +1.00000000 * np.einsum('UV,WX,IWVA,YUJX->IYJA', eta1['b'], gamma1['b'], h['bb'][C,A,A,V], t['bb'][pA,pA,hC,hA], optimize=True)
    O['bb'][C,A,C,V] += scale * +1.00000000 * np.einsum('UV,IJVA,WUKJ->IWKA', eta1['b'], h['bb'][C,C,A,V], t['bb'][pA,pA,hC,hC], optimize=True)
    O['bb'][C,A,C,V] += scale * -1.00000000 * np.einsum('UV,IUAB,WBJV->IWJA', gamma1['b'], h['bb'][C,A,V,V], t['bb'][pA,pV,hC,hA], optimize=True)
    O['bb'][C,A,C,V] += scale * -1.00000000 * np.einsum('IJAB,UBKJ->IUKA', h['bb'][C,C,V,V], t['bb'][pA,pV,hC,hC], optimize=True)
    O['bb'][A,V,C,V] += scale * -0.50000000 * np.einsum('UV,WX,UWIA,YBVX->YBIA', eta1['b'], eta1['b'], h['bb'][A,A,C,V], t['bb'][pA,pV,hA,hA], optimize=True)
    O['bb'][A,V,C,V] += scale * +1.00000000 * np.einsum('UV,WX,YUXA,WBIV->YBIA', eta1['b'], gamma1['b'], h['bb'][A,A,A,V], t['bb'][pA,pV,hC,hA], optimize=True)
    O['bb'][A,V,C,V] += scale * -1.00000000 * np.einsum('UV,WX,YWVA,UBIX->YBIA', eta1['b'], gamma1['b'], h['bb'][A,A,A,V], t['bb'][pA,pV,hC,hA], optimize=True)
    O['bb'][A,V,C,V] += scale * -1.00000000 * np.einsum('UV,WX,UAXB,YWIV->YAIB', eta1['b'], gamma1['b'], h['bb'][A,V,A,V], t['bb'][pA,pA,hC,hA], optimize=True)
    O['bb'][A,V,C,V] += scale * +1.00000000 * np.einsum('UV,WX,WAVB,YUIX->YAIB', eta1['b'], gamma1['b'], h['bb'][A,V,A,V], t['bb'][pA,pA,hC,hA], optimize=True)
    O['bb'][A,V,C,V] += scale * +1.00000000 * np.einsum('UV,IWVA,UBJI->WBJA', eta1['b'], h['bb'][C,A,A,V], t['bb'][pA,pV,hC,hC], optimize=True)
    O['bb'][A,V,C,V] += scale * +1.00000000 * np.einsum('UV,IAVB,WUJI->WAJB', eta1['b'], h['bb'][C,V,A,V], t['bb'][pA,pA,hC,hC], optimize=True)
    O['bb'][A,V,C,V] += scale * +0.50000000 * np.einsum('UV,WX,UWIA,YBVX->YBIA', gamma1['b'], gamma1['b'], h['bb'][A,A,C,V], t['bb'][pA,pV,hA,hA], optimize=True)
    O['bb'][A,V,C,V] += scale * +1.00000000 * np.einsum('UV,IUJA,WBIV->WBJA', gamma1['b'], h['bb'][C,A,C,V], t['bb'][pA,pV,hC,hA], optimize=True)
    O['bb'][A,V,C,V] += scale * -1.00000000 * np.einsum('UV,WUAB,CBIV->WCIA', gamma1['b'], h['bb'][A,A,V,V], t['bb'][pV,pV,hC,hA], optimize=True)
    O['bb'][A,V,C,V] += scale * -1.00000000 * np.einsum('UV,UABC,WCIV->WAIB', gamma1['b'], h['bb'][A,V,V,V], t['bb'][pA,pV,hC,hA], optimize=True)
    O['bb'][A,V,C,V] += scale * +0.50000000 * np.einsum('IJKA,UBIJ->UBKA', h['bb'][C,C,C,V], t['bb'][pA,pV,hC,hC], optimize=True)
    O['bb'][A,V,C,V] += scale * +1.00000000 * np.einsum('IUAB,CBJI->UCJA', h['bb'][C,A,V,V], t['bb'][pV,pV,hC,hC], optimize=True)
    O['bb'][A,V,C,V] += scale * -1.00000000 * np.einsum('IABC,UCJI->UAJB', h['bb'][C,V,V,V], t['bb'][pA,pV,hC,hC], optimize=True)
    O['bb'][C,V,C,V] += scale * +1.00000000 * np.einsum('UV,WX,IUXA,WBJV->IBJA', eta1['b'], gamma1['b'], h['bb'][C,A,A,V], t['bb'][pA,pV,hC,hA], optimize=True)
    O['bb'][C,V,C,V] += scale * -1.00000000 * np.einsum('UV,WX,IWVA,UBJX->IBJA', eta1['b'], gamma1['b'], h['bb'][C,A,A,V], t['bb'][pA,pV,hC,hA], optimize=True)
    O['bb'][C,V,C,V] += scale * -1.00000000 * np.einsum('UV,IJVA,UBKJ->IBKA', eta1['b'], h['bb'][C,C,A,V], t['bb'][pA,pV,hC,hC], optimize=True)
    O['bb'][C,V,C,V] += scale * -1.00000000 * np.einsum('UV,IUAB,CBJV->ICJA', gamma1['b'], h['bb'][C,A,V,V], t['bb'][pV,pV,hC,hA], optimize=True)
    O['bb'][C,V,C,V] += scale * -1.00000000 * np.einsum('IJAB,CBKJ->ICKA', h['bb'][C,C,V,V], t['bb'][pV,pV,hC,hC], optimize=True)
    O['bb'][V,V,C,V] += scale * -0.25000000 * np.einsum('UV,WX,UWIA,BCVX->BCIA', eta1['b'], eta1['b'], h['bb'][A,A,C,V], t['bb'][pV,pV,hA,hA], optimize=True)
    O['bb'][V,V,C,V] += scale * -1.00000000 * np.einsum('UV,WX,UAXB,WCIV->ACIB', eta1['b'], gamma1['b'], h['bb'][A,V,A,V], t['bb'][pA,pV,hC,hA], optimize=True)
    O['bb'][V,V,C,V] += scale * +1.00000000 * np.einsum('UV,WX,WAVB,UCIX->ACIB', eta1['b'], gamma1['b'], h['bb'][A,V,A,V], t['bb'][pA,pV,hC,hA], optimize=True)
    O['bb'][V,V,C,V] += scale * +1.00000000 * np.einsum('UV,IAVB,UCJI->ACJB', eta1['b'], h['bb'][C,V,A,V], t['bb'][pA,pV,hC,hC], optimize=True)
    O['bb'][V,V,C,V] += scale * +0.25000000 * np.einsum('UV,WX,UWIA,BCVX->BCIA', gamma1['b'], gamma1['b'], h['bb'][A,A,C,V], t['bb'][pV,pV,hA,hA], optimize=True)
    O['bb'][V,V,C,V] += scale * +0.50000000 * np.einsum('UV,IUJA,BCIV->BCJA', gamma1['b'], h['bb'][C,A,C,V], t['bb'][pV,pV,hC,hA], optimize=True)
    O['bb'][V,V,C,V] += scale * +1.00000000 * np.einsum('UV,UABC,DCIV->ADIB', gamma1['b'], h['bb'][A,V,V,V], t['bb'][pV,pV,hC,hA], optimize=True)
    O['bb'][V,V,C,V] += scale * +0.25000000 * np.einsum('IJKA,BCIJ->BCKA', h['bb'][C,C,C,V], t['bb'][pV,pV,hC,hC], optimize=True)
    O['bb'][V,V,C,V] += scale * +1.00000000 * np.einsum('IABC,DCJI->ADJB', h['bb'][C,V,V,V], t['bb'][pV,pV,hC,hC], optimize=True)
    
    
    O['bb'][A,A,V,V] += scale * +0.25000000 * np.einsum('UV,IUAB,WXIV->WXAB', gamma1['b'], h['bb'][C,A,V,V], t['bb'][pA,pA,hC,hA], optimize=True)
    O['bb'][A,A,V,V] += scale * +0.12500000 * np.einsum('IJAB,UVIJ->UVAB', h['bb'][C,C,V,V], t['bb'][pA,pA,hC,hC], optimize=True)
    O['bb'][A,V,V,V] += scale * -0.25000000 * np.einsum('UV,WX,UWAB,YCVX->YCAB', eta1['b'], eta1['b'], h['bb'][A,A,V,V], t['bb'][pA,pV,hA,hA], optimize=True)
    O['bb'][A,V,V,V] += scale * +0.25000000 * np.einsum('UV,WX,UWAB,YCVX->YCAB', gamma1['b'], gamma1['b'], h['bb'][A,A,V,V], t['bb'][pA,pV,hA,hA], optimize=True)
    O['bb'][A,V,V,V] += scale * +0.50000000 * np.einsum('UV,IUAB,WCIV->WCAB', gamma1['b'], h['bb'][C,A,V,V], t['bb'][pA,pV,hC,hA], optimize=True)
    O['bb'][A,V,V,V] += scale * +0.25000000 * np.einsum('IJAB,UCIJ->UCAB', h['bb'][C,C,V,V], t['bb'][pA,pV,hC,hC], optimize=True)
    O['bb'][V,V,V,V] += scale * -0.12500000 * np.einsum('UV,WX,UWAB,CDVX->CDAB', eta1['b'], eta1['b'], h['bb'][A,A,V,V], t['bb'][pV,pV,hA,hA], optimize=True)
    O['bb'][V,V,V,V] += scale * +0.12500000 * np.einsum('UV,WX,UWAB,CDVX->CDAB', gamma1['b'], gamma1['b'], h['bb'][A,A,V,V], t['bb'][pV,pV,hA,hA], optimize=True)
    O['bb'][V,V,V,V] += scale * +0.25000000 * np.einsum('UV,IUAB,CDIV->CDAB', gamma1['b'], h['bb'][C,A,V,V], t['bb'][pV,pV,hC,hA], optimize=True)
    O['bb'][V,V,V,V] += scale * +0.12500000 * np.einsum('IJAB,CDIJ->CDAB', h['bb'][C,C,V,V], t['bb'][pV,pV,hC,hC], optimize=True)

    t1 = time.time()
    if verbose: print("h2c_t2c_c2c took {:.4f} seconds to run.".format(t1-t0))

    return O
