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

	
	
	
	
	O['0'] += scale * -0.50000000 * np.einsum('iu,vwux,vwix->', h['a'][c,a], lambda2['aa'], t['aa'][pa,pa,hc,ha], optimize=True)
	O['0'] += scale * -0.50000000 * np.einsum('ua,uvwx,vawx->', h['a'][a,v], lambda2['aa'], t['aa'][pa,pv,ha,ha], optimize=True)

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

	
	
	
	
	O['0'] += scale * -1.00000000 * np.einsum('iu,vUuV,vUiV->', h['a'][c,a], lambda2['ab'], t['ab'][pa,pA,hc,hA], optimize=True)
	O['0'] += scale * +1.00000000 * np.einsum('ua,uUvV,aUvV->', h['a'][a,v], lambda2['ab'], t['ab'][pv,pA,ha,hA], optimize=True)

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

	
	
	
	
	O['0'] += scale * -1.00000000 * np.einsum('IU,uVvU,uVvI->', h['b'][C,A], lambda2['ab'], t['ab'][pa,pA,ha,hC], optimize=True)
	O['0'] += scale * +1.00000000 * np.einsum('UA,uUvV,uAvV->', h['b'][A,V], lambda2['ab'], t['ab'][pa,pV,ha,hA], optimize=True)

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

	
	
	
	
	O['0'] += scale * -0.50000000 * np.einsum('IU,VWUX,VWIX->', h['b'][C,A], lambda2['bb'], t['bb'][pA,pA,hC,hA], optimize=True)
	O['0'] += scale * -0.50000000 * np.einsum('UA,UVWX,VAWX->', h['b'][A,V], lambda2['bb'], t['bb'][pA,pV,hA,hA], optimize=True)

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

	
	
	
	
	O['0'] += scale * +0.50000000 * np.einsum('iuvw,uxvw,xi->', h['aa'][c,a,a,a], lambda2['aa'], t['a'][pa,hc], optimize=True)
	O['0'] += scale * +0.50000000 * np.einsum('uvwa,uvwx,ax->', h['aa'][a,a,a,v], lambda2['aa'], t['a'][pv,ha], optimize=True)

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
	O['0'] += scale * +1.00000000 * np.einsum('uv,iwvx,wyxz,uyiz->', eta1['a'], h['aa'][c,a,a,a], lambda2['aa'], t['aa'][pa,pa,hc,ha], optimize=True)
	
	
	O['0'] += scale * +0.25000000 * np.einsum('uv,wxva,wxyz,uayz->', eta1['a'], h['aa'][a,a,a,v], lambda2['aa'], t['aa'][pa,pv,ha,ha], optimize=True)
	
	O['0'] += scale * +0.25000000 * np.einsum('uv,wx,uwab,abvx->', gamma1['a'], gamma1['a'], h['aa'][a,a,v,v], t['aa'][pv,pv,ha,ha], optimize=True)
	
	O['0'] += scale * +0.25000000 * np.einsum('uv,iuwx,yzwx,yziv->', gamma1['a'], h['aa'][c,a,a,a], lambda2['aa'], t['aa'][pa,pa,hc,ha], optimize=True)
	O['0'] += scale * +0.50000000 * np.einsum('uv,iuab,abiv->', gamma1['a'], h['aa'][c,a,v,v], t['aa'][pv,pv,hc,ha], optimize=True)
	
	O['0'] += scale * +1.00000000 * np.einsum('uv,uwxa,wyxz,yavz->', gamma1['a'], h['aa'][a,a,a,v], lambda2['aa'], t['aa'][pa,pv,ha,ha], optimize=True)
	
	O['0'] += scale * +0.12500000 * np.einsum('ijuv,wxuv,wxij->', h['aa'][c,c,a,a], lambda2['aa'], t['aa'][pa,pa,hc,hc], optimize=True)
	O['0'] += scale * +0.25000000 * np.einsum('ijab,abij->', h['aa'][c,c,v,v], t['aa'][pv,pv,hc,hc], optimize=True)
	O['0'] += scale * +0.25000000 * np.einsum('iuvw,uxyvwz,xyiz->', h['aa'][c,a,a,a], lambda3['aaa'], t['aa'][pa,pa,hc,ha], optimize=True)
	O['0'] += scale * +1.00000000 * np.einsum('iuva,uwvx,waix->', h['aa'][c,a,a,v], lambda2['aa'], t['aa'][pa,pv,hc,ha], optimize=True)
	O['0'] += scale * -0.25000000 * np.einsum('uvwa,uvxwyz,xayz->', h['aa'][a,a,a,v], lambda3['aaa'], t['aa'][pa,pv,ha,ha], optimize=True)
	O['0'] += scale * +0.12500000 * np.einsum('uvab,uvwx,abwx->', h['aa'][a,a,v,v], lambda2['aa'], t['aa'][pv,pv,ha,ha], optimize=True)

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

	
	
	O['0'] += scale * +1.00000000 * np.einsum('uv,iwvx,wUxV,uUiV->', eta1['a'], h['aa'][c,a,a,a], lambda2['ab'], t['ab'][pa,pA,hc,hA], optimize=True)
	
	
	
	O['0'] += scale * -1.00000000 * np.einsum('uv,uwxa,wUxV,aUvV->', gamma1['a'], h['aa'][a,a,a,v], lambda2['ab'], t['ab'][pv,pA,ha,hA], optimize=True)
	
	O['0'] += scale * +0.50000000 * np.einsum('iuvw,uxUvwV,xUiV->', h['aa'][c,a,a,a], lambda3['aab'], t['ab'][pa,pA,hc,hA], optimize=True)
	O['0'] += scale * -1.00000000 * np.einsum('iuva,uUvV,aUiV->', h['aa'][c,a,a,v], lambda2['ab'], t['ab'][pv,pA,hc,hA], optimize=True)
	O['0'] += scale * +0.50000000 * np.einsum('uvwa,uvUwxV,aUxV->', h['aa'][a,a,a,v], lambda3['aab'], t['ab'][pv,pA,ha,hA], optimize=True)

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

	
	
	
	
	O['0'] += scale * -1.00000000 * np.einsum('iUuV,vUuV,vi->', h['ab'][c,A,a,A], lambda2['ab'], t['a'][pa,hc], optimize=True)
	O['0'] += scale * +1.00000000 * np.einsum('uUaV,uUvV,av->', h['ab'][a,A,v,A], lambda2['ab'], t['a'][pv,ha], optimize=True)

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

	
	
	
	
	O['0'] += scale * -1.00000000 * np.einsum('uIvU,uVvU,VI->', h['ab'][a,C,a,A], lambda2['ab'], t['b'][pA,hC], optimize=True)
	O['0'] += scale * +1.00000000 * np.einsum('uUvA,uUvV,AV->', h['ab'][a,A,a,V], lambda2['ab'], t['b'][pV,hA], optimize=True)

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

	
	
	O['0'] += scale * +1.00000000 * np.einsum('uv,iUvV,wUxV,uwix->', eta1['a'], h['ab'][c,A,a,A], lambda2['ab'], t['aa'][pa,pa,hc,ha], optimize=True)
	
	
	
	O['0'] += scale * -1.00000000 * np.einsum('uv,uUaV,wUxV,wavx->', gamma1['a'], h['ab'][a,A,v,A], lambda2['ab'], t['aa'][pa,pv,ha,ha], optimize=True)
	
	O['0'] += scale * -0.50000000 * np.einsum('iUuV,vwUuxV,vwix->', h['ab'][c,A,a,A], lambda3['aab'], t['aa'][pa,pa,hc,ha], optimize=True)
	O['0'] += scale * -1.00000000 * np.einsum('iUaV,uUvV,uaiv->', h['ab'][c,A,v,A], lambda2['ab'], t['aa'][pa,pv,hc,ha], optimize=True)
	O['0'] += scale * -0.50000000 * np.einsum('uUaV,uvUwxV,vawx->', h['ab'][a,A,v,A], lambda3['aab'], t['aa'][pa,pv,ha,ha], optimize=True)

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
	O['0'] += scale * +1.00000000 * np.einsum('uv,iUvV,UWVX,uWiX->', eta1['a'], h['ab'][c,A,a,A], lambda2['bb'], t['ab'][pa,pA,hc,hA], optimize=True)
	
	O['0'] += scale * -1.00000000 * np.einsum('uv,wIvU,wVxU,uVxI->', eta1['a'], h['ab'][a,C,a,A], lambda2['ab'], t['ab'][pa,pA,ha,hC], optimize=True)
	
	O['0'] += scale * +1.00000000 * np.einsum('uv,wUvA,wUxV,uAxV->', eta1['a'], h['ab'][a,A,a,V], lambda2['ab'], t['ab'][pa,pV,ha,hA], optimize=True)
	O['0'] += scale * +1.00000000 * np.einsum('UV,uv,wx,wIvV,uUxI->', eta1['b'], eta1['a'], gamma1['a'], h['ab'][a,C,a,A], t['ab'][pa,pA,ha,hC], optimize=True)
	
	
	O['0'] += scale * +1.00000000 * np.einsum('UV,uv,WX,iWvV,uUiX->', eta1['b'], eta1['a'], gamma1['b'], h['ab'][c,A,a,A], t['ab'][pa,pA,hc,hA], optimize=True)
	O['0'] += scale * +1.00000000 * np.einsum('UV,uv,iIvV,uUiI->', eta1['b'], eta1['a'], h['ab'][c,C,a,A], t['ab'][pa,pA,hc,hC], optimize=True)
	
	
	O['0'] += scale * +1.00000000 * np.einsum('UV,uv,uIaV,aUvI->', eta1['b'], gamma1['a'], h['ab'][a,C,v,A], t['ab'][pv,pA,ha,hC], optimize=True)
	
	
	O['0'] += scale * +1.00000000 * np.einsum('UV,WX,uv,uWaV,aUvX->', eta1['b'], gamma1['b'], gamma1['a'], h['ab'][a,A,v,A], t['ab'][pv,pA,ha,hA], optimize=True)
	O['0'] += scale * +1.00000000 * np.einsum('UV,WX,iWaV,aUiX->', eta1['b'], gamma1['b'], h['ab'][c,A,v,A], t['ab'][pv,pA,hc,hA], optimize=True)
	
	
	O['0'] += scale * +1.00000000 * np.einsum('UV,iIaV,aUiI->', eta1['b'], h['ab'][c,C,v,A], t['ab'][pv,pA,hc,hC], optimize=True)
	O['0'] += scale * -1.00000000 * np.einsum('UV,iWuV,vWuX,vUiX->', eta1['b'], h['ab'][c,A,a,A], lambda2['ab'], t['ab'][pa,pA,hc,hA], optimize=True)
	O['0'] += scale * +1.00000000 * np.einsum('UV,uIvV,uwvx,wUxI->', eta1['b'], h['ab'][a,C,a,A], lambda2['aa'], t['ab'][pa,pA,ha,hC], optimize=True)
	
	
	O['0'] += scale * +1.00000000 * np.einsum('UV,uWaV,uWvX,aUvX->', eta1['b'], h['ab'][a,A,v,A], lambda2['ab'], t['ab'][pv,pA,ha,hA], optimize=True)
	O['0'] += scale * +1.00000000 * np.einsum('uv,uIwU,xVwU,xVvI->', gamma1['a'], h['ab'][a,C,a,A], lambda2['ab'], t['ab'][pa,pA,ha,hC], optimize=True)
	O['0'] += scale * +1.00000000 * np.einsum('uv,uIaA,aAvI->', gamma1['a'], h['ab'][a,C,v,V], t['ab'][pv,pV,ha,hC], optimize=True)
	
	O['0'] += scale * -1.00000000 * np.einsum('uv,uUwA,xUwV,xAvV->', gamma1['a'], h['ab'][a,A,a,V], lambda2['ab'], t['ab'][pa,pV,ha,hA], optimize=True)
	O['0'] += scale * +1.00000000 * np.einsum('uv,uUaV,UWVX,aWvX->', gamma1['a'], h['ab'][a,A,v,A], lambda2['bb'], t['ab'][pv,pA,ha,hA], optimize=True)
	
	
	O['0'] += scale * +1.00000000 * np.einsum('UV,uv,uUaA,aAvV->', gamma1['b'], gamma1['a'], h['ab'][a,A,v,V], t['ab'][pv,pV,ha,hA], optimize=True)
	
	O['0'] += scale * +1.00000000 * np.einsum('UV,iUuW,vXuW,vXiV->', gamma1['b'], h['ab'][c,A,a,A], lambda2['ab'], t['ab'][pa,pA,hc,hA], optimize=True)
	O['0'] += scale * +1.00000000 * np.einsum('UV,iUaA,aAiV->', gamma1['b'], h['ab'][c,A,v,V], t['ab'][pv,pV,hc,hA], optimize=True)
	
	O['0'] += scale * +1.00000000 * np.einsum('UV,uUvA,uwvx,wAxV->', gamma1['b'], h['ab'][a,A,a,V], lambda2['aa'], t['ab'][pa,pV,ha,hA], optimize=True)
	O['0'] += scale * -1.00000000 * np.einsum('UV,uUaW,uXvW,aXvV->', gamma1['b'], h['ab'][a,A,v,A], lambda2['ab'], t['ab'][pv,pA,ha,hA], optimize=True)
	
	O['0'] += scale * +1.00000000 * np.einsum('iIuU,vVuU,vViI->', h['ab'][c,C,a,A], lambda2['ab'], t['ab'][pa,pA,hc,hC], optimize=True)
	O['0'] += scale * +1.00000000 * np.einsum('iIaA,aAiI->', h['ab'][c,C,v,V], t['ab'][pv,pV,hc,hC], optimize=True)
	O['0'] += scale * -1.00000000 * np.einsum('iUuV,vUWuVX,vWiX->', h['ab'][c,A,a,A], lambda3['abb'], t['ab'][pa,pA,hc,hA], optimize=True)
	O['0'] += scale * -1.00000000 * np.einsum('iUuA,vUuV,vAiV->', h['ab'][c,A,a,V], lambda2['ab'], t['ab'][pa,pV,hc,hA], optimize=True)
	O['0'] += scale * +1.00000000 * np.einsum('iUaV,UWVX,aWiX->', h['ab'][c,A,v,A], lambda2['bb'], t['ab'][pv,pA,hc,hA], optimize=True)
	O['0'] += scale * -1.00000000 * np.einsum('uIvU,uwVvxU,wVxI->', h['ab'][a,C,a,A], lambda3['aab'], t['ab'][pa,pA,ha,hC], optimize=True)
	O['0'] += scale * +1.00000000 * np.einsum('uIvA,uwvx,wAxI->', h['ab'][a,C,a,V], lambda2['aa'], t['ab'][pa,pV,ha,hC], optimize=True)
	O['0'] += scale * -1.00000000 * np.einsum('uIaU,uVvU,aVvI->', h['ab'][a,C,v,A], lambda2['ab'], t['ab'][pv,pA,ha,hC], optimize=True)
	O['0'] += scale * +1.00000000 * np.einsum('uUvA,uwUvxV,wAxV->', h['ab'][a,A,a,V], lambda3['aab'], t['ab'][pa,pV,ha,hA], optimize=True)
	O['0'] += scale * +1.00000000 * np.einsum('uUaV,uUWvVX,aWvX->', h['ab'][a,A,v,A], lambda3['abb'], t['ab'][pv,pA,ha,hA], optimize=True)
	O['0'] += scale * +1.00000000 * np.einsum('uUaA,uUvV,aAvV->', h['ab'][a,A,v,V], lambda2['ab'], t['ab'][pv,pV,ha,hA], optimize=True)

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

	
	
	O['0'] += scale * +1.00000000 * np.einsum('UV,uIvV,uWvX,UWIX->', eta1['b'], h['ab'][a,C,a,A], lambda2['ab'], t['bb'][pA,pA,hC,hA], optimize=True)
	
	
	
	O['0'] += scale * -1.00000000 * np.einsum('UV,uUvA,uWvX,WAVX->', gamma1['b'], h['ab'][a,A,a,V], lambda2['ab'], t['bb'][pA,pV,hA,hA], optimize=True)
	
	O['0'] += scale * -0.50000000 * np.einsum('uIvU,uVWvUX,VWIX->', h['ab'][a,C,a,A], lambda3['abb'], t['bb'][pA,pA,hC,hA], optimize=True)
	O['0'] += scale * -1.00000000 * np.einsum('uIvA,uUvV,UAIV->', h['ab'][a,C,a,V], lambda2['ab'], t['bb'][pA,pV,hC,hA], optimize=True)
	O['0'] += scale * -0.50000000 * np.einsum('uUvA,uUVvWX,VAWX->', h['ab'][a,A,a,V], lambda3['abb'], t['bb'][pA,pV,hA,hA], optimize=True)

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

	
	
	
	
	O['0'] += scale * +0.50000000 * np.einsum('IUVW,UXVW,XI->', h['bb'][C,A,A,A], lambda2['bb'], t['b'][pA,hC], optimize=True)
	O['0'] += scale * +0.50000000 * np.einsum('UVWA,UVWX,AX->', h['bb'][A,A,A,V], lambda2['bb'], t['b'][pV,hA], optimize=True)

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

	
	
	O['0'] += scale * +1.00000000 * np.einsum('UV,IWVX,uWvX,uUvI->', eta1['b'], h['bb'][C,A,A,A], lambda2['ab'], t['ab'][pa,pA,ha,hC], optimize=True)
	
	
	
	O['0'] += scale * -1.00000000 * np.einsum('UV,UWXA,uWvX,uAvV->', gamma1['b'], h['bb'][A,A,A,V], lambda2['ab'], t['ab'][pa,pV,ha,hA], optimize=True)
	
	O['0'] += scale * +0.50000000 * np.einsum('IUVW,uUXvVW,uXvI->', h['bb'][C,A,A,A], lambda3['abb'], t['ab'][pa,pA,ha,hC], optimize=True)
	O['0'] += scale * -1.00000000 * np.einsum('IUVA,uUvV,uAvI->', h['bb'][C,A,A,V], lambda2['ab'], t['ab'][pa,pV,ha,hC], optimize=True)
	O['0'] += scale * +0.50000000 * np.einsum('UVWA,uUVvWX,uAvX->', h['bb'][A,A,A,V], lambda3['abb'], t['ab'][pa,pV,ha,hA], optimize=True)

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
	O['0'] += scale * +1.00000000 * np.einsum('UV,IWVX,WYXZ,UYIZ->', eta1['b'], h['bb'][C,A,A,A], lambda2['bb'], t['bb'][pA,pA,hC,hA], optimize=True)
	
	
	O['0'] += scale * +0.25000000 * np.einsum('UV,WXVA,WXYZ,UAYZ->', eta1['b'], h['bb'][A,A,A,V], lambda2['bb'], t['bb'][pA,pV,hA,hA], optimize=True)
	
	O['0'] += scale * +0.25000000 * np.einsum('UV,WX,UWAB,ABVX->', gamma1['b'], gamma1['b'], h['bb'][A,A,V,V], t['bb'][pV,pV,hA,hA], optimize=True)
	
	O['0'] += scale * +0.25000000 * np.einsum('UV,IUWX,YZWX,YZIV->', gamma1['b'], h['bb'][C,A,A,A], lambda2['bb'], t['bb'][pA,pA,hC,hA], optimize=True)
	O['0'] += scale * +0.50000000 * np.einsum('UV,IUAB,ABIV->', gamma1['b'], h['bb'][C,A,V,V], t['bb'][pV,pV,hC,hA], optimize=True)
	
	O['0'] += scale * +1.00000000 * np.einsum('UV,UWXA,WYXZ,YAVZ->', gamma1['b'], h['bb'][A,A,A,V], lambda2['bb'], t['bb'][pA,pV,hA,hA], optimize=True)
	
	O['0'] += scale * +0.12500000 * np.einsum('IJUV,WXUV,WXIJ->', h['bb'][C,C,A,A], lambda2['bb'], t['bb'][pA,pA,hC,hC], optimize=True)
	O['0'] += scale * +0.25000000 * np.einsum('IJAB,ABIJ->', h['bb'][C,C,V,V], t['bb'][pV,pV,hC,hC], optimize=True)
	O['0'] += scale * +0.25000000 * np.einsum('IUVW,UXYVWZ,XYIZ->', h['bb'][C,A,A,A], lambda3['bbb'], t['bb'][pA,pA,hC,hA], optimize=True)
	O['0'] += scale * +1.00000000 * np.einsum('IUVA,UWVX,WAIX->', h['bb'][C,A,A,V], lambda2['bb'], t['bb'][pA,pV,hC,hA], optimize=True)
	O['0'] += scale * -0.25000000 * np.einsum('UVWA,UVXWYZ,XAYZ->', h['bb'][A,A,A,V], lambda3['bbb'], t['bb'][pA,pV,hA,hA], optimize=True)
	O['0'] += scale * +0.12500000 * np.einsum('UVAB,UVWX,ABWX->', h['bb'][A,A,V,V], lambda2['bb'], t['bb'][pV,pV,hA,hA], optimize=True)

	t1 = time.time()
	if verbose: print("h2c_t2c_c0 took {:.4f} seconds to run.".format(t1-t0))

	return O
