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

	
	
	
	
	O['ab'][a,A,a,A] += scale * -1.00000000 * np.einsum('iu,vUiV->vUuV', h['a'][c,a], t['ab'][pa,pA,hc,hA], optimize='optimal')
	O['ab'][a,A,a,A] += scale * +1.00000000 * np.einsum('ua,aUvV->uUvV', h['a'][a,v], t['ab'][pv,pA,ha,hA], optimize='optimal')
	
	
	O['ab'][c,A,a,A] += scale * +1.00000000 * np.einsum('ia,aUuV->iUuV', h['a'][c,v], t['ab'][pv,pA,ha,hA], optimize='optimal')
	O['ab'][v,A,a,A] += scale * -1.00000000 * np.einsum('uv,uw,aUvV->aUwV', eta1['a'], h['a'][a,a], t['ab'][pv,pA,ha,hA], optimize='optimal')
	
	O['ab'][v,A,a,A] += scale * -1.00000000 * np.einsum('uv,uw,aUvV->aUwV', gamma1['a'], h['a'][a,a], t['ab'][pv,pA,ha,hA], optimize='optimal')
	
	O['ab'][v,A,a,A] += scale * -1.00000000 * np.einsum('iu,aUiV->aUuV', h['a'][c,a], t['ab'][pv,pA,hc,hA], optimize='optimal')
	O['ab'][v,A,a,A] += scale * +1.00000000 * np.einsum('ab,bUuV->aUuV', h['a'][v,v], t['ab'][pv,pA,ha,hA], optimize='optimal')
	O['ab'][a,V,a,A] += scale * +1.00000000 * np.einsum('uv,wv,uAxU->wAxU', eta1['a'], h['a'][a,a], t['ab'][pa,pV,ha,hA], optimize='optimal')
	O['ab'][a,V,a,A] += scale * -1.00000000 * np.einsum('uv,uw,xAvU->xAwU', eta1['a'], h['a'][a,a], t['ab'][pa,pV,ha,hA], optimize='optimal')
	O['ab'][a,V,a,A] += scale * +1.00000000 * np.einsum('uv,wv,uAxU->wAxU', gamma1['a'], h['a'][a,a], t['ab'][pa,pV,ha,hA], optimize='optimal')
	O['ab'][a,V,a,A] += scale * -1.00000000 * np.einsum('uv,uw,xAvU->xAwU', gamma1['a'], h['a'][a,a], t['ab'][pa,pV,ha,hA], optimize='optimal')
	O['ab'][a,V,a,A] += scale * -1.00000000 * np.einsum('iu,vAiU->vAuU', h['a'][c,a], t['ab'][pa,pV,hc,hA], optimize='optimal')
	O['ab'][a,V,a,A] += scale * +1.00000000 * np.einsum('ua,aAvU->uAvU', h['a'][a,v], t['ab'][pv,pV,ha,hA], optimize='optimal')
	O['ab'][c,V,a,A] += scale * +1.00000000 * np.einsum('uv,iv,uAwU->iAwU', eta1['a'], h['a'][c,a], t['ab'][pa,pV,ha,hA], optimize='optimal')
	O['ab'][c,V,a,A] += scale * +1.00000000 * np.einsum('uv,iv,uAwU->iAwU', gamma1['a'], h['a'][c,a], t['ab'][pa,pV,ha,hA], optimize='optimal')
	O['ab'][c,V,a,A] += scale * +1.00000000 * np.einsum('ia,aAuU->iAuU', h['a'][c,v], t['ab'][pv,pV,ha,hA], optimize='optimal')
	O['ab'][v,V,a,A] += scale * -1.00000000 * np.einsum('uv,uw,aAvU->aAwU', eta1['a'], h['a'][a,a], t['ab'][pv,pV,ha,hA], optimize='optimal')
	O['ab'][v,V,a,A] += scale * +1.00000000 * np.einsum('uv,av,uAwU->aAwU', eta1['a'], h['a'][v,a], t['ab'][pa,pV,ha,hA], optimize='optimal')
	O['ab'][v,V,a,A] += scale * -1.00000000 * np.einsum('uv,uw,aAvU->aAwU', gamma1['a'], h['a'][a,a], t['ab'][pv,pV,ha,hA], optimize='optimal')
	O['ab'][v,V,a,A] += scale * +1.00000000 * np.einsum('uv,av,uAwU->aAwU', gamma1['a'], h['a'][v,a], t['ab'][pa,pV,ha,hA], optimize='optimal')
	O['ab'][v,V,a,A] += scale * -1.00000000 * np.einsum('iu,aAiU->aAuU', h['a'][c,a], t['ab'][pv,pV,hc,hA], optimize='optimal')
	O['ab'][v,V,a,A] += scale * +1.00000000 * np.einsum('ab,bAuU->aAuU', h['a'][v,v], t['ab'][pv,pV,ha,hA], optimize='optimal')
	O['ab'][a,A,a,C] += scale * +1.00000000 * np.einsum('uv,wv,uUxI->wUxI', eta1['a'], h['a'][a,a], t['ab'][pa,pA,ha,hC], optimize='optimal')
	O['ab'][a,A,a,C] += scale * -1.00000000 * np.einsum('uv,uw,xUvI->xUwI', eta1['a'], h['a'][a,a], t['ab'][pa,pA,ha,hC], optimize='optimal')
	O['ab'][a,A,a,C] += scale * +1.00000000 * np.einsum('uv,wv,uUxI->wUxI', gamma1['a'], h['a'][a,a], t['ab'][pa,pA,ha,hC], optimize='optimal')
	O['ab'][a,A,a,C] += scale * -1.00000000 * np.einsum('uv,uw,xUvI->xUwI', gamma1['a'], h['a'][a,a], t['ab'][pa,pA,ha,hC], optimize='optimal')
	O['ab'][a,A,a,C] += scale * -1.00000000 * np.einsum('iu,vUiI->vUuI', h['a'][c,a], t['ab'][pa,pA,hc,hC], optimize='optimal')
	O['ab'][a,A,a,C] += scale * +1.00000000 * np.einsum('ua,aUvI->uUvI', h['a'][a,v], t['ab'][pv,pA,ha,hC], optimize='optimal')
	O['ab'][c,A,a,C] += scale * +1.00000000 * np.einsum('uv,iv,uUwI->iUwI', eta1['a'], h['a'][c,a], t['ab'][pa,pA,ha,hC], optimize='optimal')
	O['ab'][c,A,a,C] += scale * +1.00000000 * np.einsum('uv,iv,uUwI->iUwI', gamma1['a'], h['a'][c,a], t['ab'][pa,pA,ha,hC], optimize='optimal')
	O['ab'][c,A,a,C] += scale * +1.00000000 * np.einsum('ia,aUuI->iUuI', h['a'][c,v], t['ab'][pv,pA,ha,hC], optimize='optimal')
	O['ab'][v,A,a,C] += scale * -1.00000000 * np.einsum('uv,uw,aUvI->aUwI', eta1['a'], h['a'][a,a], t['ab'][pv,pA,ha,hC], optimize='optimal')
	O['ab'][v,A,a,C] += scale * +1.00000000 * np.einsum('uv,av,uUwI->aUwI', eta1['a'], h['a'][v,a], t['ab'][pa,pA,ha,hC], optimize='optimal')
	O['ab'][v,A,a,C] += scale * -1.00000000 * np.einsum('uv,uw,aUvI->aUwI', gamma1['a'], h['a'][a,a], t['ab'][pv,pA,ha,hC], optimize='optimal')
	O['ab'][v,A,a,C] += scale * +1.00000000 * np.einsum('uv,av,uUwI->aUwI', gamma1['a'], h['a'][v,a], t['ab'][pa,pA,ha,hC], optimize='optimal')
	O['ab'][v,A,a,C] += scale * -1.00000000 * np.einsum('iu,aUiI->aUuI', h['a'][c,a], t['ab'][pv,pA,hc,hC], optimize='optimal')
	O['ab'][v,A,a,C] += scale * +1.00000000 * np.einsum('ab,bUuI->aUuI', h['a'][v,v], t['ab'][pv,pA,ha,hC], optimize='optimal')
	O['ab'][a,V,a,C] += scale * +1.00000000 * np.einsum('uv,wv,uAxI->wAxI', eta1['a'], h['a'][a,a], t['ab'][pa,pV,ha,hC], optimize='optimal')
	O['ab'][a,V,a,C] += scale * -1.00000000 * np.einsum('uv,uw,xAvI->xAwI', eta1['a'], h['a'][a,a], t['ab'][pa,pV,ha,hC], optimize='optimal')
	O['ab'][a,V,a,C] += scale * +1.00000000 * np.einsum('uv,wv,uAxI->wAxI', gamma1['a'], h['a'][a,a], t['ab'][pa,pV,ha,hC], optimize='optimal')
	O['ab'][a,V,a,C] += scale * -1.00000000 * np.einsum('uv,uw,xAvI->xAwI', gamma1['a'], h['a'][a,a], t['ab'][pa,pV,ha,hC], optimize='optimal')
	O['ab'][a,V,a,C] += scale * -1.00000000 * np.einsum('iu,vAiI->vAuI', h['a'][c,a], t['ab'][pa,pV,hc,hC], optimize='optimal')
	O['ab'][a,V,a,C] += scale * +1.00000000 * np.einsum('ua,aAvI->uAvI', h['a'][a,v], t['ab'][pv,pV,ha,hC], optimize='optimal')
	O['ab'][c,V,a,C] += scale * +1.00000000 * np.einsum('uv,iv,uAwI->iAwI', eta1['a'], h['a'][c,a], t['ab'][pa,pV,ha,hC], optimize='optimal')
	O['ab'][c,V,a,C] += scale * +1.00000000 * np.einsum('uv,iv,uAwI->iAwI', gamma1['a'], h['a'][c,a], t['ab'][pa,pV,ha,hC], optimize='optimal')
	O['ab'][c,V,a,C] += scale * +1.00000000 * np.einsum('ia,aAuI->iAuI', h['a'][c,v], t['ab'][pv,pV,ha,hC], optimize='optimal')
	O['ab'][v,V,a,C] += scale * -1.00000000 * np.einsum('uv,uw,aAvI->aAwI', eta1['a'], h['a'][a,a], t['ab'][pv,pV,ha,hC], optimize='optimal')
	O['ab'][v,V,a,C] += scale * +1.00000000 * np.einsum('uv,av,uAwI->aAwI', eta1['a'], h['a'][v,a], t['ab'][pa,pV,ha,hC], optimize='optimal')
	O['ab'][v,V,a,C] += scale * -1.00000000 * np.einsum('uv,uw,aAvI->aAwI', gamma1['a'], h['a'][a,a], t['ab'][pv,pV,ha,hC], optimize='optimal')
	O['ab'][v,V,a,C] += scale * +1.00000000 * np.einsum('uv,av,uAwI->aAwI', gamma1['a'], h['a'][v,a], t['ab'][pa,pV,ha,hC], optimize='optimal')
	O['ab'][v,V,a,C] += scale * -1.00000000 * np.einsum('iu,aAiI->aAuI', h['a'][c,a], t['ab'][pv,pV,hc,hC], optimize='optimal')
	O['ab'][v,V,a,C] += scale * +1.00000000 * np.einsum('ab,bAuI->aAuI', h['a'][v,v], t['ab'][pv,pV,ha,hC], optimize='optimal')
	O['ab'][a,A,c,A] += scale * +1.00000000 * np.einsum('uv,wv,uUiV->wUiV', eta1['a'], h['a'][a,a], t['ab'][pa,pA,hc,hA], optimize='optimal')
	
	O['ab'][a,A,c,A] += scale * +1.00000000 * np.einsum('uv,wv,uUiV->wUiV', gamma1['a'], h['a'][a,a], t['ab'][pa,pA,hc,hA], optimize='optimal')
	
	O['ab'][a,A,c,A] += scale * -1.00000000 * np.einsum('ij,uUiV->uUjV', h['a'][c,c], t['ab'][pa,pA,hc,hA], optimize='optimal')
	O['ab'][a,A,c,A] += scale * +1.00000000 * np.einsum('ua,aUiV->uUiV', h['a'][a,v], t['ab'][pv,pA,hc,hA], optimize='optimal')
	O['ab'][c,A,c,A] += scale * +1.00000000 * np.einsum('uv,iv,uUjV->iUjV', eta1['a'], h['a'][c,a], t['ab'][pa,pA,hc,hA], optimize='optimal')
	O['ab'][c,A,c,A] += scale * +1.00000000 * np.einsum('uv,iv,uUjV->iUjV', gamma1['a'], h['a'][c,a], t['ab'][pa,pA,hc,hA], optimize='optimal')
	O['ab'][c,A,c,A] += scale * +1.00000000 * np.einsum('ia,aUjV->iUjV', h['a'][c,v], t['ab'][pv,pA,hc,hA], optimize='optimal')
	O['ab'][v,A,c,A] += scale * -1.00000000 * np.einsum('uv,ui,aUvV->aUiV', eta1['a'], h['a'][a,c], t['ab'][pv,pA,ha,hA], optimize='optimal')
	O['ab'][v,A,c,A] += scale * +1.00000000 * np.einsum('uv,av,uUiV->aUiV', eta1['a'], h['a'][v,a], t['ab'][pa,pA,hc,hA], optimize='optimal')
	O['ab'][v,A,c,A] += scale * -1.00000000 * np.einsum('uv,ui,aUvV->aUiV', gamma1['a'], h['a'][a,c], t['ab'][pv,pA,ha,hA], optimize='optimal')
	O['ab'][v,A,c,A] += scale * +1.00000000 * np.einsum('uv,av,uUiV->aUiV', gamma1['a'], h['a'][v,a], t['ab'][pa,pA,hc,hA], optimize='optimal')
	O['ab'][v,A,c,A] += scale * -1.00000000 * np.einsum('ij,aUiV->aUjV', h['a'][c,c], t['ab'][pv,pA,hc,hA], optimize='optimal')
	O['ab'][v,A,c,A] += scale * +1.00000000 * np.einsum('ab,bUiV->aUiV', h['a'][v,v], t['ab'][pv,pA,hc,hA], optimize='optimal')
	O['ab'][a,V,c,A] += scale * +1.00000000 * np.einsum('uv,wv,uAiU->wAiU', eta1['a'], h['a'][a,a], t['ab'][pa,pV,hc,hA], optimize='optimal')
	O['ab'][a,V,c,A] += scale * -1.00000000 * np.einsum('uv,ui,wAvU->wAiU', eta1['a'], h['a'][a,c], t['ab'][pa,pV,ha,hA], optimize='optimal')
	O['ab'][a,V,c,A] += scale * +1.00000000 * np.einsum('uv,wv,uAiU->wAiU', gamma1['a'], h['a'][a,a], t['ab'][pa,pV,hc,hA], optimize='optimal')
	O['ab'][a,V,c,A] += scale * -1.00000000 * np.einsum('uv,ui,wAvU->wAiU', gamma1['a'], h['a'][a,c], t['ab'][pa,pV,ha,hA], optimize='optimal')
	O['ab'][a,V,c,A] += scale * -1.00000000 * np.einsum('ij,uAiU->uAjU', h['a'][c,c], t['ab'][pa,pV,hc,hA], optimize='optimal')
	O['ab'][a,V,c,A] += scale * +1.00000000 * np.einsum('ua,aAiU->uAiU', h['a'][a,v], t['ab'][pv,pV,hc,hA], optimize='optimal')
	O['ab'][c,V,c,A] += scale * +1.00000000 * np.einsum('uv,iv,uAjU->iAjU', eta1['a'], h['a'][c,a], t['ab'][pa,pV,hc,hA], optimize='optimal')
	O['ab'][c,V,c,A] += scale * +1.00000000 * np.einsum('uv,iv,uAjU->iAjU', gamma1['a'], h['a'][c,a], t['ab'][pa,pV,hc,hA], optimize='optimal')
	O['ab'][c,V,c,A] += scale * +1.00000000 * np.einsum('ia,aAjU->iAjU', h['a'][c,v], t['ab'][pv,pV,hc,hA], optimize='optimal')
	O['ab'][v,V,c,A] += scale * -1.00000000 * np.einsum('uv,ui,aAvU->aAiU', eta1['a'], h['a'][a,c], t['ab'][pv,pV,ha,hA], optimize='optimal')
	O['ab'][v,V,c,A] += scale * +1.00000000 * np.einsum('uv,av,uAiU->aAiU', eta1['a'], h['a'][v,a], t['ab'][pa,pV,hc,hA], optimize='optimal')
	O['ab'][v,V,c,A] += scale * -1.00000000 * np.einsum('uv,ui,aAvU->aAiU', gamma1['a'], h['a'][a,c], t['ab'][pv,pV,ha,hA], optimize='optimal')
	O['ab'][v,V,c,A] += scale * +1.00000000 * np.einsum('uv,av,uAiU->aAiU', gamma1['a'], h['a'][v,a], t['ab'][pa,pV,hc,hA], optimize='optimal')
	O['ab'][v,V,c,A] += scale * -1.00000000 * np.einsum('ij,aAiU->aAjU', h['a'][c,c], t['ab'][pv,pV,hc,hA], optimize='optimal')
	O['ab'][v,V,c,A] += scale * +1.00000000 * np.einsum('ab,bAiU->aAiU', h['a'][v,v], t['ab'][pv,pV,hc,hA], optimize='optimal')
	O['ab'][a,A,c,C] += scale * +1.00000000 * np.einsum('uv,wv,uUiI->wUiI', eta1['a'], h['a'][a,a], t['ab'][pa,pA,hc,hC], optimize='optimal')
	O['ab'][a,A,c,C] += scale * -1.00000000 * np.einsum('uv,ui,wUvI->wUiI', eta1['a'], h['a'][a,c], t['ab'][pa,pA,ha,hC], optimize='optimal')
	O['ab'][a,A,c,C] += scale * +1.00000000 * np.einsum('uv,wv,uUiI->wUiI', gamma1['a'], h['a'][a,a], t['ab'][pa,pA,hc,hC], optimize='optimal')
	O['ab'][a,A,c,C] += scale * -1.00000000 * np.einsum('uv,ui,wUvI->wUiI', gamma1['a'], h['a'][a,c], t['ab'][pa,pA,ha,hC], optimize='optimal')
	O['ab'][a,A,c,C] += scale * -1.00000000 * np.einsum('ij,uUiI->uUjI', h['a'][c,c], t['ab'][pa,pA,hc,hC], optimize='optimal')
	O['ab'][a,A,c,C] += scale * +1.00000000 * np.einsum('ua,aUiI->uUiI', h['a'][a,v], t['ab'][pv,pA,hc,hC], optimize='optimal')
	O['ab'][c,A,c,C] += scale * +1.00000000 * np.einsum('uv,iv,uUjI->iUjI', eta1['a'], h['a'][c,a], t['ab'][pa,pA,hc,hC], optimize='optimal')
	O['ab'][c,A,c,C] += scale * +1.00000000 * np.einsum('uv,iv,uUjI->iUjI', gamma1['a'], h['a'][c,a], t['ab'][pa,pA,hc,hC], optimize='optimal')
	O['ab'][c,A,c,C] += scale * +1.00000000 * np.einsum('ia,aUjI->iUjI', h['a'][c,v], t['ab'][pv,pA,hc,hC], optimize='optimal')
	O['ab'][v,A,c,C] += scale * -1.00000000 * np.einsum('uv,ui,aUvI->aUiI', eta1['a'], h['a'][a,c], t['ab'][pv,pA,ha,hC], optimize='optimal')
	O['ab'][v,A,c,C] += scale * +1.00000000 * np.einsum('uv,av,uUiI->aUiI', eta1['a'], h['a'][v,a], t['ab'][pa,pA,hc,hC], optimize='optimal')
	O['ab'][v,A,c,C] += scale * -1.00000000 * np.einsum('uv,ui,aUvI->aUiI', gamma1['a'], h['a'][a,c], t['ab'][pv,pA,ha,hC], optimize='optimal')
	O['ab'][v,A,c,C] += scale * +1.00000000 * np.einsum('uv,av,uUiI->aUiI', gamma1['a'], h['a'][v,a], t['ab'][pa,pA,hc,hC], optimize='optimal')
	O['ab'][v,A,c,C] += scale * -1.00000000 * np.einsum('ij,aUiI->aUjI', h['a'][c,c], t['ab'][pv,pA,hc,hC], optimize='optimal')
	O['ab'][v,A,c,C] += scale * +1.00000000 * np.einsum('ab,bUiI->aUiI', h['a'][v,v], t['ab'][pv,pA,hc,hC], optimize='optimal')
	O['ab'][a,V,c,C] += scale * +1.00000000 * np.einsum('uv,wv,uAiI->wAiI', eta1['a'], h['a'][a,a], t['ab'][pa,pV,hc,hC], optimize='optimal')
	O['ab'][a,V,c,C] += scale * -1.00000000 * np.einsum('uv,ui,wAvI->wAiI', eta1['a'], h['a'][a,c], t['ab'][pa,pV,ha,hC], optimize='optimal')
	O['ab'][a,V,c,C] += scale * +1.00000000 * np.einsum('uv,wv,uAiI->wAiI', gamma1['a'], h['a'][a,a], t['ab'][pa,pV,hc,hC], optimize='optimal')
	O['ab'][a,V,c,C] += scale * -1.00000000 * np.einsum('uv,ui,wAvI->wAiI', gamma1['a'], h['a'][a,c], t['ab'][pa,pV,ha,hC], optimize='optimal')
	O['ab'][a,V,c,C] += scale * -1.00000000 * np.einsum('ij,uAiI->uAjI', h['a'][c,c], t['ab'][pa,pV,hc,hC], optimize='optimal')
	O['ab'][a,V,c,C] += scale * +1.00000000 * np.einsum('ua,aAiI->uAiI', h['a'][a,v], t['ab'][pv,pV,hc,hC], optimize='optimal')
	O['ab'][c,V,c,C] += scale * +1.00000000 * np.einsum('uv,iv,uAjI->iAjI', eta1['a'], h['a'][c,a], t['ab'][pa,pV,hc,hC], optimize='optimal')
	O['ab'][c,V,c,C] += scale * +1.00000000 * np.einsum('uv,iv,uAjI->iAjI', gamma1['a'], h['a'][c,a], t['ab'][pa,pV,hc,hC], optimize='optimal')
	O['ab'][c,V,c,C] += scale * +1.00000000 * np.einsum('ia,aAjI->iAjI', h['a'][c,v], t['ab'][pv,pV,hc,hC], optimize='optimal')
	O['ab'][v,V,c,C] += scale * -1.00000000 * np.einsum('uv,ui,aAvI->aAiI', eta1['a'], h['a'][a,c], t['ab'][pv,pV,ha,hC], optimize='optimal')
	O['ab'][v,V,c,C] += scale * +1.00000000 * np.einsum('uv,av,uAiI->aAiI', eta1['a'], h['a'][v,a], t['ab'][pa,pV,hc,hC], optimize='optimal')
	O['ab'][v,V,c,C] += scale * -1.00000000 * np.einsum('uv,ui,aAvI->aAiI', gamma1['a'], h['a'][a,c], t['ab'][pv,pV,ha,hC], optimize='optimal')
	O['ab'][v,V,c,C] += scale * +1.00000000 * np.einsum('uv,av,uAiI->aAiI', gamma1['a'], h['a'][v,a], t['ab'][pa,pV,hc,hC], optimize='optimal')
	O['ab'][v,V,c,C] += scale * -1.00000000 * np.einsum('ij,aAiI->aAjI', h['a'][c,c], t['ab'][pv,pV,hc,hC], optimize='optimal')
	O['ab'][v,V,c,C] += scale * +1.00000000 * np.einsum('ab,bAiI->aAiI', h['a'][v,v], t['ab'][pv,pV,hc,hC], optimize='optimal')
	
	
	O['ab'][a,A,v,A] += scale * -1.00000000 * np.einsum('ia,uUiV->uUaV', h['a'][c,v], t['ab'][pa,pA,hc,hA], optimize='optimal')
	O['ab'][v,A,v,A] += scale * -1.00000000 * np.einsum('uv,ua,bUvV->bUaV', eta1['a'], h['a'][a,v], t['ab'][pv,pA,ha,hA], optimize='optimal')
	O['ab'][v,A,v,A] += scale * -1.00000000 * np.einsum('uv,ua,bUvV->bUaV', gamma1['a'], h['a'][a,v], t['ab'][pv,pA,ha,hA], optimize='optimal')
	O['ab'][v,A,v,A] += scale * -1.00000000 * np.einsum('ia,bUiV->bUaV', h['a'][c,v], t['ab'][pv,pA,hc,hA], optimize='optimal')
	O['ab'][a,V,v,A] += scale * -1.00000000 * np.einsum('uv,ua,wAvU->wAaU', eta1['a'], h['a'][a,v], t['ab'][pa,pV,ha,hA], optimize='optimal')
	O['ab'][a,V,v,A] += scale * -1.00000000 * np.einsum('uv,ua,wAvU->wAaU', gamma1['a'], h['a'][a,v], t['ab'][pa,pV,ha,hA], optimize='optimal')
	O['ab'][a,V,v,A] += scale * -1.00000000 * np.einsum('ia,uAiU->uAaU', h['a'][c,v], t['ab'][pa,pV,hc,hA], optimize='optimal')
	O['ab'][v,V,v,A] += scale * -1.00000000 * np.einsum('uv,ua,bAvU->bAaU', eta1['a'], h['a'][a,v], t['ab'][pv,pV,ha,hA], optimize='optimal')
	O['ab'][v,V,v,A] += scale * -1.00000000 * np.einsum('uv,ua,bAvU->bAaU', gamma1['a'], h['a'][a,v], t['ab'][pv,pV,ha,hA], optimize='optimal')
	O['ab'][v,V,v,A] += scale * -1.00000000 * np.einsum('ia,bAiU->bAaU', h['a'][c,v], t['ab'][pv,pV,hc,hA], optimize='optimal')
	O['ab'][a,A,v,C] += scale * -1.00000000 * np.einsum('uv,ua,wUvI->wUaI', eta1['a'], h['a'][a,v], t['ab'][pa,pA,ha,hC], optimize='optimal')
	O['ab'][a,A,v,C] += scale * -1.00000000 * np.einsum('uv,ua,wUvI->wUaI', gamma1['a'], h['a'][a,v], t['ab'][pa,pA,ha,hC], optimize='optimal')
	O['ab'][a,A,v,C] += scale * -1.00000000 * np.einsum('ia,uUiI->uUaI', h['a'][c,v], t['ab'][pa,pA,hc,hC], optimize='optimal')
	O['ab'][v,A,v,C] += scale * -1.00000000 * np.einsum('uv,ua,bUvI->bUaI', eta1['a'], h['a'][a,v], t['ab'][pv,pA,ha,hC], optimize='optimal')
	O['ab'][v,A,v,C] += scale * -1.00000000 * np.einsum('uv,ua,bUvI->bUaI', gamma1['a'], h['a'][a,v], t['ab'][pv,pA,ha,hC], optimize='optimal')
	O['ab'][v,A,v,C] += scale * -1.00000000 * np.einsum('ia,bUiI->bUaI', h['a'][c,v], t['ab'][pv,pA,hc,hC], optimize='optimal')
	O['ab'][a,V,v,C] += scale * -1.00000000 * np.einsum('uv,ua,wAvI->wAaI', eta1['a'], h['a'][a,v], t['ab'][pa,pV,ha,hC], optimize='optimal')
	O['ab'][a,V,v,C] += scale * -1.00000000 * np.einsum('uv,ua,wAvI->wAaI', gamma1['a'], h['a'][a,v], t['ab'][pa,pV,ha,hC], optimize='optimal')
	O['ab'][a,V,v,C] += scale * -1.00000000 * np.einsum('ia,uAiI->uAaI', h['a'][c,v], t['ab'][pa,pV,hc,hC], optimize='optimal')
	O['ab'][v,V,v,C] += scale * -1.00000000 * np.einsum('uv,ua,bAvI->bAaI', eta1['a'], h['a'][a,v], t['ab'][pv,pV,ha,hC], optimize='optimal')
	O['ab'][v,V,v,C] += scale * -1.00000000 * np.einsum('uv,ua,bAvI->bAaI', gamma1['a'], h['a'][a,v], t['ab'][pv,pV,ha,hC], optimize='optimal')
	O['ab'][v,V,v,C] += scale * -1.00000000 * np.einsum('ia,bAiI->bAaI', h['a'][c,v], t['ab'][pv,pV,hc,hC], optimize='optimal')

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

	
	
	
	
	O['ab'][a,A,a,A] += scale * -1.00000000 * np.einsum('IU,uVvI->uVvU', h['b'][C,A], t['ab'][pa,pA,ha,hC], optimize='optimal')
	O['ab'][a,A,a,A] += scale * +1.00000000 * np.einsum('UA,uAvV->uUvV', h['b'][A,V], t['ab'][pa,pV,ha,hA], optimize='optimal')
	O['ab'][v,A,a,A] += scale * +1.00000000 * np.einsum('UV,WV,aUuX->aWuX', eta1['b'], h['b'][A,A], t['ab'][pv,pA,ha,hA], optimize='optimal')
	O['ab'][v,A,a,A] += scale * -1.00000000 * np.einsum('UV,UW,aXuV->aXuW', eta1['b'], h['b'][A,A], t['ab'][pv,pA,ha,hA], optimize='optimal')
	O['ab'][v,A,a,A] += scale * +1.00000000 * np.einsum('UV,WV,aUuX->aWuX', gamma1['b'], h['b'][A,A], t['ab'][pv,pA,ha,hA], optimize='optimal')
	O['ab'][v,A,a,A] += scale * -1.00000000 * np.einsum('UV,UW,aXuV->aXuW', gamma1['b'], h['b'][A,A], t['ab'][pv,pA,ha,hA], optimize='optimal')
	O['ab'][v,A,a,A] += scale * -1.00000000 * np.einsum('IU,aVuI->aVuU', h['b'][C,A], t['ab'][pv,pA,ha,hC], optimize='optimal')
	O['ab'][v,A,a,A] += scale * +1.00000000 * np.einsum('UA,aAuV->aUuV', h['b'][A,V], t['ab'][pv,pV,ha,hA], optimize='optimal')
	
	
	O['ab'][a,C,a,A] += scale * +1.00000000 * np.einsum('IA,uAvU->uIvU', h['b'][C,V], t['ab'][pa,pV,ha,hA], optimize='optimal')
	O['ab'][v,C,a,A] += scale * +1.00000000 * np.einsum('UV,IV,aUuW->aIuW', eta1['b'], h['b'][C,A], t['ab'][pv,pA,ha,hA], optimize='optimal')
	O['ab'][v,C,a,A] += scale * +1.00000000 * np.einsum('UV,IV,aUuW->aIuW', gamma1['b'], h['b'][C,A], t['ab'][pv,pA,ha,hA], optimize='optimal')
	O['ab'][v,C,a,A] += scale * +1.00000000 * np.einsum('IA,aAuU->aIuU', h['b'][C,V], t['ab'][pv,pV,ha,hA], optimize='optimal')
	O['ab'][a,V,a,A] += scale * -1.00000000 * np.einsum('UV,UW,uAvV->uAvW', eta1['b'], h['b'][A,A], t['ab'][pa,pV,ha,hA], optimize='optimal')
	
	O['ab'][a,V,a,A] += scale * -1.00000000 * np.einsum('UV,UW,uAvV->uAvW', gamma1['b'], h['b'][A,A], t['ab'][pa,pV,ha,hA], optimize='optimal')
	
	O['ab'][a,V,a,A] += scale * -1.00000000 * np.einsum('IU,uAvI->uAvU', h['b'][C,A], t['ab'][pa,pV,ha,hC], optimize='optimal')
	O['ab'][a,V,a,A] += scale * +1.00000000 * np.einsum('AB,uBvU->uAvU', h['b'][V,V], t['ab'][pa,pV,ha,hA], optimize='optimal')
	O['ab'][v,V,a,A] += scale * -1.00000000 * np.einsum('UV,UW,aAuV->aAuW', eta1['b'], h['b'][A,A], t['ab'][pv,pV,ha,hA], optimize='optimal')
	O['ab'][v,V,a,A] += scale * +1.00000000 * np.einsum('UV,AV,aUuW->aAuW', eta1['b'], h['b'][V,A], t['ab'][pv,pA,ha,hA], optimize='optimal')
	O['ab'][v,V,a,A] += scale * -1.00000000 * np.einsum('UV,UW,aAuV->aAuW', gamma1['b'], h['b'][A,A], t['ab'][pv,pV,ha,hA], optimize='optimal')
	O['ab'][v,V,a,A] += scale * +1.00000000 * np.einsum('UV,AV,aUuW->aAuW', gamma1['b'], h['b'][V,A], t['ab'][pv,pA,ha,hA], optimize='optimal')
	O['ab'][v,V,a,A] += scale * -1.00000000 * np.einsum('IU,aAuI->aAuU', h['b'][C,A], t['ab'][pv,pV,ha,hC], optimize='optimal')
	O['ab'][v,V,a,A] += scale * +1.00000000 * np.einsum('AB,aBuU->aAuU', h['b'][V,V], t['ab'][pv,pV,ha,hA], optimize='optimal')
	O['ab'][a,A,a,C] += scale * +1.00000000 * np.einsum('UV,WV,uUvI->uWvI', eta1['b'], h['b'][A,A], t['ab'][pa,pA,ha,hC], optimize='optimal')
	
	O['ab'][a,A,a,C] += scale * +1.00000000 * np.einsum('UV,WV,uUvI->uWvI', gamma1['b'], h['b'][A,A], t['ab'][pa,pA,ha,hC], optimize='optimal')
	
	O['ab'][a,A,a,C] += scale * -1.00000000 * np.einsum('IJ,uUvI->uUvJ', h['b'][C,C], t['ab'][pa,pA,ha,hC], optimize='optimal')
	O['ab'][a,A,a,C] += scale * +1.00000000 * np.einsum('UA,uAvI->uUvI', h['b'][A,V], t['ab'][pa,pV,ha,hC], optimize='optimal')
	O['ab'][v,A,a,C] += scale * +1.00000000 * np.einsum('UV,WV,aUuI->aWuI', eta1['b'], h['b'][A,A], t['ab'][pv,pA,ha,hC], optimize='optimal')
	O['ab'][v,A,a,C] += scale * -1.00000000 * np.einsum('UV,UI,aWuV->aWuI', eta1['b'], h['b'][A,C], t['ab'][pv,pA,ha,hA], optimize='optimal')
	O['ab'][v,A,a,C] += scale * +1.00000000 * np.einsum('UV,WV,aUuI->aWuI', gamma1['b'], h['b'][A,A], t['ab'][pv,pA,ha,hC], optimize='optimal')
	O['ab'][v,A,a,C] += scale * -1.00000000 * np.einsum('UV,UI,aWuV->aWuI', gamma1['b'], h['b'][A,C], t['ab'][pv,pA,ha,hA], optimize='optimal')
	O['ab'][v,A,a,C] += scale * -1.00000000 * np.einsum('IJ,aUuI->aUuJ', h['b'][C,C], t['ab'][pv,pA,ha,hC], optimize='optimal')
	O['ab'][v,A,a,C] += scale * +1.00000000 * np.einsum('UA,aAuI->aUuI', h['b'][A,V], t['ab'][pv,pV,ha,hC], optimize='optimal')
	O['ab'][a,C,a,C] += scale * +1.00000000 * np.einsum('UV,IV,uUvJ->uIvJ', eta1['b'], h['b'][C,A], t['ab'][pa,pA,ha,hC], optimize='optimal')
	O['ab'][a,C,a,C] += scale * +1.00000000 * np.einsum('UV,IV,uUvJ->uIvJ', gamma1['b'], h['b'][C,A], t['ab'][pa,pA,ha,hC], optimize='optimal')
	O['ab'][a,C,a,C] += scale * +1.00000000 * np.einsum('IA,uAvJ->uIvJ', h['b'][C,V], t['ab'][pa,pV,ha,hC], optimize='optimal')
	O['ab'][v,C,a,C] += scale * +1.00000000 * np.einsum('UV,IV,aUuJ->aIuJ', eta1['b'], h['b'][C,A], t['ab'][pv,pA,ha,hC], optimize='optimal')
	O['ab'][v,C,a,C] += scale * +1.00000000 * np.einsum('UV,IV,aUuJ->aIuJ', gamma1['b'], h['b'][C,A], t['ab'][pv,pA,ha,hC], optimize='optimal')
	O['ab'][v,C,a,C] += scale * +1.00000000 * np.einsum('IA,aAuJ->aIuJ', h['b'][C,V], t['ab'][pv,pV,ha,hC], optimize='optimal')
	O['ab'][a,V,a,C] += scale * -1.00000000 * np.einsum('UV,UI,uAvV->uAvI', eta1['b'], h['b'][A,C], t['ab'][pa,pV,ha,hA], optimize='optimal')
	O['ab'][a,V,a,C] += scale * +1.00000000 * np.einsum('UV,AV,uUvI->uAvI', eta1['b'], h['b'][V,A], t['ab'][pa,pA,ha,hC], optimize='optimal')
	O['ab'][a,V,a,C] += scale * -1.00000000 * np.einsum('UV,UI,uAvV->uAvI', gamma1['b'], h['b'][A,C], t['ab'][pa,pV,ha,hA], optimize='optimal')
	O['ab'][a,V,a,C] += scale * +1.00000000 * np.einsum('UV,AV,uUvI->uAvI', gamma1['b'], h['b'][V,A], t['ab'][pa,pA,ha,hC], optimize='optimal')
	O['ab'][a,V,a,C] += scale * -1.00000000 * np.einsum('IJ,uAvI->uAvJ', h['b'][C,C], t['ab'][pa,pV,ha,hC], optimize='optimal')
	O['ab'][a,V,a,C] += scale * +1.00000000 * np.einsum('AB,uBvI->uAvI', h['b'][V,V], t['ab'][pa,pV,ha,hC], optimize='optimal')
	O['ab'][v,V,a,C] += scale * -1.00000000 * np.einsum('UV,UI,aAuV->aAuI', eta1['b'], h['b'][A,C], t['ab'][pv,pV,ha,hA], optimize='optimal')
	O['ab'][v,V,a,C] += scale * +1.00000000 * np.einsum('UV,AV,aUuI->aAuI', eta1['b'], h['b'][V,A], t['ab'][pv,pA,ha,hC], optimize='optimal')
	O['ab'][v,V,a,C] += scale * -1.00000000 * np.einsum('UV,UI,aAuV->aAuI', gamma1['b'], h['b'][A,C], t['ab'][pv,pV,ha,hA], optimize='optimal')
	O['ab'][v,V,a,C] += scale * +1.00000000 * np.einsum('UV,AV,aUuI->aAuI', gamma1['b'], h['b'][V,A], t['ab'][pv,pA,ha,hC], optimize='optimal')
	O['ab'][v,V,a,C] += scale * -1.00000000 * np.einsum('IJ,aAuI->aAuJ', h['b'][C,C], t['ab'][pv,pV,ha,hC], optimize='optimal')
	O['ab'][v,V,a,C] += scale * +1.00000000 * np.einsum('AB,aBuI->aAuI', h['b'][V,V], t['ab'][pv,pV,ha,hC], optimize='optimal')
	
	
	O['ab'][a,A,a,V] += scale * -1.00000000 * np.einsum('IA,uUvI->uUvA', h['b'][C,V], t['ab'][pa,pA,ha,hC], optimize='optimal')
	O['ab'][v,A,a,V] += scale * -1.00000000 * np.einsum('UV,UA,aWuV->aWuA', eta1['b'], h['b'][A,V], t['ab'][pv,pA,ha,hA], optimize='optimal')
	O['ab'][v,A,a,V] += scale * -1.00000000 * np.einsum('UV,UA,aWuV->aWuA', gamma1['b'], h['b'][A,V], t['ab'][pv,pA,ha,hA], optimize='optimal')
	O['ab'][v,A,a,V] += scale * -1.00000000 * np.einsum('IA,aUuI->aUuA', h['b'][C,V], t['ab'][pv,pA,ha,hC], optimize='optimal')
	O['ab'][a,V,a,V] += scale * -1.00000000 * np.einsum('UV,UA,uBvV->uBvA', eta1['b'], h['b'][A,V], t['ab'][pa,pV,ha,hA], optimize='optimal')
	O['ab'][a,V,a,V] += scale * -1.00000000 * np.einsum('UV,UA,uBvV->uBvA', gamma1['b'], h['b'][A,V], t['ab'][pa,pV,ha,hA], optimize='optimal')
	O['ab'][a,V,a,V] += scale * -1.00000000 * np.einsum('IA,uBvI->uBvA', h['b'][C,V], t['ab'][pa,pV,ha,hC], optimize='optimal')
	O['ab'][v,V,a,V] += scale * -1.00000000 * np.einsum('UV,UA,aBuV->aBuA', eta1['b'], h['b'][A,V], t['ab'][pv,pV,ha,hA], optimize='optimal')
	O['ab'][v,V,a,V] += scale * -1.00000000 * np.einsum('UV,UA,aBuV->aBuA', gamma1['b'], h['b'][A,V], t['ab'][pv,pV,ha,hA], optimize='optimal')
	O['ab'][v,V,a,V] += scale * -1.00000000 * np.einsum('IA,aBuI->aBuA', h['b'][C,V], t['ab'][pv,pV,ha,hC], optimize='optimal')
	O['ab'][a,A,c,A] += scale * +1.00000000 * np.einsum('UV,WV,uUiX->uWiX', eta1['b'], h['b'][A,A], t['ab'][pa,pA,hc,hA], optimize='optimal')
	O['ab'][a,A,c,A] += scale * -1.00000000 * np.einsum('UV,UW,uXiV->uXiW', eta1['b'], h['b'][A,A], t['ab'][pa,pA,hc,hA], optimize='optimal')
	O['ab'][a,A,c,A] += scale * +1.00000000 * np.einsum('UV,WV,uUiX->uWiX', gamma1['b'], h['b'][A,A], t['ab'][pa,pA,hc,hA], optimize='optimal')
	O['ab'][a,A,c,A] += scale * -1.00000000 * np.einsum('UV,UW,uXiV->uXiW', gamma1['b'], h['b'][A,A], t['ab'][pa,pA,hc,hA], optimize='optimal')
	O['ab'][a,A,c,A] += scale * -1.00000000 * np.einsum('IU,uViI->uViU', h['b'][C,A], t['ab'][pa,pA,hc,hC], optimize='optimal')
	O['ab'][a,A,c,A] += scale * +1.00000000 * np.einsum('UA,uAiV->uUiV', h['b'][A,V], t['ab'][pa,pV,hc,hA], optimize='optimal')
	O['ab'][v,A,c,A] += scale * +1.00000000 * np.einsum('UV,WV,aUiX->aWiX', eta1['b'], h['b'][A,A], t['ab'][pv,pA,hc,hA], optimize='optimal')
	O['ab'][v,A,c,A] += scale * -1.00000000 * np.einsum('UV,UW,aXiV->aXiW', eta1['b'], h['b'][A,A], t['ab'][pv,pA,hc,hA], optimize='optimal')
	O['ab'][v,A,c,A] += scale * +1.00000000 * np.einsum('UV,WV,aUiX->aWiX', gamma1['b'], h['b'][A,A], t['ab'][pv,pA,hc,hA], optimize='optimal')
	O['ab'][v,A,c,A] += scale * -1.00000000 * np.einsum('UV,UW,aXiV->aXiW', gamma1['b'], h['b'][A,A], t['ab'][pv,pA,hc,hA], optimize='optimal')
	O['ab'][v,A,c,A] += scale * -1.00000000 * np.einsum('IU,aViI->aViU', h['b'][C,A], t['ab'][pv,pA,hc,hC], optimize='optimal')
	O['ab'][v,A,c,A] += scale * +1.00000000 * np.einsum('UA,aAiV->aUiV', h['b'][A,V], t['ab'][pv,pV,hc,hA], optimize='optimal')
	O['ab'][a,C,c,A] += scale * +1.00000000 * np.einsum('UV,IV,uUiW->uIiW', eta1['b'], h['b'][C,A], t['ab'][pa,pA,hc,hA], optimize='optimal')
	O['ab'][a,C,c,A] += scale * +1.00000000 * np.einsum('UV,IV,uUiW->uIiW', gamma1['b'], h['b'][C,A], t['ab'][pa,pA,hc,hA], optimize='optimal')
	O['ab'][a,C,c,A] += scale * +1.00000000 * np.einsum('IA,uAiU->uIiU', h['b'][C,V], t['ab'][pa,pV,hc,hA], optimize='optimal')
	O['ab'][v,C,c,A] += scale * +1.00000000 * np.einsum('UV,IV,aUiW->aIiW', eta1['b'], h['b'][C,A], t['ab'][pv,pA,hc,hA], optimize='optimal')
	O['ab'][v,C,c,A] += scale * +1.00000000 * np.einsum('UV,IV,aUiW->aIiW', gamma1['b'], h['b'][C,A], t['ab'][pv,pA,hc,hA], optimize='optimal')
	O['ab'][v,C,c,A] += scale * +1.00000000 * np.einsum('IA,aAiU->aIiU', h['b'][C,V], t['ab'][pv,pV,hc,hA], optimize='optimal')
	O['ab'][a,V,c,A] += scale * -1.00000000 * np.einsum('UV,UW,uAiV->uAiW', eta1['b'], h['b'][A,A], t['ab'][pa,pV,hc,hA], optimize='optimal')
	O['ab'][a,V,c,A] += scale * +1.00000000 * np.einsum('UV,AV,uUiW->uAiW', eta1['b'], h['b'][V,A], t['ab'][pa,pA,hc,hA], optimize='optimal')
	O['ab'][a,V,c,A] += scale * -1.00000000 * np.einsum('UV,UW,uAiV->uAiW', gamma1['b'], h['b'][A,A], t['ab'][pa,pV,hc,hA], optimize='optimal')
	O['ab'][a,V,c,A] += scale * +1.00000000 * np.einsum('UV,AV,uUiW->uAiW', gamma1['b'], h['b'][V,A], t['ab'][pa,pA,hc,hA], optimize='optimal')
	O['ab'][a,V,c,A] += scale * -1.00000000 * np.einsum('IU,uAiI->uAiU', h['b'][C,A], t['ab'][pa,pV,hc,hC], optimize='optimal')
	O['ab'][a,V,c,A] += scale * +1.00000000 * np.einsum('AB,uBiU->uAiU', h['b'][V,V], t['ab'][pa,pV,hc,hA], optimize='optimal')
	O['ab'][v,V,c,A] += scale * -1.00000000 * np.einsum('UV,UW,aAiV->aAiW', eta1['b'], h['b'][A,A], t['ab'][pv,pV,hc,hA], optimize='optimal')
	O['ab'][v,V,c,A] += scale * +1.00000000 * np.einsum('UV,AV,aUiW->aAiW', eta1['b'], h['b'][V,A], t['ab'][pv,pA,hc,hA], optimize='optimal')
	O['ab'][v,V,c,A] += scale * -1.00000000 * np.einsum('UV,UW,aAiV->aAiW', gamma1['b'], h['b'][A,A], t['ab'][pv,pV,hc,hA], optimize='optimal')
	O['ab'][v,V,c,A] += scale * +1.00000000 * np.einsum('UV,AV,aUiW->aAiW', gamma1['b'], h['b'][V,A], t['ab'][pv,pA,hc,hA], optimize='optimal')
	O['ab'][v,V,c,A] += scale * -1.00000000 * np.einsum('IU,aAiI->aAiU', h['b'][C,A], t['ab'][pv,pV,hc,hC], optimize='optimal')
	O['ab'][v,V,c,A] += scale * +1.00000000 * np.einsum('AB,aBiU->aAiU', h['b'][V,V], t['ab'][pv,pV,hc,hA], optimize='optimal')
	O['ab'][a,A,c,C] += scale * +1.00000000 * np.einsum('UV,WV,uUiI->uWiI', eta1['b'], h['b'][A,A], t['ab'][pa,pA,hc,hC], optimize='optimal')
	O['ab'][a,A,c,C] += scale * -1.00000000 * np.einsum('UV,UI,uWiV->uWiI', eta1['b'], h['b'][A,C], t['ab'][pa,pA,hc,hA], optimize='optimal')
	O['ab'][a,A,c,C] += scale * +1.00000000 * np.einsum('UV,WV,uUiI->uWiI', gamma1['b'], h['b'][A,A], t['ab'][pa,pA,hc,hC], optimize='optimal')
	O['ab'][a,A,c,C] += scale * -1.00000000 * np.einsum('UV,UI,uWiV->uWiI', gamma1['b'], h['b'][A,C], t['ab'][pa,pA,hc,hA], optimize='optimal')
	O['ab'][a,A,c,C] += scale * -1.00000000 * np.einsum('IJ,uUiI->uUiJ', h['b'][C,C], t['ab'][pa,pA,hc,hC], optimize='optimal')
	O['ab'][a,A,c,C] += scale * +1.00000000 * np.einsum('UA,uAiI->uUiI', h['b'][A,V], t['ab'][pa,pV,hc,hC], optimize='optimal')
	O['ab'][v,A,c,C] += scale * +1.00000000 * np.einsum('UV,WV,aUiI->aWiI', eta1['b'], h['b'][A,A], t['ab'][pv,pA,hc,hC], optimize='optimal')
	O['ab'][v,A,c,C] += scale * -1.00000000 * np.einsum('UV,UI,aWiV->aWiI', eta1['b'], h['b'][A,C], t['ab'][pv,pA,hc,hA], optimize='optimal')
	O['ab'][v,A,c,C] += scale * +1.00000000 * np.einsum('UV,WV,aUiI->aWiI', gamma1['b'], h['b'][A,A], t['ab'][pv,pA,hc,hC], optimize='optimal')
	O['ab'][v,A,c,C] += scale * -1.00000000 * np.einsum('UV,UI,aWiV->aWiI', gamma1['b'], h['b'][A,C], t['ab'][pv,pA,hc,hA], optimize='optimal')
	O['ab'][v,A,c,C] += scale * -1.00000000 * np.einsum('IJ,aUiI->aUiJ', h['b'][C,C], t['ab'][pv,pA,hc,hC], optimize='optimal')
	O['ab'][v,A,c,C] += scale * +1.00000000 * np.einsum('UA,aAiI->aUiI', h['b'][A,V], t['ab'][pv,pV,hc,hC], optimize='optimal')
	O['ab'][a,C,c,C] += scale * +1.00000000 * np.einsum('UV,IV,uUiJ->uIiJ', eta1['b'], h['b'][C,A], t['ab'][pa,pA,hc,hC], optimize='optimal')
	O['ab'][a,C,c,C] += scale * +1.00000000 * np.einsum('UV,IV,uUiJ->uIiJ', gamma1['b'], h['b'][C,A], t['ab'][pa,pA,hc,hC], optimize='optimal')
	O['ab'][a,C,c,C] += scale * +1.00000000 * np.einsum('IA,uAiJ->uIiJ', h['b'][C,V], t['ab'][pa,pV,hc,hC], optimize='optimal')
	O['ab'][v,C,c,C] += scale * +1.00000000 * np.einsum('UV,IV,aUiJ->aIiJ', eta1['b'], h['b'][C,A], t['ab'][pv,pA,hc,hC], optimize='optimal')
	O['ab'][v,C,c,C] += scale * +1.00000000 * np.einsum('UV,IV,aUiJ->aIiJ', gamma1['b'], h['b'][C,A], t['ab'][pv,pA,hc,hC], optimize='optimal')
	O['ab'][v,C,c,C] += scale * +1.00000000 * np.einsum('IA,aAiJ->aIiJ', h['b'][C,V], t['ab'][pv,pV,hc,hC], optimize='optimal')
	O['ab'][a,V,c,C] += scale * -1.00000000 * np.einsum('UV,UI,uAiV->uAiI', eta1['b'], h['b'][A,C], t['ab'][pa,pV,hc,hA], optimize='optimal')
	O['ab'][a,V,c,C] += scale * +1.00000000 * np.einsum('UV,AV,uUiI->uAiI', eta1['b'], h['b'][V,A], t['ab'][pa,pA,hc,hC], optimize='optimal')
	O['ab'][a,V,c,C] += scale * -1.00000000 * np.einsum('UV,UI,uAiV->uAiI', gamma1['b'], h['b'][A,C], t['ab'][pa,pV,hc,hA], optimize='optimal')
	O['ab'][a,V,c,C] += scale * +1.00000000 * np.einsum('UV,AV,uUiI->uAiI', gamma1['b'], h['b'][V,A], t['ab'][pa,pA,hc,hC], optimize='optimal')
	O['ab'][a,V,c,C] += scale * -1.00000000 * np.einsum('IJ,uAiI->uAiJ', h['b'][C,C], t['ab'][pa,pV,hc,hC], optimize='optimal')
	O['ab'][a,V,c,C] += scale * +1.00000000 * np.einsum('AB,uBiI->uAiI', h['b'][V,V], t['ab'][pa,pV,hc,hC], optimize='optimal')
	O['ab'][v,V,c,C] += scale * -1.00000000 * np.einsum('UV,UI,aAiV->aAiI', eta1['b'], h['b'][A,C], t['ab'][pv,pV,hc,hA], optimize='optimal')
	O['ab'][v,V,c,C] += scale * +1.00000000 * np.einsum('UV,AV,aUiI->aAiI', eta1['b'], h['b'][V,A], t['ab'][pv,pA,hc,hC], optimize='optimal')
	O['ab'][v,V,c,C] += scale * -1.00000000 * np.einsum('UV,UI,aAiV->aAiI', gamma1['b'], h['b'][A,C], t['ab'][pv,pV,hc,hA], optimize='optimal')
	O['ab'][v,V,c,C] += scale * +1.00000000 * np.einsum('UV,AV,aUiI->aAiI', gamma1['b'], h['b'][V,A], t['ab'][pv,pA,hc,hC], optimize='optimal')
	O['ab'][v,V,c,C] += scale * -1.00000000 * np.einsum('IJ,aAiI->aAiJ', h['b'][C,C], t['ab'][pv,pV,hc,hC], optimize='optimal')
	O['ab'][v,V,c,C] += scale * +1.00000000 * np.einsum('AB,aBiI->aAiI', h['b'][V,V], t['ab'][pv,pV,hc,hC], optimize='optimal')
	O['ab'][a,A,c,V] += scale * -1.00000000 * np.einsum('UV,UA,uWiV->uWiA', eta1['b'], h['b'][A,V], t['ab'][pa,pA,hc,hA], optimize='optimal')
	O['ab'][a,A,c,V] += scale * -1.00000000 * np.einsum('UV,UA,uWiV->uWiA', gamma1['b'], h['b'][A,V], t['ab'][pa,pA,hc,hA], optimize='optimal')
	O['ab'][a,A,c,V] += scale * -1.00000000 * np.einsum('IA,uUiI->uUiA', h['b'][C,V], t['ab'][pa,pA,hc,hC], optimize='optimal')
	O['ab'][v,A,c,V] += scale * -1.00000000 * np.einsum('UV,UA,aWiV->aWiA', eta1['b'], h['b'][A,V], t['ab'][pv,pA,hc,hA], optimize='optimal')
	O['ab'][v,A,c,V] += scale * -1.00000000 * np.einsum('UV,UA,aWiV->aWiA', gamma1['b'], h['b'][A,V], t['ab'][pv,pA,hc,hA], optimize='optimal')
	O['ab'][v,A,c,V] += scale * -1.00000000 * np.einsum('IA,aUiI->aUiA', h['b'][C,V], t['ab'][pv,pA,hc,hC], optimize='optimal')
	O['ab'][a,V,c,V] += scale * -1.00000000 * np.einsum('UV,UA,uBiV->uBiA', eta1['b'], h['b'][A,V], t['ab'][pa,pV,hc,hA], optimize='optimal')
	O['ab'][a,V,c,V] += scale * -1.00000000 * np.einsum('UV,UA,uBiV->uBiA', gamma1['b'], h['b'][A,V], t['ab'][pa,pV,hc,hA], optimize='optimal')
	O['ab'][a,V,c,V] += scale * -1.00000000 * np.einsum('IA,uBiI->uBiA', h['b'][C,V], t['ab'][pa,pV,hc,hC], optimize='optimal')
	O['ab'][v,V,c,V] += scale * -1.00000000 * np.einsum('UV,UA,aBiV->aBiA', eta1['b'], h['b'][A,V], t['ab'][pv,pV,hc,hA], optimize='optimal')
	O['ab'][v,V,c,V] += scale * -1.00000000 * np.einsum('UV,UA,aBiV->aBiA', gamma1['b'], h['b'][A,V], t['ab'][pv,pV,hc,hA], optimize='optimal')
	O['ab'][v,V,c,V] += scale * -1.00000000 * np.einsum('IA,aBiI->aBiA', h['b'][C,V], t['ab'][pv,pV,hc,hC], optimize='optimal')

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

	
	
	O['ab'][a,A,a,A] += scale * -1.00000000 * np.einsum('uv,iwxv,uUiV->wUxV', eta1['a'], h['aa'][c,a,a,a], t['ab'][pa,pA,hc,hA], optimize='optimal')
	O['ab'][a,A,a,A] += scale * +1.00000000 * np.einsum('uv,wuxa,aUvV->wUxV', gamma1['a'], h['aa'][a,a,a,v], t['ab'][pv,pA,ha,hA], optimize='optimal')
	O['ab'][a,A,a,A] += scale * -1.00000000 * np.einsum('iuva,aUiV->uUvV', h['aa'][c,a,a,v], t['ab'][pv,pA,hc,hA], optimize='optimal')
	
	
	O['ab'][c,A,a,A] += scale * +1.00000000 * np.einsum('uv,ijwv,uUjV->iUwV', eta1['a'], h['aa'][c,c,a,a], t['ab'][pa,pA,hc,hA], optimize='optimal')
	O['ab'][c,A,a,A] += scale * +1.00000000 * np.einsum('uv,iuwa,aUvV->iUwV', gamma1['a'], h['aa'][c,a,a,v], t['ab'][pv,pA,ha,hA], optimize='optimal')
	O['ab'][c,A,a,A] += scale * +1.00000000 * np.einsum('ijua,aUjV->iUuV', h['aa'][c,c,a,v], t['ab'][pv,pA,hc,hA], optimize='optimal')
	
	
	O['ab'][v,A,a,A] += scale * -1.00000000 * np.einsum('uv,iawv,uUiV->aUwV', eta1['a'], h['aa'][c,v,a,a], t['ab'][pa,pA,hc,hA], optimize='optimal')
	O['ab'][v,A,a,A] += scale * -1.00000000 * np.einsum('uv,uawb,bUvV->aUwV', gamma1['a'], h['aa'][a,v,a,v], t['ab'][pv,pA,ha,hA], optimize='optimal')
	O['ab'][v,A,a,A] += scale * -1.00000000 * np.einsum('iaub,bUiV->aUuV', h['aa'][c,v,a,v], t['ab'][pv,pA,hc,hA], optimize='optimal')
	O['ab'][a,V,a,A] += scale * -1.00000000 * np.einsum('uv,wx,yuzx,wAvU->yAzU', eta1['a'], gamma1['a'], h['aa'][a,a,a,a], t['ab'][pa,pV,ha,hA], optimize='optimal')
	O['ab'][a,V,a,A] += scale * +1.00000000 * np.einsum('uv,wx,ywzv,uAxU->yAzU', eta1['a'], gamma1['a'], h['aa'][a,a,a,a], t['ab'][pa,pV,ha,hA], optimize='optimal')
	O['ab'][a,V,a,A] += scale * -1.00000000 * np.einsum('uv,iwxv,uAiU->wAxU', eta1['a'], h['aa'][c,a,a,a], t['ab'][pa,pV,hc,hA], optimize='optimal')
	O['ab'][a,V,a,A] += scale * +1.00000000 * np.einsum('uv,wuxa,aAvU->wAxU', gamma1['a'], h['aa'][a,a,a,v], t['ab'][pv,pV,ha,hA], optimize='optimal')
	O['ab'][a,V,a,A] += scale * -1.00000000 * np.einsum('iuva,aAiU->uAvU', h['aa'][c,a,a,v], t['ab'][pv,pV,hc,hA], optimize='optimal')
	O['ab'][c,V,a,A] += scale * -1.00000000 * np.einsum('uv,wx,iuyx,wAvU->iAyU', eta1['a'], gamma1['a'], h['aa'][c,a,a,a], t['ab'][pa,pV,ha,hA], optimize='optimal')
	O['ab'][c,V,a,A] += scale * +1.00000000 * np.einsum('uv,wx,iwyv,uAxU->iAyU', eta1['a'], gamma1['a'], h['aa'][c,a,a,a], t['ab'][pa,pV,ha,hA], optimize='optimal')
	O['ab'][c,V,a,A] += scale * +1.00000000 * np.einsum('uv,ijwv,uAjU->iAwU', eta1['a'], h['aa'][c,c,a,a], t['ab'][pa,pV,hc,hA], optimize='optimal')
	O['ab'][c,V,a,A] += scale * +1.00000000 * np.einsum('uv,iuwa,aAvU->iAwU', gamma1['a'], h['aa'][c,a,a,v], t['ab'][pv,pV,ha,hA], optimize='optimal')
	O['ab'][c,V,a,A] += scale * +1.00000000 * np.einsum('ijua,aAjU->iAuU', h['aa'][c,c,a,v], t['ab'][pv,pV,hc,hA], optimize='optimal')
	O['ab'][v,V,a,A] += scale * +1.00000000 * np.einsum('uv,wx,uayx,wAvU->aAyU', eta1['a'], gamma1['a'], h['aa'][a,v,a,a], t['ab'][pa,pV,ha,hA], optimize='optimal')
	O['ab'][v,V,a,A] += scale * -1.00000000 * np.einsum('uv,wx,wayv,uAxU->aAyU', eta1['a'], gamma1['a'], h['aa'][a,v,a,a], t['ab'][pa,pV,ha,hA], optimize='optimal')
	O['ab'][v,V,a,A] += scale * -1.00000000 * np.einsum('uv,iawv,uAiU->aAwU', eta1['a'], h['aa'][c,v,a,a], t['ab'][pa,pV,hc,hA], optimize='optimal')
	O['ab'][v,V,a,A] += scale * -1.00000000 * np.einsum('uv,uawb,bAvU->aAwU', gamma1['a'], h['aa'][a,v,a,v], t['ab'][pv,pV,ha,hA], optimize='optimal')
	O['ab'][v,V,a,A] += scale * -1.00000000 * np.einsum('iaub,bAiU->aAuU', h['aa'][c,v,a,v], t['ab'][pv,pV,hc,hA], optimize='optimal')
	O['ab'][a,A,a,C] += scale * -1.00000000 * np.einsum('uv,wx,yuzx,wUvI->yUzI', eta1['a'], gamma1['a'], h['aa'][a,a,a,a], t['ab'][pa,pA,ha,hC], optimize='optimal')
	O['ab'][a,A,a,C] += scale * +1.00000000 * np.einsum('uv,wx,ywzv,uUxI->yUzI', eta1['a'], gamma1['a'], h['aa'][a,a,a,a], t['ab'][pa,pA,ha,hC], optimize='optimal')
	O['ab'][a,A,a,C] += scale * -1.00000000 * np.einsum('uv,iwxv,uUiI->wUxI', eta1['a'], h['aa'][c,a,a,a], t['ab'][pa,pA,hc,hC], optimize='optimal')
	O['ab'][a,A,a,C] += scale * +1.00000000 * np.einsum('uv,wuxa,aUvI->wUxI', gamma1['a'], h['aa'][a,a,a,v], t['ab'][pv,pA,ha,hC], optimize='optimal')
	O['ab'][a,A,a,C] += scale * -1.00000000 * np.einsum('iuva,aUiI->uUvI', h['aa'][c,a,a,v], t['ab'][pv,pA,hc,hC], optimize='optimal')
	O['ab'][c,A,a,C] += scale * -1.00000000 * np.einsum('uv,wx,iuyx,wUvI->iUyI', eta1['a'], gamma1['a'], h['aa'][c,a,a,a], t['ab'][pa,pA,ha,hC], optimize='optimal')
	O['ab'][c,A,a,C] += scale * +1.00000000 * np.einsum('uv,wx,iwyv,uUxI->iUyI', eta1['a'], gamma1['a'], h['aa'][c,a,a,a], t['ab'][pa,pA,ha,hC], optimize='optimal')
	O['ab'][c,A,a,C] += scale * +1.00000000 * np.einsum('uv,ijwv,uUjI->iUwI', eta1['a'], h['aa'][c,c,a,a], t['ab'][pa,pA,hc,hC], optimize='optimal')
	O['ab'][c,A,a,C] += scale * +1.00000000 * np.einsum('uv,iuwa,aUvI->iUwI', gamma1['a'], h['aa'][c,a,a,v], t['ab'][pv,pA,ha,hC], optimize='optimal')
	O['ab'][c,A,a,C] += scale * +1.00000000 * np.einsum('ijua,aUjI->iUuI', h['aa'][c,c,a,v], t['ab'][pv,pA,hc,hC], optimize='optimal')
	O['ab'][v,A,a,C] += scale * +1.00000000 * np.einsum('uv,wx,uayx,wUvI->aUyI', eta1['a'], gamma1['a'], h['aa'][a,v,a,a], t['ab'][pa,pA,ha,hC], optimize='optimal')
	O['ab'][v,A,a,C] += scale * -1.00000000 * np.einsum('uv,wx,wayv,uUxI->aUyI', eta1['a'], gamma1['a'], h['aa'][a,v,a,a], t['ab'][pa,pA,ha,hC], optimize='optimal')
	O['ab'][v,A,a,C] += scale * -1.00000000 * np.einsum('uv,iawv,uUiI->aUwI', eta1['a'], h['aa'][c,v,a,a], t['ab'][pa,pA,hc,hC], optimize='optimal')
	O['ab'][v,A,a,C] += scale * -1.00000000 * np.einsum('uv,uawb,bUvI->aUwI', gamma1['a'], h['aa'][a,v,a,v], t['ab'][pv,pA,ha,hC], optimize='optimal')
	O['ab'][v,A,a,C] += scale * -1.00000000 * np.einsum('iaub,bUiI->aUuI', h['aa'][c,v,a,v], t['ab'][pv,pA,hc,hC], optimize='optimal')
	O['ab'][a,V,a,C] += scale * -1.00000000 * np.einsum('uv,wx,yuzx,wAvI->yAzI', eta1['a'], gamma1['a'], h['aa'][a,a,a,a], t['ab'][pa,pV,ha,hC], optimize='optimal')
	O['ab'][a,V,a,C] += scale * +1.00000000 * np.einsum('uv,wx,ywzv,uAxI->yAzI', eta1['a'], gamma1['a'], h['aa'][a,a,a,a], t['ab'][pa,pV,ha,hC], optimize='optimal')
	O['ab'][a,V,a,C] += scale * -1.00000000 * np.einsum('uv,iwxv,uAiI->wAxI', eta1['a'], h['aa'][c,a,a,a], t['ab'][pa,pV,hc,hC], optimize='optimal')
	O['ab'][a,V,a,C] += scale * +1.00000000 * np.einsum('uv,wuxa,aAvI->wAxI', gamma1['a'], h['aa'][a,a,a,v], t['ab'][pv,pV,ha,hC], optimize='optimal')
	O['ab'][a,V,a,C] += scale * -1.00000000 * np.einsum('iuva,aAiI->uAvI', h['aa'][c,a,a,v], t['ab'][pv,pV,hc,hC], optimize='optimal')
	O['ab'][c,V,a,C] += scale * -1.00000000 * np.einsum('uv,wx,iuyx,wAvI->iAyI', eta1['a'], gamma1['a'], h['aa'][c,a,a,a], t['ab'][pa,pV,ha,hC], optimize='optimal')
	O['ab'][c,V,a,C] += scale * +1.00000000 * np.einsum('uv,wx,iwyv,uAxI->iAyI', eta1['a'], gamma1['a'], h['aa'][c,a,a,a], t['ab'][pa,pV,ha,hC], optimize='optimal')
	O['ab'][c,V,a,C] += scale * +1.00000000 * np.einsum('uv,ijwv,uAjI->iAwI', eta1['a'], h['aa'][c,c,a,a], t['ab'][pa,pV,hc,hC], optimize='optimal')
	O['ab'][c,V,a,C] += scale * +1.00000000 * np.einsum('uv,iuwa,aAvI->iAwI', gamma1['a'], h['aa'][c,a,a,v], t['ab'][pv,pV,ha,hC], optimize='optimal')
	O['ab'][c,V,a,C] += scale * +1.00000000 * np.einsum('ijua,aAjI->iAuI', h['aa'][c,c,a,v], t['ab'][pv,pV,hc,hC], optimize='optimal')
	O['ab'][v,V,a,C] += scale * +1.00000000 * np.einsum('uv,wx,uayx,wAvI->aAyI', eta1['a'], gamma1['a'], h['aa'][a,v,a,a], t['ab'][pa,pV,ha,hC], optimize='optimal')
	O['ab'][v,V,a,C] += scale * -1.00000000 * np.einsum('uv,wx,wayv,uAxI->aAyI', eta1['a'], gamma1['a'], h['aa'][a,v,a,a], t['ab'][pa,pV,ha,hC], optimize='optimal')
	O['ab'][v,V,a,C] += scale * -1.00000000 * np.einsum('uv,iawv,uAiI->aAwI', eta1['a'], h['aa'][c,v,a,a], t['ab'][pa,pV,hc,hC], optimize='optimal')
	O['ab'][v,V,a,C] += scale * -1.00000000 * np.einsum('uv,uawb,bAvI->aAwI', gamma1['a'], h['aa'][a,v,a,v], t['ab'][pv,pV,ha,hC], optimize='optimal')
	O['ab'][v,V,a,C] += scale * -1.00000000 * np.einsum('iaub,bAiI->aAuI', h['aa'][c,v,a,v], t['ab'][pv,pV,hc,hC], optimize='optimal')
	
	
	O['ab'][a,A,c,A] += scale * -1.00000000 * np.einsum('uv,iwjv,uUiV->wUjV', eta1['a'], h['aa'][c,a,c,a], t['ab'][pa,pA,hc,hA], optimize='optimal')
	O['ab'][a,A,c,A] += scale * +1.00000000 * np.einsum('uv,wuia,aUvV->wUiV', gamma1['a'], h['aa'][a,a,c,v], t['ab'][pv,pA,ha,hA], optimize='optimal')
	O['ab'][a,A,c,A] += scale * -1.00000000 * np.einsum('iuja,aUiV->uUjV', h['aa'][c,a,c,v], t['ab'][pv,pA,hc,hA], optimize='optimal')
	
	
	O['ab'][c,A,c,A] += scale * +1.00000000 * np.einsum('uv,ijkv,uUjV->iUkV', eta1['a'], h['aa'][c,c,c,a], t['ab'][pa,pA,hc,hA], optimize='optimal')
	O['ab'][c,A,c,A] += scale * +1.00000000 * np.einsum('uv,iuja,aUvV->iUjV', gamma1['a'], h['aa'][c,a,c,v], t['ab'][pv,pA,ha,hA], optimize='optimal')
	O['ab'][c,A,c,A] += scale * +1.00000000 * np.einsum('ijka,aUjV->iUkV', h['aa'][c,c,c,v], t['ab'][pv,pA,hc,hA], optimize='optimal')
	
	
	O['ab'][v,A,c,A] += scale * -1.00000000 * np.einsum('uv,iajv,uUiV->aUjV', eta1['a'], h['aa'][c,v,c,a], t['ab'][pa,pA,hc,hA], optimize='optimal')
	O['ab'][v,A,c,A] += scale * -1.00000000 * np.einsum('uv,uaib,bUvV->aUiV', gamma1['a'], h['aa'][a,v,c,v], t['ab'][pv,pA,ha,hA], optimize='optimal')
	O['ab'][v,A,c,A] += scale * -1.00000000 * np.einsum('iajb,bUiV->aUjV', h['aa'][c,v,c,v], t['ab'][pv,pA,hc,hA], optimize='optimal')
	O['ab'][a,V,c,A] += scale * -1.00000000 * np.einsum('uv,wx,yuix,wAvU->yAiU', eta1['a'], gamma1['a'], h['aa'][a,a,c,a], t['ab'][pa,pV,ha,hA], optimize='optimal')
	O['ab'][a,V,c,A] += scale * +1.00000000 * np.einsum('uv,wx,ywiv,uAxU->yAiU', eta1['a'], gamma1['a'], h['aa'][a,a,c,a], t['ab'][pa,pV,ha,hA], optimize='optimal')
	O['ab'][a,V,c,A] += scale * -1.00000000 * np.einsum('uv,iwjv,uAiU->wAjU', eta1['a'], h['aa'][c,a,c,a], t['ab'][pa,pV,hc,hA], optimize='optimal')
	O['ab'][a,V,c,A] += scale * +1.00000000 * np.einsum('uv,wuia,aAvU->wAiU', gamma1['a'], h['aa'][a,a,c,v], t['ab'][pv,pV,ha,hA], optimize='optimal')
	O['ab'][a,V,c,A] += scale * -1.00000000 * np.einsum('iuja,aAiU->uAjU', h['aa'][c,a,c,v], t['ab'][pv,pV,hc,hA], optimize='optimal')
	O['ab'][c,V,c,A] += scale * -1.00000000 * np.einsum('uv,wx,iujx,wAvU->iAjU', eta1['a'], gamma1['a'], h['aa'][c,a,c,a], t['ab'][pa,pV,ha,hA], optimize='optimal')
	O['ab'][c,V,c,A] += scale * +1.00000000 * np.einsum('uv,wx,iwjv,uAxU->iAjU', eta1['a'], gamma1['a'], h['aa'][c,a,c,a], t['ab'][pa,pV,ha,hA], optimize='optimal')
	O['ab'][c,V,c,A] += scale * +1.00000000 * np.einsum('uv,ijkv,uAjU->iAkU', eta1['a'], h['aa'][c,c,c,a], t['ab'][pa,pV,hc,hA], optimize='optimal')
	O['ab'][c,V,c,A] += scale * +1.00000000 * np.einsum('uv,iuja,aAvU->iAjU', gamma1['a'], h['aa'][c,a,c,v], t['ab'][pv,pV,ha,hA], optimize='optimal')
	O['ab'][c,V,c,A] += scale * +1.00000000 * np.einsum('ijka,aAjU->iAkU', h['aa'][c,c,c,v], t['ab'][pv,pV,hc,hA], optimize='optimal')
	O['ab'][v,V,c,A] += scale * +1.00000000 * np.einsum('uv,wx,uaix,wAvU->aAiU', eta1['a'], gamma1['a'], h['aa'][a,v,c,a], t['ab'][pa,pV,ha,hA], optimize='optimal')
	O['ab'][v,V,c,A] += scale * -1.00000000 * np.einsum('uv,wx,waiv,uAxU->aAiU', eta1['a'], gamma1['a'], h['aa'][a,v,c,a], t['ab'][pa,pV,ha,hA], optimize='optimal')
	O['ab'][v,V,c,A] += scale * -1.00000000 * np.einsum('uv,iajv,uAiU->aAjU', eta1['a'], h['aa'][c,v,c,a], t['ab'][pa,pV,hc,hA], optimize='optimal')
	O['ab'][v,V,c,A] += scale * -1.00000000 * np.einsum('uv,uaib,bAvU->aAiU', gamma1['a'], h['aa'][a,v,c,v], t['ab'][pv,pV,ha,hA], optimize='optimal')
	O['ab'][v,V,c,A] += scale * -1.00000000 * np.einsum('iajb,bAiU->aAjU', h['aa'][c,v,c,v], t['ab'][pv,pV,hc,hA], optimize='optimal')
	O['ab'][a,A,c,C] += scale * -1.00000000 * np.einsum('uv,wx,yuix,wUvI->yUiI', eta1['a'], gamma1['a'], h['aa'][a,a,c,a], t['ab'][pa,pA,ha,hC], optimize='optimal')
	O['ab'][a,A,c,C] += scale * +1.00000000 * np.einsum('uv,wx,ywiv,uUxI->yUiI', eta1['a'], gamma1['a'], h['aa'][a,a,c,a], t['ab'][pa,pA,ha,hC], optimize='optimal')
	O['ab'][a,A,c,C] += scale * -1.00000000 * np.einsum('uv,iwjv,uUiI->wUjI', eta1['a'], h['aa'][c,a,c,a], t['ab'][pa,pA,hc,hC], optimize='optimal')
	O['ab'][a,A,c,C] += scale * +1.00000000 * np.einsum('uv,wuia,aUvI->wUiI', gamma1['a'], h['aa'][a,a,c,v], t['ab'][pv,pA,ha,hC], optimize='optimal')
	O['ab'][a,A,c,C] += scale * -1.00000000 * np.einsum('iuja,aUiI->uUjI', h['aa'][c,a,c,v], t['ab'][pv,pA,hc,hC], optimize='optimal')
	O['ab'][c,A,c,C] += scale * -1.00000000 * np.einsum('uv,wx,iujx,wUvI->iUjI', eta1['a'], gamma1['a'], h['aa'][c,a,c,a], t['ab'][pa,pA,ha,hC], optimize='optimal')
	O['ab'][c,A,c,C] += scale * +1.00000000 * np.einsum('uv,wx,iwjv,uUxI->iUjI', eta1['a'], gamma1['a'], h['aa'][c,a,c,a], t['ab'][pa,pA,ha,hC], optimize='optimal')
	O['ab'][c,A,c,C] += scale * +1.00000000 * np.einsum('uv,ijkv,uUjI->iUkI', eta1['a'], h['aa'][c,c,c,a], t['ab'][pa,pA,hc,hC], optimize='optimal')
	O['ab'][c,A,c,C] += scale * +1.00000000 * np.einsum('uv,iuja,aUvI->iUjI', gamma1['a'], h['aa'][c,a,c,v], t['ab'][pv,pA,ha,hC], optimize='optimal')
	O['ab'][c,A,c,C] += scale * +1.00000000 * np.einsum('ijka,aUjI->iUkI', h['aa'][c,c,c,v], t['ab'][pv,pA,hc,hC], optimize='optimal')
	O['ab'][v,A,c,C] += scale * +1.00000000 * np.einsum('uv,wx,uaix,wUvI->aUiI', eta1['a'], gamma1['a'], h['aa'][a,v,c,a], t['ab'][pa,pA,ha,hC], optimize='optimal')
	O['ab'][v,A,c,C] += scale * -1.00000000 * np.einsum('uv,wx,waiv,uUxI->aUiI', eta1['a'], gamma1['a'], h['aa'][a,v,c,a], t['ab'][pa,pA,ha,hC], optimize='optimal')
	O['ab'][v,A,c,C] += scale * -1.00000000 * np.einsum('uv,iajv,uUiI->aUjI', eta1['a'], h['aa'][c,v,c,a], t['ab'][pa,pA,hc,hC], optimize='optimal')
	O['ab'][v,A,c,C] += scale * -1.00000000 * np.einsum('uv,uaib,bUvI->aUiI', gamma1['a'], h['aa'][a,v,c,v], t['ab'][pv,pA,ha,hC], optimize='optimal')
	O['ab'][v,A,c,C] += scale * -1.00000000 * np.einsum('iajb,bUiI->aUjI', h['aa'][c,v,c,v], t['ab'][pv,pA,hc,hC], optimize='optimal')
	O['ab'][a,V,c,C] += scale * -1.00000000 * np.einsum('uv,wx,yuix,wAvI->yAiI', eta1['a'], gamma1['a'], h['aa'][a,a,c,a], t['ab'][pa,pV,ha,hC], optimize='optimal')
	O['ab'][a,V,c,C] += scale * +1.00000000 * np.einsum('uv,wx,ywiv,uAxI->yAiI', eta1['a'], gamma1['a'], h['aa'][a,a,c,a], t['ab'][pa,pV,ha,hC], optimize='optimal')
	O['ab'][a,V,c,C] += scale * -1.00000000 * np.einsum('uv,iwjv,uAiI->wAjI', eta1['a'], h['aa'][c,a,c,a], t['ab'][pa,pV,hc,hC], optimize='optimal')
	O['ab'][a,V,c,C] += scale * +1.00000000 * np.einsum('uv,wuia,aAvI->wAiI', gamma1['a'], h['aa'][a,a,c,v], t['ab'][pv,pV,ha,hC], optimize='optimal')
	O['ab'][a,V,c,C] += scale * -1.00000000 * np.einsum('iuja,aAiI->uAjI', h['aa'][c,a,c,v], t['ab'][pv,pV,hc,hC], optimize='optimal')
	O['ab'][c,V,c,C] += scale * -1.00000000 * np.einsum('uv,wx,iujx,wAvI->iAjI', eta1['a'], gamma1['a'], h['aa'][c,a,c,a], t['ab'][pa,pV,ha,hC], optimize='optimal')
	O['ab'][c,V,c,C] += scale * +1.00000000 * np.einsum('uv,wx,iwjv,uAxI->iAjI', eta1['a'], gamma1['a'], h['aa'][c,a,c,a], t['ab'][pa,pV,ha,hC], optimize='optimal')
	O['ab'][c,V,c,C] += scale * +1.00000000 * np.einsum('uv,ijkv,uAjI->iAkI', eta1['a'], h['aa'][c,c,c,a], t['ab'][pa,pV,hc,hC], optimize='optimal')
	O['ab'][c,V,c,C] += scale * +1.00000000 * np.einsum('uv,iuja,aAvI->iAjI', gamma1['a'], h['aa'][c,a,c,v], t['ab'][pv,pV,ha,hC], optimize='optimal')
	O['ab'][c,V,c,C] += scale * +1.00000000 * np.einsum('ijka,aAjI->iAkI', h['aa'][c,c,c,v], t['ab'][pv,pV,hc,hC], optimize='optimal')
	O['ab'][v,V,c,C] += scale * +1.00000000 * np.einsum('uv,wx,uaix,wAvI->aAiI', eta1['a'], gamma1['a'], h['aa'][a,v,c,a], t['ab'][pa,pV,ha,hC], optimize='optimal')
	O['ab'][v,V,c,C] += scale * -1.00000000 * np.einsum('uv,wx,waiv,uAxI->aAiI', eta1['a'], gamma1['a'], h['aa'][a,v,c,a], t['ab'][pa,pV,ha,hC], optimize='optimal')
	O['ab'][v,V,c,C] += scale * -1.00000000 * np.einsum('uv,iajv,uAiI->aAjI', eta1['a'], h['aa'][c,v,c,a], t['ab'][pa,pV,hc,hC], optimize='optimal')
	O['ab'][v,V,c,C] += scale * -1.00000000 * np.einsum('uv,uaib,bAvI->aAiI', gamma1['a'], h['aa'][a,v,c,v], t['ab'][pv,pV,ha,hC], optimize='optimal')
	O['ab'][v,V,c,C] += scale * -1.00000000 * np.einsum('iajb,bAiI->aAjI', h['aa'][c,v,c,v], t['ab'][pv,pV,hc,hC], optimize='optimal')
	
	
	O['ab'][a,A,v,A] += scale * +1.00000000 * np.einsum('uv,iwva,uUiV->wUaV', eta1['a'], h['aa'][c,a,a,v], t['ab'][pa,pA,hc,hA], optimize='optimal')
	O['ab'][a,A,v,A] += scale * +1.00000000 * np.einsum('uv,wuab,bUvV->wUaV', gamma1['a'], h['aa'][a,a,v,v], t['ab'][pv,pA,ha,hA], optimize='optimal')
	O['ab'][a,A,v,A] += scale * -1.00000000 * np.einsum('iuab,bUiV->uUaV', h['aa'][c,a,v,v], t['ab'][pv,pA,hc,hA], optimize='optimal')
	
	
	O['ab'][c,A,v,A] += scale * -1.00000000 * np.einsum('uv,ijva,uUjV->iUaV', eta1['a'], h['aa'][c,c,a,v], t['ab'][pa,pA,hc,hA], optimize='optimal')
	O['ab'][c,A,v,A] += scale * +1.00000000 * np.einsum('uv,iuab,bUvV->iUaV', gamma1['a'], h['aa'][c,a,v,v], t['ab'][pv,pA,ha,hA], optimize='optimal')
	O['ab'][c,A,v,A] += scale * +1.00000000 * np.einsum('ijab,bUjV->iUaV', h['aa'][c,c,v,v], t['ab'][pv,pA,hc,hA], optimize='optimal')
	
	
	O['ab'][v,A,v,A] += scale * +1.00000000 * np.einsum('uv,iavb,uUiV->aUbV', eta1['a'], h['aa'][c,v,a,v], t['ab'][pa,pA,hc,hA], optimize='optimal')
	O['ab'][v,A,v,A] += scale * -1.00000000 * np.einsum('uv,uabc,cUvV->aUbV', gamma1['a'], h['aa'][a,v,v,v], t['ab'][pv,pA,ha,hA], optimize='optimal')
	O['ab'][v,A,v,A] += scale * -1.00000000 * np.einsum('iabc,cUiV->aUbV', h['aa'][c,v,v,v], t['ab'][pv,pA,hc,hA], optimize='optimal')
	O['ab'][a,V,v,A] += scale * +1.00000000 * np.einsum('uv,wx,yuxa,wAvU->yAaU', eta1['a'], gamma1['a'], h['aa'][a,a,a,v], t['ab'][pa,pV,ha,hA], optimize='optimal')
	O['ab'][a,V,v,A] += scale * -1.00000000 * np.einsum('uv,wx,ywva,uAxU->yAaU', eta1['a'], gamma1['a'], h['aa'][a,a,a,v], t['ab'][pa,pV,ha,hA], optimize='optimal')
	O['ab'][a,V,v,A] += scale * +1.00000000 * np.einsum('uv,iwva,uAiU->wAaU', eta1['a'], h['aa'][c,a,a,v], t['ab'][pa,pV,hc,hA], optimize='optimal')
	O['ab'][a,V,v,A] += scale * +1.00000000 * np.einsum('uv,wuab,bAvU->wAaU', gamma1['a'], h['aa'][a,a,v,v], t['ab'][pv,pV,ha,hA], optimize='optimal')
	O['ab'][a,V,v,A] += scale * -1.00000000 * np.einsum('iuab,bAiU->uAaU', h['aa'][c,a,v,v], t['ab'][pv,pV,hc,hA], optimize='optimal')
	O['ab'][c,V,v,A] += scale * +1.00000000 * np.einsum('uv,wx,iuxa,wAvU->iAaU', eta1['a'], gamma1['a'], h['aa'][c,a,a,v], t['ab'][pa,pV,ha,hA], optimize='optimal')
	O['ab'][c,V,v,A] += scale * -1.00000000 * np.einsum('uv,wx,iwva,uAxU->iAaU', eta1['a'], gamma1['a'], h['aa'][c,a,a,v], t['ab'][pa,pV,ha,hA], optimize='optimal')
	O['ab'][c,V,v,A] += scale * -1.00000000 * np.einsum('uv,ijva,uAjU->iAaU', eta1['a'], h['aa'][c,c,a,v], t['ab'][pa,pV,hc,hA], optimize='optimal')
	O['ab'][c,V,v,A] += scale * +1.00000000 * np.einsum('uv,iuab,bAvU->iAaU', gamma1['a'], h['aa'][c,a,v,v], t['ab'][pv,pV,ha,hA], optimize='optimal')
	O['ab'][c,V,v,A] += scale * +1.00000000 * np.einsum('ijab,bAjU->iAaU', h['aa'][c,c,v,v], t['ab'][pv,pV,hc,hA], optimize='optimal')
	O['ab'][v,V,v,A] += scale * -1.00000000 * np.einsum('uv,wx,uaxb,wAvU->aAbU', eta1['a'], gamma1['a'], h['aa'][a,v,a,v], t['ab'][pa,pV,ha,hA], optimize='optimal')
	O['ab'][v,V,v,A] += scale * +1.00000000 * np.einsum('uv,wx,wavb,uAxU->aAbU', eta1['a'], gamma1['a'], h['aa'][a,v,a,v], t['ab'][pa,pV,ha,hA], optimize='optimal')
	O['ab'][v,V,v,A] += scale * +1.00000000 * np.einsum('uv,iavb,uAiU->aAbU', eta1['a'], h['aa'][c,v,a,v], t['ab'][pa,pV,hc,hA], optimize='optimal')
	O['ab'][v,V,v,A] += scale * -1.00000000 * np.einsum('uv,uabc,cAvU->aAbU', gamma1['a'], h['aa'][a,v,v,v], t['ab'][pv,pV,ha,hA], optimize='optimal')
	O['ab'][v,V,v,A] += scale * -1.00000000 * np.einsum('iabc,cAiU->aAbU', h['aa'][c,v,v,v], t['ab'][pv,pV,hc,hA], optimize='optimal')
	O['ab'][a,A,v,C] += scale * +1.00000000 * np.einsum('uv,wx,yuxa,wUvI->yUaI', eta1['a'], gamma1['a'], h['aa'][a,a,a,v], t['ab'][pa,pA,ha,hC], optimize='optimal')
	O['ab'][a,A,v,C] += scale * -1.00000000 * np.einsum('uv,wx,ywva,uUxI->yUaI', eta1['a'], gamma1['a'], h['aa'][a,a,a,v], t['ab'][pa,pA,ha,hC], optimize='optimal')
	O['ab'][a,A,v,C] += scale * +1.00000000 * np.einsum('uv,iwva,uUiI->wUaI', eta1['a'], h['aa'][c,a,a,v], t['ab'][pa,pA,hc,hC], optimize='optimal')
	O['ab'][a,A,v,C] += scale * +1.00000000 * np.einsum('uv,wuab,bUvI->wUaI', gamma1['a'], h['aa'][a,a,v,v], t['ab'][pv,pA,ha,hC], optimize='optimal')
	O['ab'][a,A,v,C] += scale * -1.00000000 * np.einsum('iuab,bUiI->uUaI', h['aa'][c,a,v,v], t['ab'][pv,pA,hc,hC], optimize='optimal')
	O['ab'][c,A,v,C] += scale * +1.00000000 * np.einsum('uv,wx,iuxa,wUvI->iUaI', eta1['a'], gamma1['a'], h['aa'][c,a,a,v], t['ab'][pa,pA,ha,hC], optimize='optimal')
	O['ab'][c,A,v,C] += scale * -1.00000000 * np.einsum('uv,wx,iwva,uUxI->iUaI', eta1['a'], gamma1['a'], h['aa'][c,a,a,v], t['ab'][pa,pA,ha,hC], optimize='optimal')
	O['ab'][c,A,v,C] += scale * -1.00000000 * np.einsum('uv,ijva,uUjI->iUaI', eta1['a'], h['aa'][c,c,a,v], t['ab'][pa,pA,hc,hC], optimize='optimal')
	O['ab'][c,A,v,C] += scale * +1.00000000 * np.einsum('uv,iuab,bUvI->iUaI', gamma1['a'], h['aa'][c,a,v,v], t['ab'][pv,pA,ha,hC], optimize='optimal')
	O['ab'][c,A,v,C] += scale * +1.00000000 * np.einsum('ijab,bUjI->iUaI', h['aa'][c,c,v,v], t['ab'][pv,pA,hc,hC], optimize='optimal')
	O['ab'][v,A,v,C] += scale * -1.00000000 * np.einsum('uv,wx,uaxb,wUvI->aUbI', eta1['a'], gamma1['a'], h['aa'][a,v,a,v], t['ab'][pa,pA,ha,hC], optimize='optimal')
	O['ab'][v,A,v,C] += scale * +1.00000000 * np.einsum('uv,wx,wavb,uUxI->aUbI', eta1['a'], gamma1['a'], h['aa'][a,v,a,v], t['ab'][pa,pA,ha,hC], optimize='optimal')
	O['ab'][v,A,v,C] += scale * +1.00000000 * np.einsum('uv,iavb,uUiI->aUbI', eta1['a'], h['aa'][c,v,a,v], t['ab'][pa,pA,hc,hC], optimize='optimal')
	O['ab'][v,A,v,C] += scale * -1.00000000 * np.einsum('uv,uabc,cUvI->aUbI', gamma1['a'], h['aa'][a,v,v,v], t['ab'][pv,pA,ha,hC], optimize='optimal')
	O['ab'][v,A,v,C] += scale * -1.00000000 * np.einsum('iabc,cUiI->aUbI', h['aa'][c,v,v,v], t['ab'][pv,pA,hc,hC], optimize='optimal')
	O['ab'][a,V,v,C] += scale * +1.00000000 * np.einsum('uv,wx,yuxa,wAvI->yAaI', eta1['a'], gamma1['a'], h['aa'][a,a,a,v], t['ab'][pa,pV,ha,hC], optimize='optimal')
	O['ab'][a,V,v,C] += scale * -1.00000000 * np.einsum('uv,wx,ywva,uAxI->yAaI', eta1['a'], gamma1['a'], h['aa'][a,a,a,v], t['ab'][pa,pV,ha,hC], optimize='optimal')
	O['ab'][a,V,v,C] += scale * +1.00000000 * np.einsum('uv,iwva,uAiI->wAaI', eta1['a'], h['aa'][c,a,a,v], t['ab'][pa,pV,hc,hC], optimize='optimal')
	O['ab'][a,V,v,C] += scale * +1.00000000 * np.einsum('uv,wuab,bAvI->wAaI', gamma1['a'], h['aa'][a,a,v,v], t['ab'][pv,pV,ha,hC], optimize='optimal')
	O['ab'][a,V,v,C] += scale * -1.00000000 * np.einsum('iuab,bAiI->uAaI', h['aa'][c,a,v,v], t['ab'][pv,pV,hc,hC], optimize='optimal')
	O['ab'][c,V,v,C] += scale * +1.00000000 * np.einsum('uv,wx,iuxa,wAvI->iAaI', eta1['a'], gamma1['a'], h['aa'][c,a,a,v], t['ab'][pa,pV,ha,hC], optimize='optimal')
	O['ab'][c,V,v,C] += scale * -1.00000000 * np.einsum('uv,wx,iwva,uAxI->iAaI', eta1['a'], gamma1['a'], h['aa'][c,a,a,v], t['ab'][pa,pV,ha,hC], optimize='optimal')
	O['ab'][c,V,v,C] += scale * -1.00000000 * np.einsum('uv,ijva,uAjI->iAaI', eta1['a'], h['aa'][c,c,a,v], t['ab'][pa,pV,hc,hC], optimize='optimal')
	O['ab'][c,V,v,C] += scale * +1.00000000 * np.einsum('uv,iuab,bAvI->iAaI', gamma1['a'], h['aa'][c,a,v,v], t['ab'][pv,pV,ha,hC], optimize='optimal')
	O['ab'][c,V,v,C] += scale * +1.00000000 * np.einsum('ijab,bAjI->iAaI', h['aa'][c,c,v,v], t['ab'][pv,pV,hc,hC], optimize='optimal')
	O['ab'][v,V,v,C] += scale * -1.00000000 * np.einsum('uv,wx,uaxb,wAvI->aAbI', eta1['a'], gamma1['a'], h['aa'][a,v,a,v], t['ab'][pa,pV,ha,hC], optimize='optimal')
	O['ab'][v,V,v,C] += scale * +1.00000000 * np.einsum('uv,wx,wavb,uAxI->aAbI', eta1['a'], gamma1['a'], h['aa'][a,v,a,v], t['ab'][pa,pV,ha,hC], optimize='optimal')
	O['ab'][v,V,v,C] += scale * +1.00000000 * np.einsum('uv,iavb,uAiI->aAbI', eta1['a'], h['aa'][c,v,a,v], t['ab'][pa,pV,hc,hC], optimize='optimal')
	O['ab'][v,V,v,C] += scale * -1.00000000 * np.einsum('uv,uabc,cAvI->aAbI', gamma1['a'], h['aa'][a,v,v,v], t['ab'][pv,pV,ha,hC], optimize='optimal')
	O['ab'][v,V,v,C] += scale * -1.00000000 * np.einsum('iabc,cAiI->aAbI', h['aa'][c,v,v,v], t['ab'][pv,pV,hc,hC], optimize='optimal')

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

	
	
	
	
	O['ab'][a,A,a,A] += scale * -1.00000000 * np.einsum('iUuV,vi->vUuV', h['ab'][c,A,a,A], t['a'][pa,hc], optimize='optimal')
	O['ab'][a,A,a,A] += scale * +1.00000000 * np.einsum('uUaV,av->uUvV', h['ab'][a,A,v,A], t['a'][pv,ha], optimize='optimal')
	
	
	O['ab'][c,A,a,A] += scale * +1.00000000 * np.einsum('iUaV,au->iUuV', h['ab'][c,A,v,A], t['a'][pv,ha], optimize='optimal')
	O['ab'][v,A,a,A] += scale * -1.00000000 * np.einsum('uv,uUwV,av->aUwV', eta1['a'], h['ab'][a,A,a,A], t['a'][pv,ha], optimize='optimal')
	
	O['ab'][v,A,a,A] += scale * -1.00000000 * np.einsum('uv,uUwV,av->aUwV', gamma1['a'], h['ab'][a,A,a,A], t['a'][pv,ha], optimize='optimal')
	
	O['ab'][v,A,a,A] += scale * -1.00000000 * np.einsum('iUuV,ai->aUuV', h['ab'][c,A,a,A], t['a'][pv,hc], optimize='optimal')
	O['ab'][v,A,a,A] += scale * +1.00000000 * np.einsum('aUbV,bu->aUuV', h['ab'][v,A,v,A], t['a'][pv,ha], optimize='optimal')
	
	
	
	
	O['ab'][a,C,a,A] += scale * -1.00000000 * np.einsum('iIuU,vi->vIuU', h['ab'][c,C,a,A], t['a'][pa,hc], optimize='optimal')
	O['ab'][a,C,a,A] += scale * +1.00000000 * np.einsum('uIaU,av->uIvU', h['ab'][a,C,v,A], t['a'][pv,ha], optimize='optimal')
	
	
	O['ab'][c,C,a,A] += scale * +1.00000000 * np.einsum('iIaU,au->iIuU', h['ab'][c,C,v,A], t['a'][pv,ha], optimize='optimal')
	O['ab'][v,C,a,A] += scale * -1.00000000 * np.einsum('uv,uIwU,av->aIwU', eta1['a'], h['ab'][a,C,a,A], t['a'][pv,ha], optimize='optimal')
	
	O['ab'][v,C,a,A] += scale * -1.00000000 * np.einsum('uv,uIwU,av->aIwU', gamma1['a'], h['ab'][a,C,a,A], t['a'][pv,ha], optimize='optimal')
	
	O['ab'][v,C,a,A] += scale * -1.00000000 * np.einsum('iIuU,ai->aIuU', h['ab'][c,C,a,A], t['a'][pv,hc], optimize='optimal')
	O['ab'][v,C,a,A] += scale * +1.00000000 * np.einsum('aIbU,bu->aIuU', h['ab'][v,C,v,A], t['a'][pv,ha], optimize='optimal')
	
	
	
	
	O['ab'][a,V,a,A] += scale * -1.00000000 * np.einsum('iAuU,vi->vAuU', h['ab'][c,V,a,A], t['a'][pa,hc], optimize='optimal')
	O['ab'][a,V,a,A] += scale * +1.00000000 * np.einsum('uAaU,av->uAvU', h['ab'][a,V,v,A], t['a'][pv,ha], optimize='optimal')
	
	
	O['ab'][c,V,a,A] += scale * +1.00000000 * np.einsum('iAaU,au->iAuU', h['ab'][c,V,v,A], t['a'][pv,ha], optimize='optimal')
	O['ab'][v,V,a,A] += scale * -1.00000000 * np.einsum('uv,uAwU,av->aAwU', eta1['a'], h['ab'][a,V,a,A], t['a'][pv,ha], optimize='optimal')
	
	O['ab'][v,V,a,A] += scale * -1.00000000 * np.einsum('uv,uAwU,av->aAwU', gamma1['a'], h['ab'][a,V,a,A], t['a'][pv,ha], optimize='optimal')
	
	O['ab'][v,V,a,A] += scale * -1.00000000 * np.einsum('iAuU,ai->aAuU', h['ab'][c,V,a,A], t['a'][pv,hc], optimize='optimal')
	O['ab'][v,V,a,A] += scale * +1.00000000 * np.einsum('aAbU,bu->aAuU', h['ab'][v,V,v,A], t['a'][pv,ha], optimize='optimal')
	
	
	
	
	O['ab'][a,A,a,C] += scale * -1.00000000 * np.einsum('iUuI,vi->vUuI', h['ab'][c,A,a,C], t['a'][pa,hc], optimize='optimal')
	O['ab'][a,A,a,C] += scale * +1.00000000 * np.einsum('uUaI,av->uUvI', h['ab'][a,A,v,C], t['a'][pv,ha], optimize='optimal')
	
	
	O['ab'][c,A,a,C] += scale * +1.00000000 * np.einsum('iUaI,au->iUuI', h['ab'][c,A,v,C], t['a'][pv,ha], optimize='optimal')
	O['ab'][v,A,a,C] += scale * -1.00000000 * np.einsum('uv,uUwI,av->aUwI', eta1['a'], h['ab'][a,A,a,C], t['a'][pv,ha], optimize='optimal')
	
	O['ab'][v,A,a,C] += scale * -1.00000000 * np.einsum('uv,uUwI,av->aUwI', gamma1['a'], h['ab'][a,A,a,C], t['a'][pv,ha], optimize='optimal')
	
	O['ab'][v,A,a,C] += scale * -1.00000000 * np.einsum('iUuI,ai->aUuI', h['ab'][c,A,a,C], t['a'][pv,hc], optimize='optimal')
	O['ab'][v,A,a,C] += scale * +1.00000000 * np.einsum('aUbI,bu->aUuI', h['ab'][v,A,v,C], t['a'][pv,ha], optimize='optimal')
	
	
	
	
	O['ab'][a,C,a,C] += scale * -1.00000000 * np.einsum('iIuJ,vi->vIuJ', h['ab'][c,C,a,C], t['a'][pa,hc], optimize='optimal')
	O['ab'][a,C,a,C] += scale * +1.00000000 * np.einsum('uIaJ,av->uIvJ', h['ab'][a,C,v,C], t['a'][pv,ha], optimize='optimal')
	
	
	O['ab'][c,C,a,C] += scale * +1.00000000 * np.einsum('iIaJ,au->iIuJ', h['ab'][c,C,v,C], t['a'][pv,ha], optimize='optimal')
	O['ab'][v,C,a,C] += scale * -1.00000000 * np.einsum('uv,uIwJ,av->aIwJ', eta1['a'], h['ab'][a,C,a,C], t['a'][pv,ha], optimize='optimal')
	
	O['ab'][v,C,a,C] += scale * -1.00000000 * np.einsum('uv,uIwJ,av->aIwJ', gamma1['a'], h['ab'][a,C,a,C], t['a'][pv,ha], optimize='optimal')
	
	O['ab'][v,C,a,C] += scale * -1.00000000 * np.einsum('iIuJ,ai->aIuJ', h['ab'][c,C,a,C], t['a'][pv,hc], optimize='optimal')
	O['ab'][v,C,a,C] += scale * +1.00000000 * np.einsum('aIbJ,bu->aIuJ', h['ab'][v,C,v,C], t['a'][pv,ha], optimize='optimal')
	
	
	
	
	O['ab'][a,V,a,C] += scale * -1.00000000 * np.einsum('iAuI,vi->vAuI', h['ab'][c,V,a,C], t['a'][pa,hc], optimize='optimal')
	O['ab'][a,V,a,C] += scale * +1.00000000 * np.einsum('uAaI,av->uAvI', h['ab'][a,V,v,C], t['a'][pv,ha], optimize='optimal')
	
	
	O['ab'][c,V,a,C] += scale * +1.00000000 * np.einsum('iAaI,au->iAuI', h['ab'][c,V,v,C], t['a'][pv,ha], optimize='optimal')
	O['ab'][v,V,a,C] += scale * -1.00000000 * np.einsum('uv,uAwI,av->aAwI', eta1['a'], h['ab'][a,V,a,C], t['a'][pv,ha], optimize='optimal')
	
	O['ab'][v,V,a,C] += scale * -1.00000000 * np.einsum('uv,uAwI,av->aAwI', gamma1['a'], h['ab'][a,V,a,C], t['a'][pv,ha], optimize='optimal')
	
	O['ab'][v,V,a,C] += scale * -1.00000000 * np.einsum('iAuI,ai->aAuI', h['ab'][c,V,a,C], t['a'][pv,hc], optimize='optimal')
	O['ab'][v,V,a,C] += scale * +1.00000000 * np.einsum('aAbI,bu->aAuI', h['ab'][v,V,v,C], t['a'][pv,ha], optimize='optimal')
	
	
	
	
	O['ab'][a,A,a,V] += scale * -1.00000000 * np.einsum('iUuA,vi->vUuA', h['ab'][c,A,a,V], t['a'][pa,hc], optimize='optimal')
	O['ab'][a,A,a,V] += scale * +1.00000000 * np.einsum('uUaA,av->uUvA', h['ab'][a,A,v,V], t['a'][pv,ha], optimize='optimal')
	
	
	O['ab'][c,A,a,V] += scale * +1.00000000 * np.einsum('iUaA,au->iUuA', h['ab'][c,A,v,V], t['a'][pv,ha], optimize='optimal')
	O['ab'][v,A,a,V] += scale * -1.00000000 * np.einsum('uv,uUwA,av->aUwA', eta1['a'], h['ab'][a,A,a,V], t['a'][pv,ha], optimize='optimal')
	
	O['ab'][v,A,a,V] += scale * -1.00000000 * np.einsum('uv,uUwA,av->aUwA', gamma1['a'], h['ab'][a,A,a,V], t['a'][pv,ha], optimize='optimal')
	
	O['ab'][v,A,a,V] += scale * -1.00000000 * np.einsum('iUuA,ai->aUuA', h['ab'][c,A,a,V], t['a'][pv,hc], optimize='optimal')
	O['ab'][v,A,a,V] += scale * +1.00000000 * np.einsum('aUbA,bu->aUuA', h['ab'][v,A,v,V], t['a'][pv,ha], optimize='optimal')
	
	
	
	
	O['ab'][a,C,a,V] += scale * -1.00000000 * np.einsum('iIuA,vi->vIuA', h['ab'][c,C,a,V], t['a'][pa,hc], optimize='optimal')
	O['ab'][a,C,a,V] += scale * +1.00000000 * np.einsum('uIaA,av->uIvA', h['ab'][a,C,v,V], t['a'][pv,ha], optimize='optimal')
	
	
	O['ab'][c,C,a,V] += scale * +1.00000000 * np.einsum('iIaA,au->iIuA', h['ab'][c,C,v,V], t['a'][pv,ha], optimize='optimal')
	O['ab'][v,C,a,V] += scale * -1.00000000 * np.einsum('uv,uIwA,av->aIwA', eta1['a'], h['ab'][a,C,a,V], t['a'][pv,ha], optimize='optimal')
	
	O['ab'][v,C,a,V] += scale * -1.00000000 * np.einsum('uv,uIwA,av->aIwA', gamma1['a'], h['ab'][a,C,a,V], t['a'][pv,ha], optimize='optimal')
	
	O['ab'][v,C,a,V] += scale * -1.00000000 * np.einsum('iIuA,ai->aIuA', h['ab'][c,C,a,V], t['a'][pv,hc], optimize='optimal')
	O['ab'][v,C,a,V] += scale * +1.00000000 * np.einsum('aIbA,bu->aIuA', h['ab'][v,C,v,V], t['a'][pv,ha], optimize='optimal')
	
	
	
	
	O['ab'][a,V,a,V] += scale * -1.00000000 * np.einsum('iAuB,vi->vAuB', h['ab'][c,V,a,V], t['a'][pa,hc], optimize='optimal')
	O['ab'][a,V,a,V] += scale * +1.00000000 * np.einsum('uAaB,av->uAvB', h['ab'][a,V,v,V], t['a'][pv,ha], optimize='optimal')
	
	
	O['ab'][c,V,a,V] += scale * +1.00000000 * np.einsum('iAaB,au->iAuB', h['ab'][c,V,v,V], t['a'][pv,ha], optimize='optimal')
	O['ab'][v,V,a,V] += scale * -1.00000000 * np.einsum('uv,uAwB,av->aAwB', eta1['a'], h['ab'][a,V,a,V], t['a'][pv,ha], optimize='optimal')
	
	O['ab'][v,V,a,V] += scale * -1.00000000 * np.einsum('uv,uAwB,av->aAwB', gamma1['a'], h['ab'][a,V,a,V], t['a'][pv,ha], optimize='optimal')
	
	O['ab'][v,V,a,V] += scale * -1.00000000 * np.einsum('iAuB,ai->aAuB', h['ab'][c,V,a,V], t['a'][pv,hc], optimize='optimal')
	O['ab'][v,V,a,V] += scale * +1.00000000 * np.einsum('aAbB,bu->aAuB', h['ab'][v,V,v,V], t['a'][pv,ha], optimize='optimal')
	O['ab'][a,A,c,A] += scale * +1.00000000 * np.einsum('uv,wUvV,ui->wUiV', eta1['a'], h['ab'][a,A,a,A], t['a'][pa,hc], optimize='optimal')
	
	O['ab'][a,A,c,A] += scale * +1.00000000 * np.einsum('uv,wUvV,ui->wUiV', gamma1['a'], h['ab'][a,A,a,A], t['a'][pa,hc], optimize='optimal')
	
	O['ab'][a,A,c,A] += scale * -1.00000000 * np.einsum('iUjV,ui->uUjV', h['ab'][c,A,c,A], t['a'][pa,hc], optimize='optimal')
	O['ab'][a,A,c,A] += scale * +1.00000000 * np.einsum('uUaV,ai->uUiV', h['ab'][a,A,v,A], t['a'][pv,hc], optimize='optimal')
	O['ab'][c,A,c,A] += scale * +1.00000000 * np.einsum('uv,iUvV,uj->iUjV', eta1['a'], h['ab'][c,A,a,A], t['a'][pa,hc], optimize='optimal')
	O['ab'][c,A,c,A] += scale * +1.00000000 * np.einsum('uv,iUvV,uj->iUjV', gamma1['a'], h['ab'][c,A,a,A], t['a'][pa,hc], optimize='optimal')
	O['ab'][c,A,c,A] += scale * +1.00000000 * np.einsum('iUaV,aj->iUjV', h['ab'][c,A,v,A], t['a'][pv,hc], optimize='optimal')
	O['ab'][v,A,c,A] += scale * -1.00000000 * np.einsum('uv,uUiV,av->aUiV', eta1['a'], h['ab'][a,A,c,A], t['a'][pv,ha], optimize='optimal')
	O['ab'][v,A,c,A] += scale * +1.00000000 * np.einsum('uv,aUvV,ui->aUiV', eta1['a'], h['ab'][v,A,a,A], t['a'][pa,hc], optimize='optimal')
	O['ab'][v,A,c,A] += scale * -1.00000000 * np.einsum('uv,uUiV,av->aUiV', gamma1['a'], h['ab'][a,A,c,A], t['a'][pv,ha], optimize='optimal')
	O['ab'][v,A,c,A] += scale * +1.00000000 * np.einsum('uv,aUvV,ui->aUiV', gamma1['a'], h['ab'][v,A,a,A], t['a'][pa,hc], optimize='optimal')
	O['ab'][v,A,c,A] += scale * -1.00000000 * np.einsum('iUjV,ai->aUjV', h['ab'][c,A,c,A], t['a'][pv,hc], optimize='optimal')
	O['ab'][v,A,c,A] += scale * +1.00000000 * np.einsum('aUbV,bi->aUiV', h['ab'][v,A,v,A], t['a'][pv,hc], optimize='optimal')
	O['ab'][a,C,c,A] += scale * +1.00000000 * np.einsum('uv,wIvU,ui->wIiU', eta1['a'], h['ab'][a,C,a,A], t['a'][pa,hc], optimize='optimal')
	
	O['ab'][a,C,c,A] += scale * +1.00000000 * np.einsum('uv,wIvU,ui->wIiU', gamma1['a'], h['ab'][a,C,a,A], t['a'][pa,hc], optimize='optimal')
	
	O['ab'][a,C,c,A] += scale * -1.00000000 * np.einsum('iIjU,ui->uIjU', h['ab'][c,C,c,A], t['a'][pa,hc], optimize='optimal')
	O['ab'][a,C,c,A] += scale * +1.00000000 * np.einsum('uIaU,ai->uIiU', h['ab'][a,C,v,A], t['a'][pv,hc], optimize='optimal')
	O['ab'][c,C,c,A] += scale * +1.00000000 * np.einsum('uv,iIvU,uj->iIjU', eta1['a'], h['ab'][c,C,a,A], t['a'][pa,hc], optimize='optimal')
	O['ab'][c,C,c,A] += scale * +1.00000000 * np.einsum('uv,iIvU,uj->iIjU', gamma1['a'], h['ab'][c,C,a,A], t['a'][pa,hc], optimize='optimal')
	O['ab'][c,C,c,A] += scale * +1.00000000 * np.einsum('iIaU,aj->iIjU', h['ab'][c,C,v,A], t['a'][pv,hc], optimize='optimal')
	O['ab'][v,C,c,A] += scale * -1.00000000 * np.einsum('uv,uIiU,av->aIiU', eta1['a'], h['ab'][a,C,c,A], t['a'][pv,ha], optimize='optimal')
	O['ab'][v,C,c,A] += scale * +1.00000000 * np.einsum('uv,aIvU,ui->aIiU', eta1['a'], h['ab'][v,C,a,A], t['a'][pa,hc], optimize='optimal')
	O['ab'][v,C,c,A] += scale * -1.00000000 * np.einsum('uv,uIiU,av->aIiU', gamma1['a'], h['ab'][a,C,c,A], t['a'][pv,ha], optimize='optimal')
	O['ab'][v,C,c,A] += scale * +1.00000000 * np.einsum('uv,aIvU,ui->aIiU', gamma1['a'], h['ab'][v,C,a,A], t['a'][pa,hc], optimize='optimal')
	O['ab'][v,C,c,A] += scale * -1.00000000 * np.einsum('iIjU,ai->aIjU', h['ab'][c,C,c,A], t['a'][pv,hc], optimize='optimal')
	O['ab'][v,C,c,A] += scale * +1.00000000 * np.einsum('aIbU,bi->aIiU', h['ab'][v,C,v,A], t['a'][pv,hc], optimize='optimal')
	O['ab'][a,V,c,A] += scale * +1.00000000 * np.einsum('uv,wAvU,ui->wAiU', eta1['a'], h['ab'][a,V,a,A], t['a'][pa,hc], optimize='optimal')
	
	O['ab'][a,V,c,A] += scale * +1.00000000 * np.einsum('uv,wAvU,ui->wAiU', gamma1['a'], h['ab'][a,V,a,A], t['a'][pa,hc], optimize='optimal')
	
	O['ab'][a,V,c,A] += scale * -1.00000000 * np.einsum('iAjU,ui->uAjU', h['ab'][c,V,c,A], t['a'][pa,hc], optimize='optimal')
	O['ab'][a,V,c,A] += scale * +1.00000000 * np.einsum('uAaU,ai->uAiU', h['ab'][a,V,v,A], t['a'][pv,hc], optimize='optimal')
	O['ab'][c,V,c,A] += scale * +1.00000000 * np.einsum('uv,iAvU,uj->iAjU', eta1['a'], h['ab'][c,V,a,A], t['a'][pa,hc], optimize='optimal')
	O['ab'][c,V,c,A] += scale * +1.00000000 * np.einsum('uv,iAvU,uj->iAjU', gamma1['a'], h['ab'][c,V,a,A], t['a'][pa,hc], optimize='optimal')
	O['ab'][c,V,c,A] += scale * +1.00000000 * np.einsum('iAaU,aj->iAjU', h['ab'][c,V,v,A], t['a'][pv,hc], optimize='optimal')
	O['ab'][v,V,c,A] += scale * -1.00000000 * np.einsum('uv,uAiU,av->aAiU', eta1['a'], h['ab'][a,V,c,A], t['a'][pv,ha], optimize='optimal')
	O['ab'][v,V,c,A] += scale * +1.00000000 * np.einsum('uv,aAvU,ui->aAiU', eta1['a'], h['ab'][v,V,a,A], t['a'][pa,hc], optimize='optimal')
	O['ab'][v,V,c,A] += scale * -1.00000000 * np.einsum('uv,uAiU,av->aAiU', gamma1['a'], h['ab'][a,V,c,A], t['a'][pv,ha], optimize='optimal')
	O['ab'][v,V,c,A] += scale * +1.00000000 * np.einsum('uv,aAvU,ui->aAiU', gamma1['a'], h['ab'][v,V,a,A], t['a'][pa,hc], optimize='optimal')
	O['ab'][v,V,c,A] += scale * -1.00000000 * np.einsum('iAjU,ai->aAjU', h['ab'][c,V,c,A], t['a'][pv,hc], optimize='optimal')
	O['ab'][v,V,c,A] += scale * +1.00000000 * np.einsum('aAbU,bi->aAiU', h['ab'][v,V,v,A], t['a'][pv,hc], optimize='optimal')
	O['ab'][a,A,c,C] += scale * +1.00000000 * np.einsum('uv,wUvI,ui->wUiI', eta1['a'], h['ab'][a,A,a,C], t['a'][pa,hc], optimize='optimal')
	
	O['ab'][a,A,c,C] += scale * +1.00000000 * np.einsum('uv,wUvI,ui->wUiI', gamma1['a'], h['ab'][a,A,a,C], t['a'][pa,hc], optimize='optimal')
	
	O['ab'][a,A,c,C] += scale * -1.00000000 * np.einsum('iUjI,ui->uUjI', h['ab'][c,A,c,C], t['a'][pa,hc], optimize='optimal')
	O['ab'][a,A,c,C] += scale * +1.00000000 * np.einsum('uUaI,ai->uUiI', h['ab'][a,A,v,C], t['a'][pv,hc], optimize='optimal')
	O['ab'][c,A,c,C] += scale * +1.00000000 * np.einsum('uv,iUvI,uj->iUjI', eta1['a'], h['ab'][c,A,a,C], t['a'][pa,hc], optimize='optimal')
	O['ab'][c,A,c,C] += scale * +1.00000000 * np.einsum('uv,iUvI,uj->iUjI', gamma1['a'], h['ab'][c,A,a,C], t['a'][pa,hc], optimize='optimal')
	O['ab'][c,A,c,C] += scale * +1.00000000 * np.einsum('iUaI,aj->iUjI', h['ab'][c,A,v,C], t['a'][pv,hc], optimize='optimal')
	O['ab'][v,A,c,C] += scale * -1.00000000 * np.einsum('uv,uUiI,av->aUiI', eta1['a'], h['ab'][a,A,c,C], t['a'][pv,ha], optimize='optimal')
	O['ab'][v,A,c,C] += scale * +1.00000000 * np.einsum('uv,aUvI,ui->aUiI', eta1['a'], h['ab'][v,A,a,C], t['a'][pa,hc], optimize='optimal')
	O['ab'][v,A,c,C] += scale * -1.00000000 * np.einsum('uv,uUiI,av->aUiI', gamma1['a'], h['ab'][a,A,c,C], t['a'][pv,ha], optimize='optimal')
	O['ab'][v,A,c,C] += scale * +1.00000000 * np.einsum('uv,aUvI,ui->aUiI', gamma1['a'], h['ab'][v,A,a,C], t['a'][pa,hc], optimize='optimal')
	O['ab'][v,A,c,C] += scale * -1.00000000 * np.einsum('iUjI,ai->aUjI', h['ab'][c,A,c,C], t['a'][pv,hc], optimize='optimal')
	O['ab'][v,A,c,C] += scale * +1.00000000 * np.einsum('aUbI,bi->aUiI', h['ab'][v,A,v,C], t['a'][pv,hc], optimize='optimal')
	O['ab'][a,C,c,C] += scale * +1.00000000 * np.einsum('uv,wIvJ,ui->wIiJ', eta1['a'], h['ab'][a,C,a,C], t['a'][pa,hc], optimize='optimal')
	
	O['ab'][a,C,c,C] += scale * +1.00000000 * np.einsum('uv,wIvJ,ui->wIiJ', gamma1['a'], h['ab'][a,C,a,C], t['a'][pa,hc], optimize='optimal')
	
	O['ab'][a,C,c,C] += scale * -1.00000000 * np.einsum('iIjJ,ui->uIjJ', h['ab'][c,C,c,C], t['a'][pa,hc], optimize='optimal')
	O['ab'][a,C,c,C] += scale * +1.00000000 * np.einsum('uIaJ,ai->uIiJ', h['ab'][a,C,v,C], t['a'][pv,hc], optimize='optimal')
	O['ab'][c,C,c,C] += scale * +1.00000000 * np.einsum('uv,iIvJ,uj->iIjJ', eta1['a'], h['ab'][c,C,a,C], t['a'][pa,hc], optimize='optimal')
	O['ab'][c,C,c,C] += scale * +1.00000000 * np.einsum('uv,iIvJ,uj->iIjJ', gamma1['a'], h['ab'][c,C,a,C], t['a'][pa,hc], optimize='optimal')
	O['ab'][c,C,c,C] += scale * +1.00000000 * np.einsum('iIaJ,aj->iIjJ', h['ab'][c,C,v,C], t['a'][pv,hc], optimize='optimal')
	O['ab'][v,C,c,C] += scale * -1.00000000 * np.einsum('uv,uIiJ,av->aIiJ', eta1['a'], h['ab'][a,C,c,C], t['a'][pv,ha], optimize='optimal')
	O['ab'][v,C,c,C] += scale * +1.00000000 * np.einsum('uv,aIvJ,ui->aIiJ', eta1['a'], h['ab'][v,C,a,C], t['a'][pa,hc], optimize='optimal')
	O['ab'][v,C,c,C] += scale * -1.00000000 * np.einsum('uv,uIiJ,av->aIiJ', gamma1['a'], h['ab'][a,C,c,C], t['a'][pv,ha], optimize='optimal')
	O['ab'][v,C,c,C] += scale * +1.00000000 * np.einsum('uv,aIvJ,ui->aIiJ', gamma1['a'], h['ab'][v,C,a,C], t['a'][pa,hc], optimize='optimal')
	O['ab'][v,C,c,C] += scale * -1.00000000 * np.einsum('iIjJ,ai->aIjJ', h['ab'][c,C,c,C], t['a'][pv,hc], optimize='optimal')
	O['ab'][v,C,c,C] += scale * +1.00000000 * np.einsum('aIbJ,bi->aIiJ', h['ab'][v,C,v,C], t['a'][pv,hc], optimize='optimal')
	O['ab'][a,V,c,C] += scale * +1.00000000 * np.einsum('uv,wAvI,ui->wAiI', eta1['a'], h['ab'][a,V,a,C], t['a'][pa,hc], optimize='optimal')
	
	O['ab'][a,V,c,C] += scale * +1.00000000 * np.einsum('uv,wAvI,ui->wAiI', gamma1['a'], h['ab'][a,V,a,C], t['a'][pa,hc], optimize='optimal')
	
	O['ab'][a,V,c,C] += scale * -1.00000000 * np.einsum('iAjI,ui->uAjI', h['ab'][c,V,c,C], t['a'][pa,hc], optimize='optimal')
	O['ab'][a,V,c,C] += scale * +1.00000000 * np.einsum('uAaI,ai->uAiI', h['ab'][a,V,v,C], t['a'][pv,hc], optimize='optimal')
	O['ab'][c,V,c,C] += scale * +1.00000000 * np.einsum('uv,iAvI,uj->iAjI', eta1['a'], h['ab'][c,V,a,C], t['a'][pa,hc], optimize='optimal')
	O['ab'][c,V,c,C] += scale * +1.00000000 * np.einsum('uv,iAvI,uj->iAjI', gamma1['a'], h['ab'][c,V,a,C], t['a'][pa,hc], optimize='optimal')
	O['ab'][c,V,c,C] += scale * +1.00000000 * np.einsum('iAaI,aj->iAjI', h['ab'][c,V,v,C], t['a'][pv,hc], optimize='optimal')
	O['ab'][v,V,c,C] += scale * -1.00000000 * np.einsum('uv,uAiI,av->aAiI', eta1['a'], h['ab'][a,V,c,C], t['a'][pv,ha], optimize='optimal')
	O['ab'][v,V,c,C] += scale * +1.00000000 * np.einsum('uv,aAvI,ui->aAiI', eta1['a'], h['ab'][v,V,a,C], t['a'][pa,hc], optimize='optimal')
	O['ab'][v,V,c,C] += scale * -1.00000000 * np.einsum('uv,uAiI,av->aAiI', gamma1['a'], h['ab'][a,V,c,C], t['a'][pv,ha], optimize='optimal')
	O['ab'][v,V,c,C] += scale * +1.00000000 * np.einsum('uv,aAvI,ui->aAiI', gamma1['a'], h['ab'][v,V,a,C], t['a'][pa,hc], optimize='optimal')
	O['ab'][v,V,c,C] += scale * -1.00000000 * np.einsum('iAjI,ai->aAjI', h['ab'][c,V,c,C], t['a'][pv,hc], optimize='optimal')
	O['ab'][v,V,c,C] += scale * +1.00000000 * np.einsum('aAbI,bi->aAiI', h['ab'][v,V,v,C], t['a'][pv,hc], optimize='optimal')
	O['ab'][a,A,c,V] += scale * +1.00000000 * np.einsum('uv,wUvA,ui->wUiA', eta1['a'], h['ab'][a,A,a,V], t['a'][pa,hc], optimize='optimal')
	
	O['ab'][a,A,c,V] += scale * +1.00000000 * np.einsum('uv,wUvA,ui->wUiA', gamma1['a'], h['ab'][a,A,a,V], t['a'][pa,hc], optimize='optimal')
	
	O['ab'][a,A,c,V] += scale * -1.00000000 * np.einsum('iUjA,ui->uUjA', h['ab'][c,A,c,V], t['a'][pa,hc], optimize='optimal')
	O['ab'][a,A,c,V] += scale * +1.00000000 * np.einsum('uUaA,ai->uUiA', h['ab'][a,A,v,V], t['a'][pv,hc], optimize='optimal')
	O['ab'][c,A,c,V] += scale * +1.00000000 * np.einsum('uv,iUvA,uj->iUjA', eta1['a'], h['ab'][c,A,a,V], t['a'][pa,hc], optimize='optimal')
	O['ab'][c,A,c,V] += scale * +1.00000000 * np.einsum('uv,iUvA,uj->iUjA', gamma1['a'], h['ab'][c,A,a,V], t['a'][pa,hc], optimize='optimal')
	O['ab'][c,A,c,V] += scale * +1.00000000 * np.einsum('iUaA,aj->iUjA', h['ab'][c,A,v,V], t['a'][pv,hc], optimize='optimal')
	O['ab'][v,A,c,V] += scale * -1.00000000 * np.einsum('uv,uUiA,av->aUiA', eta1['a'], h['ab'][a,A,c,V], t['a'][pv,ha], optimize='optimal')
	O['ab'][v,A,c,V] += scale * +1.00000000 * np.einsum('uv,aUvA,ui->aUiA', eta1['a'], h['ab'][v,A,a,V], t['a'][pa,hc], optimize='optimal')
	O['ab'][v,A,c,V] += scale * -1.00000000 * np.einsum('uv,uUiA,av->aUiA', gamma1['a'], h['ab'][a,A,c,V], t['a'][pv,ha], optimize='optimal')
	O['ab'][v,A,c,V] += scale * +1.00000000 * np.einsum('uv,aUvA,ui->aUiA', gamma1['a'], h['ab'][v,A,a,V], t['a'][pa,hc], optimize='optimal')
	O['ab'][v,A,c,V] += scale * -1.00000000 * np.einsum('iUjA,ai->aUjA', h['ab'][c,A,c,V], t['a'][pv,hc], optimize='optimal')
	O['ab'][v,A,c,V] += scale * +1.00000000 * np.einsum('aUbA,bi->aUiA', h['ab'][v,A,v,V], t['a'][pv,hc], optimize='optimal')
	O['ab'][a,C,c,V] += scale * +1.00000000 * np.einsum('uv,wIvA,ui->wIiA', eta1['a'], h['ab'][a,C,a,V], t['a'][pa,hc], optimize='optimal')
	
	O['ab'][a,C,c,V] += scale * +1.00000000 * np.einsum('uv,wIvA,ui->wIiA', gamma1['a'], h['ab'][a,C,a,V], t['a'][pa,hc], optimize='optimal')
	
	O['ab'][a,C,c,V] += scale * -1.00000000 * np.einsum('iIjA,ui->uIjA', h['ab'][c,C,c,V], t['a'][pa,hc], optimize='optimal')
	O['ab'][a,C,c,V] += scale * +1.00000000 * np.einsum('uIaA,ai->uIiA', h['ab'][a,C,v,V], t['a'][pv,hc], optimize='optimal')
	O['ab'][c,C,c,V] += scale * +1.00000000 * np.einsum('uv,iIvA,uj->iIjA', eta1['a'], h['ab'][c,C,a,V], t['a'][pa,hc], optimize='optimal')
	O['ab'][c,C,c,V] += scale * +1.00000000 * np.einsum('uv,iIvA,uj->iIjA', gamma1['a'], h['ab'][c,C,a,V], t['a'][pa,hc], optimize='optimal')
	O['ab'][c,C,c,V] += scale * +1.00000000 * np.einsum('iIaA,aj->iIjA', h['ab'][c,C,v,V], t['a'][pv,hc], optimize='optimal')
	O['ab'][v,C,c,V] += scale * -1.00000000 * np.einsum('uv,uIiA,av->aIiA', eta1['a'], h['ab'][a,C,c,V], t['a'][pv,ha], optimize='optimal')
	O['ab'][v,C,c,V] += scale * +1.00000000 * np.einsum('uv,aIvA,ui->aIiA', eta1['a'], h['ab'][v,C,a,V], t['a'][pa,hc], optimize='optimal')
	O['ab'][v,C,c,V] += scale * -1.00000000 * np.einsum('uv,uIiA,av->aIiA', gamma1['a'], h['ab'][a,C,c,V], t['a'][pv,ha], optimize='optimal')
	O['ab'][v,C,c,V] += scale * +1.00000000 * np.einsum('uv,aIvA,ui->aIiA', gamma1['a'], h['ab'][v,C,a,V], t['a'][pa,hc], optimize='optimal')
	O['ab'][v,C,c,V] += scale * -1.00000000 * np.einsum('iIjA,ai->aIjA', h['ab'][c,C,c,V], t['a'][pv,hc], optimize='optimal')
	O['ab'][v,C,c,V] += scale * +1.00000000 * np.einsum('aIbA,bi->aIiA', h['ab'][v,C,v,V], t['a'][pv,hc], optimize='optimal')
	O['ab'][a,V,c,V] += scale * +1.00000000 * np.einsum('uv,wAvB,ui->wAiB', eta1['a'], h['ab'][a,V,a,V], t['a'][pa,hc], optimize='optimal')
	
	O['ab'][a,V,c,V] += scale * +1.00000000 * np.einsum('uv,wAvB,ui->wAiB', gamma1['a'], h['ab'][a,V,a,V], t['a'][pa,hc], optimize='optimal')
	
	O['ab'][a,V,c,V] += scale * -1.00000000 * np.einsum('iAjB,ui->uAjB', h['ab'][c,V,c,V], t['a'][pa,hc], optimize='optimal')
	O['ab'][a,V,c,V] += scale * +1.00000000 * np.einsum('uAaB,ai->uAiB', h['ab'][a,V,v,V], t['a'][pv,hc], optimize='optimal')
	O['ab'][c,V,c,V] += scale * +1.00000000 * np.einsum('uv,iAvB,uj->iAjB', eta1['a'], h['ab'][c,V,a,V], t['a'][pa,hc], optimize='optimal')
	O['ab'][c,V,c,V] += scale * +1.00000000 * np.einsum('uv,iAvB,uj->iAjB', gamma1['a'], h['ab'][c,V,a,V], t['a'][pa,hc], optimize='optimal')
	O['ab'][c,V,c,V] += scale * +1.00000000 * np.einsum('iAaB,aj->iAjB', h['ab'][c,V,v,V], t['a'][pv,hc], optimize='optimal')
	O['ab'][v,V,c,V] += scale * -1.00000000 * np.einsum('uv,uAiB,av->aAiB', eta1['a'], h['ab'][a,V,c,V], t['a'][pv,ha], optimize='optimal')
	O['ab'][v,V,c,V] += scale * +1.00000000 * np.einsum('uv,aAvB,ui->aAiB', eta1['a'], h['ab'][v,V,a,V], t['a'][pa,hc], optimize='optimal')
	O['ab'][v,V,c,V] += scale * -1.00000000 * np.einsum('uv,uAiB,av->aAiB', gamma1['a'], h['ab'][a,V,c,V], t['a'][pv,ha], optimize='optimal')
	O['ab'][v,V,c,V] += scale * +1.00000000 * np.einsum('uv,aAvB,ui->aAiB', gamma1['a'], h['ab'][v,V,a,V], t['a'][pa,hc], optimize='optimal')
	O['ab'][v,V,c,V] += scale * -1.00000000 * np.einsum('iAjB,ai->aAjB', h['ab'][c,V,c,V], t['a'][pv,hc], optimize='optimal')
	O['ab'][v,V,c,V] += scale * +1.00000000 * np.einsum('aAbB,bi->aAiB', h['ab'][v,V,v,V], t['a'][pv,hc], optimize='optimal')
	
	
	O['ab'][a,A,v,A] += scale * -1.00000000 * np.einsum('iUaV,ui->uUaV', h['ab'][c,A,v,A], t['a'][pa,hc], optimize='optimal')
	O['ab'][v,A,v,A] += scale * -1.00000000 * np.einsum('uv,uUaV,bv->bUaV', eta1['a'], h['ab'][a,A,v,A], t['a'][pv,ha], optimize='optimal')
	O['ab'][v,A,v,A] += scale * -1.00000000 * np.einsum('uv,uUaV,bv->bUaV', gamma1['a'], h['ab'][a,A,v,A], t['a'][pv,ha], optimize='optimal')
	O['ab'][v,A,v,A] += scale * -1.00000000 * np.einsum('iUaV,bi->bUaV', h['ab'][c,A,v,A], t['a'][pv,hc], optimize='optimal')
	
	
	O['ab'][a,C,v,A] += scale * -1.00000000 * np.einsum('iIaU,ui->uIaU', h['ab'][c,C,v,A], t['a'][pa,hc], optimize='optimal')
	O['ab'][v,C,v,A] += scale * -1.00000000 * np.einsum('uv,uIaU,bv->bIaU', eta1['a'], h['ab'][a,C,v,A], t['a'][pv,ha], optimize='optimal')
	O['ab'][v,C,v,A] += scale * -1.00000000 * np.einsum('uv,uIaU,bv->bIaU', gamma1['a'], h['ab'][a,C,v,A], t['a'][pv,ha], optimize='optimal')
	O['ab'][v,C,v,A] += scale * -1.00000000 * np.einsum('iIaU,bi->bIaU', h['ab'][c,C,v,A], t['a'][pv,hc], optimize='optimal')
	
	
	O['ab'][a,V,v,A] += scale * -1.00000000 * np.einsum('iAaU,ui->uAaU', h['ab'][c,V,v,A], t['a'][pa,hc], optimize='optimal')
	O['ab'][v,V,v,A] += scale * -1.00000000 * np.einsum('uv,uAaU,bv->bAaU', eta1['a'], h['ab'][a,V,v,A], t['a'][pv,ha], optimize='optimal')
	O['ab'][v,V,v,A] += scale * -1.00000000 * np.einsum('uv,uAaU,bv->bAaU', gamma1['a'], h['ab'][a,V,v,A], t['a'][pv,ha], optimize='optimal')
	O['ab'][v,V,v,A] += scale * -1.00000000 * np.einsum('iAaU,bi->bAaU', h['ab'][c,V,v,A], t['a'][pv,hc], optimize='optimal')
	
	
	O['ab'][a,A,v,C] += scale * -1.00000000 * np.einsum('iUaI,ui->uUaI', h['ab'][c,A,v,C], t['a'][pa,hc], optimize='optimal')
	O['ab'][v,A,v,C] += scale * -1.00000000 * np.einsum('uv,uUaI,bv->bUaI', eta1['a'], h['ab'][a,A,v,C], t['a'][pv,ha], optimize='optimal')
	O['ab'][v,A,v,C] += scale * -1.00000000 * np.einsum('uv,uUaI,bv->bUaI', gamma1['a'], h['ab'][a,A,v,C], t['a'][pv,ha], optimize='optimal')
	O['ab'][v,A,v,C] += scale * -1.00000000 * np.einsum('iUaI,bi->bUaI', h['ab'][c,A,v,C], t['a'][pv,hc], optimize='optimal')
	
	
	O['ab'][a,C,v,C] += scale * -1.00000000 * np.einsum('iIaJ,ui->uIaJ', h['ab'][c,C,v,C], t['a'][pa,hc], optimize='optimal')
	O['ab'][v,C,v,C] += scale * -1.00000000 * np.einsum('uv,uIaJ,bv->bIaJ', eta1['a'], h['ab'][a,C,v,C], t['a'][pv,ha], optimize='optimal')
	O['ab'][v,C,v,C] += scale * -1.00000000 * np.einsum('uv,uIaJ,bv->bIaJ', gamma1['a'], h['ab'][a,C,v,C], t['a'][pv,ha], optimize='optimal')
	O['ab'][v,C,v,C] += scale * -1.00000000 * np.einsum('iIaJ,bi->bIaJ', h['ab'][c,C,v,C], t['a'][pv,hc], optimize='optimal')
	
	
	O['ab'][a,V,v,C] += scale * -1.00000000 * np.einsum('iAaI,ui->uAaI', h['ab'][c,V,v,C], t['a'][pa,hc], optimize='optimal')
	O['ab'][v,V,v,C] += scale * -1.00000000 * np.einsum('uv,uAaI,bv->bAaI', eta1['a'], h['ab'][a,V,v,C], t['a'][pv,ha], optimize='optimal')
	O['ab'][v,V,v,C] += scale * -1.00000000 * np.einsum('uv,uAaI,bv->bAaI', gamma1['a'], h['ab'][a,V,v,C], t['a'][pv,ha], optimize='optimal')
	O['ab'][v,V,v,C] += scale * -1.00000000 * np.einsum('iAaI,bi->bAaI', h['ab'][c,V,v,C], t['a'][pv,hc], optimize='optimal')
	
	
	O['ab'][a,A,v,V] += scale * -1.00000000 * np.einsum('iUaA,ui->uUaA', h['ab'][c,A,v,V], t['a'][pa,hc], optimize='optimal')
	O['ab'][v,A,v,V] += scale * -1.00000000 * np.einsum('uv,uUaA,bv->bUaA', eta1['a'], h['ab'][a,A,v,V], t['a'][pv,ha], optimize='optimal')
	O['ab'][v,A,v,V] += scale * -1.00000000 * np.einsum('uv,uUaA,bv->bUaA', gamma1['a'], h['ab'][a,A,v,V], t['a'][pv,ha], optimize='optimal')
	O['ab'][v,A,v,V] += scale * -1.00000000 * np.einsum('iUaA,bi->bUaA', h['ab'][c,A,v,V], t['a'][pv,hc], optimize='optimal')
	
	
	O['ab'][a,C,v,V] += scale * -1.00000000 * np.einsum('iIaA,ui->uIaA', h['ab'][c,C,v,V], t['a'][pa,hc], optimize='optimal')
	O['ab'][v,C,v,V] += scale * -1.00000000 * np.einsum('uv,uIaA,bv->bIaA', eta1['a'], h['ab'][a,C,v,V], t['a'][pv,ha], optimize='optimal')
	O['ab'][v,C,v,V] += scale * -1.00000000 * np.einsum('uv,uIaA,bv->bIaA', gamma1['a'], h['ab'][a,C,v,V], t['a'][pv,ha], optimize='optimal')
	O['ab'][v,C,v,V] += scale * -1.00000000 * np.einsum('iIaA,bi->bIaA', h['ab'][c,C,v,V], t['a'][pv,hc], optimize='optimal')
	
	
	O['ab'][a,V,v,V] += scale * -1.00000000 * np.einsum('iAaB,ui->uAaB', h['ab'][c,V,v,V], t['a'][pa,hc], optimize='optimal')
	O['ab'][v,V,v,V] += scale * -1.00000000 * np.einsum('uv,uAaB,bv->bAaB', eta1['a'], h['ab'][a,V,v,V], t['a'][pv,ha], optimize='optimal')
	O['ab'][v,V,v,V] += scale * -1.00000000 * np.einsum('uv,uAaB,bv->bAaB', gamma1['a'], h['ab'][a,V,v,V], t['a'][pv,ha], optimize='optimal')
	O['ab'][v,V,v,V] += scale * -1.00000000 * np.einsum('iAaB,bi->bAaB', h['ab'][c,V,v,V], t['a'][pv,hc], optimize='optimal')

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

	
	
	
	
	O['ab'][a,A,a,A] += scale * -1.00000000 * np.einsum('uIvU,VI->uVvU', h['ab'][a,C,a,A], t['b'][pA,hC], optimize='optimal')
	O['ab'][a,A,a,A] += scale * +1.00000000 * np.einsum('uUvA,AV->uUvV', h['ab'][a,A,a,V], t['b'][pV,hA], optimize='optimal')
	
	
	
	
	O['ab'][c,A,a,A] += scale * -1.00000000 * np.einsum('iIuU,VI->iVuU', h['ab'][c,C,a,A], t['b'][pA,hC], optimize='optimal')
	O['ab'][c,A,a,A] += scale * +1.00000000 * np.einsum('iUuA,AV->iUuV', h['ab'][c,A,a,V], t['b'][pV,hA], optimize='optimal')
	
	
	
	
	O['ab'][v,A,a,A] += scale * -1.00000000 * np.einsum('aIuU,VI->aVuU', h['ab'][v,C,a,A], t['b'][pA,hC], optimize='optimal')
	O['ab'][v,A,a,A] += scale * +1.00000000 * np.einsum('aUuA,AV->aUuV', h['ab'][v,A,a,V], t['b'][pV,hA], optimize='optimal')
	
	
	O['ab'][a,C,a,A] += scale * +1.00000000 * np.einsum('uIvA,AU->uIvU', h['ab'][a,C,a,V], t['b'][pV,hA], optimize='optimal')
	
	
	O['ab'][c,C,a,A] += scale * +1.00000000 * np.einsum('iIuA,AU->iIuU', h['ab'][c,C,a,V], t['b'][pV,hA], optimize='optimal')
	
	
	O['ab'][v,C,a,A] += scale * +1.00000000 * np.einsum('aIuA,AU->aIuU', h['ab'][v,C,a,V], t['b'][pV,hA], optimize='optimal')
	O['ab'][a,V,a,A] += scale * -1.00000000 * np.einsum('UV,uUvW,AV->uAvW', eta1['b'], h['ab'][a,A,a,A], t['b'][pV,hA], optimize='optimal')
	
	O['ab'][a,V,a,A] += scale * -1.00000000 * np.einsum('UV,uUvW,AV->uAvW', gamma1['b'], h['ab'][a,A,a,A], t['b'][pV,hA], optimize='optimal')
	
	O['ab'][a,V,a,A] += scale * -1.00000000 * np.einsum('uIvU,AI->uAvU', h['ab'][a,C,a,A], t['b'][pV,hC], optimize='optimal')
	O['ab'][a,V,a,A] += scale * +1.00000000 * np.einsum('uAvB,BU->uAvU', h['ab'][a,V,a,V], t['b'][pV,hA], optimize='optimal')
	O['ab'][c,V,a,A] += scale * -1.00000000 * np.einsum('UV,iUuW,AV->iAuW', eta1['b'], h['ab'][c,A,a,A], t['b'][pV,hA], optimize='optimal')
	
	O['ab'][c,V,a,A] += scale * -1.00000000 * np.einsum('UV,iUuW,AV->iAuW', gamma1['b'], h['ab'][c,A,a,A], t['b'][pV,hA], optimize='optimal')
	
	O['ab'][c,V,a,A] += scale * -1.00000000 * np.einsum('iIuU,AI->iAuU', h['ab'][c,C,a,A], t['b'][pV,hC], optimize='optimal')
	O['ab'][c,V,a,A] += scale * +1.00000000 * np.einsum('iAuB,BU->iAuU', h['ab'][c,V,a,V], t['b'][pV,hA], optimize='optimal')
	O['ab'][v,V,a,A] += scale * -1.00000000 * np.einsum('UV,aUuW,AV->aAuW', eta1['b'], h['ab'][v,A,a,A], t['b'][pV,hA], optimize='optimal')
	
	O['ab'][v,V,a,A] += scale * -1.00000000 * np.einsum('UV,aUuW,AV->aAuW', gamma1['b'], h['ab'][v,A,a,A], t['b'][pV,hA], optimize='optimal')
	
	O['ab'][v,V,a,A] += scale * -1.00000000 * np.einsum('aIuU,AI->aAuU', h['ab'][v,C,a,A], t['b'][pV,hC], optimize='optimal')
	O['ab'][v,V,a,A] += scale * +1.00000000 * np.einsum('aAuB,BU->aAuU', h['ab'][v,V,a,V], t['b'][pV,hA], optimize='optimal')
	O['ab'][a,A,a,C] += scale * +1.00000000 * np.einsum('UV,uWvV,UI->uWvI', eta1['b'], h['ab'][a,A,a,A], t['b'][pA,hC], optimize='optimal')
	
	O['ab'][a,A,a,C] += scale * +1.00000000 * np.einsum('UV,uWvV,UI->uWvI', gamma1['b'], h['ab'][a,A,a,A], t['b'][pA,hC], optimize='optimal')
	
	O['ab'][a,A,a,C] += scale * -1.00000000 * np.einsum('uIvJ,UI->uUvJ', h['ab'][a,C,a,C], t['b'][pA,hC], optimize='optimal')
	O['ab'][a,A,a,C] += scale * +1.00000000 * np.einsum('uUvA,AI->uUvI', h['ab'][a,A,a,V], t['b'][pV,hC], optimize='optimal')
	O['ab'][c,A,a,C] += scale * +1.00000000 * np.einsum('UV,iWuV,UI->iWuI', eta1['b'], h['ab'][c,A,a,A], t['b'][pA,hC], optimize='optimal')
	
	O['ab'][c,A,a,C] += scale * +1.00000000 * np.einsum('UV,iWuV,UI->iWuI', gamma1['b'], h['ab'][c,A,a,A], t['b'][pA,hC], optimize='optimal')
	
	O['ab'][c,A,a,C] += scale * -1.00000000 * np.einsum('iIuJ,UI->iUuJ', h['ab'][c,C,a,C], t['b'][pA,hC], optimize='optimal')
	O['ab'][c,A,a,C] += scale * +1.00000000 * np.einsum('iUuA,AI->iUuI', h['ab'][c,A,a,V], t['b'][pV,hC], optimize='optimal')
	O['ab'][v,A,a,C] += scale * +1.00000000 * np.einsum('UV,aWuV,UI->aWuI', eta1['b'], h['ab'][v,A,a,A], t['b'][pA,hC], optimize='optimal')
	
	O['ab'][v,A,a,C] += scale * +1.00000000 * np.einsum('UV,aWuV,UI->aWuI', gamma1['b'], h['ab'][v,A,a,A], t['b'][pA,hC], optimize='optimal')
	
	O['ab'][v,A,a,C] += scale * -1.00000000 * np.einsum('aIuJ,UI->aUuJ', h['ab'][v,C,a,C], t['b'][pA,hC], optimize='optimal')
	O['ab'][v,A,a,C] += scale * +1.00000000 * np.einsum('aUuA,AI->aUuI', h['ab'][v,A,a,V], t['b'][pV,hC], optimize='optimal')
	O['ab'][a,C,a,C] += scale * +1.00000000 * np.einsum('UV,uIvV,UJ->uIvJ', eta1['b'], h['ab'][a,C,a,A], t['b'][pA,hC], optimize='optimal')
	O['ab'][a,C,a,C] += scale * +1.00000000 * np.einsum('UV,uIvV,UJ->uIvJ', gamma1['b'], h['ab'][a,C,a,A], t['b'][pA,hC], optimize='optimal')
	O['ab'][a,C,a,C] += scale * +1.00000000 * np.einsum('uIvA,AJ->uIvJ', h['ab'][a,C,a,V], t['b'][pV,hC], optimize='optimal')
	O['ab'][c,C,a,C] += scale * +1.00000000 * np.einsum('UV,iIuV,UJ->iIuJ', eta1['b'], h['ab'][c,C,a,A], t['b'][pA,hC], optimize='optimal')
	O['ab'][c,C,a,C] += scale * +1.00000000 * np.einsum('UV,iIuV,UJ->iIuJ', gamma1['b'], h['ab'][c,C,a,A], t['b'][pA,hC], optimize='optimal')
	O['ab'][c,C,a,C] += scale * +1.00000000 * np.einsum('iIuA,AJ->iIuJ', h['ab'][c,C,a,V], t['b'][pV,hC], optimize='optimal')
	O['ab'][v,C,a,C] += scale * +1.00000000 * np.einsum('UV,aIuV,UJ->aIuJ', eta1['b'], h['ab'][v,C,a,A], t['b'][pA,hC], optimize='optimal')
	O['ab'][v,C,a,C] += scale * +1.00000000 * np.einsum('UV,aIuV,UJ->aIuJ', gamma1['b'], h['ab'][v,C,a,A], t['b'][pA,hC], optimize='optimal')
	O['ab'][v,C,a,C] += scale * +1.00000000 * np.einsum('aIuA,AJ->aIuJ', h['ab'][v,C,a,V], t['b'][pV,hC], optimize='optimal')
	O['ab'][a,V,a,C] += scale * -1.00000000 * np.einsum('UV,uUvI,AV->uAvI', eta1['b'], h['ab'][a,A,a,C], t['b'][pV,hA], optimize='optimal')
	O['ab'][a,V,a,C] += scale * +1.00000000 * np.einsum('UV,uAvV,UI->uAvI', eta1['b'], h['ab'][a,V,a,A], t['b'][pA,hC], optimize='optimal')
	O['ab'][a,V,a,C] += scale * -1.00000000 * np.einsum('UV,uUvI,AV->uAvI', gamma1['b'], h['ab'][a,A,a,C], t['b'][pV,hA], optimize='optimal')
	O['ab'][a,V,a,C] += scale * +1.00000000 * np.einsum('UV,uAvV,UI->uAvI', gamma1['b'], h['ab'][a,V,a,A], t['b'][pA,hC], optimize='optimal')
	O['ab'][a,V,a,C] += scale * -1.00000000 * np.einsum('uIvJ,AI->uAvJ', h['ab'][a,C,a,C], t['b'][pV,hC], optimize='optimal')
	O['ab'][a,V,a,C] += scale * +1.00000000 * np.einsum('uAvB,BI->uAvI', h['ab'][a,V,a,V], t['b'][pV,hC], optimize='optimal')
	O['ab'][c,V,a,C] += scale * -1.00000000 * np.einsum('UV,iUuI,AV->iAuI', eta1['b'], h['ab'][c,A,a,C], t['b'][pV,hA], optimize='optimal')
	O['ab'][c,V,a,C] += scale * +1.00000000 * np.einsum('UV,iAuV,UI->iAuI', eta1['b'], h['ab'][c,V,a,A], t['b'][pA,hC], optimize='optimal')
	O['ab'][c,V,a,C] += scale * -1.00000000 * np.einsum('UV,iUuI,AV->iAuI', gamma1['b'], h['ab'][c,A,a,C], t['b'][pV,hA], optimize='optimal')
	O['ab'][c,V,a,C] += scale * +1.00000000 * np.einsum('UV,iAuV,UI->iAuI', gamma1['b'], h['ab'][c,V,a,A], t['b'][pA,hC], optimize='optimal')
	O['ab'][c,V,a,C] += scale * -1.00000000 * np.einsum('iIuJ,AI->iAuJ', h['ab'][c,C,a,C], t['b'][pV,hC], optimize='optimal')
	O['ab'][c,V,a,C] += scale * +1.00000000 * np.einsum('iAuB,BI->iAuI', h['ab'][c,V,a,V], t['b'][pV,hC], optimize='optimal')
	O['ab'][v,V,a,C] += scale * -1.00000000 * np.einsum('UV,aUuI,AV->aAuI', eta1['b'], h['ab'][v,A,a,C], t['b'][pV,hA], optimize='optimal')
	O['ab'][v,V,a,C] += scale * +1.00000000 * np.einsum('UV,aAuV,UI->aAuI', eta1['b'], h['ab'][v,V,a,A], t['b'][pA,hC], optimize='optimal')
	O['ab'][v,V,a,C] += scale * -1.00000000 * np.einsum('UV,aUuI,AV->aAuI', gamma1['b'], h['ab'][v,A,a,C], t['b'][pV,hA], optimize='optimal')
	O['ab'][v,V,a,C] += scale * +1.00000000 * np.einsum('UV,aAuV,UI->aAuI', gamma1['b'], h['ab'][v,V,a,A], t['b'][pA,hC], optimize='optimal')
	O['ab'][v,V,a,C] += scale * -1.00000000 * np.einsum('aIuJ,AI->aAuJ', h['ab'][v,C,a,C], t['b'][pV,hC], optimize='optimal')
	O['ab'][v,V,a,C] += scale * +1.00000000 * np.einsum('aAuB,BI->aAuI', h['ab'][v,V,a,V], t['b'][pV,hC], optimize='optimal')
	
	
	O['ab'][a,A,a,V] += scale * -1.00000000 * np.einsum('uIvA,UI->uUvA', h['ab'][a,C,a,V], t['b'][pA,hC], optimize='optimal')
	
	
	O['ab'][c,A,a,V] += scale * -1.00000000 * np.einsum('iIuA,UI->iUuA', h['ab'][c,C,a,V], t['b'][pA,hC], optimize='optimal')
	
	
	O['ab'][v,A,a,V] += scale * -1.00000000 * np.einsum('aIuA,UI->aUuA', h['ab'][v,C,a,V], t['b'][pA,hC], optimize='optimal')
	O['ab'][a,V,a,V] += scale * -1.00000000 * np.einsum('UV,uUvA,BV->uBvA', eta1['b'], h['ab'][a,A,a,V], t['b'][pV,hA], optimize='optimal')
	O['ab'][a,V,a,V] += scale * -1.00000000 * np.einsum('UV,uUvA,BV->uBvA', gamma1['b'], h['ab'][a,A,a,V], t['b'][pV,hA], optimize='optimal')
	O['ab'][a,V,a,V] += scale * -1.00000000 * np.einsum('uIvA,BI->uBvA', h['ab'][a,C,a,V], t['b'][pV,hC], optimize='optimal')
	O['ab'][c,V,a,V] += scale * -1.00000000 * np.einsum('UV,iUuA,BV->iBuA', eta1['b'], h['ab'][c,A,a,V], t['b'][pV,hA], optimize='optimal')
	O['ab'][c,V,a,V] += scale * -1.00000000 * np.einsum('UV,iUuA,BV->iBuA', gamma1['b'], h['ab'][c,A,a,V], t['b'][pV,hA], optimize='optimal')
	O['ab'][c,V,a,V] += scale * -1.00000000 * np.einsum('iIuA,BI->iBuA', h['ab'][c,C,a,V], t['b'][pV,hC], optimize='optimal')
	O['ab'][v,V,a,V] += scale * -1.00000000 * np.einsum('UV,aUuA,BV->aBuA', eta1['b'], h['ab'][v,A,a,V], t['b'][pV,hA], optimize='optimal')
	O['ab'][v,V,a,V] += scale * -1.00000000 * np.einsum('UV,aUuA,BV->aBuA', gamma1['b'], h['ab'][v,A,a,V], t['b'][pV,hA], optimize='optimal')
	O['ab'][v,V,a,V] += scale * -1.00000000 * np.einsum('aIuA,BI->aBuA', h['ab'][v,C,a,V], t['b'][pV,hC], optimize='optimal')
	
	
	
	
	O['ab'][a,A,c,A] += scale * -1.00000000 * np.einsum('uIiU,VI->uViU', h['ab'][a,C,c,A], t['b'][pA,hC], optimize='optimal')
	O['ab'][a,A,c,A] += scale * +1.00000000 * np.einsum('uUiA,AV->uUiV', h['ab'][a,A,c,V], t['b'][pV,hA], optimize='optimal')
	
	
	
	
	O['ab'][c,A,c,A] += scale * -1.00000000 * np.einsum('iIjU,VI->iVjU', h['ab'][c,C,c,A], t['b'][pA,hC], optimize='optimal')
	O['ab'][c,A,c,A] += scale * +1.00000000 * np.einsum('iUjA,AV->iUjV', h['ab'][c,A,c,V], t['b'][pV,hA], optimize='optimal')
	
	
	
	
	O['ab'][v,A,c,A] += scale * -1.00000000 * np.einsum('aIiU,VI->aViU', h['ab'][v,C,c,A], t['b'][pA,hC], optimize='optimal')
	O['ab'][v,A,c,A] += scale * +1.00000000 * np.einsum('aUiA,AV->aUiV', h['ab'][v,A,c,V], t['b'][pV,hA], optimize='optimal')
	
	
	O['ab'][a,C,c,A] += scale * +1.00000000 * np.einsum('uIiA,AU->uIiU', h['ab'][a,C,c,V], t['b'][pV,hA], optimize='optimal')
	
	
	O['ab'][c,C,c,A] += scale * +1.00000000 * np.einsum('iIjA,AU->iIjU', h['ab'][c,C,c,V], t['b'][pV,hA], optimize='optimal')
	
	
	O['ab'][v,C,c,A] += scale * +1.00000000 * np.einsum('aIiA,AU->aIiU', h['ab'][v,C,c,V], t['b'][pV,hA], optimize='optimal')
	O['ab'][a,V,c,A] += scale * -1.00000000 * np.einsum('UV,uUiW,AV->uAiW', eta1['b'], h['ab'][a,A,c,A], t['b'][pV,hA], optimize='optimal')
	
	O['ab'][a,V,c,A] += scale * -1.00000000 * np.einsum('UV,uUiW,AV->uAiW', gamma1['b'], h['ab'][a,A,c,A], t['b'][pV,hA], optimize='optimal')
	
	O['ab'][a,V,c,A] += scale * -1.00000000 * np.einsum('uIiU,AI->uAiU', h['ab'][a,C,c,A], t['b'][pV,hC], optimize='optimal')
	O['ab'][a,V,c,A] += scale * +1.00000000 * np.einsum('uAiB,BU->uAiU', h['ab'][a,V,c,V], t['b'][pV,hA], optimize='optimal')
	O['ab'][c,V,c,A] += scale * -1.00000000 * np.einsum('UV,iUjW,AV->iAjW', eta1['b'], h['ab'][c,A,c,A], t['b'][pV,hA], optimize='optimal')
	
	O['ab'][c,V,c,A] += scale * -1.00000000 * np.einsum('UV,iUjW,AV->iAjW', gamma1['b'], h['ab'][c,A,c,A], t['b'][pV,hA], optimize='optimal')
	
	O['ab'][c,V,c,A] += scale * -1.00000000 * np.einsum('iIjU,AI->iAjU', h['ab'][c,C,c,A], t['b'][pV,hC], optimize='optimal')
	O['ab'][c,V,c,A] += scale * +1.00000000 * np.einsum('iAjB,BU->iAjU', h['ab'][c,V,c,V], t['b'][pV,hA], optimize='optimal')
	O['ab'][v,V,c,A] += scale * -1.00000000 * np.einsum('UV,aUiW,AV->aAiW', eta1['b'], h['ab'][v,A,c,A], t['b'][pV,hA], optimize='optimal')
	
	O['ab'][v,V,c,A] += scale * -1.00000000 * np.einsum('UV,aUiW,AV->aAiW', gamma1['b'], h['ab'][v,A,c,A], t['b'][pV,hA], optimize='optimal')
	
	O['ab'][v,V,c,A] += scale * -1.00000000 * np.einsum('aIiU,AI->aAiU', h['ab'][v,C,c,A], t['b'][pV,hC], optimize='optimal')
	O['ab'][v,V,c,A] += scale * +1.00000000 * np.einsum('aAiB,BU->aAiU', h['ab'][v,V,c,V], t['b'][pV,hA], optimize='optimal')
	O['ab'][a,A,c,C] += scale * +1.00000000 * np.einsum('UV,uWiV,UI->uWiI', eta1['b'], h['ab'][a,A,c,A], t['b'][pA,hC], optimize='optimal')
	
	O['ab'][a,A,c,C] += scale * +1.00000000 * np.einsum('UV,uWiV,UI->uWiI', gamma1['b'], h['ab'][a,A,c,A], t['b'][pA,hC], optimize='optimal')
	
	O['ab'][a,A,c,C] += scale * -1.00000000 * np.einsum('uIiJ,UI->uUiJ', h['ab'][a,C,c,C], t['b'][pA,hC], optimize='optimal')
	O['ab'][a,A,c,C] += scale * +1.00000000 * np.einsum('uUiA,AI->uUiI', h['ab'][a,A,c,V], t['b'][pV,hC], optimize='optimal')
	O['ab'][c,A,c,C] += scale * +1.00000000 * np.einsum('UV,iWjV,UI->iWjI', eta1['b'], h['ab'][c,A,c,A], t['b'][pA,hC], optimize='optimal')
	
	O['ab'][c,A,c,C] += scale * +1.00000000 * np.einsum('UV,iWjV,UI->iWjI', gamma1['b'], h['ab'][c,A,c,A], t['b'][pA,hC], optimize='optimal')
	
	O['ab'][c,A,c,C] += scale * -1.00000000 * np.einsum('iIjJ,UI->iUjJ', h['ab'][c,C,c,C], t['b'][pA,hC], optimize='optimal')
	O['ab'][c,A,c,C] += scale * +1.00000000 * np.einsum('iUjA,AI->iUjI', h['ab'][c,A,c,V], t['b'][pV,hC], optimize='optimal')
	O['ab'][v,A,c,C] += scale * +1.00000000 * np.einsum('UV,aWiV,UI->aWiI', eta1['b'], h['ab'][v,A,c,A], t['b'][pA,hC], optimize='optimal')
	
	O['ab'][v,A,c,C] += scale * +1.00000000 * np.einsum('UV,aWiV,UI->aWiI', gamma1['b'], h['ab'][v,A,c,A], t['b'][pA,hC], optimize='optimal')
	
	O['ab'][v,A,c,C] += scale * -1.00000000 * np.einsum('aIiJ,UI->aUiJ', h['ab'][v,C,c,C], t['b'][pA,hC], optimize='optimal')
	O['ab'][v,A,c,C] += scale * +1.00000000 * np.einsum('aUiA,AI->aUiI', h['ab'][v,A,c,V], t['b'][pV,hC], optimize='optimal')
	O['ab'][a,C,c,C] += scale * +1.00000000 * np.einsum('UV,uIiV,UJ->uIiJ', eta1['b'], h['ab'][a,C,c,A], t['b'][pA,hC], optimize='optimal')
	O['ab'][a,C,c,C] += scale * +1.00000000 * np.einsum('UV,uIiV,UJ->uIiJ', gamma1['b'], h['ab'][a,C,c,A], t['b'][pA,hC], optimize='optimal')
	O['ab'][a,C,c,C] += scale * +1.00000000 * np.einsum('uIiA,AJ->uIiJ', h['ab'][a,C,c,V], t['b'][pV,hC], optimize='optimal')
	O['ab'][c,C,c,C] += scale * +1.00000000 * np.einsum('UV,iIjV,UJ->iIjJ', eta1['b'], h['ab'][c,C,c,A], t['b'][pA,hC], optimize='optimal')
	O['ab'][c,C,c,C] += scale * +1.00000000 * np.einsum('UV,iIjV,UJ->iIjJ', gamma1['b'], h['ab'][c,C,c,A], t['b'][pA,hC], optimize='optimal')
	O['ab'][c,C,c,C] += scale * +1.00000000 * np.einsum('iIjA,AJ->iIjJ', h['ab'][c,C,c,V], t['b'][pV,hC], optimize='optimal')
	O['ab'][v,C,c,C] += scale * +1.00000000 * np.einsum('UV,aIiV,UJ->aIiJ', eta1['b'], h['ab'][v,C,c,A], t['b'][pA,hC], optimize='optimal')
	O['ab'][v,C,c,C] += scale * +1.00000000 * np.einsum('UV,aIiV,UJ->aIiJ', gamma1['b'], h['ab'][v,C,c,A], t['b'][pA,hC], optimize='optimal')
	O['ab'][v,C,c,C] += scale * +1.00000000 * np.einsum('aIiA,AJ->aIiJ', h['ab'][v,C,c,V], t['b'][pV,hC], optimize='optimal')
	O['ab'][a,V,c,C] += scale * -1.00000000 * np.einsum('UV,uUiI,AV->uAiI', eta1['b'], h['ab'][a,A,c,C], t['b'][pV,hA], optimize='optimal')
	O['ab'][a,V,c,C] += scale * +1.00000000 * np.einsum('UV,uAiV,UI->uAiI', eta1['b'], h['ab'][a,V,c,A], t['b'][pA,hC], optimize='optimal')
	O['ab'][a,V,c,C] += scale * -1.00000000 * np.einsum('UV,uUiI,AV->uAiI', gamma1['b'], h['ab'][a,A,c,C], t['b'][pV,hA], optimize='optimal')
	O['ab'][a,V,c,C] += scale * +1.00000000 * np.einsum('UV,uAiV,UI->uAiI', gamma1['b'], h['ab'][a,V,c,A], t['b'][pA,hC], optimize='optimal')
	O['ab'][a,V,c,C] += scale * -1.00000000 * np.einsum('uIiJ,AI->uAiJ', h['ab'][a,C,c,C], t['b'][pV,hC], optimize='optimal')
	O['ab'][a,V,c,C] += scale * +1.00000000 * np.einsum('uAiB,BI->uAiI', h['ab'][a,V,c,V], t['b'][pV,hC], optimize='optimal')
	O['ab'][c,V,c,C] += scale * -1.00000000 * np.einsum('UV,iUjI,AV->iAjI', eta1['b'], h['ab'][c,A,c,C], t['b'][pV,hA], optimize='optimal')
	O['ab'][c,V,c,C] += scale * +1.00000000 * np.einsum('UV,iAjV,UI->iAjI', eta1['b'], h['ab'][c,V,c,A], t['b'][pA,hC], optimize='optimal')
	O['ab'][c,V,c,C] += scale * -1.00000000 * np.einsum('UV,iUjI,AV->iAjI', gamma1['b'], h['ab'][c,A,c,C], t['b'][pV,hA], optimize='optimal')
	O['ab'][c,V,c,C] += scale * +1.00000000 * np.einsum('UV,iAjV,UI->iAjI', gamma1['b'], h['ab'][c,V,c,A], t['b'][pA,hC], optimize='optimal')
	O['ab'][c,V,c,C] += scale * -1.00000000 * np.einsum('iIjJ,AI->iAjJ', h['ab'][c,C,c,C], t['b'][pV,hC], optimize='optimal')
	O['ab'][c,V,c,C] += scale * +1.00000000 * np.einsum('iAjB,BI->iAjI', h['ab'][c,V,c,V], t['b'][pV,hC], optimize='optimal')
	O['ab'][v,V,c,C] += scale * -1.00000000 * np.einsum('UV,aUiI,AV->aAiI', eta1['b'], h['ab'][v,A,c,C], t['b'][pV,hA], optimize='optimal')
	O['ab'][v,V,c,C] += scale * +1.00000000 * np.einsum('UV,aAiV,UI->aAiI', eta1['b'], h['ab'][v,V,c,A], t['b'][pA,hC], optimize='optimal')
	O['ab'][v,V,c,C] += scale * -1.00000000 * np.einsum('UV,aUiI,AV->aAiI', gamma1['b'], h['ab'][v,A,c,C], t['b'][pV,hA], optimize='optimal')
	O['ab'][v,V,c,C] += scale * +1.00000000 * np.einsum('UV,aAiV,UI->aAiI', gamma1['b'], h['ab'][v,V,c,A], t['b'][pA,hC], optimize='optimal')
	O['ab'][v,V,c,C] += scale * -1.00000000 * np.einsum('aIiJ,AI->aAiJ', h['ab'][v,C,c,C], t['b'][pV,hC], optimize='optimal')
	O['ab'][v,V,c,C] += scale * +1.00000000 * np.einsum('aAiB,BI->aAiI', h['ab'][v,V,c,V], t['b'][pV,hC], optimize='optimal')
	
	
	O['ab'][a,A,c,V] += scale * -1.00000000 * np.einsum('uIiA,UI->uUiA', h['ab'][a,C,c,V], t['b'][pA,hC], optimize='optimal')
	
	
	O['ab'][c,A,c,V] += scale * -1.00000000 * np.einsum('iIjA,UI->iUjA', h['ab'][c,C,c,V], t['b'][pA,hC], optimize='optimal')
	
	
	O['ab'][v,A,c,V] += scale * -1.00000000 * np.einsum('aIiA,UI->aUiA', h['ab'][v,C,c,V], t['b'][pA,hC], optimize='optimal')
	O['ab'][a,V,c,V] += scale * -1.00000000 * np.einsum('UV,uUiA,BV->uBiA', eta1['b'], h['ab'][a,A,c,V], t['b'][pV,hA], optimize='optimal')
	O['ab'][a,V,c,V] += scale * -1.00000000 * np.einsum('UV,uUiA,BV->uBiA', gamma1['b'], h['ab'][a,A,c,V], t['b'][pV,hA], optimize='optimal')
	O['ab'][a,V,c,V] += scale * -1.00000000 * np.einsum('uIiA,BI->uBiA', h['ab'][a,C,c,V], t['b'][pV,hC], optimize='optimal')
	O['ab'][c,V,c,V] += scale * -1.00000000 * np.einsum('UV,iUjA,BV->iBjA', eta1['b'], h['ab'][c,A,c,V], t['b'][pV,hA], optimize='optimal')
	O['ab'][c,V,c,V] += scale * -1.00000000 * np.einsum('UV,iUjA,BV->iBjA', gamma1['b'], h['ab'][c,A,c,V], t['b'][pV,hA], optimize='optimal')
	O['ab'][c,V,c,V] += scale * -1.00000000 * np.einsum('iIjA,BI->iBjA', h['ab'][c,C,c,V], t['b'][pV,hC], optimize='optimal')
	O['ab'][v,V,c,V] += scale * -1.00000000 * np.einsum('UV,aUiA,BV->aBiA', eta1['b'], h['ab'][v,A,c,V], t['b'][pV,hA], optimize='optimal')
	O['ab'][v,V,c,V] += scale * -1.00000000 * np.einsum('UV,aUiA,BV->aBiA', gamma1['b'], h['ab'][v,A,c,V], t['b'][pV,hA], optimize='optimal')
	O['ab'][v,V,c,V] += scale * -1.00000000 * np.einsum('aIiA,BI->aBiA', h['ab'][v,C,c,V], t['b'][pV,hC], optimize='optimal')
	
	
	
	
	O['ab'][a,A,v,A] += scale * -1.00000000 * np.einsum('uIaU,VI->uVaU', h['ab'][a,C,v,A], t['b'][pA,hC], optimize='optimal')
	O['ab'][a,A,v,A] += scale * +1.00000000 * np.einsum('uUaA,AV->uUaV', h['ab'][a,A,v,V], t['b'][pV,hA], optimize='optimal')
	
	
	
	
	O['ab'][c,A,v,A] += scale * -1.00000000 * np.einsum('iIaU,VI->iVaU', h['ab'][c,C,v,A], t['b'][pA,hC], optimize='optimal')
	O['ab'][c,A,v,A] += scale * +1.00000000 * np.einsum('iUaA,AV->iUaV', h['ab'][c,A,v,V], t['b'][pV,hA], optimize='optimal')
	
	
	
	
	O['ab'][v,A,v,A] += scale * -1.00000000 * np.einsum('aIbU,VI->aVbU', h['ab'][v,C,v,A], t['b'][pA,hC], optimize='optimal')
	O['ab'][v,A,v,A] += scale * +1.00000000 * np.einsum('aUbA,AV->aUbV', h['ab'][v,A,v,V], t['b'][pV,hA], optimize='optimal')
	
	
	O['ab'][a,C,v,A] += scale * +1.00000000 * np.einsum('uIaA,AU->uIaU', h['ab'][a,C,v,V], t['b'][pV,hA], optimize='optimal')
	
	
	O['ab'][c,C,v,A] += scale * +1.00000000 * np.einsum('iIaA,AU->iIaU', h['ab'][c,C,v,V], t['b'][pV,hA], optimize='optimal')
	
	
	O['ab'][v,C,v,A] += scale * +1.00000000 * np.einsum('aIbA,AU->aIbU', h['ab'][v,C,v,V], t['b'][pV,hA], optimize='optimal')
	O['ab'][a,V,v,A] += scale * -1.00000000 * np.einsum('UV,uUaW,AV->uAaW', eta1['b'], h['ab'][a,A,v,A], t['b'][pV,hA], optimize='optimal')
	
	O['ab'][a,V,v,A] += scale * -1.00000000 * np.einsum('UV,uUaW,AV->uAaW', gamma1['b'], h['ab'][a,A,v,A], t['b'][pV,hA], optimize='optimal')
	
	O['ab'][a,V,v,A] += scale * -1.00000000 * np.einsum('uIaU,AI->uAaU', h['ab'][a,C,v,A], t['b'][pV,hC], optimize='optimal')
	O['ab'][a,V,v,A] += scale * +1.00000000 * np.einsum('uAaB,BU->uAaU', h['ab'][a,V,v,V], t['b'][pV,hA], optimize='optimal')
	O['ab'][c,V,v,A] += scale * -1.00000000 * np.einsum('UV,iUaW,AV->iAaW', eta1['b'], h['ab'][c,A,v,A], t['b'][pV,hA], optimize='optimal')
	
	O['ab'][c,V,v,A] += scale * -1.00000000 * np.einsum('UV,iUaW,AV->iAaW', gamma1['b'], h['ab'][c,A,v,A], t['b'][pV,hA], optimize='optimal')
	
	O['ab'][c,V,v,A] += scale * -1.00000000 * np.einsum('iIaU,AI->iAaU', h['ab'][c,C,v,A], t['b'][pV,hC], optimize='optimal')
	O['ab'][c,V,v,A] += scale * +1.00000000 * np.einsum('iAaB,BU->iAaU', h['ab'][c,V,v,V], t['b'][pV,hA], optimize='optimal')
	O['ab'][v,V,v,A] += scale * -1.00000000 * np.einsum('UV,aUbW,AV->aAbW', eta1['b'], h['ab'][v,A,v,A], t['b'][pV,hA], optimize='optimal')
	
	O['ab'][v,V,v,A] += scale * -1.00000000 * np.einsum('UV,aUbW,AV->aAbW', gamma1['b'], h['ab'][v,A,v,A], t['b'][pV,hA], optimize='optimal')
	
	O['ab'][v,V,v,A] += scale * -1.00000000 * np.einsum('aIbU,AI->aAbU', h['ab'][v,C,v,A], t['b'][pV,hC], optimize='optimal')
	O['ab'][v,V,v,A] += scale * +1.00000000 * np.einsum('aAbB,BU->aAbU', h['ab'][v,V,v,V], t['b'][pV,hA], optimize='optimal')
	O['ab'][a,A,v,C] += scale * +1.00000000 * np.einsum('UV,uWaV,UI->uWaI', eta1['b'], h['ab'][a,A,v,A], t['b'][pA,hC], optimize='optimal')
	
	O['ab'][a,A,v,C] += scale * +1.00000000 * np.einsum('UV,uWaV,UI->uWaI', gamma1['b'], h['ab'][a,A,v,A], t['b'][pA,hC], optimize='optimal')
	
	O['ab'][a,A,v,C] += scale * -1.00000000 * np.einsum('uIaJ,UI->uUaJ', h['ab'][a,C,v,C], t['b'][pA,hC], optimize='optimal')
	O['ab'][a,A,v,C] += scale * +1.00000000 * np.einsum('uUaA,AI->uUaI', h['ab'][a,A,v,V], t['b'][pV,hC], optimize='optimal')
	O['ab'][c,A,v,C] += scale * +1.00000000 * np.einsum('UV,iWaV,UI->iWaI', eta1['b'], h['ab'][c,A,v,A], t['b'][pA,hC], optimize='optimal')
	
	O['ab'][c,A,v,C] += scale * +1.00000000 * np.einsum('UV,iWaV,UI->iWaI', gamma1['b'], h['ab'][c,A,v,A], t['b'][pA,hC], optimize='optimal')
	
	O['ab'][c,A,v,C] += scale * -1.00000000 * np.einsum('iIaJ,UI->iUaJ', h['ab'][c,C,v,C], t['b'][pA,hC], optimize='optimal')
	O['ab'][c,A,v,C] += scale * +1.00000000 * np.einsum('iUaA,AI->iUaI', h['ab'][c,A,v,V], t['b'][pV,hC], optimize='optimal')
	O['ab'][v,A,v,C] += scale * +1.00000000 * np.einsum('UV,aWbV,UI->aWbI', eta1['b'], h['ab'][v,A,v,A], t['b'][pA,hC], optimize='optimal')
	
	O['ab'][v,A,v,C] += scale * +1.00000000 * np.einsum('UV,aWbV,UI->aWbI', gamma1['b'], h['ab'][v,A,v,A], t['b'][pA,hC], optimize='optimal')
	
	O['ab'][v,A,v,C] += scale * -1.00000000 * np.einsum('aIbJ,UI->aUbJ', h['ab'][v,C,v,C], t['b'][pA,hC], optimize='optimal')
	O['ab'][v,A,v,C] += scale * +1.00000000 * np.einsum('aUbA,AI->aUbI', h['ab'][v,A,v,V], t['b'][pV,hC], optimize='optimal')
	O['ab'][a,C,v,C] += scale * +1.00000000 * np.einsum('UV,uIaV,UJ->uIaJ', eta1['b'], h['ab'][a,C,v,A], t['b'][pA,hC], optimize='optimal')
	O['ab'][a,C,v,C] += scale * +1.00000000 * np.einsum('UV,uIaV,UJ->uIaJ', gamma1['b'], h['ab'][a,C,v,A], t['b'][pA,hC], optimize='optimal')
	O['ab'][a,C,v,C] += scale * +1.00000000 * np.einsum('uIaA,AJ->uIaJ', h['ab'][a,C,v,V], t['b'][pV,hC], optimize='optimal')
	O['ab'][c,C,v,C] += scale * +1.00000000 * np.einsum('UV,iIaV,UJ->iIaJ', eta1['b'], h['ab'][c,C,v,A], t['b'][pA,hC], optimize='optimal')
	O['ab'][c,C,v,C] += scale * +1.00000000 * np.einsum('UV,iIaV,UJ->iIaJ', gamma1['b'], h['ab'][c,C,v,A], t['b'][pA,hC], optimize='optimal')
	O['ab'][c,C,v,C] += scale * +1.00000000 * np.einsum('iIaA,AJ->iIaJ', h['ab'][c,C,v,V], t['b'][pV,hC], optimize='optimal')
	O['ab'][v,C,v,C] += scale * +1.00000000 * np.einsum('UV,aIbV,UJ->aIbJ', eta1['b'], h['ab'][v,C,v,A], t['b'][pA,hC], optimize='optimal')
	O['ab'][v,C,v,C] += scale * +1.00000000 * np.einsum('UV,aIbV,UJ->aIbJ', gamma1['b'], h['ab'][v,C,v,A], t['b'][pA,hC], optimize='optimal')
	O['ab'][v,C,v,C] += scale * +1.00000000 * np.einsum('aIbA,AJ->aIbJ', h['ab'][v,C,v,V], t['b'][pV,hC], optimize='optimal')
	O['ab'][a,V,v,C] += scale * -1.00000000 * np.einsum('UV,uUaI,AV->uAaI', eta1['b'], h['ab'][a,A,v,C], t['b'][pV,hA], optimize='optimal')
	O['ab'][a,V,v,C] += scale * +1.00000000 * np.einsum('UV,uAaV,UI->uAaI', eta1['b'], h['ab'][a,V,v,A], t['b'][pA,hC], optimize='optimal')
	O['ab'][a,V,v,C] += scale * -1.00000000 * np.einsum('UV,uUaI,AV->uAaI', gamma1['b'], h['ab'][a,A,v,C], t['b'][pV,hA], optimize='optimal')
	O['ab'][a,V,v,C] += scale * +1.00000000 * np.einsum('UV,uAaV,UI->uAaI', gamma1['b'], h['ab'][a,V,v,A], t['b'][pA,hC], optimize='optimal')
	O['ab'][a,V,v,C] += scale * -1.00000000 * np.einsum('uIaJ,AI->uAaJ', h['ab'][a,C,v,C], t['b'][pV,hC], optimize='optimal')
	O['ab'][a,V,v,C] += scale * +1.00000000 * np.einsum('uAaB,BI->uAaI', h['ab'][a,V,v,V], t['b'][pV,hC], optimize='optimal')
	O['ab'][c,V,v,C] += scale * -1.00000000 * np.einsum('UV,iUaI,AV->iAaI', eta1['b'], h['ab'][c,A,v,C], t['b'][pV,hA], optimize='optimal')
	O['ab'][c,V,v,C] += scale * +1.00000000 * np.einsum('UV,iAaV,UI->iAaI', eta1['b'], h['ab'][c,V,v,A], t['b'][pA,hC], optimize='optimal')
	O['ab'][c,V,v,C] += scale * -1.00000000 * np.einsum('UV,iUaI,AV->iAaI', gamma1['b'], h['ab'][c,A,v,C], t['b'][pV,hA], optimize='optimal')
	O['ab'][c,V,v,C] += scale * +1.00000000 * np.einsum('UV,iAaV,UI->iAaI', gamma1['b'], h['ab'][c,V,v,A], t['b'][pA,hC], optimize='optimal')
	O['ab'][c,V,v,C] += scale * -1.00000000 * np.einsum('iIaJ,AI->iAaJ', h['ab'][c,C,v,C], t['b'][pV,hC], optimize='optimal')
	O['ab'][c,V,v,C] += scale * +1.00000000 * np.einsum('iAaB,BI->iAaI', h['ab'][c,V,v,V], t['b'][pV,hC], optimize='optimal')
	O['ab'][v,V,v,C] += scale * -1.00000000 * np.einsum('UV,aUbI,AV->aAbI', eta1['b'], h['ab'][v,A,v,C], t['b'][pV,hA], optimize='optimal')
	O['ab'][v,V,v,C] += scale * +1.00000000 * np.einsum('UV,aAbV,UI->aAbI', eta1['b'], h['ab'][v,V,v,A], t['b'][pA,hC], optimize='optimal')
	O['ab'][v,V,v,C] += scale * -1.00000000 * np.einsum('UV,aUbI,AV->aAbI', gamma1['b'], h['ab'][v,A,v,C], t['b'][pV,hA], optimize='optimal')
	O['ab'][v,V,v,C] += scale * +1.00000000 * np.einsum('UV,aAbV,UI->aAbI', gamma1['b'], h['ab'][v,V,v,A], t['b'][pA,hC], optimize='optimal')
	O['ab'][v,V,v,C] += scale * -1.00000000 * np.einsum('aIbJ,AI->aAbJ', h['ab'][v,C,v,C], t['b'][pV,hC], optimize='optimal')
	O['ab'][v,V,v,C] += scale * +1.00000000 * np.einsum('aAbB,BI->aAbI', h['ab'][v,V,v,V], t['b'][pV,hC], optimize='optimal')
	
	
	O['ab'][a,A,v,V] += scale * -1.00000000 * np.einsum('uIaA,UI->uUaA', h['ab'][a,C,v,V], t['b'][pA,hC], optimize='optimal')
	
	
	O['ab'][c,A,v,V] += scale * -1.00000000 * np.einsum('iIaA,UI->iUaA', h['ab'][c,C,v,V], t['b'][pA,hC], optimize='optimal')
	
	
	O['ab'][v,A,v,V] += scale * -1.00000000 * np.einsum('aIbA,UI->aUbA', h['ab'][v,C,v,V], t['b'][pA,hC], optimize='optimal')
	O['ab'][a,V,v,V] += scale * -1.00000000 * np.einsum('UV,uUaA,BV->uBaA', eta1['b'], h['ab'][a,A,v,V], t['b'][pV,hA], optimize='optimal')
	O['ab'][a,V,v,V] += scale * -1.00000000 * np.einsum('UV,uUaA,BV->uBaA', gamma1['b'], h['ab'][a,A,v,V], t['b'][pV,hA], optimize='optimal')
	O['ab'][a,V,v,V] += scale * -1.00000000 * np.einsum('uIaA,BI->uBaA', h['ab'][a,C,v,V], t['b'][pV,hC], optimize='optimal')
	O['ab'][c,V,v,V] += scale * -1.00000000 * np.einsum('UV,iUaA,BV->iBaA', eta1['b'], h['ab'][c,A,v,V], t['b'][pV,hA], optimize='optimal')
	O['ab'][c,V,v,V] += scale * -1.00000000 * np.einsum('UV,iUaA,BV->iBaA', gamma1['b'], h['ab'][c,A,v,V], t['b'][pV,hA], optimize='optimal')
	O['ab'][c,V,v,V] += scale * -1.00000000 * np.einsum('iIaA,BI->iBaA', h['ab'][c,C,v,V], t['b'][pV,hC], optimize='optimal')
	O['ab'][v,V,v,V] += scale * -1.00000000 * np.einsum('UV,aUbA,BV->aBbA', eta1['b'], h['ab'][v,A,v,V], t['b'][pV,hA], optimize='optimal')
	O['ab'][v,V,v,V] += scale * -1.00000000 * np.einsum('UV,aUbA,BV->aBbA', gamma1['b'], h['ab'][v,A,v,V], t['b'][pV,hA], optimize='optimal')
	O['ab'][v,V,v,V] += scale * -1.00000000 * np.einsum('aIbA,BI->aBbA', h['ab'][v,C,v,V], t['b'][pV,hC], optimize='optimal')

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

	
	
	O['ab'][a,A,a,A] += scale * -1.00000000 * np.einsum('uv,iUvV,wuix->wUxV', eta1['a'], h['ab'][c,A,a,A], t['aa'][pa,pa,hc,ha], optimize='optimal')
	O['ab'][a,A,a,A] += scale * +1.00000000 * np.einsum('uv,uUaV,waxv->wUxV', gamma1['a'], h['ab'][a,A,v,A], t['aa'][pa,pv,ha,ha], optimize='optimal')
	O['ab'][a,A,a,A] += scale * -1.00000000 * np.einsum('iUaV,uaiv->uUvV', h['ab'][c,A,v,A], t['aa'][pa,pv,hc,ha], optimize='optimal')
	O['ab'][v,A,a,A] += scale * +1.00000000 * np.einsum('uv,wx,uUxV,wayv->aUyV', eta1['a'], gamma1['a'], h['ab'][a,A,a,A], t['aa'][pa,pv,ha,ha], optimize='optimal')
	O['ab'][v,A,a,A] += scale * -1.00000000 * np.einsum('uv,wx,wUvV,uayx->aUyV', eta1['a'], gamma1['a'], h['ab'][a,A,a,A], t['aa'][pa,pv,ha,ha], optimize='optimal')
	O['ab'][v,A,a,A] += scale * +1.00000000 * np.einsum('uv,iUvV,uaiw->aUwV', eta1['a'], h['ab'][c,A,a,A], t['aa'][pa,pv,hc,ha], optimize='optimal')
	O['ab'][v,A,a,A] += scale * +1.00000000 * np.einsum('uv,uUaV,bawv->bUwV', gamma1['a'], h['ab'][a,A,v,A], t['aa'][pv,pv,ha,ha], optimize='optimal')
	O['ab'][v,A,a,A] += scale * -1.00000000 * np.einsum('iUaV,baiu->bUuV', h['ab'][c,A,v,A], t['aa'][pv,pv,hc,ha], optimize='optimal')
	
	
	O['ab'][a,C,a,A] += scale * -1.00000000 * np.einsum('uv,iIvU,wuix->wIxU', eta1['a'], h['ab'][c,C,a,A], t['aa'][pa,pa,hc,ha], optimize='optimal')
	O['ab'][a,C,a,A] += scale * +1.00000000 * np.einsum('uv,uIaU,waxv->wIxU', gamma1['a'], h['ab'][a,C,v,A], t['aa'][pa,pv,ha,ha], optimize='optimal')
	O['ab'][a,C,a,A] += scale * -1.00000000 * np.einsum('iIaU,uaiv->uIvU', h['ab'][c,C,v,A], t['aa'][pa,pv,hc,ha], optimize='optimal')
	O['ab'][v,C,a,A] += scale * +1.00000000 * np.einsum('uv,wx,uIxU,wayv->aIyU', eta1['a'], gamma1['a'], h['ab'][a,C,a,A], t['aa'][pa,pv,ha,ha], optimize='optimal')
	O['ab'][v,C,a,A] += scale * -1.00000000 * np.einsum('uv,wx,wIvU,uayx->aIyU', eta1['a'], gamma1['a'], h['ab'][a,C,a,A], t['aa'][pa,pv,ha,ha], optimize='optimal')
	O['ab'][v,C,a,A] += scale * +1.00000000 * np.einsum('uv,iIvU,uaiw->aIwU', eta1['a'], h['ab'][c,C,a,A], t['aa'][pa,pv,hc,ha], optimize='optimal')
	O['ab'][v,C,a,A] += scale * +1.00000000 * np.einsum('uv,uIaU,bawv->bIwU', gamma1['a'], h['ab'][a,C,v,A], t['aa'][pv,pv,ha,ha], optimize='optimal')
	O['ab'][v,C,a,A] += scale * -1.00000000 * np.einsum('iIaU,baiu->bIuU', h['ab'][c,C,v,A], t['aa'][pv,pv,hc,ha], optimize='optimal')
	
	
	O['ab'][a,V,a,A] += scale * -1.00000000 * np.einsum('uv,iAvU,wuix->wAxU', eta1['a'], h['ab'][c,V,a,A], t['aa'][pa,pa,hc,ha], optimize='optimal')
	O['ab'][a,V,a,A] += scale * +1.00000000 * np.einsum('uv,uAaU,waxv->wAxU', gamma1['a'], h['ab'][a,V,v,A], t['aa'][pa,pv,ha,ha], optimize='optimal')
	O['ab'][a,V,a,A] += scale * -1.00000000 * np.einsum('iAaU,uaiv->uAvU', h['ab'][c,V,v,A], t['aa'][pa,pv,hc,ha], optimize='optimal')
	O['ab'][v,V,a,A] += scale * +1.00000000 * np.einsum('uv,wx,uAxU,wayv->aAyU', eta1['a'], gamma1['a'], h['ab'][a,V,a,A], t['aa'][pa,pv,ha,ha], optimize='optimal')
	O['ab'][v,V,a,A] += scale * -1.00000000 * np.einsum('uv,wx,wAvU,uayx->aAyU', eta1['a'], gamma1['a'], h['ab'][a,V,a,A], t['aa'][pa,pv,ha,ha], optimize='optimal')
	O['ab'][v,V,a,A] += scale * +1.00000000 * np.einsum('uv,iAvU,uaiw->aAwU', eta1['a'], h['ab'][c,V,a,A], t['aa'][pa,pv,hc,ha], optimize='optimal')
	O['ab'][v,V,a,A] += scale * +1.00000000 * np.einsum('uv,uAaU,bawv->bAwU', gamma1['a'], h['ab'][a,V,v,A], t['aa'][pv,pv,ha,ha], optimize='optimal')
	O['ab'][v,V,a,A] += scale * -1.00000000 * np.einsum('iAaU,baiu->bAuU', h['ab'][c,V,v,A], t['aa'][pv,pv,hc,ha], optimize='optimal')
	
	
	O['ab'][a,A,a,C] += scale * -1.00000000 * np.einsum('uv,iUvI,wuix->wUxI', eta1['a'], h['ab'][c,A,a,C], t['aa'][pa,pa,hc,ha], optimize='optimal')
	O['ab'][a,A,a,C] += scale * +1.00000000 * np.einsum('uv,uUaI,waxv->wUxI', gamma1['a'], h['ab'][a,A,v,C], t['aa'][pa,pv,ha,ha], optimize='optimal')
	O['ab'][a,A,a,C] += scale * -1.00000000 * np.einsum('iUaI,uaiv->uUvI', h['ab'][c,A,v,C], t['aa'][pa,pv,hc,ha], optimize='optimal')
	O['ab'][v,A,a,C] += scale * +1.00000000 * np.einsum('uv,wx,uUxI,wayv->aUyI', eta1['a'], gamma1['a'], h['ab'][a,A,a,C], t['aa'][pa,pv,ha,ha], optimize='optimal')
	O['ab'][v,A,a,C] += scale * -1.00000000 * np.einsum('uv,wx,wUvI,uayx->aUyI', eta1['a'], gamma1['a'], h['ab'][a,A,a,C], t['aa'][pa,pv,ha,ha], optimize='optimal')
	O['ab'][v,A,a,C] += scale * +1.00000000 * np.einsum('uv,iUvI,uaiw->aUwI', eta1['a'], h['ab'][c,A,a,C], t['aa'][pa,pv,hc,ha], optimize='optimal')
	O['ab'][v,A,a,C] += scale * +1.00000000 * np.einsum('uv,uUaI,bawv->bUwI', gamma1['a'], h['ab'][a,A,v,C], t['aa'][pv,pv,ha,ha], optimize='optimal')
	O['ab'][v,A,a,C] += scale * -1.00000000 * np.einsum('iUaI,baiu->bUuI', h['ab'][c,A,v,C], t['aa'][pv,pv,hc,ha], optimize='optimal')
	
	
	O['ab'][a,C,a,C] += scale * -1.00000000 * np.einsum('uv,iIvJ,wuix->wIxJ', eta1['a'], h['ab'][c,C,a,C], t['aa'][pa,pa,hc,ha], optimize='optimal')
	O['ab'][a,C,a,C] += scale * +1.00000000 * np.einsum('uv,uIaJ,waxv->wIxJ', gamma1['a'], h['ab'][a,C,v,C], t['aa'][pa,pv,ha,ha], optimize='optimal')
	O['ab'][a,C,a,C] += scale * -1.00000000 * np.einsum('iIaJ,uaiv->uIvJ', h['ab'][c,C,v,C], t['aa'][pa,pv,hc,ha], optimize='optimal')
	O['ab'][v,C,a,C] += scale * +1.00000000 * np.einsum('uv,wx,uIxJ,wayv->aIyJ', eta1['a'], gamma1['a'], h['ab'][a,C,a,C], t['aa'][pa,pv,ha,ha], optimize='optimal')
	O['ab'][v,C,a,C] += scale * -1.00000000 * np.einsum('uv,wx,wIvJ,uayx->aIyJ', eta1['a'], gamma1['a'], h['ab'][a,C,a,C], t['aa'][pa,pv,ha,ha], optimize='optimal')
	O['ab'][v,C,a,C] += scale * +1.00000000 * np.einsum('uv,iIvJ,uaiw->aIwJ', eta1['a'], h['ab'][c,C,a,C], t['aa'][pa,pv,hc,ha], optimize='optimal')
	O['ab'][v,C,a,C] += scale * +1.00000000 * np.einsum('uv,uIaJ,bawv->bIwJ', gamma1['a'], h['ab'][a,C,v,C], t['aa'][pv,pv,ha,ha], optimize='optimal')
	O['ab'][v,C,a,C] += scale * -1.00000000 * np.einsum('iIaJ,baiu->bIuJ', h['ab'][c,C,v,C], t['aa'][pv,pv,hc,ha], optimize='optimal')
	
	
	O['ab'][a,V,a,C] += scale * -1.00000000 * np.einsum('uv,iAvI,wuix->wAxI', eta1['a'], h['ab'][c,V,a,C], t['aa'][pa,pa,hc,ha], optimize='optimal')
	O['ab'][a,V,a,C] += scale * +1.00000000 * np.einsum('uv,uAaI,waxv->wAxI', gamma1['a'], h['ab'][a,V,v,C], t['aa'][pa,pv,ha,ha], optimize='optimal')
	O['ab'][a,V,a,C] += scale * -1.00000000 * np.einsum('iAaI,uaiv->uAvI', h['ab'][c,V,v,C], t['aa'][pa,pv,hc,ha], optimize='optimal')
	O['ab'][v,V,a,C] += scale * +1.00000000 * np.einsum('uv,wx,uAxI,wayv->aAyI', eta1['a'], gamma1['a'], h['ab'][a,V,a,C], t['aa'][pa,pv,ha,ha], optimize='optimal')
	O['ab'][v,V,a,C] += scale * -1.00000000 * np.einsum('uv,wx,wAvI,uayx->aAyI', eta1['a'], gamma1['a'], h['ab'][a,V,a,C], t['aa'][pa,pv,ha,ha], optimize='optimal')
	O['ab'][v,V,a,C] += scale * +1.00000000 * np.einsum('uv,iAvI,uaiw->aAwI', eta1['a'], h['ab'][c,V,a,C], t['aa'][pa,pv,hc,ha], optimize='optimal')
	O['ab'][v,V,a,C] += scale * +1.00000000 * np.einsum('uv,uAaI,bawv->bAwI', gamma1['a'], h['ab'][a,V,v,C], t['aa'][pv,pv,ha,ha], optimize='optimal')
	O['ab'][v,V,a,C] += scale * -1.00000000 * np.einsum('iAaI,baiu->bAuI', h['ab'][c,V,v,C], t['aa'][pv,pv,hc,ha], optimize='optimal')
	
	
	O['ab'][a,A,a,V] += scale * -1.00000000 * np.einsum('uv,iUvA,wuix->wUxA', eta1['a'], h['ab'][c,A,a,V], t['aa'][pa,pa,hc,ha], optimize='optimal')
	O['ab'][a,A,a,V] += scale * +1.00000000 * np.einsum('uv,uUaA,waxv->wUxA', gamma1['a'], h['ab'][a,A,v,V], t['aa'][pa,pv,ha,ha], optimize='optimal')
	O['ab'][a,A,a,V] += scale * -1.00000000 * np.einsum('iUaA,uaiv->uUvA', h['ab'][c,A,v,V], t['aa'][pa,pv,hc,ha], optimize='optimal')
	O['ab'][v,A,a,V] += scale * +1.00000000 * np.einsum('uv,wx,uUxA,wayv->aUyA', eta1['a'], gamma1['a'], h['ab'][a,A,a,V], t['aa'][pa,pv,ha,ha], optimize='optimal')
	O['ab'][v,A,a,V] += scale * -1.00000000 * np.einsum('uv,wx,wUvA,uayx->aUyA', eta1['a'], gamma1['a'], h['ab'][a,A,a,V], t['aa'][pa,pv,ha,ha], optimize='optimal')
	O['ab'][v,A,a,V] += scale * +1.00000000 * np.einsum('uv,iUvA,uaiw->aUwA', eta1['a'], h['ab'][c,A,a,V], t['aa'][pa,pv,hc,ha], optimize='optimal')
	O['ab'][v,A,a,V] += scale * +1.00000000 * np.einsum('uv,uUaA,bawv->bUwA', gamma1['a'], h['ab'][a,A,v,V], t['aa'][pv,pv,ha,ha], optimize='optimal')
	O['ab'][v,A,a,V] += scale * -1.00000000 * np.einsum('iUaA,baiu->bUuA', h['ab'][c,A,v,V], t['aa'][pv,pv,hc,ha], optimize='optimal')
	
	
	O['ab'][a,C,a,V] += scale * -1.00000000 * np.einsum('uv,iIvA,wuix->wIxA', eta1['a'], h['ab'][c,C,a,V], t['aa'][pa,pa,hc,ha], optimize='optimal')
	O['ab'][a,C,a,V] += scale * +1.00000000 * np.einsum('uv,uIaA,waxv->wIxA', gamma1['a'], h['ab'][a,C,v,V], t['aa'][pa,pv,ha,ha], optimize='optimal')
	O['ab'][a,C,a,V] += scale * -1.00000000 * np.einsum('iIaA,uaiv->uIvA', h['ab'][c,C,v,V], t['aa'][pa,pv,hc,ha], optimize='optimal')
	O['ab'][v,C,a,V] += scale * +1.00000000 * np.einsum('uv,wx,uIxA,wayv->aIyA', eta1['a'], gamma1['a'], h['ab'][a,C,a,V], t['aa'][pa,pv,ha,ha], optimize='optimal')
	O['ab'][v,C,a,V] += scale * -1.00000000 * np.einsum('uv,wx,wIvA,uayx->aIyA', eta1['a'], gamma1['a'], h['ab'][a,C,a,V], t['aa'][pa,pv,ha,ha], optimize='optimal')
	O['ab'][v,C,a,V] += scale * +1.00000000 * np.einsum('uv,iIvA,uaiw->aIwA', eta1['a'], h['ab'][c,C,a,V], t['aa'][pa,pv,hc,ha], optimize='optimal')
	O['ab'][v,C,a,V] += scale * +1.00000000 * np.einsum('uv,uIaA,bawv->bIwA', gamma1['a'], h['ab'][a,C,v,V], t['aa'][pv,pv,ha,ha], optimize='optimal')
	O['ab'][v,C,a,V] += scale * -1.00000000 * np.einsum('iIaA,baiu->bIuA', h['ab'][c,C,v,V], t['aa'][pv,pv,hc,ha], optimize='optimal')
	
	
	O['ab'][a,V,a,V] += scale * -1.00000000 * np.einsum('uv,iAvB,wuix->wAxB', eta1['a'], h['ab'][c,V,a,V], t['aa'][pa,pa,hc,ha], optimize='optimal')
	O['ab'][a,V,a,V] += scale * +1.00000000 * np.einsum('uv,uAaB,waxv->wAxB', gamma1['a'], h['ab'][a,V,v,V], t['aa'][pa,pv,ha,ha], optimize='optimal')
	O['ab'][a,V,a,V] += scale * -1.00000000 * np.einsum('iAaB,uaiv->uAvB', h['ab'][c,V,v,V], t['aa'][pa,pv,hc,ha], optimize='optimal')
	O['ab'][v,V,a,V] += scale * +1.00000000 * np.einsum('uv,wx,uAxB,wayv->aAyB', eta1['a'], gamma1['a'], h['ab'][a,V,a,V], t['aa'][pa,pv,ha,ha], optimize='optimal')
	O['ab'][v,V,a,V] += scale * -1.00000000 * np.einsum('uv,wx,wAvB,uayx->aAyB', eta1['a'], gamma1['a'], h['ab'][a,V,a,V], t['aa'][pa,pv,ha,ha], optimize='optimal')
	O['ab'][v,V,a,V] += scale * +1.00000000 * np.einsum('uv,iAvB,uaiw->aAwB', eta1['a'], h['ab'][c,V,a,V], t['aa'][pa,pv,hc,ha], optimize='optimal')
	O['ab'][v,V,a,V] += scale * +1.00000000 * np.einsum('uv,uAaB,bawv->bAwB', gamma1['a'], h['ab'][a,V,v,V], t['aa'][pv,pv,ha,ha], optimize='optimal')
	O['ab'][v,V,a,V] += scale * -1.00000000 * np.einsum('iAaB,baiu->bAuB', h['ab'][c,V,v,V], t['aa'][pv,pv,hc,ha], optimize='optimal')
	O['ab'][a,A,c,A] += scale * -1.00000000 * np.einsum('uv,wx,uUxV,ywiv->yUiV', eta1['a'], gamma1['a'], h['ab'][a,A,a,A], t['aa'][pa,pa,hc,ha], optimize='optimal')
	O['ab'][a,A,c,A] += scale * +1.00000000 * np.einsum('uv,wx,wUvV,yuix->yUiV', eta1['a'], gamma1['a'], h['ab'][a,A,a,A], t['aa'][pa,pa,hc,ha], optimize='optimal')
	O['ab'][a,A,c,A] += scale * +1.00000000 * np.einsum('uv,iUvV,wuji->wUjV', eta1['a'], h['ab'][c,A,a,A], t['aa'][pa,pa,hc,hc], optimize='optimal')
	O['ab'][a,A,c,A] += scale * +1.00000000 * np.einsum('uv,uUaV,waiv->wUiV', gamma1['a'], h['ab'][a,A,v,A], t['aa'][pa,pv,hc,ha], optimize='optimal')
	O['ab'][a,A,c,A] += scale * +1.00000000 * np.einsum('iUaV,uaji->uUjV', h['ab'][c,A,v,A], t['aa'][pa,pv,hc,hc], optimize='optimal')
	O['ab'][v,A,c,A] += scale * +1.00000000 * np.einsum('uv,wx,uUxV,waiv->aUiV', eta1['a'], gamma1['a'], h['ab'][a,A,a,A], t['aa'][pa,pv,hc,ha], optimize='optimal')
	O['ab'][v,A,c,A] += scale * -1.00000000 * np.einsum('uv,wx,wUvV,uaix->aUiV', eta1['a'], gamma1['a'], h['ab'][a,A,a,A], t['aa'][pa,pv,hc,ha], optimize='optimal')
	O['ab'][v,A,c,A] += scale * -1.00000000 * np.einsum('uv,iUvV,uaji->aUjV', eta1['a'], h['ab'][c,A,a,A], t['aa'][pa,pv,hc,hc], optimize='optimal')
	O['ab'][v,A,c,A] += scale * +1.00000000 * np.einsum('uv,uUaV,baiv->bUiV', gamma1['a'], h['ab'][a,A,v,A], t['aa'][pv,pv,hc,ha], optimize='optimal')
	O['ab'][v,A,c,A] += scale * +1.00000000 * np.einsum('iUaV,baji->bUjV', h['ab'][c,A,v,A], t['aa'][pv,pv,hc,hc], optimize='optimal')
	O['ab'][a,C,c,A] += scale * -1.00000000 * np.einsum('uv,wx,uIxU,ywiv->yIiU', eta1['a'], gamma1['a'], h['ab'][a,C,a,A], t['aa'][pa,pa,hc,ha], optimize='optimal')
	O['ab'][a,C,c,A] += scale * +1.00000000 * np.einsum('uv,wx,wIvU,yuix->yIiU', eta1['a'], gamma1['a'], h['ab'][a,C,a,A], t['aa'][pa,pa,hc,ha], optimize='optimal')
	O['ab'][a,C,c,A] += scale * +1.00000000 * np.einsum('uv,iIvU,wuji->wIjU', eta1['a'], h['ab'][c,C,a,A], t['aa'][pa,pa,hc,hc], optimize='optimal')
	O['ab'][a,C,c,A] += scale * +1.00000000 * np.einsum('uv,uIaU,waiv->wIiU', gamma1['a'], h['ab'][a,C,v,A], t['aa'][pa,pv,hc,ha], optimize='optimal')
	O['ab'][a,C,c,A] += scale * +1.00000000 * np.einsum('iIaU,uaji->uIjU', h['ab'][c,C,v,A], t['aa'][pa,pv,hc,hc], optimize='optimal')
	O['ab'][v,C,c,A] += scale * +1.00000000 * np.einsum('uv,wx,uIxU,waiv->aIiU', eta1['a'], gamma1['a'], h['ab'][a,C,a,A], t['aa'][pa,pv,hc,ha], optimize='optimal')
	O['ab'][v,C,c,A] += scale * -1.00000000 * np.einsum('uv,wx,wIvU,uaix->aIiU', eta1['a'], gamma1['a'], h['ab'][a,C,a,A], t['aa'][pa,pv,hc,ha], optimize='optimal')
	O['ab'][v,C,c,A] += scale * -1.00000000 * np.einsum('uv,iIvU,uaji->aIjU', eta1['a'], h['ab'][c,C,a,A], t['aa'][pa,pv,hc,hc], optimize='optimal')
	O['ab'][v,C,c,A] += scale * +1.00000000 * np.einsum('uv,uIaU,baiv->bIiU', gamma1['a'], h['ab'][a,C,v,A], t['aa'][pv,pv,hc,ha], optimize='optimal')
	O['ab'][v,C,c,A] += scale * +1.00000000 * np.einsum('iIaU,baji->bIjU', h['ab'][c,C,v,A], t['aa'][pv,pv,hc,hc], optimize='optimal')
	O['ab'][a,V,c,A] += scale * -1.00000000 * np.einsum('uv,wx,uAxU,ywiv->yAiU', eta1['a'], gamma1['a'], h['ab'][a,V,a,A], t['aa'][pa,pa,hc,ha], optimize='optimal')
	O['ab'][a,V,c,A] += scale * +1.00000000 * np.einsum('uv,wx,wAvU,yuix->yAiU', eta1['a'], gamma1['a'], h['ab'][a,V,a,A], t['aa'][pa,pa,hc,ha], optimize='optimal')
	O['ab'][a,V,c,A] += scale * +1.00000000 * np.einsum('uv,iAvU,wuji->wAjU', eta1['a'], h['ab'][c,V,a,A], t['aa'][pa,pa,hc,hc], optimize='optimal')
	O['ab'][a,V,c,A] += scale * +1.00000000 * np.einsum('uv,uAaU,waiv->wAiU', gamma1['a'], h['ab'][a,V,v,A], t['aa'][pa,pv,hc,ha], optimize='optimal')
	O['ab'][a,V,c,A] += scale * +1.00000000 * np.einsum('iAaU,uaji->uAjU', h['ab'][c,V,v,A], t['aa'][pa,pv,hc,hc], optimize='optimal')
	O['ab'][v,V,c,A] += scale * +1.00000000 * np.einsum('uv,wx,uAxU,waiv->aAiU', eta1['a'], gamma1['a'], h['ab'][a,V,a,A], t['aa'][pa,pv,hc,ha], optimize='optimal')
	O['ab'][v,V,c,A] += scale * -1.00000000 * np.einsum('uv,wx,wAvU,uaix->aAiU', eta1['a'], gamma1['a'], h['ab'][a,V,a,A], t['aa'][pa,pv,hc,ha], optimize='optimal')
	O['ab'][v,V,c,A] += scale * -1.00000000 * np.einsum('uv,iAvU,uaji->aAjU', eta1['a'], h['ab'][c,V,a,A], t['aa'][pa,pv,hc,hc], optimize='optimal')
	O['ab'][v,V,c,A] += scale * +1.00000000 * np.einsum('uv,uAaU,baiv->bAiU', gamma1['a'], h['ab'][a,V,v,A], t['aa'][pv,pv,hc,ha], optimize='optimal')
	O['ab'][v,V,c,A] += scale * +1.00000000 * np.einsum('iAaU,baji->bAjU', h['ab'][c,V,v,A], t['aa'][pv,pv,hc,hc], optimize='optimal')
	O['ab'][a,A,c,C] += scale * -1.00000000 * np.einsum('uv,wx,uUxI,ywiv->yUiI', eta1['a'], gamma1['a'], h['ab'][a,A,a,C], t['aa'][pa,pa,hc,ha], optimize='optimal')
	O['ab'][a,A,c,C] += scale * +1.00000000 * np.einsum('uv,wx,wUvI,yuix->yUiI', eta1['a'], gamma1['a'], h['ab'][a,A,a,C], t['aa'][pa,pa,hc,ha], optimize='optimal')
	O['ab'][a,A,c,C] += scale * +1.00000000 * np.einsum('uv,iUvI,wuji->wUjI', eta1['a'], h['ab'][c,A,a,C], t['aa'][pa,pa,hc,hc], optimize='optimal')
	O['ab'][a,A,c,C] += scale * +1.00000000 * np.einsum('uv,uUaI,waiv->wUiI', gamma1['a'], h['ab'][a,A,v,C], t['aa'][pa,pv,hc,ha], optimize='optimal')
	O['ab'][a,A,c,C] += scale * +1.00000000 * np.einsum('iUaI,uaji->uUjI', h['ab'][c,A,v,C], t['aa'][pa,pv,hc,hc], optimize='optimal')
	O['ab'][v,A,c,C] += scale * +1.00000000 * np.einsum('uv,wx,uUxI,waiv->aUiI', eta1['a'], gamma1['a'], h['ab'][a,A,a,C], t['aa'][pa,pv,hc,ha], optimize='optimal')
	O['ab'][v,A,c,C] += scale * -1.00000000 * np.einsum('uv,wx,wUvI,uaix->aUiI', eta1['a'], gamma1['a'], h['ab'][a,A,a,C], t['aa'][pa,pv,hc,ha], optimize='optimal')
	O['ab'][v,A,c,C] += scale * -1.00000000 * np.einsum('uv,iUvI,uaji->aUjI', eta1['a'], h['ab'][c,A,a,C], t['aa'][pa,pv,hc,hc], optimize='optimal')
	O['ab'][v,A,c,C] += scale * +1.00000000 * np.einsum('uv,uUaI,baiv->bUiI', gamma1['a'], h['ab'][a,A,v,C], t['aa'][pv,pv,hc,ha], optimize='optimal')
	O['ab'][v,A,c,C] += scale * +1.00000000 * np.einsum('iUaI,baji->bUjI', h['ab'][c,A,v,C], t['aa'][pv,pv,hc,hc], optimize='optimal')
	O['ab'][a,C,c,C] += scale * -1.00000000 * np.einsum('uv,wx,uIxJ,ywiv->yIiJ', eta1['a'], gamma1['a'], h['ab'][a,C,a,C], t['aa'][pa,pa,hc,ha], optimize='optimal')
	O['ab'][a,C,c,C] += scale * +1.00000000 * np.einsum('uv,wx,wIvJ,yuix->yIiJ', eta1['a'], gamma1['a'], h['ab'][a,C,a,C], t['aa'][pa,pa,hc,ha], optimize='optimal')
	O['ab'][a,C,c,C] += scale * +1.00000000 * np.einsum('uv,iIvJ,wuji->wIjJ', eta1['a'], h['ab'][c,C,a,C], t['aa'][pa,pa,hc,hc], optimize='optimal')
	O['ab'][a,C,c,C] += scale * +1.00000000 * np.einsum('uv,uIaJ,waiv->wIiJ', gamma1['a'], h['ab'][a,C,v,C], t['aa'][pa,pv,hc,ha], optimize='optimal')
	O['ab'][a,C,c,C] += scale * +1.00000000 * np.einsum('iIaJ,uaji->uIjJ', h['ab'][c,C,v,C], t['aa'][pa,pv,hc,hc], optimize='optimal')
	O['ab'][v,C,c,C] += scale * +1.00000000 * np.einsum('uv,wx,uIxJ,waiv->aIiJ', eta1['a'], gamma1['a'], h['ab'][a,C,a,C], t['aa'][pa,pv,hc,ha], optimize='optimal')
	O['ab'][v,C,c,C] += scale * -1.00000000 * np.einsum('uv,wx,wIvJ,uaix->aIiJ', eta1['a'], gamma1['a'], h['ab'][a,C,a,C], t['aa'][pa,pv,hc,ha], optimize='optimal')
	O['ab'][v,C,c,C] += scale * -1.00000000 * np.einsum('uv,iIvJ,uaji->aIjJ', eta1['a'], h['ab'][c,C,a,C], t['aa'][pa,pv,hc,hc], optimize='optimal')
	O['ab'][v,C,c,C] += scale * +1.00000000 * np.einsum('uv,uIaJ,baiv->bIiJ', gamma1['a'], h['ab'][a,C,v,C], t['aa'][pv,pv,hc,ha], optimize='optimal')
	O['ab'][v,C,c,C] += scale * +1.00000000 * np.einsum('iIaJ,baji->bIjJ', h['ab'][c,C,v,C], t['aa'][pv,pv,hc,hc], optimize='optimal')
	O['ab'][a,V,c,C] += scale * -1.00000000 * np.einsum('uv,wx,uAxI,ywiv->yAiI', eta1['a'], gamma1['a'], h['ab'][a,V,a,C], t['aa'][pa,pa,hc,ha], optimize='optimal')
	O['ab'][a,V,c,C] += scale * +1.00000000 * np.einsum('uv,wx,wAvI,yuix->yAiI', eta1['a'], gamma1['a'], h['ab'][a,V,a,C], t['aa'][pa,pa,hc,ha], optimize='optimal')
	O['ab'][a,V,c,C] += scale * +1.00000000 * np.einsum('uv,iAvI,wuji->wAjI', eta1['a'], h['ab'][c,V,a,C], t['aa'][pa,pa,hc,hc], optimize='optimal')
	O['ab'][a,V,c,C] += scale * +1.00000000 * np.einsum('uv,uAaI,waiv->wAiI', gamma1['a'], h['ab'][a,V,v,C], t['aa'][pa,pv,hc,ha], optimize='optimal')
	O['ab'][a,V,c,C] += scale * +1.00000000 * np.einsum('iAaI,uaji->uAjI', h['ab'][c,V,v,C], t['aa'][pa,pv,hc,hc], optimize='optimal')
	O['ab'][v,V,c,C] += scale * +1.00000000 * np.einsum('uv,wx,uAxI,waiv->aAiI', eta1['a'], gamma1['a'], h['ab'][a,V,a,C], t['aa'][pa,pv,hc,ha], optimize='optimal')
	O['ab'][v,V,c,C] += scale * -1.00000000 * np.einsum('uv,wx,wAvI,uaix->aAiI', eta1['a'], gamma1['a'], h['ab'][a,V,a,C], t['aa'][pa,pv,hc,ha], optimize='optimal')
	O['ab'][v,V,c,C] += scale * -1.00000000 * np.einsum('uv,iAvI,uaji->aAjI', eta1['a'], h['ab'][c,V,a,C], t['aa'][pa,pv,hc,hc], optimize='optimal')
	O['ab'][v,V,c,C] += scale * +1.00000000 * np.einsum('uv,uAaI,baiv->bAiI', gamma1['a'], h['ab'][a,V,v,C], t['aa'][pv,pv,hc,ha], optimize='optimal')
	O['ab'][v,V,c,C] += scale * +1.00000000 * np.einsum('iAaI,baji->bAjI', h['ab'][c,V,v,C], t['aa'][pv,pv,hc,hc], optimize='optimal')
	O['ab'][a,A,c,V] += scale * -1.00000000 * np.einsum('uv,wx,uUxA,ywiv->yUiA', eta1['a'], gamma1['a'], h['ab'][a,A,a,V], t['aa'][pa,pa,hc,ha], optimize='optimal')
	O['ab'][a,A,c,V] += scale * +1.00000000 * np.einsum('uv,wx,wUvA,yuix->yUiA', eta1['a'], gamma1['a'], h['ab'][a,A,a,V], t['aa'][pa,pa,hc,ha], optimize='optimal')
	O['ab'][a,A,c,V] += scale * +1.00000000 * np.einsum('uv,iUvA,wuji->wUjA', eta1['a'], h['ab'][c,A,a,V], t['aa'][pa,pa,hc,hc], optimize='optimal')
	O['ab'][a,A,c,V] += scale * +1.00000000 * np.einsum('uv,uUaA,waiv->wUiA', gamma1['a'], h['ab'][a,A,v,V], t['aa'][pa,pv,hc,ha], optimize='optimal')
	O['ab'][a,A,c,V] += scale * +1.00000000 * np.einsum('iUaA,uaji->uUjA', h['ab'][c,A,v,V], t['aa'][pa,pv,hc,hc], optimize='optimal')
	O['ab'][v,A,c,V] += scale * +1.00000000 * np.einsum('uv,wx,uUxA,waiv->aUiA', eta1['a'], gamma1['a'], h['ab'][a,A,a,V], t['aa'][pa,pv,hc,ha], optimize='optimal')
	O['ab'][v,A,c,V] += scale * -1.00000000 * np.einsum('uv,wx,wUvA,uaix->aUiA', eta1['a'], gamma1['a'], h['ab'][a,A,a,V], t['aa'][pa,pv,hc,ha], optimize='optimal')
	O['ab'][v,A,c,V] += scale * -1.00000000 * np.einsum('uv,iUvA,uaji->aUjA', eta1['a'], h['ab'][c,A,a,V], t['aa'][pa,pv,hc,hc], optimize='optimal')
	O['ab'][v,A,c,V] += scale * +1.00000000 * np.einsum('uv,uUaA,baiv->bUiA', gamma1['a'], h['ab'][a,A,v,V], t['aa'][pv,pv,hc,ha], optimize='optimal')
	O['ab'][v,A,c,V] += scale * +1.00000000 * np.einsum('iUaA,baji->bUjA', h['ab'][c,A,v,V], t['aa'][pv,pv,hc,hc], optimize='optimal')
	O['ab'][a,C,c,V] += scale * -1.00000000 * np.einsum('uv,wx,uIxA,ywiv->yIiA', eta1['a'], gamma1['a'], h['ab'][a,C,a,V], t['aa'][pa,pa,hc,ha], optimize='optimal')
	O['ab'][a,C,c,V] += scale * +1.00000000 * np.einsum('uv,wx,wIvA,yuix->yIiA', eta1['a'], gamma1['a'], h['ab'][a,C,a,V], t['aa'][pa,pa,hc,ha], optimize='optimal')
	O['ab'][a,C,c,V] += scale * +1.00000000 * np.einsum('uv,iIvA,wuji->wIjA', eta1['a'], h['ab'][c,C,a,V], t['aa'][pa,pa,hc,hc], optimize='optimal')
	O['ab'][a,C,c,V] += scale * +1.00000000 * np.einsum('uv,uIaA,waiv->wIiA', gamma1['a'], h['ab'][a,C,v,V], t['aa'][pa,pv,hc,ha], optimize='optimal')
	O['ab'][a,C,c,V] += scale * +1.00000000 * np.einsum('iIaA,uaji->uIjA', h['ab'][c,C,v,V], t['aa'][pa,pv,hc,hc], optimize='optimal')
	O['ab'][v,C,c,V] += scale * +1.00000000 * np.einsum('uv,wx,uIxA,waiv->aIiA', eta1['a'], gamma1['a'], h['ab'][a,C,a,V], t['aa'][pa,pv,hc,ha], optimize='optimal')
	O['ab'][v,C,c,V] += scale * -1.00000000 * np.einsum('uv,wx,wIvA,uaix->aIiA', eta1['a'], gamma1['a'], h['ab'][a,C,a,V], t['aa'][pa,pv,hc,ha], optimize='optimal')
	O['ab'][v,C,c,V] += scale * -1.00000000 * np.einsum('uv,iIvA,uaji->aIjA', eta1['a'], h['ab'][c,C,a,V], t['aa'][pa,pv,hc,hc], optimize='optimal')
	O['ab'][v,C,c,V] += scale * +1.00000000 * np.einsum('uv,uIaA,baiv->bIiA', gamma1['a'], h['ab'][a,C,v,V], t['aa'][pv,pv,hc,ha], optimize='optimal')
	O['ab'][v,C,c,V] += scale * +1.00000000 * np.einsum('iIaA,baji->bIjA', h['ab'][c,C,v,V], t['aa'][pv,pv,hc,hc], optimize='optimal')
	O['ab'][a,V,c,V] += scale * -1.00000000 * np.einsum('uv,wx,uAxB,ywiv->yAiB', eta1['a'], gamma1['a'], h['ab'][a,V,a,V], t['aa'][pa,pa,hc,ha], optimize='optimal')
	O['ab'][a,V,c,V] += scale * +1.00000000 * np.einsum('uv,wx,wAvB,yuix->yAiB', eta1['a'], gamma1['a'], h['ab'][a,V,a,V], t['aa'][pa,pa,hc,ha], optimize='optimal')
	O['ab'][a,V,c,V] += scale * +1.00000000 * np.einsum('uv,iAvB,wuji->wAjB', eta1['a'], h['ab'][c,V,a,V], t['aa'][pa,pa,hc,hc], optimize='optimal')
	O['ab'][a,V,c,V] += scale * +1.00000000 * np.einsum('uv,uAaB,waiv->wAiB', gamma1['a'], h['ab'][a,V,v,V], t['aa'][pa,pv,hc,ha], optimize='optimal')
	O['ab'][a,V,c,V] += scale * +1.00000000 * np.einsum('iAaB,uaji->uAjB', h['ab'][c,V,v,V], t['aa'][pa,pv,hc,hc], optimize='optimal')
	O['ab'][v,V,c,V] += scale * +1.00000000 * np.einsum('uv,wx,uAxB,waiv->aAiB', eta1['a'], gamma1['a'], h['ab'][a,V,a,V], t['aa'][pa,pv,hc,ha], optimize='optimal')
	O['ab'][v,V,c,V] += scale * -1.00000000 * np.einsum('uv,wx,wAvB,uaix->aAiB', eta1['a'], gamma1['a'], h['ab'][a,V,a,V], t['aa'][pa,pv,hc,ha], optimize='optimal')
	O['ab'][v,V,c,V] += scale * -1.00000000 * np.einsum('uv,iAvB,uaji->aAjB', eta1['a'], h['ab'][c,V,a,V], t['aa'][pa,pv,hc,hc], optimize='optimal')
	O['ab'][v,V,c,V] += scale * +1.00000000 * np.einsum('uv,uAaB,baiv->bAiB', gamma1['a'], h['ab'][a,V,v,V], t['aa'][pv,pv,hc,ha], optimize='optimal')
	O['ab'][v,V,c,V] += scale * +1.00000000 * np.einsum('iAaB,baji->bAjB', h['ab'][c,V,v,V], t['aa'][pv,pv,hc,hc], optimize='optimal')

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

	
	
	O['ab'][a,A,a,A] += scale * -1.00000000 * np.einsum('uv,wIvU,uVxI->wVxU', eta1['a'], h['ab'][a,C,a,A], t['ab'][pa,pA,ha,hC], optimize='optimal')
	O['ab'][a,A,a,A] += scale * +1.00000000 * np.einsum('uv,wUvA,uAxV->wUxV', eta1['a'], h['ab'][a,A,a,V], t['ab'][pa,pV,ha,hA], optimize='optimal')
	
	
	
	
	O['ab'][a,A,a,A] += scale * -1.00000000 * np.einsum('UV,iWuV,vUiX->vWuX', eta1['b'], h['ab'][c,A,a,A], t['ab'][pa,pA,hc,hA], optimize='optimal')
	O['ab'][a,A,a,A] += scale * +1.00000000 * np.einsum('UV,uWaV,aUvX->uWvX', eta1['b'], h['ab'][a,A,v,A], t['ab'][pv,pA,ha,hA], optimize='optimal')
	O['ab'][a,A,a,A] += scale * +1.00000000 * np.einsum('uv,uIwU,xVvI->xVwU', gamma1['a'], h['ab'][a,C,a,A], t['ab'][pa,pA,ha,hC], optimize='optimal')
	O['ab'][a,A,a,A] += scale * -1.00000000 * np.einsum('uv,uUwA,xAvV->xUwV', gamma1['a'], h['ab'][a,A,a,V], t['ab'][pa,pV,ha,hA], optimize='optimal')
	
	
	O['ab'][a,A,a,A] += scale * +1.00000000 * np.einsum('UV,iUuW,vXiV->vXuW', gamma1['b'], h['ab'][c,A,a,A], t['ab'][pa,pA,hc,hA], optimize='optimal')
	O['ab'][a,A,a,A] += scale * -1.00000000 * np.einsum('UV,uUaW,aXvV->uXvW', gamma1['b'], h['ab'][a,A,v,A], t['ab'][pv,pA,ha,hA], optimize='optimal')
	O['ab'][a,A,a,A] += scale * +1.00000000 * np.einsum('iIuU,vViI->vVuU', h['ab'][c,C,a,A], t['ab'][pa,pA,hc,hC], optimize='optimal')
	O['ab'][a,A,a,A] += scale * -1.00000000 * np.einsum('iUuA,vAiV->vUuV', h['ab'][c,A,a,V], t['ab'][pa,pV,hc,hA], optimize='optimal')
	O['ab'][a,A,a,A] += scale * -1.00000000 * np.einsum('uIaU,aVvI->uVvU', h['ab'][a,C,v,A], t['ab'][pv,pA,ha,hC], optimize='optimal')
	O['ab'][a,A,a,A] += scale * +1.00000000 * np.einsum('uUaA,aAvV->uUvV', h['ab'][a,A,v,V], t['ab'][pv,pV,ha,hA], optimize='optimal')
	
	O['ab'][c,A,a,A] += scale * -1.00000000 * np.einsum('uv,iIvU,uVwI->iVwU', eta1['a'], h['ab'][c,C,a,A], t['ab'][pa,pA,ha,hC], optimize='optimal')
	O['ab'][c,A,a,A] += scale * +1.00000000 * np.einsum('uv,iUvA,uAwV->iUwV', eta1['a'], h['ab'][c,A,a,V], t['ab'][pa,pV,ha,hA], optimize='optimal')
	
	
	O['ab'][c,A,a,A] += scale * +1.00000000 * np.einsum('UV,iWaV,aUuX->iWuX', eta1['b'], h['ab'][c,A,v,A], t['ab'][pv,pA,ha,hA], optimize='optimal')
	
	O['ab'][c,A,a,A] += scale * -1.00000000 * np.einsum('UV,iUaW,aXuV->iXuW', gamma1['b'], h['ab'][c,A,v,A], t['ab'][pv,pA,ha,hA], optimize='optimal')
	O['ab'][c,A,a,A] += scale * -1.00000000 * np.einsum('iIaU,aVuI->iVuU', h['ab'][c,C,v,A], t['ab'][pv,pA,ha,hC], optimize='optimal')
	O['ab'][c,A,a,A] += scale * +1.00000000 * np.einsum('iUaA,aAuV->iUuV', h['ab'][c,A,v,V], t['ab'][pv,pV,ha,hA], optimize='optimal')
	O['ab'][v,A,a,A] += scale * +1.00000000 * np.einsum('uv,UV,uWwV,aUvX->aWwX', eta1['a'], gamma1['b'], h['ab'][a,A,a,A], t['ab'][pv,pA,ha,hA], optimize='optimal')
	
	O['ab'][v,A,a,A] += scale * -1.00000000 * np.einsum('uv,aIvU,uVwI->aVwU', eta1['a'], h['ab'][v,C,a,A], t['ab'][pa,pA,ha,hC], optimize='optimal')
	O['ab'][v,A,a,A] += scale * +1.00000000 * np.einsum('uv,aUvA,uAwV->aUwV', eta1['a'], h['ab'][v,A,a,V], t['ab'][pa,pV,ha,hA], optimize='optimal')
	O['ab'][v,A,a,A] += scale * -1.00000000 * np.einsum('UV,uv,uUwW,aXvV->aXwW', eta1['b'], eta1['a'], h['ab'][a,A,a,A], t['ab'][pv,pA,ha,hA], optimize='optimal')
	
	O['ab'][v,A,a,A] += scale * -1.00000000 * np.einsum('UV,uv,uWwV,aUvX->aWwX', eta1['b'], gamma1['a'], h['ab'][a,A,a,A], t['ab'][pv,pA,ha,hA], optimize='optimal')
	
	O['ab'][v,A,a,A] += scale * -1.00000000 * np.einsum('UV,iWuV,aUiX->aWuX', eta1['b'], h['ab'][c,A,a,A], t['ab'][pv,pA,hc,hA], optimize='optimal')
	O['ab'][v,A,a,A] += scale * +1.00000000 * np.einsum('UV,aWbV,bUuX->aWuX', eta1['b'], h['ab'][v,A,v,A], t['ab'][pv,pA,ha,hA], optimize='optimal')
	O['ab'][v,A,a,A] += scale * +1.00000000 * np.einsum('uv,uIwU,aVvI->aVwU', gamma1['a'], h['ab'][a,C,a,A], t['ab'][pv,pA,ha,hC], optimize='optimal')
	O['ab'][v,A,a,A] += scale * -1.00000000 * np.einsum('uv,uUwA,aAvV->aUwV', gamma1['a'], h['ab'][a,A,a,V], t['ab'][pv,pV,ha,hA], optimize='optimal')
	O['ab'][v,A,a,A] += scale * +1.00000000 * np.einsum('UV,uv,uUwW,aXvV->aXwW', gamma1['b'], gamma1['a'], h['ab'][a,A,a,A], t['ab'][pv,pA,ha,hA], optimize='optimal')
	
	O['ab'][v,A,a,A] += scale * +1.00000000 * np.einsum('UV,iUuW,aXiV->aXuW', gamma1['b'], h['ab'][c,A,a,A], t['ab'][pv,pA,hc,hA], optimize='optimal')
	O['ab'][v,A,a,A] += scale * -1.00000000 * np.einsum('UV,aUbW,bXuV->aXuW', gamma1['b'], h['ab'][v,A,v,A], t['ab'][pv,pA,ha,hA], optimize='optimal')
	O['ab'][v,A,a,A] += scale * +1.00000000 * np.einsum('iIuU,aViI->aVuU', h['ab'][c,C,a,A], t['ab'][pv,pA,hc,hC], optimize='optimal')
	O['ab'][v,A,a,A] += scale * -1.00000000 * np.einsum('iUuA,aAiV->aUuV', h['ab'][c,A,a,V], t['ab'][pv,pV,hc,hA], optimize='optimal')
	O['ab'][v,A,a,A] += scale * -1.00000000 * np.einsum('aIbU,bVuI->aVuU', h['ab'][v,C,v,A], t['ab'][pv,pA,ha,hC], optimize='optimal')
	O['ab'][v,A,a,A] += scale * +1.00000000 * np.einsum('aUbA,bAuV->aUuV', h['ab'][v,A,v,V], t['ab'][pv,pV,ha,hA], optimize='optimal')
	
	O['ab'][a,C,a,A] += scale * +1.00000000 * np.einsum('uv,wIvA,uAxU->wIxU', eta1['a'], h['ab'][a,C,a,V], t['ab'][pa,pV,ha,hA], optimize='optimal')
	
	
	O['ab'][a,C,a,A] += scale * -1.00000000 * np.einsum('UV,iIuV,vUiW->vIuW', eta1['b'], h['ab'][c,C,a,A], t['ab'][pa,pA,hc,hA], optimize='optimal')
	O['ab'][a,C,a,A] += scale * +1.00000000 * np.einsum('UV,uIaV,aUvW->uIvW', eta1['b'], h['ab'][a,C,v,A], t['ab'][pv,pA,ha,hA], optimize='optimal')
	O['ab'][a,C,a,A] += scale * -1.00000000 * np.einsum('uv,uIwA,xAvU->xIwU', gamma1['a'], h['ab'][a,C,a,V], t['ab'][pa,pV,ha,hA], optimize='optimal')
	
	O['ab'][a,C,a,A] += scale * -1.00000000 * np.einsum('iIuA,vAiU->vIuU', h['ab'][c,C,a,V], t['ab'][pa,pV,hc,hA], optimize='optimal')
	O['ab'][a,C,a,A] += scale * +1.00000000 * np.einsum('uIaA,aAvU->uIvU', h['ab'][a,C,v,V], t['ab'][pv,pV,ha,hA], optimize='optimal')
	O['ab'][c,C,a,A] += scale * +1.00000000 * np.einsum('uv,iIvA,uAwU->iIwU', eta1['a'], h['ab'][c,C,a,V], t['ab'][pa,pV,ha,hA], optimize='optimal')
	
	O['ab'][c,C,a,A] += scale * +1.00000000 * np.einsum('UV,iIaV,aUuW->iIuW', eta1['b'], h['ab'][c,C,v,A], t['ab'][pv,pA,ha,hA], optimize='optimal')
	
	O['ab'][c,C,a,A] += scale * +1.00000000 * np.einsum('iIaA,aAuU->iIuU', h['ab'][c,C,v,V], t['ab'][pv,pV,ha,hA], optimize='optimal')
	O['ab'][v,C,a,A] += scale * +1.00000000 * np.einsum('uv,UV,uIwV,aUvW->aIwW', eta1['a'], gamma1['b'], h['ab'][a,C,a,A], t['ab'][pv,pA,ha,hA], optimize='optimal')
	O['ab'][v,C,a,A] += scale * +1.00000000 * np.einsum('uv,aIvA,uAwU->aIwU', eta1['a'], h['ab'][v,C,a,V], t['ab'][pa,pV,ha,hA], optimize='optimal')
	
	O['ab'][v,C,a,A] += scale * -1.00000000 * np.einsum('UV,uv,uIwV,aUvW->aIwW', eta1['b'], gamma1['a'], h['ab'][a,C,a,A], t['ab'][pv,pA,ha,hA], optimize='optimal')
	O['ab'][v,C,a,A] += scale * -1.00000000 * np.einsum('UV,iIuV,aUiW->aIuW', eta1['b'], h['ab'][c,C,a,A], t['ab'][pv,pA,hc,hA], optimize='optimal')
	O['ab'][v,C,a,A] += scale * +1.00000000 * np.einsum('UV,aIbV,bUuW->aIuW', eta1['b'], h['ab'][v,C,v,A], t['ab'][pv,pA,ha,hA], optimize='optimal')
	O['ab'][v,C,a,A] += scale * -1.00000000 * np.einsum('uv,uIwA,aAvU->aIwU', gamma1['a'], h['ab'][a,C,a,V], t['ab'][pv,pV,ha,hA], optimize='optimal')
	
	O['ab'][v,C,a,A] += scale * -1.00000000 * np.einsum('iIuA,aAiU->aIuU', h['ab'][c,C,a,V], t['ab'][pv,pV,hc,hA], optimize='optimal')
	O['ab'][v,C,a,A] += scale * +1.00000000 * np.einsum('aIbA,bAuU->aIuU', h['ab'][v,C,v,V], t['ab'][pv,pV,ha,hA], optimize='optimal')
	O['ab'][a,V,a,A] += scale * -1.00000000 * np.einsum('uv,UV,wUvW,uAxV->wAxW', eta1['a'], gamma1['b'], h['ab'][a,A,a,A], t['ab'][pa,pV,ha,hA], optimize='optimal')
	
	O['ab'][a,V,a,A] += scale * -1.00000000 * np.einsum('uv,wIvU,uAxI->wAxU', eta1['a'], h['ab'][a,C,a,A], t['ab'][pa,pV,ha,hC], optimize='optimal')
	O['ab'][a,V,a,A] += scale * +1.00000000 * np.einsum('uv,wAvB,uBxU->wAxU', eta1['a'], h['ab'][a,V,a,V], t['ab'][pa,pV,ha,hA], optimize='optimal')
	
	O['ab'][a,V,a,A] += scale * -1.00000000 * np.einsum('UV,uv,uUwW,xAvV->xAwW', eta1['b'], eta1['a'], h['ab'][a,A,a,A], t['ab'][pa,pV,ha,hA], optimize='optimal')
	O['ab'][a,V,a,A] += scale * +1.00000000 * np.einsum('UV,uv,wUvW,uAxV->wAxW', eta1['b'], gamma1['a'], h['ab'][a,A,a,A], t['ab'][pa,pV,ha,hA], optimize='optimal')
	
	O['ab'][a,V,a,A] += scale * -1.00000000 * np.einsum('UV,iAuV,vUiW->vAuW', eta1['b'], h['ab'][c,V,a,A], t['ab'][pa,pA,hc,hA], optimize='optimal')
	O['ab'][a,V,a,A] += scale * +1.00000000 * np.einsum('UV,uAaV,aUvW->uAvW', eta1['b'], h['ab'][a,V,v,A], t['ab'][pv,pA,ha,hA], optimize='optimal')
	O['ab'][a,V,a,A] += scale * +1.00000000 * np.einsum('uv,uIwU,xAvI->xAwU', gamma1['a'], h['ab'][a,C,a,A], t['ab'][pa,pV,ha,hC], optimize='optimal')
	O['ab'][a,V,a,A] += scale * -1.00000000 * np.einsum('uv,uAwB,xBvU->xAwU', gamma1['a'], h['ab'][a,V,a,V], t['ab'][pa,pV,ha,hA], optimize='optimal')
	
	O['ab'][a,V,a,A] += scale * +1.00000000 * np.einsum('UV,uv,uUwW,xAvV->xAwW', gamma1['b'], gamma1['a'], h['ab'][a,A,a,A], t['ab'][pa,pV,ha,hA], optimize='optimal')
	O['ab'][a,V,a,A] += scale * +1.00000000 * np.einsum('UV,iUuW,vAiV->vAuW', gamma1['b'], h['ab'][c,A,a,A], t['ab'][pa,pV,hc,hA], optimize='optimal')
	O['ab'][a,V,a,A] += scale * -1.00000000 * np.einsum('UV,uUaW,aAvV->uAvW', gamma1['b'], h['ab'][a,A,v,A], t['ab'][pv,pV,ha,hA], optimize='optimal')
	O['ab'][a,V,a,A] += scale * +1.00000000 * np.einsum('iIuU,vAiI->vAuU', h['ab'][c,C,a,A], t['ab'][pa,pV,hc,hC], optimize='optimal')
	O['ab'][a,V,a,A] += scale * -1.00000000 * np.einsum('iAuB,vBiU->vAuU', h['ab'][c,V,a,V], t['ab'][pa,pV,hc,hA], optimize='optimal')
	O['ab'][a,V,a,A] += scale * -1.00000000 * np.einsum('uIaU,aAvI->uAvU', h['ab'][a,C,v,A], t['ab'][pv,pV,ha,hC], optimize='optimal')
	O['ab'][a,V,a,A] += scale * +1.00000000 * np.einsum('uAaB,aBvU->uAvU', h['ab'][a,V,v,V], t['ab'][pv,pV,ha,hA], optimize='optimal')
	O['ab'][c,V,a,A] += scale * -1.00000000 * np.einsum('uv,UV,iUvW,uAwV->iAwW', eta1['a'], gamma1['b'], h['ab'][c,A,a,A], t['ab'][pa,pV,ha,hA], optimize='optimal')
	O['ab'][c,V,a,A] += scale * -1.00000000 * np.einsum('uv,iIvU,uAwI->iAwU', eta1['a'], h['ab'][c,C,a,A], t['ab'][pa,pV,ha,hC], optimize='optimal')
	O['ab'][c,V,a,A] += scale * +1.00000000 * np.einsum('uv,iAvB,uBwU->iAwU', eta1['a'], h['ab'][c,V,a,V], t['ab'][pa,pV,ha,hA], optimize='optimal')
	
	O['ab'][c,V,a,A] += scale * +1.00000000 * np.einsum('UV,uv,iUvW,uAwV->iAwW', eta1['b'], gamma1['a'], h['ab'][c,A,a,A], t['ab'][pa,pV,ha,hA], optimize='optimal')
	O['ab'][c,V,a,A] += scale * +1.00000000 * np.einsum('UV,iAaV,aUuW->iAuW', eta1['b'], h['ab'][c,V,v,A], t['ab'][pv,pA,ha,hA], optimize='optimal')
	
	O['ab'][c,V,a,A] += scale * -1.00000000 * np.einsum('UV,iUaW,aAuV->iAuW', gamma1['b'], h['ab'][c,A,v,A], t['ab'][pv,pV,ha,hA], optimize='optimal')
	O['ab'][c,V,a,A] += scale * -1.00000000 * np.einsum('iIaU,aAuI->iAuU', h['ab'][c,C,v,A], t['ab'][pv,pV,ha,hC], optimize='optimal')
	O['ab'][c,V,a,A] += scale * +1.00000000 * np.einsum('iAaB,aBuU->iAuU', h['ab'][c,V,v,V], t['ab'][pv,pV,ha,hA], optimize='optimal')
	O['ab'][v,V,a,A] += scale * +1.00000000 * np.einsum('uv,UV,uAwV,aUvW->aAwW', eta1['a'], gamma1['b'], h['ab'][a,V,a,A], t['ab'][pv,pA,ha,hA], optimize='optimal')
	O['ab'][v,V,a,A] += scale * -1.00000000 * np.einsum('uv,UV,aUvW,uAwV->aAwW', eta1['a'], gamma1['b'], h['ab'][v,A,a,A], t['ab'][pa,pV,ha,hA], optimize='optimal')
	O['ab'][v,V,a,A] += scale * -1.00000000 * np.einsum('uv,aIvU,uAwI->aAwU', eta1['a'], h['ab'][v,C,a,A], t['ab'][pa,pV,ha,hC], optimize='optimal')
	O['ab'][v,V,a,A] += scale * +1.00000000 * np.einsum('uv,aAvB,uBwU->aAwU', eta1['a'], h['ab'][v,V,a,V], t['ab'][pa,pV,ha,hA], optimize='optimal')
	O['ab'][v,V,a,A] += scale * -1.00000000 * np.einsum('UV,uv,uUwW,aAvV->aAwW', eta1['b'], eta1['a'], h['ab'][a,A,a,A], t['ab'][pv,pV,ha,hA], optimize='optimal')
	
	O['ab'][v,V,a,A] += scale * -1.00000000 * np.einsum('UV,uv,uAwV,aUvW->aAwW', eta1['b'], gamma1['a'], h['ab'][a,V,a,A], t['ab'][pv,pA,ha,hA], optimize='optimal')
	O['ab'][v,V,a,A] += scale * +1.00000000 * np.einsum('UV,uv,aUvW,uAwV->aAwW', eta1['b'], gamma1['a'], h['ab'][v,A,a,A], t['ab'][pa,pV,ha,hA], optimize='optimal')
	O['ab'][v,V,a,A] += scale * -1.00000000 * np.einsum('UV,iAuV,aUiW->aAuW', eta1['b'], h['ab'][c,V,a,A], t['ab'][pv,pA,hc,hA], optimize='optimal')
	O['ab'][v,V,a,A] += scale * +1.00000000 * np.einsum('UV,aAbV,bUuW->aAuW', eta1['b'], h['ab'][v,V,v,A], t['ab'][pv,pA,ha,hA], optimize='optimal')
	O['ab'][v,V,a,A] += scale * +1.00000000 * np.einsum('uv,uIwU,aAvI->aAwU', gamma1['a'], h['ab'][a,C,a,A], t['ab'][pv,pV,ha,hC], optimize='optimal')
	O['ab'][v,V,a,A] += scale * -1.00000000 * np.einsum('uv,uAwB,aBvU->aAwU', gamma1['a'], h['ab'][a,V,a,V], t['ab'][pv,pV,ha,hA], optimize='optimal')
	O['ab'][v,V,a,A] += scale * +1.00000000 * np.einsum('UV,uv,uUwW,aAvV->aAwW', gamma1['b'], gamma1['a'], h['ab'][a,A,a,A], t['ab'][pv,pV,ha,hA], optimize='optimal')
	
	O['ab'][v,V,a,A] += scale * +1.00000000 * np.einsum('UV,iUuW,aAiV->aAuW', gamma1['b'], h['ab'][c,A,a,A], t['ab'][pv,pV,hc,hA], optimize='optimal')
	O['ab'][v,V,a,A] += scale * -1.00000000 * np.einsum('UV,aUbW,bAuV->aAuW', gamma1['b'], h['ab'][v,A,v,A], t['ab'][pv,pV,ha,hA], optimize='optimal')
	O['ab'][v,V,a,A] += scale * +1.00000000 * np.einsum('iIuU,aAiI->aAuU', h['ab'][c,C,a,A], t['ab'][pv,pV,hc,hC], optimize='optimal')
	O['ab'][v,V,a,A] += scale * -1.00000000 * np.einsum('iAuB,aBiU->aAuU', h['ab'][c,V,a,V], t['ab'][pv,pV,hc,hA], optimize='optimal')
	O['ab'][v,V,a,A] += scale * -1.00000000 * np.einsum('aIbU,bAuI->aAuU', h['ab'][v,C,v,A], t['ab'][pv,pV,ha,hC], optimize='optimal')
	O['ab'][v,V,a,A] += scale * +1.00000000 * np.einsum('aAbB,bBuU->aAuU', h['ab'][v,V,v,V], t['ab'][pv,pV,ha,hA], optimize='optimal')
	
	O['ab'][a,A,a,C] += scale * +1.00000000 * np.einsum('uv,UV,uWwV,xUvI->xWwI', eta1['a'], gamma1['b'], h['ab'][a,A,a,A], t['ab'][pa,pA,ha,hC], optimize='optimal')
	O['ab'][a,A,a,C] += scale * -1.00000000 * np.einsum('uv,wIvJ,uUxI->wUxJ', eta1['a'], h['ab'][a,C,a,C], t['ab'][pa,pA,ha,hC], optimize='optimal')
	O['ab'][a,A,a,C] += scale * +1.00000000 * np.einsum('uv,wUvA,uAxI->wUxI', eta1['a'], h['ab'][a,A,a,V], t['ab'][pa,pV,ha,hC], optimize='optimal')
	O['ab'][a,A,a,C] += scale * +1.00000000 * np.einsum('UV,uv,wWvV,uUxI->wWxI', eta1['b'], eta1['a'], h['ab'][a,A,a,A], t['ab'][pa,pA,ha,hC], optimize='optimal')
	
	
	O['ab'][a,A,a,C] += scale * -1.00000000 * np.einsum('UV,uv,uWwV,xUvI->xWwI', eta1['b'], gamma1['a'], h['ab'][a,A,a,A], t['ab'][pa,pA,ha,hC], optimize='optimal')
	O['ab'][a,A,a,C] += scale * -1.00000000 * np.einsum('UV,iWuV,vUiI->vWuI', eta1['b'], h['ab'][c,A,a,A], t['ab'][pa,pA,hc,hC], optimize='optimal')
	O['ab'][a,A,a,C] += scale * +1.00000000 * np.einsum('UV,uWaV,aUvI->uWvI', eta1['b'], h['ab'][a,A,v,A], t['ab'][pv,pA,ha,hC], optimize='optimal')
	O['ab'][a,A,a,C] += scale * +1.00000000 * np.einsum('uv,uIwJ,xUvI->xUwJ', gamma1['a'], h['ab'][a,C,a,C], t['ab'][pa,pA,ha,hC], optimize='optimal')
	O['ab'][a,A,a,C] += scale * -1.00000000 * np.einsum('uv,uUwA,xAvI->xUwI', gamma1['a'], h['ab'][a,A,a,V], t['ab'][pa,pV,ha,hC], optimize='optimal')
	O['ab'][a,A,a,C] += scale * -1.00000000 * np.einsum('UV,uv,wWvV,uUxI->wWxI', gamma1['b'], gamma1['a'], h['ab'][a,A,a,A], t['ab'][pa,pA,ha,hC], optimize='optimal')
	
	O['ab'][a,A,a,C] += scale * +1.00000000 * np.einsum('UV,iUuI,vWiV->vWuI', gamma1['b'], h['ab'][c,A,a,C], t['ab'][pa,pA,hc,hA], optimize='optimal')
	O['ab'][a,A,a,C] += scale * -1.00000000 * np.einsum('UV,uUaI,aWvV->uWvI', gamma1['b'], h['ab'][a,A,v,C], t['ab'][pv,pA,ha,hA], optimize='optimal')
	O['ab'][a,A,a,C] += scale * +1.00000000 * np.einsum('iIuJ,vUiI->vUuJ', h['ab'][c,C,a,C], t['ab'][pa,pA,hc,hC], optimize='optimal')
	O['ab'][a,A,a,C] += scale * -1.00000000 * np.einsum('iUuA,vAiI->vUuI', h['ab'][c,A,a,V], t['ab'][pa,pV,hc,hC], optimize='optimal')
	O['ab'][a,A,a,C] += scale * -1.00000000 * np.einsum('uIaJ,aUvI->uUvJ', h['ab'][a,C,v,C], t['ab'][pv,pA,ha,hC], optimize='optimal')
	O['ab'][a,A,a,C] += scale * +1.00000000 * np.einsum('uUaA,aAvI->uUvI', h['ab'][a,A,v,V], t['ab'][pv,pV,ha,hC], optimize='optimal')
	
	O['ab'][c,A,a,C] += scale * -1.00000000 * np.einsum('uv,iIvJ,uUwI->iUwJ', eta1['a'], h['ab'][c,C,a,C], t['ab'][pa,pA,ha,hC], optimize='optimal')
	O['ab'][c,A,a,C] += scale * +1.00000000 * np.einsum('uv,iUvA,uAwI->iUwI', eta1['a'], h['ab'][c,A,a,V], t['ab'][pa,pV,ha,hC], optimize='optimal')
	O['ab'][c,A,a,C] += scale * +1.00000000 * np.einsum('UV,uv,iWvV,uUwI->iWwI', eta1['b'], eta1['a'], h['ab'][c,A,a,A], t['ab'][pa,pA,ha,hC], optimize='optimal')
	
	O['ab'][c,A,a,C] += scale * +1.00000000 * np.einsum('UV,iWaV,aUuI->iWuI', eta1['b'], h['ab'][c,A,v,A], t['ab'][pv,pA,ha,hC], optimize='optimal')
	O['ab'][c,A,a,C] += scale * -1.00000000 * np.einsum('UV,uv,iWvV,uUwI->iWwI', gamma1['b'], gamma1['a'], h['ab'][c,A,a,A], t['ab'][pa,pA,ha,hC], optimize='optimal')
	O['ab'][c,A,a,C] += scale * -1.00000000 * np.einsum('UV,iUaI,aWuV->iWuI', gamma1['b'], h['ab'][c,A,v,C], t['ab'][pv,pA,ha,hA], optimize='optimal')
	O['ab'][c,A,a,C] += scale * -1.00000000 * np.einsum('iIaJ,aUuI->iUuJ', h['ab'][c,C,v,C], t['ab'][pv,pA,ha,hC], optimize='optimal')
	O['ab'][c,A,a,C] += scale * +1.00000000 * np.einsum('iUaA,aAuI->iUuI', h['ab'][c,A,v,V], t['ab'][pv,pV,ha,hC], optimize='optimal')
	O['ab'][v,A,a,C] += scale * +1.00000000 * np.einsum('uv,UV,uWwV,aUvI->aWwI', eta1['a'], gamma1['b'], h['ab'][a,A,a,A], t['ab'][pv,pA,ha,hC], optimize='optimal')
	
	O['ab'][v,A,a,C] += scale * -1.00000000 * np.einsum('uv,aIvJ,uUwI->aUwJ', eta1['a'], h['ab'][v,C,a,C], t['ab'][pa,pA,ha,hC], optimize='optimal')
	O['ab'][v,A,a,C] += scale * +1.00000000 * np.einsum('uv,aUvA,uAwI->aUwI', eta1['a'], h['ab'][v,A,a,V], t['ab'][pa,pV,ha,hC], optimize='optimal')
	O['ab'][v,A,a,C] += scale * -1.00000000 * np.einsum('UV,uv,uUwI,aWvV->aWwI', eta1['b'], eta1['a'], h['ab'][a,A,a,C], t['ab'][pv,pA,ha,hA], optimize='optimal')
	O['ab'][v,A,a,C] += scale * +1.00000000 * np.einsum('UV,uv,aWvV,uUwI->aWwI', eta1['b'], eta1['a'], h['ab'][v,A,a,A], t['ab'][pa,pA,ha,hC], optimize='optimal')
	O['ab'][v,A,a,C] += scale * -1.00000000 * np.einsum('UV,uv,uWwV,aUvI->aWwI', eta1['b'], gamma1['a'], h['ab'][a,A,a,A], t['ab'][pv,pA,ha,hC], optimize='optimal')
	
	O['ab'][v,A,a,C] += scale * -1.00000000 * np.einsum('UV,iWuV,aUiI->aWuI', eta1['b'], h['ab'][c,A,a,A], t['ab'][pv,pA,hc,hC], optimize='optimal')
	O['ab'][v,A,a,C] += scale * +1.00000000 * np.einsum('UV,aWbV,bUuI->aWuI', eta1['b'], h['ab'][v,A,v,A], t['ab'][pv,pA,ha,hC], optimize='optimal')
	O['ab'][v,A,a,C] += scale * +1.00000000 * np.einsum('uv,uIwJ,aUvI->aUwJ', gamma1['a'], h['ab'][a,C,a,C], t['ab'][pv,pA,ha,hC], optimize='optimal')
	O['ab'][v,A,a,C] += scale * -1.00000000 * np.einsum('uv,uUwA,aAvI->aUwI', gamma1['a'], h['ab'][a,A,a,V], t['ab'][pv,pV,ha,hC], optimize='optimal')
	O['ab'][v,A,a,C] += scale * +1.00000000 * np.einsum('UV,uv,uUwI,aWvV->aWwI', gamma1['b'], gamma1['a'], h['ab'][a,A,a,C], t['ab'][pv,pA,ha,hA], optimize='optimal')
	O['ab'][v,A,a,C] += scale * -1.00000000 * np.einsum('UV,uv,aWvV,uUwI->aWwI', gamma1['b'], gamma1['a'], h['ab'][v,A,a,A], t['ab'][pa,pA,ha,hC], optimize='optimal')
	O['ab'][v,A,a,C] += scale * +1.00000000 * np.einsum('UV,iUuI,aWiV->aWuI', gamma1['b'], h['ab'][c,A,a,C], t['ab'][pv,pA,hc,hA], optimize='optimal')
	O['ab'][v,A,a,C] += scale * -1.00000000 * np.einsum('UV,aUbI,bWuV->aWuI', gamma1['b'], h['ab'][v,A,v,C], t['ab'][pv,pA,ha,hA], optimize='optimal')
	O['ab'][v,A,a,C] += scale * +1.00000000 * np.einsum('iIuJ,aUiI->aUuJ', h['ab'][c,C,a,C], t['ab'][pv,pA,hc,hC], optimize='optimal')
	O['ab'][v,A,a,C] += scale * -1.00000000 * np.einsum('iUuA,aAiI->aUuI', h['ab'][c,A,a,V], t['ab'][pv,pV,hc,hC], optimize='optimal')
	O['ab'][v,A,a,C] += scale * -1.00000000 * np.einsum('aIbJ,bUuI->aUuJ', h['ab'][v,C,v,C], t['ab'][pv,pA,ha,hC], optimize='optimal')
	O['ab'][v,A,a,C] += scale * +1.00000000 * np.einsum('aUbA,bAuI->aUuI', h['ab'][v,A,v,V], t['ab'][pv,pV,ha,hC], optimize='optimal')
	O['ab'][a,C,a,C] += scale * +1.00000000 * np.einsum('uv,UV,uIwV,xUvJ->xIwJ', eta1['a'], gamma1['b'], h['ab'][a,C,a,A], t['ab'][pa,pA,ha,hC], optimize='optimal')
	O['ab'][a,C,a,C] += scale * +1.00000000 * np.einsum('uv,wIvA,uAxJ->wIxJ', eta1['a'], h['ab'][a,C,a,V], t['ab'][pa,pV,ha,hC], optimize='optimal')
	O['ab'][a,C,a,C] += scale * +1.00000000 * np.einsum('UV,uv,wIvV,uUxJ->wIxJ', eta1['b'], eta1['a'], h['ab'][a,C,a,A], t['ab'][pa,pA,ha,hC], optimize='optimal')
	O['ab'][a,C,a,C] += scale * -1.00000000 * np.einsum('UV,uv,uIwV,xUvJ->xIwJ', eta1['b'], gamma1['a'], h['ab'][a,C,a,A], t['ab'][pa,pA,ha,hC], optimize='optimal')
	O['ab'][a,C,a,C] += scale * -1.00000000 * np.einsum('UV,iIuV,vUiJ->vIuJ', eta1['b'], h['ab'][c,C,a,A], t['ab'][pa,pA,hc,hC], optimize='optimal')
	O['ab'][a,C,a,C] += scale * +1.00000000 * np.einsum('UV,uIaV,aUvJ->uIvJ', eta1['b'], h['ab'][a,C,v,A], t['ab'][pv,pA,ha,hC], optimize='optimal')
	O['ab'][a,C,a,C] += scale * -1.00000000 * np.einsum('uv,uIwA,xAvJ->xIwJ', gamma1['a'], h['ab'][a,C,a,V], t['ab'][pa,pV,ha,hC], optimize='optimal')
	O['ab'][a,C,a,C] += scale * -1.00000000 * np.einsum('UV,uv,wIvV,uUxJ->wIxJ', gamma1['b'], gamma1['a'], h['ab'][a,C,a,A], t['ab'][pa,pA,ha,hC], optimize='optimal')
	O['ab'][a,C,a,C] += scale * -1.00000000 * np.einsum('iIuA,vAiJ->vIuJ', h['ab'][c,C,a,V], t['ab'][pa,pV,hc,hC], optimize='optimal')
	O['ab'][a,C,a,C] += scale * +1.00000000 * np.einsum('uIaA,aAvJ->uIvJ', h['ab'][a,C,v,V], t['ab'][pv,pV,ha,hC], optimize='optimal')
	O['ab'][c,C,a,C] += scale * +1.00000000 * np.einsum('uv,iIvA,uAwJ->iIwJ', eta1['a'], h['ab'][c,C,a,V], t['ab'][pa,pV,ha,hC], optimize='optimal')
	O['ab'][c,C,a,C] += scale * +1.00000000 * np.einsum('UV,uv,iIvV,uUwJ->iIwJ', eta1['b'], eta1['a'], h['ab'][c,C,a,A], t['ab'][pa,pA,ha,hC], optimize='optimal')
	O['ab'][c,C,a,C] += scale * +1.00000000 * np.einsum('UV,iIaV,aUuJ->iIuJ', eta1['b'], h['ab'][c,C,v,A], t['ab'][pv,pA,ha,hC], optimize='optimal')
	O['ab'][c,C,a,C] += scale * -1.00000000 * np.einsum('UV,uv,iIvV,uUwJ->iIwJ', gamma1['b'], gamma1['a'], h['ab'][c,C,a,A], t['ab'][pa,pA,ha,hC], optimize='optimal')
	O['ab'][c,C,a,C] += scale * +1.00000000 * np.einsum('iIaA,aAuJ->iIuJ', h['ab'][c,C,v,V], t['ab'][pv,pV,ha,hC], optimize='optimal')
	O['ab'][v,C,a,C] += scale * +1.00000000 * np.einsum('uv,UV,uIwV,aUvJ->aIwJ', eta1['a'], gamma1['b'], h['ab'][a,C,a,A], t['ab'][pv,pA,ha,hC], optimize='optimal')
	O['ab'][v,C,a,C] += scale * +1.00000000 * np.einsum('uv,aIvA,uAwJ->aIwJ', eta1['a'], h['ab'][v,C,a,V], t['ab'][pa,pV,ha,hC], optimize='optimal')
	O['ab'][v,C,a,C] += scale * +1.00000000 * np.einsum('UV,uv,aIvV,uUwJ->aIwJ', eta1['b'], eta1['a'], h['ab'][v,C,a,A], t['ab'][pa,pA,ha,hC], optimize='optimal')
	O['ab'][v,C,a,C] += scale * -1.00000000 * np.einsum('UV,uv,uIwV,aUvJ->aIwJ', eta1['b'], gamma1['a'], h['ab'][a,C,a,A], t['ab'][pv,pA,ha,hC], optimize='optimal')
	O['ab'][v,C,a,C] += scale * -1.00000000 * np.einsum('UV,iIuV,aUiJ->aIuJ', eta1['b'], h['ab'][c,C,a,A], t['ab'][pv,pA,hc,hC], optimize='optimal')
	O['ab'][v,C,a,C] += scale * +1.00000000 * np.einsum('UV,aIbV,bUuJ->aIuJ', eta1['b'], h['ab'][v,C,v,A], t['ab'][pv,pA,ha,hC], optimize='optimal')
	O['ab'][v,C,a,C] += scale * -1.00000000 * np.einsum('uv,uIwA,aAvJ->aIwJ', gamma1['a'], h['ab'][a,C,a,V], t['ab'][pv,pV,ha,hC], optimize='optimal')
	O['ab'][v,C,a,C] += scale * -1.00000000 * np.einsum('UV,uv,aIvV,uUwJ->aIwJ', gamma1['b'], gamma1['a'], h['ab'][v,C,a,A], t['ab'][pa,pA,ha,hC], optimize='optimal')
	O['ab'][v,C,a,C] += scale * -1.00000000 * np.einsum('iIuA,aAiJ->aIuJ', h['ab'][c,C,a,V], t['ab'][pv,pV,hc,hC], optimize='optimal')
	O['ab'][v,C,a,C] += scale * +1.00000000 * np.einsum('aIbA,bAuJ->aIuJ', h['ab'][v,C,v,V], t['ab'][pv,pV,ha,hC], optimize='optimal')
	O['ab'][a,V,a,C] += scale * -1.00000000 * np.einsum('uv,UV,wUvI,uAxV->wAxI', eta1['a'], gamma1['b'], h['ab'][a,A,a,C], t['ab'][pa,pV,ha,hA], optimize='optimal')
	O['ab'][a,V,a,C] += scale * +1.00000000 * np.einsum('uv,UV,uAwV,xUvI->xAwI', eta1['a'], gamma1['b'], h['ab'][a,V,a,A], t['ab'][pa,pA,ha,hC], optimize='optimal')
	O['ab'][a,V,a,C] += scale * -1.00000000 * np.einsum('uv,wIvJ,uAxI->wAxJ', eta1['a'], h['ab'][a,C,a,C], t['ab'][pa,pV,ha,hC], optimize='optimal')
	O['ab'][a,V,a,C] += scale * +1.00000000 * np.einsum('uv,wAvB,uBxI->wAxI', eta1['a'], h['ab'][a,V,a,V], t['ab'][pa,pV,ha,hC], optimize='optimal')
	O['ab'][a,V,a,C] += scale * +1.00000000 * np.einsum('UV,uv,wAvV,uUxI->wAxI', eta1['b'], eta1['a'], h['ab'][a,V,a,A], t['ab'][pa,pA,ha,hC], optimize='optimal')
	O['ab'][a,V,a,C] += scale * -1.00000000 * np.einsum('UV,uv,uUwI,xAvV->xAwI', eta1['b'], eta1['a'], h['ab'][a,A,a,C], t['ab'][pa,pV,ha,hA], optimize='optimal')
	O['ab'][a,V,a,C] += scale * +1.00000000 * np.einsum('UV,uv,wUvI,uAxV->wAxI', eta1['b'], gamma1['a'], h['ab'][a,A,a,C], t['ab'][pa,pV,ha,hA], optimize='optimal')
	O['ab'][a,V,a,C] += scale * -1.00000000 * np.einsum('UV,uv,uAwV,xUvI->xAwI', eta1['b'], gamma1['a'], h['ab'][a,V,a,A], t['ab'][pa,pA,ha,hC], optimize='optimal')
	O['ab'][a,V,a,C] += scale * -1.00000000 * np.einsum('UV,iAuV,vUiI->vAuI', eta1['b'], h['ab'][c,V,a,A], t['ab'][pa,pA,hc,hC], optimize='optimal')
	O['ab'][a,V,a,C] += scale * +1.00000000 * np.einsum('UV,uAaV,aUvI->uAvI', eta1['b'], h['ab'][a,V,v,A], t['ab'][pv,pA,ha,hC], optimize='optimal')
	O['ab'][a,V,a,C] += scale * +1.00000000 * np.einsum('uv,uIwJ,xAvI->xAwJ', gamma1['a'], h['ab'][a,C,a,C], t['ab'][pa,pV,ha,hC], optimize='optimal')
	O['ab'][a,V,a,C] += scale * -1.00000000 * np.einsum('uv,uAwB,xBvI->xAwI', gamma1['a'], h['ab'][a,V,a,V], t['ab'][pa,pV,ha,hC], optimize='optimal')
	O['ab'][a,V,a,C] += scale * -1.00000000 * np.einsum('UV,uv,wAvV,uUxI->wAxI', gamma1['b'], gamma1['a'], h['ab'][a,V,a,A], t['ab'][pa,pA,ha,hC], optimize='optimal')
	O['ab'][a,V,a,C] += scale * +1.00000000 * np.einsum('UV,uv,uUwI,xAvV->xAwI', gamma1['b'], gamma1['a'], h['ab'][a,A,a,C], t['ab'][pa,pV,ha,hA], optimize='optimal')
	O['ab'][a,V,a,C] += scale * +1.00000000 * np.einsum('UV,iUuI,vAiV->vAuI', gamma1['b'], h['ab'][c,A,a,C], t['ab'][pa,pV,hc,hA], optimize='optimal')
	O['ab'][a,V,a,C] += scale * -1.00000000 * np.einsum('UV,uUaI,aAvV->uAvI', gamma1['b'], h['ab'][a,A,v,C], t['ab'][pv,pV,ha,hA], optimize='optimal')
	O['ab'][a,V,a,C] += scale * +1.00000000 * np.einsum('iIuJ,vAiI->vAuJ', h['ab'][c,C,a,C], t['ab'][pa,pV,hc,hC], optimize='optimal')
	O['ab'][a,V,a,C] += scale * -1.00000000 * np.einsum('iAuB,vBiI->vAuI', h['ab'][c,V,a,V], t['ab'][pa,pV,hc,hC], optimize='optimal')
	O['ab'][a,V,a,C] += scale * -1.00000000 * np.einsum('uIaJ,aAvI->uAvJ', h['ab'][a,C,v,C], t['ab'][pv,pV,ha,hC], optimize='optimal')
	O['ab'][a,V,a,C] += scale * +1.00000000 * np.einsum('uAaB,aBvI->uAvI', h['ab'][a,V,v,V], t['ab'][pv,pV,ha,hC], optimize='optimal')
	O['ab'][c,V,a,C] += scale * -1.00000000 * np.einsum('uv,UV,iUvI,uAwV->iAwI', eta1['a'], gamma1['b'], h['ab'][c,A,a,C], t['ab'][pa,pV,ha,hA], optimize='optimal')
	O['ab'][c,V,a,C] += scale * -1.00000000 * np.einsum('uv,iIvJ,uAwI->iAwJ', eta1['a'], h['ab'][c,C,a,C], t['ab'][pa,pV,ha,hC], optimize='optimal')
	O['ab'][c,V,a,C] += scale * +1.00000000 * np.einsum('uv,iAvB,uBwI->iAwI', eta1['a'], h['ab'][c,V,a,V], t['ab'][pa,pV,ha,hC], optimize='optimal')
	O['ab'][c,V,a,C] += scale * +1.00000000 * np.einsum('UV,uv,iAvV,uUwI->iAwI', eta1['b'], eta1['a'], h['ab'][c,V,a,A], t['ab'][pa,pA,ha,hC], optimize='optimal')
	O['ab'][c,V,a,C] += scale * +1.00000000 * np.einsum('UV,uv,iUvI,uAwV->iAwI', eta1['b'], gamma1['a'], h['ab'][c,A,a,C], t['ab'][pa,pV,ha,hA], optimize='optimal')
	O['ab'][c,V,a,C] += scale * +1.00000000 * np.einsum('UV,iAaV,aUuI->iAuI', eta1['b'], h['ab'][c,V,v,A], t['ab'][pv,pA,ha,hC], optimize='optimal')
	O['ab'][c,V,a,C] += scale * -1.00000000 * np.einsum('UV,uv,iAvV,uUwI->iAwI', gamma1['b'], gamma1['a'], h['ab'][c,V,a,A], t['ab'][pa,pA,ha,hC], optimize='optimal')
	O['ab'][c,V,a,C] += scale * -1.00000000 * np.einsum('UV,iUaI,aAuV->iAuI', gamma1['b'], h['ab'][c,A,v,C], t['ab'][pv,pV,ha,hA], optimize='optimal')
	O['ab'][c,V,a,C] += scale * -1.00000000 * np.einsum('iIaJ,aAuI->iAuJ', h['ab'][c,C,v,C], t['ab'][pv,pV,ha,hC], optimize='optimal')
	O['ab'][c,V,a,C] += scale * +1.00000000 * np.einsum('iAaB,aBuI->iAuI', h['ab'][c,V,v,V], t['ab'][pv,pV,ha,hC], optimize='optimal')
	O['ab'][v,V,a,C] += scale * +1.00000000 * np.einsum('uv,UV,uAwV,aUvI->aAwI', eta1['a'], gamma1['b'], h['ab'][a,V,a,A], t['ab'][pv,pA,ha,hC], optimize='optimal')
	O['ab'][v,V,a,C] += scale * -1.00000000 * np.einsum('uv,UV,aUvI,uAwV->aAwI', eta1['a'], gamma1['b'], h['ab'][v,A,a,C], t['ab'][pa,pV,ha,hA], optimize='optimal')
	O['ab'][v,V,a,C] += scale * -1.00000000 * np.einsum('uv,aIvJ,uAwI->aAwJ', eta1['a'], h['ab'][v,C,a,C], t['ab'][pa,pV,ha,hC], optimize='optimal')
	O['ab'][v,V,a,C] += scale * +1.00000000 * np.einsum('uv,aAvB,uBwI->aAwI', eta1['a'], h['ab'][v,V,a,V], t['ab'][pa,pV,ha,hC], optimize='optimal')
	O['ab'][v,V,a,C] += scale * -1.00000000 * np.einsum('UV,uv,uUwI,aAvV->aAwI', eta1['b'], eta1['a'], h['ab'][a,A,a,C], t['ab'][pv,pV,ha,hA], optimize='optimal')
	O['ab'][v,V,a,C] += scale * +1.00000000 * np.einsum('UV,uv,aAvV,uUwI->aAwI', eta1['b'], eta1['a'], h['ab'][v,V,a,A], t['ab'][pa,pA,ha,hC], optimize='optimal')
	O['ab'][v,V,a,C] += scale * -1.00000000 * np.einsum('UV,uv,uAwV,aUvI->aAwI', eta1['b'], gamma1['a'], h['ab'][a,V,a,A], t['ab'][pv,pA,ha,hC], optimize='optimal')
	O['ab'][v,V,a,C] += scale * +1.00000000 * np.einsum('UV,uv,aUvI,uAwV->aAwI', eta1['b'], gamma1['a'], h['ab'][v,A,a,C], t['ab'][pa,pV,ha,hA], optimize='optimal')
	O['ab'][v,V,a,C] += scale * -1.00000000 * np.einsum('UV,iAuV,aUiI->aAuI', eta1['b'], h['ab'][c,V,a,A], t['ab'][pv,pA,hc,hC], optimize='optimal')
	O['ab'][v,V,a,C] += scale * +1.00000000 * np.einsum('UV,aAbV,bUuI->aAuI', eta1['b'], h['ab'][v,V,v,A], t['ab'][pv,pA,ha,hC], optimize='optimal')
	O['ab'][v,V,a,C] += scale * +1.00000000 * np.einsum('uv,uIwJ,aAvI->aAwJ', gamma1['a'], h['ab'][a,C,a,C], t['ab'][pv,pV,ha,hC], optimize='optimal')
	O['ab'][v,V,a,C] += scale * -1.00000000 * np.einsum('uv,uAwB,aBvI->aAwI', gamma1['a'], h['ab'][a,V,a,V], t['ab'][pv,pV,ha,hC], optimize='optimal')
	O['ab'][v,V,a,C] += scale * +1.00000000 * np.einsum('UV,uv,uUwI,aAvV->aAwI', gamma1['b'], gamma1['a'], h['ab'][a,A,a,C], t['ab'][pv,pV,ha,hA], optimize='optimal')
	O['ab'][v,V,a,C] += scale * -1.00000000 * np.einsum('UV,uv,aAvV,uUwI->aAwI', gamma1['b'], gamma1['a'], h['ab'][v,V,a,A], t['ab'][pa,pA,ha,hC], optimize='optimal')
	O['ab'][v,V,a,C] += scale * +1.00000000 * np.einsum('UV,iUuI,aAiV->aAuI', gamma1['b'], h['ab'][c,A,a,C], t['ab'][pv,pV,hc,hA], optimize='optimal')
	O['ab'][v,V,a,C] += scale * -1.00000000 * np.einsum('UV,aUbI,bAuV->aAuI', gamma1['b'], h['ab'][v,A,v,C], t['ab'][pv,pV,ha,hA], optimize='optimal')
	O['ab'][v,V,a,C] += scale * +1.00000000 * np.einsum('iIuJ,aAiI->aAuJ', h['ab'][c,C,a,C], t['ab'][pv,pV,hc,hC], optimize='optimal')
	O['ab'][v,V,a,C] += scale * -1.00000000 * np.einsum('iAuB,aBiI->aAuI', h['ab'][c,V,a,V], t['ab'][pv,pV,hc,hC], optimize='optimal')
	O['ab'][v,V,a,C] += scale * -1.00000000 * np.einsum('aIbJ,bAuI->aAuJ', h['ab'][v,C,v,C], t['ab'][pv,pV,ha,hC], optimize='optimal')
	O['ab'][v,V,a,C] += scale * +1.00000000 * np.einsum('aAbB,bBuI->aAuI', h['ab'][v,V,v,V], t['ab'][pv,pV,ha,hC], optimize='optimal')
	
	O['ab'][a,A,a,V] += scale * -1.00000000 * np.einsum('uv,wIvA,uUxI->wUxA', eta1['a'], h['ab'][a,C,a,V], t['ab'][pa,pA,ha,hC], optimize='optimal')
	
	
	O['ab'][a,A,a,V] += scale * +1.00000000 * np.einsum('uv,uIwA,xUvI->xUwA', gamma1['a'], h['ab'][a,C,a,V], t['ab'][pa,pA,ha,hC], optimize='optimal')
	
	O['ab'][a,A,a,V] += scale * +1.00000000 * np.einsum('UV,iUuA,vWiV->vWuA', gamma1['b'], h['ab'][c,A,a,V], t['ab'][pa,pA,hc,hA], optimize='optimal')
	O['ab'][a,A,a,V] += scale * -1.00000000 * np.einsum('UV,uUaA,aWvV->uWvA', gamma1['b'], h['ab'][a,A,v,V], t['ab'][pv,pA,ha,hA], optimize='optimal')
	O['ab'][a,A,a,V] += scale * +1.00000000 * np.einsum('iIuA,vUiI->vUuA', h['ab'][c,C,a,V], t['ab'][pa,pA,hc,hC], optimize='optimal')
	O['ab'][a,A,a,V] += scale * -1.00000000 * np.einsum('uIaA,aUvI->uUvA', h['ab'][a,C,v,V], t['ab'][pv,pA,ha,hC], optimize='optimal')
	
	O['ab'][c,A,a,V] += scale * -1.00000000 * np.einsum('uv,iIvA,uUwI->iUwA', eta1['a'], h['ab'][c,C,a,V], t['ab'][pa,pA,ha,hC], optimize='optimal')
	
	O['ab'][c,A,a,V] += scale * -1.00000000 * np.einsum('UV,iUaA,aWuV->iWuA', gamma1['b'], h['ab'][c,A,v,V], t['ab'][pv,pA,ha,hA], optimize='optimal')
	O['ab'][c,A,a,V] += scale * -1.00000000 * np.einsum('iIaA,aUuI->iUuA', h['ab'][c,C,v,V], t['ab'][pv,pA,ha,hC], optimize='optimal')
	
	O['ab'][v,A,a,V] += scale * -1.00000000 * np.einsum('uv,aIvA,uUwI->aUwA', eta1['a'], h['ab'][v,C,a,V], t['ab'][pa,pA,ha,hC], optimize='optimal')
	O['ab'][v,A,a,V] += scale * -1.00000000 * np.einsum('UV,uv,uUwA,aWvV->aWwA', eta1['b'], eta1['a'], h['ab'][a,A,a,V], t['ab'][pv,pA,ha,hA], optimize='optimal')
	
	O['ab'][v,A,a,V] += scale * +1.00000000 * np.einsum('uv,uIwA,aUvI->aUwA', gamma1['a'], h['ab'][a,C,a,V], t['ab'][pv,pA,ha,hC], optimize='optimal')
	O['ab'][v,A,a,V] += scale * +1.00000000 * np.einsum('UV,uv,uUwA,aWvV->aWwA', gamma1['b'], gamma1['a'], h['ab'][a,A,a,V], t['ab'][pv,pA,ha,hA], optimize='optimal')
	O['ab'][v,A,a,V] += scale * +1.00000000 * np.einsum('UV,iUuA,aWiV->aWuA', gamma1['b'], h['ab'][c,A,a,V], t['ab'][pv,pA,hc,hA], optimize='optimal')
	O['ab'][v,A,a,V] += scale * -1.00000000 * np.einsum('UV,aUbA,bWuV->aWuA', gamma1['b'], h['ab'][v,A,v,V], t['ab'][pv,pA,ha,hA], optimize='optimal')
	O['ab'][v,A,a,V] += scale * +1.00000000 * np.einsum('iIuA,aUiI->aUuA', h['ab'][c,C,a,V], t['ab'][pv,pA,hc,hC], optimize='optimal')
	O['ab'][v,A,a,V] += scale * -1.00000000 * np.einsum('aIbA,bUuI->aUuA', h['ab'][v,C,v,V], t['ab'][pv,pA,ha,hC], optimize='optimal')
	O['ab'][a,V,a,V] += scale * -1.00000000 * np.einsum('uv,UV,wUvA,uBxV->wBxA', eta1['a'], gamma1['b'], h['ab'][a,A,a,V], t['ab'][pa,pV,ha,hA], optimize='optimal')
	O['ab'][a,V,a,V] += scale * -1.00000000 * np.einsum('uv,wIvA,uBxI->wBxA', eta1['a'], h['ab'][a,C,a,V], t['ab'][pa,pV,ha,hC], optimize='optimal')
	O['ab'][a,V,a,V] += scale * -1.00000000 * np.einsum('UV,uv,uUwA,xBvV->xBwA', eta1['b'], eta1['a'], h['ab'][a,A,a,V], t['ab'][pa,pV,ha,hA], optimize='optimal')
	O['ab'][a,V,a,V] += scale * +1.00000000 * np.einsum('UV,uv,wUvA,uBxV->wBxA', eta1['b'], gamma1['a'], h['ab'][a,A,a,V], t['ab'][pa,pV,ha,hA], optimize='optimal')
	O['ab'][a,V,a,V] += scale * +1.00000000 * np.einsum('uv,uIwA,xBvI->xBwA', gamma1['a'], h['ab'][a,C,a,V], t['ab'][pa,pV,ha,hC], optimize='optimal')
	O['ab'][a,V,a,V] += scale * +1.00000000 * np.einsum('UV,uv,uUwA,xBvV->xBwA', gamma1['b'], gamma1['a'], h['ab'][a,A,a,V], t['ab'][pa,pV,ha,hA], optimize='optimal')
	O['ab'][a,V,a,V] += scale * +1.00000000 * np.einsum('UV,iUuA,vBiV->vBuA', gamma1['b'], h['ab'][c,A,a,V], t['ab'][pa,pV,hc,hA], optimize='optimal')
	O['ab'][a,V,a,V] += scale * -1.00000000 * np.einsum('UV,uUaA,aBvV->uBvA', gamma1['b'], h['ab'][a,A,v,V], t['ab'][pv,pV,ha,hA], optimize='optimal')
	O['ab'][a,V,a,V] += scale * +1.00000000 * np.einsum('iIuA,vBiI->vBuA', h['ab'][c,C,a,V], t['ab'][pa,pV,hc,hC], optimize='optimal')
	O['ab'][a,V,a,V] += scale * -1.00000000 * np.einsum('uIaA,aBvI->uBvA', h['ab'][a,C,v,V], t['ab'][pv,pV,ha,hC], optimize='optimal')
	O['ab'][c,V,a,V] += scale * -1.00000000 * np.einsum('uv,UV,iUvA,uBwV->iBwA', eta1['a'], gamma1['b'], h['ab'][c,A,a,V], t['ab'][pa,pV,ha,hA], optimize='optimal')
	O['ab'][c,V,a,V] += scale * -1.00000000 * np.einsum('uv,iIvA,uBwI->iBwA', eta1['a'], h['ab'][c,C,a,V], t['ab'][pa,pV,ha,hC], optimize='optimal')
	O['ab'][c,V,a,V] += scale * +1.00000000 * np.einsum('UV,uv,iUvA,uBwV->iBwA', eta1['b'], gamma1['a'], h['ab'][c,A,a,V], t['ab'][pa,pV,ha,hA], optimize='optimal')
	O['ab'][c,V,a,V] += scale * -1.00000000 * np.einsum('UV,iUaA,aBuV->iBuA', gamma1['b'], h['ab'][c,A,v,V], t['ab'][pv,pV,ha,hA], optimize='optimal')
	O['ab'][c,V,a,V] += scale * -1.00000000 * np.einsum('iIaA,aBuI->iBuA', h['ab'][c,C,v,V], t['ab'][pv,pV,ha,hC], optimize='optimal')
	O['ab'][v,V,a,V] += scale * -1.00000000 * np.einsum('uv,UV,aUvA,uBwV->aBwA', eta1['a'], gamma1['b'], h['ab'][v,A,a,V], t['ab'][pa,pV,ha,hA], optimize='optimal')
	O['ab'][v,V,a,V] += scale * -1.00000000 * np.einsum('uv,aIvA,uBwI->aBwA', eta1['a'], h['ab'][v,C,a,V], t['ab'][pa,pV,ha,hC], optimize='optimal')
	O['ab'][v,V,a,V] += scale * -1.00000000 * np.einsum('UV,uv,uUwA,aBvV->aBwA', eta1['b'], eta1['a'], h['ab'][a,A,a,V], t['ab'][pv,pV,ha,hA], optimize='optimal')
	O['ab'][v,V,a,V] += scale * +1.00000000 * np.einsum('UV,uv,aUvA,uBwV->aBwA', eta1['b'], gamma1['a'], h['ab'][v,A,a,V], t['ab'][pa,pV,ha,hA], optimize='optimal')
	O['ab'][v,V,a,V] += scale * +1.00000000 * np.einsum('uv,uIwA,aBvI->aBwA', gamma1['a'], h['ab'][a,C,a,V], t['ab'][pv,pV,ha,hC], optimize='optimal')
	O['ab'][v,V,a,V] += scale * +1.00000000 * np.einsum('UV,uv,uUwA,aBvV->aBwA', gamma1['b'], gamma1['a'], h['ab'][a,A,a,V], t['ab'][pv,pV,ha,hA], optimize='optimal')
	O['ab'][v,V,a,V] += scale * +1.00000000 * np.einsum('UV,iUuA,aBiV->aBuA', gamma1['b'], h['ab'][c,A,a,V], t['ab'][pv,pV,hc,hA], optimize='optimal')
	O['ab'][v,V,a,V] += scale * -1.00000000 * np.einsum('UV,aUbA,bBuV->aBuA', gamma1['b'], h['ab'][v,A,v,V], t['ab'][pv,pV,ha,hA], optimize='optimal')
	O['ab'][v,V,a,V] += scale * +1.00000000 * np.einsum('iIuA,aBiI->aBuA', h['ab'][c,C,a,V], t['ab'][pv,pV,hc,hC], optimize='optimal')
	O['ab'][v,V,a,V] += scale * -1.00000000 * np.einsum('aIbA,bBuI->aBuA', h['ab'][v,C,v,V], t['ab'][pv,pV,ha,hC], optimize='optimal')
	O['ab'][a,A,c,A] += scale * -1.00000000 * np.einsum('uv,UV,wUvW,uXiV->wXiW', eta1['a'], gamma1['b'], h['ab'][a,A,a,A], t['ab'][pa,pA,hc,hA], optimize='optimal')
	
	O['ab'][a,A,c,A] += scale * -1.00000000 * np.einsum('uv,wIvU,uViI->wViU', eta1['a'], h['ab'][a,C,a,A], t['ab'][pa,pA,hc,hC], optimize='optimal')
	O['ab'][a,A,c,A] += scale * +1.00000000 * np.einsum('uv,wUvA,uAiV->wUiV', eta1['a'], h['ab'][a,A,a,V], t['ab'][pa,pV,hc,hA], optimize='optimal')
	O['ab'][a,A,c,A] += scale * +1.00000000 * np.einsum('UV,uv,wWvV,uUiX->wWiX', eta1['b'], eta1['a'], h['ab'][a,A,a,A], t['ab'][pa,pA,hc,hA], optimize='optimal')
	
	O['ab'][a,A,c,A] += scale * +1.00000000 * np.einsum('UV,uv,wUvW,uXiV->wXiW', eta1['b'], gamma1['a'], h['ab'][a,A,a,A], t['ab'][pa,pA,hc,hA], optimize='optimal')
	
	O['ab'][a,A,c,A] += scale * -1.00000000 * np.einsum('UV,iWjV,uUiX->uWjX', eta1['b'], h['ab'][c,A,c,A], t['ab'][pa,pA,hc,hA], optimize='optimal')
	O['ab'][a,A,c,A] += scale * +1.00000000 * np.einsum('UV,uWaV,aUiX->uWiX', eta1['b'], h['ab'][a,A,v,A], t['ab'][pv,pA,hc,hA], optimize='optimal')
	O['ab'][a,A,c,A] += scale * +1.00000000 * np.einsum('uv,uIiU,wVvI->wViU', gamma1['a'], h['ab'][a,C,c,A], t['ab'][pa,pA,ha,hC], optimize='optimal')
	O['ab'][a,A,c,A] += scale * -1.00000000 * np.einsum('uv,uUiA,wAvV->wUiV', gamma1['a'], h['ab'][a,A,c,V], t['ab'][pa,pV,ha,hA], optimize='optimal')
	O['ab'][a,A,c,A] += scale * -1.00000000 * np.einsum('UV,uv,wWvV,uUiX->wWiX', gamma1['b'], gamma1['a'], h['ab'][a,A,a,A], t['ab'][pa,pA,hc,hA], optimize='optimal')
	
	O['ab'][a,A,c,A] += scale * +1.00000000 * np.einsum('UV,iUjW,uXiV->uXjW', gamma1['b'], h['ab'][c,A,c,A], t['ab'][pa,pA,hc,hA], optimize='optimal')
	O['ab'][a,A,c,A] += scale * -1.00000000 * np.einsum('UV,uUaW,aXiV->uXiW', gamma1['b'], h['ab'][a,A,v,A], t['ab'][pv,pA,hc,hA], optimize='optimal')
	O['ab'][a,A,c,A] += scale * +1.00000000 * np.einsum('iIjU,uViI->uVjU', h['ab'][c,C,c,A], t['ab'][pa,pA,hc,hC], optimize='optimal')
	O['ab'][a,A,c,A] += scale * -1.00000000 * np.einsum('iUjA,uAiV->uUjV', h['ab'][c,A,c,V], t['ab'][pa,pV,hc,hA], optimize='optimal')
	O['ab'][a,A,c,A] += scale * -1.00000000 * np.einsum('uIaU,aViI->uViU', h['ab'][a,C,v,A], t['ab'][pv,pA,hc,hC], optimize='optimal')
	O['ab'][a,A,c,A] += scale * +1.00000000 * np.einsum('uUaA,aAiV->uUiV', h['ab'][a,A,v,V], t['ab'][pv,pV,hc,hA], optimize='optimal')
	O['ab'][c,A,c,A] += scale * -1.00000000 * np.einsum('uv,UV,iUvW,uXjV->iXjW', eta1['a'], gamma1['b'], h['ab'][c,A,a,A], t['ab'][pa,pA,hc,hA], optimize='optimal')
	O['ab'][c,A,c,A] += scale * -1.00000000 * np.einsum('uv,iIvU,uVjI->iVjU', eta1['a'], h['ab'][c,C,a,A], t['ab'][pa,pA,hc,hC], optimize='optimal')
	O['ab'][c,A,c,A] += scale * +1.00000000 * np.einsum('uv,iUvA,uAjV->iUjV', eta1['a'], h['ab'][c,A,a,V], t['ab'][pa,pV,hc,hA], optimize='optimal')
	O['ab'][c,A,c,A] += scale * +1.00000000 * np.einsum('UV,uv,iWvV,uUjX->iWjX', eta1['b'], eta1['a'], h['ab'][c,A,a,A], t['ab'][pa,pA,hc,hA], optimize='optimal')
	O['ab'][c,A,c,A] += scale * +1.00000000 * np.einsum('UV,uv,iUvW,uXjV->iXjW', eta1['b'], gamma1['a'], h['ab'][c,A,a,A], t['ab'][pa,pA,hc,hA], optimize='optimal')
	O['ab'][c,A,c,A] += scale * +1.00000000 * np.einsum('UV,iWaV,aUjX->iWjX', eta1['b'], h['ab'][c,A,v,A], t['ab'][pv,pA,hc,hA], optimize='optimal')
	O['ab'][c,A,c,A] += scale * -1.00000000 * np.einsum('UV,uv,iWvV,uUjX->iWjX', gamma1['b'], gamma1['a'], h['ab'][c,A,a,A], t['ab'][pa,pA,hc,hA], optimize='optimal')
	O['ab'][c,A,c,A] += scale * -1.00000000 * np.einsum('UV,iUaW,aXjV->iXjW', gamma1['b'], h['ab'][c,A,v,A], t['ab'][pv,pA,hc,hA], optimize='optimal')
	O['ab'][c,A,c,A] += scale * -1.00000000 * np.einsum('iIaU,aVjI->iVjU', h['ab'][c,C,v,A], t['ab'][pv,pA,hc,hC], optimize='optimal')
	O['ab'][c,A,c,A] += scale * +1.00000000 * np.einsum('iUaA,aAjV->iUjV', h['ab'][c,A,v,V], t['ab'][pv,pV,hc,hA], optimize='optimal')
	O['ab'][v,A,c,A] += scale * +1.00000000 * np.einsum('uv,UV,uWiV,aUvX->aWiX', eta1['a'], gamma1['b'], h['ab'][a,A,c,A], t['ab'][pv,pA,ha,hA], optimize='optimal')
	O['ab'][v,A,c,A] += scale * -1.00000000 * np.einsum('uv,UV,aUvW,uXiV->aXiW', eta1['a'], gamma1['b'], h['ab'][v,A,a,A], t['ab'][pa,pA,hc,hA], optimize='optimal')
	O['ab'][v,A,c,A] += scale * -1.00000000 * np.einsum('uv,aIvU,uViI->aViU', eta1['a'], h['ab'][v,C,a,A], t['ab'][pa,pA,hc,hC], optimize='optimal')
	O['ab'][v,A,c,A] += scale * +1.00000000 * np.einsum('uv,aUvA,uAiV->aUiV', eta1['a'], h['ab'][v,A,a,V], t['ab'][pa,pV,hc,hA], optimize='optimal')
	O['ab'][v,A,c,A] += scale * -1.00000000 * np.einsum('UV,uv,uUiW,aXvV->aXiW', eta1['b'], eta1['a'], h['ab'][a,A,c,A], t['ab'][pv,pA,ha,hA], optimize='optimal')
	O['ab'][v,A,c,A] += scale * +1.00000000 * np.einsum('UV,uv,aWvV,uUiX->aWiX', eta1['b'], eta1['a'], h['ab'][v,A,a,A], t['ab'][pa,pA,hc,hA], optimize='optimal')
	O['ab'][v,A,c,A] += scale * -1.00000000 * np.einsum('UV,uv,uWiV,aUvX->aWiX', eta1['b'], gamma1['a'], h['ab'][a,A,c,A], t['ab'][pv,pA,ha,hA], optimize='optimal')
	O['ab'][v,A,c,A] += scale * +1.00000000 * np.einsum('UV,uv,aUvW,uXiV->aXiW', eta1['b'], gamma1['a'], h['ab'][v,A,a,A], t['ab'][pa,pA,hc,hA], optimize='optimal')
	O['ab'][v,A,c,A] += scale * -1.00000000 * np.einsum('UV,iWjV,aUiX->aWjX', eta1['b'], h['ab'][c,A,c,A], t['ab'][pv,pA,hc,hA], optimize='optimal')
	O['ab'][v,A,c,A] += scale * +1.00000000 * np.einsum('UV,aWbV,bUiX->aWiX', eta1['b'], h['ab'][v,A,v,A], t['ab'][pv,pA,hc,hA], optimize='optimal')
	O['ab'][v,A,c,A] += scale * +1.00000000 * np.einsum('uv,uIiU,aVvI->aViU', gamma1['a'], h['ab'][a,C,c,A], t['ab'][pv,pA,ha,hC], optimize='optimal')
	O['ab'][v,A,c,A] += scale * -1.00000000 * np.einsum('uv,uUiA,aAvV->aUiV', gamma1['a'], h['ab'][a,A,c,V], t['ab'][pv,pV,ha,hA], optimize='optimal')
	O['ab'][v,A,c,A] += scale * +1.00000000 * np.einsum('UV,uv,uUiW,aXvV->aXiW', gamma1['b'], gamma1['a'], h['ab'][a,A,c,A], t['ab'][pv,pA,ha,hA], optimize='optimal')
	O['ab'][v,A,c,A] += scale * -1.00000000 * np.einsum('UV,uv,aWvV,uUiX->aWiX', gamma1['b'], gamma1['a'], h['ab'][v,A,a,A], t['ab'][pa,pA,hc,hA], optimize='optimal')
	O['ab'][v,A,c,A] += scale * +1.00000000 * np.einsum('UV,iUjW,aXiV->aXjW', gamma1['b'], h['ab'][c,A,c,A], t['ab'][pv,pA,hc,hA], optimize='optimal')
	O['ab'][v,A,c,A] += scale * -1.00000000 * np.einsum('UV,aUbW,bXiV->aXiW', gamma1['b'], h['ab'][v,A,v,A], t['ab'][pv,pA,hc,hA], optimize='optimal')
	O['ab'][v,A,c,A] += scale * +1.00000000 * np.einsum('iIjU,aViI->aVjU', h['ab'][c,C,c,A], t['ab'][pv,pA,hc,hC], optimize='optimal')
	O['ab'][v,A,c,A] += scale * -1.00000000 * np.einsum('iUjA,aAiV->aUjV', h['ab'][c,A,c,V], t['ab'][pv,pV,hc,hA], optimize='optimal')
	O['ab'][v,A,c,A] += scale * -1.00000000 * np.einsum('aIbU,bViI->aViU', h['ab'][v,C,v,A], t['ab'][pv,pA,hc,hC], optimize='optimal')
	O['ab'][v,A,c,A] += scale * +1.00000000 * np.einsum('aUbA,bAiV->aUiV', h['ab'][v,A,v,V], t['ab'][pv,pV,hc,hA], optimize='optimal')
	
	O['ab'][a,C,c,A] += scale * +1.00000000 * np.einsum('uv,wIvA,uAiU->wIiU', eta1['a'], h['ab'][a,C,a,V], t['ab'][pa,pV,hc,hA], optimize='optimal')
	O['ab'][a,C,c,A] += scale * +1.00000000 * np.einsum('UV,uv,wIvV,uUiW->wIiW', eta1['b'], eta1['a'], h['ab'][a,C,a,A], t['ab'][pa,pA,hc,hA], optimize='optimal')
	
	O['ab'][a,C,c,A] += scale * -1.00000000 * np.einsum('UV,iIjV,uUiW->uIjW', eta1['b'], h['ab'][c,C,c,A], t['ab'][pa,pA,hc,hA], optimize='optimal')
	O['ab'][a,C,c,A] += scale * +1.00000000 * np.einsum('UV,uIaV,aUiW->uIiW', eta1['b'], h['ab'][a,C,v,A], t['ab'][pv,pA,hc,hA], optimize='optimal')
	O['ab'][a,C,c,A] += scale * -1.00000000 * np.einsum('uv,uIiA,wAvU->wIiU', gamma1['a'], h['ab'][a,C,c,V], t['ab'][pa,pV,ha,hA], optimize='optimal')
	O['ab'][a,C,c,A] += scale * -1.00000000 * np.einsum('UV,uv,wIvV,uUiW->wIiW', gamma1['b'], gamma1['a'], h['ab'][a,C,a,A], t['ab'][pa,pA,hc,hA], optimize='optimal')
	O['ab'][a,C,c,A] += scale * -1.00000000 * np.einsum('iIjA,uAiU->uIjU', h['ab'][c,C,c,V], t['ab'][pa,pV,hc,hA], optimize='optimal')
	O['ab'][a,C,c,A] += scale * +1.00000000 * np.einsum('uIaA,aAiU->uIiU', h['ab'][a,C,v,V], t['ab'][pv,pV,hc,hA], optimize='optimal')
	O['ab'][c,C,c,A] += scale * +1.00000000 * np.einsum('uv,iIvA,uAjU->iIjU', eta1['a'], h['ab'][c,C,a,V], t['ab'][pa,pV,hc,hA], optimize='optimal')
	O['ab'][c,C,c,A] += scale * +1.00000000 * np.einsum('UV,uv,iIvV,uUjW->iIjW', eta1['b'], eta1['a'], h['ab'][c,C,a,A], t['ab'][pa,pA,hc,hA], optimize='optimal')
	O['ab'][c,C,c,A] += scale * +1.00000000 * np.einsum('UV,iIaV,aUjW->iIjW', eta1['b'], h['ab'][c,C,v,A], t['ab'][pv,pA,hc,hA], optimize='optimal')
	O['ab'][c,C,c,A] += scale * -1.00000000 * np.einsum('UV,uv,iIvV,uUjW->iIjW', gamma1['b'], gamma1['a'], h['ab'][c,C,a,A], t['ab'][pa,pA,hc,hA], optimize='optimal')
	O['ab'][c,C,c,A] += scale * +1.00000000 * np.einsum('iIaA,aAjU->iIjU', h['ab'][c,C,v,V], t['ab'][pv,pV,hc,hA], optimize='optimal')
	O['ab'][v,C,c,A] += scale * +1.00000000 * np.einsum('uv,UV,uIiV,aUvW->aIiW', eta1['a'], gamma1['b'], h['ab'][a,C,c,A], t['ab'][pv,pA,ha,hA], optimize='optimal')
	O['ab'][v,C,c,A] += scale * +1.00000000 * np.einsum('uv,aIvA,uAiU->aIiU', eta1['a'], h['ab'][v,C,a,V], t['ab'][pa,pV,hc,hA], optimize='optimal')
	O['ab'][v,C,c,A] += scale * +1.00000000 * np.einsum('UV,uv,aIvV,uUiW->aIiW', eta1['b'], eta1['a'], h['ab'][v,C,a,A], t['ab'][pa,pA,hc,hA], optimize='optimal')
	O['ab'][v,C,c,A] += scale * -1.00000000 * np.einsum('UV,uv,uIiV,aUvW->aIiW', eta1['b'], gamma1['a'], h['ab'][a,C,c,A], t['ab'][pv,pA,ha,hA], optimize='optimal')
	O['ab'][v,C,c,A] += scale * -1.00000000 * np.einsum('UV,iIjV,aUiW->aIjW', eta1['b'], h['ab'][c,C,c,A], t['ab'][pv,pA,hc,hA], optimize='optimal')
	O['ab'][v,C,c,A] += scale * +1.00000000 * np.einsum('UV,aIbV,bUiW->aIiW', eta1['b'], h['ab'][v,C,v,A], t['ab'][pv,pA,hc,hA], optimize='optimal')
	O['ab'][v,C,c,A] += scale * -1.00000000 * np.einsum('uv,uIiA,aAvU->aIiU', gamma1['a'], h['ab'][a,C,c,V], t['ab'][pv,pV,ha,hA], optimize='optimal')
	O['ab'][v,C,c,A] += scale * -1.00000000 * np.einsum('UV,uv,aIvV,uUiW->aIiW', gamma1['b'], gamma1['a'], h['ab'][v,C,a,A], t['ab'][pa,pA,hc,hA], optimize='optimal')
	O['ab'][v,C,c,A] += scale * -1.00000000 * np.einsum('iIjA,aAiU->aIjU', h['ab'][c,C,c,V], t['ab'][pv,pV,hc,hA], optimize='optimal')
	O['ab'][v,C,c,A] += scale * +1.00000000 * np.einsum('aIbA,bAiU->aIiU', h['ab'][v,C,v,V], t['ab'][pv,pV,hc,hA], optimize='optimal')
	O['ab'][a,V,c,A] += scale * -1.00000000 * np.einsum('uv,UV,wUvW,uAiV->wAiW', eta1['a'], gamma1['b'], h['ab'][a,A,a,A], t['ab'][pa,pV,hc,hA], optimize='optimal')
	
	O['ab'][a,V,c,A] += scale * -1.00000000 * np.einsum('uv,wIvU,uAiI->wAiU', eta1['a'], h['ab'][a,C,a,A], t['ab'][pa,pV,hc,hC], optimize='optimal')
	O['ab'][a,V,c,A] += scale * +1.00000000 * np.einsum('uv,wAvB,uBiU->wAiU', eta1['a'], h['ab'][a,V,a,V], t['ab'][pa,pV,hc,hA], optimize='optimal')
	O['ab'][a,V,c,A] += scale * +1.00000000 * np.einsum('UV,uv,wAvV,uUiW->wAiW', eta1['b'], eta1['a'], h['ab'][a,V,a,A], t['ab'][pa,pA,hc,hA], optimize='optimal')
	O['ab'][a,V,c,A] += scale * -1.00000000 * np.einsum('UV,uv,uUiW,wAvV->wAiW', eta1['b'], eta1['a'], h['ab'][a,A,c,A], t['ab'][pa,pV,ha,hA], optimize='optimal')
	O['ab'][a,V,c,A] += scale * +1.00000000 * np.einsum('UV,uv,wUvW,uAiV->wAiW', eta1['b'], gamma1['a'], h['ab'][a,A,a,A], t['ab'][pa,pV,hc,hA], optimize='optimal')
	
	O['ab'][a,V,c,A] += scale * -1.00000000 * np.einsum('UV,iAjV,uUiW->uAjW', eta1['b'], h['ab'][c,V,c,A], t['ab'][pa,pA,hc,hA], optimize='optimal')
	O['ab'][a,V,c,A] += scale * +1.00000000 * np.einsum('UV,uAaV,aUiW->uAiW', eta1['b'], h['ab'][a,V,v,A], t['ab'][pv,pA,hc,hA], optimize='optimal')
	O['ab'][a,V,c,A] += scale * +1.00000000 * np.einsum('uv,uIiU,wAvI->wAiU', gamma1['a'], h['ab'][a,C,c,A], t['ab'][pa,pV,ha,hC], optimize='optimal')
	O['ab'][a,V,c,A] += scale * -1.00000000 * np.einsum('uv,uAiB,wBvU->wAiU', gamma1['a'], h['ab'][a,V,c,V], t['ab'][pa,pV,ha,hA], optimize='optimal')
	O['ab'][a,V,c,A] += scale * -1.00000000 * np.einsum('UV,uv,wAvV,uUiW->wAiW', gamma1['b'], gamma1['a'], h['ab'][a,V,a,A], t['ab'][pa,pA,hc,hA], optimize='optimal')
	O['ab'][a,V,c,A] += scale * +1.00000000 * np.einsum('UV,uv,uUiW,wAvV->wAiW', gamma1['b'], gamma1['a'], h['ab'][a,A,c,A], t['ab'][pa,pV,ha,hA], optimize='optimal')
	O['ab'][a,V,c,A] += scale * +1.00000000 * np.einsum('UV,iUjW,uAiV->uAjW', gamma1['b'], h['ab'][c,A,c,A], t['ab'][pa,pV,hc,hA], optimize='optimal')
	O['ab'][a,V,c,A] += scale * -1.00000000 * np.einsum('UV,uUaW,aAiV->uAiW', gamma1['b'], h['ab'][a,A,v,A], t['ab'][pv,pV,hc,hA], optimize='optimal')
	O['ab'][a,V,c,A] += scale * +1.00000000 * np.einsum('iIjU,uAiI->uAjU', h['ab'][c,C,c,A], t['ab'][pa,pV,hc,hC], optimize='optimal')
	O['ab'][a,V,c,A] += scale * -1.00000000 * np.einsum('iAjB,uBiU->uAjU', h['ab'][c,V,c,V], t['ab'][pa,pV,hc,hA], optimize='optimal')
	O['ab'][a,V,c,A] += scale * -1.00000000 * np.einsum('uIaU,aAiI->uAiU', h['ab'][a,C,v,A], t['ab'][pv,pV,hc,hC], optimize='optimal')
	O['ab'][a,V,c,A] += scale * +1.00000000 * np.einsum('uAaB,aBiU->uAiU', h['ab'][a,V,v,V], t['ab'][pv,pV,hc,hA], optimize='optimal')
	O['ab'][c,V,c,A] += scale * -1.00000000 * np.einsum('uv,UV,iUvW,uAjV->iAjW', eta1['a'], gamma1['b'], h['ab'][c,A,a,A], t['ab'][pa,pV,hc,hA], optimize='optimal')
	O['ab'][c,V,c,A] += scale * -1.00000000 * np.einsum('uv,iIvU,uAjI->iAjU', eta1['a'], h['ab'][c,C,a,A], t['ab'][pa,pV,hc,hC], optimize='optimal')
	O['ab'][c,V,c,A] += scale * +1.00000000 * np.einsum('uv,iAvB,uBjU->iAjU', eta1['a'], h['ab'][c,V,a,V], t['ab'][pa,pV,hc,hA], optimize='optimal')
	O['ab'][c,V,c,A] += scale * +1.00000000 * np.einsum('UV,uv,iAvV,uUjW->iAjW', eta1['b'], eta1['a'], h['ab'][c,V,a,A], t['ab'][pa,pA,hc,hA], optimize='optimal')
	O['ab'][c,V,c,A] += scale * +1.00000000 * np.einsum('UV,uv,iUvW,uAjV->iAjW', eta1['b'], gamma1['a'], h['ab'][c,A,a,A], t['ab'][pa,pV,hc,hA], optimize='optimal')
	O['ab'][c,V,c,A] += scale * +1.00000000 * np.einsum('UV,iAaV,aUjW->iAjW', eta1['b'], h['ab'][c,V,v,A], t['ab'][pv,pA,hc,hA], optimize='optimal')
	O['ab'][c,V,c,A] += scale * -1.00000000 * np.einsum('UV,uv,iAvV,uUjW->iAjW', gamma1['b'], gamma1['a'], h['ab'][c,V,a,A], t['ab'][pa,pA,hc,hA], optimize='optimal')
	O['ab'][c,V,c,A] += scale * -1.00000000 * np.einsum('UV,iUaW,aAjV->iAjW', gamma1['b'], h['ab'][c,A,v,A], t['ab'][pv,pV,hc,hA], optimize='optimal')
	O['ab'][c,V,c,A] += scale * -1.00000000 * np.einsum('iIaU,aAjI->iAjU', h['ab'][c,C,v,A], t['ab'][pv,pV,hc,hC], optimize='optimal')
	O['ab'][c,V,c,A] += scale * +1.00000000 * np.einsum('iAaB,aBjU->iAjU', h['ab'][c,V,v,V], t['ab'][pv,pV,hc,hA], optimize='optimal')
	O['ab'][v,V,c,A] += scale * +1.00000000 * np.einsum('uv,UV,uAiV,aUvW->aAiW', eta1['a'], gamma1['b'], h['ab'][a,V,c,A], t['ab'][pv,pA,ha,hA], optimize='optimal')
	O['ab'][v,V,c,A] += scale * -1.00000000 * np.einsum('uv,UV,aUvW,uAiV->aAiW', eta1['a'], gamma1['b'], h['ab'][v,A,a,A], t['ab'][pa,pV,hc,hA], optimize='optimal')
	O['ab'][v,V,c,A] += scale * -1.00000000 * np.einsum('uv,aIvU,uAiI->aAiU', eta1['a'], h['ab'][v,C,a,A], t['ab'][pa,pV,hc,hC], optimize='optimal')
	O['ab'][v,V,c,A] += scale * +1.00000000 * np.einsum('uv,aAvB,uBiU->aAiU', eta1['a'], h['ab'][v,V,a,V], t['ab'][pa,pV,hc,hA], optimize='optimal')
	O['ab'][v,V,c,A] += scale * -1.00000000 * np.einsum('UV,uv,uUiW,aAvV->aAiW', eta1['b'], eta1['a'], h['ab'][a,A,c,A], t['ab'][pv,pV,ha,hA], optimize='optimal')
	O['ab'][v,V,c,A] += scale * +1.00000000 * np.einsum('UV,uv,aAvV,uUiW->aAiW', eta1['b'], eta1['a'], h['ab'][v,V,a,A], t['ab'][pa,pA,hc,hA], optimize='optimal')
	O['ab'][v,V,c,A] += scale * -1.00000000 * np.einsum('UV,uv,uAiV,aUvW->aAiW', eta1['b'], gamma1['a'], h['ab'][a,V,c,A], t['ab'][pv,pA,ha,hA], optimize='optimal')
	O['ab'][v,V,c,A] += scale * +1.00000000 * np.einsum('UV,uv,aUvW,uAiV->aAiW', eta1['b'], gamma1['a'], h['ab'][v,A,a,A], t['ab'][pa,pV,hc,hA], optimize='optimal')
	O['ab'][v,V,c,A] += scale * -1.00000000 * np.einsum('UV,iAjV,aUiW->aAjW', eta1['b'], h['ab'][c,V,c,A], t['ab'][pv,pA,hc,hA], optimize='optimal')
	O['ab'][v,V,c,A] += scale * +1.00000000 * np.einsum('UV,aAbV,bUiW->aAiW', eta1['b'], h['ab'][v,V,v,A], t['ab'][pv,pA,hc,hA], optimize='optimal')
	O['ab'][v,V,c,A] += scale * +1.00000000 * np.einsum('uv,uIiU,aAvI->aAiU', gamma1['a'], h['ab'][a,C,c,A], t['ab'][pv,pV,ha,hC], optimize='optimal')
	O['ab'][v,V,c,A] += scale * -1.00000000 * np.einsum('uv,uAiB,aBvU->aAiU', gamma1['a'], h['ab'][a,V,c,V], t['ab'][pv,pV,ha,hA], optimize='optimal')
	O['ab'][v,V,c,A] += scale * +1.00000000 * np.einsum('UV,uv,uUiW,aAvV->aAiW', gamma1['b'], gamma1['a'], h['ab'][a,A,c,A], t['ab'][pv,pV,ha,hA], optimize='optimal')
	O['ab'][v,V,c,A] += scale * -1.00000000 * np.einsum('UV,uv,aAvV,uUiW->aAiW', gamma1['b'], gamma1['a'], h['ab'][v,V,a,A], t['ab'][pa,pA,hc,hA], optimize='optimal')
	O['ab'][v,V,c,A] += scale * +1.00000000 * np.einsum('UV,iUjW,aAiV->aAjW', gamma1['b'], h['ab'][c,A,c,A], t['ab'][pv,pV,hc,hA], optimize='optimal')
	O['ab'][v,V,c,A] += scale * -1.00000000 * np.einsum('UV,aUbW,bAiV->aAiW', gamma1['b'], h['ab'][v,A,v,A], t['ab'][pv,pV,hc,hA], optimize='optimal')
	O['ab'][v,V,c,A] += scale * +1.00000000 * np.einsum('iIjU,aAiI->aAjU', h['ab'][c,C,c,A], t['ab'][pv,pV,hc,hC], optimize='optimal')
	O['ab'][v,V,c,A] += scale * -1.00000000 * np.einsum('iAjB,aBiU->aAjU', h['ab'][c,V,c,V], t['ab'][pv,pV,hc,hA], optimize='optimal')
	O['ab'][v,V,c,A] += scale * -1.00000000 * np.einsum('aIbU,bAiI->aAiU', h['ab'][v,C,v,A], t['ab'][pv,pV,hc,hC], optimize='optimal')
	O['ab'][v,V,c,A] += scale * +1.00000000 * np.einsum('aAbB,bBiU->aAiU', h['ab'][v,V,v,V], t['ab'][pv,pV,hc,hA], optimize='optimal')
	O['ab'][a,A,c,C] += scale * -1.00000000 * np.einsum('uv,UV,wUvI,uWiV->wWiI', eta1['a'], gamma1['b'], h['ab'][a,A,a,C], t['ab'][pa,pA,hc,hA], optimize='optimal')
	O['ab'][a,A,c,C] += scale * +1.00000000 * np.einsum('uv,UV,uWiV,wUvI->wWiI', eta1['a'], gamma1['b'], h['ab'][a,A,c,A], t['ab'][pa,pA,ha,hC], optimize='optimal')
	O['ab'][a,A,c,C] += scale * -1.00000000 * np.einsum('uv,wIvJ,uUiI->wUiJ', eta1['a'], h['ab'][a,C,a,C], t['ab'][pa,pA,hc,hC], optimize='optimal')
	O['ab'][a,A,c,C] += scale * +1.00000000 * np.einsum('uv,wUvA,uAiI->wUiI', eta1['a'], h['ab'][a,A,a,V], t['ab'][pa,pV,hc,hC], optimize='optimal')
	O['ab'][a,A,c,C] += scale * +1.00000000 * np.einsum('UV,uv,wWvV,uUiI->wWiI', eta1['b'], eta1['a'], h['ab'][a,A,a,A], t['ab'][pa,pA,hc,hC], optimize='optimal')
	
	O['ab'][a,A,c,C] += scale * +1.00000000 * np.einsum('UV,uv,wUvI,uWiV->wWiI', eta1['b'], gamma1['a'], h['ab'][a,A,a,C], t['ab'][pa,pA,hc,hA], optimize='optimal')
	O['ab'][a,A,c,C] += scale * -1.00000000 * np.einsum('UV,uv,uWiV,wUvI->wWiI', eta1['b'], gamma1['a'], h['ab'][a,A,c,A], t['ab'][pa,pA,ha,hC], optimize='optimal')
	O['ab'][a,A,c,C] += scale * -1.00000000 * np.einsum('UV,iWjV,uUiI->uWjI', eta1['b'], h['ab'][c,A,c,A], t['ab'][pa,pA,hc,hC], optimize='optimal')
	O['ab'][a,A,c,C] += scale * +1.00000000 * np.einsum('UV,uWaV,aUiI->uWiI', eta1['b'], h['ab'][a,A,v,A], t['ab'][pv,pA,hc,hC], optimize='optimal')
	O['ab'][a,A,c,C] += scale * +1.00000000 * np.einsum('uv,uIiJ,wUvI->wUiJ', gamma1['a'], h['ab'][a,C,c,C], t['ab'][pa,pA,ha,hC], optimize='optimal')
	O['ab'][a,A,c,C] += scale * -1.00000000 * np.einsum('uv,uUiA,wAvI->wUiI', gamma1['a'], h['ab'][a,A,c,V], t['ab'][pa,pV,ha,hC], optimize='optimal')
	O['ab'][a,A,c,C] += scale * -1.00000000 * np.einsum('UV,uv,wWvV,uUiI->wWiI', gamma1['b'], gamma1['a'], h['ab'][a,A,a,A], t['ab'][pa,pA,hc,hC], optimize='optimal')
	
	O['ab'][a,A,c,C] += scale * +1.00000000 * np.einsum('UV,iUjI,uWiV->uWjI', gamma1['b'], h['ab'][c,A,c,C], t['ab'][pa,pA,hc,hA], optimize='optimal')
	O['ab'][a,A,c,C] += scale * -1.00000000 * np.einsum('UV,uUaI,aWiV->uWiI', gamma1['b'], h['ab'][a,A,v,C], t['ab'][pv,pA,hc,hA], optimize='optimal')
	O['ab'][a,A,c,C] += scale * +1.00000000 * np.einsum('iIjJ,uUiI->uUjJ', h['ab'][c,C,c,C], t['ab'][pa,pA,hc,hC], optimize='optimal')
	O['ab'][a,A,c,C] += scale * -1.00000000 * np.einsum('iUjA,uAiI->uUjI', h['ab'][c,A,c,V], t['ab'][pa,pV,hc,hC], optimize='optimal')
	O['ab'][a,A,c,C] += scale * -1.00000000 * np.einsum('uIaJ,aUiI->uUiJ', h['ab'][a,C,v,C], t['ab'][pv,pA,hc,hC], optimize='optimal')
	O['ab'][a,A,c,C] += scale * +1.00000000 * np.einsum('uUaA,aAiI->uUiI', h['ab'][a,A,v,V], t['ab'][pv,pV,hc,hC], optimize='optimal')
	O['ab'][c,A,c,C] += scale * -1.00000000 * np.einsum('uv,UV,iUvI,uWjV->iWjI', eta1['a'], gamma1['b'], h['ab'][c,A,a,C], t['ab'][pa,pA,hc,hA], optimize='optimal')
	O['ab'][c,A,c,C] += scale * -1.00000000 * np.einsum('uv,iIvJ,uUjI->iUjJ', eta1['a'], h['ab'][c,C,a,C], t['ab'][pa,pA,hc,hC], optimize='optimal')
	O['ab'][c,A,c,C] += scale * +1.00000000 * np.einsum('uv,iUvA,uAjI->iUjI', eta1['a'], h['ab'][c,A,a,V], t['ab'][pa,pV,hc,hC], optimize='optimal')
	O['ab'][c,A,c,C] += scale * +1.00000000 * np.einsum('UV,uv,iWvV,uUjI->iWjI', eta1['b'], eta1['a'], h['ab'][c,A,a,A], t['ab'][pa,pA,hc,hC], optimize='optimal')
	O['ab'][c,A,c,C] += scale * +1.00000000 * np.einsum('UV,uv,iUvI,uWjV->iWjI', eta1['b'], gamma1['a'], h['ab'][c,A,a,C], t['ab'][pa,pA,hc,hA], optimize='optimal')
	O['ab'][c,A,c,C] += scale * +1.00000000 * np.einsum('UV,iWaV,aUjI->iWjI', eta1['b'], h['ab'][c,A,v,A], t['ab'][pv,pA,hc,hC], optimize='optimal')
	O['ab'][c,A,c,C] += scale * -1.00000000 * np.einsum('UV,uv,iWvV,uUjI->iWjI', gamma1['b'], gamma1['a'], h['ab'][c,A,a,A], t['ab'][pa,pA,hc,hC], optimize='optimal')
	O['ab'][c,A,c,C] += scale * -1.00000000 * np.einsum('UV,iUaI,aWjV->iWjI', gamma1['b'], h['ab'][c,A,v,C], t['ab'][pv,pA,hc,hA], optimize='optimal')
	O['ab'][c,A,c,C] += scale * -1.00000000 * np.einsum('iIaJ,aUjI->iUjJ', h['ab'][c,C,v,C], t['ab'][pv,pA,hc,hC], optimize='optimal')
	O['ab'][c,A,c,C] += scale * +1.00000000 * np.einsum('iUaA,aAjI->iUjI', h['ab'][c,A,v,V], t['ab'][pv,pV,hc,hC], optimize='optimal')
	O['ab'][v,A,c,C] += scale * +1.00000000 * np.einsum('uv,UV,uWiV,aUvI->aWiI', eta1['a'], gamma1['b'], h['ab'][a,A,c,A], t['ab'][pv,pA,ha,hC], optimize='optimal')
	O['ab'][v,A,c,C] += scale * -1.00000000 * np.einsum('uv,UV,aUvI,uWiV->aWiI', eta1['a'], gamma1['b'], h['ab'][v,A,a,C], t['ab'][pa,pA,hc,hA], optimize='optimal')
	O['ab'][v,A,c,C] += scale * -1.00000000 * np.einsum('uv,aIvJ,uUiI->aUiJ', eta1['a'], h['ab'][v,C,a,C], t['ab'][pa,pA,hc,hC], optimize='optimal')
	O['ab'][v,A,c,C] += scale * +1.00000000 * np.einsum('uv,aUvA,uAiI->aUiI', eta1['a'], h['ab'][v,A,a,V], t['ab'][pa,pV,hc,hC], optimize='optimal')
	O['ab'][v,A,c,C] += scale * -1.00000000 * np.einsum('UV,uv,uUiI,aWvV->aWiI', eta1['b'], eta1['a'], h['ab'][a,A,c,C], t['ab'][pv,pA,ha,hA], optimize='optimal')
	O['ab'][v,A,c,C] += scale * +1.00000000 * np.einsum('UV,uv,aWvV,uUiI->aWiI', eta1['b'], eta1['a'], h['ab'][v,A,a,A], t['ab'][pa,pA,hc,hC], optimize='optimal')
	O['ab'][v,A,c,C] += scale * -1.00000000 * np.einsum('UV,uv,uWiV,aUvI->aWiI', eta1['b'], gamma1['a'], h['ab'][a,A,c,A], t['ab'][pv,pA,ha,hC], optimize='optimal')
	O['ab'][v,A,c,C] += scale * +1.00000000 * np.einsum('UV,uv,aUvI,uWiV->aWiI', eta1['b'], gamma1['a'], h['ab'][v,A,a,C], t['ab'][pa,pA,hc,hA], optimize='optimal')
	O['ab'][v,A,c,C] += scale * -1.00000000 * np.einsum('UV,iWjV,aUiI->aWjI', eta1['b'], h['ab'][c,A,c,A], t['ab'][pv,pA,hc,hC], optimize='optimal')
	O['ab'][v,A,c,C] += scale * +1.00000000 * np.einsum('UV,aWbV,bUiI->aWiI', eta1['b'], h['ab'][v,A,v,A], t['ab'][pv,pA,hc,hC], optimize='optimal')
	O['ab'][v,A,c,C] += scale * +1.00000000 * np.einsum('uv,uIiJ,aUvI->aUiJ', gamma1['a'], h['ab'][a,C,c,C], t['ab'][pv,pA,ha,hC], optimize='optimal')
	O['ab'][v,A,c,C] += scale * -1.00000000 * np.einsum('uv,uUiA,aAvI->aUiI', gamma1['a'], h['ab'][a,A,c,V], t['ab'][pv,pV,ha,hC], optimize='optimal')
	O['ab'][v,A,c,C] += scale * +1.00000000 * np.einsum('UV,uv,uUiI,aWvV->aWiI', gamma1['b'], gamma1['a'], h['ab'][a,A,c,C], t['ab'][pv,pA,ha,hA], optimize='optimal')
	O['ab'][v,A,c,C] += scale * -1.00000000 * np.einsum('UV,uv,aWvV,uUiI->aWiI', gamma1['b'], gamma1['a'], h['ab'][v,A,a,A], t['ab'][pa,pA,hc,hC], optimize='optimal')
	O['ab'][v,A,c,C] += scale * +1.00000000 * np.einsum('UV,iUjI,aWiV->aWjI', gamma1['b'], h['ab'][c,A,c,C], t['ab'][pv,pA,hc,hA], optimize='optimal')
	O['ab'][v,A,c,C] += scale * -1.00000000 * np.einsum('UV,aUbI,bWiV->aWiI', gamma1['b'], h['ab'][v,A,v,C], t['ab'][pv,pA,hc,hA], optimize='optimal')
	O['ab'][v,A,c,C] += scale * +1.00000000 * np.einsum('iIjJ,aUiI->aUjJ', h['ab'][c,C,c,C], t['ab'][pv,pA,hc,hC], optimize='optimal')
	O['ab'][v,A,c,C] += scale * -1.00000000 * np.einsum('iUjA,aAiI->aUjI', h['ab'][c,A,c,V], t['ab'][pv,pV,hc,hC], optimize='optimal')
	O['ab'][v,A,c,C] += scale * -1.00000000 * np.einsum('aIbJ,bUiI->aUiJ', h['ab'][v,C,v,C], t['ab'][pv,pA,hc,hC], optimize='optimal')
	O['ab'][v,A,c,C] += scale * +1.00000000 * np.einsum('aUbA,bAiI->aUiI', h['ab'][v,A,v,V], t['ab'][pv,pV,hc,hC], optimize='optimal')
	O['ab'][a,C,c,C] += scale * +1.00000000 * np.einsum('uv,UV,uIiV,wUvJ->wIiJ', eta1['a'], gamma1['b'], h['ab'][a,C,c,A], t['ab'][pa,pA,ha,hC], optimize='optimal')
	O['ab'][a,C,c,C] += scale * +1.00000000 * np.einsum('uv,wIvA,uAiJ->wIiJ', eta1['a'], h['ab'][a,C,a,V], t['ab'][pa,pV,hc,hC], optimize='optimal')
	O['ab'][a,C,c,C] += scale * +1.00000000 * np.einsum('UV,uv,wIvV,uUiJ->wIiJ', eta1['b'], eta1['a'], h['ab'][a,C,a,A], t['ab'][pa,pA,hc,hC], optimize='optimal')
	O['ab'][a,C,c,C] += scale * -1.00000000 * np.einsum('UV,uv,uIiV,wUvJ->wIiJ', eta1['b'], gamma1['a'], h['ab'][a,C,c,A], t['ab'][pa,pA,ha,hC], optimize='optimal')
	O['ab'][a,C,c,C] += scale * -1.00000000 * np.einsum('UV,iIjV,uUiJ->uIjJ', eta1['b'], h['ab'][c,C,c,A], t['ab'][pa,pA,hc,hC], optimize='optimal')
	O['ab'][a,C,c,C] += scale * +1.00000000 * np.einsum('UV,uIaV,aUiJ->uIiJ', eta1['b'], h['ab'][a,C,v,A], t['ab'][pv,pA,hc,hC], optimize='optimal')
	O['ab'][a,C,c,C] += scale * -1.00000000 * np.einsum('uv,uIiA,wAvJ->wIiJ', gamma1['a'], h['ab'][a,C,c,V], t['ab'][pa,pV,ha,hC], optimize='optimal')
	O['ab'][a,C,c,C] += scale * -1.00000000 * np.einsum('UV,uv,wIvV,uUiJ->wIiJ', gamma1['b'], gamma1['a'], h['ab'][a,C,a,A], t['ab'][pa,pA,hc,hC], optimize='optimal')
	O['ab'][a,C,c,C] += scale * -1.00000000 * np.einsum('iIjA,uAiJ->uIjJ', h['ab'][c,C,c,V], t['ab'][pa,pV,hc,hC], optimize='optimal')
	O['ab'][a,C,c,C] += scale * +1.00000000 * np.einsum('uIaA,aAiJ->uIiJ', h['ab'][a,C,v,V], t['ab'][pv,pV,hc,hC], optimize='optimal')
	O['ab'][c,C,c,C] += scale * +1.00000000 * np.einsum('uv,iIvA,uAjJ->iIjJ', eta1['a'], h['ab'][c,C,a,V], t['ab'][pa,pV,hc,hC], optimize='optimal')
	O['ab'][c,C,c,C] += scale * +1.00000000 * np.einsum('UV,uv,iIvV,uUjJ->iIjJ', eta1['b'], eta1['a'], h['ab'][c,C,a,A], t['ab'][pa,pA,hc,hC], optimize='optimal')
	O['ab'][c,C,c,C] += scale * +1.00000000 * np.einsum('UV,iIaV,aUjJ->iIjJ', eta1['b'], h['ab'][c,C,v,A], t['ab'][pv,pA,hc,hC], optimize='optimal')
	O['ab'][c,C,c,C] += scale * -1.00000000 * np.einsum('UV,uv,iIvV,uUjJ->iIjJ', gamma1['b'], gamma1['a'], h['ab'][c,C,a,A], t['ab'][pa,pA,hc,hC], optimize='optimal')
	O['ab'][c,C,c,C] += scale * +1.00000000 * np.einsum('iIaA,aAjJ->iIjJ', h['ab'][c,C,v,V], t['ab'][pv,pV,hc,hC], optimize='optimal')
	O['ab'][v,C,c,C] += scale * +1.00000000 * np.einsum('uv,UV,uIiV,aUvJ->aIiJ', eta1['a'], gamma1['b'], h['ab'][a,C,c,A], t['ab'][pv,pA,ha,hC], optimize='optimal')
	O['ab'][v,C,c,C] += scale * +1.00000000 * np.einsum('uv,aIvA,uAiJ->aIiJ', eta1['a'], h['ab'][v,C,a,V], t['ab'][pa,pV,hc,hC], optimize='optimal')
	O['ab'][v,C,c,C] += scale * +1.00000000 * np.einsum('UV,uv,aIvV,uUiJ->aIiJ', eta1['b'], eta1['a'], h['ab'][v,C,a,A], t['ab'][pa,pA,hc,hC], optimize='optimal')
	O['ab'][v,C,c,C] += scale * -1.00000000 * np.einsum('UV,uv,uIiV,aUvJ->aIiJ', eta1['b'], gamma1['a'], h['ab'][a,C,c,A], t['ab'][pv,pA,ha,hC], optimize='optimal')
	O['ab'][v,C,c,C] += scale * -1.00000000 * np.einsum('UV,iIjV,aUiJ->aIjJ', eta1['b'], h['ab'][c,C,c,A], t['ab'][pv,pA,hc,hC], optimize='optimal')
	O['ab'][v,C,c,C] += scale * +1.00000000 * np.einsum('UV,aIbV,bUiJ->aIiJ', eta1['b'], h['ab'][v,C,v,A], t['ab'][pv,pA,hc,hC], optimize='optimal')
	O['ab'][v,C,c,C] += scale * -1.00000000 * np.einsum('uv,uIiA,aAvJ->aIiJ', gamma1['a'], h['ab'][a,C,c,V], t['ab'][pv,pV,ha,hC], optimize='optimal')
	O['ab'][v,C,c,C] += scale * -1.00000000 * np.einsum('UV,uv,aIvV,uUiJ->aIiJ', gamma1['b'], gamma1['a'], h['ab'][v,C,a,A], t['ab'][pa,pA,hc,hC], optimize='optimal')
	O['ab'][v,C,c,C] += scale * -1.00000000 * np.einsum('iIjA,aAiJ->aIjJ', h['ab'][c,C,c,V], t['ab'][pv,pV,hc,hC], optimize='optimal')
	O['ab'][v,C,c,C] += scale * +1.00000000 * np.einsum('aIbA,bAiJ->aIiJ', h['ab'][v,C,v,V], t['ab'][pv,pV,hc,hC], optimize='optimal')
	O['ab'][a,V,c,C] += scale * -1.00000000 * np.einsum('uv,UV,wUvI,uAiV->wAiI', eta1['a'], gamma1['b'], h['ab'][a,A,a,C], t['ab'][pa,pV,hc,hA], optimize='optimal')
	O['ab'][a,V,c,C] += scale * +1.00000000 * np.einsum('uv,UV,uAiV,wUvI->wAiI', eta1['a'], gamma1['b'], h['ab'][a,V,c,A], t['ab'][pa,pA,ha,hC], optimize='optimal')
	O['ab'][a,V,c,C] += scale * -1.00000000 * np.einsum('uv,wIvJ,uAiI->wAiJ', eta1['a'], h['ab'][a,C,a,C], t['ab'][pa,pV,hc,hC], optimize='optimal')
	O['ab'][a,V,c,C] += scale * +1.00000000 * np.einsum('uv,wAvB,uBiI->wAiI', eta1['a'], h['ab'][a,V,a,V], t['ab'][pa,pV,hc,hC], optimize='optimal')
	O['ab'][a,V,c,C] += scale * +1.00000000 * np.einsum('UV,uv,wAvV,uUiI->wAiI', eta1['b'], eta1['a'], h['ab'][a,V,a,A], t['ab'][pa,pA,hc,hC], optimize='optimal')
	O['ab'][a,V,c,C] += scale * -1.00000000 * np.einsum('UV,uv,uUiI,wAvV->wAiI', eta1['b'], eta1['a'], h['ab'][a,A,c,C], t['ab'][pa,pV,ha,hA], optimize='optimal')
	O['ab'][a,V,c,C] += scale * +1.00000000 * np.einsum('UV,uv,wUvI,uAiV->wAiI', eta1['b'], gamma1['a'], h['ab'][a,A,a,C], t['ab'][pa,pV,hc,hA], optimize='optimal')
	O['ab'][a,V,c,C] += scale * -1.00000000 * np.einsum('UV,uv,uAiV,wUvI->wAiI', eta1['b'], gamma1['a'], h['ab'][a,V,c,A], t['ab'][pa,pA,ha,hC], optimize='optimal')
	O['ab'][a,V,c,C] += scale * -1.00000000 * np.einsum('UV,iAjV,uUiI->uAjI', eta1['b'], h['ab'][c,V,c,A], t['ab'][pa,pA,hc,hC], optimize='optimal')
	O['ab'][a,V,c,C] += scale * +1.00000000 * np.einsum('UV,uAaV,aUiI->uAiI', eta1['b'], h['ab'][a,V,v,A], t['ab'][pv,pA,hc,hC], optimize='optimal')
	O['ab'][a,V,c,C] += scale * +1.00000000 * np.einsum('uv,uIiJ,wAvI->wAiJ', gamma1['a'], h['ab'][a,C,c,C], t['ab'][pa,pV,ha,hC], optimize='optimal')
	O['ab'][a,V,c,C] += scale * -1.00000000 * np.einsum('uv,uAiB,wBvI->wAiI', gamma1['a'], h['ab'][a,V,c,V], t['ab'][pa,pV,ha,hC], optimize='optimal')
	O['ab'][a,V,c,C] += scale * -1.00000000 * np.einsum('UV,uv,wAvV,uUiI->wAiI', gamma1['b'], gamma1['a'], h['ab'][a,V,a,A], t['ab'][pa,pA,hc,hC], optimize='optimal')
	O['ab'][a,V,c,C] += scale * +1.00000000 * np.einsum('UV,uv,uUiI,wAvV->wAiI', gamma1['b'], gamma1['a'], h['ab'][a,A,c,C], t['ab'][pa,pV,ha,hA], optimize='optimal')
	O['ab'][a,V,c,C] += scale * +1.00000000 * np.einsum('UV,iUjI,uAiV->uAjI', gamma1['b'], h['ab'][c,A,c,C], t['ab'][pa,pV,hc,hA], optimize='optimal')
	O['ab'][a,V,c,C] += scale * -1.00000000 * np.einsum('UV,uUaI,aAiV->uAiI', gamma1['b'], h['ab'][a,A,v,C], t['ab'][pv,pV,hc,hA], optimize='optimal')
	O['ab'][a,V,c,C] += scale * +1.00000000 * np.einsum('iIjJ,uAiI->uAjJ', h['ab'][c,C,c,C], t['ab'][pa,pV,hc,hC], optimize='optimal')
	O['ab'][a,V,c,C] += scale * -1.00000000 * np.einsum('iAjB,uBiI->uAjI', h['ab'][c,V,c,V], t['ab'][pa,pV,hc,hC], optimize='optimal')
	O['ab'][a,V,c,C] += scale * -1.00000000 * np.einsum('uIaJ,aAiI->uAiJ', h['ab'][a,C,v,C], t['ab'][pv,pV,hc,hC], optimize='optimal')
	O['ab'][a,V,c,C] += scale * +1.00000000 * np.einsum('uAaB,aBiI->uAiI', h['ab'][a,V,v,V], t['ab'][pv,pV,hc,hC], optimize='optimal')
	O['ab'][c,V,c,C] += scale * -1.00000000 * np.einsum('uv,UV,iUvI,uAjV->iAjI', eta1['a'], gamma1['b'], h['ab'][c,A,a,C], t['ab'][pa,pV,hc,hA], optimize='optimal')
	O['ab'][c,V,c,C] += scale * -1.00000000 * np.einsum('uv,iIvJ,uAjI->iAjJ', eta1['a'], h['ab'][c,C,a,C], t['ab'][pa,pV,hc,hC], optimize='optimal')
	O['ab'][c,V,c,C] += scale * +1.00000000 * np.einsum('uv,iAvB,uBjI->iAjI', eta1['a'], h['ab'][c,V,a,V], t['ab'][pa,pV,hc,hC], optimize='optimal')
	O['ab'][c,V,c,C] += scale * +1.00000000 * np.einsum('UV,uv,iAvV,uUjI->iAjI', eta1['b'], eta1['a'], h['ab'][c,V,a,A], t['ab'][pa,pA,hc,hC], optimize='optimal')
	O['ab'][c,V,c,C] += scale * +1.00000000 * np.einsum('UV,uv,iUvI,uAjV->iAjI', eta1['b'], gamma1['a'], h['ab'][c,A,a,C], t['ab'][pa,pV,hc,hA], optimize='optimal')
	O['ab'][c,V,c,C] += scale * +1.00000000 * np.einsum('UV,iAaV,aUjI->iAjI', eta1['b'], h['ab'][c,V,v,A], t['ab'][pv,pA,hc,hC], optimize='optimal')
	O['ab'][c,V,c,C] += scale * -1.00000000 * np.einsum('UV,uv,iAvV,uUjI->iAjI', gamma1['b'], gamma1['a'], h['ab'][c,V,a,A], t['ab'][pa,pA,hc,hC], optimize='optimal')
	O['ab'][c,V,c,C] += scale * -1.00000000 * np.einsum('UV,iUaI,aAjV->iAjI', gamma1['b'], h['ab'][c,A,v,C], t['ab'][pv,pV,hc,hA], optimize='optimal')
	O['ab'][c,V,c,C] += scale * -1.00000000 * np.einsum('iIaJ,aAjI->iAjJ', h['ab'][c,C,v,C], t['ab'][pv,pV,hc,hC], optimize='optimal')
	O['ab'][c,V,c,C] += scale * +1.00000000 * np.einsum('iAaB,aBjI->iAjI', h['ab'][c,V,v,V], t['ab'][pv,pV,hc,hC], optimize='optimal')
	O['ab'][v,V,c,C] += scale * +1.00000000 * np.einsum('uv,UV,uAiV,aUvI->aAiI', eta1['a'], gamma1['b'], h['ab'][a,V,c,A], t['ab'][pv,pA,ha,hC], optimize='optimal')
	O['ab'][v,V,c,C] += scale * -1.00000000 * np.einsum('uv,UV,aUvI,uAiV->aAiI', eta1['a'], gamma1['b'], h['ab'][v,A,a,C], t['ab'][pa,pV,hc,hA], optimize='optimal')
	O['ab'][v,V,c,C] += scale * -1.00000000 * np.einsum('uv,aIvJ,uAiI->aAiJ', eta1['a'], h['ab'][v,C,a,C], t['ab'][pa,pV,hc,hC], optimize='optimal')
	O['ab'][v,V,c,C] += scale * +1.00000000 * np.einsum('uv,aAvB,uBiI->aAiI', eta1['a'], h['ab'][v,V,a,V], t['ab'][pa,pV,hc,hC], optimize='optimal')
	O['ab'][v,V,c,C] += scale * -1.00000000 * np.einsum('UV,uv,uUiI,aAvV->aAiI', eta1['b'], eta1['a'], h['ab'][a,A,c,C], t['ab'][pv,pV,ha,hA], optimize='optimal')
	O['ab'][v,V,c,C] += scale * +1.00000000 * np.einsum('UV,uv,aAvV,uUiI->aAiI', eta1['b'], eta1['a'], h['ab'][v,V,a,A], t['ab'][pa,pA,hc,hC], optimize='optimal')
	O['ab'][v,V,c,C] += scale * -1.00000000 * np.einsum('UV,uv,uAiV,aUvI->aAiI', eta1['b'], gamma1['a'], h['ab'][a,V,c,A], t['ab'][pv,pA,ha,hC], optimize='optimal')
	O['ab'][v,V,c,C] += scale * +1.00000000 * np.einsum('UV,uv,aUvI,uAiV->aAiI', eta1['b'], gamma1['a'], h['ab'][v,A,a,C], t['ab'][pa,pV,hc,hA], optimize='optimal')
	O['ab'][v,V,c,C] += scale * -1.00000000 * np.einsum('UV,iAjV,aUiI->aAjI', eta1['b'], h['ab'][c,V,c,A], t['ab'][pv,pA,hc,hC], optimize='optimal')
	O['ab'][v,V,c,C] += scale * +1.00000000 * np.einsum('UV,aAbV,bUiI->aAiI', eta1['b'], h['ab'][v,V,v,A], t['ab'][pv,pA,hc,hC], optimize='optimal')
	O['ab'][v,V,c,C] += scale * +1.00000000 * np.einsum('uv,uIiJ,aAvI->aAiJ', gamma1['a'], h['ab'][a,C,c,C], t['ab'][pv,pV,ha,hC], optimize='optimal')
	O['ab'][v,V,c,C] += scale * -1.00000000 * np.einsum('uv,uAiB,aBvI->aAiI', gamma1['a'], h['ab'][a,V,c,V], t['ab'][pv,pV,ha,hC], optimize='optimal')
	O['ab'][v,V,c,C] += scale * +1.00000000 * np.einsum('UV,uv,uUiI,aAvV->aAiI', gamma1['b'], gamma1['a'], h['ab'][a,A,c,C], t['ab'][pv,pV,ha,hA], optimize='optimal')
	O['ab'][v,V,c,C] += scale * -1.00000000 * np.einsum('UV,uv,aAvV,uUiI->aAiI', gamma1['b'], gamma1['a'], h['ab'][v,V,a,A], t['ab'][pa,pA,hc,hC], optimize='optimal')
	O['ab'][v,V,c,C] += scale * +1.00000000 * np.einsum('UV,iUjI,aAiV->aAjI', gamma1['b'], h['ab'][c,A,c,C], t['ab'][pv,pV,hc,hA], optimize='optimal')
	O['ab'][v,V,c,C] += scale * -1.00000000 * np.einsum('UV,aUbI,bAiV->aAiI', gamma1['b'], h['ab'][v,A,v,C], t['ab'][pv,pV,hc,hA], optimize='optimal')
	O['ab'][v,V,c,C] += scale * +1.00000000 * np.einsum('iIjJ,aAiI->aAjJ', h['ab'][c,C,c,C], t['ab'][pv,pV,hc,hC], optimize='optimal')
	O['ab'][v,V,c,C] += scale * -1.00000000 * np.einsum('iAjB,aBiI->aAjI', h['ab'][c,V,c,V], t['ab'][pv,pV,hc,hC], optimize='optimal')
	O['ab'][v,V,c,C] += scale * -1.00000000 * np.einsum('aIbJ,bAiI->aAiJ', h['ab'][v,C,v,C], t['ab'][pv,pV,hc,hC], optimize='optimal')
	O['ab'][v,V,c,C] += scale * +1.00000000 * np.einsum('aAbB,bBiI->aAiI', h['ab'][v,V,v,V], t['ab'][pv,pV,hc,hC], optimize='optimal')
	O['ab'][a,A,c,V] += scale * -1.00000000 * np.einsum('uv,UV,wUvA,uWiV->wWiA', eta1['a'], gamma1['b'], h['ab'][a,A,a,V], t['ab'][pa,pA,hc,hA], optimize='optimal')
	O['ab'][a,A,c,V] += scale * -1.00000000 * np.einsum('uv,wIvA,uUiI->wUiA', eta1['a'], h['ab'][a,C,a,V], t['ab'][pa,pA,hc,hC], optimize='optimal')
	
	O['ab'][a,A,c,V] += scale * +1.00000000 * np.einsum('UV,uv,wUvA,uWiV->wWiA', eta1['b'], gamma1['a'], h['ab'][a,A,a,V], t['ab'][pa,pA,hc,hA], optimize='optimal')
	O['ab'][a,A,c,V] += scale * +1.00000000 * np.einsum('uv,uIiA,wUvI->wUiA', gamma1['a'], h['ab'][a,C,c,V], t['ab'][pa,pA,ha,hC], optimize='optimal')
	
	O['ab'][a,A,c,V] += scale * +1.00000000 * np.einsum('UV,iUjA,uWiV->uWjA', gamma1['b'], h['ab'][c,A,c,V], t['ab'][pa,pA,hc,hA], optimize='optimal')
	O['ab'][a,A,c,V] += scale * -1.00000000 * np.einsum('UV,uUaA,aWiV->uWiA', gamma1['b'], h['ab'][a,A,v,V], t['ab'][pv,pA,hc,hA], optimize='optimal')
	O['ab'][a,A,c,V] += scale * +1.00000000 * np.einsum('iIjA,uUiI->uUjA', h['ab'][c,C,c,V], t['ab'][pa,pA,hc,hC], optimize='optimal')
	O['ab'][a,A,c,V] += scale * -1.00000000 * np.einsum('uIaA,aUiI->uUiA', h['ab'][a,C,v,V], t['ab'][pv,pA,hc,hC], optimize='optimal')
	O['ab'][c,A,c,V] += scale * -1.00000000 * np.einsum('uv,UV,iUvA,uWjV->iWjA', eta1['a'], gamma1['b'], h['ab'][c,A,a,V], t['ab'][pa,pA,hc,hA], optimize='optimal')
	O['ab'][c,A,c,V] += scale * -1.00000000 * np.einsum('uv,iIvA,uUjI->iUjA', eta1['a'], h['ab'][c,C,a,V], t['ab'][pa,pA,hc,hC], optimize='optimal')
	O['ab'][c,A,c,V] += scale * +1.00000000 * np.einsum('UV,uv,iUvA,uWjV->iWjA', eta1['b'], gamma1['a'], h['ab'][c,A,a,V], t['ab'][pa,pA,hc,hA], optimize='optimal')
	O['ab'][c,A,c,V] += scale * -1.00000000 * np.einsum('UV,iUaA,aWjV->iWjA', gamma1['b'], h['ab'][c,A,v,V], t['ab'][pv,pA,hc,hA], optimize='optimal')
	O['ab'][c,A,c,V] += scale * -1.00000000 * np.einsum('iIaA,aUjI->iUjA', h['ab'][c,C,v,V], t['ab'][pv,pA,hc,hC], optimize='optimal')
	O['ab'][v,A,c,V] += scale * -1.00000000 * np.einsum('uv,UV,aUvA,uWiV->aWiA', eta1['a'], gamma1['b'], h['ab'][v,A,a,V], t['ab'][pa,pA,hc,hA], optimize='optimal')
	O['ab'][v,A,c,V] += scale * -1.00000000 * np.einsum('uv,aIvA,uUiI->aUiA', eta1['a'], h['ab'][v,C,a,V], t['ab'][pa,pA,hc,hC], optimize='optimal')
	O['ab'][v,A,c,V] += scale * -1.00000000 * np.einsum('UV,uv,uUiA,aWvV->aWiA', eta1['b'], eta1['a'], h['ab'][a,A,c,V], t['ab'][pv,pA,ha,hA], optimize='optimal')
	O['ab'][v,A,c,V] += scale * +1.00000000 * np.einsum('UV,uv,aUvA,uWiV->aWiA', eta1['b'], gamma1['a'], h['ab'][v,A,a,V], t['ab'][pa,pA,hc,hA], optimize='optimal')
	O['ab'][v,A,c,V] += scale * +1.00000000 * np.einsum('uv,uIiA,aUvI->aUiA', gamma1['a'], h['ab'][a,C,c,V], t['ab'][pv,pA,ha,hC], optimize='optimal')
	O['ab'][v,A,c,V] += scale * +1.00000000 * np.einsum('UV,uv,uUiA,aWvV->aWiA', gamma1['b'], gamma1['a'], h['ab'][a,A,c,V], t['ab'][pv,pA,ha,hA], optimize='optimal')
	O['ab'][v,A,c,V] += scale * +1.00000000 * np.einsum('UV,iUjA,aWiV->aWjA', gamma1['b'], h['ab'][c,A,c,V], t['ab'][pv,pA,hc,hA], optimize='optimal')
	O['ab'][v,A,c,V] += scale * -1.00000000 * np.einsum('UV,aUbA,bWiV->aWiA', gamma1['b'], h['ab'][v,A,v,V], t['ab'][pv,pA,hc,hA], optimize='optimal')
	O['ab'][v,A,c,V] += scale * +1.00000000 * np.einsum('iIjA,aUiI->aUjA', h['ab'][c,C,c,V], t['ab'][pv,pA,hc,hC], optimize='optimal')
	O['ab'][v,A,c,V] += scale * -1.00000000 * np.einsum('aIbA,bUiI->aUiA', h['ab'][v,C,v,V], t['ab'][pv,pA,hc,hC], optimize='optimal')
	O['ab'][a,V,c,V] += scale * -1.00000000 * np.einsum('uv,UV,wUvA,uBiV->wBiA', eta1['a'], gamma1['b'], h['ab'][a,A,a,V], t['ab'][pa,pV,hc,hA], optimize='optimal')
	O['ab'][a,V,c,V] += scale * -1.00000000 * np.einsum('uv,wIvA,uBiI->wBiA', eta1['a'], h['ab'][a,C,a,V], t['ab'][pa,pV,hc,hC], optimize='optimal')
	O['ab'][a,V,c,V] += scale * -1.00000000 * np.einsum('UV,uv,uUiA,wBvV->wBiA', eta1['b'], eta1['a'], h['ab'][a,A,c,V], t['ab'][pa,pV,ha,hA], optimize='optimal')
	O['ab'][a,V,c,V] += scale * +1.00000000 * np.einsum('UV,uv,wUvA,uBiV->wBiA', eta1['b'], gamma1['a'], h['ab'][a,A,a,V], t['ab'][pa,pV,hc,hA], optimize='optimal')
	O['ab'][a,V,c,V] += scale * +1.00000000 * np.einsum('uv,uIiA,wBvI->wBiA', gamma1['a'], h['ab'][a,C,c,V], t['ab'][pa,pV,ha,hC], optimize='optimal')
	O['ab'][a,V,c,V] += scale * +1.00000000 * np.einsum('UV,uv,uUiA,wBvV->wBiA', gamma1['b'], gamma1['a'], h['ab'][a,A,c,V], t['ab'][pa,pV,ha,hA], optimize='optimal')
	O['ab'][a,V,c,V] += scale * +1.00000000 * np.einsum('UV,iUjA,uBiV->uBjA', gamma1['b'], h['ab'][c,A,c,V], t['ab'][pa,pV,hc,hA], optimize='optimal')
	O['ab'][a,V,c,V] += scale * -1.00000000 * np.einsum('UV,uUaA,aBiV->uBiA', gamma1['b'], h['ab'][a,A,v,V], t['ab'][pv,pV,hc,hA], optimize='optimal')
	O['ab'][a,V,c,V] += scale * +1.00000000 * np.einsum('iIjA,uBiI->uBjA', h['ab'][c,C,c,V], t['ab'][pa,pV,hc,hC], optimize='optimal')
	O['ab'][a,V,c,V] += scale * -1.00000000 * np.einsum('uIaA,aBiI->uBiA', h['ab'][a,C,v,V], t['ab'][pv,pV,hc,hC], optimize='optimal')
	O['ab'][c,V,c,V] += scale * -1.00000000 * np.einsum('uv,UV,iUvA,uBjV->iBjA', eta1['a'], gamma1['b'], h['ab'][c,A,a,V], t['ab'][pa,pV,hc,hA], optimize='optimal')
	O['ab'][c,V,c,V] += scale * -1.00000000 * np.einsum('uv,iIvA,uBjI->iBjA', eta1['a'], h['ab'][c,C,a,V], t['ab'][pa,pV,hc,hC], optimize='optimal')
	O['ab'][c,V,c,V] += scale * +1.00000000 * np.einsum('UV,uv,iUvA,uBjV->iBjA', eta1['b'], gamma1['a'], h['ab'][c,A,a,V], t['ab'][pa,pV,hc,hA], optimize='optimal')
	O['ab'][c,V,c,V] += scale * -1.00000000 * np.einsum('UV,iUaA,aBjV->iBjA', gamma1['b'], h['ab'][c,A,v,V], t['ab'][pv,pV,hc,hA], optimize='optimal')
	O['ab'][c,V,c,V] += scale * -1.00000000 * np.einsum('iIaA,aBjI->iBjA', h['ab'][c,C,v,V], t['ab'][pv,pV,hc,hC], optimize='optimal')
	O['ab'][v,V,c,V] += scale * -1.00000000 * np.einsum('uv,UV,aUvA,uBiV->aBiA', eta1['a'], gamma1['b'], h['ab'][v,A,a,V], t['ab'][pa,pV,hc,hA], optimize='optimal')
	O['ab'][v,V,c,V] += scale * -1.00000000 * np.einsum('uv,aIvA,uBiI->aBiA', eta1['a'], h['ab'][v,C,a,V], t['ab'][pa,pV,hc,hC], optimize='optimal')
	O['ab'][v,V,c,V] += scale * -1.00000000 * np.einsum('UV,uv,uUiA,aBvV->aBiA', eta1['b'], eta1['a'], h['ab'][a,A,c,V], t['ab'][pv,pV,ha,hA], optimize='optimal')
	O['ab'][v,V,c,V] += scale * +1.00000000 * np.einsum('UV,uv,aUvA,uBiV->aBiA', eta1['b'], gamma1['a'], h['ab'][v,A,a,V], t['ab'][pa,pV,hc,hA], optimize='optimal')
	O['ab'][v,V,c,V] += scale * +1.00000000 * np.einsum('uv,uIiA,aBvI->aBiA', gamma1['a'], h['ab'][a,C,c,V], t['ab'][pv,pV,ha,hC], optimize='optimal')
	O['ab'][v,V,c,V] += scale * +1.00000000 * np.einsum('UV,uv,uUiA,aBvV->aBiA', gamma1['b'], gamma1['a'], h['ab'][a,A,c,V], t['ab'][pv,pV,ha,hA], optimize='optimal')
	O['ab'][v,V,c,V] += scale * +1.00000000 * np.einsum('UV,iUjA,aBiV->aBjA', gamma1['b'], h['ab'][c,A,c,V], t['ab'][pv,pV,hc,hA], optimize='optimal')
	O['ab'][v,V,c,V] += scale * -1.00000000 * np.einsum('UV,aUbA,bBiV->aBiA', gamma1['b'], h['ab'][v,A,v,V], t['ab'][pv,pV,hc,hA], optimize='optimal')
	O['ab'][v,V,c,V] += scale * +1.00000000 * np.einsum('iIjA,aBiI->aBjA', h['ab'][c,C,c,V], t['ab'][pv,pV,hc,hC], optimize='optimal')
	O['ab'][v,V,c,V] += scale * -1.00000000 * np.einsum('aIbA,bBiI->aBiA', h['ab'][v,C,v,V], t['ab'][pv,pV,hc,hC], optimize='optimal')
	
	
	
	O['ab'][a,A,v,A] += scale * -1.00000000 * np.einsum('UV,iWaV,uUiX->uWaX', eta1['b'], h['ab'][c,A,v,A], t['ab'][pa,pA,hc,hA], optimize='optimal')
	O['ab'][a,A,v,A] += scale * +1.00000000 * np.einsum('uv,uIaU,wVvI->wVaU', gamma1['a'], h['ab'][a,C,v,A], t['ab'][pa,pA,ha,hC], optimize='optimal')
	O['ab'][a,A,v,A] += scale * -1.00000000 * np.einsum('uv,uUaA,wAvV->wUaV', gamma1['a'], h['ab'][a,A,v,V], t['ab'][pa,pV,ha,hA], optimize='optimal')
	
	O['ab'][a,A,v,A] += scale * +1.00000000 * np.einsum('UV,iUaW,uXiV->uXaW', gamma1['b'], h['ab'][c,A,v,A], t['ab'][pa,pA,hc,hA], optimize='optimal')
	O['ab'][a,A,v,A] += scale * +1.00000000 * np.einsum('iIaU,uViI->uVaU', h['ab'][c,C,v,A], t['ab'][pa,pA,hc,hC], optimize='optimal')
	O['ab'][a,A,v,A] += scale * -1.00000000 * np.einsum('iUaA,uAiV->uUaV', h['ab'][c,A,v,V], t['ab'][pa,pV,hc,hA], optimize='optimal')
	O['ab'][v,A,v,A] += scale * +1.00000000 * np.einsum('uv,UV,uWaV,bUvX->bWaX', eta1['a'], gamma1['b'], h['ab'][a,A,v,A], t['ab'][pv,pA,ha,hA], optimize='optimal')
	O['ab'][v,A,v,A] += scale * -1.00000000 * np.einsum('UV,uv,uUaW,bXvV->bXaW', eta1['b'], eta1['a'], h['ab'][a,A,v,A], t['ab'][pv,pA,ha,hA], optimize='optimal')
	O['ab'][v,A,v,A] += scale * -1.00000000 * np.einsum('UV,uv,uWaV,bUvX->bWaX', eta1['b'], gamma1['a'], h['ab'][a,A,v,A], t['ab'][pv,pA,ha,hA], optimize='optimal')
	O['ab'][v,A,v,A] += scale * -1.00000000 * np.einsum('UV,iWaV,bUiX->bWaX', eta1['b'], h['ab'][c,A,v,A], t['ab'][pv,pA,hc,hA], optimize='optimal')
	O['ab'][v,A,v,A] += scale * +1.00000000 * np.einsum('uv,uIaU,bVvI->bVaU', gamma1['a'], h['ab'][a,C,v,A], t['ab'][pv,pA,ha,hC], optimize='optimal')
	O['ab'][v,A,v,A] += scale * -1.00000000 * np.einsum('uv,uUaA,bAvV->bUaV', gamma1['a'], h['ab'][a,A,v,V], t['ab'][pv,pV,ha,hA], optimize='optimal')
	O['ab'][v,A,v,A] += scale * +1.00000000 * np.einsum('UV,uv,uUaW,bXvV->bXaW', gamma1['b'], gamma1['a'], h['ab'][a,A,v,A], t['ab'][pv,pA,ha,hA], optimize='optimal')
	O['ab'][v,A,v,A] += scale * +1.00000000 * np.einsum('UV,iUaW,bXiV->bXaW', gamma1['b'], h['ab'][c,A,v,A], t['ab'][pv,pA,hc,hA], optimize='optimal')
	O['ab'][v,A,v,A] += scale * +1.00000000 * np.einsum('iIaU,bViI->bVaU', h['ab'][c,C,v,A], t['ab'][pv,pA,hc,hC], optimize='optimal')
	O['ab'][v,A,v,A] += scale * -1.00000000 * np.einsum('iUaA,bAiV->bUaV', h['ab'][c,A,v,V], t['ab'][pv,pV,hc,hA], optimize='optimal')
	
	
	O['ab'][a,C,v,A] += scale * -1.00000000 * np.einsum('UV,iIaV,uUiW->uIaW', eta1['b'], h['ab'][c,C,v,A], t['ab'][pa,pA,hc,hA], optimize='optimal')
	O['ab'][a,C,v,A] += scale * -1.00000000 * np.einsum('uv,uIaA,wAvU->wIaU', gamma1['a'], h['ab'][a,C,v,V], t['ab'][pa,pV,ha,hA], optimize='optimal')
	O['ab'][a,C,v,A] += scale * -1.00000000 * np.einsum('iIaA,uAiU->uIaU', h['ab'][c,C,v,V], t['ab'][pa,pV,hc,hA], optimize='optimal')
	O['ab'][v,C,v,A] += scale * +1.00000000 * np.einsum('uv,UV,uIaV,bUvW->bIaW', eta1['a'], gamma1['b'], h['ab'][a,C,v,A], t['ab'][pv,pA,ha,hA], optimize='optimal')
	O['ab'][v,C,v,A] += scale * -1.00000000 * np.einsum('UV,uv,uIaV,bUvW->bIaW', eta1['b'], gamma1['a'], h['ab'][a,C,v,A], t['ab'][pv,pA,ha,hA], optimize='optimal')
	O['ab'][v,C,v,A] += scale * -1.00000000 * np.einsum('UV,iIaV,bUiW->bIaW', eta1['b'], h['ab'][c,C,v,A], t['ab'][pv,pA,hc,hA], optimize='optimal')
	O['ab'][v,C,v,A] += scale * -1.00000000 * np.einsum('uv,uIaA,bAvU->bIaU', gamma1['a'], h['ab'][a,C,v,V], t['ab'][pv,pV,ha,hA], optimize='optimal')
	O['ab'][v,C,v,A] += scale * -1.00000000 * np.einsum('iIaA,bAiU->bIaU', h['ab'][c,C,v,V], t['ab'][pv,pV,hc,hA], optimize='optimal')
	
	O['ab'][a,V,v,A] += scale * -1.00000000 * np.einsum('UV,uv,uUaW,wAvV->wAaW', eta1['b'], eta1['a'], h['ab'][a,A,v,A], t['ab'][pa,pV,ha,hA], optimize='optimal')
	
	O['ab'][a,V,v,A] += scale * -1.00000000 * np.einsum('UV,iAaV,uUiW->uAaW', eta1['b'], h['ab'][c,V,v,A], t['ab'][pa,pA,hc,hA], optimize='optimal')
	O['ab'][a,V,v,A] += scale * +1.00000000 * np.einsum('uv,uIaU,wAvI->wAaU', gamma1['a'], h['ab'][a,C,v,A], t['ab'][pa,pV,ha,hC], optimize='optimal')
	O['ab'][a,V,v,A] += scale * -1.00000000 * np.einsum('uv,uAaB,wBvU->wAaU', gamma1['a'], h['ab'][a,V,v,V], t['ab'][pa,pV,ha,hA], optimize='optimal')
	O['ab'][a,V,v,A] += scale * +1.00000000 * np.einsum('UV,uv,uUaW,wAvV->wAaW', gamma1['b'], gamma1['a'], h['ab'][a,A,v,A], t['ab'][pa,pV,ha,hA], optimize='optimal')
	O['ab'][a,V,v,A] += scale * +1.00000000 * np.einsum('UV,iUaW,uAiV->uAaW', gamma1['b'], h['ab'][c,A,v,A], t['ab'][pa,pV,hc,hA], optimize='optimal')
	O['ab'][a,V,v,A] += scale * +1.00000000 * np.einsum('iIaU,uAiI->uAaU', h['ab'][c,C,v,A], t['ab'][pa,pV,hc,hC], optimize='optimal')
	O['ab'][a,V,v,A] += scale * -1.00000000 * np.einsum('iAaB,uBiU->uAaU', h['ab'][c,V,v,V], t['ab'][pa,pV,hc,hA], optimize='optimal')
	O['ab'][v,V,v,A] += scale * +1.00000000 * np.einsum('uv,UV,uAaV,bUvW->bAaW', eta1['a'], gamma1['b'], h['ab'][a,V,v,A], t['ab'][pv,pA,ha,hA], optimize='optimal')
	O['ab'][v,V,v,A] += scale * -1.00000000 * np.einsum('UV,uv,uUaW,bAvV->bAaW', eta1['b'], eta1['a'], h['ab'][a,A,v,A], t['ab'][pv,pV,ha,hA], optimize='optimal')
	O['ab'][v,V,v,A] += scale * -1.00000000 * np.einsum('UV,uv,uAaV,bUvW->bAaW', eta1['b'], gamma1['a'], h['ab'][a,V,v,A], t['ab'][pv,pA,ha,hA], optimize='optimal')
	O['ab'][v,V,v,A] += scale * -1.00000000 * np.einsum('UV,iAaV,bUiW->bAaW', eta1['b'], h['ab'][c,V,v,A], t['ab'][pv,pA,hc,hA], optimize='optimal')
	O['ab'][v,V,v,A] += scale * +1.00000000 * np.einsum('uv,uIaU,bAvI->bAaU', gamma1['a'], h['ab'][a,C,v,A], t['ab'][pv,pV,ha,hC], optimize='optimal')
	O['ab'][v,V,v,A] += scale * -1.00000000 * np.einsum('uv,uAaB,bBvU->bAaU', gamma1['a'], h['ab'][a,V,v,V], t['ab'][pv,pV,ha,hA], optimize='optimal')
	O['ab'][v,V,v,A] += scale * +1.00000000 * np.einsum('UV,uv,uUaW,bAvV->bAaW', gamma1['b'], gamma1['a'], h['ab'][a,A,v,A], t['ab'][pv,pV,ha,hA], optimize='optimal')
	O['ab'][v,V,v,A] += scale * +1.00000000 * np.einsum('UV,iUaW,bAiV->bAaW', gamma1['b'], h['ab'][c,A,v,A], t['ab'][pv,pV,hc,hA], optimize='optimal')
	O['ab'][v,V,v,A] += scale * +1.00000000 * np.einsum('iIaU,bAiI->bAaU', h['ab'][c,C,v,A], t['ab'][pv,pV,hc,hC], optimize='optimal')
	O['ab'][v,V,v,A] += scale * -1.00000000 * np.einsum('iAaB,bBiU->bAaU', h['ab'][c,V,v,V], t['ab'][pv,pV,hc,hA], optimize='optimal')
	O['ab'][a,A,v,C] += scale * +1.00000000 * np.einsum('uv,UV,uWaV,wUvI->wWaI', eta1['a'], gamma1['b'], h['ab'][a,A,v,A], t['ab'][pa,pA,ha,hC], optimize='optimal')
	
	O['ab'][a,A,v,C] += scale * -1.00000000 * np.einsum('UV,uv,uWaV,wUvI->wWaI', eta1['b'], gamma1['a'], h['ab'][a,A,v,A], t['ab'][pa,pA,ha,hC], optimize='optimal')
	O['ab'][a,A,v,C] += scale * -1.00000000 * np.einsum('UV,iWaV,uUiI->uWaI', eta1['b'], h['ab'][c,A,v,A], t['ab'][pa,pA,hc,hC], optimize='optimal')
	O['ab'][a,A,v,C] += scale * +1.00000000 * np.einsum('uv,uIaJ,wUvI->wUaJ', gamma1['a'], h['ab'][a,C,v,C], t['ab'][pa,pA,ha,hC], optimize='optimal')
	O['ab'][a,A,v,C] += scale * -1.00000000 * np.einsum('uv,uUaA,wAvI->wUaI', gamma1['a'], h['ab'][a,A,v,V], t['ab'][pa,pV,ha,hC], optimize='optimal')
	
	O['ab'][a,A,v,C] += scale * +1.00000000 * np.einsum('UV,iUaI,uWiV->uWaI', gamma1['b'], h['ab'][c,A,v,C], t['ab'][pa,pA,hc,hA], optimize='optimal')
	O['ab'][a,A,v,C] += scale * +1.00000000 * np.einsum('iIaJ,uUiI->uUaJ', h['ab'][c,C,v,C], t['ab'][pa,pA,hc,hC], optimize='optimal')
	O['ab'][a,A,v,C] += scale * -1.00000000 * np.einsum('iUaA,uAiI->uUaI', h['ab'][c,A,v,V], t['ab'][pa,pV,hc,hC], optimize='optimal')
	O['ab'][v,A,v,C] += scale * +1.00000000 * np.einsum('uv,UV,uWaV,bUvI->bWaI', eta1['a'], gamma1['b'], h['ab'][a,A,v,A], t['ab'][pv,pA,ha,hC], optimize='optimal')
	O['ab'][v,A,v,C] += scale * -1.00000000 * np.einsum('UV,uv,uUaI,bWvV->bWaI', eta1['b'], eta1['a'], h['ab'][a,A,v,C], t['ab'][pv,pA,ha,hA], optimize='optimal')
	O['ab'][v,A,v,C] += scale * -1.00000000 * np.einsum('UV,uv,uWaV,bUvI->bWaI', eta1['b'], gamma1['a'], h['ab'][a,A,v,A], t['ab'][pv,pA,ha,hC], optimize='optimal')
	O['ab'][v,A,v,C] += scale * -1.00000000 * np.einsum('UV,iWaV,bUiI->bWaI', eta1['b'], h['ab'][c,A,v,A], t['ab'][pv,pA,hc,hC], optimize='optimal')
	O['ab'][v,A,v,C] += scale * +1.00000000 * np.einsum('uv,uIaJ,bUvI->bUaJ', gamma1['a'], h['ab'][a,C,v,C], t['ab'][pv,pA,ha,hC], optimize='optimal')
	O['ab'][v,A,v,C] += scale * -1.00000000 * np.einsum('uv,uUaA,bAvI->bUaI', gamma1['a'], h['ab'][a,A,v,V], t['ab'][pv,pV,ha,hC], optimize='optimal')
	O['ab'][v,A,v,C] += scale * +1.00000000 * np.einsum('UV,uv,uUaI,bWvV->bWaI', gamma1['b'], gamma1['a'], h['ab'][a,A,v,C], t['ab'][pv,pA,ha,hA], optimize='optimal')
	O['ab'][v,A,v,C] += scale * +1.00000000 * np.einsum('UV,iUaI,bWiV->bWaI', gamma1['b'], h['ab'][c,A,v,C], t['ab'][pv,pA,hc,hA], optimize='optimal')
	O['ab'][v,A,v,C] += scale * +1.00000000 * np.einsum('iIaJ,bUiI->bUaJ', h['ab'][c,C,v,C], t['ab'][pv,pA,hc,hC], optimize='optimal')
	O['ab'][v,A,v,C] += scale * -1.00000000 * np.einsum('iUaA,bAiI->bUaI', h['ab'][c,A,v,V], t['ab'][pv,pV,hc,hC], optimize='optimal')
	O['ab'][a,C,v,C] += scale * +1.00000000 * np.einsum('uv,UV,uIaV,wUvJ->wIaJ', eta1['a'], gamma1['b'], h['ab'][a,C,v,A], t['ab'][pa,pA,ha,hC], optimize='optimal')
	O['ab'][a,C,v,C] += scale * -1.00000000 * np.einsum('UV,uv,uIaV,wUvJ->wIaJ', eta1['b'], gamma1['a'], h['ab'][a,C,v,A], t['ab'][pa,pA,ha,hC], optimize='optimal')
	O['ab'][a,C,v,C] += scale * -1.00000000 * np.einsum('UV,iIaV,uUiJ->uIaJ', eta1['b'], h['ab'][c,C,v,A], t['ab'][pa,pA,hc,hC], optimize='optimal')
	O['ab'][a,C,v,C] += scale * -1.00000000 * np.einsum('uv,uIaA,wAvJ->wIaJ', gamma1['a'], h['ab'][a,C,v,V], t['ab'][pa,pV,ha,hC], optimize='optimal')
	O['ab'][a,C,v,C] += scale * -1.00000000 * np.einsum('iIaA,uAiJ->uIaJ', h['ab'][c,C,v,V], t['ab'][pa,pV,hc,hC], optimize='optimal')
	O['ab'][v,C,v,C] += scale * +1.00000000 * np.einsum('uv,UV,uIaV,bUvJ->bIaJ', eta1['a'], gamma1['b'], h['ab'][a,C,v,A], t['ab'][pv,pA,ha,hC], optimize='optimal')
	O['ab'][v,C,v,C] += scale * -1.00000000 * np.einsum('UV,uv,uIaV,bUvJ->bIaJ', eta1['b'], gamma1['a'], h['ab'][a,C,v,A], t['ab'][pv,pA,ha,hC], optimize='optimal')
	O['ab'][v,C,v,C] += scale * -1.00000000 * np.einsum('UV,iIaV,bUiJ->bIaJ', eta1['b'], h['ab'][c,C,v,A], t['ab'][pv,pA,hc,hC], optimize='optimal')
	O['ab'][v,C,v,C] += scale * -1.00000000 * np.einsum('uv,uIaA,bAvJ->bIaJ', gamma1['a'], h['ab'][a,C,v,V], t['ab'][pv,pV,ha,hC], optimize='optimal')
	O['ab'][v,C,v,C] += scale * -1.00000000 * np.einsum('iIaA,bAiJ->bIaJ', h['ab'][c,C,v,V], t['ab'][pv,pV,hc,hC], optimize='optimal')
	O['ab'][a,V,v,C] += scale * +1.00000000 * np.einsum('uv,UV,uAaV,wUvI->wAaI', eta1['a'], gamma1['b'], h['ab'][a,V,v,A], t['ab'][pa,pA,ha,hC], optimize='optimal')
	O['ab'][a,V,v,C] += scale * -1.00000000 * np.einsum('UV,uv,uUaI,wAvV->wAaI', eta1['b'], eta1['a'], h['ab'][a,A,v,C], t['ab'][pa,pV,ha,hA], optimize='optimal')
	O['ab'][a,V,v,C] += scale * -1.00000000 * np.einsum('UV,uv,uAaV,wUvI->wAaI', eta1['b'], gamma1['a'], h['ab'][a,V,v,A], t['ab'][pa,pA,ha,hC], optimize='optimal')
	O['ab'][a,V,v,C] += scale * -1.00000000 * np.einsum('UV,iAaV,uUiI->uAaI', eta1['b'], h['ab'][c,V,v,A], t['ab'][pa,pA,hc,hC], optimize='optimal')
	O['ab'][a,V,v,C] += scale * +1.00000000 * np.einsum('uv,uIaJ,wAvI->wAaJ', gamma1['a'], h['ab'][a,C,v,C], t['ab'][pa,pV,ha,hC], optimize='optimal')
	O['ab'][a,V,v,C] += scale * -1.00000000 * np.einsum('uv,uAaB,wBvI->wAaI', gamma1['a'], h['ab'][a,V,v,V], t['ab'][pa,pV,ha,hC], optimize='optimal')
	O['ab'][a,V,v,C] += scale * +1.00000000 * np.einsum('UV,uv,uUaI,wAvV->wAaI', gamma1['b'], gamma1['a'], h['ab'][a,A,v,C], t['ab'][pa,pV,ha,hA], optimize='optimal')
	O['ab'][a,V,v,C] += scale * +1.00000000 * np.einsum('UV,iUaI,uAiV->uAaI', gamma1['b'], h['ab'][c,A,v,C], t['ab'][pa,pV,hc,hA], optimize='optimal')
	O['ab'][a,V,v,C] += scale * +1.00000000 * np.einsum('iIaJ,uAiI->uAaJ', h['ab'][c,C,v,C], t['ab'][pa,pV,hc,hC], optimize='optimal')
	O['ab'][a,V,v,C] += scale * -1.00000000 * np.einsum('iAaB,uBiI->uAaI', h['ab'][c,V,v,V], t['ab'][pa,pV,hc,hC], optimize='optimal')
	O['ab'][v,V,v,C] += scale * +1.00000000 * np.einsum('uv,UV,uAaV,bUvI->bAaI', eta1['a'], gamma1['b'], h['ab'][a,V,v,A], t['ab'][pv,pA,ha,hC], optimize='optimal')
	O['ab'][v,V,v,C] += scale * -1.00000000 * np.einsum('UV,uv,uUaI,bAvV->bAaI', eta1['b'], eta1['a'], h['ab'][a,A,v,C], t['ab'][pv,pV,ha,hA], optimize='optimal')
	O['ab'][v,V,v,C] += scale * -1.00000000 * np.einsum('UV,uv,uAaV,bUvI->bAaI', eta1['b'], gamma1['a'], h['ab'][a,V,v,A], t['ab'][pv,pA,ha,hC], optimize='optimal')
	O['ab'][v,V,v,C] += scale * -1.00000000 * np.einsum('UV,iAaV,bUiI->bAaI', eta1['b'], h['ab'][c,V,v,A], t['ab'][pv,pA,hc,hC], optimize='optimal')
	O['ab'][v,V,v,C] += scale * +1.00000000 * np.einsum('uv,uIaJ,bAvI->bAaJ', gamma1['a'], h['ab'][a,C,v,C], t['ab'][pv,pV,ha,hC], optimize='optimal')
	O['ab'][v,V,v,C] += scale * -1.00000000 * np.einsum('uv,uAaB,bBvI->bAaI', gamma1['a'], h['ab'][a,V,v,V], t['ab'][pv,pV,ha,hC], optimize='optimal')
	O['ab'][v,V,v,C] += scale * +1.00000000 * np.einsum('UV,uv,uUaI,bAvV->bAaI', gamma1['b'], gamma1['a'], h['ab'][a,A,v,C], t['ab'][pv,pV,ha,hA], optimize='optimal')
	O['ab'][v,V,v,C] += scale * +1.00000000 * np.einsum('UV,iUaI,bAiV->bAaI', gamma1['b'], h['ab'][c,A,v,C], t['ab'][pv,pV,hc,hA], optimize='optimal')
	O['ab'][v,V,v,C] += scale * +1.00000000 * np.einsum('iIaJ,bAiI->bAaJ', h['ab'][c,C,v,C], t['ab'][pv,pV,hc,hC], optimize='optimal')
	O['ab'][v,V,v,C] += scale * -1.00000000 * np.einsum('iAaB,bBiI->bAaI', h['ab'][c,V,v,V], t['ab'][pv,pV,hc,hC], optimize='optimal')
	
	O['ab'][a,A,v,V] += scale * +1.00000000 * np.einsum('uv,uIaA,wUvI->wUaA', gamma1['a'], h['ab'][a,C,v,V], t['ab'][pa,pA,ha,hC], optimize='optimal')
	
	O['ab'][a,A,v,V] += scale * +1.00000000 * np.einsum('UV,iUaA,uWiV->uWaA', gamma1['b'], h['ab'][c,A,v,V], t['ab'][pa,pA,hc,hA], optimize='optimal')
	O['ab'][a,A,v,V] += scale * +1.00000000 * np.einsum('iIaA,uUiI->uUaA', h['ab'][c,C,v,V], t['ab'][pa,pA,hc,hC], optimize='optimal')
	O['ab'][v,A,v,V] += scale * -1.00000000 * np.einsum('UV,uv,uUaA,bWvV->bWaA', eta1['b'], eta1['a'], h['ab'][a,A,v,V], t['ab'][pv,pA,ha,hA], optimize='optimal')
	O['ab'][v,A,v,V] += scale * +1.00000000 * np.einsum('uv,uIaA,bUvI->bUaA', gamma1['a'], h['ab'][a,C,v,V], t['ab'][pv,pA,ha,hC], optimize='optimal')
	O['ab'][v,A,v,V] += scale * +1.00000000 * np.einsum('UV,uv,uUaA,bWvV->bWaA', gamma1['b'], gamma1['a'], h['ab'][a,A,v,V], t['ab'][pv,pA,ha,hA], optimize='optimal')
	O['ab'][v,A,v,V] += scale * +1.00000000 * np.einsum('UV,iUaA,bWiV->bWaA', gamma1['b'], h['ab'][c,A,v,V], t['ab'][pv,pA,hc,hA], optimize='optimal')
	O['ab'][v,A,v,V] += scale * +1.00000000 * np.einsum('iIaA,bUiI->bUaA', h['ab'][c,C,v,V], t['ab'][pv,pA,hc,hC], optimize='optimal')
	O['ab'][a,V,v,V] += scale * -1.00000000 * np.einsum('UV,uv,uUaA,wBvV->wBaA', eta1['b'], eta1['a'], h['ab'][a,A,v,V], t['ab'][pa,pV,ha,hA], optimize='optimal')
	O['ab'][a,V,v,V] += scale * +1.00000000 * np.einsum('uv,uIaA,wBvI->wBaA', gamma1['a'], h['ab'][a,C,v,V], t['ab'][pa,pV,ha,hC], optimize='optimal')
	O['ab'][a,V,v,V] += scale * +1.00000000 * np.einsum('UV,uv,uUaA,wBvV->wBaA', gamma1['b'], gamma1['a'], h['ab'][a,A,v,V], t['ab'][pa,pV,ha,hA], optimize='optimal')
	O['ab'][a,V,v,V] += scale * +1.00000000 * np.einsum('UV,iUaA,uBiV->uBaA', gamma1['b'], h['ab'][c,A,v,V], t['ab'][pa,pV,hc,hA], optimize='optimal')
	O['ab'][a,V,v,V] += scale * +1.00000000 * np.einsum('iIaA,uBiI->uBaA', h['ab'][c,C,v,V], t['ab'][pa,pV,hc,hC], optimize='optimal')
	O['ab'][v,V,v,V] += scale * -1.00000000 * np.einsum('UV,uv,uUaA,bBvV->bBaA', eta1['b'], eta1['a'], h['ab'][a,A,v,V], t['ab'][pv,pV,ha,hA], optimize='optimal')
	O['ab'][v,V,v,V] += scale * +1.00000000 * np.einsum('uv,uIaA,bBvI->bBaA', gamma1['a'], h['ab'][a,C,v,V], t['ab'][pv,pV,ha,hC], optimize='optimal')
	O['ab'][v,V,v,V] += scale * +1.00000000 * np.einsum('UV,uv,uUaA,bBvV->bBaA', gamma1['b'], gamma1['a'], h['ab'][a,A,v,V], t['ab'][pv,pV,ha,hA], optimize='optimal')
	O['ab'][v,V,v,V] += scale * +1.00000000 * np.einsum('UV,iUaA,bBiV->bBaA', gamma1['b'], h['ab'][c,A,v,V], t['ab'][pv,pV,hc,hA], optimize='optimal')
	O['ab'][v,V,v,V] += scale * +1.00000000 * np.einsum('iIaA,bBiI->bBaA', h['ab'][c,C,v,V], t['ab'][pv,pV,hc,hC], optimize='optimal')

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

	
	
	O['ab'][a,A,a,A] += scale * -1.00000000 * np.einsum('UV,uIvV,WUIX->uWvX', eta1['b'], h['ab'][a,C,a,A], t['bb'][pA,pA,hC,hA], optimize='optimal')
	O['ab'][a,A,a,A] += scale * +1.00000000 * np.einsum('UV,uUvA,WAXV->uWvX', gamma1['b'], h['ab'][a,A,a,V], t['bb'][pA,pV,hA,hA], optimize='optimal')
	O['ab'][a,A,a,A] += scale * -1.00000000 * np.einsum('uIvA,UAIV->uUvV', h['ab'][a,C,a,V], t['bb'][pA,pV,hC,hA], optimize='optimal')
	
	
	O['ab'][c,A,a,A] += scale * -1.00000000 * np.einsum('UV,iIuV,WUIX->iWuX', eta1['b'], h['ab'][c,C,a,A], t['bb'][pA,pA,hC,hA], optimize='optimal')
	O['ab'][c,A,a,A] += scale * +1.00000000 * np.einsum('UV,iUuA,WAXV->iWuX', gamma1['b'], h['ab'][c,A,a,V], t['bb'][pA,pV,hA,hA], optimize='optimal')
	O['ab'][c,A,a,A] += scale * -1.00000000 * np.einsum('iIuA,UAIV->iUuV', h['ab'][c,C,a,V], t['bb'][pA,pV,hC,hA], optimize='optimal')
	
	
	O['ab'][v,A,a,A] += scale * -1.00000000 * np.einsum('UV,aIuV,WUIX->aWuX', eta1['b'], h['ab'][v,C,a,A], t['bb'][pA,pA,hC,hA], optimize='optimal')
	O['ab'][v,A,a,A] += scale * +1.00000000 * np.einsum('UV,aUuA,WAXV->aWuX', gamma1['b'], h['ab'][v,A,a,V], t['bb'][pA,pV,hA,hA], optimize='optimal')
	O['ab'][v,A,a,A] += scale * -1.00000000 * np.einsum('aIuA,UAIV->aUuV', h['ab'][v,C,a,V], t['bb'][pA,pV,hC,hA], optimize='optimal')
	O['ab'][a,V,a,A] += scale * +1.00000000 * np.einsum('UV,WX,uUvX,WAYV->uAvY', eta1['b'], gamma1['b'], h['ab'][a,A,a,A], t['bb'][pA,pV,hA,hA], optimize='optimal')
	O['ab'][a,V,a,A] += scale * -1.00000000 * np.einsum('UV,WX,uWvV,UAYX->uAvY', eta1['b'], gamma1['b'], h['ab'][a,A,a,A], t['bb'][pA,pV,hA,hA], optimize='optimal')
	O['ab'][a,V,a,A] += scale * +1.00000000 * np.einsum('UV,uIvV,UAIW->uAvW', eta1['b'], h['ab'][a,C,a,A], t['bb'][pA,pV,hC,hA], optimize='optimal')
	O['ab'][a,V,a,A] += scale * +1.00000000 * np.einsum('UV,uUvA,BAWV->uBvW', gamma1['b'], h['ab'][a,A,a,V], t['bb'][pV,pV,hA,hA], optimize='optimal')
	O['ab'][a,V,a,A] += scale * -1.00000000 * np.einsum('uIvA,BAIU->uBvU', h['ab'][a,C,a,V], t['bb'][pV,pV,hC,hA], optimize='optimal')
	O['ab'][c,V,a,A] += scale * +1.00000000 * np.einsum('UV,WX,iUuX,WAYV->iAuY', eta1['b'], gamma1['b'], h['ab'][c,A,a,A], t['bb'][pA,pV,hA,hA], optimize='optimal')
	O['ab'][c,V,a,A] += scale * -1.00000000 * np.einsum('UV,WX,iWuV,UAYX->iAuY', eta1['b'], gamma1['b'], h['ab'][c,A,a,A], t['bb'][pA,pV,hA,hA], optimize='optimal')
	O['ab'][c,V,a,A] += scale * +1.00000000 * np.einsum('UV,iIuV,UAIW->iAuW', eta1['b'], h['ab'][c,C,a,A], t['bb'][pA,pV,hC,hA], optimize='optimal')
	O['ab'][c,V,a,A] += scale * +1.00000000 * np.einsum('UV,iUuA,BAWV->iBuW', gamma1['b'], h['ab'][c,A,a,V], t['bb'][pV,pV,hA,hA], optimize='optimal')
	O['ab'][c,V,a,A] += scale * -1.00000000 * np.einsum('iIuA,BAIU->iBuU', h['ab'][c,C,a,V], t['bb'][pV,pV,hC,hA], optimize='optimal')
	O['ab'][v,V,a,A] += scale * +1.00000000 * np.einsum('UV,WX,aUuX,WAYV->aAuY', eta1['b'], gamma1['b'], h['ab'][v,A,a,A], t['bb'][pA,pV,hA,hA], optimize='optimal')
	O['ab'][v,V,a,A] += scale * -1.00000000 * np.einsum('UV,WX,aWuV,UAYX->aAuY', eta1['b'], gamma1['b'], h['ab'][v,A,a,A], t['bb'][pA,pV,hA,hA], optimize='optimal')
	O['ab'][v,V,a,A] += scale * +1.00000000 * np.einsum('UV,aIuV,UAIW->aAuW', eta1['b'], h['ab'][v,C,a,A], t['bb'][pA,pV,hC,hA], optimize='optimal')
	O['ab'][v,V,a,A] += scale * +1.00000000 * np.einsum('UV,aUuA,BAWV->aBuW', gamma1['b'], h['ab'][v,A,a,V], t['bb'][pV,pV,hA,hA], optimize='optimal')
	O['ab'][v,V,a,A] += scale * -1.00000000 * np.einsum('aIuA,BAIU->aBuU', h['ab'][v,C,a,V], t['bb'][pV,pV,hC,hA], optimize='optimal')
	O['ab'][a,A,a,C] += scale * -1.00000000 * np.einsum('UV,WX,uUvX,YWIV->uYvI', eta1['b'], gamma1['b'], h['ab'][a,A,a,A], t['bb'][pA,pA,hC,hA], optimize='optimal')
	O['ab'][a,A,a,C] += scale * +1.00000000 * np.einsum('UV,WX,uWvV,YUIX->uYvI', eta1['b'], gamma1['b'], h['ab'][a,A,a,A], t['bb'][pA,pA,hC,hA], optimize='optimal')
	O['ab'][a,A,a,C] += scale * +1.00000000 * np.einsum('UV,uIvV,WUJI->uWvJ', eta1['b'], h['ab'][a,C,a,A], t['bb'][pA,pA,hC,hC], optimize='optimal')
	O['ab'][a,A,a,C] += scale * +1.00000000 * np.einsum('UV,uUvA,WAIV->uWvI', gamma1['b'], h['ab'][a,A,a,V], t['bb'][pA,pV,hC,hA], optimize='optimal')
	O['ab'][a,A,a,C] += scale * +1.00000000 * np.einsum('uIvA,UAJI->uUvJ', h['ab'][a,C,a,V], t['bb'][pA,pV,hC,hC], optimize='optimal')
	O['ab'][c,A,a,C] += scale * -1.00000000 * np.einsum('UV,WX,iUuX,YWIV->iYuI', eta1['b'], gamma1['b'], h['ab'][c,A,a,A], t['bb'][pA,pA,hC,hA], optimize='optimal')
	O['ab'][c,A,a,C] += scale * +1.00000000 * np.einsum('UV,WX,iWuV,YUIX->iYuI', eta1['b'], gamma1['b'], h['ab'][c,A,a,A], t['bb'][pA,pA,hC,hA], optimize='optimal')
	O['ab'][c,A,a,C] += scale * +1.00000000 * np.einsum('UV,iIuV,WUJI->iWuJ', eta1['b'], h['ab'][c,C,a,A], t['bb'][pA,pA,hC,hC], optimize='optimal')
	O['ab'][c,A,a,C] += scale * +1.00000000 * np.einsum('UV,iUuA,WAIV->iWuI', gamma1['b'], h['ab'][c,A,a,V], t['bb'][pA,pV,hC,hA], optimize='optimal')
	O['ab'][c,A,a,C] += scale * +1.00000000 * np.einsum('iIuA,UAJI->iUuJ', h['ab'][c,C,a,V], t['bb'][pA,pV,hC,hC], optimize='optimal')
	O['ab'][v,A,a,C] += scale * -1.00000000 * np.einsum('UV,WX,aUuX,YWIV->aYuI', eta1['b'], gamma1['b'], h['ab'][v,A,a,A], t['bb'][pA,pA,hC,hA], optimize='optimal')
	O['ab'][v,A,a,C] += scale * +1.00000000 * np.einsum('UV,WX,aWuV,YUIX->aYuI', eta1['b'], gamma1['b'], h['ab'][v,A,a,A], t['bb'][pA,pA,hC,hA], optimize='optimal')
	O['ab'][v,A,a,C] += scale * +1.00000000 * np.einsum('UV,aIuV,WUJI->aWuJ', eta1['b'], h['ab'][v,C,a,A], t['bb'][pA,pA,hC,hC], optimize='optimal')
	O['ab'][v,A,a,C] += scale * +1.00000000 * np.einsum('UV,aUuA,WAIV->aWuI', gamma1['b'], h['ab'][v,A,a,V], t['bb'][pA,pV,hC,hA], optimize='optimal')
	O['ab'][v,A,a,C] += scale * +1.00000000 * np.einsum('aIuA,UAJI->aUuJ', h['ab'][v,C,a,V], t['bb'][pA,pV,hC,hC], optimize='optimal')
	O['ab'][a,V,a,C] += scale * +1.00000000 * np.einsum('UV,WX,uUvX,WAIV->uAvI', eta1['b'], gamma1['b'], h['ab'][a,A,a,A], t['bb'][pA,pV,hC,hA], optimize='optimal')
	O['ab'][a,V,a,C] += scale * -1.00000000 * np.einsum('UV,WX,uWvV,UAIX->uAvI', eta1['b'], gamma1['b'], h['ab'][a,A,a,A], t['bb'][pA,pV,hC,hA], optimize='optimal')
	O['ab'][a,V,a,C] += scale * -1.00000000 * np.einsum('UV,uIvV,UAJI->uAvJ', eta1['b'], h['ab'][a,C,a,A], t['bb'][pA,pV,hC,hC], optimize='optimal')
	O['ab'][a,V,a,C] += scale * +1.00000000 * np.einsum('UV,uUvA,BAIV->uBvI', gamma1['b'], h['ab'][a,A,a,V], t['bb'][pV,pV,hC,hA], optimize='optimal')
	O['ab'][a,V,a,C] += scale * +1.00000000 * np.einsum('uIvA,BAJI->uBvJ', h['ab'][a,C,a,V], t['bb'][pV,pV,hC,hC], optimize='optimal')
	O['ab'][c,V,a,C] += scale * +1.00000000 * np.einsum('UV,WX,iUuX,WAIV->iAuI', eta1['b'], gamma1['b'], h['ab'][c,A,a,A], t['bb'][pA,pV,hC,hA], optimize='optimal')
	O['ab'][c,V,a,C] += scale * -1.00000000 * np.einsum('UV,WX,iWuV,UAIX->iAuI', eta1['b'], gamma1['b'], h['ab'][c,A,a,A], t['bb'][pA,pV,hC,hA], optimize='optimal')
	O['ab'][c,V,a,C] += scale * -1.00000000 * np.einsum('UV,iIuV,UAJI->iAuJ', eta1['b'], h['ab'][c,C,a,A], t['bb'][pA,pV,hC,hC], optimize='optimal')
	O['ab'][c,V,a,C] += scale * +1.00000000 * np.einsum('UV,iUuA,BAIV->iBuI', gamma1['b'], h['ab'][c,A,a,V], t['bb'][pV,pV,hC,hA], optimize='optimal')
	O['ab'][c,V,a,C] += scale * +1.00000000 * np.einsum('iIuA,BAJI->iBuJ', h['ab'][c,C,a,V], t['bb'][pV,pV,hC,hC], optimize='optimal')
	O['ab'][v,V,a,C] += scale * +1.00000000 * np.einsum('UV,WX,aUuX,WAIV->aAuI', eta1['b'], gamma1['b'], h['ab'][v,A,a,A], t['bb'][pA,pV,hC,hA], optimize='optimal')
	O['ab'][v,V,a,C] += scale * -1.00000000 * np.einsum('UV,WX,aWuV,UAIX->aAuI', eta1['b'], gamma1['b'], h['ab'][v,A,a,A], t['bb'][pA,pV,hC,hA], optimize='optimal')
	O['ab'][v,V,a,C] += scale * -1.00000000 * np.einsum('UV,aIuV,UAJI->aAuJ', eta1['b'], h['ab'][v,C,a,A], t['bb'][pA,pV,hC,hC], optimize='optimal')
	O['ab'][v,V,a,C] += scale * +1.00000000 * np.einsum('UV,aUuA,BAIV->aBuI', gamma1['b'], h['ab'][v,A,a,V], t['bb'][pV,pV,hC,hA], optimize='optimal')
	O['ab'][v,V,a,C] += scale * +1.00000000 * np.einsum('aIuA,BAJI->aBuJ', h['ab'][v,C,a,V], t['bb'][pV,pV,hC,hC], optimize='optimal')
	
	
	O['ab'][a,A,c,A] += scale * -1.00000000 * np.einsum('UV,uIiV,WUIX->uWiX', eta1['b'], h['ab'][a,C,c,A], t['bb'][pA,pA,hC,hA], optimize='optimal')
	O['ab'][a,A,c,A] += scale * +1.00000000 * np.einsum('UV,uUiA,WAXV->uWiX', gamma1['b'], h['ab'][a,A,c,V], t['bb'][pA,pV,hA,hA], optimize='optimal')
	O['ab'][a,A,c,A] += scale * -1.00000000 * np.einsum('uIiA,UAIV->uUiV', h['ab'][a,C,c,V], t['bb'][pA,pV,hC,hA], optimize='optimal')
	
	
	O['ab'][c,A,c,A] += scale * -1.00000000 * np.einsum('UV,iIjV,WUIX->iWjX', eta1['b'], h['ab'][c,C,c,A], t['bb'][pA,pA,hC,hA], optimize='optimal')
	O['ab'][c,A,c,A] += scale * +1.00000000 * np.einsum('UV,iUjA,WAXV->iWjX', gamma1['b'], h['ab'][c,A,c,V], t['bb'][pA,pV,hA,hA], optimize='optimal')
	O['ab'][c,A,c,A] += scale * -1.00000000 * np.einsum('iIjA,UAIV->iUjV', h['ab'][c,C,c,V], t['bb'][pA,pV,hC,hA], optimize='optimal')
	
	
	O['ab'][v,A,c,A] += scale * -1.00000000 * np.einsum('UV,aIiV,WUIX->aWiX', eta1['b'], h['ab'][v,C,c,A], t['bb'][pA,pA,hC,hA], optimize='optimal')
	O['ab'][v,A,c,A] += scale * +1.00000000 * np.einsum('UV,aUiA,WAXV->aWiX', gamma1['b'], h['ab'][v,A,c,V], t['bb'][pA,pV,hA,hA], optimize='optimal')
	O['ab'][v,A,c,A] += scale * -1.00000000 * np.einsum('aIiA,UAIV->aUiV', h['ab'][v,C,c,V], t['bb'][pA,pV,hC,hA], optimize='optimal')
	O['ab'][a,V,c,A] += scale * +1.00000000 * np.einsum('UV,WX,uUiX,WAYV->uAiY', eta1['b'], gamma1['b'], h['ab'][a,A,c,A], t['bb'][pA,pV,hA,hA], optimize='optimal')
	O['ab'][a,V,c,A] += scale * -1.00000000 * np.einsum('UV,WX,uWiV,UAYX->uAiY', eta1['b'], gamma1['b'], h['ab'][a,A,c,A], t['bb'][pA,pV,hA,hA], optimize='optimal')
	O['ab'][a,V,c,A] += scale * +1.00000000 * np.einsum('UV,uIiV,UAIW->uAiW', eta1['b'], h['ab'][a,C,c,A], t['bb'][pA,pV,hC,hA], optimize='optimal')
	O['ab'][a,V,c,A] += scale * +1.00000000 * np.einsum('UV,uUiA,BAWV->uBiW', gamma1['b'], h['ab'][a,A,c,V], t['bb'][pV,pV,hA,hA], optimize='optimal')
	O['ab'][a,V,c,A] += scale * -1.00000000 * np.einsum('uIiA,BAIU->uBiU', h['ab'][a,C,c,V], t['bb'][pV,pV,hC,hA], optimize='optimal')
	O['ab'][c,V,c,A] += scale * +1.00000000 * np.einsum('UV,WX,iUjX,WAYV->iAjY', eta1['b'], gamma1['b'], h['ab'][c,A,c,A], t['bb'][pA,pV,hA,hA], optimize='optimal')
	O['ab'][c,V,c,A] += scale * -1.00000000 * np.einsum('UV,WX,iWjV,UAYX->iAjY', eta1['b'], gamma1['b'], h['ab'][c,A,c,A], t['bb'][pA,pV,hA,hA], optimize='optimal')
	O['ab'][c,V,c,A] += scale * +1.00000000 * np.einsum('UV,iIjV,UAIW->iAjW', eta1['b'], h['ab'][c,C,c,A], t['bb'][pA,pV,hC,hA], optimize='optimal')
	O['ab'][c,V,c,A] += scale * +1.00000000 * np.einsum('UV,iUjA,BAWV->iBjW', gamma1['b'], h['ab'][c,A,c,V], t['bb'][pV,pV,hA,hA], optimize='optimal')
	O['ab'][c,V,c,A] += scale * -1.00000000 * np.einsum('iIjA,BAIU->iBjU', h['ab'][c,C,c,V], t['bb'][pV,pV,hC,hA], optimize='optimal')
	O['ab'][v,V,c,A] += scale * +1.00000000 * np.einsum('UV,WX,aUiX,WAYV->aAiY', eta1['b'], gamma1['b'], h['ab'][v,A,c,A], t['bb'][pA,pV,hA,hA], optimize='optimal')
	O['ab'][v,V,c,A] += scale * -1.00000000 * np.einsum('UV,WX,aWiV,UAYX->aAiY', eta1['b'], gamma1['b'], h['ab'][v,A,c,A], t['bb'][pA,pV,hA,hA], optimize='optimal')
	O['ab'][v,V,c,A] += scale * +1.00000000 * np.einsum('UV,aIiV,UAIW->aAiW', eta1['b'], h['ab'][v,C,c,A], t['bb'][pA,pV,hC,hA], optimize='optimal')
	O['ab'][v,V,c,A] += scale * +1.00000000 * np.einsum('UV,aUiA,BAWV->aBiW', gamma1['b'], h['ab'][v,A,c,V], t['bb'][pV,pV,hA,hA], optimize='optimal')
	O['ab'][v,V,c,A] += scale * -1.00000000 * np.einsum('aIiA,BAIU->aBiU', h['ab'][v,C,c,V], t['bb'][pV,pV,hC,hA], optimize='optimal')
	O['ab'][a,A,c,C] += scale * -1.00000000 * np.einsum('UV,WX,uUiX,YWIV->uYiI', eta1['b'], gamma1['b'], h['ab'][a,A,c,A], t['bb'][pA,pA,hC,hA], optimize='optimal')
	O['ab'][a,A,c,C] += scale * +1.00000000 * np.einsum('UV,WX,uWiV,YUIX->uYiI', eta1['b'], gamma1['b'], h['ab'][a,A,c,A], t['bb'][pA,pA,hC,hA], optimize='optimal')
	O['ab'][a,A,c,C] += scale * +1.00000000 * np.einsum('UV,uIiV,WUJI->uWiJ', eta1['b'], h['ab'][a,C,c,A], t['bb'][pA,pA,hC,hC], optimize='optimal')
	O['ab'][a,A,c,C] += scale * +1.00000000 * np.einsum('UV,uUiA,WAIV->uWiI', gamma1['b'], h['ab'][a,A,c,V], t['bb'][pA,pV,hC,hA], optimize='optimal')
	O['ab'][a,A,c,C] += scale * +1.00000000 * np.einsum('uIiA,UAJI->uUiJ', h['ab'][a,C,c,V], t['bb'][pA,pV,hC,hC], optimize='optimal')
	O['ab'][c,A,c,C] += scale * -1.00000000 * np.einsum('UV,WX,iUjX,YWIV->iYjI', eta1['b'], gamma1['b'], h['ab'][c,A,c,A], t['bb'][pA,pA,hC,hA], optimize='optimal')
	O['ab'][c,A,c,C] += scale * +1.00000000 * np.einsum('UV,WX,iWjV,YUIX->iYjI', eta1['b'], gamma1['b'], h['ab'][c,A,c,A], t['bb'][pA,pA,hC,hA], optimize='optimal')
	O['ab'][c,A,c,C] += scale * +1.00000000 * np.einsum('UV,iIjV,WUJI->iWjJ', eta1['b'], h['ab'][c,C,c,A], t['bb'][pA,pA,hC,hC], optimize='optimal')
	O['ab'][c,A,c,C] += scale * +1.00000000 * np.einsum('UV,iUjA,WAIV->iWjI', gamma1['b'], h['ab'][c,A,c,V], t['bb'][pA,pV,hC,hA], optimize='optimal')
	O['ab'][c,A,c,C] += scale * +1.00000000 * np.einsum('iIjA,UAJI->iUjJ', h['ab'][c,C,c,V], t['bb'][pA,pV,hC,hC], optimize='optimal')
	O['ab'][v,A,c,C] += scale * -1.00000000 * np.einsum('UV,WX,aUiX,YWIV->aYiI', eta1['b'], gamma1['b'], h['ab'][v,A,c,A], t['bb'][pA,pA,hC,hA], optimize='optimal')
	O['ab'][v,A,c,C] += scale * +1.00000000 * np.einsum('UV,WX,aWiV,YUIX->aYiI', eta1['b'], gamma1['b'], h['ab'][v,A,c,A], t['bb'][pA,pA,hC,hA], optimize='optimal')
	O['ab'][v,A,c,C] += scale * +1.00000000 * np.einsum('UV,aIiV,WUJI->aWiJ', eta1['b'], h['ab'][v,C,c,A], t['bb'][pA,pA,hC,hC], optimize='optimal')
	O['ab'][v,A,c,C] += scale * +1.00000000 * np.einsum('UV,aUiA,WAIV->aWiI', gamma1['b'], h['ab'][v,A,c,V], t['bb'][pA,pV,hC,hA], optimize='optimal')
	O['ab'][v,A,c,C] += scale * +1.00000000 * np.einsum('aIiA,UAJI->aUiJ', h['ab'][v,C,c,V], t['bb'][pA,pV,hC,hC], optimize='optimal')
	O['ab'][a,V,c,C] += scale * +1.00000000 * np.einsum('UV,WX,uUiX,WAIV->uAiI', eta1['b'], gamma1['b'], h['ab'][a,A,c,A], t['bb'][pA,pV,hC,hA], optimize='optimal')
	O['ab'][a,V,c,C] += scale * -1.00000000 * np.einsum('UV,WX,uWiV,UAIX->uAiI', eta1['b'], gamma1['b'], h['ab'][a,A,c,A], t['bb'][pA,pV,hC,hA], optimize='optimal')
	O['ab'][a,V,c,C] += scale * -1.00000000 * np.einsum('UV,uIiV,UAJI->uAiJ', eta1['b'], h['ab'][a,C,c,A], t['bb'][pA,pV,hC,hC], optimize='optimal')
	O['ab'][a,V,c,C] += scale * +1.00000000 * np.einsum('UV,uUiA,BAIV->uBiI', gamma1['b'], h['ab'][a,A,c,V], t['bb'][pV,pV,hC,hA], optimize='optimal')
	O['ab'][a,V,c,C] += scale * +1.00000000 * np.einsum('uIiA,BAJI->uBiJ', h['ab'][a,C,c,V], t['bb'][pV,pV,hC,hC], optimize='optimal')
	O['ab'][c,V,c,C] += scale * +1.00000000 * np.einsum('UV,WX,iUjX,WAIV->iAjI', eta1['b'], gamma1['b'], h['ab'][c,A,c,A], t['bb'][pA,pV,hC,hA], optimize='optimal')
	O['ab'][c,V,c,C] += scale * -1.00000000 * np.einsum('UV,WX,iWjV,UAIX->iAjI', eta1['b'], gamma1['b'], h['ab'][c,A,c,A], t['bb'][pA,pV,hC,hA], optimize='optimal')
	O['ab'][c,V,c,C] += scale * -1.00000000 * np.einsum('UV,iIjV,UAJI->iAjJ', eta1['b'], h['ab'][c,C,c,A], t['bb'][pA,pV,hC,hC], optimize='optimal')
	O['ab'][c,V,c,C] += scale * +1.00000000 * np.einsum('UV,iUjA,BAIV->iBjI', gamma1['b'], h['ab'][c,A,c,V], t['bb'][pV,pV,hC,hA], optimize='optimal')
	O['ab'][c,V,c,C] += scale * +1.00000000 * np.einsum('iIjA,BAJI->iBjJ', h['ab'][c,C,c,V], t['bb'][pV,pV,hC,hC], optimize='optimal')
	O['ab'][v,V,c,C] += scale * +1.00000000 * np.einsum('UV,WX,aUiX,WAIV->aAiI', eta1['b'], gamma1['b'], h['ab'][v,A,c,A], t['bb'][pA,pV,hC,hA], optimize='optimal')
	O['ab'][v,V,c,C] += scale * -1.00000000 * np.einsum('UV,WX,aWiV,UAIX->aAiI', eta1['b'], gamma1['b'], h['ab'][v,A,c,A], t['bb'][pA,pV,hC,hA], optimize='optimal')
	O['ab'][v,V,c,C] += scale * -1.00000000 * np.einsum('UV,aIiV,UAJI->aAiJ', eta1['b'], h['ab'][v,C,c,A], t['bb'][pA,pV,hC,hC], optimize='optimal')
	O['ab'][v,V,c,C] += scale * +1.00000000 * np.einsum('UV,aUiA,BAIV->aBiI', gamma1['b'], h['ab'][v,A,c,V], t['bb'][pV,pV,hC,hA], optimize='optimal')
	O['ab'][v,V,c,C] += scale * +1.00000000 * np.einsum('aIiA,BAJI->aBiJ', h['ab'][v,C,c,V], t['bb'][pV,pV,hC,hC], optimize='optimal')
	
	
	O['ab'][a,A,v,A] += scale * -1.00000000 * np.einsum('UV,uIaV,WUIX->uWaX', eta1['b'], h['ab'][a,C,v,A], t['bb'][pA,pA,hC,hA], optimize='optimal')
	O['ab'][a,A,v,A] += scale * +1.00000000 * np.einsum('UV,uUaA,WAXV->uWaX', gamma1['b'], h['ab'][a,A,v,V], t['bb'][pA,pV,hA,hA], optimize='optimal')
	O['ab'][a,A,v,A] += scale * -1.00000000 * np.einsum('uIaA,UAIV->uUaV', h['ab'][a,C,v,V], t['bb'][pA,pV,hC,hA], optimize='optimal')
	
	
	O['ab'][c,A,v,A] += scale * -1.00000000 * np.einsum('UV,iIaV,WUIX->iWaX', eta1['b'], h['ab'][c,C,v,A], t['bb'][pA,pA,hC,hA], optimize='optimal')
	O['ab'][c,A,v,A] += scale * +1.00000000 * np.einsum('UV,iUaA,WAXV->iWaX', gamma1['b'], h['ab'][c,A,v,V], t['bb'][pA,pV,hA,hA], optimize='optimal')
	O['ab'][c,A,v,A] += scale * -1.00000000 * np.einsum('iIaA,UAIV->iUaV', h['ab'][c,C,v,V], t['bb'][pA,pV,hC,hA], optimize='optimal')
	
	
	O['ab'][v,A,v,A] += scale * -1.00000000 * np.einsum('UV,aIbV,WUIX->aWbX', eta1['b'], h['ab'][v,C,v,A], t['bb'][pA,pA,hC,hA], optimize='optimal')
	O['ab'][v,A,v,A] += scale * +1.00000000 * np.einsum('UV,aUbA,WAXV->aWbX', gamma1['b'], h['ab'][v,A,v,V], t['bb'][pA,pV,hA,hA], optimize='optimal')
	O['ab'][v,A,v,A] += scale * -1.00000000 * np.einsum('aIbA,UAIV->aUbV', h['ab'][v,C,v,V], t['bb'][pA,pV,hC,hA], optimize='optimal')
	O['ab'][a,V,v,A] += scale * +1.00000000 * np.einsum('UV,WX,uUaX,WAYV->uAaY', eta1['b'], gamma1['b'], h['ab'][a,A,v,A], t['bb'][pA,pV,hA,hA], optimize='optimal')
	O['ab'][a,V,v,A] += scale * -1.00000000 * np.einsum('UV,WX,uWaV,UAYX->uAaY', eta1['b'], gamma1['b'], h['ab'][a,A,v,A], t['bb'][pA,pV,hA,hA], optimize='optimal')
	O['ab'][a,V,v,A] += scale * +1.00000000 * np.einsum('UV,uIaV,UAIW->uAaW', eta1['b'], h['ab'][a,C,v,A], t['bb'][pA,pV,hC,hA], optimize='optimal')
	O['ab'][a,V,v,A] += scale * +1.00000000 * np.einsum('UV,uUaA,BAWV->uBaW', gamma1['b'], h['ab'][a,A,v,V], t['bb'][pV,pV,hA,hA], optimize='optimal')
	O['ab'][a,V,v,A] += scale * -1.00000000 * np.einsum('uIaA,BAIU->uBaU', h['ab'][a,C,v,V], t['bb'][pV,pV,hC,hA], optimize='optimal')
	O['ab'][c,V,v,A] += scale * +1.00000000 * np.einsum('UV,WX,iUaX,WAYV->iAaY', eta1['b'], gamma1['b'], h['ab'][c,A,v,A], t['bb'][pA,pV,hA,hA], optimize='optimal')
	O['ab'][c,V,v,A] += scale * -1.00000000 * np.einsum('UV,WX,iWaV,UAYX->iAaY', eta1['b'], gamma1['b'], h['ab'][c,A,v,A], t['bb'][pA,pV,hA,hA], optimize='optimal')
	O['ab'][c,V,v,A] += scale * +1.00000000 * np.einsum('UV,iIaV,UAIW->iAaW', eta1['b'], h['ab'][c,C,v,A], t['bb'][pA,pV,hC,hA], optimize='optimal')
	O['ab'][c,V,v,A] += scale * +1.00000000 * np.einsum('UV,iUaA,BAWV->iBaW', gamma1['b'], h['ab'][c,A,v,V], t['bb'][pV,pV,hA,hA], optimize='optimal')
	O['ab'][c,V,v,A] += scale * -1.00000000 * np.einsum('iIaA,BAIU->iBaU', h['ab'][c,C,v,V], t['bb'][pV,pV,hC,hA], optimize='optimal')
	O['ab'][v,V,v,A] += scale * +1.00000000 * np.einsum('UV,WX,aUbX,WAYV->aAbY', eta1['b'], gamma1['b'], h['ab'][v,A,v,A], t['bb'][pA,pV,hA,hA], optimize='optimal')
	O['ab'][v,V,v,A] += scale * -1.00000000 * np.einsum('UV,WX,aWbV,UAYX->aAbY', eta1['b'], gamma1['b'], h['ab'][v,A,v,A], t['bb'][pA,pV,hA,hA], optimize='optimal')
	O['ab'][v,V,v,A] += scale * +1.00000000 * np.einsum('UV,aIbV,UAIW->aAbW', eta1['b'], h['ab'][v,C,v,A], t['bb'][pA,pV,hC,hA], optimize='optimal')
	O['ab'][v,V,v,A] += scale * +1.00000000 * np.einsum('UV,aUbA,BAWV->aBbW', gamma1['b'], h['ab'][v,A,v,V], t['bb'][pV,pV,hA,hA], optimize='optimal')
	O['ab'][v,V,v,A] += scale * -1.00000000 * np.einsum('aIbA,BAIU->aBbU', h['ab'][v,C,v,V], t['bb'][pV,pV,hC,hA], optimize='optimal')
	O['ab'][a,A,v,C] += scale * -1.00000000 * np.einsum('UV,WX,uUaX,YWIV->uYaI', eta1['b'], gamma1['b'], h['ab'][a,A,v,A], t['bb'][pA,pA,hC,hA], optimize='optimal')
	O['ab'][a,A,v,C] += scale * +1.00000000 * np.einsum('UV,WX,uWaV,YUIX->uYaI', eta1['b'], gamma1['b'], h['ab'][a,A,v,A], t['bb'][pA,pA,hC,hA], optimize='optimal')
	O['ab'][a,A,v,C] += scale * +1.00000000 * np.einsum('UV,uIaV,WUJI->uWaJ', eta1['b'], h['ab'][a,C,v,A], t['bb'][pA,pA,hC,hC], optimize='optimal')
	O['ab'][a,A,v,C] += scale * +1.00000000 * np.einsum('UV,uUaA,WAIV->uWaI', gamma1['b'], h['ab'][a,A,v,V], t['bb'][pA,pV,hC,hA], optimize='optimal')
	O['ab'][a,A,v,C] += scale * +1.00000000 * np.einsum('uIaA,UAJI->uUaJ', h['ab'][a,C,v,V], t['bb'][pA,pV,hC,hC], optimize='optimal')
	O['ab'][c,A,v,C] += scale * -1.00000000 * np.einsum('UV,WX,iUaX,YWIV->iYaI', eta1['b'], gamma1['b'], h['ab'][c,A,v,A], t['bb'][pA,pA,hC,hA], optimize='optimal')
	O['ab'][c,A,v,C] += scale * +1.00000000 * np.einsum('UV,WX,iWaV,YUIX->iYaI', eta1['b'], gamma1['b'], h['ab'][c,A,v,A], t['bb'][pA,pA,hC,hA], optimize='optimal')
	O['ab'][c,A,v,C] += scale * +1.00000000 * np.einsum('UV,iIaV,WUJI->iWaJ', eta1['b'], h['ab'][c,C,v,A], t['bb'][pA,pA,hC,hC], optimize='optimal')
	O['ab'][c,A,v,C] += scale * +1.00000000 * np.einsum('UV,iUaA,WAIV->iWaI', gamma1['b'], h['ab'][c,A,v,V], t['bb'][pA,pV,hC,hA], optimize='optimal')
	O['ab'][c,A,v,C] += scale * +1.00000000 * np.einsum('iIaA,UAJI->iUaJ', h['ab'][c,C,v,V], t['bb'][pA,pV,hC,hC], optimize='optimal')
	O['ab'][v,A,v,C] += scale * -1.00000000 * np.einsum('UV,WX,aUbX,YWIV->aYbI', eta1['b'], gamma1['b'], h['ab'][v,A,v,A], t['bb'][pA,pA,hC,hA], optimize='optimal')
	O['ab'][v,A,v,C] += scale * +1.00000000 * np.einsum('UV,WX,aWbV,YUIX->aYbI', eta1['b'], gamma1['b'], h['ab'][v,A,v,A], t['bb'][pA,pA,hC,hA], optimize='optimal')
	O['ab'][v,A,v,C] += scale * +1.00000000 * np.einsum('UV,aIbV,WUJI->aWbJ', eta1['b'], h['ab'][v,C,v,A], t['bb'][pA,pA,hC,hC], optimize='optimal')
	O['ab'][v,A,v,C] += scale * +1.00000000 * np.einsum('UV,aUbA,WAIV->aWbI', gamma1['b'], h['ab'][v,A,v,V], t['bb'][pA,pV,hC,hA], optimize='optimal')
	O['ab'][v,A,v,C] += scale * +1.00000000 * np.einsum('aIbA,UAJI->aUbJ', h['ab'][v,C,v,V], t['bb'][pA,pV,hC,hC], optimize='optimal')
	O['ab'][a,V,v,C] += scale * +1.00000000 * np.einsum('UV,WX,uUaX,WAIV->uAaI', eta1['b'], gamma1['b'], h['ab'][a,A,v,A], t['bb'][pA,pV,hC,hA], optimize='optimal')
	O['ab'][a,V,v,C] += scale * -1.00000000 * np.einsum('UV,WX,uWaV,UAIX->uAaI', eta1['b'], gamma1['b'], h['ab'][a,A,v,A], t['bb'][pA,pV,hC,hA], optimize='optimal')
	O['ab'][a,V,v,C] += scale * -1.00000000 * np.einsum('UV,uIaV,UAJI->uAaJ', eta1['b'], h['ab'][a,C,v,A], t['bb'][pA,pV,hC,hC], optimize='optimal')
	O['ab'][a,V,v,C] += scale * +1.00000000 * np.einsum('UV,uUaA,BAIV->uBaI', gamma1['b'], h['ab'][a,A,v,V], t['bb'][pV,pV,hC,hA], optimize='optimal')
	O['ab'][a,V,v,C] += scale * +1.00000000 * np.einsum('uIaA,BAJI->uBaJ', h['ab'][a,C,v,V], t['bb'][pV,pV,hC,hC], optimize='optimal')
	O['ab'][c,V,v,C] += scale * +1.00000000 * np.einsum('UV,WX,iUaX,WAIV->iAaI', eta1['b'], gamma1['b'], h['ab'][c,A,v,A], t['bb'][pA,pV,hC,hA], optimize='optimal')
	O['ab'][c,V,v,C] += scale * -1.00000000 * np.einsum('UV,WX,iWaV,UAIX->iAaI', eta1['b'], gamma1['b'], h['ab'][c,A,v,A], t['bb'][pA,pV,hC,hA], optimize='optimal')
	O['ab'][c,V,v,C] += scale * -1.00000000 * np.einsum('UV,iIaV,UAJI->iAaJ', eta1['b'], h['ab'][c,C,v,A], t['bb'][pA,pV,hC,hC], optimize='optimal')
	O['ab'][c,V,v,C] += scale * +1.00000000 * np.einsum('UV,iUaA,BAIV->iBaI', gamma1['b'], h['ab'][c,A,v,V], t['bb'][pV,pV,hC,hA], optimize='optimal')
	O['ab'][c,V,v,C] += scale * +1.00000000 * np.einsum('iIaA,BAJI->iBaJ', h['ab'][c,C,v,V], t['bb'][pV,pV,hC,hC], optimize='optimal')
	O['ab'][v,V,v,C] += scale * +1.00000000 * np.einsum('UV,WX,aUbX,WAIV->aAbI', eta1['b'], gamma1['b'], h['ab'][v,A,v,A], t['bb'][pA,pV,hC,hA], optimize='optimal')
	O['ab'][v,V,v,C] += scale * -1.00000000 * np.einsum('UV,WX,aWbV,UAIX->aAbI', eta1['b'], gamma1['b'], h['ab'][v,A,v,A], t['bb'][pA,pV,hC,hA], optimize='optimal')
	O['ab'][v,V,v,C] += scale * -1.00000000 * np.einsum('UV,aIbV,UAJI->aAbJ', eta1['b'], h['ab'][v,C,v,A], t['bb'][pA,pV,hC,hC], optimize='optimal')
	O['ab'][v,V,v,C] += scale * +1.00000000 * np.einsum('UV,aUbA,BAIV->aBbI', gamma1['b'], h['ab'][v,A,v,V], t['bb'][pV,pV,hC,hA], optimize='optimal')
	O['ab'][v,V,v,C] += scale * +1.00000000 * np.einsum('aIbA,BAJI->aBbJ', h['ab'][v,C,v,V], t['bb'][pV,pV,hC,hC], optimize='optimal')

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

	
	
	O['ab'][a,A,a,A] += scale * -1.00000000 * np.einsum('UV,IWXV,uUvI->uWvX', eta1['b'], h['bb'][C,A,A,A], t['ab'][pa,pA,ha,hC], optimize='optimal')
	O['ab'][a,A,a,A] += scale * +1.00000000 * np.einsum('UV,WUXA,uAvV->uWvX', gamma1['b'], h['bb'][A,A,A,V], t['ab'][pa,pV,ha,hA], optimize='optimal')
	O['ab'][a,A,a,A] += scale * -1.00000000 * np.einsum('IUVA,uAvI->uUvV', h['bb'][C,A,A,V], t['ab'][pa,pV,ha,hC], optimize='optimal')
	O['ab'][v,A,a,A] += scale * -1.00000000 * np.einsum('UV,WX,YUZX,aWuV->aYuZ', eta1['b'], gamma1['b'], h['bb'][A,A,A,A], t['ab'][pv,pA,ha,hA], optimize='optimal')
	O['ab'][v,A,a,A] += scale * +1.00000000 * np.einsum('UV,WX,YWZV,aUuX->aYuZ', eta1['b'], gamma1['b'], h['bb'][A,A,A,A], t['ab'][pv,pA,ha,hA], optimize='optimal')
	O['ab'][v,A,a,A] += scale * -1.00000000 * np.einsum('UV,IWXV,aUuI->aWuX', eta1['b'], h['bb'][C,A,A,A], t['ab'][pv,pA,ha,hC], optimize='optimal')
	O['ab'][v,A,a,A] += scale * +1.00000000 * np.einsum('UV,WUXA,aAuV->aWuX', gamma1['b'], h['bb'][A,A,A,V], t['ab'][pv,pV,ha,hA], optimize='optimal')
	O['ab'][v,A,a,A] += scale * -1.00000000 * np.einsum('IUVA,aAuI->aUuV', h['bb'][C,A,A,V], t['ab'][pv,pV,ha,hC], optimize='optimal')
	
	
	O['ab'][a,C,a,A] += scale * +1.00000000 * np.einsum('UV,IJWV,uUvJ->uIvW', eta1['b'], h['bb'][C,C,A,A], t['ab'][pa,pA,ha,hC], optimize='optimal')
	O['ab'][a,C,a,A] += scale * +1.00000000 * np.einsum('UV,IUWA,uAvV->uIvW', gamma1['b'], h['bb'][C,A,A,V], t['ab'][pa,pV,ha,hA], optimize='optimal')
	O['ab'][a,C,a,A] += scale * +1.00000000 * np.einsum('IJUA,uAvJ->uIvU', h['bb'][C,C,A,V], t['ab'][pa,pV,ha,hC], optimize='optimal')
	O['ab'][v,C,a,A] += scale * -1.00000000 * np.einsum('UV,WX,IUYX,aWuV->aIuY', eta1['b'], gamma1['b'], h['bb'][C,A,A,A], t['ab'][pv,pA,ha,hA], optimize='optimal')
	O['ab'][v,C,a,A] += scale * +1.00000000 * np.einsum('UV,WX,IWYV,aUuX->aIuY', eta1['b'], gamma1['b'], h['bb'][C,A,A,A], t['ab'][pv,pA,ha,hA], optimize='optimal')
	O['ab'][v,C,a,A] += scale * +1.00000000 * np.einsum('UV,IJWV,aUuJ->aIuW', eta1['b'], h['bb'][C,C,A,A], t['ab'][pv,pA,ha,hC], optimize='optimal')
	O['ab'][v,C,a,A] += scale * +1.00000000 * np.einsum('UV,IUWA,aAuV->aIuW', gamma1['b'], h['bb'][C,A,A,V], t['ab'][pv,pV,ha,hA], optimize='optimal')
	O['ab'][v,C,a,A] += scale * +1.00000000 * np.einsum('IJUA,aAuJ->aIuU', h['bb'][C,C,A,V], t['ab'][pv,pV,ha,hC], optimize='optimal')
	
	
	O['ab'][a,V,a,A] += scale * -1.00000000 * np.einsum('UV,IAWV,uUvI->uAvW', eta1['b'], h['bb'][C,V,A,A], t['ab'][pa,pA,ha,hC], optimize='optimal')
	O['ab'][a,V,a,A] += scale * -1.00000000 * np.einsum('UV,UAWB,uBvV->uAvW', gamma1['b'], h['bb'][A,V,A,V], t['ab'][pa,pV,ha,hA], optimize='optimal')
	O['ab'][a,V,a,A] += scale * -1.00000000 * np.einsum('IAUB,uBvI->uAvU', h['bb'][C,V,A,V], t['ab'][pa,pV,ha,hC], optimize='optimal')
	O['ab'][v,V,a,A] += scale * +1.00000000 * np.einsum('UV,WX,UAYX,aWuV->aAuY', eta1['b'], gamma1['b'], h['bb'][A,V,A,A], t['ab'][pv,pA,ha,hA], optimize='optimal')
	O['ab'][v,V,a,A] += scale * -1.00000000 * np.einsum('UV,WX,WAYV,aUuX->aAuY', eta1['b'], gamma1['b'], h['bb'][A,V,A,A], t['ab'][pv,pA,ha,hA], optimize='optimal')
	O['ab'][v,V,a,A] += scale * -1.00000000 * np.einsum('UV,IAWV,aUuI->aAuW', eta1['b'], h['bb'][C,V,A,A], t['ab'][pv,pA,ha,hC], optimize='optimal')
	O['ab'][v,V,a,A] += scale * -1.00000000 * np.einsum('UV,UAWB,aBuV->aAuW', gamma1['b'], h['bb'][A,V,A,V], t['ab'][pv,pV,ha,hA], optimize='optimal')
	O['ab'][v,V,a,A] += scale * -1.00000000 * np.einsum('IAUB,aBuI->aAuU', h['bb'][C,V,A,V], t['ab'][pv,pV,ha,hC], optimize='optimal')
	
	
	O['ab'][a,A,a,C] += scale * -1.00000000 * np.einsum('UV,IWJV,uUvI->uWvJ', eta1['b'], h['bb'][C,A,C,A], t['ab'][pa,pA,ha,hC], optimize='optimal')
	O['ab'][a,A,a,C] += scale * +1.00000000 * np.einsum('UV,WUIA,uAvV->uWvI', gamma1['b'], h['bb'][A,A,C,V], t['ab'][pa,pV,ha,hA], optimize='optimal')
	O['ab'][a,A,a,C] += scale * -1.00000000 * np.einsum('IUJA,uAvI->uUvJ', h['bb'][C,A,C,V], t['ab'][pa,pV,ha,hC], optimize='optimal')
	O['ab'][v,A,a,C] += scale * -1.00000000 * np.einsum('UV,WX,YUIX,aWuV->aYuI', eta1['b'], gamma1['b'], h['bb'][A,A,C,A], t['ab'][pv,pA,ha,hA], optimize='optimal')
	O['ab'][v,A,a,C] += scale * +1.00000000 * np.einsum('UV,WX,YWIV,aUuX->aYuI', eta1['b'], gamma1['b'], h['bb'][A,A,C,A], t['ab'][pv,pA,ha,hA], optimize='optimal')
	O['ab'][v,A,a,C] += scale * -1.00000000 * np.einsum('UV,IWJV,aUuI->aWuJ', eta1['b'], h['bb'][C,A,C,A], t['ab'][pv,pA,ha,hC], optimize='optimal')
	O['ab'][v,A,a,C] += scale * +1.00000000 * np.einsum('UV,WUIA,aAuV->aWuI', gamma1['b'], h['bb'][A,A,C,V], t['ab'][pv,pV,ha,hA], optimize='optimal')
	O['ab'][v,A,a,C] += scale * -1.00000000 * np.einsum('IUJA,aAuI->aUuJ', h['bb'][C,A,C,V], t['ab'][pv,pV,ha,hC], optimize='optimal')
	
	
	O['ab'][a,C,a,C] += scale * +1.00000000 * np.einsum('UV,IJKV,uUvJ->uIvK', eta1['b'], h['bb'][C,C,C,A], t['ab'][pa,pA,ha,hC], optimize='optimal')
	O['ab'][a,C,a,C] += scale * +1.00000000 * np.einsum('UV,IUJA,uAvV->uIvJ', gamma1['b'], h['bb'][C,A,C,V], t['ab'][pa,pV,ha,hA], optimize='optimal')
	O['ab'][a,C,a,C] += scale * +1.00000000 * np.einsum('IJKA,uAvJ->uIvK', h['bb'][C,C,C,V], t['ab'][pa,pV,ha,hC], optimize='optimal')
	O['ab'][v,C,a,C] += scale * -1.00000000 * np.einsum('UV,WX,IUJX,aWuV->aIuJ', eta1['b'], gamma1['b'], h['bb'][C,A,C,A], t['ab'][pv,pA,ha,hA], optimize='optimal')
	O['ab'][v,C,a,C] += scale * +1.00000000 * np.einsum('UV,WX,IWJV,aUuX->aIuJ', eta1['b'], gamma1['b'], h['bb'][C,A,C,A], t['ab'][pv,pA,ha,hA], optimize='optimal')
	O['ab'][v,C,a,C] += scale * +1.00000000 * np.einsum('UV,IJKV,aUuJ->aIuK', eta1['b'], h['bb'][C,C,C,A], t['ab'][pv,pA,ha,hC], optimize='optimal')
	O['ab'][v,C,a,C] += scale * +1.00000000 * np.einsum('UV,IUJA,aAuV->aIuJ', gamma1['b'], h['bb'][C,A,C,V], t['ab'][pv,pV,ha,hA], optimize='optimal')
	O['ab'][v,C,a,C] += scale * +1.00000000 * np.einsum('IJKA,aAuJ->aIuK', h['bb'][C,C,C,V], t['ab'][pv,pV,ha,hC], optimize='optimal')
	
	
	O['ab'][a,V,a,C] += scale * -1.00000000 * np.einsum('UV,IAJV,uUvI->uAvJ', eta1['b'], h['bb'][C,V,C,A], t['ab'][pa,pA,ha,hC], optimize='optimal')
	O['ab'][a,V,a,C] += scale * -1.00000000 * np.einsum('UV,UAIB,uBvV->uAvI', gamma1['b'], h['bb'][A,V,C,V], t['ab'][pa,pV,ha,hA], optimize='optimal')
	O['ab'][a,V,a,C] += scale * -1.00000000 * np.einsum('IAJB,uBvI->uAvJ', h['bb'][C,V,C,V], t['ab'][pa,pV,ha,hC], optimize='optimal')
	O['ab'][v,V,a,C] += scale * +1.00000000 * np.einsum('UV,WX,UAIX,aWuV->aAuI', eta1['b'], gamma1['b'], h['bb'][A,V,C,A], t['ab'][pv,pA,ha,hA], optimize='optimal')
	O['ab'][v,V,a,C] += scale * -1.00000000 * np.einsum('UV,WX,WAIV,aUuX->aAuI', eta1['b'], gamma1['b'], h['bb'][A,V,C,A], t['ab'][pv,pA,ha,hA], optimize='optimal')
	O['ab'][v,V,a,C] += scale * -1.00000000 * np.einsum('UV,IAJV,aUuI->aAuJ', eta1['b'], h['bb'][C,V,C,A], t['ab'][pv,pA,ha,hC], optimize='optimal')
	O['ab'][v,V,a,C] += scale * -1.00000000 * np.einsum('UV,UAIB,aBuV->aAuI', gamma1['b'], h['bb'][A,V,C,V], t['ab'][pv,pV,ha,hA], optimize='optimal')
	O['ab'][v,V,a,C] += scale * -1.00000000 * np.einsum('IAJB,aBuI->aAuJ', h['bb'][C,V,C,V], t['ab'][pv,pV,ha,hC], optimize='optimal')
	
	
	O['ab'][a,A,a,V] += scale * +1.00000000 * np.einsum('UV,IWVA,uUvI->uWvA', eta1['b'], h['bb'][C,A,A,V], t['ab'][pa,pA,ha,hC], optimize='optimal')
	O['ab'][a,A,a,V] += scale * +1.00000000 * np.einsum('UV,WUAB,uBvV->uWvA', gamma1['b'], h['bb'][A,A,V,V], t['ab'][pa,pV,ha,hA], optimize='optimal')
	O['ab'][a,A,a,V] += scale * -1.00000000 * np.einsum('IUAB,uBvI->uUvA', h['bb'][C,A,V,V], t['ab'][pa,pV,ha,hC], optimize='optimal')
	O['ab'][v,A,a,V] += scale * +1.00000000 * np.einsum('UV,WX,YUXA,aWuV->aYuA', eta1['b'], gamma1['b'], h['bb'][A,A,A,V], t['ab'][pv,pA,ha,hA], optimize='optimal')
	O['ab'][v,A,a,V] += scale * -1.00000000 * np.einsum('UV,WX,YWVA,aUuX->aYuA', eta1['b'], gamma1['b'], h['bb'][A,A,A,V], t['ab'][pv,pA,ha,hA], optimize='optimal')
	O['ab'][v,A,a,V] += scale * +1.00000000 * np.einsum('UV,IWVA,aUuI->aWuA', eta1['b'], h['bb'][C,A,A,V], t['ab'][pv,pA,ha,hC], optimize='optimal')
	O['ab'][v,A,a,V] += scale * +1.00000000 * np.einsum('UV,WUAB,aBuV->aWuA', gamma1['b'], h['bb'][A,A,V,V], t['ab'][pv,pV,ha,hA], optimize='optimal')
	O['ab'][v,A,a,V] += scale * -1.00000000 * np.einsum('IUAB,aBuI->aUuA', h['bb'][C,A,V,V], t['ab'][pv,pV,ha,hC], optimize='optimal')
	
	
	O['ab'][a,C,a,V] += scale * -1.00000000 * np.einsum('UV,IJVA,uUvJ->uIvA', eta1['b'], h['bb'][C,C,A,V], t['ab'][pa,pA,ha,hC], optimize='optimal')
	O['ab'][a,C,a,V] += scale * +1.00000000 * np.einsum('UV,IUAB,uBvV->uIvA', gamma1['b'], h['bb'][C,A,V,V], t['ab'][pa,pV,ha,hA], optimize='optimal')
	O['ab'][a,C,a,V] += scale * +1.00000000 * np.einsum('IJAB,uBvJ->uIvA', h['bb'][C,C,V,V], t['ab'][pa,pV,ha,hC], optimize='optimal')
	O['ab'][v,C,a,V] += scale * +1.00000000 * np.einsum('UV,WX,IUXA,aWuV->aIuA', eta1['b'], gamma1['b'], h['bb'][C,A,A,V], t['ab'][pv,pA,ha,hA], optimize='optimal')
	O['ab'][v,C,a,V] += scale * -1.00000000 * np.einsum('UV,WX,IWVA,aUuX->aIuA', eta1['b'], gamma1['b'], h['bb'][C,A,A,V], t['ab'][pv,pA,ha,hA], optimize='optimal')
	O['ab'][v,C,a,V] += scale * -1.00000000 * np.einsum('UV,IJVA,aUuJ->aIuA', eta1['b'], h['bb'][C,C,A,V], t['ab'][pv,pA,ha,hC], optimize='optimal')
	O['ab'][v,C,a,V] += scale * +1.00000000 * np.einsum('UV,IUAB,aBuV->aIuA', gamma1['b'], h['bb'][C,A,V,V], t['ab'][pv,pV,ha,hA], optimize='optimal')
	O['ab'][v,C,a,V] += scale * +1.00000000 * np.einsum('IJAB,aBuJ->aIuA', h['bb'][C,C,V,V], t['ab'][pv,pV,ha,hC], optimize='optimal')
	
	
	O['ab'][a,V,a,V] += scale * +1.00000000 * np.einsum('UV,IAVB,uUvI->uAvB', eta1['b'], h['bb'][C,V,A,V], t['ab'][pa,pA,ha,hC], optimize='optimal')
	O['ab'][a,V,a,V] += scale * -1.00000000 * np.einsum('UV,UABC,uCvV->uAvB', gamma1['b'], h['bb'][A,V,V,V], t['ab'][pa,pV,ha,hA], optimize='optimal')
	O['ab'][a,V,a,V] += scale * -1.00000000 * np.einsum('IABC,uCvI->uAvB', h['bb'][C,V,V,V], t['ab'][pa,pV,ha,hC], optimize='optimal')
	O['ab'][v,V,a,V] += scale * -1.00000000 * np.einsum('UV,WX,UAXB,aWuV->aAuB', eta1['b'], gamma1['b'], h['bb'][A,V,A,V], t['ab'][pv,pA,ha,hA], optimize='optimal')
	O['ab'][v,V,a,V] += scale * +1.00000000 * np.einsum('UV,WX,WAVB,aUuX->aAuB', eta1['b'], gamma1['b'], h['bb'][A,V,A,V], t['ab'][pv,pA,ha,hA], optimize='optimal')
	O['ab'][v,V,a,V] += scale * +1.00000000 * np.einsum('UV,IAVB,aUuI->aAuB', eta1['b'], h['bb'][C,V,A,V], t['ab'][pv,pA,ha,hC], optimize='optimal')
	O['ab'][v,V,a,V] += scale * -1.00000000 * np.einsum('UV,UABC,aCuV->aAuB', gamma1['b'], h['bb'][A,V,V,V], t['ab'][pv,pV,ha,hA], optimize='optimal')
	O['ab'][v,V,a,V] += scale * -1.00000000 * np.einsum('IABC,aCuI->aAuB', h['bb'][C,V,V,V], t['ab'][pv,pV,ha,hC], optimize='optimal')
	O['ab'][a,A,c,A] += scale * -1.00000000 * np.einsum('UV,WX,YUZX,uWiV->uYiZ', eta1['b'], gamma1['b'], h['bb'][A,A,A,A], t['ab'][pa,pA,hc,hA], optimize='optimal')
	O['ab'][a,A,c,A] += scale * +1.00000000 * np.einsum('UV,WX,YWZV,uUiX->uYiZ', eta1['b'], gamma1['b'], h['bb'][A,A,A,A], t['ab'][pa,pA,hc,hA], optimize='optimal')
	O['ab'][a,A,c,A] += scale * -1.00000000 * np.einsum('UV,IWXV,uUiI->uWiX', eta1['b'], h['bb'][C,A,A,A], t['ab'][pa,pA,hc,hC], optimize='optimal')
	O['ab'][a,A,c,A] += scale * +1.00000000 * np.einsum('UV,WUXA,uAiV->uWiX', gamma1['b'], h['bb'][A,A,A,V], t['ab'][pa,pV,hc,hA], optimize='optimal')
	O['ab'][a,A,c,A] += scale * -1.00000000 * np.einsum('IUVA,uAiI->uUiV', h['bb'][C,A,A,V], t['ab'][pa,pV,hc,hC], optimize='optimal')
	O['ab'][v,A,c,A] += scale * -1.00000000 * np.einsum('UV,WX,YUZX,aWiV->aYiZ', eta1['b'], gamma1['b'], h['bb'][A,A,A,A], t['ab'][pv,pA,hc,hA], optimize='optimal')
	O['ab'][v,A,c,A] += scale * +1.00000000 * np.einsum('UV,WX,YWZV,aUiX->aYiZ', eta1['b'], gamma1['b'], h['bb'][A,A,A,A], t['ab'][pv,pA,hc,hA], optimize='optimal')
	O['ab'][v,A,c,A] += scale * -1.00000000 * np.einsum('UV,IWXV,aUiI->aWiX', eta1['b'], h['bb'][C,A,A,A], t['ab'][pv,pA,hc,hC], optimize='optimal')
	O['ab'][v,A,c,A] += scale * +1.00000000 * np.einsum('UV,WUXA,aAiV->aWiX', gamma1['b'], h['bb'][A,A,A,V], t['ab'][pv,pV,hc,hA], optimize='optimal')
	O['ab'][v,A,c,A] += scale * -1.00000000 * np.einsum('IUVA,aAiI->aUiV', h['bb'][C,A,A,V], t['ab'][pv,pV,hc,hC], optimize='optimal')
	O['ab'][a,C,c,A] += scale * -1.00000000 * np.einsum('UV,WX,IUYX,uWiV->uIiY', eta1['b'], gamma1['b'], h['bb'][C,A,A,A], t['ab'][pa,pA,hc,hA], optimize='optimal')
	O['ab'][a,C,c,A] += scale * +1.00000000 * np.einsum('UV,WX,IWYV,uUiX->uIiY', eta1['b'], gamma1['b'], h['bb'][C,A,A,A], t['ab'][pa,pA,hc,hA], optimize='optimal')
	O['ab'][a,C,c,A] += scale * +1.00000000 * np.einsum('UV,IJWV,uUiJ->uIiW', eta1['b'], h['bb'][C,C,A,A], t['ab'][pa,pA,hc,hC], optimize='optimal')
	O['ab'][a,C,c,A] += scale * +1.00000000 * np.einsum('UV,IUWA,uAiV->uIiW', gamma1['b'], h['bb'][C,A,A,V], t['ab'][pa,pV,hc,hA], optimize='optimal')
	O['ab'][a,C,c,A] += scale * +1.00000000 * np.einsum('IJUA,uAiJ->uIiU', h['bb'][C,C,A,V], t['ab'][pa,pV,hc,hC], optimize='optimal')
	O['ab'][v,C,c,A] += scale * -1.00000000 * np.einsum('UV,WX,IUYX,aWiV->aIiY', eta1['b'], gamma1['b'], h['bb'][C,A,A,A], t['ab'][pv,pA,hc,hA], optimize='optimal')
	O['ab'][v,C,c,A] += scale * +1.00000000 * np.einsum('UV,WX,IWYV,aUiX->aIiY', eta1['b'], gamma1['b'], h['bb'][C,A,A,A], t['ab'][pv,pA,hc,hA], optimize='optimal')
	O['ab'][v,C,c,A] += scale * +1.00000000 * np.einsum('UV,IJWV,aUiJ->aIiW', eta1['b'], h['bb'][C,C,A,A], t['ab'][pv,pA,hc,hC], optimize='optimal')
	O['ab'][v,C,c,A] += scale * +1.00000000 * np.einsum('UV,IUWA,aAiV->aIiW', gamma1['b'], h['bb'][C,A,A,V], t['ab'][pv,pV,hc,hA], optimize='optimal')
	O['ab'][v,C,c,A] += scale * +1.00000000 * np.einsum('IJUA,aAiJ->aIiU', h['bb'][C,C,A,V], t['ab'][pv,pV,hc,hC], optimize='optimal')
	O['ab'][a,V,c,A] += scale * +1.00000000 * np.einsum('UV,WX,UAYX,uWiV->uAiY', eta1['b'], gamma1['b'], h['bb'][A,V,A,A], t['ab'][pa,pA,hc,hA], optimize='optimal')
	O['ab'][a,V,c,A] += scale * -1.00000000 * np.einsum('UV,WX,WAYV,uUiX->uAiY', eta1['b'], gamma1['b'], h['bb'][A,V,A,A], t['ab'][pa,pA,hc,hA], optimize='optimal')
	O['ab'][a,V,c,A] += scale * -1.00000000 * np.einsum('UV,IAWV,uUiI->uAiW', eta1['b'], h['bb'][C,V,A,A], t['ab'][pa,pA,hc,hC], optimize='optimal')
	O['ab'][a,V,c,A] += scale * -1.00000000 * np.einsum('UV,UAWB,uBiV->uAiW', gamma1['b'], h['bb'][A,V,A,V], t['ab'][pa,pV,hc,hA], optimize='optimal')
	O['ab'][a,V,c,A] += scale * -1.00000000 * np.einsum('IAUB,uBiI->uAiU', h['bb'][C,V,A,V], t['ab'][pa,pV,hc,hC], optimize='optimal')
	O['ab'][v,V,c,A] += scale * +1.00000000 * np.einsum('UV,WX,UAYX,aWiV->aAiY', eta1['b'], gamma1['b'], h['bb'][A,V,A,A], t['ab'][pv,pA,hc,hA], optimize='optimal')
	O['ab'][v,V,c,A] += scale * -1.00000000 * np.einsum('UV,WX,WAYV,aUiX->aAiY', eta1['b'], gamma1['b'], h['bb'][A,V,A,A], t['ab'][pv,pA,hc,hA], optimize='optimal')
	O['ab'][v,V,c,A] += scale * -1.00000000 * np.einsum('UV,IAWV,aUiI->aAiW', eta1['b'], h['bb'][C,V,A,A], t['ab'][pv,pA,hc,hC], optimize='optimal')
	O['ab'][v,V,c,A] += scale * -1.00000000 * np.einsum('UV,UAWB,aBiV->aAiW', gamma1['b'], h['bb'][A,V,A,V], t['ab'][pv,pV,hc,hA], optimize='optimal')
	O['ab'][v,V,c,A] += scale * -1.00000000 * np.einsum('IAUB,aBiI->aAiU', h['bb'][C,V,A,V], t['ab'][pv,pV,hc,hC], optimize='optimal')
	O['ab'][a,A,c,C] += scale * -1.00000000 * np.einsum('UV,WX,YUIX,uWiV->uYiI', eta1['b'], gamma1['b'], h['bb'][A,A,C,A], t['ab'][pa,pA,hc,hA], optimize='optimal')
	O['ab'][a,A,c,C] += scale * +1.00000000 * np.einsum('UV,WX,YWIV,uUiX->uYiI', eta1['b'], gamma1['b'], h['bb'][A,A,C,A], t['ab'][pa,pA,hc,hA], optimize='optimal')
	O['ab'][a,A,c,C] += scale * -1.00000000 * np.einsum('UV,IWJV,uUiI->uWiJ', eta1['b'], h['bb'][C,A,C,A], t['ab'][pa,pA,hc,hC], optimize='optimal')
	O['ab'][a,A,c,C] += scale * +1.00000000 * np.einsum('UV,WUIA,uAiV->uWiI', gamma1['b'], h['bb'][A,A,C,V], t['ab'][pa,pV,hc,hA], optimize='optimal')
	O['ab'][a,A,c,C] += scale * -1.00000000 * np.einsum('IUJA,uAiI->uUiJ', h['bb'][C,A,C,V], t['ab'][pa,pV,hc,hC], optimize='optimal')
	O['ab'][v,A,c,C] += scale * -1.00000000 * np.einsum('UV,WX,YUIX,aWiV->aYiI', eta1['b'], gamma1['b'], h['bb'][A,A,C,A], t['ab'][pv,pA,hc,hA], optimize='optimal')
	O['ab'][v,A,c,C] += scale * +1.00000000 * np.einsum('UV,WX,YWIV,aUiX->aYiI', eta1['b'], gamma1['b'], h['bb'][A,A,C,A], t['ab'][pv,pA,hc,hA], optimize='optimal')
	O['ab'][v,A,c,C] += scale * -1.00000000 * np.einsum('UV,IWJV,aUiI->aWiJ', eta1['b'], h['bb'][C,A,C,A], t['ab'][pv,pA,hc,hC], optimize='optimal')
	O['ab'][v,A,c,C] += scale * +1.00000000 * np.einsum('UV,WUIA,aAiV->aWiI', gamma1['b'], h['bb'][A,A,C,V], t['ab'][pv,pV,hc,hA], optimize='optimal')
	O['ab'][v,A,c,C] += scale * -1.00000000 * np.einsum('IUJA,aAiI->aUiJ', h['bb'][C,A,C,V], t['ab'][pv,pV,hc,hC], optimize='optimal')
	O['ab'][a,C,c,C] += scale * -1.00000000 * np.einsum('UV,WX,IUJX,uWiV->uIiJ', eta1['b'], gamma1['b'], h['bb'][C,A,C,A], t['ab'][pa,pA,hc,hA], optimize='optimal')
	O['ab'][a,C,c,C] += scale * +1.00000000 * np.einsum('UV,WX,IWJV,uUiX->uIiJ', eta1['b'], gamma1['b'], h['bb'][C,A,C,A], t['ab'][pa,pA,hc,hA], optimize='optimal')
	O['ab'][a,C,c,C] += scale * +1.00000000 * np.einsum('UV,IJKV,uUiJ->uIiK', eta1['b'], h['bb'][C,C,C,A], t['ab'][pa,pA,hc,hC], optimize='optimal')
	O['ab'][a,C,c,C] += scale * +1.00000000 * np.einsum('UV,IUJA,uAiV->uIiJ', gamma1['b'], h['bb'][C,A,C,V], t['ab'][pa,pV,hc,hA], optimize='optimal')
	O['ab'][a,C,c,C] += scale * +1.00000000 * np.einsum('IJKA,uAiJ->uIiK', h['bb'][C,C,C,V], t['ab'][pa,pV,hc,hC], optimize='optimal')
	O['ab'][v,C,c,C] += scale * -1.00000000 * np.einsum('UV,WX,IUJX,aWiV->aIiJ', eta1['b'], gamma1['b'], h['bb'][C,A,C,A], t['ab'][pv,pA,hc,hA], optimize='optimal')
	O['ab'][v,C,c,C] += scale * +1.00000000 * np.einsum('UV,WX,IWJV,aUiX->aIiJ', eta1['b'], gamma1['b'], h['bb'][C,A,C,A], t['ab'][pv,pA,hc,hA], optimize='optimal')
	O['ab'][v,C,c,C] += scale * +1.00000000 * np.einsum('UV,IJKV,aUiJ->aIiK', eta1['b'], h['bb'][C,C,C,A], t['ab'][pv,pA,hc,hC], optimize='optimal')
	O['ab'][v,C,c,C] += scale * +1.00000000 * np.einsum('UV,IUJA,aAiV->aIiJ', gamma1['b'], h['bb'][C,A,C,V], t['ab'][pv,pV,hc,hA], optimize='optimal')
	O['ab'][v,C,c,C] += scale * +1.00000000 * np.einsum('IJKA,aAiJ->aIiK', h['bb'][C,C,C,V], t['ab'][pv,pV,hc,hC], optimize='optimal')
	O['ab'][a,V,c,C] += scale * +1.00000000 * np.einsum('UV,WX,UAIX,uWiV->uAiI', eta1['b'], gamma1['b'], h['bb'][A,V,C,A], t['ab'][pa,pA,hc,hA], optimize='optimal')
	O['ab'][a,V,c,C] += scale * -1.00000000 * np.einsum('UV,WX,WAIV,uUiX->uAiI', eta1['b'], gamma1['b'], h['bb'][A,V,C,A], t['ab'][pa,pA,hc,hA], optimize='optimal')
	O['ab'][a,V,c,C] += scale * -1.00000000 * np.einsum('UV,IAJV,uUiI->uAiJ', eta1['b'], h['bb'][C,V,C,A], t['ab'][pa,pA,hc,hC], optimize='optimal')
	O['ab'][a,V,c,C] += scale * -1.00000000 * np.einsum('UV,UAIB,uBiV->uAiI', gamma1['b'], h['bb'][A,V,C,V], t['ab'][pa,pV,hc,hA], optimize='optimal')
	O['ab'][a,V,c,C] += scale * -1.00000000 * np.einsum('IAJB,uBiI->uAiJ', h['bb'][C,V,C,V], t['ab'][pa,pV,hc,hC], optimize='optimal')
	O['ab'][v,V,c,C] += scale * +1.00000000 * np.einsum('UV,WX,UAIX,aWiV->aAiI', eta1['b'], gamma1['b'], h['bb'][A,V,C,A], t['ab'][pv,pA,hc,hA], optimize='optimal')
	O['ab'][v,V,c,C] += scale * -1.00000000 * np.einsum('UV,WX,WAIV,aUiX->aAiI', eta1['b'], gamma1['b'], h['bb'][A,V,C,A], t['ab'][pv,pA,hc,hA], optimize='optimal')
	O['ab'][v,V,c,C] += scale * -1.00000000 * np.einsum('UV,IAJV,aUiI->aAiJ', eta1['b'], h['bb'][C,V,C,A], t['ab'][pv,pA,hc,hC], optimize='optimal')
	O['ab'][v,V,c,C] += scale * -1.00000000 * np.einsum('UV,UAIB,aBiV->aAiI', gamma1['b'], h['bb'][A,V,C,V], t['ab'][pv,pV,hc,hA], optimize='optimal')
	O['ab'][v,V,c,C] += scale * -1.00000000 * np.einsum('IAJB,aBiI->aAiJ', h['bb'][C,V,C,V], t['ab'][pv,pV,hc,hC], optimize='optimal')
	O['ab'][a,A,c,V] += scale * +1.00000000 * np.einsum('UV,WX,YUXA,uWiV->uYiA', eta1['b'], gamma1['b'], h['bb'][A,A,A,V], t['ab'][pa,pA,hc,hA], optimize='optimal')
	O['ab'][a,A,c,V] += scale * -1.00000000 * np.einsum('UV,WX,YWVA,uUiX->uYiA', eta1['b'], gamma1['b'], h['bb'][A,A,A,V], t['ab'][pa,pA,hc,hA], optimize='optimal')
	O['ab'][a,A,c,V] += scale * +1.00000000 * np.einsum('UV,IWVA,uUiI->uWiA', eta1['b'], h['bb'][C,A,A,V], t['ab'][pa,pA,hc,hC], optimize='optimal')
	O['ab'][a,A,c,V] += scale * +1.00000000 * np.einsum('UV,WUAB,uBiV->uWiA', gamma1['b'], h['bb'][A,A,V,V], t['ab'][pa,pV,hc,hA], optimize='optimal')
	O['ab'][a,A,c,V] += scale * -1.00000000 * np.einsum('IUAB,uBiI->uUiA', h['bb'][C,A,V,V], t['ab'][pa,pV,hc,hC], optimize='optimal')
	O['ab'][v,A,c,V] += scale * +1.00000000 * np.einsum('UV,WX,YUXA,aWiV->aYiA', eta1['b'], gamma1['b'], h['bb'][A,A,A,V], t['ab'][pv,pA,hc,hA], optimize='optimal')
	O['ab'][v,A,c,V] += scale * -1.00000000 * np.einsum('UV,WX,YWVA,aUiX->aYiA', eta1['b'], gamma1['b'], h['bb'][A,A,A,V], t['ab'][pv,pA,hc,hA], optimize='optimal')
	O['ab'][v,A,c,V] += scale * +1.00000000 * np.einsum('UV,IWVA,aUiI->aWiA', eta1['b'], h['bb'][C,A,A,V], t['ab'][pv,pA,hc,hC], optimize='optimal')
	O['ab'][v,A,c,V] += scale * +1.00000000 * np.einsum('UV,WUAB,aBiV->aWiA', gamma1['b'], h['bb'][A,A,V,V], t['ab'][pv,pV,hc,hA], optimize='optimal')
	O['ab'][v,A,c,V] += scale * -1.00000000 * np.einsum('IUAB,aBiI->aUiA', h['bb'][C,A,V,V], t['ab'][pv,pV,hc,hC], optimize='optimal')
	O['ab'][a,C,c,V] += scale * +1.00000000 * np.einsum('UV,WX,IUXA,uWiV->uIiA', eta1['b'], gamma1['b'], h['bb'][C,A,A,V], t['ab'][pa,pA,hc,hA], optimize='optimal')
	O['ab'][a,C,c,V] += scale * -1.00000000 * np.einsum('UV,WX,IWVA,uUiX->uIiA', eta1['b'], gamma1['b'], h['bb'][C,A,A,V], t['ab'][pa,pA,hc,hA], optimize='optimal')
	O['ab'][a,C,c,V] += scale * -1.00000000 * np.einsum('UV,IJVA,uUiJ->uIiA', eta1['b'], h['bb'][C,C,A,V], t['ab'][pa,pA,hc,hC], optimize='optimal')
	O['ab'][a,C,c,V] += scale * +1.00000000 * np.einsum('UV,IUAB,uBiV->uIiA', gamma1['b'], h['bb'][C,A,V,V], t['ab'][pa,pV,hc,hA], optimize='optimal')
	O['ab'][a,C,c,V] += scale * +1.00000000 * np.einsum('IJAB,uBiJ->uIiA', h['bb'][C,C,V,V], t['ab'][pa,pV,hc,hC], optimize='optimal')
	O['ab'][v,C,c,V] += scale * +1.00000000 * np.einsum('UV,WX,IUXA,aWiV->aIiA', eta1['b'], gamma1['b'], h['bb'][C,A,A,V], t['ab'][pv,pA,hc,hA], optimize='optimal')
	O['ab'][v,C,c,V] += scale * -1.00000000 * np.einsum('UV,WX,IWVA,aUiX->aIiA', eta1['b'], gamma1['b'], h['bb'][C,A,A,V], t['ab'][pv,pA,hc,hA], optimize='optimal')
	O['ab'][v,C,c,V] += scale * -1.00000000 * np.einsum('UV,IJVA,aUiJ->aIiA', eta1['b'], h['bb'][C,C,A,V], t['ab'][pv,pA,hc,hC], optimize='optimal')
	O['ab'][v,C,c,V] += scale * +1.00000000 * np.einsum('UV,IUAB,aBiV->aIiA', gamma1['b'], h['bb'][C,A,V,V], t['ab'][pv,pV,hc,hA], optimize='optimal')
	O['ab'][v,C,c,V] += scale * +1.00000000 * np.einsum('IJAB,aBiJ->aIiA', h['bb'][C,C,V,V], t['ab'][pv,pV,hc,hC], optimize='optimal')
	O['ab'][a,V,c,V] += scale * -1.00000000 * np.einsum('UV,WX,UAXB,uWiV->uAiB', eta1['b'], gamma1['b'], h['bb'][A,V,A,V], t['ab'][pa,pA,hc,hA], optimize='optimal')
	O['ab'][a,V,c,V] += scale * +1.00000000 * np.einsum('UV,WX,WAVB,uUiX->uAiB', eta1['b'], gamma1['b'], h['bb'][A,V,A,V], t['ab'][pa,pA,hc,hA], optimize='optimal')
	O['ab'][a,V,c,V] += scale * +1.00000000 * np.einsum('UV,IAVB,uUiI->uAiB', eta1['b'], h['bb'][C,V,A,V], t['ab'][pa,pA,hc,hC], optimize='optimal')
	O['ab'][a,V,c,V] += scale * -1.00000000 * np.einsum('UV,UABC,uCiV->uAiB', gamma1['b'], h['bb'][A,V,V,V], t['ab'][pa,pV,hc,hA], optimize='optimal')
	O['ab'][a,V,c,V] += scale * -1.00000000 * np.einsum('IABC,uCiI->uAiB', h['bb'][C,V,V,V], t['ab'][pa,pV,hc,hC], optimize='optimal')
	O['ab'][v,V,c,V] += scale * -1.00000000 * np.einsum('UV,WX,UAXB,aWiV->aAiB', eta1['b'], gamma1['b'], h['bb'][A,V,A,V], t['ab'][pv,pA,hc,hA], optimize='optimal')
	O['ab'][v,V,c,V] += scale * +1.00000000 * np.einsum('UV,WX,WAVB,aUiX->aAiB', eta1['b'], gamma1['b'], h['bb'][A,V,A,V], t['ab'][pv,pA,hc,hA], optimize='optimal')
	O['ab'][v,V,c,V] += scale * +1.00000000 * np.einsum('UV,IAVB,aUiI->aAiB', eta1['b'], h['bb'][C,V,A,V], t['ab'][pv,pA,hc,hC], optimize='optimal')
	O['ab'][v,V,c,V] += scale * -1.00000000 * np.einsum('UV,UABC,aCiV->aAiB', gamma1['b'], h['bb'][A,V,V,V], t['ab'][pv,pV,hc,hA], optimize='optimal')
	O['ab'][v,V,c,V] += scale * -1.00000000 * np.einsum('IABC,aCiI->aAiB', h['bb'][C,V,V,V], t['ab'][pv,pV,hc,hC], optimize='optimal')

	t1 = time.time()
	if verbose: print("h2c_t2b_c2b took {:.4f} seconds to run.".format(t1-t0))

	return O
