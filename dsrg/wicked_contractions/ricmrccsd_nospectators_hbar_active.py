import time
import numpy as np

def Hbar_ncomm1_nbody1(X, h, t, gamma1, eta1, lambdas, orbspace, verbose=False):
	tic = time.time()
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

	# A|A
	X['b'] += -1.0 * np.einsum('uv,iIuU,vViI->VU', eta1['a'], h['ab'][c,C,a,A], t['ab'][pa,pA,hc,hC], optimize=True)
	X['b'] += 1.0 * np.einsum('uv,iUuV,vi->UV', eta1['a'], h['ab'][c,A,a,A], t['a'][pa,hc], optimize=True)
	X['b'] += 1.0 * np.einsum('uv,iUuA,vAiV->UV', eta1['a'], h['ab'][c,A,a,V], t['ab'][pa,pV,hc,hA], optimize=True)
	X['b'] += -0.5 * np.einsum('WX,IJUW,VXIJ->VU', eta1['b'], h['bb'][C,C,A,A], t['bb'][pA,pA,hC,hC], optimize=True)
	X['b'] += -1.0 * np.einsum('WX,IUVW,XI->UV', eta1['b'], h['bb'][C,A,A,A], t['b'][pA,hC], optimize=True)
	X['b'] += -1.0 * np.einsum('uv,vIaU,aVuI->VU', gamma1['a'], h['ab'][a,C,v,A], t['ab'][pv,pA,ha,hC], optimize=True)
	X['b'] += 1.0 * np.einsum('uv,vUaV,au->UV', gamma1['a'], h['ab'][a,A,v,A], t['a'][pv,ha], optimize=True)
	X['b'] += 1.0 * np.einsum('uv,vUaA,aAuV->UV', gamma1['a'], h['ab'][a,A,v,V], t['ab'][pv,pV,ha,hA], optimize=True)
	X['b'] += 1.0 * np.einsum('WX,UXVA,AW->UV', gamma1['b'], h['bb'][A,A,A,V], t['b'][pV,hA], optimize=True)
	X['b'] += 0.5 * np.einsum('WX,UXAB,ABVW->UV', gamma1['b'], h['bb'][A,A,V,V], t['bb'][pV,pV,hA,hA], optimize=True)
	X['b'] += -1.0 * np.einsum('iIaU,aViI->VU', h['ab'][c,C,v,A], t['ab'][pv,pA,hc,hC], optimize=True)
	X['b'] += 1.0 * np.einsum('iUaV,ai->UV', h['ab'][c,A,v,A], t['a'][pv,hc], optimize=True)
	X['b'] += 1.0 * np.einsum('iUaA,aAiV->UV', h['ab'][c,A,v,V], t['ab'][pv,pV,hc,hA], optimize=True)
	X['b'] += -1.0 * np.einsum('IU,VI->VU', h['b'][C,A], t['b'][pA,hC], optimize=True)
	X['b'] += -0.5 * np.einsum('IJUA,VAIJ->VU', h['bb'][C,C,A,V], t['bb'][pA,pV,hC,hC], optimize=True)
	X['b'] += -1.0 * np.einsum('IUVA,AI->UV', h['bb'][C,A,A,V], t['b'][pV,hC], optimize=True)
	X['b'] += 0.5 * np.einsum('IUAB,ABIV->UV', h['bb'][C,A,V,V], t['bb'][pV,pV,hC,hA], optimize=True)
	X['b'] += 1.0 * np.einsum('UA,AV->UV', h['b'][A,V], t['b'][pV,hA], optimize=True)
	# a|a
	X['a'] += -0.5 * np.einsum('wx,ijuw,vxij->vu', eta1['a'], h['aa'][c,c,a,a], t['aa'][pa,pa,hc,hc], optimize=True)
	X['a'] += -1.0 * np.einsum('wx,iuvw,xi->uv', eta1['a'], h['aa'][c,a,a,a], t['a'][pa,hc], optimize=True)
	X['a'] += -1.0 * np.einsum('UV,iIuU,vViI->vu', eta1['b'], h['ab'][c,C,a,A], t['ab'][pa,pA,hc,hC], optimize=True)
	X['a'] += 1.0 * np.einsum('UV,uIvU,VI->uv', eta1['b'], h['ab'][a,C,a,A], t['b'][pA,hC], optimize=True)
	X['a'] += 1.0 * np.einsum('UV,uIaU,aVvI->uv', eta1['b'], h['ab'][a,C,v,A], t['ab'][pv,pA,ha,hC], optimize=True)
	X['a'] += 1.0 * np.einsum('wx,uxva,aw->uv', gamma1['a'], h['aa'][a,a,a,v], t['a'][pv,ha], optimize=True)
	X['a'] += 0.5 * np.einsum('wx,uxab,abvw->uv', gamma1['a'], h['aa'][a,a,v,v], t['aa'][pv,pv,ha,ha], optimize=True)
	X['a'] += -1.0 * np.einsum('UV,iVuA,vAiU->vu', gamma1['b'], h['ab'][c,A,a,V], t['ab'][pa,pV,hc,hA], optimize=True)
	X['a'] += 1.0 * np.einsum('UV,uVvA,AU->uv', gamma1['b'], h['ab'][a,A,a,V], t['b'][pV,hA], optimize=True)
	X['a'] += 1.0 * np.einsum('UV,uVaA,aAvU->uv', gamma1['b'], h['ab'][a,A,v,V], t['ab'][pv,pV,ha,hA], optimize=True)
	X['a'] += -1.0 * np.einsum('iu,vi->vu', h['a'][c,a], t['a'][pa,hc], optimize=True)
	X['a'] += -0.5 * np.einsum('ijua,vaij->vu', h['aa'][c,c,a,v], t['aa'][pa,pv,hc,hc], optimize=True)
	X['a'] += -1.0 * np.einsum('iuva,ai->uv', h['aa'][c,a,a,v], t['a'][pv,hc], optimize=True)
	X['a'] += 0.5 * np.einsum('iuab,abiv->uv', h['aa'][c,a,v,v], t['aa'][pv,pv,hc,ha], optimize=True)
	X['a'] += -1.0 * np.einsum('iIuA,vAiI->vu', h['ab'][c,C,a,V], t['ab'][pa,pV,hc,hC], optimize=True)
	X['a'] += 1.0 * np.einsum('ua,av->uv', h['a'][a,v], t['a'][pv,ha], optimize=True)
	X['a'] += 1.0 * np.einsum('uIvA,AI->uv', h['ab'][a,C,a,V], t['b'][pV,hC], optimize=True)
	X['a'] += 1.0 * np.einsum('uIaA,aAvI->uv', h['ab'][a,C,v,V], t['ab'][pv,pV,ha,hC], optimize=True)
	toc = time.time()
	elapsed_time = toc - tic
	if verbose:
		print(f'Took {elapsed_time} seconds.')
	return X


def Hbar_ncomm1_nbody2(X, h, t, gamma1, eta1, lambdas, orbspace, verbose=False):
	tic = time.time()
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

	# AA|AA
	X['bb'] += 0.125 * np.einsum('IJUV,WXIJ->WXUV', h['bb'][C,C,A,A], t['bb'][pA,pA,hC,hC], optimize=True)
	X['bb'] += 0.5 * np.einsum('IUVW,XI->UXVW', h['bb'][C,A,A,A], t['b'][pA,hC], optimize=True)
	X['bb'] += 0.5 * np.einsum('UVWA,AX->UVWX', h['bb'][A,A,A,V], t['b'][pV,hA], optimize=True)
	X['bb'] += 0.125 * np.einsum('UVAB,ABWX->UVWX', h['bb'][A,A,V,V], t['bb'][pV,pV,hA,hA], optimize=True)
	# aA|Aa
	X['ab'] += 1.0 * np.einsum('iIuU,vViI->vVuU', h['ab'][c,C,a,A], t['ab'][pa,pA,hc,hC], optimize=True)
	X['ab'] += -1.0 * np.einsum('iUuV,vi->vUuV', h['ab'][c,A,a,A], t['a'][pa,hc], optimize=True)
	X['ab'] += -1.0 * np.einsum('iUuA,vAiV->vUuV', h['ab'][c,A,a,V], t['ab'][pa,pV,hc,hA], optimize=True)
	X['ab'] += -1.0 * np.einsum('uIvU,VI->uVvU', h['ab'][a,C,a,A], t['b'][pA,hC], optimize=True)
	X['ab'] += -1.0 * np.einsum('uIaU,aVvI->uVvU', h['ab'][a,C,v,A], t['ab'][pv,pA,ha,hC], optimize=True)
	X['ab'] += 1.0 * np.einsum('uUvA,AV->uUvV', h['ab'][a,A,a,V], t['b'][pV,hA], optimize=True)
	X['ab'] += 1.0 * np.einsum('uUaV,av->uUvV', h['ab'][a,A,v,A], t['a'][pv,ha], optimize=True)
	X['ab'] += 1.0 * np.einsum('uUaA,aAvV->uUvV', h['ab'][a,A,v,V], t['ab'][pv,pV,ha,hA], optimize=True)
	# aa|aa
	X['aa'] += 0.125 * np.einsum('ijuv,wxij->wxuv', h['aa'][c,c,a,a], t['aa'][pa,pa,hc,hc], optimize=True)
	X['aa'] += 0.5 * np.einsum('iuvw,xi->uxvw', h['aa'][c,a,a,a], t['a'][pa,hc], optimize=True)
	X['aa'] += 0.5 * np.einsum('uvwa,ax->uvwx', h['aa'][a,a,a,v], t['a'][pv,ha], optimize=True)
	X['aa'] += 0.125 * np.einsum('uvab,abwx->uvwx', h['aa'][a,a,v,v], t['aa'][pv,pv,ha,ha], optimize=True)
	toc = time.time()
	elapsed_time = toc - tic
	if verbose:
		print(f'Took {elapsed_time} seconds.')
	return X


def Hbar_ncomm2_nbody1(X, h, t, gamma1, eta1, lambdas, orbspace, verbose=False):
	tic = time.time()
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

	# A|A
	X['b'] += -1.0 * np.einsum('uv,iIuU,VI,vi->VU', eta1['a'], h['ab'][c,C,a,A], t['b'][pA,hC], t['a'][pa,hc], optimize=True)
	X['b'] += -1.0 * np.einsum('uv,iIuA,UI,vAiV->UV', eta1['a'], h['ab'][c,C,a,V], t['b'][pA,hC], t['ab'][pa,pV,hc,hA], optimize=True)
	X['b'] += -1.0 * np.einsum('uv,iIuA,AU,vViI->VU', eta1['a'], h['ab'][c,C,a,V], t['b'][pV,hA], t['ab'][pa,pA,hc,hC], optimize=True)
	X['b'] += -0.5 * np.einsum('uv,iIaU,vi,aVuI->VU', eta1['a'], h['ab'][c,C,v,A], t['a'][pa,hc], t['ab'][pv,pA,ha,hC], optimize=True)
	X['b'] += -0.5 * np.einsum('uv,iIaU,au,vViI->VU', eta1['a'], h['ab'][c,C,v,A], t['a'][pv,ha], t['ab'][pa,pA,hc,hC], optimize=True)
	X['b'] += -0.5 * np.einsum('uv,iIaA,aUuI,vAiV->UV', eta1['a'], h['ab'][c,C,v,V], t['ab'][pv,pA,ha,hC], t['ab'][pa,pV,hc,hA], optimize=True)
	X['b'] += -0.5 * np.einsum('uv,iIaA,aAuU,vViI->VU', eta1['a'], h['ab'][c,C,v,V], t['ab'][pv,pV,ha,hA], t['ab'][pa,pA,hc,hC], optimize=True)
	X['b'] += 1.0 * np.einsum('uv,iUuA,AV,vi->UV', eta1['a'], h['ab'][c,A,a,V], t['b'][pV,hA], t['a'][pa,hc], optimize=True)
	X['b'] += 0.5 * np.einsum('uv,iUaV,au,vi->UV', eta1['a'], h['ab'][c,A,v,A], t['a'][pv,ha], t['a'][pa,hc], optimize=True)
	X['b'] += 0.5 * np.einsum('uv,iUaA,vi,aAuV->UV', eta1['a'], h['ab'][c,A,v,V], t['a'][pa,hc], t['ab'][pv,pV,ha,hA], optimize=True)
	X['b'] += 0.5 * np.einsum('uv,iUaA,au,vAiV->UV', eta1['a'], h['ab'][c,A,v,V], t['a'][pv,ha], t['ab'][pa,pV,hc,hA], optimize=True)
	X['b'] += 1.0 * np.einsum('WX,IJUW,VJ,XI->VU', eta1['b'], h['bb'][C,C,A,A], t['b'][pA,hC], t['b'][pA,hC], optimize=True)
	X['b'] += -0.25 * np.einsum('WX,IJUA,AW,VXIJ->VU', eta1['b'], h['bb'][C,C,A,V], t['b'][pV,hA], t['bb'][pA,pA,hC,hC], optimize=True)
	X['b'] += 0.5 * np.einsum('WX,IJWA,AU,VXIJ->VU', eta1['b'], h['bb'][C,C,A,V], t['b'][pV,hA], t['bb'][pA,pA,hC,hC], optimize=True)
	X['b'] += -0.125 * np.einsum('WX,IJAB,ABUW,VXIJ->VU', eta1['b'], h['bb'][C,C,V,V], t['bb'][pV,pV,hA,hA], t['bb'][pA,pA,hC,hC], optimize=True)
	X['b'] += -0.5 * np.einsum('WX,IUVA,AW,XI->UV', eta1['b'], h['bb'][C,A,A,V], t['b'][pV,hA], t['b'][pA,hC], optimize=True)
	X['b'] += 1.0 * np.einsum('WX,IUWA,AV,XI->UV', eta1['b'], h['bb'][C,A,A,V], t['b'][pV,hA], t['b'][pA,hC], optimize=True)
	X['b'] += -0.25 * np.einsum('WX,IUAB,XI,ABVW->UV', eta1['b'], h['bb'][C,A,V,V], t['b'][pA,hC], t['bb'][pV,pV,hA,hA], optimize=True)
	X['b'] += 0.5 * np.einsum('uv,iIaU,vi,aVuI->VU', gamma1['a'], h['ab'][c,C,v,A], t['a'][pa,hc], t['ab'][pv,pA,ha,hC], optimize=True)
	X['b'] += 0.5 * np.einsum('uv,iIaU,au,vViI->VU', gamma1['a'], h['ab'][c,C,v,A], t['a'][pv,ha], t['ab'][pa,pA,hc,hC], optimize=True)
	X['b'] += 0.5 * np.einsum('uv,iIaA,aUuI,vAiV->UV', gamma1['a'], h['ab'][c,C,v,V], t['ab'][pv,pA,ha,hC], t['ab'][pa,pV,hc,hA], optimize=True)
	X['b'] += 0.5 * np.einsum('uv,iIaA,aAuU,vViI->VU', gamma1['a'], h['ab'][c,C,v,V], t['ab'][pv,pV,ha,hA], t['ab'][pa,pA,hc,hC], optimize=True)
	X['b'] += -0.5 * np.einsum('uv,iUaV,au,vi->UV', gamma1['a'], h['ab'][c,A,v,A], t['a'][pv,ha], t['a'][pa,hc], optimize=True)
	X['b'] += -0.5 * np.einsum('uv,iUaA,vi,aAuV->UV', gamma1['a'], h['ab'][c,A,v,V], t['a'][pa,hc], t['ab'][pv,pV,ha,hA], optimize=True)
	X['b'] += -0.5 * np.einsum('uv,iUaA,au,vAiV->UV', gamma1['a'], h['ab'][c,A,v,V], t['a'][pv,ha], t['ab'][pa,pV,hc,hA], optimize=True)
	X['b'] += -1.0 * np.einsum('uv,vIaU,VI,au->VU', gamma1['a'], h['ab'][a,C,v,A], t['b'][pA,hC], t['a'][pv,ha], optimize=True)
	X['b'] += -1.0 * np.einsum('uv,vIaA,UI,aAuV->UV', gamma1['a'], h['ab'][a,C,v,V], t['b'][pA,hC], t['ab'][pv,pV,ha,hA], optimize=True)
	X['b'] += -1.0 * np.einsum('uv,vIaA,AU,aVuI->VU', gamma1['a'], h['ab'][a,C,v,V], t['b'][pV,hA], t['ab'][pv,pA,ha,hC], optimize=True)
	X['b'] += 1.0 * np.einsum('uv,vUaA,AV,au->UV', gamma1['a'], h['ab'][a,A,v,V], t['b'][pV,hA], t['a'][pv,ha], optimize=True)
	X['b'] += 0.25 * np.einsum('WX,IJUA,AW,VXIJ->VU', gamma1['b'], h['bb'][C,C,A,V], t['b'][pV,hA], t['bb'][pA,pA,hC,hC], optimize=True)
	X['b'] += 0.125 * np.einsum('WX,IJAB,ABUW,VXIJ->VU', gamma1['b'], h['bb'][C,C,V,V], t['bb'][pV,pV,hA,hA], t['bb'][pA,pA,hC,hC], optimize=True)
	X['b'] += 0.5 * np.einsum('WX,IUVA,AW,XI->UV', gamma1['b'], h['bb'][C,A,A,V], t['b'][pV,hA], t['b'][pA,hC], optimize=True)
	X['b'] += 0.25 * np.einsum('WX,IUAB,XI,ABVW->UV', gamma1['b'], h['bb'][C,A,V,V], t['b'][pA,hC], t['bb'][pV,pV,hA,hA], optimize=True)
	X['b'] += -1.0 * np.einsum('WX,IXUA,AW,VI->VU', gamma1['b'], h['bb'][C,A,A,V], t['b'][pV,hA], t['b'][pA,hC], optimize=True)
	X['b'] += -0.5 * np.einsum('WX,IXAB,UI,ABVW->UV', gamma1['b'], h['bb'][C,A,V,V], t['b'][pA,hC], t['bb'][pV,pV,hA,hA], optimize=True)
	X['b'] += -1.0 * np.einsum('WX,UXAB,BV,AW->UV', gamma1['b'], h['bb'][A,A,V,V], t['b'][pV,hA], t['b'][pV,hA], optimize=True)
	X['b'] += -1.0 * np.einsum('iIaU,VI,ai->VU', h['ab'][c,C,v,A], t['b'][pA,hC], t['a'][pv,hc], optimize=True)
	X['b'] += -1.0 * np.einsum('iIaA,UI,aAiV->UV', h['ab'][c,C,v,V], t['b'][pA,hC], t['ab'][pv,pV,hc,hA], optimize=True)
	X['b'] += -1.0 * np.einsum('iIaA,AU,aViI->VU', h['ab'][c,C,v,V], t['b'][pV,hA], t['ab'][pv,pA,hc,hC], optimize=True)
	X['b'] += 1.0 * np.einsum('iUaA,AV,ai->UV', h['ab'][c,A,v,V], t['b'][pV,hA], t['a'][pv,hc], optimize=True)
	X['b'] += -1.0 * np.einsum('IA,AU,VI->VU', h['b'][C,V], t['b'][pV,hA], t['b'][pA,hC], optimize=True)
	X['b'] += 1.0 * np.einsum('IJUA,AI,VJ->VU', h['bb'][C,C,A,V], t['b'][pV,hC], t['b'][pA,hC], optimize=True)
	X['b'] += -0.5 * np.einsum('IJAB,UJ,ABIV->UV', h['bb'][C,C,V,V], t['b'][pA,hC], t['bb'][pV,pV,hC,hA], optimize=True)
	X['b'] += -0.5 * np.einsum('IJAB,AU,VBIJ->VU', h['bb'][C,C,V,V], t['b'][pV,hA], t['bb'][pA,pV,hC,hC], optimize=True)
	X['b'] += -1.0 * np.einsum('IUAB,AV,BI->UV', h['bb'][C,A,V,V], t['b'][pV,hA], t['b'][pV,hC], optimize=True)
	# a|a
	X['a'] += 1.0 * np.einsum('wx,ijuw,vj,xi->vu', eta1['a'], h['aa'][c,c,a,a], t['a'][pa,hc], t['a'][pa,hc], optimize=True)
	X['a'] += -0.25 * np.einsum('wx,ijua,aw,vxij->vu', eta1['a'], h['aa'][c,c,a,v], t['a'][pv,ha], t['aa'][pa,pa,hc,hc], optimize=True)
	X['a'] += 0.5 * np.einsum('wx,ijwa,au,vxij->vu', eta1['a'], h['aa'][c,c,a,v], t['a'][pv,ha], t['aa'][pa,pa,hc,hc], optimize=True)
	X['a'] += -0.125 * np.einsum('wx,ijab,abuw,vxij->vu', eta1['a'], h['aa'][c,c,v,v], t['aa'][pv,pv,ha,ha], t['aa'][pa,pa,hc,hc], optimize=True)
	X['a'] += -0.5 * np.einsum('wx,iuva,aw,xi->uv', eta1['a'], h['aa'][c,a,a,v], t['a'][pv,ha], t['a'][pa,hc], optimize=True)
	X['a'] += 1.0 * np.einsum('wx,iuwa,av,xi->uv', eta1['a'], h['aa'][c,a,a,v], t['a'][pv,ha], t['a'][pa,hc], optimize=True)
	X['a'] += -0.25 * np.einsum('wx,iuab,xi,abvw->uv', eta1['a'], h['aa'][c,a,v,v], t['a'][pa,hc], t['aa'][pv,pv,ha,ha], optimize=True)
	X['a'] += -1.0 * np.einsum('UV,iIuU,VI,vi->vu', eta1['b'], h['ab'][c,C,a,A], t['b'][pA,hC], t['a'][pa,hc], optimize=True)
	X['a'] += -0.5 * np.einsum('UV,iIuA,VI,vAiU->vu', eta1['b'], h['ab'][c,C,a,V], t['b'][pA,hC], t['ab'][pa,pV,hc,hA], optimize=True)
	X['a'] += -0.5 * np.einsum('UV,iIuA,AU,vViI->vu', eta1['b'], h['ab'][c,C,a,V], t['b'][pV,hA], t['ab'][pa,pA,hc,hC], optimize=True)
	X['a'] += -1.0 * np.einsum('UV,iIaU,ui,aVvI->uv', eta1['b'], h['ab'][c,C,v,A], t['a'][pa,hc], t['ab'][pv,pA,ha,hC], optimize=True)
	X['a'] += -1.0 * np.einsum('UV,iIaU,au,vViI->vu', eta1['b'], h['ab'][c,C,v,A], t['a'][pv,ha], t['ab'][pa,pA,hc,hC], optimize=True)
	X['a'] += -0.5 * np.einsum('UV,iIaA,aVuI,vAiU->vu', eta1['b'], h['ab'][c,C,v,V], t['ab'][pv,pA,ha,hC], t['ab'][pa,pV,hc,hA], optimize=True)
	X['a'] += -0.5 * np.einsum('UV,iIaA,aAuU,vViI->vu', eta1['b'], h['ab'][c,C,v,V], t['ab'][pv,pV,ha,hA], t['ab'][pa,pA,hc,hC], optimize=True)
	X['a'] += 0.5 * np.einsum('UV,uIvA,AU,VI->uv', eta1['b'], h['ab'][a,C,a,V], t['b'][pV,hA], t['b'][pA,hC], optimize=True)
	X['a'] += 1.0 * np.einsum('UV,uIaU,VI,av->uv', eta1['b'], h['ab'][a,C,v,A], t['b'][pA,hC], t['a'][pv,ha], optimize=True)
	X['a'] += 0.5 * np.einsum('UV,uIaA,VI,aAvU->uv', eta1['b'], h['ab'][a,C,v,V], t['b'][pA,hC], t['ab'][pv,pV,ha,hA], optimize=True)
	X['a'] += 0.5 * np.einsum('UV,uIaA,AU,aVvI->uv', eta1['b'], h['ab'][a,C,v,V], t['b'][pV,hA], t['ab'][pv,pA,ha,hC], optimize=True)
	X['a'] += 0.25 * np.einsum('wx,ijua,aw,vxij->vu', gamma1['a'], h['aa'][c,c,a,v], t['a'][pv,ha], t['aa'][pa,pa,hc,hc], optimize=True)
	X['a'] += 0.125 * np.einsum('wx,ijab,abuw,vxij->vu', gamma1['a'], h['aa'][c,c,v,v], t['aa'][pv,pv,ha,ha], t['aa'][pa,pa,hc,hc], optimize=True)
	X['a'] += 0.5 * np.einsum('wx,iuva,aw,xi->uv', gamma1['a'], h['aa'][c,a,a,v], t['a'][pv,ha], t['a'][pa,hc], optimize=True)
	X['a'] += 0.25 * np.einsum('wx,iuab,xi,abvw->uv', gamma1['a'], h['aa'][c,a,v,v], t['a'][pa,hc], t['aa'][pv,pv,ha,ha], optimize=True)
	X['a'] += -1.0 * np.einsum('wx,ixua,aw,vi->vu', gamma1['a'], h['aa'][c,a,a,v], t['a'][pv,ha], t['a'][pa,hc], optimize=True)
	X['a'] += -0.5 * np.einsum('wx,ixab,ui,abvw->uv', gamma1['a'], h['aa'][c,a,v,v], t['a'][pa,hc], t['aa'][pv,pv,ha,ha], optimize=True)
	X['a'] += -1.0 * np.einsum('wx,uxab,bv,aw->uv', gamma1['a'], h['aa'][a,a,v,v], t['a'][pv,ha], t['a'][pv,ha], optimize=True)
	X['a'] += 0.5 * np.einsum('UV,iIuA,VI,vAiU->vu', gamma1['b'], h['ab'][c,C,a,V], t['b'][pA,hC], t['ab'][pa,pV,hc,hA], optimize=True)
	X['a'] += 0.5 * np.einsum('UV,iIuA,AU,vViI->vu', gamma1['b'], h['ab'][c,C,a,V], t['b'][pV,hA], t['ab'][pa,pA,hc,hC], optimize=True)
	X['a'] += 0.5 * np.einsum('UV,iIaA,aVuI,vAiU->vu', gamma1['b'], h['ab'][c,C,v,V], t['ab'][pv,pA,ha,hC], t['ab'][pa,pV,hc,hA], optimize=True)
	X['a'] += 0.5 * np.einsum('UV,iIaA,aAuU,vViI->vu', gamma1['b'], h['ab'][c,C,v,V], t['ab'][pv,pV,ha,hA], t['ab'][pa,pA,hc,hC], optimize=True)
	X['a'] += -1.0 * np.einsum('UV,iVuA,AU,vi->vu', gamma1['b'], h['ab'][c,A,a,V], t['b'][pV,hA], t['a'][pa,hc], optimize=True)
	X['a'] += -1.0 * np.einsum('UV,iVaA,ui,aAvU->uv', gamma1['b'], h['ab'][c,A,v,V], t['a'][pa,hc], t['ab'][pv,pV,ha,hA], optimize=True)
	X['a'] += -1.0 * np.einsum('UV,iVaA,au,vAiU->vu', gamma1['b'], h['ab'][c,A,v,V], t['a'][pv,ha], t['ab'][pa,pV,hc,hA], optimize=True)
	X['a'] += -0.5 * np.einsum('UV,uIvA,AU,VI->uv', gamma1['b'], h['ab'][a,C,a,V], t['b'][pV,hA], t['b'][pA,hC], optimize=True)
	X['a'] += -0.5 * np.einsum('UV,uIaA,VI,aAvU->uv', gamma1['b'], h['ab'][a,C,v,V], t['b'][pA,hC], t['ab'][pv,pV,ha,hA], optimize=True)
	X['a'] += -0.5 * np.einsum('UV,uIaA,AU,aVvI->uv', gamma1['b'], h['ab'][a,C,v,V], t['b'][pV,hA], t['ab'][pv,pA,ha,hC], optimize=True)
	X['a'] += 1.0 * np.einsum('UV,uVaA,AU,av->uv', gamma1['b'], h['ab'][a,A,v,V], t['b'][pV,hA], t['a'][pv,ha], optimize=True)
	X['a'] += -1.0 * np.einsum('ia,au,vi->vu', h['a'][c,v], t['a'][pv,ha], t['a'][pa,hc], optimize=True)
	X['a'] += 1.0 * np.einsum('ijua,ai,vj->vu', h['aa'][c,c,a,v], t['a'][pv,hc], t['a'][pa,hc], optimize=True)
	X['a'] += -0.5 * np.einsum('ijab,uj,abiv->uv', h['aa'][c,c,v,v], t['a'][pa,hc], t['aa'][pv,pv,hc,ha], optimize=True)
	X['a'] += -0.5 * np.einsum('ijab,au,vbij->vu', h['aa'][c,c,v,v], t['a'][pv,ha], t['aa'][pa,pv,hc,hc], optimize=True)
	X['a'] += -1.0 * np.einsum('iuab,av,bi->uv', h['aa'][c,a,v,v], t['a'][pv,ha], t['a'][pv,hc], optimize=True)
	X['a'] += -1.0 * np.einsum('iIuA,AI,vi->vu', h['ab'][c,C,a,V], t['b'][pV,hC], t['a'][pa,hc], optimize=True)
	X['a'] += -1.0 * np.einsum('iIaA,ui,aAvI->uv', h['ab'][c,C,v,V], t['a'][pa,hc], t['ab'][pv,pV,ha,hC], optimize=True)
	X['a'] += -1.0 * np.einsum('iIaA,au,vAiI->vu', h['ab'][c,C,v,V], t['a'][pv,ha], t['ab'][pa,pV,hc,hC], optimize=True)
	X['a'] += 1.0 * np.einsum('uIaA,AI,av->uv', h['ab'][a,C,v,V], t['b'][pV,hC], t['a'][pv,ha], optimize=True)
	toc = time.time()
	elapsed_time = toc - tic
	if verbose:
		print(f'Took {elapsed_time} seconds.')
	return X


def Hbar_ncomm2_nbody2(X, h, t, gamma1, eta1, lambdas, orbspace, verbose=False):
	tic = time.time()
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

	# AA|AA
	X['bb'] += 0.25 * np.einsum('IJUV,WI,XJ->WXUV', h['bb'][C,C,A,A], t['b'][pA,hC], t['b'][pA,hC], optimize=True)
	X['bb'] += 0.25 * np.einsum('IJUA,AV,WXIJ->WXUV', h['bb'][C,C,A,V], t['b'][pV,hA], t['bb'][pA,pA,hC,hC], optimize=True)
	X['bb'] += 0.0625 * np.einsum('IJAB,ABUV,WXIJ->WXUV', h['bb'][C,C,V,V], t['bb'][pV,pV,hA,hA], t['bb'][pA,pA,hC,hC], optimize=True)
	X['bb'] += 1.0 * np.einsum('IUVA,AW,XI->UXVW', h['bb'][C,A,A,V], t['b'][pV,hA], t['b'][pA,hC], optimize=True)
	X['bb'] += 0.25 * np.einsum('IUAB,VI,ABWX->UVWX', h['bb'][C,A,V,V], t['b'][pA,hC], t['bb'][pV,pV,hA,hA], optimize=True)
	X['bb'] += -0.25 * np.einsum('UVAB,BW,AX->UVWX', h['bb'][A,A,V,V], t['b'][pV,hA], t['b'][pV,hA], optimize=True)
	# aA|Aa
	X['ab'] += 1.0 * np.einsum('iIuU,VI,vi->vVuU', h['ab'][c,C,a,A], t['b'][pA,hC], t['a'][pa,hc], optimize=True)
	X['ab'] += 1.0 * np.einsum('iIuA,UI,vAiV->vUuV', h['ab'][c,C,a,V], t['b'][pA,hC], t['ab'][pa,pV,hc,hA], optimize=True)
	X['ab'] += 1.0 * np.einsum('iIuA,AU,vViI->vVuU', h['ab'][c,C,a,V], t['b'][pV,hA], t['ab'][pa,pA,hc,hC], optimize=True)
	X['ab'] += 1.0 * np.einsum('iIaU,ui,aVvI->uVvU', h['ab'][c,C,v,A], t['a'][pa,hc], t['ab'][pv,pA,ha,hC], optimize=True)
	X['ab'] += 1.0 * np.einsum('iIaU,au,vViI->vVuU', h['ab'][c,C,v,A], t['a'][pv,ha], t['ab'][pa,pA,hc,hC], optimize=True)
	X['ab'] += 1.0 * np.einsum('iIaA,aUuI,vAiV->vUuV', h['ab'][c,C,v,V], t['ab'][pv,pA,ha,hC], t['ab'][pa,pV,hc,hA], optimize=True)
	X['ab'] += 1.0 * np.einsum('iIaA,aAuU,vViI->vVuU', h['ab'][c,C,v,V], t['ab'][pv,pV,ha,hA], t['ab'][pa,pA,hc,hC], optimize=True)
	X['ab'] += -1.0 * np.einsum('iUuA,AV,vi->vUuV', h['ab'][c,A,a,V], t['b'][pV,hA], t['a'][pa,hc], optimize=True)
	X['ab'] += -1.0 * np.einsum('iUaV,au,vi->vUuV', h['ab'][c,A,v,A], t['a'][pv,ha], t['a'][pa,hc], optimize=True)
	X['ab'] += -1.0 * np.einsum('iUaA,ui,aAvV->uUvV', h['ab'][c,A,v,V], t['a'][pa,hc], t['ab'][pv,pV,ha,hA], optimize=True)
	X['ab'] += -1.0 * np.einsum('iUaA,au,vAiV->vUuV', h['ab'][c,A,v,V], t['a'][pv,ha], t['ab'][pa,pV,hc,hA], optimize=True)
	X['ab'] += -1.0 * np.einsum('uIvA,AU,VI->uVvU', h['ab'][a,C,a,V], t['b'][pV,hA], t['b'][pA,hC], optimize=True)
	X['ab'] += -1.0 * np.einsum('uIaU,VI,av->uVvU', h['ab'][a,C,v,A], t['b'][pA,hC], t['a'][pv,ha], optimize=True)
	X['ab'] += -1.0 * np.einsum('uIaA,UI,aAvV->uUvV', h['ab'][a,C,v,V], t['b'][pA,hC], t['ab'][pv,pV,ha,hA], optimize=True)
	X['ab'] += -1.0 * np.einsum('uIaA,AU,aVvI->uVvU', h['ab'][a,C,v,V], t['b'][pV,hA], t['ab'][pv,pA,ha,hC], optimize=True)
	X['ab'] += 1.0 * np.einsum('uUaA,AV,av->uUvV', h['ab'][a,A,v,V], t['b'][pV,hA], t['a'][pv,ha], optimize=True)
	# aa|aa
	X['aa'] += 0.25 * np.einsum('ijuv,wi,xj->wxuv', h['aa'][c,c,a,a], t['a'][pa,hc], t['a'][pa,hc], optimize=True)
	X['aa'] += 0.25 * np.einsum('ijua,av,wxij->wxuv', h['aa'][c,c,a,v], t['a'][pv,ha], t['aa'][pa,pa,hc,hc], optimize=True)
	X['aa'] += 0.0625 * np.einsum('ijab,abuv,wxij->wxuv', h['aa'][c,c,v,v], t['aa'][pv,pv,ha,ha], t['aa'][pa,pa,hc,hc], optimize=True)
	X['aa'] += 1.0 * np.einsum('iuva,aw,xi->uxvw', h['aa'][c,a,a,v], t['a'][pv,ha], t['a'][pa,hc], optimize=True)
	X['aa'] += 0.25 * np.einsum('iuab,vi,abwx->uvwx', h['aa'][c,a,v,v], t['a'][pa,hc], t['aa'][pv,pv,ha,ha], optimize=True)
	X['aa'] += -0.25 * np.einsum('uvab,bw,ax->uvwx', h['aa'][a,a,v,v], t['a'][pv,ha], t['a'][pv,ha], optimize=True)
	toc = time.time()
	elapsed_time = toc - tic
	if verbose:
		print(f'Took {elapsed_time} seconds.')
	return X
