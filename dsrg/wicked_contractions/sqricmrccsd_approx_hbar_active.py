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
	X['b'] += -1.0 * np.einsum('uv,wx,xIuU,vVwI->VU', eta1['a'], gamma1['a'], h['ab'][a,C,a,A], t['ab'][pa,pA,ha,hC], optimize=True)
	X['b'] += 1.0 * np.einsum('uv,wx,xUuA,vAwV->UV', eta1['a'], gamma1['a'], h['ab'][a,A,a,V], t['ab'][pa,pV,ha,hA], optimize=True)
	X['b'] += -1.0 * np.einsum('uv,WX,iXuU,vViW->VU', eta1['a'], gamma1['b'], h['ab'][c,A,a,A], t['ab'][pa,pA,hc,hA], optimize=True)
	X['b'] += 1.0 * np.einsum('uv,iu,vUiV->UV', eta1['a'], h['a'][c,a], t['ab'][pa,pA,hc,hA], optimize=True)
	X['b'] += -1.0 * np.einsum('uv,iIuU,vViI->VU', eta1['a'], h['ab'][c,C,a,A], t['ab'][pa,pA,hc,hC], optimize=True)
	X['b'] += 1.0 * np.einsum('uv,iUuA,vAiV->UV', eta1['a'], h['ab'][c,A,a,V], t['ab'][pa,pV,hc,hA], optimize=True)
	X['b'] += 1.0 * np.einsum('WX,uv,iUuW,vXiV->UV', eta1['b'], eta1['a'], h['ab'][c,A,a,A], t['ab'][pa,pA,hc,hA], optimize=True)
	X['b'] += 0.5 * np.einsum('WX,YZ,IUWY,XZIV->UV', eta1['b'], eta1['b'], h['bb'][C,A,A,A], t['bb'][pA,pA,hC,hA], optimize=True)
	X['b'] += 1.0 * np.einsum('WX,uv,vUaW,aXuV->UV', eta1['b'], gamma1['a'], h['ab'][a,A,v,A], t['ab'][pv,pA,ha,hA], optimize=True)
	X['b'] += -1.0 * np.einsum('WX,YZ,IZUW,VXIY->VU', eta1['b'], gamma1['b'], h['bb'][C,A,A,A], t['bb'][pA,pA,hC,hA], optimize=True)
	X['b'] += 1.0 * np.einsum('WX,YZ,UZWA,XAVY->UV', eta1['b'], gamma1['b'], h['bb'][A,A,A,V], t['bb'][pA,pV,hA,hA], optimize=True)
	X['b'] += 1.0 * np.einsum('WX,iUaW,aXiV->UV', eta1['b'], h['ab'][c,A,v,A], t['ab'][pv,pA,hc,hA], optimize=True)
	X['b'] += -1.0 * np.einsum('WX,IW,UXIV->UV', eta1['b'], h['b'][C,A], t['bb'][pA,pA,hC,hA], optimize=True)
	X['b'] += -0.5 * np.einsum('WX,IJUW,VXIJ->VU', eta1['b'], h['bb'][C,C,A,A], t['bb'][pA,pA,hC,hC], optimize=True)
	X['b'] += 1.0 * np.einsum('WX,IUWA,XAIV->UV', eta1['b'], h['bb'][C,A,A,V], t['bb'][pA,pV,hC,hA], optimize=True)
	X['b'] += 1.0 * np.einsum('uv,va,aUuV->UV', gamma1['a'], h['a'][a,v], t['ab'][pv,pA,ha,hA], optimize=True)
	X['b'] += -1.0 * np.einsum('uv,vIaU,aVuI->VU', gamma1['a'], h['ab'][a,C,v,A], t['ab'][pv,pA,ha,hC], optimize=True)
	X['b'] += 1.0 * np.einsum('uv,vUaA,aAuV->UV', gamma1['a'], h['ab'][a,A,v,V], t['ab'][pv,pV,ha,hA], optimize=True)
	X['b'] += -1.0 * np.einsum('WX,uv,vXaU,aVuW->VU', gamma1['b'], gamma1['a'], h['ab'][a,A,v,A], t['ab'][pv,pA,ha,hA], optimize=True)
	X['b'] += -0.5 * np.einsum('WX,YZ,XZUA,VAWY->VU', gamma1['b'], gamma1['b'], h['bb'][A,A,A,V], t['bb'][pA,pV,hA,hA], optimize=True)
	X['b'] += -1.0 * np.einsum('WX,iXaU,aViW->VU', gamma1['b'], h['ab'][c,A,v,A], t['ab'][pv,pA,hc,hA], optimize=True)
	X['b'] += -1.0 * np.einsum('WX,IXUA,VAIW->VU', gamma1['b'], h['bb'][C,A,A,V], t['bb'][pA,pV,hC,hA], optimize=True)
	X['b'] += 0.5 * np.einsum('WX,UXAB,ABVW->UV', gamma1['b'], h['bb'][A,A,V,V], t['bb'][pV,pV,hA,hA], optimize=True)
	X['b'] += 1.0 * np.einsum('WX,XA,UAVW->UV', gamma1['b'], h['b'][A,V], t['bb'][pA,pV,hA,hA], optimize=True)
	X['b'] += 1.0 * np.einsum('ia,aUiV->UV', h['a'][c,v], t['ab'][pv,pA,hc,hA], optimize=True)
	X['b'] += 0.5 * np.einsum('iuvw,vwux,xUiV->UV', h['aa'][c,a,a,a], lambdas['aa'], t['ab'][pa,pA,hc,hA], optimize=True)
	X['b'] += -1.0 * np.einsum('iIaU,aViI->VU', h['ab'][c,C,v,A], t['ab'][pv,pA,hc,hC], optimize=True)
	X['b'] += -0.5 * np.einsum('iUuV,uvwx,wxiv->UV', h['ab'][c,A,a,A], lambdas['aa'], t['aa'][pa,pa,hc,ha], optimize=True)
	X['b'] += -1.0 * np.einsum('iUuV,uWvX,vXiW->UV', h['ab'][c,A,a,A], lambdas['ab'], t['ab'][pa,pA,hc,hA], optimize=True)
	X['b'] += 1.0 * np.einsum('iUuW,uWvX,vXiV->UV', h['ab'][c,A,a,A], lambdas['ab'], t['ab'][pa,pA,hc,hA], optimize=True)
	X['b'] += 1.0 * np.einsum('iUaA,aAiV->UV', h['ab'][c,A,v,V], t['ab'][pv,pV,hc,hA], optimize=True)
	X['b'] += 1.0 * np.einsum('iWuU,uXvW,vViX->VU', h['ab'][c,A,a,A], lambdas['ab'], t['ab'][pa,pA,hc,hA], optimize=True)
	X['b'] += -1.0 * np.einsum('iWuX,uXvW,vUiV->UV', h['ab'][c,A,a,A], lambdas['ab'], t['ab'][pa,pA,hc,hA], optimize=True)
	X['b'] += 0.5 * np.einsum('uvwa,wxuv,aUxV->UV', h['aa'][a,a,a,v], lambdas['aa'], t['ab'][pv,pA,ha,hA], optimize=True)
	X['b'] += -1.0 * np.einsum('uIvU,vwux,xVwI->VU', h['ab'][a,C,a,A], lambdas['aa'], t['ab'][pa,pA,ha,hC], optimize=True)
	X['b'] += -1.0 * np.einsum('uIvU,vWuX,VXIW->VU', h['ab'][a,C,a,A], lambdas['ab'], t['bb'][pA,pA,hC,hA], optimize=True)
	X['b'] += 1.0 * np.einsum('uIvW,vWuX,UXIV->UV', h['ab'][a,C,a,A], lambdas['ab'], t['bb'][pA,pA,hC,hA], optimize=True)
	X['b'] += 1.0 * np.einsum('uUvA,vwux,xAwV->UV', h['ab'][a,A,a,V], lambdas['aa'], t['ab'][pa,pV,ha,hA], optimize=True)
	X['b'] += -1.0 * np.einsum('uUvA,vWuX,XAVW->UV', h['ab'][a,A,a,V], lambdas['ab'], t['bb'][pA,pV,hA,hA], optimize=True)
	X['b'] += -0.5 * np.einsum('uUaV,vwux,xavw->UV', h['ab'][a,A,v,A], lambdas['aa'], t['aa'][pa,pv,ha,ha], optimize=True)
	X['b'] += 1.0 * np.einsum('uUaV,vWuX,aXvW->UV', h['ab'][a,A,v,A], lambdas['ab'], t['ab'][pv,pA,ha,hA], optimize=True)
	X['b'] += -1.0 * np.einsum('uUaW,vWuX,aXvV->UV', h['ab'][a,A,v,A], lambdas['ab'], t['ab'][pv,pA,ha,hA], optimize=True)
	X['b'] += 1.0 * np.einsum('uWvA,vXuW,UAVX->UV', h['ab'][a,A,a,V], lambdas['ab'], t['bb'][pA,pV,hA,hA], optimize=True)
	X['b'] += -1.0 * np.einsum('uWaU,vXuW,aVvX->VU', h['ab'][a,A,v,A], lambdas['ab'], t['ab'][pv,pA,ha,hA], optimize=True)
	X['b'] += 1.0 * np.einsum('uWaX,vXuW,aUvV->UV', h['ab'][a,A,v,A], lambdas['ab'], t['ab'][pv,pA,ha,hA], optimize=True)
	X['b'] += -1.0 * np.einsum('IA,UAIV->UV', h['b'][C,V], t['bb'][pA,pV,hC,hA], optimize=True)
	X['b'] += -0.5 * np.einsum('IJUA,VAIJ->VU', h['bb'][C,C,A,V], t['bb'][pA,pV,hC,hC], optimize=True)
	X['b'] += 1.0 * np.einsum('IUVW,uWvX,vXuI->UV', h['bb'][C,A,A,A], lambdas['ab'], t['ab'][pa,pA,ha,hC], optimize=True)
	X['b'] += 0.5 * np.einsum('IUVW,WXYZ,YZIX->UV', h['bb'][C,A,A,A], lambdas['bb'], t['bb'][pA,pA,hC,hA], optimize=True)
	X['b'] += 0.25 * np.einsum('IUWX,WXYZ,YZIV->UV', h['bb'][C,A,A,A], lambdas['bb'], t['bb'][pA,pA,hC,hA], optimize=True)
	X['b'] += 0.5 * np.einsum('IUAB,ABIV->UV', h['bb'][C,A,V,V], t['bb'][pV,pV,hC,hA], optimize=True)
	X['b'] += -1.0 * np.einsum('IWUX,uXvW,vVuI->VU', h['bb'][C,A,A,A], lambdas['ab'], t['ab'][pa,pA,ha,hC], optimize=True)
	X['b'] += -1.0 * np.einsum('IWUX,XYWZ,VZIY->VU', h['bb'][C,A,A,A], lambdas['bb'], t['bb'][pA,pA,hC,hA], optimize=True)
	X['b'] += -0.5 * np.einsum('IWXY,XYWZ,UZIV->UV', h['bb'][C,A,A,A], lambdas['bb'], t['bb'][pA,pA,hC,hA], optimize=True)
	X['b'] += 1.0 * np.einsum('UWVA,uXvW,vAuX->UV', h['bb'][A,A,A,V], lambdas['ab'], t['ab'][pa,pV,ha,hA], optimize=True)
	X['b'] += -0.5 * np.einsum('UWVA,XYWZ,ZAXY->UV', h['bb'][A,A,A,V], lambdas['bb'], t['bb'][pA,pV,hA,hA], optimize=True)
	X['b'] += -1.0 * np.einsum('UWXA,uXvW,vAuV->UV', h['bb'][A,A,A,V], lambdas['ab'], t['ab'][pa,pV,ha,hA], optimize=True)
	X['b'] += 1.0 * np.einsum('UWXA,XYWZ,ZAVY->UV', h['bb'][A,A,A,V], lambdas['bb'], t['bb'][pA,pV,hA,hA], optimize=True)
	X['b'] += -0.25 * np.einsum('WXUA,YZWX,VAYZ->VU', h['bb'][A,A,A,V], lambdas['bb'], t['bb'][pA,pV,hA,hA], optimize=True)
	X['b'] += 0.5 * np.einsum('WXYA,YZWX,UAVZ->UV', h['bb'][A,A,A,V], lambdas['bb'], t['bb'][pA,pV,hA,hA], optimize=True)
	# a|a
	X['a'] += 0.5 * np.einsum('wx,yz,iuwy,xziv->uv', eta1['a'], eta1['a'], h['aa'][c,a,a,a], t['aa'][pa,pa,hc,ha], optimize=True)
	X['a'] += -1.0 * np.einsum('wx,yz,izuw,vxiy->vu', eta1['a'], gamma1['a'], h['aa'][c,a,a,a], t['aa'][pa,pa,hc,ha], optimize=True)
	X['a'] += 1.0 * np.einsum('wx,yz,uzwa,xavy->uv', eta1['a'], gamma1['a'], h['aa'][a,a,a,v], t['aa'][pa,pv,ha,ha], optimize=True)
	X['a'] += 1.0 * np.einsum('wx,UV,uVwA,xAvU->uv', eta1['a'], gamma1['b'], h['ab'][a,A,a,V], t['ab'][pa,pV,ha,hA], optimize=True)
	X['a'] += -1.0 * np.einsum('wx,iw,uxiv->uv', eta1['a'], h['a'][c,a], t['aa'][pa,pa,hc,ha], optimize=True)
	X['a'] += -0.5 * np.einsum('wx,ijuw,vxij->vu', eta1['a'], h['aa'][c,c,a,a], t['aa'][pa,pa,hc,hc], optimize=True)
	X['a'] += 1.0 * np.einsum('wx,iuwa,xaiv->uv', eta1['a'], h['aa'][c,a,a,v], t['aa'][pa,pv,hc,ha], optimize=True)
	X['a'] += 1.0 * np.einsum('wx,uIwA,xAvI->uv', eta1['a'], h['ab'][a,C,a,V], t['ab'][pa,pV,ha,hC], optimize=True)
	X['a'] += 1.0 * np.einsum('UV,wx,uIwU,xVvI->uv', eta1['b'], eta1['a'], h['ab'][a,C,a,A], t['ab'][pa,pA,ha,hC], optimize=True)
	X['a'] += -1.0 * np.einsum('UV,wx,xIuU,vVwI->vu', eta1['b'], gamma1['a'], h['ab'][a,C,a,A], t['ab'][pa,pA,ha,hC], optimize=True)
	X['a'] += -1.0 * np.einsum('UV,WX,iXuU,vViW->vu', eta1['b'], gamma1['b'], h['ab'][c,A,a,A], t['ab'][pa,pA,hc,hA], optimize=True)
	X['a'] += 1.0 * np.einsum('UV,WX,uXaU,aVvW->uv', eta1['b'], gamma1['b'], h['ab'][a,A,v,A], t['ab'][pv,pA,ha,hA], optimize=True)
	X['a'] += -1.0 * np.einsum('UV,iIuU,vViI->vu', eta1['b'], h['ab'][c,C,a,A], t['ab'][pa,pA,hc,hC], optimize=True)
	X['a'] += 1.0 * np.einsum('UV,uIaU,aVvI->uv', eta1['b'], h['ab'][a,C,v,A], t['ab'][pv,pA,ha,hC], optimize=True)
	X['a'] += 1.0 * np.einsum('UV,IU,uVvI->uv', eta1['b'], h['b'][C,A], t['ab'][pa,pA,ha,hC], optimize=True)
	X['a'] += -0.5 * np.einsum('wx,yz,xzua,vawy->vu', gamma1['a'], gamma1['a'], h['aa'][a,a,a,v], t['aa'][pa,pv,ha,ha], optimize=True)
	X['a'] += -1.0 * np.einsum('wx,ixua,vaiw->vu', gamma1['a'], h['aa'][c,a,a,v], t['aa'][pa,pv,hc,ha], optimize=True)
	X['a'] += 0.5 * np.einsum('wx,uxab,abvw->uv', gamma1['a'], h['aa'][a,a,v,v], t['aa'][pv,pv,ha,ha], optimize=True)
	X['a'] += 1.0 * np.einsum('wx,xa,uavw->uv', gamma1['a'], h['a'][a,v], t['aa'][pa,pv,ha,ha], optimize=True)
	X['a'] += -1.0 * np.einsum('wx,xIuA,vAwI->vu', gamma1['a'], h['ab'][a,C,a,V], t['ab'][pa,pV,ha,hC], optimize=True)
	X['a'] += -1.0 * np.einsum('UV,wx,xVuA,vAwU->vu', gamma1['b'], gamma1['a'], h['ab'][a,A,a,V], t['ab'][pa,pV,ha,hA], optimize=True)
	X['a'] += -1.0 * np.einsum('UV,iVuA,vAiU->vu', gamma1['b'], h['ab'][c,A,a,V], t['ab'][pa,pV,hc,hA], optimize=True)
	X['a'] += 1.0 * np.einsum('UV,uVaA,aAvU->uv', gamma1['b'], h['ab'][a,A,v,V], t['ab'][pv,pV,ha,hA], optimize=True)
	X['a'] += 1.0 * np.einsum('UV,VA,uAvU->uv', gamma1['b'], h['b'][A,V], t['ab'][pa,pV,ha,hA], optimize=True)
	X['a'] += -1.0 * np.einsum('ia,uaiv->uv', h['a'][c,v], t['aa'][pa,pv,hc,ha], optimize=True)
	X['a'] += -0.5 * np.einsum('ijua,vaij->vu', h['aa'][c,c,a,v], t['aa'][pa,pv,hc,hc], optimize=True)
	X['a'] += 0.5 * np.einsum('iuvw,wxyz,yzix->uv', h['aa'][c,a,a,a], lambdas['aa'], t['aa'][pa,pa,hc,ha], optimize=True)
	X['a'] += 1.0 * np.einsum('iuvw,wUxV,xViU->uv', h['aa'][c,a,a,a], lambdas['ab'], t['ab'][pa,pA,hc,hA], optimize=True)
	X['a'] += 0.25 * np.einsum('iuwx,wxyz,yziv->uv', h['aa'][c,a,a,a], lambdas['aa'], t['aa'][pa,pa,hc,ha], optimize=True)
	X['a'] += 0.5 * np.einsum('iuab,abiv->uv', h['aa'][c,a,v,v], t['aa'][pv,pv,hc,ha], optimize=True)
	X['a'] += -1.0 * np.einsum('iwux,xywz,vziy->vu', h['aa'][c,a,a,a], lambdas['aa'], t['aa'][pa,pa,hc,ha], optimize=True)
	X['a'] += -1.0 * np.einsum('iwux,xUwV,vViU->vu', h['aa'][c,a,a,a], lambdas['ab'], t['ab'][pa,pA,hc,hA], optimize=True)
	X['a'] += -0.5 * np.einsum('iwxy,xywz,uziv->uv', h['aa'][c,a,a,a], lambdas['aa'], t['aa'][pa,pa,hc,ha], optimize=True)
	X['a'] += -1.0 * np.einsum('iIuA,vAiI->vu', h['ab'][c,C,a,V], t['ab'][pa,pV,hc,hC], optimize=True)
	X['a'] += -1.0 * np.einsum('iUuV,wVxU,vxiw->vu', h['ab'][c,A,a,A], lambdas['ab'], t['aa'][pa,pa,hc,ha], optimize=True)
	X['a'] += -1.0 * np.einsum('iUuV,VWUX,vXiW->vu', h['ab'][c,A,a,A], lambdas['bb'], t['ab'][pa,pA,hc,hA], optimize=True)
	X['a'] += 1.0 * np.einsum('iUwV,wVxU,uxiv->uv', h['ab'][c,A,a,A], lambdas['ab'], t['aa'][pa,pa,hc,ha], optimize=True)
	X['a'] += -0.5 * np.einsum('uwva,xywz,zaxy->uv', h['aa'][a,a,a,v], lambdas['aa'], t['aa'][pa,pv,ha,ha], optimize=True)
	X['a'] += 1.0 * np.einsum('uwva,xUwV,aVxU->uv', h['aa'][a,a,a,v], lambdas['ab'], t['ab'][pv,pA,ha,hA], optimize=True)
	X['a'] += 1.0 * np.einsum('uwxa,xywz,zavy->uv', h['aa'][a,a,a,v], lambdas['aa'], t['aa'][pa,pv,ha,ha], optimize=True)
	X['a'] += -1.0 * np.einsum('uwxa,xUwV,aVvU->uv', h['aa'][a,a,a,v], lambdas['ab'], t['ab'][pv,pA,ha,hA], optimize=True)
	X['a'] += -1.0 * np.einsum('uIvU,wUxV,xVwI->uv', h['ab'][a,C,a,A], lambdas['ab'], t['ab'][pa,pA,ha,hC], optimize=True)
	X['a'] += -0.5 * np.einsum('uIvU,UVWX,WXIV->uv', h['ab'][a,C,a,A], lambdas['bb'], t['bb'][pA,pA,hC,hA], optimize=True)
	X['a'] += 1.0 * np.einsum('uIwU,wUxV,xVvI->uv', h['ab'][a,C,a,A], lambdas['ab'], t['ab'][pa,pA,ha,hC], optimize=True)
	X['a'] += 1.0 * np.einsum('uIaA,aAvI->uv', h['ab'][a,C,v,V], t['ab'][pv,pV,ha,hC], optimize=True)
	X['a'] += 1.0 * np.einsum('uUvA,wVxU,xAwV->uv', h['ab'][a,A,a,V], lambdas['ab'], t['ab'][pa,pV,ha,hA], optimize=True)
	X['a'] += -0.5 * np.einsum('uUvA,VWUX,XAVW->uv', h['ab'][a,A,a,V], lambdas['bb'], t['bb'][pA,pV,hA,hA], optimize=True)
	X['a'] += -1.0 * np.einsum('uUwA,wVxU,xAvV->uv', h['ab'][a,A,a,V], lambdas['ab'], t['ab'][pa,pV,ha,hA], optimize=True)
	X['a'] += -1.0 * np.einsum('uUaV,wVxU,xavw->uv', h['ab'][a,A,v,A], lambdas['ab'], t['aa'][pa,pv,ha,ha], optimize=True)
	X['a'] += 1.0 * np.einsum('uUaV,VWUX,aXvW->uv', h['ab'][a,A,v,A], lambdas['bb'], t['ab'][pv,pA,ha,hA], optimize=True)
	X['a'] += -0.25 * np.einsum('wxua,yzwx,vayz->vu', h['aa'][a,a,a,v], lambdas['aa'], t['aa'][pa,pv,ha,ha], optimize=True)
	X['a'] += 0.5 * np.einsum('wxya,yzwx,uavz->uv', h['aa'][a,a,a,v], lambdas['aa'], t['aa'][pa,pv,ha,ha], optimize=True)
	X['a'] += 1.0 * np.einsum('wIuU,xUwV,vVxI->vu', h['ab'][a,C,a,A], lambdas['ab'], t['ab'][pa,pA,ha,hC], optimize=True)
	X['a'] += -1.0 * np.einsum('wIxU,xUwV,uVvI->uv', h['ab'][a,C,a,A], lambdas['ab'], t['ab'][pa,pA,ha,hC], optimize=True)
	X['a'] += -1.0 * np.einsum('wUuA,xVwU,vAxV->vu', h['ab'][a,A,a,V], lambdas['ab'], t['ab'][pa,pV,ha,hA], optimize=True)
	X['a'] += 1.0 * np.einsum('wUxA,xVwU,uAvV->uv', h['ab'][a,A,a,V], lambdas['ab'], t['ab'][pa,pV,ha,hA], optimize=True)
	X['a'] += 1.0 * np.einsum('wUaV,xVwU,uavx->uv', h['ab'][a,A,v,A], lambdas['ab'], t['aa'][pa,pv,ha,ha], optimize=True)
	X['a'] += 1.0 * np.einsum('IA,uAvI->uv', h['b'][C,V], t['ab'][pa,pV,ha,hC], optimize=True)
	X['a'] += 0.5 * np.einsum('IUVW,VWUX,uXvI->uv', h['bb'][C,A,A,A], lambdas['bb'], t['ab'][pa,pA,ha,hC], optimize=True)
	X['a'] += 0.5 * np.einsum('UVWA,WXUV,uAvX->uv', h['bb'][A,A,A,V], lambdas['bb'], t['ab'][pa,pV,ha,hA], optimize=True)
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
	X['bb'] += 1.0 * np.einsum('uv,iUuV,vWiX->UWVX', eta1['a'], h['ab'][c,A,a,A], t['ab'][pa,pA,hc,hA], optimize=True)
	X['bb'] += 1.0 * np.einsum('YZ,IUVY,WZIX->UWVX', eta1['b'], h['bb'][C,A,A,A], t['bb'][pA,pA,hC,hA], optimize=True)
	X['bb'] += 0.25 * np.einsum('YZ,UVYA,ZAWX->UVWX', eta1['b'], h['bb'][A,A,A,V], t['bb'][pA,pV,hA,hA], optimize=True)
	X['bb'] += 1.0 * np.einsum('uv,vUaV,aWuX->UWVX', gamma1['a'], h['ab'][a,A,v,A], t['ab'][pv,pA,ha,hA], optimize=True)
	X['bb'] += 0.25 * np.einsum('YZ,IZUV,WXIY->WXUV', gamma1['b'], h['bb'][C,A,A,A], t['bb'][pA,pA,hC,hA], optimize=True)
	X['bb'] += 1.0 * np.einsum('YZ,UZVA,WAXY->UWVX', gamma1['b'], h['bb'][A,A,A,V], t['bb'][pA,pV,hA,hA], optimize=True)
	X['bb'] += 1.0 * np.einsum('iUaV,aWiX->UWVX', h['ab'][c,A,v,A], t['ab'][pv,pA,hc,hA], optimize=True)
	X['bb'] += -0.5 * np.einsum('IU,VWIX->VWUX', h['b'][C,A], t['bb'][pA,pA,hC,hA], optimize=True)
	X['bb'] += 0.125 * np.einsum('IJUV,WXIJ->WXUV', h['bb'][C,C,A,A], t['bb'][pA,pA,hC,hC], optimize=True)
	X['bb'] += 1.0 * np.einsum('IUVA,WAIX->UWVX', h['bb'][C,A,A,V], t['bb'][pA,pV,hC,hA], optimize=True)
	X['bb'] += -0.5 * np.einsum('UA,VAWX->UVWX', h['b'][A,V], t['bb'][pA,pV,hA,hA], optimize=True)
	X['bb'] += 0.125 * np.einsum('UVAB,ABWX->UVWX', h['bb'][A,A,V,V], t['bb'][pV,pV,hA,hA], optimize=True)
	# aA|Aa
	X['ab'] += -1.0 * np.einsum('wx,iuvw,xUiV->uUvV', eta1['a'], h['aa'][c,a,a,a], t['ab'][pa,pA,hc,hA], optimize=True)
	X['ab'] += -1.0 * np.einsum('wx,iUwV,uxiv->uUvV', eta1['a'], h['ab'][c,A,a,A], t['aa'][pa,pa,hc,ha], optimize=True)
	X['ab'] += -1.0 * np.einsum('wx,uIwU,xVvI->uVvU', eta1['a'], h['ab'][a,C,a,A], t['ab'][pa,pA,ha,hC], optimize=True)
	X['ab'] += 1.0 * np.einsum('wx,uUwA,xAvV->uUvV', eta1['a'], h['ab'][a,A,a,V], t['ab'][pa,pV,ha,hA], optimize=True)
	X['ab'] += -1.0 * np.einsum('WX,iUuW,vXiV->vUuV', eta1['b'], h['ab'][c,A,a,A], t['ab'][pa,pA,hc,hA], optimize=True)
	X['ab'] += -1.0 * np.einsum('WX,uIvW,UXIV->uUvV', eta1['b'], h['ab'][a,C,a,A], t['bb'][pA,pA,hC,hA], optimize=True)
	X['ab'] += 1.0 * np.einsum('WX,uUaW,aXvV->uUvV', eta1['b'], h['ab'][a,A,v,A], t['ab'][pv,pA,ha,hA], optimize=True)
	X['ab'] += -1.0 * np.einsum('WX,IUVW,uXvI->uUvV', eta1['b'], h['bb'][C,A,A,A], t['ab'][pa,pA,ha,hC], optimize=True)
	X['ab'] += 1.0 * np.einsum('wx,uxva,aUwV->uUvV', gamma1['a'], h['aa'][a,a,a,v], t['ab'][pv,pA,ha,hA], optimize=True)
	X['ab'] += 1.0 * np.einsum('wx,xIuU,vVwI->vVuU', gamma1['a'], h['ab'][a,C,a,A], t['ab'][pa,pA,ha,hC], optimize=True)
	X['ab'] += -1.0 * np.einsum('wx,xUuA,vAwV->vUuV', gamma1['a'], h['ab'][a,A,a,V], t['ab'][pa,pV,ha,hA], optimize=True)
	X['ab'] += 1.0 * np.einsum('wx,xUaV,uavw->uUvV', gamma1['a'], h['ab'][a,A,v,A], t['aa'][pa,pv,ha,ha], optimize=True)
	X['ab'] += 1.0 * np.einsum('WX,iXuU,vViW->vVuU', gamma1['b'], h['ab'][c,A,a,A], t['ab'][pa,pA,hc,hA], optimize=True)
	X['ab'] += 1.0 * np.einsum('WX,uXvA,UAVW->uUvV', gamma1['b'], h['ab'][a,A,a,V], t['bb'][pA,pV,hA,hA], optimize=True)
	X['ab'] += -1.0 * np.einsum('WX,uXaU,aVvW->uVvU', gamma1['b'], h['ab'][a,A,v,A], t['ab'][pv,pA,ha,hA], optimize=True)
	X['ab'] += 1.0 * np.einsum('WX,UXVA,uAvW->uUvV', gamma1['b'], h['bb'][A,A,A,V], t['ab'][pa,pV,ha,hA], optimize=True)
	X['ab'] += -1.0 * np.einsum('iu,vUiV->vUuV', h['a'][c,a], t['ab'][pa,pA,hc,hA], optimize=True)
	X['ab'] += -1.0 * np.einsum('iuva,aUiV->uUvV', h['aa'][c,a,a,v], t['ab'][pv,pA,hc,hA], optimize=True)
	X['ab'] += 1.0 * np.einsum('iIuU,vViI->vVuU', h['ab'][c,C,a,A], t['ab'][pa,pA,hc,hC], optimize=True)
	X['ab'] += -1.0 * np.einsum('iUuA,vAiV->vUuV', h['ab'][c,A,a,V], t['ab'][pa,pV,hc,hA], optimize=True)
	X['ab'] += -1.0 * np.einsum('iUaV,uaiv->uUvV', h['ab'][c,A,v,A], t['aa'][pa,pv,hc,ha], optimize=True)
	X['ab'] += 1.0 * np.einsum('ua,aUvV->uUvV', h['a'][a,v], t['ab'][pv,pA,ha,hA], optimize=True)
	X['ab'] += -1.0 * np.einsum('uIvA,UAIV->uUvV', h['ab'][a,C,a,V], t['bb'][pA,pV,hC,hA], optimize=True)
	X['ab'] += -1.0 * np.einsum('uIaU,aVvI->uVvU', h['ab'][a,C,v,A], t['ab'][pv,pA,ha,hC], optimize=True)
	X['ab'] += 1.0 * np.einsum('uUaA,aAvV->uUvV', h['ab'][a,A,v,V], t['ab'][pv,pV,ha,hA], optimize=True)
	X['ab'] += -1.0 * np.einsum('IU,uVvI->uVvU', h['b'][C,A], t['ab'][pa,pA,ha,hC], optimize=True)
	X['ab'] += -1.0 * np.einsum('IUVA,uAvI->uUvV', h['bb'][C,A,A,V], t['ab'][pa,pV,ha,hC], optimize=True)
	X['ab'] += 1.0 * np.einsum('UA,uAvV->uUvV', h['b'][A,V], t['ab'][pa,pV,ha,hA], optimize=True)
	# aa|aa
	X['aa'] += 1.0 * np.einsum('yz,iuvy,wzix->uwvx', eta1['a'], h['aa'][c,a,a,a], t['aa'][pa,pa,hc,ha], optimize=True)
	X['aa'] += 0.25 * np.einsum('yz,uvya,zawx->uvwx', eta1['a'], h['aa'][a,a,a,v], t['aa'][pa,pv,ha,ha], optimize=True)
	X['aa'] += 1.0 * np.einsum('UV,uIvU,wVxI->uwvx', eta1['b'], h['ab'][a,C,a,A], t['ab'][pa,pA,ha,hC], optimize=True)
	X['aa'] += 0.25 * np.einsum('yz,izuv,wxiy->wxuv', gamma1['a'], h['aa'][c,a,a,a], t['aa'][pa,pa,hc,ha], optimize=True)
	X['aa'] += 1.0 * np.einsum('yz,uzva,waxy->uwvx', gamma1['a'], h['aa'][a,a,a,v], t['aa'][pa,pv,ha,ha], optimize=True)
	X['aa'] += 1.0 * np.einsum('UV,uVvA,wAxU->uwvx', gamma1['b'], h['ab'][a,A,a,V], t['ab'][pa,pV,ha,hA], optimize=True)
	X['aa'] += -0.5 * np.einsum('iu,vwix->vwux', h['a'][c,a], t['aa'][pa,pa,hc,ha], optimize=True)
	X['aa'] += 0.125 * np.einsum('ijuv,wxij->wxuv', h['aa'][c,c,a,a], t['aa'][pa,pa,hc,hc], optimize=True)
	X['aa'] += 1.0 * np.einsum('iuva,waix->uwvx', h['aa'][c,a,a,v], t['aa'][pa,pv,hc,ha], optimize=True)
	X['aa'] += -0.5 * np.einsum('ua,vawx->uvwx', h['a'][a,v], t['aa'][pa,pv,ha,ha], optimize=True)
	X['aa'] += 0.125 * np.einsum('uvab,abwx->uvwx', h['aa'][a,a,v,v], t['aa'][pv,pv,ha,ha], optimize=True)
	X['aa'] += 1.0 * np.einsum('uIvA,wAxI->uwvx', h['ab'][a,C,a,V], t['ab'][pa,pV,ha,hC], optimize=True)
	toc = time.time()
	elapsed_time = toc - tic
	if verbose:
		print(f'Took {elapsed_time} seconds.')
	return X


