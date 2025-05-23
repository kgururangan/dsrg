import numpy as np

def denormal_order_ints(hbar, ref):

    e_scalar_1 = (
            np.einsum("uv,vu->", hbar['a'], ref.gam1['a'], optimize=True)
          + np.einsum("uv,vu->", hbar['b'], ref.gam1['b'], optimize=True)
    )
    print(f"    <HBar_1> = {-e_scalar_1}")

    e_scalar_2 = (
              0.25 * np.einsum("uvxy,xyuv->", hbar['aa'], ref.rdms['aa'], optimize=True)
            + np.einsum("uvxy,xyuv->", hbar['ab'], ref.rdms['ab'], optimize=True)
            + 0.25 * np.einsum("uvxy,xyuv->", hbar['bb'], ref.rdms['bb'], optimize=True)
            - np.einsum("uvxy,xu,yv->", hbar['aa'], ref.gam1['a'], ref.gam1['a'], optimize=True)
            - 2.0 * np.einsum("uvxy,xu,yv->", hbar['ab'], ref.gam1['a'], ref.gam1['b'], optimize=True)
            - np.einsum("uvxy,xu,yv->", hbar['bb'], ref.gam1['b'], ref.gam1['b'], optimize=True)
    )
    print(f"    <HBar_2> = {-e_scalar_2}")

    e_scalar = -e_scalar_1 - e_scalar_2

    hbar['a'] -= (
                    np.einsum("vxuy,xy->vu", hbar['aa'], ref.gam1['a'])
                    + np.einsum("vxuy,xy->vu", hbar['ab'], ref.gam1['b'])
    )

    hbar['b'] -= (
                    np.einsum("vxuy,xy->vu", hbar['bb'], ref.gam1['b'])
                    + np.einsum("xvyu,xy->vu", hbar['ab'], ref.gam1['a'])
    )

    return hbar, e_scalar

# def mr_ldsrg2_reference_relaxation(self, _eri, herm=True):
#     hbar_aa = self.hbar_1b[self.active,self.active].copy()
#     hbar_aaaa = self.hbar_2b[self.active,self.active,self.active,self.active].copy()
#
#     self.relax_e_scalar = (-np.einsum('vu,uv->', hbar_aa, self.cumulants['gamma1'])
#                            -0.25*np.einsum('xyuv,uvxy->',hbar_aaaa,self.rdms['2rdm'])
#                            +np.einsum('xyuv,ux,vy->',hbar_aaaa,self.cumulants['gamma1'],self.cumulants['gamma1'])
#
#     hbar_aa -= np.einsum('uyvx,xy->uv',hbar_aaaa,self.cumulants['gamma1'])
#
#     # For now, all things to do with CASCI are in the physicist's notation
#     hbar_aa = np.conjugate(hbar_aa)
#     hbar_aaaa = np.conjugate(hbar_aaaa)
#
#     hbar_aa_canon = np.einsum('ip,pq,jq->ij', np.conj(self.semicanonicalizer_active), hbar_aa, (self.semicanonicalizer_active), optimize='optimal')
#     hbar_aaaa_canon = np.einsum('ip,jq,pqrs,kr,ls->ijkl', np.conj(self.semicanonicalizer_active), np.conj(self.semicanonicalizer_active), hbar_aaaa, (self.semicanonicalizer_active),(self.semicanonicalizer_active), optimize='optimal')
#
#     _ref_relax_hamil = form_cas_hamiltonian(hbar_aa_canon, hbar_aaaa_canon, self.det_strings, self.verbose, self.cas, dtype=self.dtype)
#     self.mr_ldsrg2_relax_eigvals, self.mr_ldsrg2_relax_eigvecs = np.linalg.eigh(_ref_relax_hamil)
#
#     self.e_relax = np.dot(self.mr_ldsrg2_relax_eigvals[self.state_avg], self.sa_weights) + self.relax_e_scalar
#     self.e_mr_ldsrg2_relaxed = self.e_mr_ldsrg2_tot_energy + self.e_relax
#     self.mr_ldsrg2_relax_eigvals_shifted = self.e_mr_ldsrg2_tot_energy + self.mr_ldsrg2_relax_eigvals + self.relax_e_scalar
