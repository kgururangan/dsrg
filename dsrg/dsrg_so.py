import time
import numpy as np
from dsrg.utilities import get_memory_usage
from dsrg.spinorbital_contractions import *

def spatial_index(p):
    if p % 2 == 0:
        return int(p / 2)
    else:
        return int((p + 1) / 2)

def spin_label(p):
    if p % 2 == 0:
        return "B"
    else:
        return "A"

def regularized_denominator(x, s):
    '''Compute the denomiator factor [1 - exp(s*x^2)]/x. For small
    values of s*x^2, apply Taylor expansion of exp(s*x^2). This allows
    one to recover the s -> infty limit.'''
    z = np.sqrt(s) * x
    if abs(z) <= 1.0e-09:
        return np.sqrt(s)*(z - z**3/2 + z**5/6)
    return (1. - np.exp(-s * x**2)) / x

class SpinDSRG:

    def __init__(self, ref, print_threshold=0.09):
        self.ref = ref
        self.print_threshold = print_threshold

    def form_denominators(self, s):
        nhole = self.ref.ncore + self.ref.nact
        npart = self.ref.nact + self.ref.nvirt

        eps = np.real(np.diagonal(self.ref.F))

        self.d1 = np.zeros((npart, nhole))
        self.d2 = np.zeros((npart, npart, nhole, nhole))
        self.denom1 = np.zeros((npart, nhole))
        self.denom2 = np.zeros((npart, npart, nhole, nhole))
        for i in range(nhole):
            for a in range(npart):
                self.d1[a, i] = eps[i] - eps[a + self.ref.ncore]
                self.denom1[a, i] = regularized_denominator(eps[i] - eps[a + self.ref.ncore], s)
                for j in range(nhole):
                    for b in range(npart):
                        self.d2[a, b, i, j] = eps[i] + eps[j] - eps[a + self.ref.ncore] - eps[b + self.ref.ncore]
                        self.denom2[a, b, i, j] = regularized_denominator(eps[i] + eps[j] - eps[a + self.ref.ncore] - eps[b + self.ref.ncore], s)
        
        self.denom1_exp = np.zeros((npart, nhole))
        self.denom2_exp = np.zeros((npart, npart, nhole, nhole))
        for i in range(nhole):
            for a in range(npart):
                self.denom1_exp[a, i] = np.exp(-s * (eps[i] - eps[a + self.ref.ncore])**2)
                for j in range(nhole):
                    for b in range(npart):
                        self.denom2_exp[a, b, i, j] = np.exp(-s * (eps[i] + eps[j] - eps[a + self.ref.ncore] - eps[b + self.ref.ncore])**2)

    def compute_energy_pt2(self, s, herm=True):
        print("    ==> DSRG-MRPT(2) Correction <==")
        print("")
        t_start = time.time()

        self.form_denominators(s)

        h = self.ref.orbspace['hole']
        p = self.ref.orbspace['particle']
        a = self.ref.orbspace['active']
        ha = self.ref.orbspace['hole_active']
        hc = self.ref.orbspace['hole_core']
        pa = self.ref.orbspace['particle_active']
        pv = self.ref.orbspace['particle_virt']

        # 1st-order t1, t2 amplitudes
        tic = time.time()
        self.t2 = self.ref.V[p, p, h, h] * self.denom2
        self.t2[pa, pa, ha, ha] *= 0.
        self.t1 = (self.ref.F[p, h] + np.einsum("axiu,ux,ux->ai", self.t2[:, pa, :, ha], self.d1[pa, ha], self.ref.gam1, optimize=True)) * self.denom1
        self.t1[pa, ha] *= 0.
        toc = time.time()
        print(f'   ... compute first-order amplitudes: {toc - tic}s')
        # modified first-order Fock matrix
        tic = time.time()
        f1 = (
                self.ref.F[h, p] 
                + self.ref.F[h, p] * self.denom1_exp.T
                + np.einsum("axiu,ux,ux->ia", self.t2[:, pa, :, ha], self.d1[pa, ha], self.ref.gam1, optimize=True) * self.denom1_exp.T
        )
        toc = time.time()
        print(f'   ... renormalize F: {toc - tic}s')
        # modified 2-electron integrals
        tic = time.time()
        v1 = self.ref.V[h, h, p, p] + self.ref.V[h, h, p, p] * self.denom2_exp.transpose(2, 3, 0, 1)
        toc = time.time()
        print(f'   ... renormalize V: {toc - tic}s')

        #
        # Compute MRPT(2) energy
        #
        tic = time.time()
        ept2 = 0.0
        ept2 += np.einsum("jb,ai,ij,ba->", f1, self.t1, self.ref.gam1_full[h, h], self.ref.eta1_full[p, p], optimize=True)
        ept2 += 0.5 * (
                np.einsum("xyev,eu,uvxy->", v1[ha, ha, pv, pa], self.t1[pv, ha], self.ref.lambdas['2'], optimize=True)
                - np.einsum("myuv,xm,uvxy->", v1[hc, ha, pa, pa], self.t1[pa, hc], self.ref.lambdas['2'], optimize=True)
        )
        ept2 += 0.5 * (
                np.einsum("xe,eyuv,uvxy->", f1[ha, pv], self.t2[pv, pa, ha, ha], self.ref.lambdas['2'], optimize=True)
                - np.einsum("mv,xyum,uvxy->", f1[hc, pa], self.t2[pa, pa, ha, hc], self.ref.lambdas['2'], optimize=True)
        )
        ept2 += 0.25 * np.einsum("klcd,abij,ik,jl,ac,db->", v1, self.t2, self.ref.gam1_full[h, h], self.ref.gam1_full[h, h], self.ref.eta1_full[p, p], self.ref.eta1_full[p, p], optimize=True)
        ept2 += 0.125 * (
                np.einsum("xycd,abuv,ca,db,uvxy->", v1[ha, ha, :, :], self.t2[:, :, ha, ha], self.ref.eta1_full[p, p], self.ref.eta1_full[p, p], self.ref.lambdas['2'], optimize=True)
                + np.einsum("kluv,xyij,ik,jl,uvxy->", v1[:, :, pa, pa], self.t2[pa, pa, :, :], self.ref.gam1_full[h, h], self.ref.gam1_full[h, h], self.ref.lambdas['2'], optimize=True)
        )
        ept2 += np.einsum("jxvb,ayiu,ij,ba,uvxy->", v1[:, ha, pa, :], self.t2[:, pa, :, ha], self.ref.gam1_full[h, h], self.ref.eta1_full[p, p], self.ref.lambdas['2'], optimize=True)
        if not herm: ept2 /= 2.0
        self.e_dsrg_mrpt2 = ept2

        #
        # The following also works (flipped)
        #
        #FF = np.zeros_like(self.ref.F)
        #FF[p, h] = f1.T # integrals are swapped in Wicked
        #VV = np.zeros_like(self.ref.V) 
        #VV[p, p, h, h] = v1.transpose(2, 3, 0, 1) # integrals are swapped in wicked
        # T1 and T2 are also swapped in Wicked
        #self.e_dsrg_mrpt2 = H_T_C0_flipped(FF, VV, self.t1.T, self.t2.transpose(2, 3, 0, 1), self.ref.gam1, self.ref.eta1, self.ref.lambdas, self.ref.orbspace)

        #
        # The following also works (not flipped)
        #
        #FF = np.zeros_like(self.ref.F)
        #FF[h, p] = f1.copy()
        #VV = np.zeros_like(self.ref.V) 
        #VV[h, h, p, p] = v1.copy()
        #self.e_dsrg_mrpt2 = H_T_C0(FF, VV, self.t1, self.t2, self.ref.gam1, self.ref.eta1, self.ref.lambdas, self.ref.orbspace)

        toc = time.time()
        print(f'   ... compute_energy_pt2: {toc - tic}s')
        minutes, seconds = divmod(time.time() - t_start, 60)
        print("")
        print("    Reference Energy: {: 20.12f}".format(self.ref.e_cas))
        print("    Unrelaxed DSRG-MRPT(2) Correlation Energy: {: 20.12f}".format(self.e_dsrg_mrpt2))
        print("    Unrelaxed DSRG-MRPT(2) Total Energy: {: 20.12f}".format(self.ref.e_cas + self.e_dsrg_mrpt2))
        print("")
        print("    Largest Singly and Doubly Excited Amplitudes")
        print("    --------------------------------------------")
        self.print_amplitudes()
        print("")
        print("    DSRG-MRPT(2) calculation completed in {:.2f}m {:.2f}s".format(minutes, seconds))
        print(f"    Memory usage: {get_memory_usage()} MB")
        print("")

    def run_ldsrg2(self, s, maxiter=80, herm=True, conv=1.0e-07, max_ncomm=12):
        print("    ==> MR-LDSRG(2) Amplitude Equations <==")
        print("")
        t_start = time.time()
        # Slicing arrays
        h = self.ref.orbspace['hole']
        p = self.ref.orbspace['particle']
        a = self.ref.orbspace['active']
        ha = self.ref.orbspace['hole_active']
        hc = self.ref.orbspace['hole_core']
        pa = self.ref.orbspace['particle_active']
        pv = self.ref.orbspace['particle_virt']
        # [WARNING]: Set 3-cumulant to 0
        print("  WARNING: Setting 3-cumulant to 0!")
        self.ref.lambdas['3'] *= 0.

        # Form 1- and 2-body (regularized) energy denominators
        self.form_denominators(s)

        # 1st-order t1, t2 amplitudes
        tic = time.time()
        self.t2 = self.ref.V[p, p, h, h] * self.denom2
        self.t2[pa, pa, ha, ha] *= 0.
        self.t1 = (self.ref.F[p, h] + np.einsum("axiu,ux,ux->ai", self.t2[:, pa, :, ha], self.d1[pa, ha], self.ref.gam1, optimize=True)) * self.denom1
        self.t1[pa, ha] *= 0.
        toc = time.time()
        print(f'   ... compute first-order amplitudes: {toc - tic}s')

        #
        # LDSRG(2) iterations
        #
        t_start = time.time()
        #hbar1 = self.ref.F.copy()
        #hbar2 = self.ref.V.copy()
        converged = False
        e_old = .0
        it = 0
        print("")
        print("     Iter               Energy                 |dE|               |HBar|  Ncomm     Wall Time     Memory")
        while it < maxiter:
            tic = time.perf_counter()
            # Compute HBar
            ncomm, energy, hbar1, hbar2, resid = self.compute_hbar(max_ncomm, herm=herm)
            # Record iteration information
            delta_e = energy - e_old
            toc = time.perf_counter()
            minutes, seconds = divmod(toc - tic, 60)
            print("    {: 5d} {: 20.12f} {: 20.12f} {: 20.12f} {: 5d}    {:.2f}m {:.2f}s    {:.2f} MB".format(it, energy, delta_e, resid, ncomm, minutes, seconds, get_memory_usage()))
            if abs(delta_e) < conv:
                print(f"    MR-LDSRG(2) successfully converged after {it} iterations.")
                break
            e_old = energy.copy()
            # Update amplitudes
            self.t1 = (hbar1[p, h] + self.t1 * self.d1) * self.denom1
            self.t2 = (hbar2[p, p, h, h] + self.t2 * self.d2) * self.denom2
            self.t1[pa, ha] = .0
            self.t2[pa, pa, ha, ha] = .0
            # Update iteration counter
            it += 1
        else:
            print("   MR-LDSRG(2) did not converge")
        # Record the energy
        self.e_dsrg2 = energy
        # Record total time and print summary
        minutes, seconds = divmod(time.time() - t_start, 60)
        print("")
        print("    Reference Energy: {: 20.12f}".format(self.ref.e_cas))
        print("    Unrelaxed MR-LDSRG(2) Correlation Energy: {: 20.12f}".format(self.e_dsrg2))
        print("    Unrelaxed MR-LDSRG(2) Total Energy: {: 20.12f}".format(self.ref.e_cas + self.e_dsrg2))
        print("")
        print("    Largest Singly and Doubly Excited Amplitudes")
        print("    --------------------------------------------")
        self.print_amplitudes()
        print("")
        print("    MR-LDSRG(2) calculation completed in {:.2f}m {:.2f}s".format(minutes, seconds))
        print(f"    Memory usage: {get_memory_usage()} MB")
        print("")

    def compute_hbar(self, max_ncomm, herm, verbose=False):
        energy = .0
        hbar1 = self.ref.F.copy()
        hbar2 = self.ref.V.copy()

        o1_old = self.ref.F.copy()
        o2_old = self.ref.V.copy()

        o0 = .0
        o1 = np.zeros_like(o1_old)
        o2 = np.zeros_like(o2_old)
        ncomm = 0
        while ncomm < max_ncomm:
            # zerobody
            # _t0 = time.time()
            o0 = 0.
            o0 = h1_t1_c0(o0, (o1_old, o2_old), (self.t1, self.t2), self.ref.gam1, self.ref.eta1, self.ref.lambdas, self.ref.orbspace, scale=2.0 if herm else 1.0)
            o0 = h1_t2_c0(o0, (o1_old, o2_old), (self.t1, self.t2), self.ref.gam1, self.ref.eta1, self.ref.lambdas, self.ref.orbspace, scale=2.0 if herm else 1.0)
            o0 = h2_t1_c0(o0, (o1_old, o2_old), (self.t1, self.t2), self.ref.gam1, self.ref.eta1, self.ref.lambdas, self.ref.orbspace, scale=2.0 if herm else 1.0)
            o0 = h2_t2_c0(o0, (o1_old, o2_old), (self.t1, self.t2), self.ref.gam1, self.ref.eta1, self.ref.lambdas, self.ref.orbspace, scale=2.0 if herm else 1.0)
            # print(f"energy took {time.time() - _t0}")
            # onebody
            # _t0 = time.time()
            o1 = h1_t1_c1(o1, (o1_old, o2_old), (self.t1, self.t2), self.ref.gam1, self.ref.eta1, self.ref.lambdas, self.ref.orbspace)
            o1 = h1_t2_c1(o1, (o1_old, o2_old), (self.t1, self.t2), self.ref.gam1, self.ref.eta1, self.ref.lambdas, self.ref.orbspace)
            o1 = h2_t1_c1(o1, (o1_old, o2_old), (self.t1, self.t2), self.ref.gam1, self.ref.eta1, self.ref.lambdas, self.ref.orbspace)
            o1 = h2_t2_c1(o1, (o1_old, o2_old), (self.t1, self.t2), self.ref.gam1, self.ref.eta1, self.ref.lambdas, self.ref.orbspace)
            # print(f"onebody took {time.time() - _t0}")
            # twobody
            # _t0 = time.time()
            o2 = h1_t2_c2(o2, (o1_old, o2_old), (self.t1, self.t2), self.ref.gam1, self.ref.eta1, self.ref.lambdas, self.ref.orbspace)
            o2 = h2_t1_c2(o2, (o1_old, o2_old), (self.t1, self.t2), self.ref.gam1, self.ref.eta1, self.ref.lambdas, self.ref.orbspace)
            o2 = h2_t2_c2(o2, (o1_old, o2_old), (self.t1, self.t2), self.ref.gam1, self.ref.eta1, self.ref.lambdas, self.ref.orbspace)
            # print(f"twobody took {time.time() - _t0}")
            # antisymmetrize twobody
            o2 -= o2.transpose(1, 0, 2, 3)
            o2 -= o2.transpose(0, 1, 3, 2)
            if herm: 
                o1 += o1.T.conj()
                o2 += o2.transpose(2, 3, 0, 1)
            ncomm += 1
            o0 /= ncomm
            o1 /= ncomm
            o2 /= ncomm
            residual_op = np.sqrt((o1**2).sum() + (o2**2).sum())
            if verbose:
                print(f"    {ncomm}   |C| = {residual_op}")
            if residual_op < 1e-12: 
                break
            # Increment many-body components of Hbar
            energy += o0
            hbar1 += o1
            hbar2 += o2
            # Store old delta H
            o0_old = o0
            o1_old = o1.copy()
            o2_old = o2.copy()
            # reset delta H
            o1 = np.zeros_like(o1_old)
            o2 = np.zeros_like(o2_old)
        return ncomm, energy, hbar1, hbar2, residual_op

    def print_amplitudes(self):
        nu, no = self.t1.shape
        n = 1
        print("          i -> a")
        for a in range(nu):
            for i in range(no):
                if abs(self.t1[a, i]) <= self.print_threshold: continue
                print(f"     [{n}]  {spatial_index(i + 1)}{spin_label(i + 1)} -> {spatial_index(a + self.ref.ncore + 1)}{spin_label(a + self.ref.ncore + 1)}    {self.t1[a, i]}") 
                n += 1
        print("          i j -> a b")
        for a in range(nu):
            for b in range(a + 1, nu):
                for i in range(no):
                    for j in range(i + 1, no):
                        if abs(self.t2[a, b, i, j]) <= self.print_threshold: continue
                        print(f"     [{n}]  {spatial_index(i + 1)}{spin_label(i + 1)} {spatial_index(j + 1)}{spin_label(j + 1)} -> {spatial_index(a + self.ref.ncore + 1)}{spin_label(a + self.ref.ncore + 1)} {spatial_index(b + self.ref.ncore + 1)}{spin_label(b + self.ref.ncore + 1)}    {self.t2[a, b, i, j]}") 
                        n += 1





















