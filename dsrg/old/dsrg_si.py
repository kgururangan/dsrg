import time
import numpy as np
from dsrg.utilities import get_memory_usage, spin_label, spatial_index, regularized_denominator
from dsrg.old.spinintegrated_contractions import *
from dsrg.wicked_contractions.ldsrg2_contractions import *

REMOVE_L3 = False
USE_OPT = True

class DSRG:

    def __init__(self, ref, print_threshold=0.09):
        self.ref = ref
        self.print_threshold = print_threshold

    def form_denominators(self, s):
        nhole_alpha = self.ref.ncore_alpha + self.ref.nact_alpha
        npart_alpha = self.ref.nact_alpha + self.ref.nvirt_alpha
        nhole_beta = self.ref.ncore_beta + self.ref.nact_beta
        npart_beta = self.ref.nact_beta + self.ref.nvirt_beta

        eps_a = np.real(np.diagonal(self.ref.F['a']))
        eps_b = np.real(np.diagonal(self.ref.F['b']))

        self.d1a = np.zeros((npart_alpha, nhole_alpha))
        self.d1b = np.zeros((npart_beta, nhole_beta))
        self.d2aa = np.zeros((npart_alpha, npart_alpha, nhole_alpha, nhole_alpha))
        self.d2ab = np.zeros((npart_alpha, npart_beta, nhole_alpha, nhole_beta))
        self.d2bb = np.zeros((npart_beta, npart_beta, nhole_beta, nhole_beta))
        self.regdenom1a = np.zeros((npart_alpha, nhole_alpha))
        self.regdenom1b = np.zeros((npart_beta, nhole_beta))
        self.regdenom2aa = np.zeros((npart_alpha, npart_alpha, nhole_alpha, nhole_alpha))
        self.regdenom2ab = np.zeros((npart_alpha, npart_beta, nhole_alpha, nhole_beta))
        self.regdenom2bb = np.zeros((npart_beta, npart_beta, nhole_beta, nhole_beta))
        #
        for i in range(nhole_alpha):
            for a in range(npart_alpha):
                self.d1a[a, i] = eps_a[i] - eps_a[a + self.ref.ncore_alpha]
                self.regdenom1a[a, i] = regularized_denominator(eps_a[i] - eps_a[a + self.ref.ncore_alpha], s)
                for j in range(nhole_alpha):
                    for b in range(npart_alpha):
                        self.d2aa[a, b, i, j] = eps_a[i] + eps_a[j] - eps_a[a + self.ref.ncore_alpha] - eps_a[b + self.ref.ncore_alpha]
                        self.regdenom2aa[a, b, i, j] = regularized_denominator(eps_a[i] + eps_a[j] - eps_a[a + self.ref.ncore_alpha] - eps_a[b + self.ref.ncore_alpha], s)
                for j in range(nhole_beta):
                    for b in range(npart_beta):
                        self.d2ab[a, b, i, j] = eps_a[i] + eps_b[j] - eps_a[a + self.ref.ncore_alpha] - eps_b[b + self.ref.ncore_beta]
                        self.regdenom2ab[a, b, i, j] = regularized_denominator(eps_a[i] + eps_b[j] - eps_a[a + self.ref.ncore_alpha] - eps_b[b + self.ref.ncore_beta], s)
        for i in range(nhole_beta):
            for a in range(npart_beta):
                self.d1b[a, i] = eps_b[i] - eps_b[a + self.ref.ncore_beta]
                self.regdenom1b[a, i] = regularized_denominator(eps_b[i] - eps_b[a + self.ref.ncore_beta], s)
                for j in range(nhole_beta):
                    for b in range(npart_beta):
                        self.d2bb[a, b, i, j] = eps_b[i] + eps_b[j] - eps_b[a + self.ref.ncore_beta] - eps_b[b + self.ref.ncore_beta]
                        self.regdenom2bb[a, b, i, j] = regularized_denominator(eps_b[i] + eps_b[j] - eps_b[a + self.ref.ncore_beta] - eps_b[b + self.ref.ncore_beta], s)
        
        self.denom1a_exp = np.zeros((npart_alpha, nhole_alpha))
        self.denom1b_exp = np.zeros((npart_beta, nhole_beta))
        self.denom2aa_exp = np.zeros((npart_alpha, npart_alpha, nhole_alpha, nhole_alpha))
        self.denom2ab_exp = np.zeros((npart_alpha, npart_beta, nhole_alpha, nhole_beta))
        self.denom2bb_exp = np.zeros((npart_beta, npart_beta, nhole_beta, nhole_beta))
        for i in range(nhole_alpha):
            for a in range(npart_alpha):
                self.denom1a_exp[a, i] = np.exp(-s * (eps_a[i] - eps_a[a + self.ref.ncore_alpha])**2)
                for j in range(nhole_alpha):
                    for b in range(npart_alpha):
                        self.denom2aa_exp[a, b, i, j] = np.exp(-s * (eps_a[i] + eps_a[j] - eps_a[a + self.ref.ncore_alpha] - eps_a[b + self.ref.ncore_alpha])**2)
                for j in range(nhole_beta):
                    for b in range(npart_beta):
                        self.denom2ab_exp[a, b, i, j] = np.exp(-s * (eps_a[i] + eps_b[j] - eps_a[a + self.ref.ncore_alpha] - eps_b[b + self.ref.ncore_beta])**2)
        for i in range(nhole_beta):
            for a in range(npart_beta):
                self.denom1b_exp[a, i] = np.exp(-s * (eps_b[i] - eps_b[a + self.ref.ncore_beta])**2)
                for j in range(nhole_beta):
                    for b in range(npart_beta):
                        self.denom2bb_exp[a, b, i, j] = np.exp(-s * (eps_b[i] + eps_b[j] - eps_b[a + self.ref.ncore_beta] - eps_b[b + self.ref.ncore_beta])**2)

    def compute_energy_pt2(self, s, herm=True):
        print("    ==> DSRG-MRPT(2) Correction <==")
        print("")
        t_start = time.time()

        self.form_denominators(s)

        h = self.ref.orbspace['hole_alpha']
        p = self.ref.orbspace['particle_alpha']
        a = self.ref.orbspace['active_alpha']
        ha = self.ref.orbspace['hole_active_alpha']
        hc = self.ref.orbspace['hole_core_alpha']
        pa = self.ref.orbspace['particle_active_alpha']
        pv = self.ref.orbspace['particle_virt_alpha']
        H = self.ref.orbspace['hole_beta']
        P = self.ref.orbspace['particle_beta']
        A = self.ref.orbspace['active_beta']
        hA = self.ref.orbspace['hole_active_beta']
        hC = self.ref.orbspace['hole_core_beta']
        pA = self.ref.orbspace['particle_active_beta']
        pV = self.ref.orbspace['particle_virt_beta']

        # 1st-order t1, t2 amplitudes
        tic = time.time()
        self.T = {}
        self.T['aa'] = self.ref.V['aa'][p, p, h, h] * self.regdenom2aa
        self.T['ab'] = self.ref.V['ab'][p, p, h, h] * self.regdenom2ab
        self.T['bb'] = self.ref.V['bb'][p, p, h, h] * self.regdenom2bb
        self.T['aa'][pa, pa, ha, ha] *= 0.
        self.T['ab'][pa, pA, ha, hA] *= 0.
        self.T['bb'][pA, pA, hA, hA] *= 0.
        self.T['a'] = (
                         self.ref.F['a'][p, h] 
                       + np.einsum("axiu,ux,ux->ai", self.T['aa'][:, pa, :, ha], self.d1a[pa, ha], self.ref.gam1['a'], optimize=True)
                       + np.einsum("axiu,ux,ux->ai", self.T['ab'][:, pA, :, hA], self.d1b[pA, hA], self.ref.gam1['b'], optimize=True)
        )
        self.T['a'] *= self.regdenom1a
        self.T['a'][pa, ha] *= 0.
        self.T['b'] = (
                         self.ref.F['b'][P, H] 
                       + np.einsum("axiu,ux,ux->ai", self.T['bb'][:, pA, :, hA], self.d1b[pA, hA], self.ref.gam1['b'], optimize=True)
                       + np.einsum("xaui,ux,ux->ai", self.T['ab'][pA, :, hA, :], self.d1a[pa, ha], self.ref.gam1['a'], optimize=True)
        )
        self.T['b'] *= self.regdenom1b
        self.T['b'][pA, hA] *= 0.
        toc = time.time()
        print(f'   ... compute first-order amplitudes: {toc - tic}s')
        # modified first-order Fock matrix
        tic = time.time()
        f1a = (
                  self.ref.F['a'][h, p] 
                + self.ref.F['a'][h, p] * self.denom1a_exp.T
                + np.einsum("axiu,ux,ux->ia", self.T['aa'][:, pa, :, ha], self.d1a[pa, ha], self.ref.gam1['a'], optimize=True) * self.denom1a_exp.T
                + np.einsum("axiu,ux,ux->ia", self.T['ab'][:, pA, :, hA], self.d1b[pA, hA], self.ref.gam1['b'], optimize=True) * self.denom1a_exp.T
        )
        f1b = (
                  self.ref.F['b'][H, P] 
                + self.ref.F['b'][H, P] * self.denom1b_exp.T
                + np.einsum("axiu,ux,ux->ia", self.T['bb'][:, pA, :, hA], self.d1b[pA, hA], self.ref.gam1['b'], optimize=True) * self.denom1b_exp.T
                + np.einsum("xaui,ux,ux->ia", self.T['ab'][pa, :, ha, :], self.d1a[pa, ha], self.ref.gam1['a'], optimize=True) * self.denom1b_exp.T
        )
        toc = time.time()
        print(f'   ... renormalize F: {toc - tic}s')
        # modified 2-electron integrals
        tic = time.time()
        v1aa = self.ref.V['aa'][h, h, p, p] + self.ref.V['aa'][h, h, p, p] * self.denom2aa_exp.transpose(2, 3, 0, 1)
        v1ab = self.ref.V['ab'][h, H, p, P] + self.ref.V['ab'][h, H, p, P] * self.denom2ab_exp.transpose(2, 3, 0, 1)
        v1bb = self.ref.V['bb'][H, H, P, P] + self.ref.V['bb'][H, H, P, P] * self.denom2bb_exp.transpose(2, 3, 0, 1)
        toc = time.time()
        print(f'   ... renormalize V: {toc - tic}s')

        #
        # Compute MRPT(2) energy
        #
        tic = time.time()
        Hbar = {'a': np.zeros_like(self.ref.F['a']), 
                'b': np.zeros_like(self.ref.F['b']),
                'aa': np.zeros_like(self.ref.V['aa']), 
                'ab': np.zeros_like(self.ref.V['ab']), 
                'bb': np.zeros_like(self.ref.V['bb'])}
        Hbar['a'][h, p] = f1a.copy()
        Hbar['b'][H, P] = f1b.copy()
        Hbar['aa'][h, h, p, p] = v1aa.copy()
        Hbar['ab'][h, H, p, P] = v1ab.copy()
        Hbar['bb'][H, H, P, P] = v1bb.copy()
        #
        e = {'0': 0.}
        # [H1, T1]
        e = h1a_t1a_c0(e, Hbar, self.T, self.ref.gam1, self.ref.eta1, self.ref.lambdas, self.ref.orbspace)
        e = h1b_t1b_c0(e, Hbar, self.T, self.ref.gam1, self.ref.eta1, self.ref.lambdas, self.ref.orbspace)
        # [H1, T2]
        e = h1a_t2a_c0(e, Hbar, self.T, self.ref.gam1, self.ref.eta1, self.ref.lambdas, self.ref.orbspace)
        e = h1a_t2b_c0(e, Hbar, self.T, self.ref.gam1, self.ref.eta1, self.ref.lambdas, self.ref.orbspace)
        e = h1b_t2b_c0(e, Hbar, self.T, self.ref.gam1, self.ref.eta1, self.ref.lambdas, self.ref.orbspace)
        e = h1b_t2c_c0(e, Hbar, self.T, self.ref.gam1, self.ref.eta1, self.ref.lambdas, self.ref.orbspace)
        # [H2, T1]
        e = h2a_t1a_c0(e, Hbar, self.T, self.ref.gam1, self.ref.eta1, self.ref.lambdas, self.ref.orbspace)
        e = h2b_t1a_c0(e, Hbar, self.T, self.ref.gam1, self.ref.eta1, self.ref.lambdas, self.ref.orbspace)
        e = h2b_t1b_c0(e, Hbar, self.T, self.ref.gam1, self.ref.eta1, self.ref.lambdas, self.ref.orbspace)
        e = h2c_t1b_c0(e, Hbar, self.T, self.ref.gam1, self.ref.eta1, self.ref.lambdas, self.ref.orbspace)
        # [H2, T2]
        e = h2a_t2a_c0(e, Hbar, self.T, self.ref.gam1, self.ref.eta1, self.ref.lambdas, self.ref.orbspace)
        e = h2a_t2b_c0(e, Hbar, self.T, self.ref.gam1, self.ref.eta1, self.ref.lambdas, self.ref.orbspace)
        e = h2b_t2a_c0(e, Hbar, self.T, self.ref.gam1, self.ref.eta1, self.ref.lambdas, self.ref.orbspace)
        e = h2b_t2b_c0(e, Hbar, self.T, self.ref.gam1, self.ref.eta1, self.ref.lambdas, self.ref.orbspace)
        e = h2b_t2c_c0(e, Hbar, self.T, self.ref.gam1, self.ref.eta1, self.ref.lambdas, self.ref.orbspace)
        e = h2c_t2b_c0(e, Hbar, self.T, self.ref.gam1, self.ref.eta1, self.ref.lambdas, self.ref.orbspace)
        e = h2c_t2c_c0(e, Hbar, self.T, self.ref.gam1, self.ref.eta1, self.ref.lambdas, self.ref.orbspace)
        #
        if not herm: e['0'] /= 2.0
        self.e_dsrg_mrpt2 = e['0']
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
        self.print_amplitudes(self.print_threshold)
        print("")
        print("    DSRG-MRPT(2) calculation completed in {:.2f}m {:.2f}s".format(minutes, seconds))
        print(f"    Memory usage: {get_memory_usage()} MB")
        print("")

    def run_ldsrg2(self, s, maxiter=80, herm=True, conv=1.0e-09, max_ncomm=12):
        print("    ==> MR-LDSRG(2) Amplitude Equations <==")
        print(f"     USE_OPT = {USE_OPT}")
        if REMOVE_L3:
            print("   WARNING: SETTING LAMBDA3 = 0")
            self.ref.lambdas['aaa'] *= 0.0
            self.ref.lambdas['aab'] *= 0.0
            self.ref.lambdas['abb'] *= 0.0
            self.ref.lambdas['bbb'] *= 0.0
        print("")
        t_start = time.time()

        # Slicing
        h = self.ref.orbspace['hole_alpha']
        p = self.ref.orbspace['particle_alpha']
        a = self.ref.orbspace['active_alpha']
        ha = self.ref.orbspace['hole_active_alpha']
        hc = self.ref.orbspace['hole_core_alpha']
        pa = self.ref.orbspace['particle_active_alpha']
        pv = self.ref.orbspace['particle_virt_alpha']
        H = self.ref.orbspace['hole_beta']
        P = self.ref.orbspace['particle_beta']
        A = self.ref.orbspace['active_beta']
        hA = self.ref.orbspace['hole_active_beta']
        hC = self.ref.orbspace['hole_core_beta']
        pA = self.ref.orbspace['particle_active_beta']
        pV = self.ref.orbspace['particle_virt_beta']

        # Form 1- and 2-body (regularized) energy denominators
        self.form_denominators(s)

        # 1st-order t1, t2 amplitudes
        tic = time.time()
        self.T = {}
        self.T['aa'] = self.ref.V['aa'][p, p, h, h] * self.regdenom2aa
        self.T['ab'] = self.ref.V['ab'][p, p, h, h] * self.regdenom2ab
        self.T['bb'] = self.ref.V['bb'][p, p, h, h] * self.regdenom2bb
        self.T['aa'][pa, pa, ha, ha] *= 0.
        self.T['ab'][pa, pA, ha, hA] *= 0.
        self.T['bb'][pA, pA, hA, hA] *= 0.
        self.T['a'] = (
                         self.ref.F['a'][p, h] 
                       + np.einsum("axiu,ux,ux->ai", self.T['aa'][:, pa, :, ha], self.d1a[pa, ha], self.ref.gam1['a'], optimize=True)
                       + np.einsum("axiu,ux,ux->ai", self.T['ab'][:, pA, :, hA], self.d1b[pA, hA], self.ref.gam1['b'], optimize=True)
        )
        self.T['a'] *= self.regdenom1a
        self.T['a'][pa, ha] *= 0.
        self.T['b'] = (
                         self.ref.F['b'][P, H] 
                       + np.einsum("axiu,ux,ux->ai", self.T['bb'][:, pA, :, hA], self.d1b[pA, hA], self.ref.gam1['b'], optimize=True)
                       + np.einsum("xaui,ux,ux->ai", self.T['ab'][pA, :, hA, :], self.d1a[pa, ha], self.ref.gam1['a'], optimize=True)
        )
        self.T['b'] *= self.regdenom1b
        self.T['b'][pA, hA] *= 0.
        toc = time.time()
        print(f'   ... compute first-order amplitudes: {toc - tic}s')

        print(f'   ... allocating Hbar arrays')
        self.hbar = {
                '0': 0.0,
                'a': self.ref.F['a'].copy(),
                'b': self.ref.F['b'].copy(),
                'aa': self.ref.V['aa'].copy(),
                'ab': self.ref.V['ab'].copy(),
                'bb': self.ref.V['bb'].copy()
        }
        o = { 
                '0': 0.0,
                'a': np.zeros_like(self.ref.F['a']),
                'b': np.zeros_like(self.ref.F['b']),
                'aa': np.zeros_like(self.ref.V['aa']),
                'ab': np.zeros_like(self.ref.V['ab']),
                'bb': np.zeros_like(self.ref.V['bb'])
        }
        #o_old = {
        #        '0': 0.0,
        #        'a': self.ref.F['a'].copy(),
        #        'b': self.ref.F['b'].copy(),
        #        'aa': self.ref.V['aa'].copy(),
        #        'ab': self.ref.V['ab'].copy(),
        #        'bb': self.ref.V['bb'].copy()
        #}

        #
        # LDSRG(2) iterations
        #
        t_start = time.time()
        converged = False
        e_old = .0
        it = 0
        print("")
        print("     Iter               Energy                 |dE|               |HBar|  Ncomm     Wall Time     Memory")
        while it < maxiter:
            tic = time.perf_counter()
            # Compute HBar
            ncomm, resid = self.compute_hbar(o, max_ncomm, herm=herm)
            energy = self.hbar['0']
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
            self.T['a'] = (self.hbar['a'][p, h] + self.T['a'] * self.d1a) * self.regdenom1a
            self.T['a'][pa, ha] = .0
            self.T['b'] = (self.hbar['b'][P, H] + self.T['b'] * self.d1b) * self.regdenom1b
            self.T['b'][pA, hA] = .0
            self.T['aa'] = (self.hbar['aa'][p, p, h, h] + self.T['aa'] * self.d2aa) * self.regdenom2aa
            self.T['aa'][pa, pa, ha, ha] = .0
            self.T['ab'] = (self.hbar['ab'][p, P, h, H] + self.T['ab'] * self.d2ab) * self.regdenom2ab
            self.T['ab'][pa, pA, ha, hA] = .0
            self.T['bb'] = (self.hbar['bb'][P, P, h, H] + self.T['bb'] * self.d2bb) * self.regdenom2bb
            self.T['bb'][pA, pA, hA, hA] = .0
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
        self.print_amplitudes(self.print_threshold)
        print("")
        print("    MR-LDSRG(2) calculation completed in {:.2f}m {:.2f}s".format(minutes, seconds))
        print(f"    Memory usage: {get_memory_usage()} MB")
        print("")

    def compute_hbar(self, o, max_ncomm, herm, verbose=False):
        self.hbar = {
                '0': 0.0,
                'a': self.ref.F['a'].copy(),
                'b': self.ref.F['b'].copy(),
                'aa': self.ref.V['aa'].copy(),
                'ab': self.ref.V['ab'].copy(),
                'bb': self.ref.V['bb'].copy()
        }

        o_old = {
                '0': 0.0,
                'a': self.ref.F['a'].copy(),
                'b': self.ref.F['b'].copy(),
                'aa': self.ref.V['aa'].copy(),
                'ab': self.ref.V['ab'].copy(),
                'bb': self.ref.V['bb'].copy()
        }
        for key in o.keys():
            o[key] *= 0.0

        ncomm = 0
        while ncomm < max_ncomm:
            o['0'] = 0.0
            if USE_OPT:
                # 0-body (energy)
                _t0 = time.time()
                o = h2a_t2a_c0(o, o_old, self.T, self.ref.gam1, self.ref.eta1, self.ref.lambdas, self.ref.orbspace, scale=2.0 if herm else 1.0)
                o = h2a_t2b_c0(o, o_old, self.T, self.ref.gam1, self.ref.eta1, self.ref.lambdas, self.ref.orbspace, scale=2.0 if herm else 1.0)
                o = h2b_t2a_c0(o, o_old, self.T, self.ref.gam1, self.ref.eta1, self.ref.lambdas, self.ref.orbspace, scale=2.0 if herm else 1.0)
                o = h2b_t2b_c0(o, o_old, self.T, self.ref.gam1, self.ref.eta1, self.ref.lambdas, self.ref.orbspace, scale=2.0 if herm else 1.0)
                o = h2b_t2c_c0(o, o_old, self.T, self.ref.gam1, self.ref.eta1, self.ref.lambdas, self.ref.orbspace, scale=2.0 if herm else 1.0)
                o = h2c_t2b_c0(o, o_old, self.T, self.ref.gam1, self.ref.eta1, self.ref.lambdas, self.ref.orbspace, scale=2.0 if herm else 1.0)
                o = h2c_t2c_c0(o, o_old, self.T, self.ref.gam1, self.ref.eta1, self.ref.lambdas, self.ref.orbspace, scale=2.0 if herm else 1.0)
                o = h_t_c0(o, o_old, self.T, self.ref.gam1, self.ref.eta1, self.ref.lambdas, self.ref.orbspace, scale=2.0 if herm else 1.0)
                # print(f"time for zerobody {time.time() - _t0}")
                # 1-body
                _t0 = time.time()
                o = h1_t1_c1(o, o_old, self.T, self.ref.gam1, self.ref.eta1, self.ref.lambdas, self.ref.orbspace)
                o = h1_t2_c1(o, o_old, self.T, self.ref.gam1, self.ref.eta1, self.ref.lambdas, self.ref.orbspace)
                o = h2_t1_c1(o, o_old, self.T, self.ref.gam1, self.ref.eta1, self.ref.lambdas, self.ref.orbspace)
                o = h2_t2_c1(o, o_old, self.T, self.ref.gam1, self.ref.eta1, self.ref.lambdas, self.ref.orbspace)
                # print(f"time for onebody {time.time() - _t0}")
                # 2-body
                _t0 = time.time()
                o = h1_t2_c2(o, o_old, self.T, self.ref.gam1, self.ref.eta1, self.ref.lambdas, self.ref.orbspace)
                o = h2_t1_c2(o, o_old, self.T, self.ref.gam1, self.ref.eta1, self.ref.lambdas, self.ref.orbspace)
                o = h2_t2_c2(o, o_old, self.T, self.ref.gam1, self.ref.eta1, self.ref.lambdas, self.ref.orbspace)
                # print(f"time for twobody {time.time() - _t0}")
            else:
                _t0 = time.time()
                o = h1a_t1a_c0(o, o_old, self.T, self.ref.gam1, self.ref.eta1, self.ref.lambdas, self.ref.orbspace, scale=2.0 if herm else 1.0)
                o = h1b_t1b_c0(o, o_old, self.T, self.ref.gam1, self.ref.eta1, self.ref.lambdas, self.ref.orbspace, scale=2.0 if herm else 1.0)
                o = h1a_t2a_c0(o, o_old, self.T, self.ref.gam1, self.ref.eta1, self.ref.lambdas, self.ref.orbspace, scale=2.0 if herm else 1.0)
                o = h1a_t2b_c0(o, o_old, self.T, self.ref.gam1, self.ref.eta1, self.ref.lambdas, self.ref.orbspace, scale=2.0 if herm else 1.0)
                o = h1b_t2b_c0(o, o_old, self.T, self.ref.gam1, self.ref.eta1, self.ref.lambdas, self.ref.orbspace, scale=2.0 if herm else 1.0)
                o = h1b_t2c_c0(o, o_old, self.T, self.ref.gam1, self.ref.eta1, self.ref.lambdas, self.ref.orbspace, scale=2.0 if herm else 1.0)
                o = h2a_t1a_c0(o, o_old, self.T, self.ref.gam1, self.ref.eta1, self.ref.lambdas, self.ref.orbspace, scale=2.0 if herm else 1.0)
                o = h2b_t1a_c0(o, o_old, self.T, self.ref.gam1, self.ref.eta1, self.ref.lambdas, self.ref.orbspace, scale=2.0 if herm else 1.0)
                o = h2b_t1b_c0(o, o_old, self.T, self.ref.gam1, self.ref.eta1, self.ref.lambdas, self.ref.orbspace, scale=2.0 if herm else 1.0)
                o = h2c_t1b_c0(o, o_old, self.T, self.ref.gam1, self.ref.eta1, self.ref.lambdas, self.ref.orbspace, scale=2.0 if herm else 1.0)
                o = h2a_t2a_c0(o, o_old, self.T, self.ref.gam1, self.ref.eta1, self.ref.lambdas, self.ref.orbspace, scale=2.0 if herm else 1.0)
                o = h2a_t2b_c0(o, o_old, self.T, self.ref.gam1, self.ref.eta1, self.ref.lambdas, self.ref.orbspace, scale=2.0 if herm else 1.0)
                o = h2b_t2a_c0(o, o_old, self.T, self.ref.gam1, self.ref.eta1, self.ref.lambdas, self.ref.orbspace, scale=2.0 if herm else 1.0)
                o = h2b_t2b_c0(o, o_old, self.T, self.ref.gam1, self.ref.eta1, self.ref.lambdas, self.ref.orbspace, scale=2.0 if herm else 1.0)
                o = h2b_t2c_c0(o, o_old, self.T, self.ref.gam1, self.ref.eta1, self.ref.lambdas, self.ref.orbspace, scale=2.0 if herm else 1.0)
                o = h2c_t2b_c0(o, o_old, self.T, self.ref.gam1, self.ref.eta1, self.ref.lambdas, self.ref.orbspace, scale=2.0 if herm else 1.0)
                o = h2c_t2c_c0(o, o_old, self.T, self.ref.gam1, self.ref.eta1, self.ref.lambdas, self.ref.orbspace, scale=2.0 if herm else 1.0)
                # print(f"energy took {time.time() - _t0}")
                ### onebody
                # c1a
                _t0 = time.time()
                o = h1a_t1a_c1a(o, o_old, self.T, self.ref.gam1, self.ref.eta1, self.ref.lambdas, self.ref.orbspace)
                o = h1a_t2a_c1a(o, o_old, self.T, self.ref.gam1, self.ref.eta1, self.ref.lambdas, self.ref.orbspace)
                o = h1b_t2b_c1a(o, o_old, self.T, self.ref.gam1, self.ref.eta1, self.ref.lambdas, self.ref.orbspace)
                o = h2a_t1a_c1a(o, o_old, self.T, self.ref.gam1, self.ref.eta1, self.ref.lambdas, self.ref.orbspace)
                o = h2b_t1b_c1a(o, o_old, self.T, self.ref.gam1, self.ref.eta1, self.ref.lambdas, self.ref.orbspace)
                o = h2a_t2a_c1a(o, o_old, self.T, self.ref.gam1, self.ref.eta1, self.ref.lambdas, self.ref.orbspace)
                o = h2a_t2b_c1a(o, o_old, self.T, self.ref.gam1, self.ref.eta1, self.ref.lambdas, self.ref.orbspace)
                o = h2b_t2a_c1a(o, o_old, self.T, self.ref.gam1, self.ref.eta1, self.ref.lambdas, self.ref.orbspace)
                o = h2b_t2b_c1a(o, o_old, self.T, self.ref.gam1, self.ref.eta1, self.ref.lambdas, self.ref.orbspace)
                o = h2b_t2c_c1a(o, o_old, self.T, self.ref.gam1, self.ref.eta1, self.ref.lambdas, self.ref.orbspace)
                o = h2c_t2b_c1a(o, o_old, self.T, self.ref.gam1, self.ref.eta1, self.ref.lambdas, self.ref.orbspace)
                # print(f"c1a took {time.time() - _t0}")
                # c1b
                _t0 = time.time()
                o = h1b_t1b_c1b(o, o_old, self.T, self.ref.gam1, self.ref.eta1, self.ref.lambdas, self.ref.orbspace)
                o = h1b_t2c_c1b(o, o_old, self.T, self.ref.gam1, self.ref.eta1, self.ref.lambdas, self.ref.orbspace)
                o = h1a_t2b_c1b(o, o_old, self.T, self.ref.gam1, self.ref.eta1, self.ref.lambdas, self.ref.orbspace)
                o = h2c_t1b_c1b(o, o_old, self.T, self.ref.gam1, self.ref.eta1, self.ref.lambdas, self.ref.orbspace)
                o = h2b_t1a_c1b(o, o_old, self.T, self.ref.gam1, self.ref.eta1, self.ref.lambdas, self.ref.orbspace)
                o = h2c_t2c_c1b(o, o_old, self.T, self.ref.gam1, self.ref.eta1, self.ref.lambdas, self.ref.orbspace)
                o = h2c_t2b_c1b(o, o_old, self.T, self.ref.gam1, self.ref.eta1, self.ref.lambdas, self.ref.orbspace)
                o = h2b_t2c_c1b(o, o_old, self.T, self.ref.gam1, self.ref.eta1, self.ref.lambdas, self.ref.orbspace)
                o = h2b_t2b_c1b(o, o_old, self.T, self.ref.gam1, self.ref.eta1, self.ref.lambdas, self.ref.orbspace)
                o = h2b_t2a_c1b(o, o_old, self.T, self.ref.gam1, self.ref.eta1, self.ref.lambdas, self.ref.orbspace)
                o = h2a_t2b_c1b(o, o_old, self.T, self.ref.gam1, self.ref.eta1, self.ref.lambdas, self.ref.orbspace)
                # print(f"c1b took {time.time() - _t0}")
                ### twobody
                # c2a
                _t0 = time.time()
                o = h1a_t2a_c2a(o, o_old, self.T, self.ref.gam1, self.ref.eta1, self.ref.lambdas, self.ref.orbspace)
                o = h2a_t1a_c2a(o, o_old, self.T, self.ref.gam1, self.ref.eta1, self.ref.lambdas, self.ref.orbspace)
                o = h2a_t2a_c2a(o, o_old, self.T, self.ref.gam1, self.ref.eta1, self.ref.lambdas, self.ref.orbspace)
                o = h2b_t2b_c2a(o, o_old, self.T, self.ref.gam1, self.ref.eta1, self.ref.lambdas, self.ref.orbspace)
                # print(f"c2a took {time.time() - _t0}")
                # c2b
                _t0 = time.time()
                o = h1a_t2b_c2b(o, o_old, self.T, self.ref.gam1, self.ref.eta1, self.ref.lambdas, self.ref.orbspace)
                o = h1b_t2b_c2b(o, o_old, self.T, self.ref.gam1, self.ref.eta1, self.ref.lambdas, self.ref.orbspace)
                o = h2b_t1a_c2b(o, o_old, self.T, self.ref.gam1, self.ref.eta1, self.ref.lambdas, self.ref.orbspace)
                o = h2b_t1b_c2b(o, o_old, self.T, self.ref.gam1, self.ref.eta1, self.ref.lambdas, self.ref.orbspace)
                o = h2a_t2b_c2b(o, o_old, self.T, self.ref.gam1, self.ref.eta1, self.ref.lambdas, self.ref.orbspace)
                o = h2b_t2a_c2b(o, o_old, self.T, self.ref.gam1, self.ref.eta1, self.ref.lambdas, self.ref.orbspace)
                o = h2b_t2b_c2b(o, o_old, self.T, self.ref.gam1, self.ref.eta1, self.ref.lambdas, self.ref.orbspace)
                o = h2b_t2c_c2b(o, o_old, self.T, self.ref.gam1, self.ref.eta1, self.ref.lambdas, self.ref.orbspace)
                o = h2c_t2b_c2b(o, o_old, self.T, self.ref.gam1, self.ref.eta1, self.ref.lambdas, self.ref.orbspace)
                # print(f"c2b took {time.time() - _t0}")
                # c2c
                _t0 = time.time()
                o = h1b_t2c_c2c(o, o_old, self.T, self.ref.gam1, self.ref.eta1, self.ref.lambdas, self.ref.orbspace)
                o = h2c_t1b_c2c(o, o_old, self.T, self.ref.gam1, self.ref.eta1, self.ref.lambdas, self.ref.orbspace)
                o = h2c_t2c_c2c(o, o_old, self.T, self.ref.gam1, self.ref.eta1, self.ref.lambdas, self.ref.orbspace)
                o = h2b_t2b_c2c(o, o_old, self.T, self.ref.gam1, self.ref.eta1, self.ref.lambdas, self.ref.orbspace)
                # print(f"c2c took {time.time() - _t0}")
            # antisymmetrize twobody
            o['aa'] -= o['aa'].transpose(1, 0, 2, 3)
            o['aa'] -= o['aa'].transpose(0, 1, 3, 2)
            o['bb'] -= o['bb'].transpose(1, 0, 2, 3)
            o['bb'] -= o['bb'].transpose(0, 1, 3, 2)
            if herm: 
                o['a'] += o['a'].T.conj()
                o['b'] += o['b'].T.conj()
                o['aa'] += o['aa'].transpose(2, 3, 0, 1)
                o['ab'] += o['ab'].transpose(2, 3, 0, 1)
                o['bb'] += o['bb'].transpose(2, 3, 0, 1)
            ncomm += 1
            for key, value in o.items():
                o[key] /= ncomm
            residual_op = np.sqrt((o['a']**2).sum() + (o['b']**2).sum() + (o['aa']**2).sum() + (o['ab']**2).sum() + (o['bb']**2).sum())
            if verbose:
                print(f"    {ncomm}   |C| = {residual_op}")
            if residual_op < 1e-12: 
                break
            # Increment many-body components of Hbar
            for key, value in o.items():
                self.hbar[key] += value
            # Store old delta H
            for key, value in o.items():
                o_old[key] = o[key].copy()
            # reset delta H
            for key, value in o.items():
                o[key] = np.zeros_like(o_old[key])
        ### zerobody
        return ncomm, residual_op

    def print_amplitudes(self, thresh_print):

        nua, nub, noa, nob = self.T['ab'].shape

        # Zero out the non-unique T amplitudes related by permutational symmetry
        for a in range(nua):
            for b in range(a + 1, nua):
                for i in range(noa):
                    for j in range(i + 1, noa):
                        self.T['aa'][b, a, j, i] = 0.0
                        self.T['aa'][a, b, j, i] = 0.0
                        self.T['aa'][b, a, i, j] = 0.0
        for a in range(nub):
            for b in range(a + 1, nub):
                for i in range(nob):
                    for j in range(i + 1, nob):
                        self.T['bb'][b, a, j, i] = 0.0
                        self.T['bb'][a, b, j, i] = 0.0
                        self.T['bb'][b, a, i, j] = 0.0

        print("\n   Largest Singly and Doubly Excited Amplitudes:")
        n = 1
        for a in range(nua):
            for i in range(noa):
                if abs(self.T['a'][a, i]) <= thresh_print: continue
                print(
                    "      [{}]     {}A  ->  {}A   =   {:.6f}".format(
                        n,
                        i + self.ref.nfrozen + 1,
                        a + self.ref.nfrozen + self.ref.ncore_alpha + 1,
                        self.T['a'][a, i],
                    )
                )
                n += 1
        for a in range(nub):
            for i in range(nob):
                if abs(self.T['b'][a, i]) <= thresh_print: continue
                print(
                    "      [{}]     {}B  ->  {}B   =   {:.6f}".format(
                        n,
                        i + self.ref.nfrozen + 1,
                        a + self.ref.nfrozen + self.ref.ncore_beta + 1,
                        self.T['b'][a, i],
                    )
                )
                n += 1
        for a in range(nua):
            for b in range(a + 1, nua):
                for i in range(noa):
                    for j in range(i + 1, noa):
                        if abs(self.T['aa'][a, b, i, j]) <= thresh_print: continue
                        print(
                            "      [{}]     {}A  {}A  ->  {}A  {}A  =   {:.6f}".format(
                                n,
                                i + self.ref.nfrozen + 1,
                                j + self.ref.nfrozen + 1,
                                a + self.ref.ncore_alpha + self.ref.nfrozen + 1,
                                b + self.ref.ncore_alpha + self.ref.nfrozen + 1,
                                self.T['aa'][a, b, i, j],
                            )
                        )
                        n += 1
        for a in range(nub):
            for b in range(a + 1, nub):
                for i in range(nob):
                    for j in range(i + 1, nob):
                        if abs(self.T['bb'][a, b, i, j]) <= thresh_print: continue
                        print(
                            "      [{}]     {}B  {}B  ->  {}B  {}B  =   {:.6f}".format(
                                n,
                                i + self.ref.nfrozen + 1,
                                j + self.ref.nfrozen + 1,
                                a + self.ref.ncore_beta + self.ref.nfrozen + 1,
                                b + self.ref.ncore_beta + self.ref.nfrozen + 1,
                                self.T['bb'][a, b, i, j],
                            )
                        )
                        n += 1
        for a in range(nua):
            for b in range(nub):
                for i in range(noa):
                    for j in range(nob):
                        if abs(self.T['ab'][a, b, i, j]) <= thresh_print: continue
                        print(
                            "      [{}]     {}A  {}B  ->  {}A  {}B  =   {:.6f}".format(
                                n,
                                i + self.ref.nfrozen + 1,
                                j + self.ref.nfrozen + 1,
                                a + self.ref.ncore_alpha + self.ref.nfrozen + 1,
                                b + self.ref.ncore_beta + self.ref.nfrozen + 1,
                                self.T['ab'][a, b, i, j],
                            )
                        )
                        n += 1
        # Restore permutationally redundant amplitudes
        for a in range(nua):
            for b in range(a + 1, nua):
                for i in range(noa):
                    for j in range(i + 1, noa):
                        self.T['aa'][b, a, j, i] = self.T['aa'][a, b, i, j]
                        self.T['aa'][a, b, j, i] = -1.0 * self.T['aa'][a, b, i, j]
                        self.T['aa'][b, a, i, j] = -1.0 * self.T['aa'][a, b, i, j]
        for a in range(nub):
            for b in range(a + 1, nub):
                for i in range(nob):
                    for j in range(i + 1, nob):
                        self.T['bb'][b, a, j, i] = self.T['bb'][a, b, i, j]
                        self.T['bb'][a, b, j, i] = -1.0 * self.T['bb'][a, b, i, j]
                        self.T['bb'][b, a, i, j] = -1.0 * self.T['bb'][a, b, i, j]

######### DSRG 3 #########
class DSRG3:

    def __init__(self, ref, print_threshold=0.09):
        self.ref = ref
        self.print_threshold = print_threshold

    def form_denominators(self, s):
        nhole_alpha = self.ref.ncore_alpha + self.ref.nact_alpha
        npart_alpha = self.ref.nact_alpha + self.ref.nvirt_alpha
        nhole_beta = self.ref.ncore_beta + self.ref.nact_beta
        npart_beta = self.ref.nact_beta + self.ref.nvirt_beta

        eps_a = np.real(np.diagonal(self.ref.F['a']))
        eps_b = np.real(np.diagonal(self.ref.F['b']))

        self.d1a = np.zeros((npart_alpha, nhole_alpha))
        self.d1b = np.zeros((npart_beta, nhole_beta))
        self.d2aa = np.zeros((npart_alpha, npart_alpha, nhole_alpha, nhole_alpha))
        self.d2ab = np.zeros((npart_alpha, npart_beta, nhole_alpha, nhole_beta))
        self.d2bb = np.zeros((npart_beta, npart_beta, nhole_beta, nhole_beta))
        self.d3aaa = np.zeros((npart_alpha, npart_alpha, npart_alpha, nhole_alpha, nhole_alpha, nhole_alpha))
        self.d3aab = np.zeros((npart_alpha, npart_alpha, npart_beta, nhole_alpha, nhole_alpha, nhole_beta))
        self.d3abb = np.zeros((npart_alpha, npart_beta, npart_beta, nhole_alpha, nhole_beta, nhole_beta))
        self.d3bbb = np.zeros((npart_beta, npart_beta, npart_beta, nhole_beta, nhole_beta, nhole_beta))
        self.regdenom1a = np.zeros((npart_alpha, nhole_alpha))
        self.regdenom1b = np.zeros((npart_beta, nhole_beta))
        self.regdenom2aa = np.zeros((npart_alpha, npart_alpha, nhole_alpha, nhole_alpha))
        self.regdenom2ab = np.zeros((npart_alpha, npart_beta, nhole_alpha, nhole_beta))
        self.regdenom2bb = np.zeros((npart_beta, npart_beta, nhole_beta, nhole_beta))
        self.regdenom3aaa = np.zeros((npart_alpha, npart_alpha, npart_alpha, nhole_alpha, nhole_alpha, nhole_alpha))
        self.regdenom3aab = np.zeros((npart_alpha, npart_alpha, npart_beta, nhole_alpha, nhole_alpha, nhole_beta))
        self.regdenom3abb = np.zeros((npart_alpha, npart_beta, npart_beta, nhole_alpha, nhole_beta, nhole_beta))
        self.regdenom3bbb = np.zeros((npart_beta, npart_beta, npart_beta, nhole_beta, nhole_beta, nhole_beta))
        #
        for i in range(nhole_alpha):
            for a in range(npart_alpha):
                self.d1a[a, i] = eps_a[i] - eps_a[a + self.ref.ncore_alpha]
                self.regdenom1a[a, i] = regularized_denominator(eps_a[i] - eps_a[a + self.ref.ncore_alpha], s)
                for j in range(nhole_alpha):
                    for b in range(npart_alpha):
                        self.d2aa[a, b, i, j] = eps_a[i] + eps_a[j] - eps_a[a + self.ref.ncore_alpha] - eps_a[b + self.ref.ncore_alpha]
                        self.regdenom2aa[a, b, i, j] = regularized_denominator(eps_a[i] + eps_a[j] - eps_a[a + self.ref.ncore_alpha] - eps_a[b + self.ref.ncore_alpha], s)
                for j in range(nhole_beta):
                    for b in range(npart_beta):
                        self.d2ab[a, b, i, j] = eps_a[i] + eps_b[j] - eps_a[a + self.ref.ncore_alpha] - eps_b[b + self.ref.ncore_beta]
                        self.regdenom2ab[a, b, i, j] = regularized_denominator(eps_a[i] + eps_b[j] - eps_a[a + self.ref.ncore_alpha] - eps_b[b + self.ref.ncore_beta], s)
        for i in range(nhole_beta):
            for a in range(npart_beta):
                self.d1b[a, i] = eps_b[i] - eps_b[a + self.ref.ncore_beta]
                self.regdenom1b[a, i] = regularized_denominator(eps_b[i] - eps_b[a + self.ref.ncore_beta], s)
                for j in range(nhole_beta):
                    for b in range(npart_beta):
                        self.d2bb[a, b, i, j] = eps_b[i] + eps_b[j] - eps_b[a + self.ref.ncore_beta] - eps_b[b + self.ref.ncore_beta]
                        self.regdenom2bb[a, b, i, j] = regularized_denominator(eps_b[i] + eps_b[j] - eps_b[a + self.ref.ncore_beta] - eps_b[b + self.ref.ncore_beta], s)
        for i in range(nhole_alpha):
            for a in range(npart_alpha):
                for j in range(nhole_alpha):
                    for b in range(npart_alpha):
                        for k in range(nhole_alpha):
                            for c in range(npart_alpha):
                                self.d3aaa[a, b, c, i, j, k] = (eps_a[i] + eps_a[j] + eps_a[k] 
                                                                - eps_a[a + self.ref.ncore_alpha] - eps_a[b + self.ref.ncore_alpha] - eps_a[c + self.ref.ncore_alpha])
                                self.regdenom3aaa[a, b, c, i, j, k] = regularized_denomator(self.d3aaa[a, b, c, i, j, k], s)
                        for k in range(nhole_beta):
                            for c in range(npart_beta):
                                self.d3aab[a, b, c, i, j, k] = (eps_a[i] + eps_a[j] + eps_b[k] 
                                                                - eps_a[a + self.ref.ncore_alpha] - eps_a[b + self.ref.ncore_alpha] - eps_b[c + self.ref.ncore_beta])
                                self.regdenom3aab[a, b, c, i, j, k] = regularized_denomator(self.d3aab[a, b, c, i, j, k], s)
        for j in range(nhole_alpha):
            for b in range(npart_alpha):
                for k in range(nhole_alpha):
                    for c in range(npart_alpha):
                        for i in range(nhole_alpha):
                            for a in range(npart_alpha):
                                self.d3abb[a, b, c, i, j, k] = (eps_a[i] + eps_b[j] + eps_b[k] 
                                                                - eps_a[a + self.ref.ncore_alpha] - eps_b[b + self.ref.ncore_beta] - eps_b[c + self.ref.ncore_beta])
                                self.regdenom3abb[a, b, c, i, j, k] = regularized_denomator(self.d3abb[a, b, c, i, j, k], s)
                        for i in range(nhole_beta):
                            for a in range(npart_beta):
                                self.d3bbb[a, b, c, i, j, k] = (eps_b[i] + eps_b[j] + eps_b[k] 
                                                                - eps_b[a + self.ref.ncore_alpha] - eps_b[b + self.ref.ncore_alpha] - eps_b[c + self.ref.ncore_beta])
                                self.regdenom3bbb[a, b, c, i, j, k] = regularized_denomator(self.d3bbb[a, b, c, i, j, k], s)

    def run_ldsrg3(self, s, maxiter=80, herm=True, conv=1.0e-09, max_ncomm=12):
        print("    ==> MR-LDSRG(3) Amplitude Equations <==")
        print("")
        t_start = time.time()

        # Slicing
        h = self.ref.orbspace['hole_alpha']
        p = self.ref.orbspace['particle_alpha']
        a = self.ref.orbspace['active_alpha']
        ha = self.ref.orbspace['hole_active_alpha']
        hc = self.ref.orbspace['hole_core_alpha']
        pa = self.ref.orbspace['particle_active_alpha']
        pv = self.ref.orbspace['particle_virt_alpha']
        H = self.ref.orbspace['hole_beta']
        P = self.ref.orbspace['particle_beta']
        A = self.ref.orbspace['active_beta']
        hA = self.ref.orbspace['hole_active_beta']
        hC = self.ref.orbspace['hole_core_beta']
        pA = self.ref.orbspace['particle_active_beta']
        pV = self.ref.orbspace['particle_virt_beta']

        norb = self.ref.F.shape[0]

        # Form 1- and 2-body (regularized) energy denominators
        self.form_denominators(s)

        # 1st-order t1, t2 amplitudes
        tic = time.time()
        self.T = {}
        self.T['aa'] = self.ref.V['aa'][p, p, h, h] * self.regdenom2aa
        self.T['ab'] = self.ref.V['ab'][p, p, h, h] * self.regdenom2ab
        self.T['bb'] = self.ref.V['bb'][p, p, h, h] * self.regdenom2bb
        self.T['aa'][pa, pa, ha, ha] *= 0.
        self.T['ab'][pa, pA, ha, hA] *= 0.
        self.T['bb'][pA, pA, hA, hA] *= 0.
        self.T['a'] = (
                         self.ref.F['a'][p, h] 
                       + np.einsum("axiu,ux,ux->ai", self.T['aa'][:, pa, :, ha], self.d1a[pa, ha], self.ref.gam1['a'], optimize=True)
                       + np.einsum("axiu,ux,ux->ai", self.T['ab'][:, pA, :, hA], self.d1b[pA, hA], self.ref.gam1['b'], optimize=True)
        )
        self.T['a'] *= self.regdenom1a
        self.T['a'][pa, ha] *= 0.
        self.T['b'] = (
                         self.ref.F['b'][P, H] 
                       + np.einsum("axiu,ux,ux->ai", self.T['bb'][:, pA, :, hA], self.d1b[pA, hA], self.ref.gam1['b'], optimize=True)
                       + np.einsum("xaui,ux,ux->ai", self.T['ab'][pA, :, hA, :], self.d1a[pa, ha], self.ref.gam1['a'], optimize=True)
        )
        self.T['b'] *= self.regdenom1b
        self.T['b'][pA, hA] *= 0.
        toc = time.time()
        print(f'   ... compute first-order amplitudes: {toc - tic}s')

        print(f'   ... allocating Hbar arrays')
        self.hbar = {
                '0': 0.0,
                'a': self.ref.F['a'].copy(),
                'b': self.ref.F['b'].copy(),
                'aa': self.ref.V['aa'].copy(),
                'ab': self.ref.V['ab'].copy(),
                'bb': self.ref.V['bb'].copy(),
                'aaa': np.zeros((norb, norb, norb, norb, norb, norb)),
                'aab': np.zeros((norb, norb, norb, norb, norb, norb)),
                'abb': np.zeros((norb, norb, norb, norb, norb, norb)),
                'bbb': np.zeros((norb, norb, norb, norb, norb, norb)),
        }
        o = { 
                '0': 0.0,
                'a': np.zeros_like(self.ref.F['a']),
                'b': np.zeros_like(self.ref.F['b']),
                'aa': np.zeros_like(self.ref.V['aa']),
                'ab': np.zeros_like(self.ref.V['ab']),
                'bb': np.zeros_like(self.ref.V['bb']),
                'aaa': np.zeros((norb, norb, norb, norb, norb, norb)),
                'aab': np.zeros((norb, norb, norb, norb, norb, norb)),
                'abb': np.zeros((norb, norb, norb, norb, norb, norb)),
                'bbb': np.zeros((norb, norb, norb, norb, norb, norb)),
        }

        #
        # LDSRG(3) iterations
        #
        t_start = time.time()
        converged = False
        e_old = .0
        it = 0
        print("")
        print("     Iter               Energy                 |dE|               |HBar|  Ncomm     Wall Time     Memory")
        while it < maxiter:
            tic = time.perf_counter()
            # Compute HBar
            ncomm, resid = self.compute_hbar(o, max_ncomm, herm=herm)
            energy = self.hbar['0']
            # Record iteration information
            delta_e = energy - e_old
            toc = time.perf_counter()
            minutes, seconds = divmod(toc - tic, 60)
            print("    {: 5d} {: 20.12f} {: 20.12f} {: 20.12f} {: 5d}    {:.2f}m {:.2f}s    {:.2f} MB".format(it, energy, delta_e, resid, ncomm, minutes, seconds, get_memory_usage()))
            if abs(delta_e) < conv:
                print(f"    MR-LDSRG(3) successfully converged after {it} iterations.")
                break
            e_old = energy.copy()
            # Update amplitudes
            self.T['a'] = (self.hbar['a'][p, h] + self.T['a'] * self.d1a) * self.regdenom1a
            self.T['a'][pa, ha] = .0
            self.T['b'] = (self.hbar['b'][P, H] + self.T['b'] * self.d1b) * self.regdenom1b
            self.T['b'][pA, hA] = .0
            self.T['aa'] = (self.hbar['aa'][p, p, h, h] + self.T['aa'] * self.d2aa) * self.regdenom2aa
            self.T['aa'][pa, pa, ha, ha] = .0
            self.T['ab'] = (self.hbar['ab'][p, P, h, H] + self.T['ab'] * self.d2ab) * self.regdenom2ab
            self.T['ab'][pa, pA, ha, hA] = .0
            self.T['bb'] = (self.hbar['bb'][P, P, h, H] + self.T['bb'] * self.d2bb) * self.regdenom2bb
            self.T['bb'][pA, pA, hA, hA] = .0
            self.T['aaa'] = (self.hbar['aaa'][p, p, p, h, h, h] + self.T['aaa'] * self.d3aaa) * self.regdenom3aaa
            self.T['aaa'][pa, pa, pa, ha, ha, ha] = .0
            self.T['aab'] = (self.hbar['aab'][p, p, P, h, h, H] + self.T['aab'] * self.d3aab) * self.regdenom3aab
            self.T['aab'][pa, pa, pA, ha, ha, hA] = .0
            self.T['abb'] = (self.hbar['abb'][p, P, P, h, H, H] + self.T['abb'] * self.d3abb) * self.regdenom3abb
            self.T['abb'][pa, pA, pA, ha, hA, hA] = .0
            self.T['bbb'] = (self.hbar['bbb'][P, P, P, H, H, H] + self.T['bbb'] * self.d3bbb) * self.regdenom3bbb
            self.T['bbb'][pA, pA, pA, hA, hA, hA] = .0
            # Update iteration counter
            it += 1
        else:
            print("   MR-LDSRG(3) did not converge")
        # Record the energy
        self.e_dsrg3 = energy
        # Record total time and print summary
        minutes, seconds = divmod(time.time() - t_start, 60)
        print("")
        print("    Reference Energy: {: 20.12f}".format(self.ref.e_cas))
        print("    Unrelaxed MR-LDSRG(3) Correlation Energy: {: 20.12f}".format(self.e_dsrg3))
        print("    Unrelaxed MR-LDSRG(3) Total Energy: {: 20.12f}".format(self.ref.e_cas + self.e_dsrg3))
        print("")
        print("    Largest Singly and Doubly Excited Amplitudes")
        print("    --------------------------------------------")
        self.print_amplitudes(self.print_threshold)
        print("")
        print("    MR-LDSRG(3) calculation completed in {:.2f}m {:.2f}s".format(minutes, seconds))
        print(f"    Memory usage: {get_memory_usage()} MB")
        print("")

    def compute_hbar(self, o, max_ncomm, herm, verbose=False):
        self.hbar = {
                '0': 0.0,
                'a': self.ref.F['a'].copy(),
                'b': self.ref.F['b'].copy(),
                'aa': self.ref.V['aa'].copy(),
                'ab': self.ref.V['ab'].copy(),
                'bb': self.ref.V['bb'].copy(),
        }

        o_old = {
                '0': 0.0,
                'a': self.ref.F['a'].copy(),
                'b': self.ref.F['b'].copy(),
                'aa': self.ref.V['aa'].copy(),
                'ab': self.ref.V['ab'].copy(),
                'bb': self.ref.V['bb'].copy(),
        }
        for key in o.keys():
            o[key] *= 0.0

        ncomm = 0
        while ncomm < max_ncomm:
            o['0'] = 0.0
            if USE_OPT:
                # 0-body (energy)
                _t0 = time.time()
                o = h2a_t2a_c0(o, o_old, self.T, self.ref.gam1, self.ref.eta1, self.ref.lambdas, self.ref.orbspace, scale=2.0 if herm else 1.0)
                o = h2a_t2b_c0(o, o_old, self.T, self.ref.gam1, self.ref.eta1, self.ref.lambdas, self.ref.orbspace, scale=2.0 if herm else 1.0)
                o = h2b_t2a_c0(o, o_old, self.T, self.ref.gam1, self.ref.eta1, self.ref.lambdas, self.ref.orbspace, scale=2.0 if herm else 1.0)
                o = h2b_t2b_c0(o, o_old, self.T, self.ref.gam1, self.ref.eta1, self.ref.lambdas, self.ref.orbspace, scale=2.0 if herm else 1.0)
                o = h2b_t2c_c0(o, o_old, self.T, self.ref.gam1, self.ref.eta1, self.ref.lambdas, self.ref.orbspace, scale=2.0 if herm else 1.0)
                o = h2c_t2b_c0(o, o_old, self.T, self.ref.gam1, self.ref.eta1, self.ref.lambdas, self.ref.orbspace, scale=2.0 if herm else 1.0)
                o = h2c_t2c_c0(o, o_old, self.T, self.ref.gam1, self.ref.eta1, self.ref.lambdas, self.ref.orbspace, scale=2.0 if herm else 1.0)
                o = h_t_c0(o, o_old, self.T, self.ref.gam1, self.ref.eta1, self.ref.lambdas, self.ref.orbspace, scale=2.0 if herm else 1.0)
                # print(f"time for zerobody {time.time() - _t0}")
                # 1-body
                _t0 = time.time()
                o = h1_t1_c1(o, o_old, self.T, self.ref.gam1, self.ref.eta1, self.ref.lambdas, self.ref.orbspace)
                o = h1_t2_c1(o, o_old, self.T, self.ref.gam1, self.ref.eta1, self.ref.lambdas, self.ref.orbspace)
                o = h2_t1_c1(o, o_old, self.T, self.ref.gam1, self.ref.eta1, self.ref.lambdas, self.ref.orbspace)
                o = h2_t2_c1(o, o_old, self.T, self.ref.gam1, self.ref.eta1, self.ref.lambdas, self.ref.orbspace)
                # print(f"time for onebody {time.time() - _t0}")
                # 2-body
                _t0 = time.time()
                o = h1_t2_c2(o, o_old, self.T, self.ref.gam1, self.ref.eta1, self.ref.lambdas, self.ref.orbspace)
                o = h2_t1_c2(o, o_old, self.T, self.ref.gam1, self.ref.eta1, self.ref.lambdas, self.ref.orbspace)
                o = h2_t2_c2(o, o_old, self.T, self.ref.gam1, self.ref.eta1, self.ref.lambdas, self.ref.orbspace)
                # print(f"time for twobody {time.time() - _t0}")
            else:
                _t0 = time.time()
                o = h1a_t1a_c0(o, o_old, self.T, self.ref.gam1, self.ref.eta1, self.ref.lambdas, self.ref.orbspace, scale=2.0 if herm else 1.0)
                o = h1b_t1b_c0(o, o_old, self.T, self.ref.gam1, self.ref.eta1, self.ref.lambdas, self.ref.orbspace, scale=2.0 if herm else 1.0)
                o = h1a_t2a_c0(o, o_old, self.T, self.ref.gam1, self.ref.eta1, self.ref.lambdas, self.ref.orbspace, scale=2.0 if herm else 1.0)
                o = h1a_t2b_c0(o, o_old, self.T, self.ref.gam1, self.ref.eta1, self.ref.lambdas, self.ref.orbspace, scale=2.0 if herm else 1.0)
                o = h1b_t2b_c0(o, o_old, self.T, self.ref.gam1, self.ref.eta1, self.ref.lambdas, self.ref.orbspace, scale=2.0 if herm else 1.0)
                o = h1b_t2c_c0(o, o_old, self.T, self.ref.gam1, self.ref.eta1, self.ref.lambdas, self.ref.orbspace, scale=2.0 if herm else 1.0)
                o = h2a_t1a_c0(o, o_old, self.T, self.ref.gam1, self.ref.eta1, self.ref.lambdas, self.ref.orbspace, scale=2.0 if herm else 1.0)
                o = h2b_t1a_c0(o, o_old, self.T, self.ref.gam1, self.ref.eta1, self.ref.lambdas, self.ref.orbspace, scale=2.0 if herm else 1.0)
                o = h2b_t1b_c0(o, o_old, self.T, self.ref.gam1, self.ref.eta1, self.ref.lambdas, self.ref.orbspace, scale=2.0 if herm else 1.0)
                o = h2c_t1b_c0(o, o_old, self.T, self.ref.gam1, self.ref.eta1, self.ref.lambdas, self.ref.orbspace, scale=2.0 if herm else 1.0)
                o = h2a_t2a_c0(o, o_old, self.T, self.ref.gam1, self.ref.eta1, self.ref.lambdas, self.ref.orbspace, scale=2.0 if herm else 1.0)
                o = h2a_t2b_c0(o, o_old, self.T, self.ref.gam1, self.ref.eta1, self.ref.lambdas, self.ref.orbspace, scale=2.0 if herm else 1.0)
                o = h2b_t2a_c0(o, o_old, self.T, self.ref.gam1, self.ref.eta1, self.ref.lambdas, self.ref.orbspace, scale=2.0 if herm else 1.0)
                o = h2b_t2b_c0(o, o_old, self.T, self.ref.gam1, self.ref.eta1, self.ref.lambdas, self.ref.orbspace, scale=2.0 if herm else 1.0)
                o = h2b_t2c_c0(o, o_old, self.T, self.ref.gam1, self.ref.eta1, self.ref.lambdas, self.ref.orbspace, scale=2.0 if herm else 1.0)
                o = h2c_t2b_c0(o, o_old, self.T, self.ref.gam1, self.ref.eta1, self.ref.lambdas, self.ref.orbspace, scale=2.0 if herm else 1.0)
                o = h2c_t2c_c0(o, o_old, self.T, self.ref.gam1, self.ref.eta1, self.ref.lambdas, self.ref.orbspace, scale=2.0 if herm else 1.0)
                print(f"energy took {time.time() - _t0}")
                ### onebody
                # c1a
                _t0 = time.time()
                o = h1a_t1a_c1a(o, o_old, self.T, self.ref.gam1, self.ref.eta1, self.ref.lambdas, self.ref.orbspace)
                o = h1a_t2a_c1a(o, o_old, self.T, self.ref.gam1, self.ref.eta1, self.ref.lambdas, self.ref.orbspace)
                o = h1b_t2b_c1a(o, o_old, self.T, self.ref.gam1, self.ref.eta1, self.ref.lambdas, self.ref.orbspace)
                o = h2a_t1a_c1a(o, o_old, self.T, self.ref.gam1, self.ref.eta1, self.ref.lambdas, self.ref.orbspace)
                o = h2b_t1b_c1a(o, o_old, self.T, self.ref.gam1, self.ref.eta1, self.ref.lambdas, self.ref.orbspace)
                o = h2a_t2a_c1a(o, o_old, self.T, self.ref.gam1, self.ref.eta1, self.ref.lambdas, self.ref.orbspace)
                o = h2a_t2b_c1a(o, o_old, self.T, self.ref.gam1, self.ref.eta1, self.ref.lambdas, self.ref.orbspace)
                o = h2b_t2a_c1a(o, o_old, self.T, self.ref.gam1, self.ref.eta1, self.ref.lambdas, self.ref.orbspace)
                o = h2b_t2b_c1a(o, o_old, self.T, self.ref.gam1, self.ref.eta1, self.ref.lambdas, self.ref.orbspace)
                o = h2b_t2c_c1a(o, o_old, self.T, self.ref.gam1, self.ref.eta1, self.ref.lambdas, self.ref.orbspace)
                o = h2c_t2b_c1a(o, o_old, self.T, self.ref.gam1, self.ref.eta1, self.ref.lambdas, self.ref.orbspace)
                print(f"c1a took {time.time() - _t0}")
                # c1b
                _t0 = time.time()
                o = h1b_t1b_c1b(o, o_old, self.T, self.ref.gam1, self.ref.eta1, self.ref.lambdas, self.ref.orbspace)
                o = h1b_t2c_c1b(o, o_old, self.T, self.ref.gam1, self.ref.eta1, self.ref.lambdas, self.ref.orbspace)
                o = h1a_t2b_c1b(o, o_old, self.T, self.ref.gam1, self.ref.eta1, self.ref.lambdas, self.ref.orbspace)
                o = h2c_t1b_c1b(o, o_old, self.T, self.ref.gam1, self.ref.eta1, self.ref.lambdas, self.ref.orbspace)
                o = h2b_t1a_c1b(o, o_old, self.T, self.ref.gam1, self.ref.eta1, self.ref.lambdas, self.ref.orbspace)
                o = h2c_t2c_c1b(o, o_old, self.T, self.ref.gam1, self.ref.eta1, self.ref.lambdas, self.ref.orbspace)
                o = h2c_t2b_c1b(o, o_old, self.T, self.ref.gam1, self.ref.eta1, self.ref.lambdas, self.ref.orbspace)
                o = h2b_t2c_c1b(o, o_old, self.T, self.ref.gam1, self.ref.eta1, self.ref.lambdas, self.ref.orbspace)
                o = h2b_t2b_c1b(o, o_old, self.T, self.ref.gam1, self.ref.eta1, self.ref.lambdas, self.ref.orbspace)
                o = h2b_t2a_c1b(o, o_old, self.T, self.ref.gam1, self.ref.eta1, self.ref.lambdas, self.ref.orbspace)
                o = h2a_t2b_c1b(o, o_old, self.T, self.ref.gam1, self.ref.eta1, self.ref.lambdas, self.ref.orbspace)
                print(f"c1b took {time.time() - _t0}")
                ### twobody
                # c2a
                _t0 = time.time()
                o = h1a_t2a_c2a(o, o_old, self.T, self.ref.gam1, self.ref.eta1, self.ref.lambdas, self.ref.orbspace)
                o = h2a_t1a_c2a(o, o_old, self.T, self.ref.gam1, self.ref.eta1, self.ref.lambdas, self.ref.orbspace)
                o = h2a_t2a_c2a(o, o_old, self.T, self.ref.gam1, self.ref.eta1, self.ref.lambdas, self.ref.orbspace)
                o = h2b_t2b_c2a(o, o_old, self.T, self.ref.gam1, self.ref.eta1, self.ref.lambdas, self.ref.orbspace)
                print(f"c2a took {time.time() - _t0}")
                # c2b
                _t0 = time.time()
                o = h1a_t2b_c2b(o, o_old, self.T, self.ref.gam1, self.ref.eta1, self.ref.lambdas, self.ref.orbspace)
                o = h1b_t2b_c2b(o, o_old, self.T, self.ref.gam1, self.ref.eta1, self.ref.lambdas, self.ref.orbspace)
                o = h2b_t1a_c2b(o, o_old, self.T, self.ref.gam1, self.ref.eta1, self.ref.lambdas, self.ref.orbspace)
                o = h2b_t1b_c2b(o, o_old, self.T, self.ref.gam1, self.ref.eta1, self.ref.lambdas, self.ref.orbspace)
                o = h2a_t2b_c2b(o, o_old, self.T, self.ref.gam1, self.ref.eta1, self.ref.lambdas, self.ref.orbspace)
                o = h2b_t2a_c2b(o, o_old, self.T, self.ref.gam1, self.ref.eta1, self.ref.lambdas, self.ref.orbspace)
                o = h2b_t2b_c2b(o, o_old, self.T, self.ref.gam1, self.ref.eta1, self.ref.lambdas, self.ref.orbspace)
                o = h2b_t2c_c2b(o, o_old, self.T, self.ref.gam1, self.ref.eta1, self.ref.lambdas, self.ref.orbspace)
                o = h2c_t2b_c2b(o, o_old, self.T, self.ref.gam1, self.ref.eta1, self.ref.lambdas, self.ref.orbspace)
                print(f"c2b took {time.time() - _t0}")
                # c2c
                _t0 = time.time()
                o = h1b_t2c_c2c(o, o_old, self.T, self.ref.gam1, self.ref.eta1, self.ref.lambdas, self.ref.orbspace)
                o = h2c_t1b_c2c(o, o_old, self.T, self.ref.gam1, self.ref.eta1, self.ref.lambdas, self.ref.orbspace)
                o = h2c_t2c_c2c(o, o_old, self.T, self.ref.gam1, self.ref.eta1, self.ref.lambdas, self.ref.orbspace)
                o = h2b_t2b_c2c(o, o_old, self.T, self.ref.gam1, self.ref.eta1, self.ref.lambdas, self.ref.orbspace)
                print(f"c2c took {time.time() - _t0}")
            # antisymmetrize twobody
            o['aa'] -= o['aa'].transpose(1, 0, 2, 3)
            o['aa'] -= o['aa'].transpose(0, 1, 3, 2)
            o['bb'] -= o['bb'].transpose(1, 0, 2, 3)
            o['bb'] -= o['bb'].transpose(0, 1, 3, 2)
            if herm: 
                o['a'] += o['a'].T.conj()
                o['b'] += o['b'].T.conj()
                o['aa'] += o['aa'].transpose(2, 3, 0, 1)
                o['ab'] += o['ab'].transpose(2, 3, 0, 1)
                o['bb'] += o['bb'].transpose(2, 3, 0, 1)
            ncomm += 1
            for key, value in o.items():
                o[key] /= ncomm
            residual_op = np.sqrt((o['a']**2).sum() + (o['b']**2).sum() + (o['aa']**2).sum() + (o['ab']**2).sum() + (o['bb']**2).sum())
            if verbose:
                print(f"    {ncomm}   |C| = {residual_op}")
            if residual_op < 1e-12: 
                break
            # Increment many-body components of Hbar
            for key, value in o.items():
                self.hbar[key] += value
            # Store old delta H
            for key, value in o.items():
                o_old[key] = o[key].copy()
            # reset delta H
            for key, value in o.items():
                o[key] = np.zeros_like(o_old[key])
        ### zerobody
        return ncomm, residual_op

    def print_amplitudes(self, thresh_print):

        nua, nub, noa, nob = self.T['ab'].shape

        # Zero out the non-unique T amplitudes related by permutational symmetry
        for a in range(nua):
            for b in range(a + 1, nua):
                for i in range(noa):
                    for j in range(i + 1, noa):
                        self.T['aa'][b, a, j, i] = 0.0
                        self.T['aa'][a, b, j, i] = 0.0
                        self.T['aa'][b, a, i, j] = 0.0
        for a in range(nub):
            for b in range(a + 1, nub):
                for i in range(nob):
                    for j in range(i + 1, nob):
                        self.T['bb'][b, a, j, i] = 0.0
                        self.T['bb'][a, b, j, i] = 0.0
                        self.T['bb'][b, a, i, j] = 0.0

        print("\n   Largest Singly and Doubly Excited Amplitudes:")
        n = 1
        for a in range(nua):
            for i in range(noa):
                if abs(self.T['a'][a, i]) <= thresh_print: continue
                print(
                    "      [{}]     {}A  ->  {}A   =   {:.6f}".format(
                        n,
                        i + self.ref.nfrozen + 1,
                        a + self.ref.nfrozen + self.ref.ncore_alpha + 1,
                        self.T['a'][a, i],
                    )
                )
                n += 1
        for a in range(nub):
            for i in range(nob):
                if abs(self.T['b'][a, i]) <= thresh_print: continue
                print(
                    "      [{}]     {}B  ->  {}B   =   {:.6f}".format(
                        n,
                        i + self.ref.nfrozen + 1,
                        a + self.ref.nfrozen + self.ref.ncore_beta + 1,
                        self.T['b'][a, i],
                    )
                )
                n += 1
        for a in range(nua):
            for b in range(a + 1, nua):
                for i in range(noa):
                    for j in range(i + 1, noa):
                        if abs(self.T['aa'][a, b, i, j]) <= thresh_print: continue
                        print(
                            "      [{}]     {}A  {}A  ->  {}A  {}A  =   {:.6f}".format(
                                n,
                                i + self.ref.nfrozen + 1,
                                j + self.ref.nfrozen + 1,
                                a + self.ref.ncore_alpha + self.ref.nfrozen + 1,
                                b + self.ref.ncore_alpha + self.ref.nfrozen + 1,
                                self.T['aa'][a, b, i, j],
                            )
                        )
                        n += 1
        for a in range(nub):
            for b in range(a + 1, nub):
                for i in range(nob):
                    for j in range(i + 1, nob):
                        if abs(self.T['bb'][a, b, i, j]) <= thresh_print: continue
                        print(
                            "      [{}]     {}B  {}B  ->  {}B  {}B  =   {:.6f}".format(
                                n,
                                i + self.ref.nfrozen + 1,
                                j + self.ref.nfrozen + 1,
                                a + self.ref.ncore_beta + self.ref.nfrozen + 1,
                                b + self.ref.ncore_beta + self.ref.nfrozen + 1,
                                self.T['bb'][a, b, i, j],
                            )
                        )
                        n += 1
        for a in range(nua):
            for b in range(nub):
                for i in range(noa):
                    for j in range(nob):
                        if abs(self.T['ab'][a, b, i, j]) <= thresh_print: continue
                        print(
                            "      [{}]     {}A  {}B  ->  {}A  {}B  =   {:.6f}".format(
                                n,
                                i + self.ref.nfrozen + 1,
                                j + self.ref.nfrozen + 1,
                                a + self.ref.ncore_alpha + self.ref.nfrozen + 1,
                                b + self.ref.ncore_beta + self.ref.nfrozen + 1,
                                self.T['ab'][a, b, i, j],
                            )
                        )
                        n += 1
        # Restore permutationally redundant amplitudes
        for a in range(nua):
            for b in range(a + 1, nua):
                for i in range(noa):
                    for j in range(i + 1, noa):
                        self.T['aa'][b, a, j, i] = self.T['aa'][a, b, i, j]
                        self.T['aa'][a, b, j, i] = -1.0 * self.T['aa'][a, b, i, j]
                        self.T['aa'][b, a, i, j] = -1.0 * self.T['aa'][a, b, i, j]
        for a in range(nub):
            for b in range(a + 1, nub):
                for i in range(nob):
                    for j in range(i + 1, nob):
                        self.T['bb'][b, a, j, i] = self.T['bb'][a, b, i, j]
                        self.T['bb'][a, b, j, i] = -1.0 * self.T['bb'][a, b, i, j]
                        self.T['bb'][b, a, i, j] = -1.0 * self.T['bb'][a, b, i, j]

