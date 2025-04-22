import time
import numpy as np
from dsrg.utilities import get_memory_usage, spin_label, spatial_index, regularized_denominator
from dsrg.mrldsrg3_contractions import *

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
                                self.regdenom3aaa[a, b, c, i, j, k] = regularized_denominator(self.d3aaa[a, b, c, i, j, k], s)
                        for k in range(nhole_beta):
                            for c in range(npart_beta):
                                self.d3aab[a, b, c, i, j, k] = (eps_a[i] + eps_a[j] + eps_b[k] 
                                                                - eps_a[a + self.ref.ncore_alpha] - eps_a[b + self.ref.ncore_alpha] - eps_b[c + self.ref.ncore_beta])
                                self.regdenom3aab[a, b, c, i, j, k] = regularized_denominator(self.d3aab[a, b, c, i, j, k], s)
        for j in range(nhole_alpha):
            for b in range(npart_alpha):
                for k in range(nhole_alpha):
                    for c in range(npart_alpha):
                        for i in range(nhole_alpha):
                            for a in range(npart_alpha):
                                self.d3abb[a, b, c, i, j, k] = (eps_a[i] + eps_b[j] + eps_b[k] 
                                                                - eps_a[a + self.ref.ncore_alpha] - eps_b[b + self.ref.ncore_beta] - eps_b[c + self.ref.ncore_beta])
                                self.regdenom3abb[a, b, c, i, j, k] = regularized_denominator(self.d3abb[a, b, c, i, j, k], s)
                        for i in range(nhole_beta):
                            for a in range(npart_beta):
                                self.d3bbb[a, b, c, i, j, k] = (eps_b[i] + eps_b[j] + eps_b[k] 
                                                                - eps_b[a + self.ref.ncore_alpha] - eps_b[b + self.ref.ncore_alpha] - eps_b[c + self.ref.ncore_beta])
                                self.regdenom3bbb[a, b, c, i, j, k] = regularized_denominator(self.d3bbb[a, b, c, i, j, k], s)

    def run_ldsrg3(self, s, maxiter=80, herm=True, conv=1.0e-07, max_ncomm=12):
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

        norb = self.ref.F['a'].shape[0]

        # Form 1- and 2-body (regularized) energy denominators
        tic = time.time()
        self.form_denominators(s)
        toc = time.time()
        print(f'form n-body regularized denominators   ... : {toc - tic}s')

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

        nua, nub, noa, nob = self.T['ab'].shape
        self.T['aaa'] = np.zeros((nua, nua, nua, noa, noa, noa))
        self.T['aab'] = np.zeros((nua, nua, nub, noa, noa, nob))
        self.T['abb'] = np.zeros((nua, nub, nub, noa, nob, nob))
        self.T['bbb'] = np.zeros((nub, nub, nub, nob, nob, nob))
        toc = time.time()
        print(f'   ... compute first-order amplitudes: {toc - tic}s')

        tic = time.time()
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
                #'aaa': np.zeros((nua, nua, nua, noa, noa, noa)),
                #'aab': np.zeros((nua, nua, nub, noa, noa, nob)),
                #'abb': np.zeros((nua, nub, nub, noa, nob, nob)),
                #'bbb': np.zeros((nub, nub, nub, nob, nob, nob)),
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
                #'aaa': np.zeros((nua, nua, nua, noa, noa, noa)),
                #'aab': np.zeros((nua, nua, nub, noa, noa, nob)),
                #'abb': np.zeros((nua, nub, nub, noa, nob, nob)),
                #'bbb': np.zeros((nub, nub, nub, nob, nob, nob)),
        }
        toc = time.time()
        print(f'   ... allocate HBar arrays: {toc - tic}s')

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
            #self.T['aaa'] = (self.hbar['aaa'] + self.T['aaa'] * self.d3aaa) * self.regdenom3aaa
            self.T['aaa'][pa, pa, pa, ha, ha, ha] = .0
            self.T['aab'] = (self.hbar['aab'][p, p, P, h, h, H] + self.T['aab'] * self.d3aab) * self.regdenom3aab
            #self.T['aab'] = (self.hbar['aab'] + self.T['aab'] * self.d3aab) * self.regdenom3aab
            self.T['aab'][pa, pa, pA, ha, ha, hA] = .0
            self.T['abb'] = (self.hbar['abb'][p, P, P, h, H, H] + self.T['abb'] * self.d3abb) * self.regdenom3abb
            #self.T['abb'] = (self.hbar['abb'] + self.T['abb'] * self.d3abb) * self.regdenom3abb
            self.T['abb'][pa, pA, pA, ha, hA, hA] = .0
            self.T['bbb'] = (self.hbar['bbb'][P, P, P, H, H, H] + self.T['bbb'] * self.d3bbb) * self.regdenom3bbb
            #self.T['bbb'] = (self.hbar['bbb'] + self.T['bbb'] * self.d3bbb) * self.regdenom3bbb
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
        nua, nub, noa, nob = self.T['ab'].shape
        self.hbar = {
                '0': 0.0,
                'a': self.ref.F['a'].copy(),
                'b': self.ref.F['b'].copy(),
                'aa': self.ref.V['aa'].copy(),
                'ab': self.ref.V['ab'].copy(),
                'bb': self.ref.V['bb'].copy(),
                'aaa': np.zeros((self.ref.norb, self.ref.norb, self.ref.norb, self.ref.norb, self.ref.norb, self.ref.norb)),
                'aab': np.zeros((self.ref.norb, self.ref.norb, self.ref.norb, self.ref.norb, self.ref.norb, self.ref.norb)),
                'abb': np.zeros((self.ref.norb, self.ref.norb, self.ref.norb, self.ref.norb, self.ref.norb, self.ref.norb)),
                'bbb': np.zeros((self.ref.norb, self.ref.norb, self.ref.norb, self.ref.norb, self.ref.norb, self.ref.norb)),
                #'aaa': np.zeros((nua, nua, nua, noa, noa, noa)),
                #'aab': np.zeros((nua, nua, nub, noa, noa, nob)),
                #'abb': np.zeros((nua, nub, nub, noa, nob, nob)),
                #'bbb': np.zeros((nub, nub, nub, nob, nob, nob)),
        }

        o_old = {
                '0': 0.0,
                'a': self.ref.F['a'].copy(),
                'b': self.ref.F['b'].copy(),
                'aa': self.ref.V['aa'].copy(),
                'ab': self.ref.V['ab'].copy(),
                'bb': self.ref.V['bb'].copy(),
                'aaa': np.zeros((self.ref.norb, self.ref.norb, self.ref.norb, self.ref.norb, self.ref.norb, self.ref.norb)),
                'aab': np.zeros((self.ref.norb, self.ref.norb, self.ref.norb, self.ref.norb, self.ref.norb, self.ref.norb)),
                'abb': np.zeros((self.ref.norb, self.ref.norb, self.ref.norb, self.ref.norb, self.ref.norb, self.ref.norb)),
                'bbb': np.zeros((self.ref.norb, self.ref.norb, self.ref.norb, self.ref.norb, self.ref.norb, self.ref.norb)),
                #'aaa': np.zeros((nua, nua, nua, noa, noa, noa)),
                #'aab': np.zeros((nua, nua, nub, noa, noa, nob)),
                #'abb': np.zeros((nua, nub, nub, noa, nob, nob)),
                #'bbb': np.zeros((nub, nub, nub, nob, nob, nob)),
        }
        for key in o.keys():
            o[key] *= 0.0

        ncomm = 0
        while ncomm < max_ncomm:
            o['0'] = 0.0
            # 0-body (energy)
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
            o = h1a_t3a_c0(o, o_old, self.T, self.ref.gam1, self.ref.eta1, self.ref.lambdas, self.ref.orbspace, scale=2.0 if herm else 1.0)
            o = h1a_t3b_c0(o, o_old, self.T, self.ref.gam1, self.ref.eta1, self.ref.lambdas, self.ref.orbspace, scale=2.0 if herm else 1.0)
            o = h1a_t3c_c0(o, o_old, self.T, self.ref.gam1, self.ref.eta1, self.ref.lambdas, self.ref.orbspace, scale=2.0 if herm else 1.0)
            o = h1b_t3b_c0(o, o_old, self.T, self.ref.gam1, self.ref.eta1, self.ref.lambdas, self.ref.orbspace, scale=2.0 if herm else 1.0)
            o = h1b_t3c_c0(o, o_old, self.T, self.ref.gam1, self.ref.eta1, self.ref.lambdas, self.ref.orbspace, scale=2.0 if herm else 1.0)
            o = h1b_t3d_c0(o, o_old, self.T, self.ref.gam1, self.ref.eta1, self.ref.lambdas, self.ref.orbspace, scale=2.0 if herm else 1.0)
            o = h2a_t3a_c0(o, o_old, self.T, self.ref.gam1, self.ref.eta1, self.ref.lambdas, self.ref.orbspace, scale=2.0 if herm else 1.0)
            o = h2a_t3b_c0(o, o_old, self.T, self.ref.gam1, self.ref.eta1, self.ref.lambdas, self.ref.orbspace, scale=2.0 if herm else 1.0)
            o = h2a_t3c_c0(o, o_old, self.T, self.ref.gam1, self.ref.eta1, self.ref.lambdas, self.ref.orbspace, scale=2.0 if herm else 1.0)
            o = h2b_t3a_c0(o, o_old, self.T, self.ref.gam1, self.ref.eta1, self.ref.lambdas, self.ref.orbspace, scale=2.0 if herm else 1.0)
            o = h2b_t3b_c0(o, o_old, self.T, self.ref.gam1, self.ref.eta1, self.ref.lambdas, self.ref.orbspace, scale=2.0 if herm else 1.0)
            o = h2b_t3c_c0(o, o_old, self.T, self.ref.gam1, self.ref.eta1, self.ref.lambdas, self.ref.orbspace, scale=2.0 if herm else 1.0)
            o = h2b_t3d_c0(o, o_old, self.T, self.ref.gam1, self.ref.eta1, self.ref.lambdas, self.ref.orbspace, scale=2.0 if herm else 1.0)
            o = h2c_t3b_c0(o, o_old, self.T, self.ref.gam1, self.ref.eta1, self.ref.lambdas, self.ref.orbspace, scale=2.0 if herm else 1.0)
            o = h2c_t3c_c0(o, o_old, self.T, self.ref.gam1, self.ref.eta1, self.ref.lambdas, self.ref.orbspace, scale=2.0 if herm else 1.0)
            o = h2c_t3d_c0(o, o_old, self.T, self.ref.gam1, self.ref.eta1, self.ref.lambdas, self.ref.orbspace, scale=2.0 if herm else 1.0)
            ### onebody
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
            o = h1a_t3a_c1a(o, o_old, self.T, self.ref.gam1, self.ref.eta1, self.ref.lambdas, self.ref.orbspace)
            o = h1a_t3b_c1a(o, o_old, self.T, self.ref.gam1, self.ref.eta1, self.ref.lambdas, self.ref.orbspace)
            o = h1b_t3b_c1a(o, o_old, self.T, self.ref.gam1, self.ref.eta1, self.ref.lambdas, self.ref.orbspace)
            o = h1b_t3c_c1a(o, o_old, self.T, self.ref.gam1, self.ref.eta1, self.ref.lambdas, self.ref.orbspace)
            o = h2a_t3a_c1a(o, o_old, self.T, self.ref.gam1, self.ref.eta1, self.ref.lambdas, self.ref.orbspace)
            o = h2a_t3b_c1a(o, o_old, self.T, self.ref.gam1, self.ref.eta1, self.ref.lambdas, self.ref.orbspace)
            o = h2a_t3c_c1a(o, o_old, self.T, self.ref.gam1, self.ref.eta1, self.ref.lambdas, self.ref.orbspace)
            o = h2b_t3a_c1a(o, o_old, self.T, self.ref.gam1, self.ref.eta1, self.ref.lambdas, self.ref.orbspace)
            o = h2b_t3b_c1a(o, o_old, self.T, self.ref.gam1, self.ref.eta1, self.ref.lambdas, self.ref.orbspace)
            o = h2b_t3c_c1a(o, o_old, self.T, self.ref.gam1, self.ref.eta1, self.ref.lambdas, self.ref.orbspace)
            o = h2b_t3d_c1a(o, o_old, self.T, self.ref.gam1, self.ref.eta1, self.ref.lambdas, self.ref.orbspace)
            o = h2c_t3b_c1a(o, o_old, self.T, self.ref.gam1, self.ref.eta1, self.ref.lambdas, self.ref.orbspace)
            o = h2c_t3c_c1a(o, o_old, self.T, self.ref.gam1, self.ref.eta1, self.ref.lambdas, self.ref.orbspace)
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
            o = h1a_t3b_c1b(o, o_old, self.T, self.ref.gam1, self.ref.eta1, self.ref.lambdas, self.ref.orbspace)
            o = h1a_t3c_c1b(o, o_old, self.T, self.ref.gam1, self.ref.eta1, self.ref.lambdas, self.ref.orbspace)
            o = h1b_t3c_c1b(o, o_old, self.T, self.ref.gam1, self.ref.eta1, self.ref.lambdas, self.ref.orbspace)
            o = h1b_t3d_c1b(o, o_old, self.T, self.ref.gam1, self.ref.eta1, self.ref.lambdas, self.ref.orbspace)
            o = h2a_t3b_c1b(o, o_old, self.T, self.ref.gam1, self.ref.eta1, self.ref.lambdas, self.ref.orbspace)
            o = h2a_t3c_c1b(o, o_old, self.T, self.ref.gam1, self.ref.eta1, self.ref.lambdas, self.ref.orbspace)
            o = h2b_t3a_c1b(o, o_old, self.T, self.ref.gam1, self.ref.eta1, self.ref.lambdas, self.ref.orbspace)
            o = h2b_t3b_c1b(o, o_old, self.T, self.ref.gam1, self.ref.eta1, self.ref.lambdas, self.ref.orbspace)
            o = h2b_t3c_c1b(o, o_old, self.T, self.ref.gam1, self.ref.eta1, self.ref.lambdas, self.ref.orbspace)
            o = h2b_t3d_c1b(o, o_old, self.T, self.ref.gam1, self.ref.eta1, self.ref.lambdas, self.ref.orbspace)
            o = h2c_t3b_c1b(o, o_old, self.T, self.ref.gam1, self.ref.eta1, self.ref.lambdas, self.ref.orbspace)
            o = h2c_t3c_c1b(o, o_old, self.T, self.ref.gam1, self.ref.eta1, self.ref.lambdas, self.ref.orbspace)
            o = h2c_t3d_c1b(o, o_old, self.T, self.ref.gam1, self.ref.eta1, self.ref.lambdas, self.ref.orbspace)
            ### twobody
            o = h1a_t2a_c2a(o, o_old, self.T, self.ref.gam1, self.ref.eta1, self.ref.lambdas, self.ref.orbspace)
            o = h2a_t1a_c2a(o, o_old, self.T, self.ref.gam1, self.ref.eta1, self.ref.lambdas, self.ref.orbspace)
            o = h2a_t2a_c2a(o, o_old, self.T, self.ref.gam1, self.ref.eta1, self.ref.lambdas, self.ref.orbspace)
            o = h2b_t2b_c2a(o, o_old, self.T, self.ref.gam1, self.ref.eta1, self.ref.lambdas, self.ref.orbspace)
            o = h1a_t3a_c2a(o, o_old, self.T, self.ref.gam1, self.ref.eta1, self.ref.lambdas, self.ref.orbspace)
            o = h1b_t3b_c2a(o, o_old, self.T, self.ref.gam1, self.ref.eta1, self.ref.lambdas, self.ref.orbspace)
            o = h2a_t3a_c2a(o, o_old, self.T, self.ref.gam1, self.ref.eta1, self.ref.lambdas, self.ref.orbspace)
            o = h2a_t3b_c2a(o, o_old, self.T, self.ref.gam1, self.ref.eta1, self.ref.lambdas, self.ref.orbspace)
            o = h2b_t3a_c2a(o, o_old, self.T, self.ref.gam1, self.ref.eta1, self.ref.lambdas, self.ref.orbspace)
            o = h2b_t3b_c2a(o, o_old, self.T, self.ref.gam1, self.ref.eta1, self.ref.lambdas, self.ref.orbspace)
            o = h2b_t3c_c2a(o, o_old, self.T, self.ref.gam1, self.ref.eta1, self.ref.lambdas, self.ref.orbspace)
            o = h2c_t3b_c2a(o, o_old, self.T, self.ref.gam1, self.ref.eta1, self.ref.lambdas, self.ref.orbspace)
            o = h1a_t2b_c2b(o, o_old, self.T, self.ref.gam1, self.ref.eta1, self.ref.lambdas, self.ref.orbspace)
            o = h1b_t2b_c2b(o, o_old, self.T, self.ref.gam1, self.ref.eta1, self.ref.lambdas, self.ref.orbspace)
            o = h2b_t1a_c2b(o, o_old, self.T, self.ref.gam1, self.ref.eta1, self.ref.lambdas, self.ref.orbspace)
            o = h2b_t1b_c2b(o, o_old, self.T, self.ref.gam1, self.ref.eta1, self.ref.lambdas, self.ref.orbspace)
            o = h2a_t2b_c2b(o, o_old, self.T, self.ref.gam1, self.ref.eta1, self.ref.lambdas, self.ref.orbspace)
            o = h2b_t2a_c2b(o, o_old, self.T, self.ref.gam1, self.ref.eta1, self.ref.lambdas, self.ref.orbspace)
            o = h2b_t2b_c2b(o, o_old, self.T, self.ref.gam1, self.ref.eta1, self.ref.lambdas, self.ref.orbspace)
            o = h2b_t2c_c2b(o, o_old, self.T, self.ref.gam1, self.ref.eta1, self.ref.lambdas, self.ref.orbspace)
            o = h2c_t2b_c2b(o, o_old, self.T, self.ref.gam1, self.ref.eta1, self.ref.lambdas, self.ref.orbspace)
            o = h1a_t3b_c2b(o, o_old, self.T, self.ref.gam1, self.ref.eta1, self.ref.lambdas, self.ref.orbspace)
            o = h1b_t3c_c2b(o, o_old, self.T, self.ref.gam1, self.ref.eta1, self.ref.lambdas, self.ref.orbspace)
            o = h2a_t3b_c2b(o, o_old, self.T, self.ref.gam1, self.ref.eta1, self.ref.lambdas, self.ref.orbspace)
            o = h2a_t3c_c2b(o, o_old, self.T, self.ref.gam1, self.ref.eta1, self.ref.lambdas, self.ref.orbspace)
            o = h2b_t3a_c2b(o, o_old, self.T, self.ref.gam1, self.ref.eta1, self.ref.lambdas, self.ref.orbspace)
            o = h2b_t3b_c2b(o, o_old, self.T, self.ref.gam1, self.ref.eta1, self.ref.lambdas, self.ref.orbspace)
            o = h2b_t3c_c2b(o, o_old, self.T, self.ref.gam1, self.ref.eta1, self.ref.lambdas, self.ref.orbspace)
            o = h2b_t3d_c2b(o, o_old, self.T, self.ref.gam1, self.ref.eta1, self.ref.lambdas, self.ref.orbspace)
            o = h2c_t3b_c2b(o, o_old, self.T, self.ref.gam1, self.ref.eta1, self.ref.lambdas, self.ref.orbspace)
            o = h2c_t3c_c2b(o, o_old, self.T, self.ref.gam1, self.ref.eta1, self.ref.lambdas, self.ref.orbspace)
            o = h1b_t2c_c2c(o, o_old, self.T, self.ref.gam1, self.ref.eta1, self.ref.lambdas, self.ref.orbspace)
            o = h2c_t1b_c2c(o, o_old, self.T, self.ref.gam1, self.ref.eta1, self.ref.lambdas, self.ref.orbspace)
            o = h2c_t2c_c2c(o, o_old, self.T, self.ref.gam1, self.ref.eta1, self.ref.lambdas, self.ref.orbspace)
            o = h2b_t2b_c2c(o, o_old, self.T, self.ref.gam1, self.ref.eta1, self.ref.lambdas, self.ref.orbspace)
            o = h1a_t3c_c2c(o, o_old, self.T, self.ref.gam1, self.ref.eta1, self.ref.lambdas, self.ref.orbspace)
            o = h1b_t3d_c2c(o, o_old, self.T, self.ref.gam1, self.ref.eta1, self.ref.lambdas, self.ref.orbspace)
            o = h2a_t3c_c2c(o, o_old, self.T, self.ref.gam1, self.ref.eta1, self.ref.lambdas, self.ref.orbspace)
            o = h2b_t3b_c2c(o, o_old, self.T, self.ref.gam1, self.ref.eta1, self.ref.lambdas, self.ref.orbspace)
            o = h2b_t3c_c2c(o, o_old, self.T, self.ref.gam1, self.ref.eta1, self.ref.lambdas, self.ref.orbspace)
            o = h2b_t3d_c2c(o, o_old, self.T, self.ref.gam1, self.ref.eta1, self.ref.lambdas, self.ref.orbspace)
            o = h2c_t3c_c2c(o, o_old, self.T, self.ref.gam1, self.ref.eta1, self.ref.lambdas, self.ref.orbspace)
            o = h2c_t3d_c2c(o, o_old, self.T, self.ref.gam1, self.ref.eta1, self.ref.lambdas, self.ref.orbspace)
            # 3-body
            o = h2a_t2a_c3a(o, o_old, self.T, self.ref.gam1, self.ref.eta1, self.ref.lambdas, self.ref.orbspace)
            o = h2a_t2b_c3b(o, o_old, self.T, self.ref.gam1, self.ref.eta1, self.ref.lambdas, self.ref.orbspace)
            o = h2b_t2a_c3b(o, o_old, self.T, self.ref.gam1, self.ref.eta1, self.ref.lambdas, self.ref.orbspace)
            o = h2b_t2b_c3b(o, o_old, self.T, self.ref.gam1, self.ref.eta1, self.ref.lambdas, self.ref.orbspace)
            o = h2b_t2b_c3c(o, o_old, self.T, self.ref.gam1, self.ref.eta1, self.ref.lambdas, self.ref.orbspace)
            o = h2b_t2c_c3c(o, o_old, self.T, self.ref.gam1, self.ref.eta1, self.ref.lambdas, self.ref.orbspace)
            o = h2c_t2b_c3c(o, o_old, self.T, self.ref.gam1, self.ref.eta1, self.ref.lambdas, self.ref.orbspace)
            o = h2c_t2c_c3d(o, o_old, self.T, self.ref.gam1, self.ref.eta1, self.ref.lambdas, self.ref.orbspace)
            #
            o = h1a_t3a_c3a(o, o_old, self.T, self.ref.gam1, self.ref.eta1, self.ref.lambdas, self.ref.orbspace)
            o = h2a_t3a_c3a(o, o_old, self.T, self.ref.gam1, self.ref.eta1, self.ref.lambdas, self.ref.orbspace)
            o = h2b_t3b_c3a(o, o_old, self.T, self.ref.gam1, self.ref.eta1, self.ref.lambdas, self.ref.orbspace)
            o = h1a_t3b_c3b(o, o_old, self.T, self.ref.gam1, self.ref.eta1, self.ref.lambdas, self.ref.orbspace)
            o = h1b_t3b_c3b(o, o_old, self.T, self.ref.gam1, self.ref.eta1, self.ref.lambdas, self.ref.orbspace)
            o = h2a_t3b_c3b(o, o_old, self.T, self.ref.gam1, self.ref.eta1, self.ref.lambdas, self.ref.orbspace)
            o = h2b_t3a_c3b(o, o_old, self.T, self.ref.gam1, self.ref.eta1, self.ref.lambdas, self.ref.orbspace)
            o = h2b_t3b_c3b(o, o_old, self.T, self.ref.gam1, self.ref.eta1, self.ref.lambdas, self.ref.orbspace)
            o = h2b_t3c_c3b(o, o_old, self.T, self.ref.gam1, self.ref.eta1, self.ref.lambdas, self.ref.orbspace)
            o = h2c_t3b_c3b(o, o_old, self.T, self.ref.gam1, self.ref.eta1, self.ref.lambdas, self.ref.orbspace)
            o = h1a_t3c_c3c(o, o_old, self.T, self.ref.gam1, self.ref.eta1, self.ref.lambdas, self.ref.orbspace)
            o = h1b_t3c_c3c(o, o_old, self.T, self.ref.gam1, self.ref.eta1, self.ref.lambdas, self.ref.orbspace)
            o = h2a_t3c_c3c(o, o_old, self.T, self.ref.gam1, self.ref.eta1, self.ref.lambdas, self.ref.orbspace)
            o = h2b_t3b_c3c(o, o_old, self.T, self.ref.gam1, self.ref.eta1, self.ref.lambdas, self.ref.orbspace)
            o = h2b_t3c_c3c(o, o_old, self.T, self.ref.gam1, self.ref.eta1, self.ref.lambdas, self.ref.orbspace)
            o = h2b_t3d_c3c(o, o_old, self.T, self.ref.gam1, self.ref.eta1, self.ref.lambdas, self.ref.orbspace)
            o = h2c_t3c_c3c(o, o_old, self.T, self.ref.gam1, self.ref.eta1, self.ref.lambdas, self.ref.orbspace)
            o = h1b_t3d_c3d(o, o_old, self.T, self.ref.gam1, self.ref.eta1, self.ref.lambdas, self.ref.orbspace)
            o = h2b_t3c_c3d(o, o_old, self.T, self.ref.gam1, self.ref.eta1, self.ref.lambdas, self.ref.orbspace)
            o = h2c_t3d_c3d(o, o_old, self.T, self.ref.gam1, self.ref.eta1, self.ref.lambdas, self.ref.orbspace)

            # antisymmetrize
            o['aa'] -= o['aa'].transpose(1, 0, 2, 3)
            o['aa'] -= o['aa'].transpose(0, 1, 3, 2)
            o['bb'] -= o['bb'].transpose(1, 0, 2, 3)
            o['bb'] -= o['bb'].transpose(0, 1, 3, 2)

            o['aaa'] -= o['aaa'].transpose(0, 2, 1, 3, 4, 5) # (bc)
            o['aaa'] -= o['aaa'].transpose(1, 0, 2, 3, 4, 5) + o['aaa'].transpose(2, 1, 0, 3, 4, 5) # (a/bc)
            o['aaa'] -= o['aaa'].transpose(0, 1, 2, 3, 5, 4) # (jk)
            o['aaa'] -= o['aaa'].transpose(0, 1, 2, 4, 3, 5) + o['aaa'].transpose(0, 1, 2, 5, 4, 3) # (i/jk)

            o['aab'] -= o['aab'].transpose(1, 0, 2, 3, 4, 5) # (ab)
            o['aab'] -= o['aab'].transpose(0, 1, 2, 4, 3, 5) # (ij)

            o['abb'] -= o['abb'].transpose(0, 2, 1, 3, 4, 5) # (bc)
            o['abb'] -= o['abb'].transpose(0, 1, 2, 3, 5, 4) # (jk)

            o['bbb'] -= o['bbb'].transpose(0, 2, 1, 3, 4, 5) # (bc)
            o['bbb'] -= o['bbb'].transpose(1, 0, 2, 3, 4, 5) + o['bbb'].transpose(2, 1, 0, 3, 4, 5) # (a/bc)
            o['bbb'] -= o['bbb'].transpose(0, 1, 2, 3, 5, 4) # (jk)
            o['bbb'] -= o['bbb'].transpose(0, 1, 2, 4, 3, 5) + o['bbb'].transpose(0, 1, 2, 5, 4, 3) # (i/jk)

            if herm: 
                o['a'] += o['a'].T.conj()
                o['b'] += o['b'].T.conj()
                o['aa'] += o['aa'].transpose(2, 3, 0, 1)
                o['ab'] += o['ab'].transpose(2, 3, 0, 1)
                o['bb'] += o['bb'].transpose(2, 3, 0, 1)
                o['aaa'] += o['aaa'].transpose(3, 4, 5, 0, 1, 2)
                o['aab'] += o['aab'].transpose(3, 4, 5, 0, 1, 2)
                o['abb'] += o['abb'].transpose(3, 4, 5, 0, 1, 2)
                o['bbb'] += o['bbb'].transpose(3, 4, 5, 0, 1, 2)

            # Increment commutator count
            ncomm += 1

            for key, value in o.items():
                o[key] /= ncomm

            res1 = np.linalg.norm(o['a'].flatten() + o['b'].flatten())
            res2 = np.linalg.norm(o['aa'].flatten() + o['ab'].flatten() + o['bb'].flatten())
            res3 = np.linalg.norm(o['aaa'].flatten() + o['aab'].flatten() + o['abb'].flatten() + o['bbb'].flatten())
            residual_op = res1 + res2 + res3
            if verbose:
                print(f"    {ncomm}   |C| = {residual_op}   |c1| = {res1}   |c2| = {res2}   |c3| = {res3}")
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

