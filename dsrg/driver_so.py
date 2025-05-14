import time
from importlib import import_module
import numpy as np

import dsrg
import dsrg.methods
from dsrg.utilities import get_memory_usage, spin_label, spatial_index

class DSRG:

    def __init__(self, ref, print_threshold=0.09):
        self.ref = ref
        self.reference_energy = self.ref.e_cas
        self._print_threshold = print_threshold
        self._par = {'nbody_t': 0,
                     'nbody_h': 0,
                     'comm_approx': 0}
        self.T = None


    def load_calculation(self, method):

        if method in ["ldsrg2_so"]:
            self._par['nbody_t'] = 2
            self._par['nbody_h'] = 2
            self._par['comm_approx'] = 1

        elif method in ['ldsrg2t_so']:
            self._par['nbody_t'] = 3
            self._par['nbody_h'] = 3
            self._par['comm_approx'] = 1

        elif method in ['ldsrg3_so']:
            self._par['nbody_t'] = 3
            self._par['nbody_h'] = 3
            self._par['comm_approx'] = 1

        if method.lower() not in dsrg.methods.MODULES:
            raise NotImplementedError(f"Method {method.upper()} not implemented!")
        self.calc_module = import_module("dsrg.methods." + method.lower())
        self.update_hbar = getattr(self.calc_module, 'update_hbar')
        self.update_t = getattr(self.calc_module, 'update_t')
        self.initial_guess = getattr(self.calc_module, 'initial_guess')
        self.denom_builder = getattr(self.calc_module, 'build_denominators')


    def initialize_hbar(self):
        self.hbar = {}
        self.hbar['0'] = np.zeros(1)
        for n in range(1, self._par['nbody_h'] + 1):
            key = str(n)
            self.hbar[key] = np.zeros(2*n * (self.ref.norb,))


    def run_dsrg(self, method, s, maxiter=80, herm=True, conv=1.0e-07, max_ncomm=12):

        # Mount the desired calculation
        self.load_calculation(method)
        print(f"   ... loaded calculation modules from 'dsrg.{method.lower()}.py'")

        if herm:
            _method_name = method.upper()
        else:
            _method_name = 'NH-' + method.upper()
        print(f"    ==> {_method_name} Amplitude Equations <==")
        print("")
        print(f"   Flow parameter (s): {s} 1/Eh^2")
        print("")
        t_start = time.time()

        # [WARNING]: Set 3-cumulant to 0
        print("   WARNING: Setting 3-cumulant to 0!")
        self.ref.lambdas['3'] *= 0.

        # Build n-body (regularized) MP denominators
        tic = time.time()
        denom, reg_denom = self.denom_builder(
            s,
            np.real(np.diagonal(self.ref.F)),
            self.ref,
        )
        toc = time.time()
        print(f"   ... build n-body regularized denominators: {toc - tic}s")

        # Obtain initial guess
        tic = time.time()
        if not self.T:
            self.T = self.initial_guess(self.ref, denom, reg_denom)
        toc = time.time()
        print(f"   ... initial T amplitudes: {toc - tic}s")

        # Initialize HBar
        tic = time.time()
        self.initialize_hbar()
        o = {}
        for key, value in self.hbar.items():
            o[key] = np.zeros_like(value)
        toc = time.time()
        print(f'   ... allocate HBar arrays: {toc - tic}s')

        #
        # DSRG iterations
        #
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
            print("    {: 5d} {: 20.12f} {: 20.12f} {: 20.12f} {: 5d}    {:.2f}m {:.2f}s    {:.2f} MB".format(it,
                                                                                                              energy[0],
                                                                                                              delta_e[0],
                                                                                                              resid,
                                                                                                              ncomm,
                                                                                                              minutes, seconds,
                                                                                                              get_memory_usage()))
            if abs(delta_e) < conv:
                print(f"    MR-DSRG successfully converged after {it} iterations.")
                break
            e_old = energy.copy()
            # Update amplitudes
            self.T = self.update_t(self.T, self.hbar, self.ref, denom, reg_denom)
            # Update iteration counter
            it += 1
        else:
            print("   MR-DSRG did not converge")
        # Record the energy
        self.correlation_energy = energy[0]
        self.total_energy = self.correlation_energy + self.reference_energy
        # Record total time and print summary
        minutes, seconds = divmod(time.time() - t_start, 60)
        print("")
        print("    Calculation Summary:")
        print("    --------------------")
        print("    Reference Energy: {: 20.12f}".format(self.ref.e_cas))
        print("    Unrelaxed MR-DSRG Correlation Energy: {: 20.12f}".format(self.correlation_energy))
        print("    Unrelaxed MR-DSRG Total Energy: {: 20.12f}".format(self.total_energy))
        print("")
        print("    Largest Singly and Doubly Excited Amplitudes")
        print("    --------------------------------------------")
        self.print_amplitudes()
        print("")
        print("    MR-DSRG calculation completed in {:.2f}m {:.2f}s".format(minutes, seconds))
        print(f"    Memory usage: {get_memory_usage()} MB")
        print("")

    def compute_hbar(self, o, max_ncomm, herm, verbose=False):
        # Initial values
        for key in self.hbar.keys():
            self.hbar[key] *= 0.0
        self.hbar['0'] = np.array([0.0])
        self.hbar['1'] = self.ref.F.copy()
        self.hbar['2'] = self.ref.V.copy()
        #
        o_old = {}
        for key in o.keys():
            o[key] *= 0.0
            o_old[key] = self.hbar[key].copy()
        #
        ncomm = 0
        while ncomm < max_ncomm:
            o['0'] *= 0.0
            o = self.update_hbar(o, o_old, self.T, self.ref, herm)
            # Increment commutator count
            ncomm += 1
            # Compute residual
            resid = 0.0
            for key, value in o.items():
                o[key] /= ncomm
                resid += np.linalg.norm(o[key].flatten())
            if resid < 1e-12:
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
        return ncomm, resid

    def print_amplitudes(self):
        t1 = self.T['1']
        t2 = self.T['2']
        nu, no = t1.shape
        n = 1
        print("          i -> a")
        for a in range(nu):
            for i in range(no):
                if abs(t1[a, i]) <= self._print_threshold: continue
                print(
                    f"     [{n}]  {spatial_index(i + 1)}{spin_label(i + 1)} -> {spatial_index(a + self.ref.ncore + 1)}{spin_label(a + self.ref.ncore + 1)}    {t1[a, i]}")
                n += 1
        print("          i j -> a b")
        for a in range(nu):
            for b in range(a + 1, nu):
                for i in range(no):
                    for j in range(i + 1, no):
                        if abs(t2[a, b, i, j]) <= self._print_threshold: continue
                        print(
                            f"     [{n}]  {spatial_index(i + 1)}{spin_label(i + 1)} {spatial_index(j + 1)}{spin_label(j + 1)} -> {spatial_index(a + self.ref.ncore + 1)}{spin_label(a + self.ref.ncore + 1)} {spatial_index(b + self.ref.ncore + 1)}{spin_label(b + self.ref.ncore + 1)}    {t2[a, b, i, j]}")
                        n += 1