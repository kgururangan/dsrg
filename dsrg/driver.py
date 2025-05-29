import time
from importlib import import_module
import numpy as np
from copy import deepcopy

from dsrg.methods import MODULES
from dsrg.utilities import (get_memory_usage,
                            numel_in_dict,
                            unflatten_vector_to_dict,
                            semicanonicalize_active)
from dsrg.diis import DIIS
from dsrg.gno import denormal_order_ints


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

        if method in ["ldsrg2"]:
            self._par['nbody_t'] = 2
            self._par['nbody_h'] = 2
            self._par['comm_approx'] = 1

        elif method in ['ldsrg2t']:
            self._par['nbody_t'] = 3
            self._par['nbody_h'] = 3
            self._par['comm_approx'] = 1

        elif method in ['ldsrg3']:
            self._par['nbody_t'] = 3
            self._par['nbody_h'] = 3
            self._par['comm_approx'] = 1

        if method.lower() not in MODULES:
            raise NotImplementedError(f"Method {method.upper()} not implemented!")
        self.calc_module = import_module(f"dsrg.methods.{method.lower()}")
        self.update_hbar = getattr(self.calc_module, 'update_hbar')
        self.update_t = getattr(self.calc_module, 'update_t')
        self.initial_guess = getattr(self.calc_module, 'initial_guess')
        self.denom_builder = getattr(self.calc_module, 'build_denominators')


    def initialize_hbar(self):
        self.hbar = {}
        self.hbar['0'] = np.zeros(1)
        for n in range(1, self._par['nbody_h'] + 1):
            for m in range(n + 1):
                key = 'a'*(n - m) + 'b'*m
                self.hbar[key] = np.zeros(2*n * (self.ref.norb,))


    def run_dsrg(self, method, s, maxiter=80, herm=True, conv=1.0e-07, max_ncomm=12, diis_size=6, out_of_core=False):

        # Mount the desired calculation
        self.load_calculation(method)
        print(f"   ... loaded calculation modules from 'dsrg.{method.lower()}.py'")

        if herm:
            _method_name = method.upper()
        else:
            _method_name = 'NH-' + method.upper()
        print(f"    ==> {_method_name} Amplitude Equations <==")
        print("")
        print(f"   Flow parameter (s): {s} Eh^-2")
        print("")
        t_start = time.time()

        # Build n-body (regularized) MP denominators
        tic = time.time()
        denom, reg_denom = self.denom_builder(
            s,
            np.real(np.diagonal(self.ref.F['a'])),
            np.real(np.diagonal(self.ref.F['b'])),
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

        # Initialize DIIS engine
        # diis_engine = DIIS(
        #     ndim=numel_in_dict(self.T),
        #     diis_size=diis_size,
        #     out_of_core=out_of_core
        # )

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
            
            # DIIS extrapolate
            # diis_engine.push(self.T, dT, it)
            # if it >= diis_size:  
            
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
        self.hbar['a'] = self.ref.F['a'].copy()
        self.hbar['b'] = self.ref.F['b'].copy()
        self.hbar['aa'] = self.ref.V['aa'].copy()
        self.hbar['ab'] = self.ref.V['ab'].copy()
        self.hbar['bb'] = self.ref.V['bb'].copy()
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
            if verbose:
                print(f"         E{ncomm} = {self.hbar['0'][0]:.10f}  |H(T1)| = {resid:.10f}")
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

    def diagonalize_hbar(self, herm, state_index=None):

        print(f"     ==> Similarity-Transformed Hamiltonian Diagonalization <==")

        if state_index is None:
            state_index = list(range(self.ref.nstates))

        self.relaxation_energy = np.zeros(len(state_index))
        self.total_energy_relaxed = np.zeros(len(state_index))

        # Slicing
        a = self.ref.orbspace['active_alpha']
        A = self.ref.orbspace['active_beta']
        c = self.ref.orbspace['core_alpha']
        C = self.ref.orbspace['core_beta']
        # Obtain the similarity-transformed Hamiltonian (1- and 2-body) in the active space
        # coulomb = np.einsum("ipiq->pq", self.hbar['ab'][c, A, c, A])
        # exchange = np.einsum("ipqi->pq", self.hbar['ab'][c, A, a, C])
        # hbar1_act = self.hbar['a'][a, a] + 2*coulomb - exchange
        #
        # hbar_act = {'a': hbar1_act,
        #             'b': hbar1_act,
        #             'aa': self.hbar['aa'][a, a, a, a],
        #             'ab': self.hbar['ab'][a, A, a, A],
        #             'bb': self.hbar['bb'][A, A, A, A]}
        hbar_act = {'a': self.hbar['a'][a, a],
                    'b': self.hbar['b'][A, A],
                    'aa': self.hbar['aa'][a, a, a, a],
                    'ab': self.hbar['ab'][a, A, a, A],
                    'bb': self.hbar['bb'][A, A, A, A]}
        # Denormal order the Hbar integrals
        print(f"    Denormal order active HBar integrals")
        hbar_act, e_scalar = denormal_order_ints(hbar_act, self.ref)
        print(f"    <HBar> = {e_scalar}")
        # Semicanonicalize the active-space HBar integrals
        print(f"    Semicanonicalize denormal-ordered active HBar integrals")
        hbar_act = semicanonicalize_active(hbar_act, self.ref)
        # Diagonalize Hamiltonian in the CAS space using fcipy
        self.ref.cisolver.e1int = hbar_act['a'].copy()
        self.ref.cisolver.e2int = hbar_act['ab'].copy()
        # self.ref.cisolver.run_ci(self.ref.nstates, opt=True, herm=herm)
        self.ref.cisolver.build_hamiltonian(herm=herm)
        self.ref.cisolver.diagonalize_hamiltonian(herm=herm)

        self.relaxation_energy = self.ref.cisolver.state_eigval.real + e_scalar
        self.total_energy_relaxed = self.total_energy + self.relaxation_energy

        state_index, overlap = self.ref.get_state_indices_in_spectrum()
        i_ground = state_index[0]

        for i, istate in enumerate(state_index):
            # ground_state = False
            print("")
            print(f"    Calculation Summary: State {i}")
            print(f"    initial root {i} -> relaxed root {istate}")
            print(f"    overlap = {overlap[i, istate]}")
            # if istate == i_ground:
            #     ground_state = True
            #     print("    [Ground State]")
            print("    ------------------------------------")
            # if ground_state:
            #     print("    Reference Energy: {: 20.12f}".format(self.ref.e_cas))
            #     print("    Unrelaxed MR-DSRG Total Energy: {: 20.12f}".format(self.total_energy))
            # else:
            #     print("    Excitation Energy: {: 20.12f}".format(self.total_energy_relaxed[istate] - self.total_energy_relaxed[i_ground]))
            print("    Relaxation Energy: {: 20.12f}".format(self.relaxation_energy[istate]))
            print("    Relaxed MR-DSRG Total Energy: {: 20.12f}".format(self.total_energy_relaxed[istate]))
            self.ref.cisolver.print_ci_vector(state=istate, prtol=0.19)

    def print_amplitudes(self):

        nua, nub, noa, nob = self.T['ab'].shape

        n = 1
        for a in range(nua):
            for i in range(noa):
                if abs(self.T['a'][a, i]) <= self._print_threshold: continue
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
                if abs(self.T['b'][a, i]) <= self._print_threshold: continue
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
                        if abs(self.T['aa'][a, b, i, j]) <= self._print_threshold: continue
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
                        if abs(self.T['bb'][a, b, i, j]) <= self._print_threshold: continue
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
                        if abs(self.T['ab'][a, b, i, j]) <= self._print_threshold: continue
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
                        
class RICMRCC:

    def __init__(self, ref, print_threshold=0.09):

        self.ref = ref
        self.reference_energy = self.ref.e_cas
        self._print_threshold = print_threshold
        self._par = {'nbody_t': 0,
                     'nbody_h': 0,
                     'comm_approx': 0}
        self.T = None


    def load_calculation(self, method):

        if method in ["ricmrccsd", "ricmrccsd_approx", "sqricmrccsd", "sqricmrccsd_approx",
                      "ricmrccsd1"]:
            self._par['nbody_t'] = 2
            self._par['nbody_h'] = 2
            self._par['comm_approx'] = 2
        elif method in ["ricmrccsdt1_approx"]:
            self._par['nbody_t'] = 3
            self._par['nbody_h'] = 3
            self._par['comm_approx'] = 2

        if method.lower() not in MODULES:
            raise NotImplementedError(f"Method {method.upper()} not implemented!")
        self.calc_module = import_module(f"dsrg.methods.{method.lower()}")
        self.residual_function = getattr(self.calc_module, 'compute_residual')
        self.update_t = getattr(self.calc_module, 'update_t')
        self.initial_guess = getattr(self.calc_module, 'initial_guess')
        self.denom_builder = getattr(self.calc_module, 'build_denominators')
        try:
            self.build_hbar_active = getattr(self.calc_module, 'compute_hbar_active')
        except AttributeError:
            self.build_hbar_active = None


    def initialize_hbar(self):
        
        # Assemble bare Hamiltonian dictionary
        self.hamiltonian = {'a': self.ref.F['a'],
                            'b': self.ref.F['b'],
                            'aa': self.ref.V['aa'],
                            'ab': self.ref.V['ab'], 
                            'bb': self.ref.V['bb'],
                            'aaa': None,
                            'aab': None,
                            'abb': None,
                            'bbb': None}


    def run_ricmrcc(self, method, s, maxiter=80, herm=False, e_conv=1.0e-07, t_conv=1.0e-05, diis_size=6, out_of_core=False):

        # Set hermiticity flag
        self.herm = herm

        # Mount the desired calculation
        self.load_calculation(method)
        print(f"   ... loaded calculation modules from 'dsrg.{method.lower()}.py'")

        if self.herm:
            _method_name = method.upper()
        else:
            _method_name = 'NH-' + method.upper()
        print(f"    ==> {_method_name} Amplitude Equations <==")
        print("")
        print(f"   Flow parameter (s): {s} Eh^-2")
        print("")
        t_start = time.time()

        # Build n-body (regularized) MP denominators
        tic = time.time()
        denom, reg_denom = self.denom_builder(
            s,
            np.real(np.diagonal(self.ref.F['a'])),
            np.real(np.diagonal(self.ref.F['b'])),
            self.ref,
        )
        toc = time.time()
        print(f"   ... build n-body regularized denominators: {toc - tic}s")

        # Obtain initial guess
        tic = time.time()
        if not self.T:
            self.T = self.initial_guess(self.ref, denom, reg_denom)
        T_pert = deepcopy(self.T)
        toc = time.time()
        print(f"   ... initial T amplitudes: {toc - tic}s")

        # Initialize HBar
        tic = time.time()
        self.initialize_hbar()
        toc = time.time()
        print(f'   ... allocate HBar arrays: {toc - tic}s')
        
        # Initialize DIIS engine
        diis_engine = DIIS(
            ndim=numel_in_dict(self.T),
            diis_size=diis_size,
            out_of_core=out_of_core
        )

        T_shapes = {k: v.shape for k, v in self.T.items()}
        T_sizes = {k: v.size for k, v in self.T.items()}

        #
        # ric-MRCC iterations
        #
        e_old = .0
        it = 0
        print("")
        print("     Iter               Energy                 |dE|               |dT|  Wall Time     Memory")
        while it < maxiter:
            
            tic = time.perf_counter()
            
            # Compute residual
            X = self.residual_function(self.hamiltonian, self.T, self.ref, herm=self.herm)
            energy = X['0']
            
            # Update amplitudes
            self.T, dT = self.update_t(self.T, X, self.ref, denom, reg_denom, T_pert=T_pert)
            toc = time.perf_counter()
            minutes, seconds = divmod(toc - tic, 60)
            
            # Record iteration information
            delta_e = energy - e_old
            resid = sum([np.linalg.norm(value.flatten()) for _, value in dT.items()])
            e_old = energy.copy()
            
            # Print the iteration
            print("    {: 5d} {: 20.12f} {: 20.12f} {: 20.12f} {:.2f}m {:.2f}s    {:.2f} MB".format(it,
                                                                                                    energy,
                                                                                                    delta_e,
                                                                                                    resid,
                                                                                                    minutes, seconds,
                                                                                                    get_memory_usage()))
            # Check convergence criterion
            if abs(delta_e) < e_conv and resid < t_conv:
                print(f"    ric-MRCC successfully converged after {it} iterations.")
                break
                
            # DIIS extrapolate
            diis_engine.push(self.T, dT, it)
            if it >= diis_size:  
               self.T = unflatten_vector_to_dict(diis_engine.extrapolate(), T_shapes, T_sizes)
                
            # Update iteration counter
            it += 1
        else:
            print("   ric-MRCC did not converge")
        # Record the energy
        self.correlation_energy = energy
        self.total_energy = self.correlation_energy + self.reference_energy
        # Record total time and print summary
        minutes, seconds = divmod(time.time() - t_start, 60)
        print("")
        print("    Calculation Summary:")
        print("    --------------------")
        print("    Reference Energy: {: 20.12f}".format(self.ref.e_cas))
        print("    Unrelaxed ric-MRCC Correlation Energy: {: 20.12f}".format(self.correlation_energy))
        print("    Unrelaxed ric-MRCC Total Energy: {: 20.12f}".format(self.total_energy))
        print("")
        print("    Largest Singly and Doubly Excited Amplitudes")
        print("    --------------------------------------------")
        self.print_amplitudes()
        print("")
        print("    ric-MRCC calculation completed in {:.2f}m {:.2f}s".format(minutes, seconds))
        print(f"    Memory usage: {get_memory_usage()} MB")
        print("")
        

    def diagonalize_hbar(self, herm):
        print(f"     ==> Similarity-Transformed Hamiltonian Diagonalization <==")
        self.relaxation_energy = np.zeros(self.ref.nstates)
        self.total_energy_relaxed = np.zeros(self.ref.nstates)
        # [TODO]: Construct the entire hole 1- and 2-body hbars. This may improve things.
        # Obtain the similarity-transformed Hamiltonian (1- and 2-body) in the active space
        print("    Building 1- and 2-body components of HBar in the active space... ", end='')
        _t0 = time.time()
        hbar_act = self.build_hbar_active(self.hamiltonian, self.T, self.ref, self.herm)
        print(f"   {time.time() - _t0} seconds\n")
        # Denormal order the Hbar integrals
        print(f"    Denormal order active HBar integrals...")
        hbar_act, e_scalar = denormal_order_ints(hbar_act, self.ref)
        print(f"    <HBar> = {e_scalar}")
        # Semicanonicalize the active-space integrals
        print(f"    Semicanonicalize denormal-ordered active HBar integrals...")
        hbar_act = semicanonicalize_active(hbar_act, self.ref)
        # Diagonalize Hamiltonian in the CAS space using fcipy
        self.ref.cisolver.e1int = hbar_act['a'].copy()
        self.ref.cisolver.e2int = hbar_act['ab'].copy()
        #
        # self.ref.cisolver.run_ci(self.ref.nstates, opt=True, herm=herm)
        #
        self.ref.cisolver.build_hamiltonian(herm=herm)
        self.ref.cisolver.diagonalize_hamiltonian(herm=herm)

        self.relaxation_energy = self.ref.cisolver.state_eigval.real + e_scalar
        self.total_energy_relaxed = self.total_energy + self.relaxation_energy

        state_index, overlap = self.ref.get_state_indices_in_spectrum()
        # i_ground = state_index[0]

        # overlap = np.dot(self.ref.ci_coeff_init.T, self.ref.cisolver.coef)
        # print(state_index)

        # for i in range(self.ref.nstates):
        #
        #     self.relaxation_energy[i] = self.ref.cisolver.state_eigval[i] + e_scalar
        #     self.total_energy_relaxed[i] = self.total_energy + self.relaxation_energy[i]
        #     print("")
        #     print(f"    Calculation Summary: State {i}")
        #     print("    ------------------------------------")
        #     if i == 0:
        #         print("    Reference Energy: {: 20.12f}".format(self.ref.e_cas))
        #         print("    Unrelaxed ric-MRCC Total Energy: {: 20.12f}".format(self.total_energy))
        #     else:
        #         print("    Excitation Energy: {: 20.12f}".format(self.total_energy_relaxed[i] - self.total_energy_relaxed[0]))
        #     print("    Relaxation Energy: {: 20.12f}".format(self.relaxation_energy[i]))
        #     print("    Relaxed ric-MRCC Total Energy: {: 20.12f}".format(self.total_energy_relaxed[i]))
        #     self.ref.cisolver.print_ci_vector(state=i, prtol=0.19)

        # for i in range(overlap.shape[0]):
        #     print(f"State {i}")
        #     for j in range(overlap.shape[1]):
        #         print(f"S[{i}, {j}] = {overlap[i, j]}")
        #     print("")

        # for i, (e_t, e_r) in enumerate(zip(self.total_energy_relaxed, self.relaxation_energy)):
        #     idx = np.argsort(overlap, axis=1)
        #     print("")
        #     print(f"    Calculation Summary: State {i}")
        #     print("    ------------------------------------")
        #     # if i == 0:
        #     #     print("    Reference Energy: {: 20.12f}".format(self.ref.e_cas))
        #     #     print("    Unrelaxed ric-MRCC Total Energy: {: 20.12f}".format(self.total_energy))
        #     # else:
        #     #     print("    Excitation Energy: {: 20.12f}".format(e_t - self.total_energy_relaxed[0]))
        #     print("    Relaxation Energy: {: 20.12f}".format(e_r))
        #     print("    Relaxed ric-MRCC Total Energy: {: 20.12f}".format(e_t))
        #     print("    State Overlaps:")
        #     for j in range(5):
        #         print(f"       [{j + 1}]     {idx[j]}    {overlap[idx[j], i]}")
        #
        #     self.ref.cisolver.print_ci_vector(state=i, prtol=0.19)

        for i, istate in enumerate(state_index):
            # ground_state = False
            print("")
            print(f"    Calculation Summary: State {i}")
            print(f"    initial root {i} -> relaxed root {istate}")
            print(f"    overlap = {overlap[i, istate]}")
            # if istate == i_ground:
            #     ground_state = True
            #     print("    [Ground State]")
            print("    ------------------------------------")
            # if ground_state:
            #     print("    Reference Energy: {: 20.12f}".format(self.ref.e_cas))
            #     print("    Unrelaxed ric-MRCC Total Energy: {: 20.12f}".format(self.total_energy))
            # else:
            #     print("    Excitation Energy: {: 20.12f}".format(self.total_energy_relaxed[istate] - self.total_energy_relaxed[i_ground]))
            print("    Relaxation Energy: {: 20.12f}".format(self.relaxation_energy[istate]))
            print("    Relaxed ric-MRCC Total Energy: {: 20.12f}".format(self.total_energy_relaxed[istate]))
            self.ref.cisolver.print_ci_vector(state=istate, prtol=0.19)


    def print_amplitudes(self):

        nua, nub, noa, nob = self.T['ab'].shape

        n = 1
        for a in range(nua):
            for i in range(noa):
                if abs(self.T['a'][a, i]) <= self._print_threshold: continue
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
                if abs(self.T['b'][a, i]) <= self._print_threshold: continue
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
                        if abs(self.T['aa'][a, b, i, j]) <= self._print_threshold: continue
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
                        if abs(self.T['bb'][a, b, i, j]) <= self._print_threshold: continue
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
                        if abs(self.T['ab'][a, b, i, j]) <= self._print_threshold: continue
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
