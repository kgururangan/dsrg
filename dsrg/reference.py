import time
import numpy as np
import scipy
from dsrg.pyscf_tools import make_casci_rdm123s, make_casci_rdm123, get_pyscf_orbsym, make_hf_integrals
from dsrg.utilities import rotate_1, rotate_2, rotate_2s, rotate_3b, rotate_3c, rotate_3s
try:
    from fcipy.driver import Driver as CI
    from fcipy.system import System as CI_system
except ImportError:
    print("FCIpy not installed - you will not be able to run relaxed or excited-state calculatiosn!")
    pass


# [TODO]: Change how semicanonicalization is handled. In relaxed calculations, the semicanonicalizer
# changes based on how the one-body active integrals change. We sould separate semicanonicalization
# of integrals and RDMs.
class Reference:

    @classmethod
    def from_pyscf(cls, mc, mf, nfrozen, sa_weights=np.array([1.0])):

        nelectron = mf.mol.nelectron
        nuclear_repulsion = mf.mol.energy_nuc()

        mo_coeff = mc.mo_coeff
        e1int_ao, e2int_ao = make_hf_integrals(mf)

        point_group = mf.mol.groupname
        orbsym = get_pyscf_orbsym(mf.mol, mo_coeff)

        nelcas_alpha, nelcas_beta = mc.nelecas
        cas = ((nelcas_alpha, nelcas_beta), mc.ncas)

        ci0 = []
        if isinstance(mc.ci, list):
            for i, ci_vec in enumerate(mc.ci):
                ci0.append(ci_vec.flatten())
        else:
            ci0.append(mc.ci.flatten())

        ci0 = np.array(ci0).T  # shape (ndet, nstates)

        assert ci0.shape[1] == len(sa_weights)

        ref = cls(
            e1int_ao, e2int_ao, mo_coeff, nuclear_repulsion,
            cas=cas, nelectron=nelectron, ci_coeff_init=ci0, nfrozen=nfrozen,
            point_group=point_group, orbsym=orbsym,
            sa_weights=sa_weights
        )

        # Initialize the reference
        ref.initialize()

        # Check that the CAS energy computed via RDMs in reference matches the PySCF result
        print("    Expected State-Averaged CAS Energy = ", mc.e_tot)
        try:
            assert np.allclose(ref.e_cas, mc.e_tot)
            assert np.allclose(ref.e_cas_from_fock, mc.e_tot)
            print("    >> All is well! :)\n")
        except AssertionError:
            raise ValueError("State-averaged CAS energy computed via RDMs does not match!")
        return ref
    
    def __init__(self, 
                 e1int_ao, e2int_ao, mo_coeff, nuclear_repulsion,
                 cas, nelectron, 
                 ci_coeff_init,
                 nfrozen=0,
                 point_group=None, orbsym=None,
                 sa_weights=np.array([1.0])):
        
        self.nfrozen = nfrozen
        self.nelectron = nelectron
        self.mo_coeff = mo_coeff
        self.norb = self.mo_coeff.shape[1]
        self.nuclear_repulsion = nuclear_repulsion
        self.nelcas_alpha, self.nelcas_beta = cas[0]
        self.cas = (self.nelcas_alpha + self.nelcas_beta, cas[1])

        self.e1int_ao = e1int_ao
        self.e2int_ao = e2int_ao

        self.ncore_alpha = self.nelectron // 2 - self.nelcas_alpha 
        self.ncore_beta = self.nelectron // 2 - self.nelcas_beta
        self.nact_alpha = self.cas[1]
        self.nact_beta = self.cas[1]
        self.nvirt_alpha = self.norb - self.nact_alpha - self.ncore_alpha
        self.nvirt_beta = self.norb - self.nact_beta - self.ncore_beta
        #
        self.nstates = len(sa_weights)
        self.sa_weights = sa_weights
        self.cas_state_energies = np.zeros(len(self.sa_weights))
        self.e_frozen_core = 0.0
        self.mo_coeff = mo_coeff
        self.ci_coeff_init = ci_coeff_init
        #
        self.nhole_alpha = self.ncore_alpha + self.nact_alpha
        self.nhole_beta = self.ncore_beta + self.nact_beta
        self.npart_alpha = self.nact_alpha + self.nvirt_alpha
        self.npart_beta = self.nact_beta + self.nvirt_beta
        #
        self.orbspace = {
            'core_alpha': slice(0, self.ncore_alpha),
            'active_alpha': slice(self.ncore_alpha, self.ncore_alpha + self.nact_alpha),
            'virt_alpha': slice(self.ncore_alpha + self.nact_alpha, self.norb),
            'core_beta': slice(0, self.ncore_beta),
            'active_beta': slice(self.ncore_beta, self.ncore_beta + self.nact_beta),
            'virt_beta': slice(self.ncore_beta + self.nact_beta, self.norb),
            'hole_core_alpha': slice(0, self.ncore_alpha),
            'hole_active_alpha': slice(self.ncore_alpha, self.ncore_alpha + self.nact_alpha),
            'particle_active_alpha': slice(0, self.nact_alpha),
            'particle_virt_alpha': slice(self.nact_alpha, self.nact_alpha + self.nvirt_alpha),
            'hole_core_beta': slice(0, self.ncore_beta),
            'hole_active_beta': slice(self.ncore_beta, self.ncore_beta + self.nact_beta),
            'particle_active_beta': slice(0, self.nact_beta),
            'particle_virt_beta': slice(self.nact_beta, self.nact_beta + self.nvirt_beta),
            'hole_alpha': slice(0, self.ncore_alpha + self.nact_alpha),
            'particle_alpha': slice(self.ncore_alpha, self.norb),
            'hole_beta': slice(0, self.ncore_beta + self.nact_beta),
            'particle_beta': slice(self.ncore_beta, self.norb),
        }
        #
        self.rdms = None
        #
        if point_group is None:
            self.point_group = "C1"
        else:
            self.point_group = point_group
        if orbsym is None:
            self.orbsym = ["A" for _ in range(self.norb)]
        else:
            self.orbsym = orbsym

    def initialize(self):
        print(f"\n    ==> Spin-Integrated CAS Reference <==")
        # Make integrals in MO basis (CASSCF or RHF)
        tic = time.time()
        self.ao_to_mo()
        toc = time.time()
        print(f"    Transforming integrals to MO basis... {toc - tic}s")
        # diagonalize the Hamiltonian in the CAS space (this will recover the CASSCF/CASCI energies)
        tic = time.time()
        self.initialize_cisolver()
        self.diagonalize_cas_hamiltonian()
        toc = time.time()
        print(f"    Hamiltonian CAS diagonalization... {toc - tic}s")
        # get RDMs from CAS wave function
        tic = time.time()
        self.make_state_avg_rdms()
        toc = time.time()
        print(f"    RDM construction... {toc - tic}s")
        # make generalized Fock operator
        tic = time.time()
        self.make_fock()
        toc = time.time()
        print(f"    Fock construction... {toc - tic}s")
        # semi-canonicalize integrals and RDMs
        tic = time.time()
        self.get_semicanonicalizer()
        self.semicanonicalize_integrals()
        self.semicanonicalize_rdms()
        toc = time.time()
        print(f"    Semicanonicalize integrals and RDMs... {toc - tic}s")
        # make cumulants from RDMs
        tic = time.time()
        self.make_cumulants()
        toc = time.time()
        print(f"    Make cumulants... {toc - tic}s")
        # check that 1- and 2-RDMs in semicanonical basis is correct by computing CAS energy
        self.compute_sa_cas_energy()
        #
        self.compute_cas_energy_from_fock()
        #
        self.freeze_orbitals()
        print(f"    Freezing {self.nfrozen} doubly occupied orbitals for correlated treatment...")
        print(f"    Frozen core energy = {self.e_frozen_core}")
        #
        print("")
        print("    CAS Reference Summary:")
        print("    -----------------------")
        for i, e in enumerate(self.cas_state_energies):
            print(f"    CAS State {i} Energy = {e}")
        print("")
        print("    State-Averaged CAS Energy (from RDMs) = ", self.e_cas)
        print("    State-Averaged CAS (from RDMs, Fock) = ", self.e_cas_from_fock)
        try:
            assert np.allclose(self.e_cas, self.e_cas_from_fock, atol=1.0e-06, rtol=1.0e-06)
        except AssertionError:
            raise ValueError("CAS energy from RDMs does not match CAS energy from Fock!")

    def ao_to_mo(self):
        """Transform the one- and two-electron integrals to the MO basis."""
        # Make 1- and 2-electron integrals in MO basis
        hcore = np.einsum('pi,pq,qj->ij', self.mo_coeff, self.e1int_ao, self.mo_coeff)
        eri = np.einsum("ijkl,ip,jq,kr,ls->pqrs", self.e2int_ao, self.mo_coeff, self.mo_coeff, self.mo_coeff, self.mo_coeff, optimize=True)
        eri_sym = eri - eri.transpose(0, 1, 3, 2)
        self.Z = {'a': hcore, 'b': hcore}
        self.V = {'aa': eri_sym, 'ab': eri, 'bb': eri_sym}

    def update(self):
        """Update the reference state with a new set of CI vectors."""
        print(f"\n    ==> Updating Spin-Integrated CAS Reference <==")
        # Clear original RDMs
        print('    Making new state-averaged RDMs...')
        self.rdms = None      
        self.make_state_avg_rdms()
        # Make new Fock matrix
        print('    Making new Fock matrix...')
        self.make_fock()
        # Make new semicanonicalizer
        print('    Making new semicanonicaliezer...')
        self.get_semicanonicalizer()
        # Semi-canonicalize integrals and RDMs
        print('    Semicanonicalizing integrals and RDMs...')
        self.semicanonicalize_integrals()
        self.semicanonicalize_rdms()
        # Make new cumulants
        print('    Making new cumulants...')
        self.make_cumulants()
        # Compute SA CAS energy from RDMs
        self.compute_sa_cas_energy()
        self.compute_cas_energy_from_fock()
        print("")
        print("    CAS Reference Summary:")
        print("    -----------------------")
        for i, e in enumerate(self.cas_state_energies):
            print(f"    CAS State {i} Energy = {e}")
        print("")
        print("    State-Averaged CAS Energy (from RDMs) = ", self.e_cas)
        print("    State-Averaged CAS (from RDMs, Fock) = ", self.e_cas_from_fock)
        try:
            assert np.allclose(self.e_cas, self.e_cas_from_fock, atol=1.0e-06, rtol=1.0e-06)
        except AssertionError:
            raise ValueError("CAS energy from RDMs does not match CAS energy from Fock!")

    def get_state_indices_in_spectrum(self):
        overlap = np.dot(self.ci_coeff_init.T, self.cisolver.coef)
        # self.state_index = np.argmax(np.abs(overlap), axis=1)
        return np.argmax(np.abs(overlap), axis=1), overlap

    def initialize_cisolver(self):
        # Important to copy one- and two-electron integrals - take only hole part (core + active)
        assert self.ncore_alpha == self.ncore_beta
        assert self.nact_alpha == self.nact_beta
        a = self.orbspace['active_alpha']
        A = self.orbspace['active_beta']
        h = self.orbspace['hole_alpha']
        H = self.orbspace['hole_beta']
        nel = self.nelectron
        norb = self.ncore_alpha + self.nact_alpha
        nfroz = self.ncore_alpha
        ndelete = 0
        mult = 1
        self.cisolver = CI.from_custom(
            nel, norb, nfroz, ndelete, mult,
            self.Z['a'][h, h].copy(), self.V['ab'][h, H, h, H].copy(), self.nuclear_repulsion,
            point_group=self.point_group, orbital_symmetry=self.orbsym,
        )
        self.cisolver.load_determinants(target_irrep=None)

    def diagonalize_cas_hamiltonian(self):
        # Initialize CI coefficients
        self.cisolver.build_hamiltonian(herm=True)
        self.cisolver.diagonalize_hamiltonian(herm=True)

        state_index, overlap = self.get_state_indices_in_spectrum()

        print("   State Energies from CAS Diagonalization")
        print("   ----------------------------------------")
        for i, istate in enumerate(state_index):
            print(f"   Overlap with initial state {i}: {overlap[i, istate]}")
            self.ci_coeff_init[:, i] = self.cisolver.coef[:, istate].copy()
            print(f"   State {istate}: Energy = {self.cisolver.total_energy[istate]}")
            self.cisolver.print_ci_vector(state=istate, prtol=0.19)
            print("")

    def make_state_avg_rdms(self):

        state_index, _ = self.get_state_indices_in_spectrum()

        for i, (istate, w) in enumerate(zip(state_index, self.sa_weights)):
            # Use FCIpy to get 1-, 2-, and 3-RDMs
            rdms_i = self.cisolver.compute_rdm123s(istate)
            # Compute CAS state energies using RDMs
            self.cas_state_energies[i] = self.compute_cas_state_energy(rdms_i)
            # Check that the CI diagonalization and the RDMs produce the same energy
            # assert np.allclose(self.cas_state_energies[i], self.cisolver.total_energy[istate], atol=1.0e-06, rtol=1.0e-06)
            if self.rdms is None:
                print('    Initializing RDMs')
                self.rdms = {key: None for key in rdms_i}
                for key, value in rdms_i.items():
                    self.rdms[key] = w * value
            else:
                for key, value in rdms_i.items():
                    self.rdms[key] += w * value

            # Print out natural occupancies from 1-RDM
            occ_a, _ = np.linalg.eigh(rdms_i['a'])
            occ_b, _ = np.linalg.eigh(rdms_i['b'])
            print(f"\n    Natural occupancies of CAS state {i}")
            print("    -------------------------------------")
            for j, (na, nb) in enumerate(zip(occ_a, occ_b)):
                print(f"    CAS orbital {j + 1}:  {na + nb:.5f}")
        print('')

    def make_cumulants(self):

        # Make 1-body cumulants
        gam1a = self.rdms['a'].copy()
        eta1a = np.eye(gam1a.shape[0]) - gam1a
        gam1b = self.rdms['b'].copy()
        eta1b = np.eye(gam1b.shape[0]) - gam1b

        # Make full 1-body cumulants (just in case)
        c = self.orbspace['core_alpha']
        C = self.orbspace['core_beta']
        a = self.orbspace['active_alpha']
        A = self.orbspace['active_beta']
        v = self.orbspace['virt_alpha']
        V = self.orbspace['virt_beta']
        #
        gam1a_full = np.zeros((self.norb, self.norb))
        gam1a_full[c, c] = np.eye(self.ncore_alpha)
        gam1a_full[a, a] = gam1a.copy()
        #
        gam1b_full = np.zeros((self.norb, self.norb))
        gam1b_full[C, C] = np.eye(self.ncore_beta)
        gam1b_full[A, A] = gam1b.copy()
        #
        eta1a_full = np.zeros((self.norb, self.norb))
        eta1a_full[a, a] = eta1a.copy()
        eta1a_full[v, v] = np.eye(self.nvirt_alpha)
        #
        eta1b_full = np.zeros((self.norb, self.norb)) 
        eta1b_full[A, A] = eta1b.copy()
        eta1b_full[V, V] = np.eye(self.nvirt_beta)

        # Make 2-body cumulants
        lam2aa = -np.einsum("uw,vx->uvwx", gam1a, gam1a, optimize=True)
        lam2aa -= lam2aa.transpose(1, 0, 2, 3)
        lam2aa += self.rdms['aa']
        #
        lam2ab = -np.einsum("uw,vx->uvwx", gam1a, gam1b, optimize=True)
        lam2ab += self.rdms['ab']
        #
        lam2bb = -np.einsum("uw,vx->uvwx", gam1b, gam1b, optimize=True)
        lam2bb -= lam2bb.transpose(1, 0, 2, 3)
        lam2bb += self.rdms['bb']

        # Make 3-body cumulants
        # l(abcijk) = g(abcijk) - A(a/bc)A(i/jk) g(ai)g(bcjk) + 2*A(ijk) g(ai)g(bj)g(ck)
        lam3aaa = self.rdms['aaa'].copy()
        temp = np.einsum("ai,bcjk->abcijk", self.rdms['a'], self.rdms['aa'], optimize=True)
        temp -= np.transpose(temp, (1, 0, 2, 3, 4, 5)) + np.transpose(temp, (2, 1, 0, 3, 4, 5)) # (a/bc)
        temp -= np.transpose(temp, (0, 1, 2, 4, 3, 5)) + np.transpose(temp, (0, 1, 2, 5, 4, 3)) # (i/jk)
        lam3aaa -= temp
        temp = np.einsum("ai,bj,ck->abcijk", self.rdms['a'], self.rdms['a'], self.rdms['a'], optimize=True)
        temp -= np.transpose(temp, (0, 1, 2, 3, 5, 4)) # (jk)
        temp -= np.transpose(temp, (0, 1, 2, 4, 3, 5)) + np.transpose(temp, (0, 1, 2, 5, 4, 3)) # (i/jk)
        lam3aaa += 2.0 * temp
        # l(abc~ijk~) = g(abc~ijk~) - A(ab)A(ij) g(ai)g(bc~jk~) - g(c~k~)g(abij) + 2*A(ij) g(ai)g(bj)g(c~k~)
        lam3aab = self.rdms['aab'].copy()
        temp = np.einsum("ai,bcjk->abcijk", self.rdms['a'], self.rdms['ab'], optimize=True)
        temp -= np.transpose(temp, (1, 0, 2, 3, 4, 5)) # (ab)
        temp -= np.transpose(temp, (0, 1, 2, 4, 3, 5)) # (ij)
        lam3aab -= temp
        temp = np.einsum("ck,abij->abcijk", self.rdms['b'], self.rdms['aa'], optimize=True)
        lam3aab -= temp
        temp = np.einsum("ai,bj,ck->abcijk", self.rdms['a'], self.rdms['a'], self.rdms['b'], optimize=True)
        temp -= np.transpose(temp, (0, 1, 2, 4, 3, 5)) # (ij)
        lam3aab += 2.0 * temp
        # l(ab~c~ij~k~) = g(ab~c~ij~k~) - g(ai)g(b~c~j~k~) - A(bc)A(jk) g(b~j~)g(ac~ik~) + 2*A(jk) g(ai)g(b~j~)g(c~k~)
        lam3abb = self.rdms['abb'].copy()
        temp = np.einsum("ai,bcjk->abcijk", self.rdms['a'], self.rdms['bb'], optimize=True)
        lam3abb -= temp
        temp = np.einsum("bj,acik->abcijk", self.rdms['b'], self.rdms['ab'], optimize=True)
        temp -= np.transpose(temp, (0, 2, 1, 3, 4, 5)) # (bc)
        temp -= np.transpose(temp, (0, 1, 2, 3, 5, 4)) # (jk)
        lam3abb -= temp
        temp = np.einsum("ai,bj,ck->abcijk", self.rdms['a'], self.rdms['b'], self.rdms['b'], optimize=True)
        temp -= np.transpose(temp, (0, 1, 2, 3, 5, 4)) # (jk)
        lam3abb += 2.0 * temp
        # l(a~b~c~i~j~k~) = g(a~b~c~i~j~k~) - A(a/bc)A(i/jk) g(a~i~)g(b~c~j~k~) + 2*A(ijk) g(a~i~)g(b~j~)g(c~k~)
        lam3bbb = self.rdms['bbb'].copy()
        temp = np.einsum("ai,bcjk->abcijk", self.rdms['b'], self.rdms['bb'], optimize=True)
        temp -= np.transpose(temp, (1, 0, 2, 3, 4, 5)) + np.transpose(temp, (2, 1, 0, 3, 4, 5)) # (a/bc)
        temp -= np.transpose(temp, (0, 1, 2, 4, 3, 5)) + np.transpose(temp, (0, 1, 2, 5, 4, 3)) # (i/jk)
        lam3bbb -= temp
        temp = np.einsum("ai,bj,ck->abcijk", self.rdms['b'], self.rdms['b'], self.rdms['b'], optimize=True)
        temp -= np.transpose(temp, (0, 1, 2, 3, 5, 4)) # (jk)
        temp -= np.transpose(temp, (0, 1, 2, 4, 3, 5)) + np.transpose(temp, (0, 1, 2, 5, 4, 3)) # (i/jk)
        lam3bbb += 2.0 * temp

        self.gam1 = {'a': gam1a, 'b': gam1b}
        self.gam1_full = {'a': gam1a_full, 'b': gam1b_full}
        self.eta1 = {'a': eta1a, 'b': eta1b}
        self.eta1_full = {'a': eta1a_full, 'b': eta1b_full}
        self.lambdas = {'aa': lam2aa, 'ab': lam2ab, 'bb': lam2bb,
                        'aaa': lam3aaa, 'aab': lam3aab, 'abb': lam3abb, 'bbb': lam3bbb}
        
    def make_fock(self):
        c = self.orbspace['core_alpha']
        C = self.orbspace['core_beta']
        a = self.orbspace['active_alpha']
        A = self.orbspace['active_beta']
        
        fock_a = (
                    self.Z['a'] 
                    + np.einsum("piqi->pq", self.V['aa'][:, c, :, c])
                    + np.einsum("puqv,uv->pq", self.V['aa'][:, a, :, a], self.rdms['a'])
                    + np.einsum("piqi->pq", self.V['ab'][:, C, :, C])
                    + np.einsum("puqv,uv->pq", self.V['ab'][:, A, :, A], self.rdms['b'])
        )
        fock_b = (
                    self.Z['b'] 
                    + np.einsum("piqi->pq", self.V['bb'][:, C, :, C])
                    + np.einsum("puqv,uv->pq", self.V['bb'][:, A, :, A], self.rdms['b'])
                    + np.einsum("ipiq->pq", self.V['ab'][c, :, c, :])
                    + np.einsum("upvq,uv->pq", self.V['ab'][a, :, a, :], self.rdms['a'])
        )
        self.F = {'a': fock_a, 'b': fock_b}
        
    def get_semicanonicalizer(self):

        def _diagonalize_and_reorder(f):

            herm_error = np.linalg.norm(f - np.conj(f).T)
            # print(f"    |F - F.T| = {herm_error}")
            if herm_error > 1.0e-09: # non-Hermitian

                print("Warning: Fock matrix is not Hermitian! Diagonalizing as non-Hermitian matrix.")

                e, u_left, u_right = scipy.linalg.eig(f, left=True, right=True)
                isort = np.argsort(np.real(e))
                u_left = u_left[:, isort]
                u_right = u_right[:, isort]

                # Why is this throwing for virtuals?
                # test = np.dot(np.conj(u_left).T, u_right)
                # for i in range(test.shape[0]):
                #     u_left[:, i] /= test[i, i]

                n = e.shape[0]
                for i in range(n):
                    # (a) normalize the i-th pair so that <U_L[:,i] | U_R[:,i]> = 1
                    scale = np.vdot(u_left[:, i], u_right[:, i])   # vdot = conj(U_L[:,i]) @ U_R[:,i]
                    u_left[:, i] /= scale

                    # (b) orthogonalize U_L[:, i] against all previously fixed U_R[:, j], j < i
                    for j in range(i):
                        # project out any component of U_R[:, j] from U_L[:, i]
                        coeff = np.vdot(u_left[:, i], u_right[:, j])
                        u_left[:, i] -= coeff * u_left[:, j]

                    # (c) Now enforce <U_L[:,i] | U_R[:,i]> = 1 again (since step (b) can re‐introduce
                    #     a small scaling error)
                    scale2 = np.vdot(u_left[:, i], u_right[:, i])
                    u_left[:, i] /= scale2

                test = np.dot(np.conj(u_left).T, u_right)

                print(np.linalg.norm(test - np.eye(u_left.shape[1])))
                try:
                    assert np.allclose(test, np.eye(u_left.shape[1]), atol=1.0e-09, rtol=1.0e-09)
                except AssertionError:
                    "Left and right eigenvectors are not orthogonal!"
                    for i in range(f.shape[0]):
                        for j in range(f.shape[1]):
                            print(f"L*R[{i}, {j}] = {test[i, j]}")

            else: # Hermitian
                e, u = scipy.linalg.eigh(f)
                isort = np.argsort(e)
                u_left = u[:, isort]
                u_right = u[:, isort]

            return u_left, u_right

        # diagonalize fock_a and fock_b in cc, aa, and vv sectors to get alpha and beta semicanonicalizers
        c = self.orbspace['core_alpha']
        C = self.orbspace['core_beta']
        a = self.orbspace['active_alpha']
        A = self.orbspace['active_beta']
        v = self.orbspace['virt_alpha']
        V = self.orbspace['virt_beta']

        self.U_L = {'a': np.zeros_like(self.F['a']), 'b': np.zeros_like(self.F['b'])}
        self.U_R = {'a': np.zeros_like(self.F['a']), 'b': np.zeros_like(self.F['b'])}
        for ispin, f in self.F.items():
            if ispin == 'a':
                self.U_L[ispin][c, c], self.U_R[ispin][c, c] = _diagonalize_and_reorder(f[c, c].copy())
                self.U_L[ispin][a, a], self.U_R[ispin][a, a] = _diagonalize_and_reorder(f[a, a].copy())
                self.U_L[ispin][v, v], self.U_R[ispin][v, v] = _diagonalize_and_reorder(f[v, v].copy())
            if ispin == 'b':
                self.U_L[ispin][C, C], self.U_R[ispin][C, C] = _diagonalize_and_reorder(f[C, C].copy())
                self.U_L[ispin][A, A], self.U_R[ispin][A, A] = _diagonalize_and_reorder(f[A, A].copy())
                self.U_L[ispin][V, V], self.U_R[ispin][V, V] = _diagonalize_and_reorder(f[V, V].copy())

        self.U_L['b'] = self.U_L['a'].copy()
        self.U_R['b'] = self.U_R['a'].copy()
        
    def semicanonicalize_integrals(self):
        # semi-canonicalize 1- and 2-body integrals
        self.Z['a'] = rotate_1(self.U_L['a'], self.U_R['a'], self.Z['a'].copy())
        self.Z['b'] = rotate_1(self.U_L['b'], self.U_R['b'], self.Z['b'].copy())
        self.F['a'] = rotate_1(self.U_L['a'], self.U_R['a'], self.F['a'].copy())
        self.F['b'] = rotate_1(self.U_L['b'], self.U_R['b'], self.F['b'].copy())
        self.V['aa'] = rotate_2s(self.U_L['a'], self.U_R['a'], self.V['aa'].copy())
        self.V['ab'] = rotate_2(self.U_L['a'], self.U_L['b'], self.U_R['a'], self.U_R['b'], self.V['ab'].copy())
        self.V['bb'] = rotate_2s(self.U_L['b'], self.U_R['b'], self.V['bb'].copy())

        # for key, value in self.Z.items():
        #     print(f"{key} -> {np.linalg.norm(value.flatten())}")
        # for key, value in self.F.items():
        #     print(f"{key} -> {np.linalg.norm(value.flatten())}")
        # for key, value in self.V.items():
        #     print(f"{key} -> {np.linalg.norm(value.flatten())}")

    def semicanonicalize_rdms(self):
        a = self.orbspace['active_alpha']
        A = self.orbspace['active_beta']
        # semi-canonicalize 1-, 2-, and 3-body RDMs
        self.rdms['a'] = rotate_1(self.U_L['a'][a, a], self.U_R['a'][a, a], self.rdms['a'].copy())
        self.rdms['b'] = rotate_1(self.U_L['b'][A, A], self.U_L['b'][A, A], self.rdms['b'].copy())
        self.rdms['aa'] = rotate_2s(self.U_L['a'][a, a], self.U_L['a'][a, a], self.rdms['aa'].copy())
        self.rdms['ab'] = rotate_2(self.U_L['a'][a, a], self.U_L['b'][A, A], self.U_R['a'][a, a], self.U_R['b'][A, A], self.rdms['ab'].copy())
        self.rdms['bb'] = rotate_2s(self.U_L['b'][A, A], self.U_R['b'][A, A], self.rdms['bb'].copy())
        self.rdms['aaa'] = rotate_3s(self.U_L['a'][a, a], self.U_R['a'][a, a], self.rdms['aaa'].copy())
        self.rdms['aab'] = rotate_3b(self.U_L['a'][a, a], self.U_L['b'][A, A], self.U_R['a'][a, a], self.U_R['b'][A, A], self.rdms['aab'].copy())
        self.rdms['abb'] = rotate_3c(self.U_L['a'][a, a], self.U_L['b'][A, A], self.U_R['a'][a, a], self.U_R['b'][A, A], self.rdms['abb'].copy())
        self.rdms['bbb'] = rotate_3s(self.U_L['b'][A, A], self.U_R['b'][A, A], self.rdms['bbb'].copy())

    def compute_cas_state_energy(self, rdms):
        c = self.orbspace['core_alpha']
        C = self.orbspace['core_beta']
        a = self.orbspace['active_alpha']
        A = self.orbspace['active_beta']
        e_test = (
            np.einsum("mm->", self.Z['a'][c, c]) 
          + np.einsum("mm->", self.Z['b'][C, C])
          + np.einsum("uv,uv->", self.Z['a'][a, a], rdms['a'])
          + np.einsum("uv,uv->", self.Z['b'][A, A], rdms['b'])
          + 0.5 * np.einsum("mnmn->", self.V['aa'][c, c, c, c])
          + np.einsum("mnmn->", self.V['ab'][c, C, c, C])
          + 0.5 * np.einsum("mnmn->", self.V['bb'][C, C, C, C])
          + np.einsum("mumv,uv->", self.V['aa'][c, a, c, a], rdms['a'])
          + np.einsum("umvm,uv->", self.V['ab'][a, C, a, C], rdms['a'])
          + np.einsum("mumv,uv->", self.V['ab'][c, A, c, A], rdms['b'])
          + np.einsum("mumv,uv->", self.V['bb'][C, A, C, A], rdms['b'])
          + 0.25 * np.einsum("uvxy,uvxy->", self.V['aa'][a, a, a, a], rdms['aa'])
          + np.einsum("uvxy,uvxy->", self.V['ab'][a, A, a, A], rdms['ab'])
          + 0.25 * np.einsum("uvxy,uvxy->", self.V['bb'][A, A, A, A], rdms['bb'])
          + self.nuclear_repulsion
        )
        return e_test

    def compute_sa_cas_energy(self):
        c = self.orbspace['core_alpha']
        C = self.orbspace['core_beta']
        a = self.orbspace['active_alpha']
        A = self.orbspace['active_beta']
        e_test = (
            np.einsum("mm->", self.Z['a'][c, c])
          + np.einsum("mm->", self.Z['b'][C, C])
          + np.einsum("uv,uv->", self.Z['a'][a, a], self.rdms['a'])
          + np.einsum("uv,uv->", self.Z['b'][A, A], self.rdms['b'])
          + 0.5 * np.einsum("mnmn->", self.V['aa'][c, c, c, c])
          + np.einsum("mnmn->", self.V['ab'][c, C, c, C])
          + 0.5 * np.einsum("mnmn->", self.V['bb'][C, C, C, C])
          + np.einsum("mumv,uv->", self.V['aa'][c, a, c, a], self.rdms['a'])
          + np.einsum("umvm,uv->", self.V['ab'][a, C, a, C], self.rdms['a'])
          + np.einsum("mumv,uv->", self.V['ab'][c, A, c, A], self.rdms['b'])
          + np.einsum("mumv,uv->", self.V['bb'][C, A, C, A], self.rdms['b'])
          + 0.25 * np.einsum("uvxy,uvxy->", self.V['aa'][a, a, a, a], self.rdms['aa'])
          + np.einsum("uvxy,uvxy->", self.V['ab'][a, A, a, A], self.rdms['ab'])
          + 0.25 * np.einsum("uvxy,uvxy->", self.V['bb'][A, A, A, A], self.rdms['bb'])
          + self.nuclear_repulsion
        )

        e_test2 = np.dot(self.sa_weights, self.cas_state_energies)
        assert np.allclose(e_test, e_test2, atol=1.0e-08, rtol=1.0e-08)
        self.e_cas = e_test

    def compute_cas_energy_from_fock(self):
        c = self.orbspace['core_alpha']
        C = self.orbspace['core_beta']
        a = self.orbspace['active_alpha']
        A = self.orbspace['active_beta']
        e_test = (
            np.einsum("mm->", self.F['a'][c, c])
          + np.einsum("mm->", self.F['b'][C, C])
          + np.einsum("uv,uv->", self.F['a'][a, a], self.rdms['a'])
          + np.einsum("uv,uv->", self.F['b'][A, A], self.rdms['b'])
          - 0.5 * np.einsum("mnmn->", self.V['aa'][c, c, c, c])
          - np.einsum("mnmn->", self.V['ab'][c, C, c, C])
          - 0.5 * np.einsum("mnmn->", self.V['bb'][C, C, C, C])
          - np.einsum("mumv,uv->", self.V['aa'][c, a, c, a], self.rdms['a'])
          - np.einsum("umvm,uv->", self.V['ab'][a, C, a, C], self.rdms['a'])
          - np.einsum("mumv,uv->", self.V['ab'][c, A, c, A], self.rdms['b'])
          - np.einsum("mumv,uv->", self.V['bb'][C, A, C, A], self.rdms['b'])
          - 0.5 * np.einsum("xuyv,xy,uv->", self.V['aa'][a, a, a, a], self.rdms['a'], self.rdms['a'])
          - np.einsum("xuyv,xy,uv->", self.V['ab'][a, A, a, A], self.rdms['a'], self.rdms['b'])
          - 0.5 * np.einsum("xuyv,xy,uv->", self.V['bb'][A, A, A, A], self.rdms['b'], self.rdms['b'])
          + 0.25 * np.einsum("uvxy,uvxy->", self.V['aa'][a, a, a, a], self.lambdas['aa'])
          + np.einsum("uvxy,uvxy->", self.V['ab'][a, A, a, A], self.lambdas['ab'])
          + 0.25 * np.einsum("uvxy,uvxy->", self.V['bb'][A, A, A, A], self.lambdas['bb'])
          + self.nuclear_repulsion
        )
        # Using full gamma and eta matrices
        #
        #np.einsum("pq,pq->", self.F['a'], self.gam1_full['a'], optimize=True)
        #+ np.einsum("pq,pq->", self.F['b'], self.gam1_full['b'], optimize=True)
        #- 0.5 * np.einsum("pqrs,pr,qs->", self.V['aa'], self.gam1_full['a'], self.gam1_full['a'], optimize=True)
        #- np.einsum("pqrs,pr,qs->", self.V['ab'], self.gam1_full['a'], self.gam1_full['b'], optimize=True)
        #- 0.5 * np.einsum("pqrs,pr,qs->", self.V['bb'], self.gam1_full['b'], self.gam1_full['b'], optimize=True)
        #+ 0.25 * np.einsum("uvxy,uvxy->", self.V['aa'][a, a, a, a], self.lambdas['aa'])
        #+ np.einsum("uvxy,uvxy->", self.V['ab'][a, A, a, A], self.lambdas['ab'])
        #+ 0.25 * np.einsum("uvxy,uvxy->", self.V['bb'][A, A, A, A], self.lambdas['bb'])
        #+ self.nuclear_repulsion
        self.e_cas_from_fock = e_test

    def freeze_orbitals(self):
        # compute the energy of frozen-core orbitals in the basis of CAS orbitals
        f = slice(0, self.nfrozen)
        a = self.orbspace['active_alpha']
        A = self.orbspace['active_beta']
        # self.e_frozen_core = (
        #     # fc-fc
        #       np.einsum("ii->", self.Z['a'][f, f], optimize=True)
        #     + np.einsum("ii->", self.Z['b'][f, f], optimize=True)
        #     + 0.5 * np.einsum("ijij->", self.V['aa'][f, f, f, f], optimize=True)
        #     + np.einsum("ijij->", self.V['ab'][f, f, f, f], optimize=True)
        #     + 0.5 * np.einsum('ijij->', self.V['bb'][f, f, f, f], optimize=True)
        #     # fc-act
        #     + 0.5 * np.einsum("iuiv,vu->", self.V['aa'][f, a, f, a], self.rdms['a'], optimize=True)
        #     + np.einsum("iuiv,vu->", self.V['ab'][f, A, f, A], self.rdms['b'], optimize=True)
        #     + np.einsum("uivi,vu->", self.V['ab'][a, f, a, f], self.rdms['a'], optimize=True)
        #     + 0.5 * np.einsum("iuiv,vu->", self.V['bb'][f, A, f, A], self.rdms['b'], optimize=True)
        # )
        self.e_frozen_core = (
              np.einsum("mm->", self.F['a'][f, f])
            + np.einsum("mm->", self.F['b'][f, f])
            - 0.5 * np.einsum("mnmn->", self.V['aa'][f, f, f, f])
            - np.einsum("mnmn->", self.V['ab'][f, f, f, f])
            - 0.5 * np.einsum("mnmn->", self.V['bb'][f, f, f, f])
            - np.einsum("mumv,uv->", self.V['aa'][f, a, f, a], self.rdms['a'])
            - np.einsum("umvm,uv->", self.V['ab'][a, f, a, f], self.rdms['a'])
            - np.einsum("mumv,uv->", self.V['ab'][f, A, f, A], self.rdms['b'])
            - np.einsum("mumv,uv->", self.V['bb'][f, A, f, A], self.rdms['b'])
        )
        self.nelectron -= 2*self.nfrozen
        self.norb -= self.nfrozen
        self.ncore_alpha -= self.nfrozen 
        self.ncore_beta -= self.nfrozen 
        #
        self.nhole_alpha = self.ncore_alpha + self.nact_alpha
        self.nhole_beta = self.ncore_beta + self.nact_beta
        #
        self.orbspace = {
            'core_alpha': slice(0, self.ncore_alpha),
            'active_alpha': slice(self.ncore_alpha, self.ncore_alpha + self.nact_alpha),
            'virt_alpha': slice(self.ncore_alpha + self.nact_alpha, self.norb),
            'core_beta': slice(0, self.ncore_beta),
            'active_beta': slice(self.ncore_beta, self.ncore_beta + self.nact_beta),
            'virt_beta': slice(self.ncore_beta + self.nact_beta, self.norb),
            'hole_core_alpha': slice(0, self.ncore_alpha),
            'hole_active_alpha': slice(self.ncore_alpha, self.ncore_alpha + self.nact_alpha),
            'particle_active_alpha': slice(0, self.nact_alpha),
            'particle_virt_alpha': slice(self.nact_alpha, self.nact_alpha + self.nvirt_alpha),
            'hole_core_beta': slice(0, self.ncore_beta),
            'hole_active_beta': slice(self.ncore_beta, self.ncore_beta + self.nact_beta),
            'particle_active_beta': slice(0, self.nact_beta),
            'particle_virt_beta': slice(self.nact_beta, self.nact_beta + self.nvirt_beta),
            'hole_alpha': slice(0, self.ncore_alpha + self.nact_alpha),
            'particle_alpha': slice(self.ncore_alpha, self.norb),
            'hole_beta': slice(0, self.ncore_beta + self.nact_beta),
            'particle_beta': slice(self.ncore_beta, self.norb),
        }
        #
        corr = slice(self.nfrozen, self.norb + self.nfrozen)
        self.F['a'] = self.F['a'][corr, corr]
        self.F['b'] = self.F['b'][corr, corr]
        self.V['aa'] = self.V['aa'][corr, corr, corr, corr]
        self.V['ab'] = self.V['ab'][corr, corr, corr, corr]
        self.V['bb'] = self.V['bb'][corr, corr, corr, corr]

class FakeReference:
    
    def __init__(self, mf, nfrozen=0, verbose=False):
        self.mf = mf
        self.nfrozen = nfrozen
        self.nelectron = mf.mol.nelectron
        self.norb = mf.mo_coeff.shape[1]
        self.nuclear_repulsion = self.mf.mol.energy_nuc()
        self.ncore_alpha = self.nelectron // 2
        self.ncore_beta = self.nelectron // 2
        self.nact_alpha = 0
        self.nact_beta = 0
        self.nvirt_alpha = self.norb - self.nact_alpha - self.ncore_alpha
        self.nvirt_beta = self.norb - self.nact_beta - self.ncore_beta
        #
        self.mo_coeff = self.mf.mo_coeff
        #
        self.nhole_alpha = self.ncore_alpha + self.nact_alpha
        self.nhole_beta = self.ncore_beta + self.nact_beta
        self.npart_alpha = self.nact_alpha + self.nvirt_alpha
        self.npart_beta = self.nact_beta + self.nvirt_beta
        #
        self.orbspace = {
            'core_alpha': slice(0, self.ncore_alpha),
            'active_alpha': slice(self.ncore_alpha, self.ncore_alpha + self.nact_alpha),
            'virt_alpha': slice(self.ncore_alpha + self.nact_alpha, self.norb),
            'core_beta': slice(0, self.ncore_beta),
            'active_beta': slice(self.ncore_beta, self.ncore_beta + self.nact_beta),
            'virt_beta': slice(self.ncore_beta + self.nact_beta, self.norb),
            'hole_core_alpha': slice(0, self.ncore_alpha),
            'hole_active_alpha': slice(self.ncore_alpha, self.ncore_alpha + self.nact_alpha),
            'particle_active_alpha': slice(0, self.nact_alpha),
            'particle_virt_alpha': slice(self.nact_alpha, self.nact_alpha + self.nvirt_alpha),
            'hole_core_beta': slice(0, self.ncore_beta),
            'hole_active_beta': slice(self.ncore_beta, self.ncore_beta + self.nact_beta),
            'particle_active_beta': slice(0, self.nact_beta),
            'particle_virt_beta': slice(self.nact_beta, self.nact_beta + self.nvirt_beta),
            'hole_alpha': slice(0, self.ncore_alpha + self.nact_alpha),
            'particle_alpha': slice(self.ncore_alpha, self.norb),
            'hole_beta': slice(0, self.ncore_beta + self.nact_beta),
            'particle_beta': slice(self.ncore_beta, self.norb),
        }
        #
        self.verbose = verbose

    def kernel(self, semi=True):
        if self.verbose: print("Spin-Integrated CAS Reference")
        if self.verbose: print(f"Semicanonicalization = {semi}")
        # first make the hcore/V integrals in HF basis
        tic = time.time()
        self.make_hf_integrals()
        toc = time.time()
        if self.verbose: print(f"HF integral construction... {toc - tic}s")
        # get RDMs from CAS wave function
        tic = time.time()
        self.rdms = {}
        self.rdms['a'] = np.zeros(shape=(0, 0))
        self.rdms['b'] = np.zeros(shape=(0, 0))
        self.rdms['aa'] = np.zeros(shape=(0, 0, 0, 0))
        self.rdms['ab'] = np.zeros(shape=(0, 0, 0, 0))
        self.rdms['bb'] = np.zeros(shape=(0, 0, 0, 0))
        self.rdms['aaa'] = np.zeros(shape=(0, 0, 0, 0, 0, 0))
        self.rdms['aab'] = np.zeros(shape=(0, 0, 0, 0, 0, 0))
        self.rdms['abb'] = np.zeros(shape=(0, 0, 0, 0, 0, 0))
        self.rdms['bbb'] = np.zeros(shape=(0, 0, 0, 0, 0, 0))
        #self.make_rdms()
        toc = time.time()
        if self.verbose: print(f"RDM construction... {toc - tic}s")
        # make generalized Fock operator
        tic = time.time()
        self.make_fock()
        toc = time.time()
        if self.verbose: print(f"Fock construction... {toc - tic}s")
        # semi-canonicalize integrals and RDMs
        if semi:
            pass
        # make cumulants from RDMs
        tic = time.time()
        self.make_cumulants()
        toc = time.time()
        if self.verbose: print(f"Make cumulants... {toc - tic}s")
        # check that 1- and 2-RDMs in semicanonical basis is correct by computing CAS energy
        self.compute_cas_energy()
        #
        self.compute_cas_energy_from_fock()
        #
        self.freeze_orbitals()
        if self.verbose: print(f"Freezing {self.nfrozen} doubly occupied orbitals for correlated treatment...")
        #
        if self.verbose:
            print("CAS (from RDMs) = ", self.e_cas)
            print("CAS (from RDMs, Fock) = ", self.e_cas_from_fock)
            print("Expected CAS = ", self.mf.e_tot)
        try:
            assert np.allclose(self.e_cas, self.mf.e_tot)
            assert np.allclose(self.e_cas_from_fock, self.mf.e_tot)
            if self.verbose: print("All is well!")
        except AssertionError:
            print("CAS energy computed via RDMs does not match!")
        
    def make_hf_integrals(self):

        # Note: You cannot replace this the T + V construction with mf.get_hcore() when using
        # a CASCI calculation in conjunction with mf!
        hcore_ao = self.mf.mol.intor_symmetric('int1e_kin') + self.mf.mol.intor_symmetric('int1e_nuc')
        hcore = np.einsum('pi,pq,qj->ij', self.mo_coeff, hcore_ao, self.mo_coeff)
        
        eri = self.mf.mol.intor("int2e_sph", aosym="s1").transpose(0, 2, 1, 3)
        eri = np.einsum("ijkl,ip,jq,kr,ls->pqrs", eri, self.mo_coeff, self.mo_coeff, self.mo_coeff, self.mo_coeff, optimize=True)
        
        eri_aa = eri - eri.transpose(0, 1, 3, 2)
        eri_ab = eri.copy()
        eri_bb = eri - eri.transpose(0, 1, 3, 2)
        
        self.Z = {'a': hcore, 'b': hcore}
        self.V = {'aa': eri_aa, 'ab': eri_ab, 'bb': eri_bb}

    def make_cumulants(self):

        # Make 1-body cumulants
        gam1a = self.rdms['a'].copy()
        eta1a = np.eye(gam1a.shape[0]) - gam1a
        gam1b = self.rdms['b'].copy()
        eta1b = np.eye(gam1b.shape[0]) - gam1b

        # Make full 1-body cumulants (just in case)
        c = self.orbspace['core_alpha']
        C = self.orbspace['core_beta']
        a = self.orbspace['active_alpha']
        A = self.orbspace['active_beta']
        v = self.orbspace['virt_alpha']
        V = self.orbspace['virt_beta']
        #
        gam1a_full = np.zeros((self.norb, self.norb))
        gam1a_full[c, c] = np.eye(self.ncore_alpha)
        gam1a_full[a, a] = gam1a.copy()
        #
        gam1b_full = np.zeros((self.norb, self.norb))
        gam1b_full[C, C] = np.eye(self.ncore_beta)
        gam1b_full[A, A] = gam1b.copy()
        #
        eta1a_full = np.zeros((self.norb, self.norb))
        eta1a_full[a, a] = eta1a.copy()
        eta1a_full[v, v] = np.eye(self.nvirt_alpha)
        #
        eta1b_full = np.zeros((self.norb, self.norb)) 
        eta1b_full[A, A] = eta1b.copy()
        eta1b_full[V, V] = np.eye(self.nvirt_beta)

        # Make 2-body cumulants
        lam2aa = -np.einsum("uw,vx->uvwx", gam1a, gam1a, optimize=True)
        lam2aa -= lam2aa.transpose(1, 0, 2, 3)
        lam2aa += self.rdms['aa']
        #
        lam2ab = -np.einsum("uw,vx->uvwx", gam1a, gam1b, optimize=True)
        lam2ab += self.rdms['ab']
        #
        lam2bb = -np.einsum("uw,vx->uvwx", gam1b, gam1b, optimize=True)
        lam2bb -= lam2bb.transpose(1, 0, 2, 3)
        lam2bb += self.rdms['bb']

        # Make 3-body cumulants
        # l(abcijk) = g(abcijk) - A(a/bc)A(i/jk) g(ai)g(bcjk) + 2*A(ijk) g(ai)g(bj)g(ck)
        lam3aaa = self.rdms['aaa'].copy()
        temp = np.einsum("ai,bcjk->abcijk", self.rdms['a'], self.rdms['aa'], optimize=True)
        temp -= np.transpose(temp, (1, 0, 2, 3, 4, 5)) + np.transpose(temp, (2, 1, 0, 3, 4, 5)) # (a/bc)
        temp -= np.transpose(temp, (0, 1, 2, 4, 3, 5)) + np.transpose(temp, (0, 1, 2, 5, 4, 3)) # (i/jk)
        lam3aaa -= temp
        temp = np.einsum("ai,bj,ck->abcijk", self.rdms['a'], self.rdms['a'], self.rdms['a'], optimize=True)
        temp -= np.transpose(temp, (0, 1, 2, 3, 5, 4)) # (jk)
        temp -= np.transpose(temp, (0, 1, 2, 4, 3, 5)) + np.transpose(temp, (0, 1, 2, 5, 4, 3)) # (i/jk)
        lam3aaa += 2.0 * temp
        # l(abc~ijk~) = g(abc~ijk~) - A(ab)A(ij) g(ai)g(bc~jk~) - g(c~k~)g(abij) + 2*A(ij) g(ai)g(bj)g(c~k~)
        lam3aab = self.rdms['aab'].copy()
        temp = np.einsum("ai,bcjk->abcijk", self.rdms['a'], self.rdms['ab'], optimize=True)
        temp -= np.transpose(temp, (1, 0, 2, 3, 4, 5)) # (ab)
        temp -= np.transpose(temp, (0, 1, 2, 4, 3, 5)) # (ij)
        lam3aab -= temp
        temp = np.einsum("ck,abij->abcijk", self.rdms['b'], self.rdms['aa'], optimize=True)
        lam3aab -= temp
        temp = np.einsum("ai,bj,ck->abcijk", self.rdms['a'], self.rdms['a'], self.rdms['b'], optimize=True)
        temp -= np.transpose(temp, (0, 1, 2, 4, 3, 5)) # (ij)
        lam3aab += 2.0 * temp
        # l(ab~c~ij~k~) = g(ab~c~ij~k~) - g(ai)g(b~c~j~k~) - A(bc)A(jk) g(b~j~)g(ac~ik~) + 2*A(jk) g(ai)g(b~j~)g(c~k~)
        lam3abb = self.rdms['abb'].copy()
        temp = np.einsum("ai,bcjk->abcijk", self.rdms['a'], self.rdms['bb'], optimize=True)
        lam3abb -= temp
        temp = np.einsum("bj,acik->abcijk", self.rdms['b'], self.rdms['ab'], optimize=True)
        temp -= np.transpose(temp, (0, 2, 1, 3, 4, 5)) # (bc)
        temp -= np.transpose(temp, (0, 1, 2, 3, 5, 4)) # (jk)
        lam3abb -= temp
        temp = np.einsum("ai,bj,ck->abcijk", self.rdms['a'], self.rdms['b'], self.rdms['b'], optimize=True)
        temp -= np.transpose(temp, (0, 1, 2, 3, 5, 4)) # (jk)
        lam3abb += 2.0 * temp
        # l(a~b~c~i~j~k~) = g(a~b~c~i~j~k~) - A(a/bc)A(i/jk) g(a~i~)g(b~c~j~k~) + 2*A(ijk) g(a~i~)g(b~j~)g(c~k~)
        lam3bbb = self.rdms['bbb'].copy()
        temp = np.einsum("ai,bcjk->abcijk", self.rdms['b'], self.rdms['bb'], optimize=True)
        temp -= np.transpose(temp, (1, 0, 2, 3, 4, 5)) + np.transpose(temp, (2, 1, 0, 3, 4, 5)) # (a/bc)
        temp -= np.transpose(temp, (0, 1, 2, 4, 3, 5)) + np.transpose(temp, (0, 1, 2, 5, 4, 3)) # (i/jk)
        lam3bbb -= temp
        temp = np.einsum("ai,bj,ck->abcijk", self.rdms['b'], self.rdms['b'], self.rdms['b'], optimize=True)
        temp -= np.transpose(temp, (0, 1, 2, 3, 5, 4)) # (jk)
        temp -= np.transpose(temp, (0, 1, 2, 4, 3, 5)) + np.transpose(temp, (0, 1, 2, 5, 4, 3)) # (i/jk)
        lam3bbb += 2.0 * temp

        self.gam1 = {'a': gam1a, 'b': gam1b}
        self.gam1_full = {'a': gam1a_full, 'b': gam1b_full}
        self.eta1 = {'a': eta1a, 'b': eta1b}
        self.eta1_full = {'a': eta1a_full, 'b': eta1b_full}
        self.lambdas = {'aa': lam2aa, 'ab': lam2ab, 'bb': lam2bb,
                        'aaa': lam3aaa, 'aab': lam3aab, 'abb': lam3abb, 'bbb': lam3bbb}
        
    def make_fock(self):
        c = self.orbspace['core_alpha']
        C = self.orbspace['core_beta']
        a = self.orbspace['active_alpha']
        A = self.orbspace['active_beta']
        
        fock_a = (
                    self.Z['a'] 
                    + np.einsum("piqi->pq", self.V['aa'][:, c, :, c])
                    + np.einsum("puqv,uv->pq", self.V['aa'][:, a, :, a], self.rdms['a'])
                    + np.einsum("piqi->pq", self.V['ab'][:, C, :, C])
                    + np.einsum("puqv,uv->pq", self.V['ab'][:, A, :, A], self.rdms['b'])
        )
        fock_b = (
                    self.Z['b'] 
                    + np.einsum("piqi->pq", self.V['bb'][:, C, :, C])
                    + np.einsum("puqv,uv->pq", self.V['bb'][:, A, :, A], self.rdms['b'])
                    + np.einsum("ipiq->pq", self.V['ab'][c, :, c, :])
                    + np.einsum("upvq,uv->pq", self.V['ab'][a, :, a, :], self.rdms['a'])
        )
        self.F = {'a': fock_a, 'b': fock_b}
        #fock = np.einsum("pi,pq,qj->ij", self.mc.mo_coeff, self.mc.get_fock(), self.mc.mo_coeff)
        #self.F = {'a': fock, 'b': fock}

    def compute_cas_energy(self):
        c = self.orbspace['core_alpha']
        C = self.orbspace['core_beta']
        a = self.orbspace['active_alpha']
        A = self.orbspace['active_beta']
        e_test = (
            np.einsum("mm->", self.Z['a'][c, c]) 
          + np.einsum("mm->", self.Z['b'][C, C])
          + np.einsum("uv,uv->", self.Z['a'][a, a], self.rdms['a'])
          + np.einsum("uv,uv->", self.Z['b'][A, A], self.rdms['b'])
          + 0.5 * np.einsum("mnmn->", self.V['aa'][c, c, c, c])
          + np.einsum("mnmn->", self.V['ab'][c, C, c, C])
          + 0.5 * np.einsum("mnmn->", self.V['bb'][C, C, C, C])
          + np.einsum("mumv,uv->", self.V['aa'][c, a, c, a], self.rdms['a'])
          + np.einsum("umvm,uv->", self.V['ab'][a, C, a, C], self.rdms['a'])
          + np.einsum("mumv,uv->", self.V['ab'][c, A, c, A], self.rdms['b'])
          + np.einsum("mumv,uv->", self.V['bb'][C, A, C, A], self.rdms['b'])
          + 0.25 * np.einsum("uvxy,uvxy->", self.V['aa'][a, a, a, a], self.rdms['aa'])
          + np.einsum("uvxy,uvxy->", self.V['ab'][a, A, a, A], self.rdms['ab'])
          + 0.25 * np.einsum("uvxy,uvxy->", self.V['bb'][A, A, A, A], self.rdms['bb'])
          + self.nuclear_repulsion
        )
        self.e_cas = e_test

    def compute_cas_energy_from_fock(self):
        c = self.orbspace['core_alpha']
        C = self.orbspace['core_beta']
        a = self.orbspace['active_alpha']
        A = self.orbspace['active_beta']
        e_test = (
            np.einsum("mm->", self.F['a'][c, c]) 
          + np.einsum("mm->", self.F['b'][C, C])
          + np.einsum("uv,uv->", self.F['a'][a, a], self.rdms['a'])
          + np.einsum("uv,uv->", self.F['b'][A, A], self.rdms['b'])
          - 0.5 * np.einsum("mnmn->", self.V['aa'][c, c, c, c])
          - np.einsum("mnmn->", self.V['ab'][c, C, c, C])
          - 0.5 * np.einsum("mnmn->", self.V['bb'][C, C, C, C])
          - np.einsum("mumv,uv->", self.V['aa'][c, a, c, a], self.rdms['a'])
          - np.einsum("umvm,uv->", self.V['ab'][a, C, a, C], self.rdms['a'])
          - np.einsum("mumv,uv->", self.V['ab'][c, A, c, A], self.rdms['b'])
          - np.einsum("mumv,uv->", self.V['bb'][C, A, C, A], self.rdms['b'])
          - 0.5 * np.einsum("xuyv,xy,uv->", self.V['aa'][a, a, a, a], self.rdms['a'], self.rdms['a'])
          - np.einsum("xuyv,xy,uv->", self.V['ab'][a, A, a, A], self.rdms['a'], self.rdms['b'])
          - 0.5 * np.einsum("xuyv,xy,uv->", self.V['bb'][A, A, A, A], self.rdms['b'], self.rdms['b'])
          + 0.25 * np.einsum("uvxy,uvxy->", self.V['aa'][a, a, a, a], self.lambdas['aa'])
          + np.einsum("uvxy,uvxy->", self.V['ab'][a, A, a, A], self.lambdas['ab'])
          + 0.25 * np.einsum("uvxy,uvxy->", self.V['bb'][A, A, A, A], self.lambdas['bb'])
          + self.nuclear_repulsion
        )
        self.e_cas_from_fock = e_test

    def freeze_orbitals(self):
        self.nelectron -= 2*self.nfrozen
        self.norb -= self.nfrozen
        self.ncore_alpha -= self.nfrozen 
        self.ncore_beta -= self.nfrozen 
        #
        self.nhole_alpha = self.ncore_alpha + self.nact_alpha
        self.nhole_beta = self.ncore_beta + self.nact_beta
        #
        self.orbspace = {
            'core_alpha': slice(0, self.ncore_alpha),
            'active_alpha': slice(self.ncore_alpha, self.ncore_alpha + self.nact_alpha),
            'virt_alpha': slice(self.ncore_alpha + self.nact_alpha, self.norb),
            'core_beta': slice(0, self.ncore_beta),
            'active_beta': slice(self.ncore_beta, self.ncore_beta + self.nact_beta),
            'virt_beta': slice(self.ncore_beta + self.nact_beta, self.norb),
            'hole_core_alpha': slice(0, self.ncore_alpha),
            'hole_active_alpha': slice(self.ncore_alpha, self.ncore_alpha + self.nact_alpha),
            'particle_active_alpha': slice(0, self.nact_alpha),
            'particle_virt_alpha': slice(self.nact_alpha, self.nact_alpha + self.nvirt_alpha),
            'hole_core_beta': slice(0, self.ncore_beta),
            'hole_active_beta': slice(self.ncore_beta, self.ncore_beta + self.nact_beta),
            'particle_active_beta': slice(0, self.nact_beta),
            'particle_virt_beta': slice(self.nact_beta, self.nact_beta + self.nvirt_beta),
            'hole_alpha': slice(0, self.ncore_alpha + self.nact_alpha),
            'particle_alpha': slice(self.ncore_alpha, self.norb),
            'hole_beta': slice(0, self.ncore_beta + self.nact_beta),
            'particle_beta': slice(self.ncore_beta, self.norb),
        }
        #
        corr = slice(self.nfrozen, self.norb + self.nfrozen)
        self.F['a'] = self.F['a'][corr, corr]
        self.F['b'] = self.F['b'][corr, corr]
        self.V['aa'] = self.V['aa'][corr, corr, corr, corr]
        self.V['ab'] = self.V['ab'][corr, corr, corr, corr]
        self.V['bb'] = self.V['bb'][corr, corr, corr, corr]

class SpinReference:
    
    def __init__(self, mc, mf, verbose=False):
        self.mc = mc
        self.mf = mf
        self.nelectron = mc.mol.nelectron
        self.norb = 2 * mc.mo_coeff.shape[1]
        self.nuclear_repulsion = self.mf.mol.energy_nuc()
        self.nelecas = sum(self.mc.nelecas)
        self.nelcas_alpha, self.nelcas_beta = self.mc.nelecas
        self.cas = (self.nelecas, 2*self.mc.ncas)
        self.ncore = self.nelectron - self.nelecas 
        self.nact = self.cas[1]
        self.nvirt = self.norb - self.nact - self.ncore
        #
        self.orbspace = {
            'core': slice(0, self.ncore),
            'active': slice(self.ncore, self.ncore + self.nact),
            'virt': slice(self.ncore + self.nact, self.norb),
            'hole_core': slice(0, self.ncore),
            'hole_active': slice(self.ncore, self.ncore + self.nact),
            'particle_active': slice(0, self.nact),
            'particle_virt': slice(self.nact, self.nact + self.nvirt),
            'hole': slice(0, self.ncore + self.nact),
            #'particle': slice(0, self.nact + self.nvirt),
            'particle': slice(self.ncore, self.norb),
        }
        #
        self.verbose = verbose

    def kernel(self, semi=True):
        if self.verbose: print("Spinorbital CAS Reference")
        if self.verbose: print(f"Semicanonicalization = {semi}")
        # first make the hcore/V integrals in HF basis
        tic = time.time()
        self.make_hf_integrals()
        toc = time.time()
        if self.verbose: print(f"HF integral construction... {toc - tic}s")
        # get RDMs from CAS wave function
        tic = time.time()
        self.make_rdms()
        toc = time.time()
        if self.verbose: print(f"RDM construction... {toc - tic}s")
        # make generalized Fock operator
        tic = time.time()
        self.make_fock()
        toc = time.time()
        if self.verbose: print(f"Fock construction... {toc - tic}s")
        # semi-canonicalize integrals and RDMs
        if semi:
            tic = time.time()
            self.get_semicanonicalizer()
            self.semicanonicalize()
            toc = time.time()
            if self.verbose: print(f"Semicanonicalize integrals and RDMs... {toc - tic}s")
        # make cumulants from RDMs
        tic = time.time()
        self.make_cumulants()
        toc = time.time()
        if self.verbose: print(f"Make cumulants... {toc - tic}s")
        # check that 1- and 2-RDMs in semicanonical basis is correct by computing CAS energy
        self.compute_cas_energy()
        #
        self.compute_cas_energy_from_fock()
        if self.verbose:
            print("CAS (from RDMs) = ", self.e_cas)
            print("CAS (from RDMs, Fock) = ", self.e_cas_from_fock)
            print("Expected CAS = ", self.mc.e_tot)
        try:
            assert np.allclose(self.e_cas, self.mc.e_tot)
            assert np.allclose(self.e_cas_from_fock, self.mc.e_tot)
            if self.verbose: print("All is well!")
        except AssertionError:
            print("CAS energy computed via RDMs does not match!")
        
    def make_rdms(self):
        self.rdms = make_casci_rdm123(self.mc, self.cas[1], self.nelcas_alpha, self.nelcas_beta)

    def make_cumulants(self):

        # Make 1-body cumulants
        gam1 = self.rdms['1'].copy()
        eta1 = np.eye(gam1.shape[0]) - gam1

        # Make full 1-body cumulants (just in case)
        c = self.orbspace['core']
        a = self.orbspace['active']
        v = self.orbspace['virt']
        #
        gam1_full = np.zeros((self.norb, self.norb))
        gam1_full[c, c] = np.eye(self.ncore)
        gam1_full[a, a] = gam1.copy()
        #
        eta1_full = np.zeros((self.norb, self.norb))
        eta1_full[a, a] = eta1.copy()
        eta1_full[v, v] = np.eye(self.nvirt)

        # Make 2-body cumulants
        lam2 = -np.einsum("uw,vx->uvwx", gam1, gam1, optimize=True)
        lam2 -= lam2.transpose(1, 0, 2, 3)
        lam2 += self.rdms['2']

        # Make 3-body cumulants
        # l(abcijk) = g(abcijk) - A(a/bc)A(i/jk) g(ai)g(bcjk) + 2*A(ijk) g(ai)g(bj)g(ck)
        lam3 = self.rdms['3'].copy()
        temp = np.einsum("ai,bcjk->abcijk", self.rdms['1'], self.rdms['2'], optimize=True)
        temp -= np.transpose(temp, (1, 0, 2, 3, 4, 5)) + np.transpose(temp, (2, 1, 0, 3, 4, 5)) # (a/bc)
        temp -= np.transpose(temp, (0, 1, 2, 4, 3, 5)) + np.transpose(temp, (0, 1, 2, 5, 4, 3)) # (i/jk)
        lam3 -= temp
        temp = np.einsum("ai,bj,ck->abcijk", self.rdms['1'], self.rdms['1'], self.rdms['1'], optimize=True)
        temp -= np.transpose(temp, (0, 1, 2, 3, 5, 4)) # (jk)
        temp -= np.transpose(temp, (0, 1, 2, 4, 3, 5)) + np.transpose(temp, (0, 1, 2, 5, 4, 3)) # (i/jk)
        lam3 += 2.0 * temp

        self.gam1 = gam1
        self.eta1 = eta1
        self.gam1_full = gam1_full
        self.eta1_full = eta1_full
        self.lambdas = {'2': lam2, '3': lam3}
        print("|lam2| = ", np.linalg.norm(lam2.flatten()))
        print("|lam3| = ", np.linalg.norm(lam3.flatten()))
        
    def make_hf_integrals(self):
        mo_coeff = self.mf.mo_coeff

        # Note: You cannot replace this the T + V construction with mf.get_hcore() when using
        # a CASCI calculation in conjunction with mf!
        hcore_ao = self.mf.mol.intor_symmetric('int1e_kin') + self.mf.mol.intor_symmetric('int1e_nuc')
        hcore = np.einsum('pi,pq,qj->ij', mo_coeff, hcore_ao, mo_coeff)
        Z = np.zeros((self.norb, self.norb))
        Z[::2, ::2] = Z[1::2, 1::2] = hcore
        
        eri = self.mf.mol.intor("int2e_sph", aosym="s1").transpose(0, 2, 1, 3)
        eri = np.einsum("ijkl,ip,jq,kr,ls->pqrs", eri, mo_coeff, mo_coeff, mo_coeff, mo_coeff, optimize=True)
        eri = eri
        V = np.zeros((self.norb, self.norb, self.norb, self.norb))
        V[::2, ::2, ::2, ::2] = V[1::2, 1::2, 1::2, 1::2] = eri - eri.transpose(0, 1, 3, 2) 
        V[::2, 1::2, ::2, 1::2] = V[1::2, ::2, 1::2, ::2] = eri 
        V[::2, 1::2, 1::2, ::2] = V[1::2, ::2, ::2, 1::2] = -eri.transpose(0, 1, 3, 2) 

        # Store 1- and 2-electron integrals 
        self.Z = Z
        self.V = V 
        
    def make_fock(self):
        c = self.orbspace['core']
        a = self.orbspace['active']
        
        fock = (
                    self.Z 
                    + np.einsum("piqi->pq", self.V[:, c, :, c])
                    + np.einsum("puqv,uv->pq", self.V[:, a, :, a], self.rdms['1'])
        )
        self.F = fock 
        
    def get_semicanonicalizer(self):

        def _diagonalize_and_reorder(f):
            #e, u = np.linalg.eigh(f)
            #isort = np.argsort(e)
            #return u[:, isort]
            e, u = np.linalg.eigh(f)
            return u 

        # diagonalize fock_a and fock_b in cc, aa, and vv sectors to get semicanonicalizer
        c = self.orbspace['core']
        a = self.orbspace['active']
        v = self.orbspace['virt']

        self.U = np.zeros_like(self.F) 
        self.U[c, c] = _diagonalize_and_reorder(self.F[c, c])
        self.U[a, a] = _diagonalize_and_reorder(self.F[a, a])
        self.U[v, v] = _diagonalize_and_reorder(self.F[v, v])
    
    def semicanonicalize(self):
        def _rotate_1(U, F):
            return np.einsum("ij,ip,jq->pq", F, np.conj(U), U, optimize=True)
        def _rotate_2(U, V):
            return np.einsum("ijkl,ip,jq,kr,ls->pqrs", V, np.conj(U), np.conj(U), U, U, optimize=True)
        # semi-canonicalize 1- and 2-body integrals
        self.Z = _rotate_1(self.U, self.Z)
        self.F = _rotate_1(self.U, self.F)
        self.V = _rotate_2(self.U, self.V)
        # semi-canonicalize 1-, 2-, and 3-body RDMs
        a = self.orbspace['active']
        self.rdms['1'] = _rotate_1(self.U[a, a], self.rdms['1'])
        self.rdms['2'] = _rotate_2(self.U[a, a], self.rdms['2'])

    def compute_cas_energy(self):
        c = self.orbspace['core']
        a = self.orbspace['active']
        e_test = (
            np.einsum("mm->", self.Z[c, c]) 
          + np.einsum("uv,uv->", self.Z[a, a], self.rdms['1'])
          + 0.5 * np.einsum("mnmn->", self.V[c, c, c, c])
          + np.einsum("mumv,uv->", self.V[c, a, c, a], self.rdms['1'])
          + 0.25 * np.einsum("uvxy,uvxy->", self.V[a, a, a, a], self.rdms['2'])
          + self.nuclear_repulsion
        )
        self.e_cas = e_test

    def compute_cas_energy_from_fock(self):
        c = self.orbspace['core']
        a = self.orbspace['active']
        e_test = (
            np.einsum("mm->", self.F[c, c]) 
          + np.einsum("uv,uv->", self.F[a, a], self.rdms['1'])
          - 0.5 * np.einsum("mnmn->", self.V[c, c, c, c])
          - np.einsum("mumv,uv->", self.V[c, a, c, a], self.rdms['1'])
          - 0.5 * np.einsum("xuyv,xy,uv->", self.V[a, a, a, a], self.rdms['1'], self.rdms['1'])
          + 0.25 * np.einsum("uvxy,uvxy->", self.V[a, a, a, a], self.lambdas['2'])
          + self.nuclear_repulsion
        )
        self.e_cas_from_fock = e_test
