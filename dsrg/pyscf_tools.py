import numpy as np
from pyscf import gto, scf, mcscf, fci

def make_casci_rdm123s(mc, norb_cas, nelcas_a, nelcas_b):

    #(dm1a, dm1b), (dm2aa, dm2ab, dm2bb) = fci.direct_uhf.make_rdm12s(mc.ci, cas[1], cas[0], reorder=True)
    (dm1a, dm1b), (dm2aa, dm2ab, dm2bb), (dm3aaa, dm3aab, dm3abb, dm3bbb) = fci.direct_spin1.make_rdm123s(mc.ci, norb_cas, (nelcas_a, nelcas_b), reorder=True)
    # [WARNING]: 2-RDMs and 3-RDMs from PySCF must be transposed from Chemist to Physics notation
    dm2aa = dm2aa.transpose(0, 2, 1, 3)
    dm2ab = dm2ab.transpose(0, 2, 1, 3)
    dm2bb = dm2bb.transpose(0, 2, 1, 3)
    dm3aaa = dm3aaa.transpose(0, 2, 4, 1, 3, 5)
    dm3aab = dm3aab.transpose(0, 2, 4, 1, 3, 5)
    dm3abb = dm3abb.transpose(0, 2, 4, 1, 3, 5)
    dm3bbb = dm3bbb.transpose(0, 2, 4, 1, 3, 5)

    rdms = {'a': dm1a, 'b': dm1b, 
            'aa': dm2aa, 'ab': dm2ab, 'bb': dm2bb,
            'aaa': dm3aaa, 'aab': dm3aab, 'abb': dm3abb, 'bbb': dm3bbb}

    return rdms

def make_casci_rdm123(mc, norb_cas, nelcas_a, nelcas_b):

    #(dm1a, dm1b), (dm2aa, dm2ab, dm2bb) = fci.direct_uhf.make_rdm12s(mc.ci, cas[1]//2, cas[0], reorder=True)
    (dm1a, dm1b), (dm2aa, dm2ab, dm2bb), (dm3aaa, dm3aab, dm3abb, dm3bbb) = fci.direct_spin1.make_rdm123s(mc.ci, norb_cas//2, (nelcas_a, nelcas_b), reorder=True)
    # [WARNING]: 2-RDMs and 3-RDMs from PySCF must be transposed from Chemist to Physics notation
    dm2aa = dm2aa.transpose(0, 2, 1, 3)
    dm2ab = dm2ab.transpose(0, 2, 1, 3)
    dm2bb = dm2bb.transpose(0, 2, 1, 3)
    dm3aaa = dm3aaa.transpose(0, 2, 4, 1, 3, 5)
    dm3aab = dm3aab.transpose(0, 2, 4, 1, 3, 5)
    dm3abb = dm3abb.transpose(0, 2, 4, 1, 3, 5)
    dm3bbb = dm3bbb.transpose(0, 2, 4, 1, 3, 5)

    dm1 = np.zeros((norb_cas, norb_cas))
    dm2 = np.zeros((norb_cas, norb_cas, norb_cas, norb_cas))
    dm3 = np.zeros((norb_cas, norb_cas, norb_cas, norb_cas, norb_cas, norb_cas))

    dm1[::2, ::2] = dm1a
    dm1[1::2, 1::2] = dm1b

    dm2[::2, ::2, ::2, ::2] = dm2aa
    dm2[1::2, 1::2, 1::2, 1::2] = dm2bb
    dm2[::2, 1::2, ::2, 1::2] = dm2ab
    dm2[1::2, ::2, 1::2, ::2] = dm2ab.transpose(1, 0, 3, 2)
    dm2[::2, 1::2, 1::2, ::2] = -dm2ab.transpose(0, 1, 3, 2)
    dm2[1::2, ::2, ::2, 1::2] = -dm2ab.transpose(1, 0, 2, 3)

    dm3[::2, ::2, ::2, ::2, ::2, ::2] = dm3aaa
    dm3[1::2, 1::2, 1::2, 1::2, 1::2, 1::2] = dm3bbb

    dm3[::2, ::2, 1::2, ::2, ::2, 1::2] = dm3aab
    dm3[::2, 1::2, ::2, ::2, ::2, 1::2] = -dm3aab.transpose(0, 2, 1, 3, 4, 5)
    dm3[1::2, ::2, ::2, ::2, ::2, 1::2] = dm3aab.transpose(2, 0, 1, 3, 4, 5)

    dm3[::2, ::2, 1::2, ::2, 1::2, ::2] = -dm3aab.transpose(0, 1, 2, 3, 5, 4)
    dm3[::2, 1::2, ::2, ::2, 1::2, ::2] = dm3aab.transpose(0, 2, 1, 3, 5, 4)
    dm3[1::2, ::2, ::2, ::2, 1::2, ::2] = -dm3aab.transpose(2, 0, 1, 3, 5, 4)

    dm3[::2, ::2, 1::2, 1::2, ::2, ::2] = dm3aab.transpose(0, 1, 2, 5, 3, 4)
    dm3[::2, 1::2, ::2, 1::2, ::2, ::2] = -dm3aab.transpose(0, 2, 1, 5, 3, 4)
    dm3[1::2, ::2, ::2, 1::2, ::2, ::2] = dm3aab.transpose(2, 0, 1, 5, 3, 4)

    dm3[::2, 1::2, 1::2, ::2, 1::2, 1::2] = dm3abb
    dm3[1::2, ::2, 1::2, ::2, 1::2, 1::2] = -dm3abb.transpose(1, 0, 2, 3, 4, 5)
    dm3[1::2, 1::2, ::2, ::2, 1::2, 1::2] = dm3abb.transpose(1, 2, 0, 3, 4, 5)

    dm3[::2, 1::2, 1::2, 1::2, ::2, 1::2] = -dm3abb.transpose(0, 1, 2, 4, 3, 5)
    dm3[1::2, ::2, 1::2, 1::2, ::2, 1::2] = dm3abb.transpose(1, 0, 2, 4, 3, 5)
    dm3[1::2, 1::2, ::2, 1::2, ::2, 1::2] = -dm3abb.transpose(1, 2, 0, 4, 3, 5)

    dm3[::2, 1::2, 1::2, 1::2, 1::2, ::2] = dm3abb.transpose(0, 1, 2, 4, 5, 3)
    dm3[1::2, ::2, 1::2, 1::2, 1::2, ::2] = -dm3abb.transpose(1, 0, 2, 4, 5, 3)
    dm3[1::2, 1::2, ::2, 1::2, 1::2, ::2] = dm3abb.transpose(1, 2, 0, 4, 5, 3)

    print("|dm3| = ", np.linalg.norm(dm3.flatten()))
    
    rdms = {'1': dm1, '2': dm2, '3': dm3}
    return rdms
