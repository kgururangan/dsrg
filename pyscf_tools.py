import numpy as np
from pyscf import gto, scf, mcscf, fci

def make_casci_rdm12s(mc, cas):

    (dm1a, dm1b), (dm2aa, dm2ab, dm2bb) = fci.direct_uhf.make_rdm12s(mc.ci, cas[1], cas[0], reorder=True)
    # [WARNING]: 2-RDMs from PySCF must be transposed from Chemist to Physics notation
    dm2aa = dm2aa.transpose(0, 2, 1, 3)
    dm2ab = dm2ab.transpose(0, 2, 1, 3)
    dm2bb = dm2bb.transpose(0, 2, 1, 3)


    rdms = {'a': dm1a, 'b': dm1b, 'aa': dm2aa, 'ab': dm2ab, 'bb': dm2bb}

    return rdms

def make_casci_rdm12(mc, cas):

    (dm1a, dm1b), (dm2aa, dm2ab, dm2bb) = fci.direct_uhf.make_rdm12s(mc.ci, cas[1]//2, cas[0], reorder=True)
    # [WARNING]: 2-RDMs from PySCF must be transposed from Chemist to Physics notation
    dm2aa = dm2aa.transpose(0, 2, 1, 3)
    dm2ab = dm2ab.transpose(0, 2, 1, 3)
    dm2bb = dm2bb.transpose(0, 2, 1, 3)

    dm1 = np.zeros((cas[1], cas[1]))
    dm2 = np.zeros((cas[1], cas[1], cas[1], cas[1]))

    dm1[::2, ::2] = dm1a
    dm1[1::2, 1::2] = dm1b

    dm2[::2, ::2, ::2, ::2] = dm2aa
    dm2[1::2, 1::2, 1::2, 1::2] = dm2bb
    dm2[::2, 1::2, ::2, 1::2] = dm2ab
    dm2[1::2, ::2, 1::2, ::2] = dm2ab.transpose(1, 0, 3, 2)
    dm2[::2, 1::2, 1::2, ::2] = -dm2ab.transpose(0, 1, 3, 2)
    dm2[1::2, ::2, ::2, 1::2] = -dm2ab.transpose(1, 0, 2, 3)

    #print(cas)
    #dm1, dm2 = fci.fci_dhf_slow.make_rdm12(mc.ci[0], cas[1], cas[0], reorder=True)
    # [WARNING]: 2-RDMs from PySCF must be transposed from Chemist to Physics notation
    #dm2 = dm2.transpose(0, 2, 1, 3)
    
    rdms = {'1': dm1, '2': dm2}
    return rdms
