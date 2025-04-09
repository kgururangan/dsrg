import numpy as np
from pyscf import gto, scf, mcscf, fci
from reference import SpinReference
from correlation import SpinDSRG

VERBOSE = 0

if __name__ == "__main__":

    mol = gto.M(atom='''H 0 0 0; F 0 0 1.5''', 
                spin=0, 
                basis='6-31g', 
                unit='angstrom', 
                verbose=VERBOSE, 
                symmetry="C2v"
    )
    mf = scf.RHF(mol).run()

    # Total orbital space
    nels = mol.nelectron
    norb = mf.mo_coeff.shape[1]

    # Set up active space
    nelcas_a = 3
    nelcas_b = 3
    nact = 4

    nelcas = nelcas_a + nelcas_b
    ncore = (nels - nelcas)//2
    nvirt = norb - nact - ncore

    cas = (nelcas, nact)

    # Run CASCI
    mycas = mcscf.CASCI(mf, cas[1], cas[0])
    mycas.run()

    ref = SpinReference(mycas, mf, verbose=True)
    ref.kernel(semi=True)

    driver = SpinDSRG(ref)
    driver.run_ldsrg2(s=2.0, herm=False)

