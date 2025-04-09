import sys, os
# Get the current file's directory
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory
parent_dir = os.path.dirname(current_dir)
# Add the parent directory to sys.path
sys.path.append(parent_dir)

import numpy as np
from pyscf import gto, scf, mcscf, fci
from reference import SpinReference
from dsrg_spinorb import SpinDSRG

VERBOSE = 0
RTOL = 1.0e-07

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
    driver.run_ldsrg2(s=2.0, herm=True)

    print("   MR-LDSRG(2) correlation energy should be: -0.124561555381")

    #
    # Check the results
    # Source: MR-LDSRG(2) Results from 4c-DSRG-MRPT (Brian's code)
    # (excludes 3-cumulant in energy)
    #
    assert np.isclose(ref.e_cas, -99.9015526, rtol=RTOL)
    assert np.isclose(driver.e_dsrg2 + ref.e_cas, -100.026114187584, rtol=RTOL)
