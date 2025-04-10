import sys, os
# Get the current file's directory
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory
parent_dir = os.path.dirname(current_dir)
# Add the parent directory to sys.path
sys.path.append(parent_dir)

import numpy as np
from pyscf import gto, scf, mcscf
from reference import Reference
from dsrg import DSRG

RTOL = 1.0e-06

def test_mrldsrg2_h2o():

    mol = gto.M(
        atom = '''
            O  0.0000  0.0000  0.1173
            H  0.0000  0.7572 -0.4692
            H  0.0000 -0.7572 -0.4692''',
        basis = '6-31g',
        spin=0,
        charge=0,
        symmetry="C2V",
    )
    mol.verbose = 4

    # Perform RHF calculation
    mf = scf.RHF(mol).run()

    # Perform CASSCF calculation
    mc = mcscf.CASSCF(mf, 4, 4) # 4 electrons in 4 orbitals
    mc.kernel()

    # Create the reference
    # Pass in MO coefficients from CASSCF to use those orbitals in AO-to-MO transformation
    ref = Reference(mc, mf, mo_coeff=mc.mo_coeff, nfrozen=0, verbose=True)
    ref.kernel(semi=True)

    # Run DSRG
    driver = DSRG(ref)
    driver.run_ldsrg2(s=0.5)

    #
    # Check the results
    # Source: MR-LDSRG(2) Results from Forte 
    # (includes 3-cumulant in energy)
    #
    assert np.isclose(driver.ref.e_cas, -75.999851588600450, rtol=RTOL)
    assert np.isclose(driver.e_dsrg2 + driver.ref.e_cas, -76.118891492221735, rtol=RTOL)

if __name__ == "__main__":
    test_mrldsrg2_h2o()

