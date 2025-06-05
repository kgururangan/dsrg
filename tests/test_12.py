import numpy as np
from pyscf import gto, scf, mcscf, fci
from dsrg.reference import Reference
from dsrg.driver import DSRG

RTOL = 1.0e-06
ATOL = 1.0e-06

def test_mrdsrg_ldsrg2_hf():

    mol = gto.M(atom='''H 0 0 0; F 0 0 1.5''', 
                spin=0, 
                basis='cc-pvdz', 
                unit='angstrom', 
                verbose=4, 
                symmetry="C2v"
    )
    mf = scf.RHF(mol).run()
    scf.rhf_symm.analyze(mf)

    # Run CASCI
    mc = mcscf.CASSCF(mf, 2, 2)
    ncore = {'A1': 2, 'A2': 0, 'B1': 1, 'B2': 1}
    ncas = {'A1': 2, 'A2': 0, 'B1': 0, 'B2': 0}
    mo = mcscf.sort_mo_by_irrep(mc, mf.mo_coeff, ncas, ncore)
    mc.kernel(mo)
    # Print out CASSCF orbitals
    mc.analyze()

    ref = Reference.from_pyscf(mc, mf, nfrozen=0)

    driver = DSRG(ref)
    driver.run_dsrg(method='ldsrg2', s=1.0, herm=True)
    driver.diagonalize_hbar(herm=True)

    #
    # Check the results
    #
    assert np.isclose(driver.reference_energy, -99.939316382624, rtol=RTOL, atol=ATOL)
    assert np.isclose(driver.total_energy, -100.112784378794, rtol=RTOL, atol=ATOL)
    assert np.isclose(driver.total_energy_relaxed[0], -100.115962352711, rtol=RTOL, atol=ATOL)


if __name__ == "__main__":
    test_mrdsrg_ldsrg2_hf()
