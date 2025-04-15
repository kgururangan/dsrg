import numpy as np
from pyscf import gto, scf, mcscf, fci
from dsrg.reference import SpinReference
from dsrg.dsrg_spinorb import SpinDSRG

VERBOSE = 0
RTOL = 1.0e-06
ATOL = 1.0e-06

if __name__ == "__main__":

    mol = gto.M(
        atom = '''
            N  0.0000  0.0000  0.000
            N  0.0000  0.0000  1.100''',
        basis = '6-31g',
        spin=0,
        charge=0,
        symmetry="D2H",
    )
    mol.verbose = 4

    # Perform RHF calculation
    mf = scf.RHF(mol).run()
    # Ag B1g, B2g, B3g, Au, B1u, B2u, B3u
    # restricted_docc         [2,0,0,0,0,2,0,0]
    # active                  [1,0,1,1,0,1,1,1]
    # Perform CASSCF calculation
    mc = mcscf.CASCI(mf, 6, 6)
    ncore = {'Ag': 2, 'B1u': 2}
    ncas = {'Ag': 1, 'B2g': 1, 'B3g': 1, 'B1u': 1, 'B2u': 1, 'B3u': 1}
    mo = mcscf.sort_mo_by_irrep(mc, mf.mo_coeff, ncas, ncore)
    mc.kernel(mo)

    ref = SpinReference(mc, mf, verbose=True)
    ref.kernel(semi=True)

    driver = SpinDSRG(ref)
    driver.run_ldsrg2(s=1.0, herm=True)

    #
    # Check the results
    # Source: MR-LDSRG(2) Results from Forte 
    # (includes 3-cumulant in energy)
    #
    assert np.isclose(driver.ref.e_cas, -108.947010693494292, rtol=RTOL, atol=ATOL)
    assert np.isclose(driver.e_dsrg2, -0.153866605751881, rtol=RTOL, atol=ATOL)
    assert np.isclose(driver.e_dsrg2 + driver.ref.e_cas, -109.100877299246179, rtol=RTOL, atol=ATOL)
