import numpy as np
from pyscf import gto, scf, mcscf
from dsrg.reference import Reference
from dsrg.driver import RICMRCC

RTOL = 1.0e-06
ATOL = 1.0e-06

def test_ricmrccsd_1():

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
    scf.rhf_symm.analyze(mf)
    # Ag B1g, B2g, B3g, Au, B1u, B2u, B3u
    # restricted_docc         [2,0,0,0,0,2,0,0]
    # active                  [1,0,1,1,0,1,1,1]
    # Perform CASCI calculation
    mc = mcscf.CASSCF(mf, 6, 6)
    ncore = {'Ag': 2, 'B1u': 2}
    ncas = {'Ag': 1, 'B2g': 1, 'B3g': 1, 'B1u': 1, 'B2u': 1, 'B3u': 1}
    mo = mcscf.sort_mo_by_irrep(mc, mf.mo_coeff, ncas, ncore)
    mc.kernel(mo)
    mc.analyze()

    # Create the reference
    ref = Reference(mc, mf, mo_coeff=mc.mo_coeff, nfrozen=0, verbose=True)
    ref.kernel(semi=True)

    # Run DSRG
    driver = RICMRCC(ref)
    driver.run_ricmrcc(method='ricmrccsd', s=1.0, herm=False)
    
    #
    # Check the results
    # Source: ric-MRCCSD Results from Robin's code (ricmrcc-publish) 
    # (neglects 4-cumulant everywhere)
    #
    assert np.isclose(driver.reference_energy, -109.015943955217224, rtol=RTOL, atol=ATOL)
    assert np.isclose(driver.correlation_energy, -0.091885948941, rtol=RTOL, atol=ATOL)
    assert np.isclose(driver.total_energy, -109.10782990415822, rtol=RTOL, atol=ATOL)


if __name__ == "__main__":
    test_ricmrccsd_1()

