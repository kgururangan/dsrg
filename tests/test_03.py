import numpy as np
from pyscf import gto, scf, mcscf
from dsrg.reference import Reference
from dsrg.dsrg_si import DSRG

RTOL = 1.0e-06
ATOL = 1.0e-06

def test_mrdsrg_ldsrg2_nh_1():

    mol = gto.M(
        atom = '''
            N  0.0000  0.0000  0.000
            N  0.0000  0.0000  1.251''',
        basis = 'cc-pvdz',
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
    mc = mcscf.CASSCF(mf, 6, 6)
    ncore = {'Ag': 2, 'B1u': 2}
    ncas = {'Ag': 1, 'B2g': 1, 'B3g': 1, 'B1u': 1, 'B2u': 1, 'B3u': 1}
    mo = mcscf.sort_mo_by_irrep(mc, mf.mo_coeff, ncas, ncore)
    mc.kernel(mo)

    # Create the reference
    # Pass in MO coefficients from CASSCF to use those orbitals in AO-to-MO transformation
    ref = Reference(mc, mf, mo_coeff=mc.mo_coeff, nfrozen=0, verbose=True)
    ref.kernel(semi=True)

    # Run DSRG
    driver = DSRG(ref)
    driver.run_ldsrg2(s=0.5, herm=False)

    #
    # Check the results
    # Source: MR-LDSRG(2) Results from Forte 
    # (includes 3-cumulant in energy)
    #
    assert np.isclose(driver.ref.e_cas, -109.055262183808921, rtol=RTOL, atol=ATOL)
    assert np.isclose(driver.e_dsrg2, -0.195362408475867, rtol=RTOL, atol=ATOL)
    assert np.isclose(driver.e_dsrg2 + driver.ref.e_cas, -109.250624592284794, rtol=RTOL, atol=ATOL)

if __name__ == "__main__":
    test_mrdsrg_ldsrg_nh_1()

