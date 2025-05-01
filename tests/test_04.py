import numpy as np
from pyscf import gto, scf, mcscf, fci
from dsrg.reference import Reference
from dsrg.driver import DSRG

VERBOSE = 0
RTOL = 1.0e-06
ATOL = 1.0e-06

def test_mrdsrg_ldsrg2_hf():

    mol = gto.M(atom='''H 0 0 0; F 0 0 1.5''', 
                spin=0, 
                basis='6-31g', 
                unit='angstrom', 
                verbose=VERBOSE, 
                symmetry="C2v"
    )
    mf = scf.RHF(mol).run()
    scf.rhf_symm.analyze(mf)

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
    mycas.analyze()

    ref = Reference(mycas, mf, verbose=True)
    ref.kernel(semi=True)

    driver = DSRG(ref)
    driver.run_dsrg(method='ldsrg2', s=2.0, herm=True)

    #
    # Check the results
    # Source: MR-LDSRG(2) Results from 4c-DSRG-MRPT (Brian's code)
    # (includes 3-cumulant in energy)
    #
    assert np.isclose(driver.reference_energy, -99.9015526, rtol=RTOL, atol=ATOL)
    assert np.isclose(driver.correlation_energy, -0.124514912932, rtol=RTOL, atol=ATOL)
    assert np.isclose(driver.total_energy, -100.0260675, rtol=RTOL, atol=ATOL)

if __name__ == "__main__":
    test_mrdsrg_ldsrg2_hf()
