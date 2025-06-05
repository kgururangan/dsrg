import re
import numpy as np
from pyscf import fci
from pyscf import symm

def make_hf_integrals(mf):
    # Note: You cannot replace this the T + V construction with mf.get_hcore() when using
    # a CASCI calculation in conjunction with mf!
    e1int_ao = mf.mol.intor_symmetric('int1e_kin') + mf.mol.intor_symmetric('int1e_nuc')    
    e2int_ao = mf.mol.intor("int2e_sph", aosym="s1").transpose(0, 2, 1, 3)
    return e1int_ao, e2int_ao

def get_pyscf_orbsym(molecule, mo_coeff):
    return [x.upper() for x in symm.label_orb_symm(molecule, molecule.irrep_name, molecule.symm_orb, mo_coeff)]

def make_casci_rdm123s(ci_coeff, norb_cas, nelcas_a, nelcas_b):

    #(dm1a, dm1b), (dm2aa, dm2ab, dm2bb) = fci.direct_uhf.make_rdm12s(mc.ci, cas[1], cas[0], reorder=True)
    (dm1a, dm1b), (dm2aa, dm2ab, dm2bb), (dm3aaa, dm3aab, dm3abb, dm3bbb) = fci.direct_spin1.make_rdm123s(ci_coeff, norb_cas, (nelcas_a, nelcas_b), reorder=True)
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

    # print("|dm3| = ", np.linalg.norm(dm3.flatten()))
    
    rdms = {'1': dm1, '2': dm2, '3': dm3}
    return rdms

# def print_active_ao_content(mol, mo_coeff, ncore, ncas,
#                             label="",
#                             thresh=1e-3,
#                             per_block=5,
#                             coeff_fmt="{:.4f}"):
#     """
#     Block‐style print of Mulliken AO pops for each ACTIVE MO.
#
#     Parameters
#     ----------
#     mol       : pyscf.gto.Mole
#     mo_coeff  : (nao,nmo) ndarray or dict (if symmetry=True)
#     ncore     : int      # of core/frozen MOs
#     ncas      : int      # of active MOs
#     label     : str      header to print before the blocks
#     thresh    : float    only show pops >= thresh
#     per_block : int      how many ACTIVE MOs per row
#     coeff_fmt : str      e.g. "{:.4f}"  (precision only; width auto)
#     """
#     # 1) Flatten mo_coeff if symm=True
#     if isinstance(mo_coeff, dict):
#         irrep_names = mol.irrep_name
#         C_blocks, symlbls = [], []
#         for ir in irrep_names:
#             C_ir = mo_coeff.get(ir, np.zeros((mol.nao,0)))
#             C_blocks.append(C_ir)
#             symlbls += [ir]*C_ir.shape[1]
#         C_full = np.hstack(C_blocks)
#     else:
#         C_full    = mo_coeff.copy()
#         symlbls   = None
#
#     nao, nmo = C_full.shape
#
#     # 2) Overlap and AO labels
#     S  = mol.intor('int1e_ovlp')
#     ao_labels = mol.ao_labels()    # list of strings
#
#     # 3) Extract just the active block and compute pops
#     iact = list(range(ncore, ncore+ncas))
#     C_act = C_full[:, iact]        # shape (nao, ncas)
#     pops  = C_act * (S @ C_act)     # shape (nao, ncas)
#
#     # header for the entire section
#     print(f"\n=== AO content of ACTIVE orbitals  ({label}) ===\n")
#
#     # 4) Loop over blocks of active MOs
#     for bstart in range(0, ncas, per_block):
#         block     = list(range(bstart, min(bstart+per_block, ncas)))
#         mo_numbers= [ncore + i for i in block]   # global MO idx
#
#         # Build header labels: "MO5[A1]" or just "MO5" if no sym
#         hdr_labels = []
#         for im in mo_numbers:
#             txt = f"MO{im+1}"
#             if symlbls:
#                 txt += f"[{symlbls[im]}]"
#             hdr_labels.append(txt)
#
#         # column widths
#         ao_w    = max(len(lbl) for lbl in ao_labels) + 2
#         num_w   = max(len(coeff_fmt.format(pops[:,i].max())) for i in block)
#         label_w = max(len(h) for h in hdr_labels)
#         col_w   = max(num_w, label_w)
#
#         # print header row
#         row = f"{'AO':<{ao_w}}"
#         for h in hdr_labels:
#             row += f"{h:>{col_w}}"
#         print(row)
#         print("-" * len(row))
#
#         # print each AO line
#         for iao, ao in enumerate(ao_labels):
#             line = f"{ao:<{ao_w}}"
#             for j in block:
#                 val = pops[iao,j]
#                 if abs(val) < thresh:
#                     line += " " * col_w
#                 else:
#                     line += f"{coeff_fmt.format(val):>{col_w}}"
#             print(line)
#         print()  # blank between blocks
#
#     print("----------------------------------------------------------\n")

def print_active_ao_content(mol, mo_coeff, ncore, ncas,
                            label="",
                            thresh=1e-3):
    """
    Mulliken AO contributions of every active orbital.

    Parameters
    ----------
    mol       : pyscf.gto.Mole
    mo_coeff  : (nao,nmo) MO coefficient matrix
    ncore     : # frozen/core MOs
    ncas      : # active orbitals
    label     : header string
    thresh    : show |pop| ≥ thresh
    """
    S  = mol.intor('int1e_ovlp')
    ao = mol.ao_labels(fmt=True)

    act = mo_coeff[:, ncore:ncore+ncas]        # the active block

    print(f"\n=== AO content of ACTIVE orbitals  ({label}) ===")
    for i in range(ncas):
        c   = act[:, i]
        pop = c * (S @ c)                      # Mulliken pop on every AO
        print(f"\n Active MO #{i:2d}")
        for iao, lbl in enumerate(ao):
            if abs(pop[iao]) >= thresh:
                print(f"   {lbl:22s}  {pop[iao]:8.4f}")
    print("--------------------------------------------------\n")


def print_mo_matrix(mf, per_block=5,
                    coeff_fmt="{:12.6f}",  # e.g. width=12, prec=6
                    occ_fmt="({:.1f})"):  # prints occupancy "(2.0)"
    """
    Print MO coefficients in the AO basis, per_block MOs as columns,
    including MO occupancy and MO symmetry (if symmetry=True in mol).
    Works for both symm=False (array) and symm=True (dict) in PySCF.
    """
    # Retrieve AO labels
    ao_labels = mf.mol.ao_labels()  # list of strings for each AO

    # Handle MO coefficients/occupancies for symm=False or symm=True
    C_raw = mf.mo_coeff
    occ_raw = mf.mo_occ
    if isinstance(C_raw, dict):
        # Symmetry enabled: flatten dict in irreps order
        irrep_names = mf.mol.irrep_name  # ordered list of irrep labels
        C_blocks = []
        occ_blocks = []
        sym_flat = []
        for ir in irrep_names:
            C_ir = C_raw.get(ir, np.zeros((mf.mo_coeff[next(iter(C_raw))].shape[0], 0)))
            occ_ir = occ_raw.get(ir, np.array([]))
            C_blocks.append(C_ir)
            occ_blocks.append(occ_ir)
            sym_flat.extend([ir] * C_ir.shape[1])
        C = np.hstack(C_blocks)
        occ = np.concatenate(occ_blocks)
        sym_labels = sym_flat
    else:
        C = C_raw
        occ = occ_raw
        sym_labels = None

    nao, nmo = C.shape

    # Extract precision from coeff_fmt (e.g., "{:12.6f}" -> precision=6)
    m = re.search(r"\.(\d+)f", coeff_fmt)
    precision = int(m.group(1)) if m else 6

    # Iterate in blocks
    for start in range(0, nmo, per_block):
        block = list(range(start, min(start + per_block, nmo)))

        # Build header labels MO#:occ:sym
        hdr_labels = []
        for j in block:
            lbl = f"MO{j + 1}{occ_fmt.format(occ[j])}"
            if sym_labels is not None:
                lbl += f"[{sym_labels[j]}]"
            hdr_labels.append(lbl)

        # Determine column width
        coeff_w = len(coeff_fmt.format(0.0))
        label_w = max(len(l) for l in hdr_labels)
        col_w = max(coeff_w, label_w)
        ao_w = max(len(a) for a in ao_labels) + 2

        # Print header
        header = f"{'AO':<{ao_w}s}"
        for lbl in hdr_labels:
            header += f"{lbl:>{col_w}s}"
        print(header)
        print("-" * len(header))

        # Print each row
        for iao, ao in enumerate(ao_labels):
            row = f"{ao:<{ao_w}s}"
            for j in block:
                row += f"{C[iao, j]:{col_w}.{precision}f}"
            print(row)
        print()  # blank line between blocks