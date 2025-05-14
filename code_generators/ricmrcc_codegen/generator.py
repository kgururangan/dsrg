import wicked as w
import re
import time
import itertools
from fractions import Fraction

_OUTDIR = "output"

_MAXIMUM_CUMULANT = 3

_CANONICAL_INDICES = {'active_alpha': ['u', 'v', 'w', 'x', 'y', 'z', 'p', 'q', 'r', 's', 't'],
                      'active_beta': ['U', 'V', 'W', 'X', 'Y', 'Z', 'P', 'Q', 'R', 'S', 'T'],
                      'core_alpha': ['i', 'j', 'k', 'l', 'm', 'n', 'o'],
                      'core_beta': ['I', 'J', 'K', 'L', 'M', 'N', 'O'],
                      'virtual_alpha': ['a', 'b', 'c', 'd', 'e', 'f', 'g'],
                      'virtual_beta': ['A', 'B', 'C', 'D', 'E', 'F', 'G'],
}

_LIST_OF_KEYS = [
    '|',
    'c|v',
    'a|v',
    'c|a',
    'cc|vv',
    'ca|vv',
    'ac|vv',
    'aa|vv',
    'cc|av',
    'ca|av',
    'ac|av',
    'aa|av',
    'cc|va',
    'ca|va',
    'ac|va',
    'aa|va',
    'cc|aa',
    'ca|aa',
    'ac|aa',
]


def get_mbeq(H, T, ncomm, nbody):

    w.reset_space()
    # alpha
    w.add_space("c", "fermion", "occupied", list('ijklmno'))
    w.add_space("a", "fermion", "general", list('uvwxyzrstpq'))
    w.add_space("v", "fermion", "unoccupied", list('abcdefg'))
    # beta
    w.add_space("C", "fermion", "occupied", list('IJKLMNO'))
    w.add_space("A", "fermion", "general", list('UVWXYZRSTPQ'))
    w.add_space("V", "fermion", "unoccupied", list('ABCDEFG'))
    wt = w.WickTheorem()
    wt.set_max_cumulant(_MAXIMUM_CUMULANT)

    if ncomm == 1:
        comm = w.commutator(H, T)
    elif ncomm == 2:
        comm = w.rational(1,2) * w.commutator(w.commutator(H, T), T)

    mbeq = wt.contract(comm, nbody*2, nbody*2, inter_general=True).to_manybody_equation('X')

    # Return the set of projections corresponding to the content of T(nbody)
    mbeq_t = {}
    for i, (key, value) in enumerate(mbeq.items()):
        # skip the first entry of mbeq, as this is always all active A|A, AA|AA, etc.
        # if i == 0: continue
        if key.lower() not in _LIST_OF_KEYS: continue
        # if (key == 'A|A' or key == 'AA|AA' or key == 'AAA|AAA'
        #     or key == 'a|a' or key == 'aa|aa' or key == 'aaa|aaa'
        #     or key == 'aA|aA' or key == 'aaA|aaA' or key == 'aAA|aAA'): continue
        ket, bra = key.split('|')
        if 'C' in bra or 'c' in bra or 'V' in ket or 'v' in ket:
            continue
        mbeq_t[key] = value
    
    return mbeq_t


def canonicalize_indices(list_of_indices):
    canonicalized_indices = []
    for idx in list_of_indices:
        
        cav = idx[0]
        if cav == 'a':
            orb_type = 'active_alpha'
        elif cav == 'A':
            orb_type = 'active_beta'
        elif cav == 'c':
            orb_type = 'core_alpha'
        elif cav == 'C':
            orb_type = 'core_beta'
        elif cav == 'v':
            orb_type = 'virtual_alpha'
        elif cav == 'V':
            orb_type = 'virtual_beta'
            
        number = int(idx[-1])
        
        canonicalized_indices.append(_CANONICAL_INDICES[orb_type][number])
        
    return canonicalized_indices

def get_factor(expression):
    """
    Returns the prefactor of a right hand side expression, 
    taking care of edge cases where an empty space (for +1.0) 
    or a negative sign (for -1.0) are present.
    """
    factor = str(expression).split('+=')[-1].split(' ')[1]
    try:
        return float(Fraction(factor))
    except ValueError:
        if factor == '-':
            return -1.0
        else:
            return 1.0

def finetune_label(label):
    if label == 'lambda2' or label == 'lambda3':
        label = 'lambdas'
    # elif label == 'gamma1':
    #     label = 'self.ref.gam1'
    # elif label == 'eta1':
    #     label = 'self.ref.eta1'
    return label

def unpack_tensor(tensor):
    """
    Split a single expression or string of the form
    >>> H^{a0,a3}_{a1,a2}
    into a tuple containing its label, the upper, and lower indices, i.e.,
    >>> ('H', [a0, a3], [a1, a2])
    """
    tensor = str(tensor)
    label, indices = str(tensor).split('^{')
    upper = indices.split('}_')[0].split(',')
    lower = indices.split('_{')[1].split('}')[0].split(',')
    
    # My convention: t_{a}^{i} = <a|t|i>, so bottom should be bra and top should be ket.
    # Wicked flips this, so assign Wicked's top to bra and bottom to ket. Now, we have
    # a proper contravariant notation.
    # bra = upper
    # ket = lower
    bra = lower
    ket = upper

    label = finetune_label(label)

    if bra == [''] and ket == ['']:
        spincase = '0'
        label += f"['{spincase}']"
        return label, bra, ket

    # convert the bra and ket indices to a canonically defined set
    # a{n), n = 0, 1, 2, 3, ... => {u, v, w, x, y, z, p, q, r, s, t}
    # c{n}, n = 0, 1, 2, 3, ... => {i, j, k, l, m, n, o}
    # v{n}, n = 0, 1, 2, 3, ... => {a, b, c, d, e, f, g}
    bra = canonicalize_indices(bra)
    ket = canonicalize_indices(ket)

    # determine the spincase
    spincase = get_spincase(''.join(bra), ''.join(ket))

    # determine the slicing pattern
    slice_pattern = get_slice_pattern(label, bra, ket)

    # append spincase to label, assuming that spincases are stored in a dictionary
    label += f"['{spincase}']"
    if slice_pattern: label += f"[{','.join(slice_pattern)}]"
    
    return label, bra, ket

def get_spincase(bra, ket):

    assert len(bra) == len(ket)

    num_alpha = len(re.findall(r'[a-z]', bra))
    num_beta = len(re.findall(r'[A-Z]', bra))

    return num_alpha * 'a' + num_beta * 'b'

def get_slice_pattern(label, bra, ket):

    if label == 'eta1' or label == 'gamma1' or label == 'lambda2' or label == 'lambda3' or label == 'lambdas':
        return ''

    slice_pattern = []
    if label == 'h':
        for char in bra:
            if char in _CANONICAL_INDICES['core_alpha']:
                slice_pattern.append('c')
            elif char in _CANONICAL_INDICES['active_alpha']:
                slice_pattern.append('a')
            elif char in _CANONICAL_INDICES['virtual_alpha']:
                slice_pattern.append('v')
            elif char in _CANONICAL_INDICES['core_beta']:
                slice_pattern.append('C')
            elif char in _CANONICAL_INDICES['active_beta']:
                slice_pattern.append('A')
            elif char in _CANONICAL_INDICES['virtual_beta']:
                slice_pattern.append('V')
        for char in ket:
            if char in _CANONICAL_INDICES['core_alpha']:
                slice_pattern.append('c')
            elif char in _CANONICAL_INDICES['active_alpha']:
                slice_pattern.append('a')
            elif char in _CANONICAL_INDICES['virtual_alpha']:
                slice_pattern.append('v')
            elif char in _CANONICAL_INDICES['core_beta']:
                slice_pattern.append('C')
            elif char in _CANONICAL_INDICES['active_beta']:
                slice_pattern.append('A')
            elif char in _CANONICAL_INDICES['virtual_beta']:
                slice_pattern.append('V')
    elif label == 't' or 'X':
        for char in bra:
            if char in _CANONICAL_INDICES['core_alpha']:
                print(label, bra, ket)
                raise ValueError("NOOO")
            elif char in _CANONICAL_INDICES['core_beta']:
                print(label, bra, ket)
                raise ValueError("NOOO")
            elif char in _CANONICAL_INDICES['active_alpha']:
                slice_pattern.append('pa')
            elif char in _CANONICAL_INDICES['virtual_alpha']:
                slice_pattern.append('pv')
            elif char in _CANONICAL_INDICES['active_beta']:
                slice_pattern.append('pA')
            elif char in _CANONICAL_INDICES['virtual_beta']:
                slice_pattern.append('pV')
        for char in ket:
            if char in _CANONICAL_INDICES['virtual_alpha']:
                print(label, bra, ket)
                raise ValueError("NOOO")
            elif char in _CANONICAL_INDICES['virtual_beta']:
                print(label, bra, ket)
                raise ValueError("NOOO")
            elif char in _CANONICAL_INDICES['active_alpha']:
                slice_pattern.append('ha')
            elif char in _CANONICAL_INDICES['core_alpha']:
                slice_pattern.append('hc')
            elif char in _CANONICAL_INDICES['active_beta']:
                slice_pattern.append('hA')
            elif char in _CANONICAL_INDICES['core_beta']:
                slice_pattern.append('hC')
                
    return slice_pattern

            
def term_to_einsum(factor, term_labels, term_indices, output_label, output_indices):
    contr = ','.join(term_indices) + '->' + output_indices
    terms = ', '.join(term_labels)
        
    einsum_str = f"{output_label} += {factor} * np.einsum('{contr}', {terms}, optimize=True)"
    return einsum_str           
           
def make_function(H, T, ncomm, nbody, output=None, approx1=False, approx2=False, verbose=False):

    approximation = {'act-ext': (ncomm == 2) and approx1, # neglect t(u..) * t(v..) contractions
                     '3-act': (ncomm == 2) and approx2,   # neglect t2(uvxp) in C2 <- [[H, T], T] (neglect lambda2)
                     }

    func_name = f'H_T_ncomm{ncomm}_nbody{nbody}'
    time_print = "print(f'Took {elapsed_time} seconds.')"


    with open(f'{func_name}.py', 'w') as f:
        f.write("import time\n")
        f.write("import numpy as np\n")
        f.write("\n")
        f.write(f'def {func_name}(X, h, t, gamma1, eta1, lambdas, orbspace, verbose=False):\n')
        f.write(f'\ttic = time.time()\n')
        f.write(f"\tc = orbspace['core_alpha']\n")
        f.write(f"\tC = orbspace['core_beta']\n")
        f.write(f"\ta = orbspace['active_alpha']\n")
        f.write(f"\tA = orbspace['active_beta']\n")
        f.write(f"\tv = orbspace['virt_alpha']\n")
        f.write(f"\tV = orbspace['virt_beta']\n")
        f.write(f"\thc = orbspace['hole_core_alpha']\n")
        f.write(f"\thC = orbspace['hole_core_beta']\n")
        f.write(f"\tha = orbspace['hole_active_alpha']\n")
        f.write(f"\thA = orbspace['hole_active_beta']\n")
        f.write(f"\tpa = orbspace['particle_active_alpha']\n")
        f.write(f"\tpA = orbspace['particle_active_beta']\n")
        f.write(f"\tpv = orbspace['particle_virt_alpha']\n")
        f.write(f"\tpV = orbspace['particle_virt_beta']\n")
        f.write("\n")

        mbeq = get_mbeq(H, T, ncomm, nbody)

        # Loop over many-body components ("keys") in the operator
        for key, equations in mbeq.items():

            f.write(f"\t# {key}\n")
            # Loop over the different diagrams contributing to the given component
            for equation in equations:

                skip = False

                if verbose:
                    print(equation)

                # Obtain the LHS indices
                output_label, out_bra, out_ket = unpack_tensor(equation.lhs())
                output_indices = ''.join(out_bra) + ''.join(out_ket)

                # Obtain the factor on the diagram
                factor = get_factor(equation)
                # rhs_expression = equation.rhs_expression()

                # Loop over the different RHS tensors entering each diagram
                term_labels = []
                term_indices = []
                for expression in equation.rhs().tensors():
                    # Upack the tensor in a given expression
                    # print(expression)
                    label, bra, ket = unpack_tensor(expression)
                    term_labels.append(label)
                    term_indices.append(''.join(bra) + ''.join(ket))

                # Apply approximations, if applicable
                # active-ext approximation (approx1)
                num_act_t = [0 for _ in range(len(term_labels))]
                for i, (lab, ind) in enumerate(zip(term_labels, term_indices)):
                    if lab == 't':
                        for char in inds:
                            if (char in _CANONICALIZED_INDICES['active_alpha']
                                or char in _CANONICALIZED_INDICES['active_beta']):
                                num_act_t[i] += 1
                                # break
                flag1 = len([1 for n in num_act_t if n > 0])
                # 3-active approximation
                flag2 = len([1 for n in num_act_t if n >= 3])
                if flag1 >= 2 and approximation['act-ext']: 
                    skip = True
                if flag2 >= 1 and approximation['3-act']:
                    skip = True

                # Obtain einsum contraction
                einsum_str = term_to_einsum(factor, term_labels, term_indices, output_label, output_indices)

                # Print the einsum contraction (either to file or to stdout)
                # if output:
                #     with open(
                f.write(f"\t{einsum_str}\n")
                # else:
                #     print(f"\t{einsum_str}")

        # f.write("\n")
        f.write("\ttoc = time.time()\n")
        f.write("\telapsed_time = toc - tic\n")
        f.write("\tif verbose:\n")
        f.write(f"\t\t{time_print}\n")
        f.write("\treturn X")
