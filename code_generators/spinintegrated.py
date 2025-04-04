import wicked as w
import numpy as np
import re
from fractions import Fraction
import time

def split_single_tensor(tensor, flip=True):
    """
    Split a single expression or string of the form
    >>> H^{a0,a3}_{a1,a2}
    into a tuple containing its label, the upper, and lower indices, i.e.,
    >>> ('H', [a0, a3], [a1, a2])
    If the label is 'F' or 'V', the upper and lower indices are swapped 
    (see 4c-dsrg-mrpt2.ipynb::dsrg_mrpt2_update to see why)
    """
    tensor = str(tensor)
    label, indices = str(tensor).split('^{')
    upper = indices.split('}_')[0].split(',')
    lower = indices.split('_{')[1].split('}')[0].split(',')
    if (flip):
        # Need to include 'H' here, since the output of residuals is default 'H'. This switches the
        # output indices, so that ia->ai and ijab->abij, which is how we like it.
        if label in ['F', 'V', 'h', 'H', 't', 'f', 'v', 'h1a', 'h1b', 'h2a', 'h2b', 'h2c', 'o', 'O']: 
            upper, lower = lower, upper
    return label, upper, lower

def get_unique_tensor_indices(tensor, unused_indices, index_dict):
    """
    >>> ('V', [a0, c0], [a1, a2]) -> 'uiwx'
    Get the indices of a tensor for use in an einsum contraction.
    For example, given the tensor from split_single_tensor, 
    the list of available indices and indices that have already been assigned, 
    we can generate the index string.
    """
    label, upper, lower = tensor
    indstr = ''
    for i in upper+lower:
        if index_dict.get(i):
            indstr += index_dict[i]
        else:
            index = unused_indices[i[0]].pop(0)
            index_dict[i] = index
            indstr += index
    
    return indstr

def get_tensor_slice(tensor, fmt):
    """
    >>> ('V', [a0, c0], [a1, a2]) -> "V['acaa']" (fmt = 'dict')
    >>> ('V', [a0, c0], [a1, a2]) -> "V[a,c,a,a]" (fmt = 'slice')
    """
    if tensor[0] in ['gamma1', 'eta1', 'lambda2', 'lambda3']:
        return tensor[0], ''
    else:
        if (fmt == 'dict'):
            return tensor[0], "['" + ''.join([i[0] for i in tensor[1]]) + ''.join([i[0] for i in tensor[2]]) + "']"
        elif (fmt == 'slice'):
            if ('t' in tensor[0]): # T tensors are always particle-hole sized.
                return tensor[0], '[' + ','.join(['p'+i[0] for i in tensor[1]]) + ',' + ','.join(['h'+i[0] for i in tensor[2]]) + ']'
            else:
                return tensor[0], '[' + ','.join([i[0] for i in tensor[1]]) + ',' + ','.join([i[0] for i in tensor[2]]) + ']'

def get_lhs_tensor_name(tensor):
    """
    >>> ('V', [a0, c0], [a1, a2]) -> "Vacaa"
    """
    return tensor[0] + '_' + ''.join([i[0] for i in tensor[1]]) + ''.join([i[0] for i in tensor[2]])

def get_factor(expression):
    """
    Returns the prefactor of a right hand side expression, taking care of edge cases where an empty space (for +1.0) or a negative sign (for -1.0) are present.
    """
    factor = str(expression).split('+=')[-1].split(' ')[1]
    try:
        return float(Fraction(factor))
    except ValueError:
        if factor == '-':
            return -1.0
        else:
            return 1.0

def compile_einsum(expression, fmt='dict', tensor_name=None, remove_lambda3=False):
    """
    Compile an expression into a valid einsum expression.
    Turns a Wick&d expression (wicked._wicked.Equation) like H^{c0,a0}_{a1,a2} += 1/4 T2^{c0,a0}_{a3,a4} V^{a5,a6}_{a1,a2} eta1^{a4}_{a6} eta1^{a3}_{a5}
    into the einsum code string "H_caaa += +0.25000000 * np.einsum('iuvw,xyzr,wr,vz->iuxy', T2['caaa'], V['aaaa'], eta1, eta1, optimize=True)"
    """
    unused_indices = {'a':list('uvwxyzrst'), 'A':list('UVWXYZRST'), 
                      'c':list('ijklmn'), 'C':list('IJKLMN'),
                      'v':list('abcdef'), 'V':list('ABCDEF')}
    index_dict = {}

    lhs = expression.lhs()
    rhs = expression.rhs()

    factor = get_factor(expression)

    exstr = ''  # holds the expression part of the einsum contraction, e.g., iuvw,xyzr,wr,vz->iuxy 
    tenstr = '' # holds the tensor label part of the einsum contraction, e.g., T2['caaa'], V['aaaa'], eta1, eta1, optimize='optimal')

    for i in str(rhs).split(' '):
        _ = split_single_tensor(i)
        t_name, arr = get_tensor_slice(_, fmt) # this is the tensor name
        # process arr to find out if it's all active in T; then skip it
        skip = False
        if t_name in ['t', 'T']:
            # arr is something like [pa,pa,ha,ha]
            # arr[1:-1] is pa,pa,ha,ha
            ll = [x[-1] for x in arr[1:-1].split(',')]
            skip = all([l in ['a', 'A'] for l in ll])
        # check if lambda3 is in here
        if remove_lambda3:
            if t_name == 'lambda3':
                skip = True
        if skip:
            return ''
                
        t_inds = get_unique_tensor_indices(_, unused_indices, index_dict) # these are the indices
        # add labels for spin-integration
        order = len(t_inds) // 2
        num_alpha = len(re.findall(r'[a-z]',t_inds))
        num_beta = len(re.findall(r'[A-Z]',t_inds))
        if order == 1:
            if num_alpha == 2:
                t_name += "['a']"
            else:
                t_name += "['b']"
        elif order == 2:
            if num_alpha == 4:
                t_name += "['aa']"
            elif num_alpha == 2:
                t_name += "['ab']"
            elif num_alpha == 0:
                t_name += "['bb']"
        elif order == 3:
            if num_alpha == 6:
                t_name += "['aaa']"
            elif num_alpha == 4:
                t_name += "['aab']"
            elif num_alpha == 2:
                t_name += "['abb']"
            elif num_alpha == 0:
                t_name += "['bbb']"
        t_name += arr
        # print(t_name)
        tenstr += t_name + ', ' 
        exstr += t_inds 
        exstr += ','
    exstr = exstr[:-1] + '->'
    tenstr += "optimize=True)"

    _ = split_single_tensor(lhs)
    if (_[1] != ['']):
        left = get_lhs_tensor_name(_)
        res_indx = get_unique_tensor_indices(_, unused_indices, index_dict)
        exstr += res_indx
    else:
        left = _[0] # If it's scalar, just return the label.

    if tensor_name is not None:
        left = tensor_name

    einsumstr = left \
        + ' ' \
        + f"+= scale * {factor:+.8f} * np.einsum('"\
        + exstr \
        + "', " \
        + tenstr
    # print(einsumstr)

    return einsumstr    

def make_code(mbeq, fmt='dict', tensor_name=None, remove_lambda3=False):
    code = ''
    nlines = 0
    for i in mbeq:
        einsum = compile_einsum(i, fmt, tensor_name, remove_lambda3=remove_lambda3)
        code += '\t' + einsum + '\n'
        nlines += 1
    return code, nlines

def get_many_body_equations(op1, op2, nbody):
    """
    Returns the elements of the commutator of two operators.
    """
    comm = w.commutator(op1, op2)
    comm_expr = wt.contract(comm, nbody*2, nbody*2, inter_general=True)
    return comm_expr.to_manybody_equation("H")

def make_nbody_elements(op1, op2, nbody, spincase, fmt='slice', keys_in=None, do_lhs_slice=True, remove_lambda3=False):
    """
    Returns the elements of the commutator of two operators in einsum format.
    """
    code = ''
    nlines = 0
    mbeq = get_many_body_equations(op1, op2, nbody)
    keys = mbeq.keys() if keys_in is None else keys_in

    for key in keys:
        # only select the keys that are compatible with the chosen spincase
        num_alpha = len(re.findall(r'[a-z]', key))
        num_beta = len(re.findall(r'[A-Z]', key))
        if nbody == 1:
            if spincase == 'a': 
                if num_alpha != 2: continue
            elif spincase == 'b':
                if num_beta != 2: continue
        if nbody == 2:
            if spincase == 'aa':
                if num_alpha != 4: continue
            elif spincase == 'ab':
                if num_alpha != 2: continue
            elif spincase == 'bb':
                if num_beta != 4: continue

        if (nbody != 0 and do_lhs_slice):
            if (fmt == 'slice'):
                lhs_slice = re.findall(r'[a-zA-Z]', key)
                if (len(lhs_slice) == 4):
                    lhs_slice = ','.join(lhs_slice[2:][::-1]) + ',' + ','.join(lhs_slice[:2]) # H^ij_ab -> H[i,j,b,a]
                else:
                    lhs_slice = ','.join(reversed(lhs_slice))
            _ = make_code(mbeq[key], fmt, f"O['{spincase}'][{lhs_slice}]", remove_lambda3=remove_lambda3)
        else:
            _ = make_code(mbeq[key], fmt, f"O['{spincase}']", remove_lambda3=remove_lambda3)
        code += _[0]
        nlines += _[1]
    return code, nlines

if __name__ == "__main__":
    import itertools

    remove_lambda3 = True
    if remove_lambda3:
        print("WARNING: no 3-cumulant included!")

    w.reset_space()
    # alpha
    w.add_space("c", "fermion", "occupied", list('ijklmn'))
    w.add_space("a", "fermion", "general", list('uvwxyzrst'))
    w.add_space("v", "fermion", "unoccupied", list('abcdef'))
    # beta
    w.add_space("C", "fermion", "occupied", list('IJKLMN'))
    w.add_space("A", "fermion", "general", list('UVWXYZRST'))
    w.add_space("V", "fermion", "unoccupied", list('ABCDEF'))
    wt = w.WickTheorem()

    t0 = time.time()

    # t1a
    temp = []
    for i in itertools.product(['v+', 'a+'],['a', 'c']):
        temp.append(' '.join(i))
    t1a = w.op('t', temp, unique=True)
    # t1b
    temp = []
    for i in itertools.product(['V+', 'A+'],['A', 'C']):
        temp.append(' '.join(i))
    t1b = w.op("t", temp, unique=True)

    # t2a
    temp = []
    for i in itertools.product(['v+', 'a+'],['v+', 'a+'], ['a', 'c'], ['a', 'c']):
        temp.append(' '.join(i))
    t2a = w.op('t', temp, unique=True)
    # t2c
    temp = []
    for i in itertools.product(['V+', 'A+'],['V+', 'A+'], ['A', 'C'], ['A', 'C']):
        temp.append(' '.join(i))
    t2c = w.op('t', temp, unique=True)
    # t2b
    temp = []
    for i in itertools.product(['v+', 'a+'],['V+', 'A+'], ['a', 'c'], ['A', 'C']):
        temp.append(' '.join(i))
    t2b = w.op("t", temp, unique=True)

    # h1a
    temp = []
    for i in itertools.product(['v+', 'a+', 'c+'],['v', 'a', 'c']):
        temp.append(' '.join(i))
    h1a = w.op('h', temp, unique=True)
    # h1b
    temp = []
    for i in itertools.product(['V+', 'A+', 'C+'],['V', 'A', 'C']):
        temp.append(' '.join(i))
    h1b = w.op('h', temp, unique=True)

    # h2a
    temp = []
    for i in itertools.product(['v+', 'a+', 'c+'],['v+', 'a+', 'c+'], ['v', 'a', 'c'], ['v', 'a', 'c']):
        temp.append(' '.join(i))
    h2a = w.op('h', temp, unique=True)
    # h2c
    temp = []
    for i in itertools.product(['V+', 'A+', 'C+'],['V+', 'A+', 'C+'], ['V', 'A', 'C'], ['V', 'A', 'C']):
        temp.append(' '.join(i))
    h2c = w.op('h', temp, unique=True)
    # h2b
    temp = []
    for i in itertools.product(['v+', 'a+', 'c+'],['V+', 'A+', 'C+'], ['v', 'a', 'c'], ['V', 'A', 'C']):
        temp.append(' '.join(i))
    h2b = w.op("h", temp, unique=True)

    fmt = 'slice'

    H = [h1a, h1b, h2a, h2b, h2c]
    T = [t1a, t1b, t2a, t2b, t2c]

    output = ['c0', 'c1a', 'c1b', 'c2a', 'c2b', 'c2c']
    interaction = ['h1a', 'h1b', 'h2a', 'h2b', 'h2c']
    cluster = ['t1a', 't1b', 't2a', 't2b', 't2c']

    input_dict = {}
    for o in output:
        if '0' in o:
            nbody = 0
            spincase = '0'
        elif 'c1' in o:
            nbody = 1
            spincase = o[-1]
        elif 'c2' in o:
            nbody = 2
            spincase = o[-1]
            if spincase == 'a':
                spincase = 'aa'
            elif spincase == 'b':
                spincase = 'ab'
            elif spincase == 'c':
                spincase = 'bb'
        for i, h in zip(interaction, H):
            for j, t in zip(cluster, T):
                key = f'{i}_{j}_{o}'
                value = (h, t, nbody, spincase, fmt)
                input_dict[key] = value

    slicedef = ''
    # slicedef += "\tc = mf.core['alpha']\n"
    # slicedef += "\tC = mf.core['beta']\n"
    # slicedef += "\ta = mf.active['alpha']\n"
    # slicedef += "\tA = mf.active['beta']\n"
    # slicedef += "\tv = mf.virt['alpha']\n"
    # slicedef += "\tV = mf.virt['beta']\n"

    # slicedef += "\thc = mf.hc\n"
    # slicedef += "\thC = mf.hC\n"
    # slicedef += "\tha = mf.ha\n"
    # slicedef += "\thA = mf.hA\n"
    # slicedef += "\tpa = mf.pa\n"
    # slicedef += "\tpA = mf.pA\n"
    # slicedef += "\tpv = mf.pv\n"
    # slicedef += "\tpV = mf.pV\n"
    slicedef += "\tc = orbspace['core_alpha']\n"
    slicedef += "\tC = orbspace['core_beta']\n"
    slicedef += "\ta = orbspace['active_alpha']\n"
    slicedef += "\tA = orbspace['active_beta']\n"
    slicedef += "\tv = orbspace['virt_alpha']\n"
    slicedef += "\tV = orbspace['virt_beta']\n"

    slicedef += "\thc = orbspace['hole_core_alpha']\n"
    slicedef += "\thC = orbspace['hole_core_beta']\n"
    slicedef += "\tha = orbspace['hole_active_alpha']\n"
    slicedef += "\thA = orbspace['hole_active_beta']\n"
    slicedef += "\tpa = orbspace['particle_active_alpha']\n"
    slicedef += "\tpA = orbspace['particle_active_beta']\n"
    slicedef += "\tpv = orbspace['particle_virt_alpha']\n"
    slicedef += "\tpV = orbspace['particle_virt_beta']\n"

    nfiles = 0
    with open(f'/Users/karthik/Dropbox/dsrgpy/spinintegrated_contractions.py', 'w') as f:
    #with open(f'tmp.py', 'w') as f:
        for key in input_dict.keys():
            #if 'c2b' not in key: continue
            code, nlines = make_nbody_elements(*input_dict[key], remove_lambda3=remove_lambda3)
            print(key, nlines)
            if nlines == 0: continue
            nfiles += 1
            #with open(f'/Users/karthik/Dropbox/uforte/updates/{key}.py', 'w') as f:
            #f.write('import numpy as np\nimport time\n\n')
            f.write('def ' + key + '(O, h, t, gamma1, eta1, lambdas, orbspace, verbose=False, scale=1.0):\n')
            f.write('\t# ' + str(nlines) + ' lines\n')
            f.write('\tt0 = time.time()\n')
            f.write(slicedef+'\n')
            f.write(code + '\n')
            f.write('\tt1 = time.time()\n')
            f.write('\tif verbose: print("'+key+' took {:.4f} seconds to run.".format(t1-t0))\n\n')
            f.write('\treturn O')
            f.write('\n')
    print(f'wrote {nfiles} updates')
