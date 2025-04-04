import wicked as w
import numpy as np
import re
from fractions import Fraction
import time

#
# IMPORTANT: Turn inter_general off for spinorbitals! Only use for spin-integrated!
#

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
        if label in ['F', 'V', 'h', 't', 't1', 't2', 'f', 'v', 'h1', 'h2', 'H']: 
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

def compile_einsum(expression, fmt='dict', tensor_name=None):
    """
    Compile an expression into a valid einsum expression.
    Turns a Wick&d expression (wicked._wicked.Equation) like H^{c0,a0}_{a1,a2} += 1/4 T2^{c0,a0}_{a3,a4} V^{a5,a6}_{a1,a2} eta1^{a4}_{a6} eta1^{a3}_{a5}
    into the einsum code string "H_caaa += +0.25000000 * np.einsum('iuvw,xyzr,wr,vz->iuxy', T2['caaa'], V['aaaa'], eta1, eta1, optimize='optimal')"
    """
    unused_indices = {'a':list('uvwxyzrst'), 
                      'c':list('ijklmn'),
                      'v':list('abcdef')}
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
        # This is not needed for spinorbitals!
        t_inds = get_unique_tensor_indices(_, unused_indices, index_dict) # these are the indices
        tenstr += t_name + arr + ', '
        exstr += t_inds 
        exstr += ','
    exstr = exstr[:-1] + '->'
    tenstr += "optimize='optimal')"

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

def make_code(mbeq, fmt='dict', tensor_name=None):
    code = ''
    nlines = 0
    for i in mbeq:
        einsum = compile_einsum(i, fmt, tensor_name)
        code += '\t' + einsum + '\n'
        nlines += 1
    return code, nlines

def get_many_body_equations(op1, op2, nbody):
    """
    Returns the elements of the commutator of two operators.
    """
    comm = w.commutator(op1, op2)
    comm_expr = wt.contract(comm, nbody*2, nbody*2)
    return comm_expr.to_manybody_equation("H")

def make_nbody_elements(op1, op2, nbody, fmt='slice', keys_in=None, do_lhs_slice=True):
    """
    Returns the elements of the commutator of two operators in einsum format.
    """
    code = ''
    nlines = 0
    mbeq = get_many_body_equations(op1, op2, nbody)
    keys = mbeq.keys() if keys_in is None else keys_in

    for key in keys:
        if (nbody != 0 and do_lhs_slice):
            if (fmt == 'slice'):
                lhs_slice = re.findall(r'[a-zA-Z]', key)
                if (len(lhs_slice) == 4):
                    #lhs_slice = ','.join(lhs_slice[:2]) + ',' + ','.join(lhs_slice[2:][::-1]) # H^ij_ab -> H[i,j,b,a]
                    lhs_slice = ','.join(lhs_slice[2:][::-1]) + ',' + ','.join(lhs_slice[:2]) # H^ij_ab -> H[i,j,b,a]
                else:
                    #lhs_slice = ','.join(lhs_slice)
                    lhs_slice = ','.join(reversed(lhs_slice))
                #print(lhs_slice)
                # Reverse lhs_slice for my convention
                #if len(lhs_slice) == 2:
                #    lhs_slice = ','.join(lhs_slice[::-1])
                #elif len(lhs_slice) == 4:
                #    lhs_slice = ','.join(lhs_slice[2:]) + ',' + ','.join(lhs_slice[:2])
                #print(lhs_slice)
            _ = make_code(mbeq[key], fmt, f"O[{lhs_slice}]")
        else:
            _ = make_code(mbeq[key], fmt, f"O")
        code += _[0]
        nlines += _[1]
    return code, nlines

if __name__ == "__main__":
    import itertools

    w.reset_space()
    w.add_space("c", "fermion", "occupied", list('ijklmn'))
    w.add_space("a", "fermion", "general", list('uvwxyzrst'))
    w.add_space("v", "fermion", "unoccupied", list('abcdef'))
    wt = w.WickTheorem()

    t0 = time.time()

    # t1
    #temp = []
    #for i in itertools.product(['v+', 'a+'],['a', 'c']):
    #    temp.append(' '.join(i))
    #t1 = w.op('t1', temp, unique=True)
    # t2
    #temp = []
    #for i in itertools.product(['v+', 'a+'],['v+', 'a+'], ['a', 'c'], ['a', 'c']):
    #    temp.append(' '.join(i))
    #t2 = w.op('t2', temp, unique=True)

    # h1
    #temp = []
    #for i in itertools.product(['v+', 'a+', 'c+'],['v', 'a', 'c']):
    #    temp.append(' '.join(i))
    #h1 = w.op('h1', temp, unique=True)
    # h2a
    #temp = []
    #for i in itertools.product(['v+', 'a+', 'c+'],['v+', 'a+', 'c+'], ['v', 'a', 'c'], ['v', 'a', 'c']):
    #    temp.append(' '.join(i))
    #h2 = w.op('h2', temp, unique=True)

    h1 = w.utils.gen_op('h1',1,'cav','cav',diagonal=True)
    h2 = w.utils.gen_op('h2',2,'cav','cav',diagonal=True)

    t1 = w.utils.gen_op('t1',1,'av','ca',diagonal=False)
    t2 = w.utils.gen_op('t2',2,'av','ca',diagonal=False)

    fmt = 'slice'

    H = [h1, h2] 
    T = [t1, t2]

    output = ['c0', 'c1', 'c2']
    interaction = ['h1', 'h2'] 
    cluster = ['t1', 't2']

    input_dict = {}
    for o in output:
        if '0' in o:
            nbody = 0
        elif 'c1' in o:
            nbody = 1
        elif 'c2' in o:
            nbody = 2
        for i, h in zip(interaction, H):
            for j, t in zip(cluster, T):
                key = f'{i}_{j}_{o}'
                value = (h, t, nbody, fmt)
                input_dict[key] = value

    slicedef = ''
    slicedef += "\tc = orbspace['core']\n"
    slicedef += "\ta = orbspace['active']\n"
    slicedef += "\tv = orbspace['virt']\n"

    slicedef += "\thc = orbspace['hole_core']\n"
    slicedef += "\tha = orbspace['hole_active']\n"
    slicedef += "\tpa = orbspace['particle_active']\n"
    slicedef += "\tpv = orbspace['particle_virt']\n"

    nfiles = 0
    with open(f'/Users/karthik/Dropbox/dsrgpy/spinorbital_contractions.py', 'w') as f:
        for key in input_dict.keys():
            # if 'c0' not in key: continue
            code, nlines = make_nbody_elements(*input_dict[key])
            print(key, nlines)
            if nlines == 0: continue
            nfiles += 1
            #with open(f'/Users/karthik/Dropbox/uforte/updates/{key}.py', 'w') as f:
            #f.write('import numpy as np\nimport time\n\n')
            f.write('def ' + key + '(O, Hbar, t, gamma1, eta1, lambdas, orbspace, verbose=False, scale=1.0):\n')
            f.write('\t# ' + str(nlines) + ' lines\n')
            f.write('\tt0 = time.time()\n')
            f.write(slicedef+'\n')
            f.write('\tt1, t2 = t # Unpack cluster operator\n')
            f.write('\th1, h2 = Hbar # Unpack Hamiltonian\n')
            f.write("\tlambda2 = lambdas['2'] # 2-cumulant\n")
            f.write("\t# lambda3 = lambdas['3'] # 3-cumulant\n")
            f.write(code + '\n')
            f.write('\tt1 = time.time()\n')
            f.write('\tif verbose: print("'+key+' took {:.4f} seconds to run.".format(t1-t0))\n\n')
            f.write('\treturn O')
            f.write('\n')
    print(f'wrote {nfiles} updates')
