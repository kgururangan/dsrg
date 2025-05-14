import itertools
import wicked as w

def get_spinint_clusterop():
    w.reset_space()
    # alpha
    w.add_space("c", "fermion", "occupied", list('ijklmno'))
    w.add_space("a", "fermion", "general", list('uvwxyzrstpq'))
    w.add_space("v", "fermion", "unoccupied", list('abcdefg'))
    # beta
    w.add_space("C", "fermion", "occupied", list('IJKLMNO'))
    w.add_space("A", "fermion", "general", list('UVWXYZRSTPQ'))
    w.add_space("V", "fermion", "unoccupied", list('ABCDEFG'))

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
    return t1a, t1b, t2a, t2b, t2c

def get_spinint_hamiltonian():
    w.reset_space()
    # alpha
    w.add_space("c", "fermion", "occupied", list('ijklmno'))
    w.add_space("a", "fermion", "general", list('uvwxyzrstpq'))
    w.add_space("v", "fermion", "unoccupied", list('abcdefg'))
    # beta
    w.add_space("C", "fermion", "occupied", list('IJKLMNO'))
    w.add_space("A", "fermion", "general", list('UVWXYZRSTPQ'))
    w.add_space("V", "fermion", "unoccupied", list('ABCDEFG'))

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
    return h1a, h1b, h2a, h2b, h2c
