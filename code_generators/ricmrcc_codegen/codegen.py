import argparse
import time

from utils import get_spinint_clusterop, get_spinint_hamiltonian
from generator import make_function 

def main(args):

    ncomm = args.ncomm
    nbody = args.nbody

    approx1 = True
    approx2 = True

    h1a, h1b, h2a, h2b, h2c = get_spinint_hamiltonian()
    t1a, t1b, t2a, t2b, t2c = get_spinint_clusterop()

    H = h1a + h1b + h2a + h2b + h2c
    T = t1a + t1b + t2a + t2b + t2c

    _t0 = time.time()
    make_function(H, T, ncomm, nbody, approx1=approx1, approx2=approx2)
    _t1 = time.time()
    print(f"Make function in {_t1 - _t0} seconds.")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Create equations for ric-MRCC methods.")
    parser.add_argument('ncomm', type=int, help="Number of commutators.")
    parser.add_argument('nbody', type=int, help="Many-body rank of output tensor.")
    args = parser.parse_args()

    main(args)


