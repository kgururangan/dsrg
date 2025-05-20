import numpy as np
import h5py
import tempfile
from dsrg.utilities import remove_file, flatten_dict_to_vector

class DIIS:
    def __init__(self, ndim, diis_size, out_of_core):

        ftmp = tempfile.NamedTemporaryFile()
        self.diis_size = diis_size
        self.out_of_core = out_of_core
        self.ndim = ndim
        self.file_name = ftmp.name

        if self.out_of_core:
            remove_file(self.file_name)
            f = h5py.File(self.file_name, "w")
            self.T_list = f.create_dataset("t-vectors", (self.diis_size, self.ndim))
            self.T_residuum_list = f.create_dataset("resid-vectors", (self.diis_size, self.ndim))
        else:
            self.T_list = np.zeros((self.diis_size, self.ndim))
            self.T_residuum_list = np.zeros((self.diis_size, self.ndim))

    def cleanup(self):
        if self.out_of_core:
            remove_file(self.file_name)
            
    def push(self, T, T_residuum, iteration):
            self.T_list[iteration % self.diis_size, :] = flatten_dict_to_vector(T)
            self.T_residuum_list[iteration % self.diis_size, :] = flatten_dict_to_vector(T_residuum)

    def extrapolate(self):
        B_dim = self.diis_size + 1
        B = -1.0 * np.ones((B_dim, B_dim))
        for i in range(self.diis_size):
            for j in range(i, self.diis_size):
                B[i, j] = np.dot(self.T_residuum_list[i, :].T.conj(), self.T_residuum_list[j, :])
                B[j, i] = B[i, j]
        B[-1, -1] = 0.0

        rhs = np.zeros(B_dim)
        rhs[-1] = -1.0
        
        coeff = np.linalg.solve(B, rhs)       
        x_xtrap = np.zeros(self.ndim)
        for i in range(self.diis_size):
            x_xtrap += coeff[i] * self.T_list[i, :]

        return x_xtrap
