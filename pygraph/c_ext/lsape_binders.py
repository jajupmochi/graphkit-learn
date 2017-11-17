import numpy as np
import ctypes as c
from ctypes import cdll
import os.path

def lsap_solverHG(C):
    ''' Binding for lsape hungarian solver '''

    nm = C.shape[0]
    dll_name = 'liblsap.so'
    lib = cdll.LoadLibrary(os.path.abspath(
        os.path.join(os.path.dirname(__file__), dll_name)))
    lib.lsap.restype = c.c_int
    rho = np.zeros((nm, 1), int)
    varrho = np.zeros((nm, 1), int)
    C[C == np.inf] = 10000

    lib.lsap(c.c_void_p(C.transpose().ctypes.data),
             c.c_int(nm),
             c.c_void_p(rho.ctypes.data),
             c.c_void_p(varrho.ctypes.data))

    return np.array(range(0, nm)), np.array([c.c_int(i).value for i in varrho])
