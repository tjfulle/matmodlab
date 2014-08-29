#!/usr/bin/env python
import numpy as np
from subprocess import Popen

try:
    import tensalg_test as tt
except ImportError:
    cmd = ["gfortran", "-c", "numbers.f90", "tensalg.f90", "min_io.f90",
           "../../utils/fortran/blas_lapack-lite.f",
           "../../utils/fortran/dgpadm.f"]
    stat = Popen(cmd)
    stat.wait()
    cmd = ["f2py", "-c", "-m", "tensalg_test", "tensalg_test.f90", "*.o"]
    stat = Popen(cmd)
    stat.wait()
    import tensalg_test as tt

def asarray(a):
    return np.array([a[0,0], a[1,1], a[2,2], a[0,1], a[0,2], a[1,2]])

a = np.random.random(9).reshape(3,3)
ainv = np.linalg.inv(a)
AINV = tt.test_inv_3x3(a)
print np.allclose(ainv, AINV)

asym = .5 * (a + a.T)
ainv =  asarray(np.linalg.inv(asym))
AINV = tt.test_inv_6x1(asarray(asym))
print np.allclose(ainv, AINV)
