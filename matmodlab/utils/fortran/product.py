import os
from numpy.distutils.misc_util import get_shared_lib_extension as np_so_ext
from matmodlab.product import PKG_D
_D = os.path.dirname(os.path.realpath(__file__))
LAPACK = os.path.join(_D, "blas_lapack-lite.f")
LAPACK_OBJ = os.path.join(PKG_D, "blas_lapack-lite.o")
MMLABPACK_F90 = os.path.join(_D, "mmlabpack.f90")
DGPADM_F = os.path.join(_D, "dgpadm.f")
SO_EXT = np_so_ext()
SO_EXT = ".so"
IO_F90 = os.path.join(_D, 'mml_io.f90')

MMLABPACK = [MMLABPACK_F90, DGPADM_F]

def fortran_libraries():
    return {"mmlabpack": {"source_files": MMLABPACK, "lapack": "lite"}}
