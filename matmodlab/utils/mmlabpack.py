"""Simple wrapper around mmlabpack, trying first to find the fortran version
and using the python version as a back up

"""
import os
import logging
import numpy as np
from sys import modules, argv
from ..product import PKG_D

mmlabpack_so = os.path.join(PKG_D, "mmlabpack.so")
warned = False
def should_warn():
    for x in ("build", "clean", "config", "convert",
              "-h", "--help", "help"):
        if x in argv:
            return False
    return True

try:
    from ..lib.mmlabpack import mmlabpack as m
except ImportError:
    from . import _mmlabpack as m
    if not warned and should_warn():
        d = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../lib")
        if not os.path.isfile(mmlabpack_so):
            logging.warn("""\
fortran mmlabpack.so not found, using python backup
             run the build script to create mmlabpack.so""")
        else:
            logging.warn("error importing fortran mmlabpack, "
                         "using python backup")
        logging.warn("python backup is significantly slower\n")
        warned = True

for method in dir(m):
    if method.startswith("__"):
        continue
    setattr(modules[__name__], method, getattr(m, method))

def isotropic_part(A):
    A[:, 3:] *= np.sqrt(2.)
    A[3:, :] *= np.sqrt(2.)
    alpha = np.sum(A[:3,:3])
    beta = np.trace(A)
    a = (2. * alpha - beta) / 15.
    b = (3. * beta - alpha) / 15.
    Aiso = b * np.eye(6)
    Aiso[:3,:3] += a * np.ones((3,3))
    Aiso[:, 3:] /= np.sqrt(2.)
    Aiso[3:, :] /= np.sqrt(2.)
    return Aiso
