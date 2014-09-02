"""Simple wrapper around mmlabpack, trying first to find the fortran version
and using the python version as a back up

"""
from sys import modules
from utils.mmlio import log_warning
try:
    from lib.mmlabpack import mmlabpack as m
except ImportError:
    log_warning("fortran mmlabpack not imported, using python backup")
    import _mmlabpack as m

for method in dir(m):
    if method.startswith("__"):
        continue
    setattr(modules[__name__], method, getattr(m, method))
