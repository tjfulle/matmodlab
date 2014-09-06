"""Simple wrapper around mmlabpack, trying first to find the fortran version
and using the python version as a back up

"""
import os
from sys import modules
from core.logger import ConsoleLogger
try:
    from lib.mmlabpack import mmlabpack as m
except ImportError:
    d = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../lib")
    if not os.path.isfile(os.path.join(d, "mmlabpack.so")):
        ConsoleLogger.warn("""\
fortran mmlabpack.so not found, using python backup
             run the build script to create mmlabpack.so""")
    else:
        ConsoleLogger.warn("error importing fortran mmlabpack, using python backup")
    ConsoleLogger.warn("python backup is significantly slower")
    import _mmlabpack as m

for method in dir(m):
    if method.startswith("__"):
        continue
    setattr(modules[__name__], method, getattr(m, method))
