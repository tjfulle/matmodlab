import os
import sys
import inspect
import warnings
from math import *
from distutils.spawn import find_executable as which

__version__ = (2, 0, 0)

errors = []
(major, minor, micro, relev, ser) = sys.version_info
if (major != 3 and major != 2) or (major == 2 and minor < 7):
    errors.append("python >= 2.7 required")
    errors.append("  {0} provides {1}.{2}.{3}".format(
        sys.executable, major, minor, micro))

# --- numpy
try: import numpy as np
except ImportError: errors.append("numpy not found")

# --- scipy
try: import scipy
except ImportError: errors.append("scipy not found")

# check prerequisites
if errors:
    raise SystemExit("*** error: matmodlab could not run due to the "
                     "following errors:\n  {0}".format("\n  ".join(errors)))
del errors

# ------------------------------------------------ PROJECT WIDE CONSTANTS --- #
PLATFORM = sys.platform
PYEXE = os.path.realpath(sys.executable)

ROOT_D = os.path.dirname(os.path.realpath(__file__))
CORE = os.path.join(ROOT_D, "core")
VIZ_D = os.path.join(ROOT_D, "viz")
UTL_D = os.path.join(ROOT_D, "utils")
TLS_D = os.path.join(ROOT_D, "toolset")
TEST_D = os.path.join(ROOT_D, "tests")
PKG_D = os.path.join(ROOT_D, "lib")
BLD_D = os.path.join(ROOT_D, "build")
LIB_D = os.path.join(ROOT_D, "lib")
EXO_D = os.path.join(UTL_D, "exojac")
MATLIB = os.path.join(ROOT_D, "materials")

# --- INFORMATION FILE
MML_IFILE = "mml_i.py"

# --- MATERIAL FILE
MML_MFILE = "mml_m.py"

# --- OUTPUT DATABASE FILE
F_EVALDB = "mml-evaldb.xml"

# --- ENVIRONMENT VARIABLES
PATH = os.getenv("PATH").split(os.pathsep)
if TLS_D not in PATH:
    PATH.insert(0, TLS_D)
UMATS = [d for d in os.getenv("MMLMTLS", "").split(os.pathsep) if d.split()]
FFLAGS = [x for x in os.getenv("FFLAGS", "").split() if x.split()]
FC = which(os.getenv("FC", "gfortran"))

# --- ADD CWD TO sys.path
sys.path.insert(0, os.getcwd())

# --- ENVIRONMENT TO USE WHEN RUNNING subprocess.Popen OR subprocess.call
MML_ENV = dict(os.environ)
pypath = MML_ENV.get("PYTHONPATH", "").split(os.pathsep)
pypath.extend([ROOT_D, EXO_D])
MML_ENV["PYTHONPATH"] = os.pathsep.join(p for p in pypath if p.split())
MML_ENV["PATH"] = os.pathsep.join(PATH)
del pypath

SPLASH = """\
                  M           M    M           M    L
                 M M       M M    M M       M M    L
                M   M   M   M    M   M   M   M    L
               M     M     M    M     M     M    L
              M           M    M           M    L
             M           M    M           M    L
            M           M    M           M    L
           M           M    M           M    LLLLLLLLL
                     Material Model Laboratory v {0}

""".format(".".join("{0}".format(i) for i in __version__))

# ------------------------ FACTORY METHODS TO SET UP AND RUN A SIMULATION --- #
from core.driver import Driver
from core.material import Material
from core.mat_point_sim import MaterialPointSimulator
from core.permutator import Permutator, PermutateVariable
from core.optimizer import Optimizer, OptimizeVariable
from utils.functions import Function
from core.logger import Logger
from core.test import TestBase

# --- DECORATOR FOR SIMULATION
already_splashed = False
def matmodlab(func):
    """Decorator for func

    Parameters
    ----------
    func : callable
        Any callable function

    Returns
    -------
    decorated_func : callable
        The decorated function

    Notes
    -----
    Decorator parses command line arguments, executes them, calls func, and
    does any clean up

    """
    from toolset.clparse import parse_sim_argv
    from core.runtime import opts, set_runtime_opt
    def decorated_func(*args, **kwargs):
        global already_splashed
        clargs = parse_sim_argv()
        if clargs.v > 0 and not already_splashed:
            sys.stdout.write(SPLASH)
            already_splashed = True
        if not opts.Wall:
            warnings.simplefilter("ignore")

        # execute the function
        out = func(*args, **kwargs)

        if clargs.v:
            from utils.quotes import write_random_quote
            write_random_quote()

    return decorated_func


def gen_runid():
    stack = inspect.stack()[1]
    return os.path.splitext(os.path.basename(stack[1]))[0]


def get_my_directory():
    """return the directory of the calling function"""
    stack = inspect.stack()[1]
    d = os.path.dirname(os.path.realpath(stack[1]))
    return d
