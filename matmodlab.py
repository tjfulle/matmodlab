import os
import sys
import inspect
import warnings
from math import *
from core.product import *
from core.configurer import cfgswitch_and_warn

if os.getenv("MMLMTLS"):
    cfgswitch_and_warn()

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

# --- ADD CWD TO sys.path
sys.path.insert(0, os.getcwd())

# ------------------------ FACTORY METHODS TO SET UP AND RUN A SIMULATION --- #
from core.driver import Driver
from core.material import Material
from core.mat_point_sim import MaterialPointSimulator
from core.permutator import Permutator, PermutateVariable
from core.optimizer import Optimizer, OptimizeVariable
from core.functions import Function
from core.logger import Logger
from core.test import TestBase, TestError as TestError
from materials.addon_expansion import Expansion
from materials.addon_trs import TRS
from materials.addon_viscoelastic import Viscoelastic
RAND = np.random.RandomState()


def genrand():
    return RAND.random_sample()
randreal = genrand()

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
    from utils.clparse import parse_sim_argv
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

        return out

    return decorated_func


def gen_runid():
    stack = inspect.stack()[1]
    return os.path.splitext(os.path.basename(stack[1]))[0]


def get_my_directory():
    """return the directory of the calling function"""
    stack = inspect.stack()[1]
    d = os.path.dirname(os.path.realpath(stack[1]))
    return d
