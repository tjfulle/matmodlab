import os
import sys
import inspect
import warnings
from math import *
from core.product import *
from core.configurer import cfgswitch_and_warn, cfgparse

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
already_wiped = False
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
    from core.runtime import opts, set_runtime_opt

    prog = "mml run"
    desc = """{0}: run a matmodlab simulation script in the matmodlab
    environment. Simulation scripts can be run directly by the python
    interpreter if {1} is on your PYTHONPATH.""".format(prog, ROOT_D)

    parser = argparse.ArgumentParser(prog=prog, description=desc)
    parser.add_argument("-v", default=opts.verbosity,
       type=int, help="Verbosity [default: %(default)s]")
    parser.add_argument("--debug", default=opts.debug, action="store_true",
       help="Debug mode [default: %(default)s]")
    parser.add_argument("--sqa", default=opts.sqa, action="store_true",
       help="SQA mode [default: %(default)s]")
    parser.add_argument("--switch", metavar="MATERIAL", default=opts.switch,
       help="Switch material in input with MATERIAL [default: %(default)s]")
    parser.add_argument("--mimic", metavar="MATX:MATY", default=opts.mimic,
       nargs="*", help=("Run with MATY instead of MATX, if present"
             "(not supported by all models) [default: %(default)s]"))
    parser.add_argument("-I", default=os.getcwd(), help=argparse.SUPPRESS)
    parser.add_argument("-B", metavar="MATERIAL",
        help="Wipe and rebuild MATERIAL before running [default: %(default)s]")
    parser.add_argument("-V", default=False, action="store_true",
        help="Launch results viewer on completion [default: %(default)s]")
    parser.add_argument("-j", "--nprocs", type=int, default=opts.nprocs,
        help="Number of simultaneous jobs [default: %(default)s]")
    parser.add_argument("-E", action="store_true", default=False,
        help="Do not use matmodlabrc configuration file [default: False]")
    parser.add_argument("-W", choices=["std", "all", "error"], default=opts.warn,
        help="Warning level [default: %(default)s]")

    def decorated_func(*args, **kwargs):
        global already_splashed, already_wiped
        get_f = kwargs.pop("get_f", None)
        if get_f:
            parser.add_argument("source", help="Source file [default: %(default)s]")

        argv = kwargs.pop("argv", None)
        if argv is None:
            argv = sys.argv[1:]
        clargs = parser.parse_args(argv)
        if get_f and not os.path.isfile(clargs.source):
            raise SystemExit("{0}: file not found".format(clargs.source))

        # set runtime options
        set_runtime_opt("debug", clargs.debug)
        set_runtime_opt("sqa", clargs.sqa)
        set_runtime_opt("nprocs", clargs.nprocs)
        set_runtime_opt("verbosity", clargs.v)
        if clargs.W == "error":
            set_runtime_opt("Wall", True)
            set_runtime_opt("Werror", True)
        elif clargs.W == "all":
            set_runtime_opt("Wall", True)

        # directory to look for hrefs and other files
        set_runtime_opt("I", clargs.I)
        if clargs.switch:
            set_runtime_opt("switch", clargs.switch)
        if clargs.mimic:
            set_runtime_opt("mimic", clargs.mimic)
        if clargs.V:
            set_runtime_opt("viz_on_completion", True)

        if clargs.B and not already_wiped:
            name = clargs.B.strip()
            verbosity = 3 if clargs.v > 1 else 0
            if os.path.isfile(os.path.join(PKG_D, "{0}.so".format(name))):
                # removing is sufficient since the material class will attempt
                # to build non-existent materials
                os.remove(os.path.join(PKG_D, "{0}.so".format(name)))
            already_wiped = True

        if clargs.v > 0 and not already_splashed:
            sys.stdout.write(SPLASH)
            already_splashed = True
        if not opts.Wall:
            warnings.simplefilter("ignore")

        # execute the function
        if get_f:
            out = func(clargs.source)
        else:
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


def main():
    filename = exec_runner(None, get_f=1)
    return


@matmodlab
def exec_runner(filename, **kwargs):
    exec(compile(open(filename, "rb").read(), filename, 'exec'))
    return

if __name__ == "__main__":
    main()
