import os
import sys
import shutil
from numpy.distutils.misc_util import get_shared_lib_extension as np_so_ext
from distutils.spawn import find_executable as which

import materials.materialdb as mdb
import utils.namespace as ns


__version__ = (1, 0, 0)
ROOT_D = os.path.dirname(os.path.realpath(__file__))
MTL_DB_D = os.path.join(ROOT_D, "materials/db")
F_MTL_PARAM_DB = os.path.join(MTL_DB_D, "material_properties.db")
F_EVALDB = "mml-evaldb.xml"
RESTART = -2

ROOT_D = os.path.dirname(os.path.realpath(__file__))

PYEXE = os.path.realpath(sys.executable)
CORE = os.path.join(ROOT_D, "core")
VIZ_D = os.path.join(ROOT_D, "viz")
UTL_D = os.path.join(ROOT_D, "utils")
TLS_D = os.path.join(ROOT_D, "toolset")
TESTS_D = os.path.join(ROOT_D, "tests")
PKG_D = os.path.join(ROOT_D, "lib")
BLD_D = os.path.join(ROOT_D, "build")
LIB_D = os.path.join(ROOT_D, "lib")
EXO_D = os.path.join(UTL_D, "exo")

FIO = os.path.join(ROOT_D, "utils/fortran/mmlfio.f90")
ABQIO = os.path.join(ROOT_D, "utils/fortran/abqio.f90")
ABQUMAT = os.path.join(ROOT_D, "utils/fortran/abaumat.pyf")
ABQUAHI = os.path.join(ROOT_D, "utils/fortran/abauanisohypinv.pyf")

SO_EXT = np_so_ext()

# environment variables
PATH = os.getenv("PATH").split(os.pathsep)
if TLS_D not in PATH:
    PATH.insert(0, TLS_D)
UMATS = [d for d in os.getenv("MMLMTLS", "").split(os.pathsep) if d.split()]
FFLAGS = [x for x in os.getenv("FFLAGS", "").split() if x.split()]
FC = which(os.getenv("FC", "gfortran"))

# Add cwd to sys.path
sys.path.insert(0, os.getcwd())

# Environment to use when running subprocess.Popen or subprocess.call
MML_ENV = dict(os.environ)
pypath = MML_ENV.get("PYTHONPATH", "").split(os.pathsep)
pypath.extend([ROOT_D, EXO_D])
MML_ENV["PYTHONPATH"] = os.pathsep.join(p for p in pypath if p.split())
MML_ENV["PATH"] = os.pathsep.join(PATH)

# The material database - modify sys.path to find materials
MTL_DB = mdb.MaterialDB.gen_db(UMATS)
sys.path = MTL_DB.path + [os.getcwd()] + sys.path


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


cfg = ns.Namespace()
cfg.debug = False
cfg.sqa = False
cfg.I = None
cfg.verbosity = 1
cfg.runid = None


def cout(message, end="\n"):
    """Write message to stdout """
    if cfg.verbosity:
        sys.__stdout__.write(message + end)
        sys.__stdout__.flush()


def cerr(message):
    """Write message to stderr """
    sys.__stderr__.write(message + "\n")
    sys.__stderr__.flush()


def remove(path):
    """Remove file or directory -- dangerous!

    """
    if not os.path.exists(path): return
    try: os.remove(path)
    except OSError: shutil.rmtree(path)
    return


def check_prereqs():
    errors = []
    platform = sys.platform
    (major, minor, micro, relev, ser) = sys.version_info
    if (major != 3 and major != 2) or (major == 2 and minor < 7):
        errors.append("python >= 2.7 required")
        errors.append("  {0} provides {1}.{2}.{3}".format(
                sys.executable, major, minor, micro))

    # --- numpy
    try: import numpy
    except ImportError: errors.append("numpy not found")

    # --- scipy
    try: import scipy
    except ImportError: errors.append("scipy not found")
    return errors


