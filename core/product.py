import os
import sys
import ConfigParser
from distutils.spawn import find_executable as which
from core.configurer import cfgswitch_and_warn, cfgparse


# ------------------------------------------------ PROJECT WIDE CONSTANTS --- #
__version__ = (2, 0, 0)

PLATFORM = sys.platform
PYEXE = os.path.realpath(sys.executable)

ROOT_D = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
CORE = os.path.join(ROOT_D, "core")
assert os.path.isdir(CORE)
VIZ_D = os.path.join(ROOT_D, "viz")
UTL_D = os.path.join(ROOT_D, "utils")
TLS_D = os.path.join(ROOT_D, "toolset")
TEST_D = os.path.join(ROOT_D, "tests")
PKG_D = os.path.join(ROOT_D, "lib")
BLD_D = os.path.join(ROOT_D, "build")
LIB_D = os.path.join(ROOT_D, "lib")
EXO_D = os.path.join(UTL_D, "exojac")
MATLIB = os.path.join(ROOT_D, "materials")

# --- OUTPUT DATABASE FILE
F_EVALDB = "mml-evaldb.xml"
F_PRODUCT = "product.py"

# --- ENVIRONMENT VARIABLES
PATH = os.getenv("PATH").split(os.pathsep)
if TLS_D not in PATH:
    PATH.insert(0, TLS_D)
MAT_LIB_DIRS = [MATLIB]
TEST_DIRS = [TEST_D]

# --- APPLY USER CONFIGURATIONS
if os.getenv("MMLMTLS"):
    cfgswitch_and_warn()
cfg = cfgparse()
MAT_LIB_DIRS.extend(cfg.user_mats)
TEST_DIRS.extend(cfg.user_tests)
TEST_CONS_WIDTH = 80

FFLAGS = [x for x in os.getenv("FFLAGS", "").split() if x.split()]
FC = which(os.getenv("FC", "gfortran"))

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
