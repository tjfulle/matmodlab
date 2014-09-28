import os
import sys
import argparse
import ConfigParser
from distutils.spawn import find_executable as which


# ------------------------------------------------ PROJECT WIDE CONSTANTS --- #
__version__ = (2, 0, 0)

PLATFORM = sys.platform
PYEXE = os.path.realpath(sys.executable)

ROOT_D = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
CORE_D = os.path.join(ROOT_D, "core")
assert os.path.isdir(CORE_D)
VIZ_D = os.path.join(ROOT_D, "viz")
UTL_D = os.path.join(ROOT_D, "utils")
BIN_D = os.path.join(ROOT_D, "bin")
TEST_D = os.path.join(ROOT_D, "tests")
PKG_D = os.path.join(ROOT_D, "lib")
LIB_D = os.path.join(ROOT_D, "lib")
EXO_D = os.path.join(UTL_D, "exojac")
MAT_D = os.path.join(ROOT_D, "materials")
BLD_D = os.path.join(LIB_D, "build")

# --- OUTPUT DATABASE FILE
F_EVALDB = "mml-evaldb.xml"
F_PRODUCT = "product.py"

# --- MATERIAL AND TEST SEARCH DIRECTORIES
MAT_LIB_DIRS = [MAT_D]
TEST_DIRS = [os.path.join(TEST_D, d) for d in os.listdir(TEST_D)
             if os.path.isdir(os.path.join(TEST_D, d))]

# User configuration
f = "matmodlabrc"
if os.path.isfile(f):
    RCFILE = os.path.realpath(f)
else:
    RCFILE = os.getenv("MATMODLABRC") or os.path.expanduser("~/.{0}".format(f))
p = argparse.ArgumentParser(add_help=False)
p.add_argument("-E", action="store_true", default=False,
    help="Do not use matmodlabrc configuration file [default: False]")
_a, sys.argv[1:] = p.parse_known_args()
SUPRESS_USER_ENV = _a.E

# OTHER CONSTANTS
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
