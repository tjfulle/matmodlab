import os
import sys
import shutil
from distutils.spawn import find_executable as which

__version__ = (1, 2, 0)
ROOT_D = os.path.dirname(os.path.realpath(__file__))
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
