import os
import sys
from os.path import join, dirname, realpath, isdir

# ------------------------------------------------ PROJECT WIDE CONSTANTS --- #
VERSION = (3, 0, 1)

PLATFORM = sys.platform
PYEXE = realpath(sys.executable)

ROOT_D = dirname(realpath(__file__))
MMD_D = join(ROOT_D, "mmd")
assert isdir(MMD_D)
VIZ_D = join(ROOT_D, "viz")
UTL_D = join(ROOT_D, "utils")
BIN_D = join(ROOT_D, "bin")
TEST_D = join(ROOT_D, "tests")
PKG_D = join(ROOT_D, "lib")
LIB_D = join(ROOT_D, "lib")
MAT_D = join(ROOT_D, "materials")
BLD_D = join(LIB_D, "build")
EXMPL_D = join(ROOT_D, "examples")
TUT_D = join(ROOT_D, "tutorial")
IPY_D = join(ROOT_D, "ipynb")

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

""".format(".".join("{0}".format(i) for i in VERSION))
