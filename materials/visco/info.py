import os
from mml import ROOT_D
from utils.fortran.info import FIO

_D = os.path.dirname(os.path.realpath(__file__))
LIBS = {}

VISCO_F90 = os.path.join(_D, "visco.f90")
VISCO_PYF = os.path.join(_D, "visco.pyf")
LIBS["visco"] = {"source_files": [VISCO_F90, VISCO_PYF, FIO]}

THERMO_F90 = os.path.join(_D, "thermomech.f90")
THERMO_PYF = os.path.join(_D, "thermomech.pyf")
LIBS["thermomech"] = {"source_files": [THERMO_F90, THERMO_PYF, FIO]}

def fortran_libraries():
    return LIBS
