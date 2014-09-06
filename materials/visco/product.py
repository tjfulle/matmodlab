import os
from utils.fortran.product import FIO

def fortran_libraries():
    _d = os.path.dirname(os.path.realpath(__file__))
    libs = {}

    visco_f90 = os.path.join(_d, "visco.f90")
    visco_pyf = os.path.join(_d, "visco.pyf")
    libs["visco"] = {"source_files": [visco_f90, visco_pyf, FIO]}

    thermo_f90 = os.path.join(_d, "thermomech.f90")
    thermo_pyf = os.path.join(_d, "thermomech.pyf")
    libs["thermomech"] = {"source_files": [thermo_f90, thermo_pyf, FIO]}

    return libs
