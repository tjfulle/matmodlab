import os
from utils.fortran.product import DGPADM_F, FIO
from core.product import MAT_D
D = os.path.join(MAT_D, "src")

# Auxiliary files
ABA_UANISOHYPER_JAC_F90 = os.path.join(D, "uanisohyper_inv_jac.f90")
ABA_UHYPER_JAC_F90 = os.path.join(D, "uhyper_jac.f90")
ABA_TENSALG_F90 = os.path.join(D, "tensalg.f90")
ABA_IO_F90 = os.path.join(D, "aba_io.f90")

# Signature files
ABA_UANISOHYPER_PYF = os.path.join(D, "uanisohyper_inv.pyf")
ABA_UHYPER_PYF = os.path.join(D, "uhyper.pyf")
ABA_UMAT_PYF = os.path.join(D, "umat.pyf")

ABA_MATS = ["umat", "uhyper", "uanisohyper_inv"]
USER_MAT = ["user"]


def fortran_libraries():
    libs = {}

    visco_f90 = os.path.join(D, "visco.f90")
    visco_pyf = os.path.join(D, "visco.pyf")
    libs["visco"] = {"source_files": [visco_f90, visco_pyf, FIO]}

    expansion_f90 = os.path.join(D, "expansion.f90")
    expansion_pyf = os.path.join(D, "expansion.pyf")
    libs["expansion"] = {"source_files": [expansion_f90, expansion_pyf],
                         "mmlabpack": True}

    return libs
