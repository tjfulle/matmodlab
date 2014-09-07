import os
from utils.fortran.product import DGPADM_F
from core.product import MATLIB
D = os.path.join(MATLIB, "src")

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
