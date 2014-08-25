import os
from materials.materialdb import _Material
from utils.fortran.info import DGPADM_F
D = os.path.dirname(os.path.realpath(__file__))

# Auxiliary files
ABA_IO_F90 = os.path.join(D, "aba_io.f90")
ABA_UHYPER_JAC_F90 = os.path.join(D, "uhyper_jac.f90")
ABA_NUMBERS_F90 = os.path.join(D, "numbers.f90")
ABA_TENSALG_F90 = os.path.join(D, "tensalg.f90")

# Signature files
ABA_UANISOHYPER_INV_PYF = os.path.join(D, "uanisohyper_inv.pyf")
ABA_UHYPER_PYF = os.path.join(D, "uhyper.pyf")
ABA_UMAT_PYF = os.path.join(D, "umat.pyf")

# Interface files
ABA_UANISOHYPER_INV_PY = os.path.join(D, "uanisohyper_inv.py")
ABA_UHYPER_PY = os.path.join(D, "uhyper.py")
ABA_UMAT_PY = os.path.join(D, "umat.py")

# Abaqus materials
ABA_MATS = {}

kwargs = {"source_files": [ABA_IO_F90, ABA_UMAT_PYF],
          "interface_file": ABA_UMAT_PY,
          "class": "UMat", "type": "abaqus_umat"}
ABA_MATS["umat"] = _Material("umat", **kwargs)

kwargs = {"source_files": [ABA_IO_F90, ABA_UANISOHYPER_INV_PYF],
          "interface_file": ABA_UANISOHYPER_INV_PY,
          "class": "UAnisoHyper", "type": "abaqus_uanioshyper_inv"}
ABA_MATS["uanisohyper_inv"] = _Material("umat", **kwargs)

kwargs = {"source_files": [ABA_IO_F90, ABA_NUMBERS_F90, DGPADM_F, ABA_TENSALG_F90,
                           ABA_UHYPER_PYF, ABA_UHYPER_JAC_F90],
          "interface_file": ABA_UHYPER_PYF,
          "class": "UHyper", "type": "abaqus_uhyper"}
ABA_MATS["uhyper"] = _Material("uhyper", **kwargs)
