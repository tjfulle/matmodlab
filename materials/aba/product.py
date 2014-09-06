import os
from utils.fortran.product import DGPADM_F

D = os.path.dirname(os.path.realpath(__file__))

# Auxiliary files
ABA_UANISOHYPER_JAC_F90 = os.path.join(D, "uanisohyper_jac.f90")
ABA_UHYPER_JAC_F90 = os.path.join(D, "uhyper_jac.f90")
ABA_TENSALG_F90 = os.path.join(D, "tensalg.f90")
ABA_IO_F90 = os.path.join(D, "aba_io.f90")

# Signature files
ABA_UANISOHYPER_PYF = os.path.join(D, "uanisohyper.pyf")
ABA_UHYPER_PYF = os.path.join(D, "uhyper.pyf")
ABA_UMAT_PYF = os.path.join(D, "umat.pyf")

# Interface files
ABA_UANISOHYPER_PY = os.path.join(D, "uanisohyper.py")
ABA_UHYPER_PY = os.path.join(D, "uhyper.py")
ABA_UMAT_PY = os.path.join(D, "umat.py")

def material_libraries():
    # Abaqus materials
    return {"umat": {"source_files": [ABA_IO_F90, ABA_UMAT_PYF],
                     "interface": ABA_UMAT_PY, "class": "UMat"},
            "uanisohyper": {"source_files": [ABA_IO_F90, DGPADM_F,
                                             ABA_TENSALG_F90,
                                             ABA_UANISOHYPER_PYF,
                                             ABA_UANISOHYPER_JAC_F90],
                            "interface": ABA_UANISOHYPER_PY,
                            "class": "UAnisoHyper"},
            "uhyper": {"source_files": [ABA_IO_F90, DGPADM_F, ABA_TENSALG_F90,
                                        ABA_UHYPER_PYF, ABA_UHYPER_JAC_F90],
                       "interface": ABA_UHYPER_PY, "class": "UHyper"}}
