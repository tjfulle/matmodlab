from core.material import AbaqusMaterial
from utils.constants import SET_AT_RUNTIME
from utils.errors import ModelNotImportedError
from materials.product import (ABA_IO_F90, DGPADM_F, ABA_TENSALG_F90,
                               ABA_UANISOHYPER_PYF, ABA_UANISOHYPER_JAC_F90)
mat = None


class UAnisoHyper(AbaqusMaterial):
    """Constitutive model class for the uanisohyper model"""
    name = "uanisohyper_inv"
    aux_files = [ABA_IO_F90, DGPADM_F, ABA_TENSALG_F90,
                 ABA_UANISOHYPER_PYF, ABA_UANISOHYPER_JAC_F90]

    def __init__(self):
        self.param_names = SET_AT_RUNTIME

    def import_model(self):
        global mat
        try:
            import lib.uanisohyper_inv as mat
        except ImportError:
            raise ModelNotImportedError("uanisohyper_inv")

    def update_state_umat(self, Ainv, zeta, nfibers, temp, noel,
                          cmname, incmpflag, ihybflag, statev,
                          fieldv, fieldvinc):
        comm = (self.logger.write, self.logger.warn, self.logger.raise_error)
        return mat.uanisohyper_inv(Ainv, zeta, nfibers, temp, noel, cmname,
                   incmpflag, ihybflag, statev, fieldv, fieldvinc,
                   self.params[:-4], *comm)
