import numpy as np
from core.material import AbaqusMaterial
from utils.constants import SET_AT_RUNTIME
from utils.errors import ModelNotImportedError
from materials.product import (ABA_IO_F90, DGPADM_F, ABA_TENSALG_F90,
                               ABA_UHYPER_PYF, ABA_UHYPER_JAC_F90)
mat = None

class UHyper(AbaqusMaterial):
    """Constitutive model class for the umat model"""
    name = "uhyper"
    lapack = "lite"
    aux_files = [ABA_IO_F90, DGPADM_F, ABA_TENSALG_F90,
                 ABA_UHYPER_PYF, ABA_UHYPER_JAC_F90]

    def __init__(self):
        self.param_names = SET_AT_RUNTIME

    def import_model(self):
        global mat
        try:
            import lib.uhyper as mat
        except ImportError:
            raise ModelNotImportedError("uhyper")

    def update_state_umat(self, stress, statev, ddsdde,
            sse, spd, scd, rpl, ddsddt, drplde, drpldt, stran, dstran,
            time, dtime, temp, dtemp, predef, dpred, cmname, ndi, nshr,
            nxtra, params, coords, drot, pnewdt, celent, dfgrd0,
            dfgrd1, noel, npt, layer, kspt, kstep, kinc):
        """update the material state"""
        comm = (self.logger.write, self.logger.warn, self.logger.raise_error)
        ddsdde = np.zeros((6,6), order="F")
        mat.umat(stress, statev, ddsdde,
            sse, spd, scd, rpl, ddsddt, drplde, drpldt, stran, dstran,
            time, dtime, temp, dtemp, predef, dpred, cmname, ndi, nshr,
            nxtra, params, coords, drot, pnewdt, celent, dfgrd0,
            dfgrd1, noel, npt, layer, kspt, kstep, kinc, *comm)
        return stress, statev, ddsdde
