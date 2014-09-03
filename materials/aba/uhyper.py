import sys
import numpy as np
from materials.aba.abamat import AbaqusMaterial
from utils.errors import ModelNotImportedError
import utils.mmlabpack as mmlabpack
try:
    import lib.uhyper as uhyper
except ImportError:
    uhyper = None


class UHyper(AbaqusMaterial):
    """Constitutive model class for the umat model"""
    name = "uhyper"
    def check_import(self):
        if uhyper is None:
            raise ModelNotImportedError("umat")
    def update_state_umat(self, stress, statev, ddsdde,
            sse, spd, scd, rpl, ddsddt, drplde, drpldt, stran, dstran,
            time, dtime, temp, dtemp, predef, dpred, cmname, ndi, nshr,
            nxtra, params, coords, drot, pnewdt, celent, dfgrd0,
            dfgrd1, noel, npt, layer, kspt, kstep, kinc, logger):
        """update the material state"""
        uhyper.umat(stress, statev, ddsdde,
            sse, spd, scd, rpl, ddsddt, drplde, drpldt, stran, dstran,
            time, dtime, temp, dtemp, predef, dpred, cmname, ndi, nshr,
        nxtra, params, coords, drot, pnewdt, celent, dfgrd0,
            dfgrd1, noel, npt, layer, kspt, kstep, kinc, logger.error,
            logger.write, logger.warn)
        return stress, statev, ddsdde
