from utils.constants import SET_AT_RUNTIME
from utils.errors import ModelNotImportedError
from materials.aba.abamat import AbaqusMaterial
try: import lib.uhyper as uhyper
except ImportError: uhyper = None


class UHyper(AbaqusMaterial):
    """Constitutive model class for the umat model"""

    def __init__(self):
        self.name = "uhyper"
        self.param_names = SET_AT_RUNTIME

    def check_import(self):
        if uhyper is None:
            raise ModelNotImportedError("uhyper")

    def update_state_umat(self, stress, statev, ddsdde,
            sse, spd, scd, rpl, ddsddt, drplde, drpldt, stran, dstran,
            time, dtime, temp, dtemp, predef, dpred, cmname, ndi, nshr,
            nxtra, params, coords, drot, pnewdt, celent, dfgrd0,
            dfgrd1, noel, npt, layer, kspt, kstep, kinc):
        """update the material state"""
        uhyper.umat(stress, statev, ddsdde,
            sse, spd, scd, rpl, ddsddt, drplde, drpldt, stran, dstran,
            time, dtime, temp, dtemp, predef, dpred, cmname, ndi, nshr,
        nxtra, params, coords, drot, pnewdt, celent, dfgrd0,
            dfgrd1, noel, npt, layer, kspt, kstep, kinc, self.logger.error,
            self.logger.write, self.logger.warn)
        return stress, statev, ddsdde
