from materials.product import ABA_IO_F90, ABA_UMAT_PYF
from core.material import AbaqusMaterial
from utils.constants import SET_AT_RUNTIME
from utils.errors import ModelNotImportedError
try: import lib.umat as umat
except ImportError: umat = None


class UMat(AbaqusMaterial):
    """Constitutive model class for the umat model"""

    def __init__(self):
        self.name = "umat"
        self.param_names = SET_AT_RUNTIME
        self.aux_files = [ABA_IO_F90, ABA_UMAT_PYF]
        self.lapack = "lite"
        self.aba_model = True

    def check_import(self):
        if umat is None:
            raise ModelNotImportedError("umat")

    def update_state_umat(self, stress, statev, ddsdde,
            sse, spd, scd, rpl, ddsddt, drplde, drpldt, stran, dstran,
            time, dtime, temp, dtemp, predef, dpred, cmname, ndi, nshr,
            nxtra, params, coords, drot, pnewdt, celent, dfgrd0,
            dfgrd1, noel, npt, layer, kspt, kstep, kinc):
        """update the material state"""
        umat.umat(stress, statev, ddsdde,
            sse, spd, scd, rpl, ddsddt, drplde, drpldt, stran, dstran,
            time, dtime, temp, dtemp, predef, dpred, cmname, ndi, nshr,
            nxtra, params[:-1], coords, drot, pnewdt, celent, dfgrd0,
            dfgrd1, noel, npt, layer, kspt, kstep, kinc, self.logger.error,
            self.logger.write, self.logger.warn)
        return stress, statev, ddsdde
