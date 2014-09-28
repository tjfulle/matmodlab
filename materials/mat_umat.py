from materials.product import ABA_IO_F90, ABA_UMAT_PYF
from core.material import AbaqusMaterial
from utils.constants import SET_AT_RUNTIME
from utils.errors import ModelNotImportedError
mat = None


class UMat(AbaqusMaterial):
    """Constitutive model class for the umat model"""
    name = "umat"
    aux_files = [ABA_IO_F90, ABA_UMAT_PYF]
    lapack = "lite"

    def __init__(self):
        self.param_names = SET_AT_RUNTIME

    def import_model(self):
        global mat
        try:
            import lib.umat as mat
        except ImportError:
            raise ModelNotImportedError("umat")

    def update_state_umat(self, stress, statev, ddsdde,
            sse, spd, scd, rpl, ddsddt, drplde, drpldt, stran, dstran,
            time, dtime, temp, dtemp, predef, dpred, cmname, ndi, nshr,
            nxtra, params, coords, drot, pnewdt, celent, dfgrd0,
            dfgrd1, noel, npt, layer, kspt, kstep, kinc):
        """update the material state"""
        mat.umat(stress, statev, ddsdde,
            sse, spd, scd, rpl, ddsddt, drplde, drpldt, stran, dstran,
            time, dtime, temp, dtemp, predef, dpred, cmname, ndi, nshr,
            nxtra, params[:-1], coords, drot, pnewdt, celent, dfgrd0,
            dfgrd1, noel, npt, layer, kspt, kstep, kinc, self.logger.write,
            self.logger.warn, self.logger.raise_error)
        return stress, statev, ddsdde
