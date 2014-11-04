from materials.product import ABA_IO_F90, ABA_UMAT_PYF
from core.material import AbaqusMaterial
from utils.constants import SET_AT_RUNTIME
from core.logger import logmes, logwrn, bombed

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
        import lib.umat as mat

    def update_state_umat(self, stress, statev, ddsdde,
            sse, spd, scd, rpl, ddsddt, drplde, drpldt, stran, dstran,
            time, dtime, temp, dtemp, predef, dpred, cmname, ndi, nshr,
            nxtra, params, coords, drot, pnewdt, celent, dfgrd0,
            dfgrd1, noel, npt, layer, kspt, kstep, kinc):
        """update the material state"""
        lid = self.logger.logger_id
        extra = (len(ddsddt), len(params), (lid,), (lid,), (lid,))
        mat.umat(stress, statev, ddsdde,
            sse, spd, scd, rpl, ddsddt, drplde, drpldt, stran, dstran,
            time, dtime, temp, dtemp, predef, dpred, cmname, ndi, nshr,
            nxtra, params, coords, drot, pnewdt, celent, dfgrd0,
            dfgrd1, noel, npt, layer, kspt, kstep, kinc, logmes, logwrn,
            bombed, **extra)
        return stress, statev, ddsdde
