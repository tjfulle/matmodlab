import os
from core.product import MAT_D
from core.material import AbaqusMaterial
from utils.constants import SET_AT_RUNTIME
from utils.errors import ModelNotImportedError
from core.logger import logmes, logwrn, bombed

mat = None

d = os.path.join(MAT_D, "src")
f = os.path.join(d, "user.pyf")
class UserMat(AbaqusMaterial):
    """Constitutive model class for the user model"""
    name = "user"
    aux_files = [f]

    def __init__(self):
        self.param_names = SET_AT_RUNTIME

    def import_model(self):
        global mat
        import lib.user as mat

    def update_state_umat(self, stress, statev, ddsdde,
            sse, spd, scd, rpl, ddsddt, drplde, drpldt, stran, dstran,
            time, dtime, temp, dtemp, predef, dpred, cmname, ndi, nshr,
            nxtra, params, coords, drot, pnewdt, celent, dfgrd0,
            dfgrd1, noel, npt, layer, kspt, kstep, kinc):
        """update the material state"""
        lid = self.logger.logger_id
        extra = (len(params), (lid,), (lid,), (lid,))
        mat.user_mat(stress, statev, ddsdde,
            stran, dstran, time, dtime, temp, dtemp,
            nxtra, params, dfgrd0, dfgrd1, logmes, logwrn, bombed, *extra)
        return stress, statev, ddsdde
