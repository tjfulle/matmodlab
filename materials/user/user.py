import sys
import numpy as np
from utils.constants import SET_AT_RUNTIME
from materials.aba.abamat import AbaqusMaterial
from utils.errors import ModelNotImportedError
try: import lib.user as user
except ImportError: user = None

class UserMat(AbaqusMaterial):
    """Constitutive model class for the user model"""
    def __init__(self):
        self.name = "user"
        self.param_names = SET_AT_RUNTIME

    def check_import(self):
        if user is None:
            raise ModelNotImportedError("user")

    def update_state_umat(self, stress, statev, ddsdde,
            sse, spd, scd, rpl, ddsddt, drplde, drpldt, stran, dstran,
            time, dtime, temp, dtemp, predef, dpred, cmname, ndi, nshr,
            nxtra, params, coords, drot, pnewdt, celent, dfgrd0,
            dfgrd1, noel, npt, layer, kspt, kstep, kinc):
        """update the material state"""
        user.user_mat(stress, statev, ddsdde,
            stran, dstran, time, dtime, temp, dtemp,
            nxtra, params, dfgrd0, dfgrd1, self.logger.error,
            self.logger.write, self.logger.warn)
        return stress, statev, ddsdde
