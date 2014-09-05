import sys
import numpy as np
import utils.mmlabpack as mmlabpack
from utils.errors import ModelNotImportedError
from materials.aba.abamat import AbaqusMaterial
try:
    import lib.uanisohyper as umat
except ImportError:
    umat = None


class UAnisoHyper(AbaqusMaterial):
    """Constitutive model class for the uanisohyper model"""
    name = "uanisohyper_inv"
    def check_import(self):
        if umat is None:
            raise ModelNotImportedError("uanisohyper_inv")
    def update_state_umat(self, Ainv, zeta, nfibers, temp, noel,
                          cmname, incmpflag, ihybflag, statev,
                          fieldv, fieldvinc):
        return umat.uanisohyper_inv(Ainv, zeta, nfibers, temp, noel, cmname,
                   incmpflag, ihybflag, statev, fieldv, fieldvinc,
                   self.params[:-4], self.logger.error, self.logger.write,
                   self.logger.warn)
