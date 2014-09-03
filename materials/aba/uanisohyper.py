import sys
import numpy as np
from materials.aba.abamat import AbaqusMaterial
from utils.conlog import Error1, log_message, log_error, log_warning
from utils.errors import ModelNotImportedError
import utils.mmlabpack as mmlabpack
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
                                    incmpflag, ihybflag, statev, fieldv,
                                    fieldvinc, self.params[:-4],
                                    log_error, log_message, log_warning)
