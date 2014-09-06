from utils.constants import SET_AT_RUNTIME
from utils.errors import ModelNotImportedError
from materials.aba.abamat import AbaqusMaterial
try: import lib.uanisohyper as umat
except ImportError: umat = None


class UAnisoHyper(AbaqusMaterial):
    """Constitutive model class for the uanisohyper model"""
    def __init__(self):
        self.name = "uanisohyper_inv"
        self.param_names = SET_AT_RUNTIME

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
