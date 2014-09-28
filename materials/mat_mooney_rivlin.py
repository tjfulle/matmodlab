import os
import numpy as np
from core.product import MAT_D
import utils.mmlabpack as mmlabpack
from core.material import MaterialModel
from utils.errors import ModelNotImportedError
from materials.completion import EC_C10, EC_C01, EC_NU, TEMP0
try: import lib.mooney_rivlin as mat
except ImportError: mat = None

d = os.path.join(MAT_D, "src")
f1 = os.path.join(d, "mooney_rivlin.f90")
f2 = os.path.join(d, "mooney_rivlin.pyf")

class MooneyRivlin(MaterialModel):
    """Constitutive model class for the Mooney-Rivlin model

    """
    name = "mooney_rivlin"
    source_files = [f1, f2]
    lapack = "lite"
    def __init__(self):
        self.param_names = ["C10", "C01", "NU", "T0", "MC10", "MC01"]
        self.prop_names = [EC_C10, EC_C01, EC_NU, TEMP0, None, None]

    def setup(self):
        """Set up the domain Mooney-Rivlin materia

        """
        if mat is None:
            raise ModelNotImportedError("mooney_rivlin")
        return

    #    def set_constant_jacobian(self):
        #     Vij = mmlabpack.asarray(np.eye(3), 6)
        #     T0 = 298. if not self.params["T0"] else self.params["T0"]
        #     Rij = np.eye(3)
        #     comm = (self.logger.error, self.logger.write)
        #     _, ddsdde = mat.mnrv_mat(self.params, Rij, Vij, *comm)
        #     return ddsdde

    def update_state(self, time, dtime, temp, dtemp, energy, rho, F0, F,
        stran, d, elec_field, user_field, stress, xtra, **kwargs):
        """ update the material state based on current state and stretch """

        Fij = np.reshape(F, (3, 3))
        Vij = mmlabpack.sqrtm(np.dot(Fij, Fij.T))
        Rij = np.reshape(np.dot(mmlabpack.inv(Vij), Fij), (9,))
        Vij = mmlabpack.asarray(Vij, 6)
        comm = (self.logger.write, self.logger.warn, self.logger.raise_error)
        sig, ddsdde = mat.mnrv_mat(self.params, Rij, Vij, *comm)

        return np.reshape(sig, (6,)), np.reshape(xtra, (self.nxtra,)), ddsdde
