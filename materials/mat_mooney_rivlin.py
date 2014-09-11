import os
import numpy as np
from core.product import MATLIB
import utils.mmlabpack as mmlabpack
from core.material import MaterialModel
from utils.errors import ModelNotImportedError
try: import lib.mooney_rivlin as mat
except ImportError: mat = None


class MooneyRivlin(MaterialModel):
    """Constitutive model class for the Mooney-Rivlin model

    """

    def __init__(self):
        self.name = "mooney_rivlin"
        self.param_names = ["C10", "C01", "NU", "T0", "MC10", "MC01"]
        d = os.path.join(MATLIB, "src")
        f1 = os.path.join(d, "mooney_rivlin.f90")
        f2 = os.path.join(d, "mooney_rivlin.pyf")
        self.source_files = [f1, f2]
        self.lapack = "lite"

    def setup(self):
        """Set up the domain Mooney-Rivlin materia

        """
        if mat is None:
            raise ModelNotImportedError("mooney_rivlin")
        smod = 2. * (self.params["C10"] + self.params["C01"])
        nu = self.params["NU"]
        bmod = 2. * smod * (1.+ nu) / 3. / (1 - 2. * nu)
        self.bulk_modulus = bmod
        self.shear_modulus = smod
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
