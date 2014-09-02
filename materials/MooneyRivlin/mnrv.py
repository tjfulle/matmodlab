import sys
import numpy as np

from core.material import MaterialModel
from utils.mmlio import log_message, log_error
from utils.errors import ModelNotImportedError
import utils.mmlabpack as mmlabpack
try:
    import lib.mnrv as mnrv
except ImportError:
    mnrv = None


class MooneyRivlin(MaterialModel):
    """Constitutive model class for the Mooney-Rivlin model

    """
    name = "mnrv"
    param_names = ["C10", "C01", "NU", "T0", "MC10", "MC01"]

    def setup(self):
        """Set up the domain Mooney-Rivlin materia

        """
        if mnrv is None:
            raise ModelNotImportedError("mnrv model not imported")

        mnrv.mnrvcp(self.params, log_error, log_message)
        smod = 2. * (self.params["C10"] + self.params["C01"])
        nu = self.params["NU"]
        bmod = 2. * smod * (1.+ nu) / 3. / (1 - 2. * nu)

        self.bulk_modulus = bmod
        self.shear_modulus = smod

        # register extra variables
        nxtra, keya, xinit = mnrv.mnrvxv(self.params, log_error, log_message)
        self.register_xtra_variables(keya, xinit, mig=True)

        Rij = np.reshape(np.eye(3), (9,))
        Vij = mmlabpack.asarray(np.eye(3), 6)
        T = 298.
        v = np.arange(6, dtype=np.int)
        return

    def update_state(self, time, dtime, temp, dtemp, energy, rho, F0, F,
        stran, d, elec_field, user_field, stress, xtra, **kwargs):
        """ update the material state based on current state and stretch """

        Fij = np.reshape(F, (3, 3))

        # left stretch
        Vij = mmlabpack.sqrtm(np.dot(Fij, Fij.T))

        # rotation
        Rij = np.reshape(np.dot(mmlabpack.inv(Vij), Fij), (9,))
        Vij = mmlabpack.asarray(Vij, 6)

        # temperature
        T = 298.

        sig = mnrv.mnrvus(self.params, Rij, Vij, T, xtra, log_error, log_message)
        ddsdde = mnrv.mnrvjm(self.params, Vij, T, xtra, log_error, log_message)

        return np.reshape(sig, (6,)), np.reshape(xtra, (self.nxtra,)), ddsdde

    def set_constant_jacobian(self):
        Vij = mmlabpack.asarray(np.eye(3), 6)
        T0 = 298. if not self.params["T0"] else self.params["T0"]
        self.J0 = mnrv.mnrvjm(self.params, Vij, T0, self.xinit,
                              log_error, log_message)
