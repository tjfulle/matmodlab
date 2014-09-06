import numpy as np

from core.material import MaterialModel
from utils.errors import ModelNotImportedError
try: import lib.elastic as mat
except ImportError: mat=None

class Elastic(MaterialModel):

    def __init__(self):
        self.name = "elastic"
        self.param_names = ["K", "G"]

    def setup(self):
        """Set up the Elastic material

        """
        if mat is None:
            raise ModelNotImportedError("elastic")
        mat.elastic_check(self.params, self.logger.error, self.logger.write)
        self.bulk_modulus = self.params["K"]
        self.shear_modulus = self.params["G"]
        self.use_constant_jacobian = True

    def update_state(self, time, dtime, temp, dtemp, energy, rho, F0, F,
        stran, d, elec_field, user_field, stress, xtra, **kwargs):
        """Compute updated stress given strain increment

        Parameters
        ----------
        dtime : float
            Time step

        d : array_like
            Deformation rate

        stress : array_like
            Stress at beginning of step

        xtra : array_like
            Extra variables

        Returns
        -------
        S : array_like
            Updated stress

        xtra : array_like
            Updated extra variables

        """
        mat.elastic_update_state(dtime, self.params, d, stress,
                                 self.logger.error, self.logger.write)
        return stress, xtra, self.constant_jacobian
