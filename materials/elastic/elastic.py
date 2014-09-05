import numpy as np

from core.material import MaterialModel
from utils.errors import ModelNotImportedError
try:
    import lib.elastic as elastic
except ImportError:
    elastic = None

class Elastic(MaterialModel):
    name = "elastic"
    param_names = ["K", "G"]
    constant_j = True
    def setup(self):
        """Set up the Elastic material

        """
        if elastic is None:
            raise ModelNotImportedError("elastic")
        elastic.elastic_check(self.params, self.logger.error, self.logger.write)
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
        elastic.elastic_update_state(dtime, self.params, d, stress,
                                     self.logger.error, self.logger.write)
        return stress, xtra, self.constant_jacobian
