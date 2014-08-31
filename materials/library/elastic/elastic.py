import numpy as np

from materials.material import Material
from core.mmlio import Error1, log_error, log_message
try:
    import lib.elastic as elastic
except ImportError:
    elastic = None

class Elastic(Material):
    name = "elastic"
    param_names = ["K", "G"]
    constant_j = True

    def setup(self):
        """Set up the Elastic material

        """
        if elastic is None:
            raise Error1("elastic model not imported")
        elastic.elastic_check(self.params, log_error, log_message)
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
                                     log_error, log_message)
        return stress, xtra, self.constant_jacobian
