import numpy as np

from materials.material import Material
from core.mmlio import Error1, log_error, log_message
try:
    import lib.plastic as plastic
except ImportError:
    plastic = None

class Plastic(Material):
    name = "plastic"
    param_names = ["K", "G", "A1", "A4"]
    def setup(self):
        """Set up the Plastic material

        """
        if plastic is None:
            raise Error1("plastic model not imported")
        plastic.plastic_check(self.params, log_error, log_message)
        K, G, = self.params
        self.bulk_modulus = K
        self.shear_modulus = G

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
        plastic.plastic_update_state(dtime, self.params, d, stress,
                                     log_error, log_message)
        return stress, xtra, self.constant_jacobian
