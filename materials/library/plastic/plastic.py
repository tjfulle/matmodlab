import numpy as np

from materials._material import Material
from core.io import Error1, log_error, log_message
try:
    import lib.plastic as plastic
except ImportError:
    plastic = None

class Plastic(Material):
    name = "plastic"
    param_names = ["K", "G", "A1", "A4"]
    def setup(self, params):
        """Set up the Elastic material

        Parameters
        ----------
        params : ndarray
            Material parameters

        """
        if plastic is None:
            raise Error1("plastic model not imported")
        plastic.plastic_check(params, log_error, log_message)
        K, G, = params
        self.set_param_vals(params)
        self.bulk_modulus = K
        self.shear_modulus = G

    def update_state(self, dt, d, stress, xtra, *args):
        """Compute updated stress given strain increment

        Parameters
        ----------
        dt : float
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
        plastic.plastic_update_state(dt, self._param_vals, d, stress,
                                     log_error, log_message)
        return stress, xtra

    def jacobian(self, dt, d, stress, xtra, v, *args):
        """Return the constant stiffness
        dt : float
            time step

        d : array_like
            Deformation rate

        stress : array_like
            Stress at beginning of step

        """
        return self.constant_jacobian(v)
