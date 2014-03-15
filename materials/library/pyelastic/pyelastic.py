import numpy as np

from materials.material import Material
from core.mmlio import Error1, log_error, log_message

class PyElastic(Material):
    name = "pyelastic"
    param_names = ["K", "G"]

    def setup(self):
        """Set up the Elastic material

        """
        # Check inputs
        K, G, = self.params
        if K <= 0.0: log_error = "Bulk modulus K must be positive"
        if G <= 0.0: log_error = "Shear modulus G must be positive"
        nu = (3.0 * K - 2.0 * G) / (6.0 * K + 2.0 * G)
        if nu > 0.5: log_error = "Poisson's ratio > .5"
        if nu < -1.0: log_error = "Poisson's ratio < -1."
        if nu < 0.0: log_message = "#---- WARNING: negative Poisson's ratio"

        self.bulk_modulus = K
        self.shear_modulus = G

    def update_state(self, dt, d, stress, xtra, *args, **kwargs):
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
        de = d * dt

        iso = de[:3].sum() / 3.0 * np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
        dev = de - iso

        stress = stress + 3.0 * self.bulk_modulus * iso + 2.0 * self.shear_modulus * dev
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
