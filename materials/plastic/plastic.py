import numpy as np

from materials.material import MaterialModel
from utils.errors import ModelNotImportedError
try: import lib.plastic as plastic
except ImportError: plastic = None

class Plastic(MaterialModel):

    def __init__(self):
        self.name = "plastic"
        self.param_names = ["K", "G", "A1", "A4"]
        self.constant_j = True

    def setup(self):
        """Set up the Plastic material

        """
        if plastic is None:
            raise ModelNotImportedError("plastic model not imported")
        plastic.plastic_check(self.params, self.logger.error, self.logger.write)
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
                                     self.logger.error, self.logger.write)
        return stress, xtra, self.constant_jacobian
