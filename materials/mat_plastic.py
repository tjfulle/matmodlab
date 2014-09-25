import os
from core.product import MAT_D
from core.material import MaterialModel
from utils.errors import ModelNotImportedError
from materials.completion import EC_BULK, EC_SHEAR, DP_A, DP_B
try: import lib.plastic as mat
except ImportError: mat = None

class Plastic(MaterialModel):

    def __init__(self):
        self.name = "plastic"
        self.param_names = ["K", "G", "A1", "A4"]
        self.prop_names = [EC_BULK, EC_SHEAR, DP_A, DP_B]
        d = os.path.join(MAT_D, "src")
        f1 = os.path.join(d, "plastic.f90")
        f2 = os.path.join(d, "plastic.pyf")
        self.source_files = [f1, f2]

    def setup(self):
        """Set up the Plastic material

        """
        if mat is None:
            raise ModelNotImportedError("plastic")
        comm = (self.logger.write, self.logger.warn, self.logger.raise_error)
        mat.plastic_check(self.params, *comm)

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
        comm = (self.logger.write, self.logger.warn, self.logger.raise_error)
        mat.plastic_update_state(dtime, self.params, d, stress, *comm)
        return stress, xtra, None
