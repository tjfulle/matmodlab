import os
from core.product import MAT_D
from core.material import MaterialModel
from utils.errors import ModelNotImportedError
from utils.constants import VOIGHT
from materials.completion import EC_BULK, EC_SHEAR, DP_A, DP_B
from core.logger import logmes, logwrn, bombed

d = os.path.join(MAT_D, "src")
f1 = os.path.join(d, "plastic.f90")
f2 = os.path.join(d, "plastic.pyf")

class Plastic(MaterialModel):
    name = "plastic"
    source_files = [f1, f2]

    def __init__(self):
        self.param_names = ["K", "G", "A1", "A4"]
        self.prop_names = [EC_BULK, EC_SHEAR, DP_A, DP_B]

    def setup(self):
        """Set up the Plastic material

        """
        global mat
        try:
            import lib.plastic as mat
        except ImportError:
            raise ModelNotImportedError("plastic")
        lid = self.logger.logger_id
        extra = ((lid,), (lid,), (lid,))
        mat.plastic_check(self.params, logmes, logwrn, bombed, *extra)

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
        lid = self.logger.logger_id
        extra = ((lid,), (lid,), (lid,))
        d = d / VOIGHT
        mat.plastic_update_state(dtime, self.params, d, stress,
                                 logmes, logwrn, bombed, *extra)
        return stress, xtra, None
