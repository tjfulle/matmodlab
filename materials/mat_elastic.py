import os
import sys
from core.product import MAT_D
from core.material import MaterialModel
from core.logger import logmes, logwrn, bombed
from utils.errors import ModelNotImportedError
from utils.constants import VOIGHT
from materials.completion import EC_BULK, EC_SHEAR

mat = None

d = os.path.join(MAT_D, "src")
f1 = os.path.join(d, "elastic.f90")
f2 = os.path.join(d, "elastic.pyf")


class Elastic(MaterialModel):
    name = "elastic"
    source_files = [f1, f2]
    def __init__(self):
        self.param_names = ["K:BMOD", "G:SMOD:MU"]
        self.prop_names = [EC_BULK, EC_SHEAR]

    def setup(self):
        """Set up the Elastic material

        """
        global mat
        try:
            import lib.elastic as mat
        except ImportError:
            raise ModelNotImportedError("elastic")
        comm = (logmes, logwrn, bombed, (self.logger.logger_id,),
                (self.logger.logger_id,), (self.logger.logger_id,))
        mat.elastic_check(self.params, *comm)

    def mimicking(self, mat_mimic):
        # reset the initial parameter array to represent this material
        iparray = np.zeros(2)
        iparray[0] = mat_mimic.completions[EC_BULK] or 0.
        iparray[1] = mat_mimic.completions[EC_SHEAR] or 0.
        if mat_mimic.completions[Y_TENSION]:
            self.logger.warn("{0} cannot mimic a strength limit, only "
                             "an elastic response will occur".format(self.name))
        return iparray

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
        # model is written with strain components given as tensor strain,
        # matmodlab uses engineering shear strains
        d = d / VOIGHT
        mat.elastic_update_state(dtime, self.params, d, stress,
                                 logmes, logwrn, bombed, *extra)
        return stress, xtra, self.constant_jacobian
