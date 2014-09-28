import os
from core.product import MAT_D
from core.material import MaterialModel
from utils.errors import ModelNotImportedError
from materials.completion import EC_BULK, EC_SHEAR
try: import lib.elastic as mat
except ImportError: mat = None

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
        if mat is None:
            raise ModelNotImportedError("elastic")
        comm = (self.logger.write, self.logger.warn, self.logger.raise_error)
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
        comm = (self.logger.write, self.logger.warn, self.logger.raise_error)
        mat.elastic_update_state(dtime, self.params, d, stress, *comm)
        return stress, xtra, self.constant_jacobian
