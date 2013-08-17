import numpy as np

from materials._material import Material
from utils.errors import Error1
try:
    import lib.elastic as elastic
except ImportError:
    elastic = None

class Elastic(Material):
    name = "elastic"
    driver = "solid"
    def __init__(self):
        """Instantiate the Plastic material

        """
        super(Elastic, self).__init__()
        self.register_parameters("K", "G")

    def setup(self, params):
        """Set up the Elastic material

        Parameters
        ----------
        params : ndarray
            Material parameters

        """
        if elastic is None:
            raise Error1("elastic model not imported")
        elastic.elastic_check(params)
        K, G, = params
        self.params = params
        self.bulk_modulus = K
        self.shear_modulus = G

    def update_state(self, dt, d, stress, xtra):
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
        elastic.elastic_update_state(dt, self.params, d, stress)
        return stress, xtra

    def _jacobian(self, dt, d, stress, xtra, v):
        """Return the constant stiffness
        dt : float
            time step

        d : array_like
            Deformation rate

        stress : array_like
            Stress at beginning of step

        """
        return self.constant_jacobian(v)
        #J = elastic.elastic_stiff(dt, self.params, d, stress)
        #return J
