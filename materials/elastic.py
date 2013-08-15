import numpy as np

from materials._material import Material
from utils.tensor import iso, dev
from utils.errors import Error1

class Elastic(Material):
    name = "elastic"
    def __init__(self):
        """Instantiate the Elastic material

        """
        super(Elastic, self).__init__()
        self.register_parameters("E", "NU")

    def setup(self, params):
        """Set up the Elastic material

        Parameters
        ----------
        params : ndarray
            Material parameters

        """
        self.params = self._check_params(params)
        nu = self.params[self.NU]

        # Bulk modulus, Youngs modulus and Poissons ratio
        self.K = self.params[self.E] / 3. / (1. - 2. * nu)
        self.G = self.params[self.E] / 2. / (1. + nu)

        # compute the constant stiffness
        Eh = self.params[self.E] / (1. + nu) / (1. - 2 * nu)
        self.C = Eh * np.array([[1. - nu, nu, nu, 0, 0, 0],
                                [nu, 1. - nu, nu, 0, 0, 0],
                                [nu, nu, 1. - nu, 0, 0, 0],
                                [0, 0, 0, (1. - 2 * nu) / 2., 0, 0],
                                [0, 0, 0, 0, (1. - 2 * nu) / 2., 0],
                                [0, 0, 0, 0, 0, (1. - 2 * nu) / 2.]],
                               dtype=np.float)

    def _check_params(self, params):
        """Check parameters and set defaults

        """
        if params[self.E] < 0.:
            raise Error1("Young's modulus E must be > 0")
        if not -1 < params[self.NU] < .5:
            raise Error1("Poisson's ratio NU out of bounds")
        return params

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
        dstrain = d * dt
        dstress = 3. * self.K * iso(dstrain) + 2 * self.G * dev(dstrain)
        return stress + dstress, xtra

    def stiffness(self, dt, d, stress, xtra):
        """Return the constant stiffness
        dt : float
            time step

        d : array_like
            Deformation rate

        stress : array_like
            Stress at beginning of step

        """
        return self.C
