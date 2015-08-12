import logging
from numpy import dot, ix_, zeros
from matmodlab.mmd.material import MaterialModel

class PyElastic(MaterialModel):
    name = "pyelastic"

    @classmethod
    def param_names(cls, n):
        return ['K', 'G']

    @staticmethod
    def completions_map():
        return {'K': 0, 'G': 1}

    @classmethod
    def from_other(cls, other_mat):
        # reset the initial parameter array to represent this material
        iparray = zeros(2)
        iparray[0] = other_mat.completions['K'] or 0.
        iparray[1] = other_mat.completions['G'] or 0.
        if other_mat.completions['YT']:
            logging.getLogger('matmodlab.mmd.simulator').warn(
                '{0!r} cannot mimic a strength limit, '
                'only an elastic response will occur'.format(cls.name))
        return cls(iparray)

    def setup(self, **kwargs):
        """Set up the Elastic material

        """
        logger = logging.getLogger('matmodlab.mmd.simulator')
        # Check inputs
        K, G, = self.params
        errors = 0
        if K <= 0.0:
            errors += 1
            logger.error('Bulk modulus K must be positive')
        if G <= 0.0:
            errors += 1
            logger.error('Shear modulus G must be positive')
        nu = (3.0 * K - 2.0 * G) / (6.0 * K + 2.0 * G)
        if nu > 0.5:
            errors += 1
            logger.error("Poisson's ratio > .5")
        if nu < -1.0:
            errors += 1
            logger.error("Poisson's ratio < -1.")
        if nu < 0.0:
            logger.warn("Negative Poisson's ratio")
        if errors:
            raise ValueError("stopping due to previous errors")

    def update_state(self, time, dtime, temp, dtemp, energy, rho, F0, F,
        stran, d, elec_field, stress, statev, **kwargs):
        """Compute updated stress given strain increment"""

        # elastic properties
        K = self.parameters['K']
        G = self.parameters['G']

        K3 = 3. * K
        G2 = 2. * G
        Lam = (K3 - G2) / 3.

        # elastic stiffness
        ddsdde = zeros((6,6))
        ddsdde[ix_(range(3), range(3))] = Lam
        ddsdde[range(3),range(3)] += G2
        ddsdde[range(3,6),range(3,6)] = G

        # stress update
        stress += dot(ddsdde, d * dtime)

        return stress, statev, ddsdde
