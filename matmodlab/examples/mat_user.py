import logging
from numpy import dot, ix_, zeros
from matmodlab.mmd.material import MaterialModel
from matmodlab.constants import VOIGHT

class UserElastic(MaterialModel):
    name = "uelastic"

    @classmethod
    def param_names(cls, n):
        return ['E', 'Nu']

    def setup(self, **kwargs):
        """Set up the Elastic material

        """
        logger = logging.getLogger('matmodlab.mmd.simulator')

        # Check inputs
        E, Nu = self.params

        errors = 0
        if E <= 0.0:
            errors += 1
            logger.error("Young's modulus E must be positive")
        if Nu > 0.5:
            errors += 1
            logger.error("Poisson's ratio > .5")
        if Nu < -1.0:
            errors += 1
            logger.error("Poisson's ratio < -1.")
        if Nu < 0.0:
            logger.warn("#---- WARNING: negative Poisson's ratio")
        if errors:
            raise ValueError("stopping due to previous errors")

    def update_state(self, time, dtime, temp, dtemp, energy, rho, F0, F,
        stran, d, elec_field, stress, statev, **kwargs):
        """Compute updated stress given strain increment"""

        # elastic properties
        E = self.parameters['E']
        Nu = self.parameters['Nu']

        K = E / 3. / (1. - 2. * Nu)
        G = E / 2. / (1. + Nu)

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
