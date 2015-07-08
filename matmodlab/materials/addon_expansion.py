from math import log
from numpy import asarray, dot, eye, zeros, allclose
from numpy.linalg import inv
from matmodlab.constants import I6
from matmodlab.materials.product import ISOTROPIC
import matmodlab.utils.mmlabpack as mmlabpack

class Expansion(object):
    def __init__(self, type, data):

        if type == ISOTROPIC:
            if len(data) != 1:
                raise ValueError("unexpected value for isotropic expansion")
        else:
            raise ValueError("{0}: unknown expansion type".format(type))
        self.data = asarray(data)
        self.type = type

    def setup(self):
        keys = ['EM.{0}'.format(s) for s in ('XX', 'YY', 'ZZ', 'XY', 'YZ', 'XZ')]
        vals = zeros(6)
        return keys, vals

    def update_state(self, kappa, itemp, temp, dtemp, dtime, F, E, d):

        # update the strain and deformation gradient
        if self.type == ISOTROPIC:

            cte = self.data[0]

            # The total strain at the end of the increment is E

            # Mechanical strain
            Eth = cte * (temp + dtemp - itemp) * I6
            Em = E - Eth

            # Updated deformation gradient
            Fm = mmlabpack.f_from_e(kappa, Em)

            # mechanical deformation rate
            # F0 = F.reshape(3,3)
            # F1 = Fm.reshape(3,3)
            # Lth = mmlabpack.logm(dot(inv(F0), F1))
            # dm = d + mmlabpack.asarray(Lth, 6)
            dtherm = cte * dtemp * I6 / dtime * I6
            dm = d - dtherm

        return Fm, Em, dm
