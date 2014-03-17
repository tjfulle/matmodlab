import numpy as np
try:
    from lib.mmlabpack import mmlabpack
except ImportError:
    import utils.mmlabpack as mmlabpack
from materials.material import Material


class AnisoHyper(Material):
    def setup_new_material(self, params):
        self.fiber_direction = np.array([1,0,0], dtype=np.float64)
        super(AnisoHyper, self).setup_new_material(params)

    def jacobian(self, dt, d, sig, xtra, v, *args):
        return self.constant_jacobian(v)

    def update_state(self, dtime, dstran, stress, statev, *args, **kwargs):
        """Update the state of an abaqus umat model.  Implementation
        based on Abaqus conventions

        """
        time = args[0]
        F0 = np.reshape(args[1], (3, 3), order="F")
        F = np.reshape(args[2], (3, 3), order="F")
        stran = args[3]
        temp = args[5]
        dtemp = args[6]

        # Find C and Cbar
        C = mmlabpack.dot(F.T, F)
        J = mmlabpack.det(F)
        Cb = J ** (-2. / 3.) * C
        Cbsq = mmlabpack.dot(Cb, Cb)

        # Isochoric invariants
        Ainv = mmlabpack.get_invariants(Cb, self.fiber_direction)

        # Setup to call material model
        cmname = "umat    "
        ninv = 5
        ninv2 = ninv * (ninv + 1) / 2
        nfibers = 1
        zeta = np.zeros(1)
        noel = 1
        incmpflag = 1
        ihybflag = 1
        numstatev = self.nxtra
        numfieldv = 1 # just for dimensioning
        fieldv = np.zeros(numfieldv)
        fieldvinc = np.zeros(numfieldv)

        hold = Ainv[2]
        Ainv[2] = J
        resp = self.update_state_anisohyper(Ainv, zeta, nfibers, temp, noel,
                                            cmname, incmpflag, ihybflag,
                                            statev, fieldv, fieldvinc)
        ua, ui1, ui2, ui3 = resp

        # convert C to 6x1
        stress = self._get_stress_from_w(C, J, F, ui1)

        return stress, statev

    def _get_stress_from_w(self, C, J, F, dWdIb):
        """Compute stress from variations in energy

        Notes
        -----
        The stress S is

                    Sij = 2/J Fik dW/dCkl Fjl

        and
                 dW/dCkl = Sum[dW/dIi dIi/dCkl, i=1..N]

                 I1 = Cii
                 I2 = .5 (Cii^2 - CijCji)
                 I3 = det(C)
                 I4 = ni Cij nj
                 I5 = ni CikCkj nj


        """
        dIdC = self._get_dIdC(C)
        dIbdI = self._get_dIbdI(C)

        dWdC = np.zeros(6, dtype=np.float64)
        for ij in range(6):
            for k in range(5):
                for m in range(5):
                    dWdC[ij] += dWdIb[m] * dIbdI[m, k] * dIdC[k, ij]

        dWdC = mmlabpack.asmat(dWdC)
        stress = 2. / J * mmlabpack.dot(mmlabpack.dot(F, dWdC), F.T)

        return mmlabpack.asarray(stress, 6)

    def _get_dIdC(self, C):
        """Derivative of C wrt to invariants

        Returned as a 5x6 array with rows, columns:

                      dIdC = dIi / dCj

        """
        N = self.fiber_direction
        n = np.dot(C, N)

        Identity = np.array([1, 1, 1, 0, 0, 0], dtype=np.float64)
        trC = np.trace(C)
        invC = mmlabpack.inv(C)
        detC = mmlabpack.det(C)

        dIdC = np.zeros((5, 6), dtype=np.float64)
        dIdC[0] = Identity
        dIdC[1] = trC * Identity - mmlabpack.asarray(C, 6)
        dIdC[2] = detC * mmlabpack.asarray(invC, 6)
        dIdC[3] = mmlabpack.dyad(N, N)
        dIdC[4] = mmlabpack.dyad(N, n) + mmlabpack.dyad(n, N)
        return dIdC

    def _get_dIbdI(self, C):
        """Derivative of Ibar wrt to I

        """
        N = self.fiber_direction

        # Invariants of C
        I1, I2, I3, I4, I5 = mmlabpack.get_invariants(C, N)
        I3_ = I3 ** (-1. / 3.)

        # Derivative of Ibi wrt Ij
        dIbdI = np.zeros((5, 5), dtype=np.float64)
        dIbdI[0, 0] = I3_  # dIb1 / dI1
        dIbdI[0, 2] = -1. / 3. * I1 * I3_ ** 4  # dIb1 / dI3

        dIbdI[1, 1] = I3_ ** 2  # dIb2 / dI2
        dIbdI[1, 2] = -2. / 3. * I2 * I3_ ** 5  # dIb2 / dI3

        dIbdI[2, 2] = .5 * I3 ** (-.5)  # dJ / dI3

        dIbdI[3, 2] = -1. / 3. * I4 * I3_ ** 4  # dIb4 / dI3
        dIbdI[3, 3] = I3_  # dIb4 / dI4

        dIbdI[4, 2] = -2. / 3. * I5 * I3_ ** 5  # dIb5 / dI3
        dIbdI[4, 4] = I3_ ** 2  # dIb5 / dI5

        return dIbdI
