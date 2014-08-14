import sys
import numpy as np

form core.runtime import opts
from core.mmlio import Error1, log_message, log_error, log_warning
try:
    from lib.mmlabpack import mmlabpack
except ImportError:
    import utils.mmlabpack as mmlabpack
from materials.material import Material
try:
    from mml_user_sub import get_invars_time
except ImportError:
    get_invars_time = None
try:
    import lib.uanisohyper as umat
except ImportError:
    umat = None


Z3 = np.zeros(3)
NINV = 5
Identity = np.array([1, 1, 1, 0, 0, 0], dtype=np.float64)


class AbaUAnisoHyper(Material):
    def setup_new_material(self, params):
        self.fiber_direction = np.array([1,0,0], dtype=np.float64)
        super(AnisoHyper, self).setup_new_material(params)

    def jacobian(self, dt, d, sig, xtra, v, *args):
        return self.constant_jacobian(v)

    def update_state(self, dtime, dstran, stress, statev, *args, **kwargs):
        """Update the state of an abaqus umat model.  Implementation
        based on Abaqus conventions

        """
        invars_time = None
        N = self.fiber_direction
        time = args[0]
        F0 = np.reshape(args[1], (3, 3), order="F")
        F = np.reshape(args[2], (3, 3), order="F")
        stran = args[3]
        temp = args[5]
        dtemp = args[6]

        # Find C and Cbar
        C = mmlabpack.dot(F.T, F)
        J = mmlabpack.det(F)

        # Invariants and isochoric invariants
        if get_invars_time:
            invars_time = get_invars_time(opts.runid, time)
            if invars_time is not None:
                I1b, I2b, J, I4b, I5b = invars_time
                I1 = J ** (2. / 3.) * I1b
                I2 = J ** (4. / 3.) * I2b
                I4 = J ** (2. / 3.) * I4b
                I5 = J ** (4. / 3.) * I5b
        if invars_time is None:
            I1, I2, I3, I4, I5 = mmlabpack.get_invariants(C, N)
            I1b = J ** (-2. / 3.) * I1
            I2b = J ** (-4. / 3.) * I2
            I4b = J ** (-2. / 3.) * I4
            I5b = J ** (-4. / 3.) * I5

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

        Ainv = np.array([I1b, I2b, J, I4b, I5b], dtype=np.float64, order="F")
        resp = self.update_state_anisohyper(Ainv, zeta, nfibers, temp, noel,
                                            cmname, incmpflag, ihybflag,
                                            statev, fieldv, fieldvinc)
        ua, ui1, ui2, ui3 = resp

        dIbdC = get_dIbdC(C, N)

        # derivative of energy wrt C
        dWdC = np.zeros(6, dtype=np.float64)
        dWdIb = ui1
        for ij in range(6):
            for k in range(NINV):
                dWdC[ij] += dWdIb[k] * dIbdC[k, ij]
        dWdC = mmlabpack.asmat(dWdC)

        # Cauchy stress
        stress = 2. / J * mmlabpack.dot(mmlabpack.dot(F, dWdC), F.T)
        stress = mmlabpack.asarray(stress, 6)

        return stress, statev

        # old way
        hold = dIbdC.copy()
        hold1 = stress.copy()
        # derivative of isochoric invariants wrt to invariants
        dIbdI = get_dIbdI(I1, I2, J, I4, I5)

        # derivative of invariants wrt C
        dIdC = get_dIdC(C, N)

        dIbdC = np.zeros((NINV, 6))
        for ij in range(6):
            for k in range(NINV):
                for m in range(NINV):
                    dIbdC[k, ij] += dIbdI[k, m] * dIdC[m, ij]

        # derivative of energy wrt C
        dWdC = np.zeros(6, dtype=np.float64)
        for ij in range(6):
            for k in range(NINV):
                for m in range(NINV):
                    dWdC[ij] += ui1[m] * dIbdI[m, k] * dIdC[k, ij]
        dWdC = mmlabpack.asmat(dWdC)

        # Cauchy stress
        stress = 2. / J * mmlabpack.dot(mmlabpack.dot(F, dWdC), F.T)
        stress = mmlabpack.asarray(stress, 6)

        for (i, row) in enumerate(hold):
            print i+1
            print " ".join("{0:.8f}".format(float(x)) for x in row)
            print " ".join("{0:.8f}".format(float(x)) for x in dIbdC[i])
            print
        print " ".join("{0:.8f}".format(float(x)) for x in hold1)
        print " ".join("{0:.8f}".format(float(x)) for x in stress)
        print
        print

        return stress, statev


def get_dIdC(C, N):
    """Derivative of C wrt to invariants

    Returned as a 5x6 array with rows, columns:

                  dIdC = dIi / dCj

    """
    n = np.dot(C, N)

    trC = np.trace(C)
    invC = mmlabpack.inv(C)
    detC = mmlabpack.det(C)

    dIdC = np.zeros((5, 6), dtype=np.float64)
    dIdC[0] = Identity  # dI1 / dC
    dIdC[1] = trC * Identity - mmlabpack.asarray(C, 6)  # dI2 / dC
    dIdC[2] = .5 * np.sqrt(detC) * mmlabpack.asarray(invC, 6)  # dJ / dC
    dIdC[3] = mmlabpack.dyad(N, N)  # dI4 / dC
    dIdC[4] = mmlabpack.dyad(N, n) + mmlabpack.dyad(n, N)  # dI5 / dC
    return dIdC


def get_dIbdI(I1, I2, J, I4, I5):
    """Derivative of Ibar wrt to I"""

    # Derivative of Ibi wrt Ij
    dIbdI = np.zeros((5, 5), dtype=np.float64)
    dIbdI[0, 0] = J ** (-2. / 3.)  # dIb1 / dI1
    dIbdI[0, 2] = -2. / 3. * I1 * J ** (-5. / 3.)  # dIb1 / dJ

    dIbdI[1, 1] = J ** (-4. / 3.)  # dIb2 / dI2
    dIbdI[1, 2] = -4. / 3. * I2 * J ** (-7. / 3.)  # dIb2 / dJ

    dIbdI[2, 2] = 1.  # dJ / dJ

    dIbdI[3, 2] = -2. / 3. * I4 * J ** (-5. / 3.)  # dIb4 / dJ
    dIbdI[3, 3] = J ** (-2. / 3.)  # dIb4 / dI4

    dIbdI[4, 2] = -4. / 3. * I5 * J ** (-7. / 3.)  # dIb5 / dJ
    dIbdI[4, 4] = J ** (-4. / 3.)  # dIb5 / dI5

    return dIbdI


def get_dIbdC(C, N):

    n = np.dot(C, N)
    I1, I2, I3, I4, I5 = mmlabpack.get_invariants(C, N)
    Cinv = mmlabpack.inv(C)
    J = np.sqrt(I3)
    NN = mmlabpack.dyad(N, N)
    Nn = mmlabpack.dyad(N, n)
    nN = mmlabpack.dyad(n, N)

    dJdC = .5 * J * mmlabpack.asarray(Cinv, 6)
    C = mmlabpack.asarray(C, 6)

    dIbdC = np.zeros((NINV, 6), dtype=np.float64)

    dIbdC[0] = J ** (-2. / 3.) * Identity
    dIbdC[0] += -2. / 3. * I1 * J ** (-5. / 3.) * dJdC

    dIbdC[1] = J ** (-4. / 3.) * (I1 * Identity - C)
    dIbdC[1] += -4. / 3. * I2 * J ** (-7. / 3.) * dJdC

    dIbdC[2] = dJdC

    dIbdC[3] = J ** (-2. / 3.) * NN
    dIbdC[3] += -2. / 3. * I4 * J ** (-5. / 3.) * dJdC

    dIbdC[4] = J ** (-4. / 3.) * (Nn + nN)
    dIbdC[4] += -4. / 3. * I5 * J ** (-7. / 3.) * dJdC

    return dIbdC

