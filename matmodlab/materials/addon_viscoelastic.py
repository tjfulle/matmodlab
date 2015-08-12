import logging
from numpy.linalg import inv, det
from numpy import reshape, dot, zeros, array, exp
from matmodlab.materials.product import PRONY
from matmodlab.utils.mmlabpack import dev, asmat, asarray

class Viscoelastic(object):

    """

    Parameters
    ----------
    WC1=1  ! WLF C1
    WC2=2  ! WLF C2
    WTR=3  ! WLF TREF
    GOO=4  ! PRONY SHEAR INFINITY
    G01=IPGOO+1  ! PRONY SHEAR COEFFICIENTS (10)
    G02=IPGOO+2
    G03=IPGOO+3
    G04=IPGOO+4
    G05=IPGOO+5
    G06=IPGOO+6
    G07=IPGOO+7
    G08=IPGOO+8
    G09=IPGOO+9
    G10=IPGOO+10
    T=IPG10 ! SHEAR RELAX TIME (10)
    T01=IPT+1
    T02=IPT+2
    T03=IPT+3
    T04=IPT+4
    T05=IPT+5
    T06=IPT+6
    T07=IPT+7
    T08=IPT+8
    T09=IPT+9
    T10=IPT+10

    State Dependent Variables
    -------------------------
         (1) : WLF AVERAGE SHIFT FACTOR
         (2) : AVERAGE NUMERICAL SHIFT FACTOR
       (3:8) : INSTANTANEOUS DEVIATORIC PK2 STRESS COMPONENTS AT START OF
               CURRENT TIME STEP WRITTEN USING THE INITIAL CONFIGURATION AS
               THE REFERENCE STATE
      (9:14) : XX,YY,ZZ,XY,YZ,ZX COMPONENTS OF VISCOELASTIC DEVIATORIC 2ND
               PIOLA KIRCHHOFF (PK2) STRESS FOR 1ST PRONY TERM USING THE
               INITIAL CONFIGURATION AS THE REFERENCE STATE
     (15:20) : VISCO DEV PK2 STRESS FOR 2ND PRONY TERM
     (21:26) : VISCO DEV PK2 STRESS FOR 3RD PRONY TERM
     (27:32) : VISCO DEV PK2 STRESS FOR 4TH PRONY TERM
     (33:38) : VISCO DEV PK2 STRESS FOR 5TH PRONY TERM
     (39:44) : VISCO DEV PK2 STRESS FOR 6TH PRONY TERM
     (45:50) : VISCO DEV PK2 STRESS FOR 7TH PRONY TERM
     (51:56) : VISCO DEV PK2 STRESS FOR 8TH PRONY TERM
     (57:62) : VISCO DEV PK2 STRESS FOR 9TH PRONY TERM
     (63:68) : VISCO DEV PK2 STRESS FOR 10TH PRONY TERM

    """
    def __init__(self, time, data):
        self.time = time
        data = array(data)
        if self.time == PRONY:
            # check data
            if data.shape[1] != 2:
                raise ValueError("expected Prony series data to be 2 columns")
            self.data = data
        else:
            raise ValueError("{0}: unkown time type".format(time))

        self.Goo = 1. - sum(self.data[:, 0])
        if self.Goo < 0.:
            raise ValueError("expected sum of shear Prony coefficients, "
                             "including infinity term to be one")

    def setup(self, trs_model=None):

        # setup viscoelastic params
        self.params = zeros(24)

        # starting location of G and T Prony terms
        n = self.nprony
        I, J = (4, 14)
        self.params[I:I+n] = self.data[:, 0]
        self.params[J:J+n] = self.data[:, 1]

        # Ginf
        self.params[3] = self.Ginf

        # Allocate storage for visco data
        keys = []

        # Shift factors
        keys.extend(["SHIFT_{0}".format(i+1) for i in range(2)])

        # Instantaneous deviatoric PK2
        m = {0: "XX", 1: "YY", 2: "ZZ", 3: "XY", 4: "YZ", 5: "XZ"}
        keys.extend(["TE_{0}".format(m[i]) for i in range(6)])

        # Visco elastic model supports up to 10 Prony series terms,
        # allocate storage for stress corresponding to each
        nprony = 10
        for l in range(nprony):
            for i in range(6):
                keys.append("H{0}_{1}".format(l+1, m[i]))

        self.nvisco = len(keys)
        idata = zeros(self.nvisco)
        idata[:2] = 1.

        if trs_model is not None:
            self.params[0] = trs_model.wlf_coeffs[0] # C1
            self.params[1] = trs_model.wlf_coeffs[1] # C2
            self.params[2] = trs_model.temp_ref # REF TEMP

        log = logging.getLogger('matmodlab.mmd.simulator')

        #visco.propcheck(self.params, log.info, log.warn, StopFortran)
        # Check property array for viscoelastic model

        # Check sum of prony series coefficients
        psum  = sum(self.params[I:I+n])
        if any(self.params[I:I+n] < 0.):
            raise ValueError('Expected all shear Prony series coefficients > 0')

        if abs(psum) < 1e-10:
            log.warn('Sum of normalized shear prony series coefficients\n'
                     'including infinity term is zero. normalized infinity\n'
                     'coefficient set to one for elastic response.')
            self.params[3] = 1.

        elif abs(psum - 1.) > 1e-3:
            message = ('Expected sum of normalized shear prony series\n'
                       'coefficients including inf term to be 1,\n'
                       'got {0}'.format(psum))
            if abs(psum - 1) < .03:
                log.warn(message)
            else:
                raise ValueError(message)

        # Verify that all relaxation times are positive
        for (i, param) in enumerate(self.params[J:], start=J):
            if param <= 0.:
                log.warn('Shear relaxation time term <=0, SETTING TO 1')
                self.params[i] = 1.

        return keys, idata

    @property
    def nprony(self):
        return self.data.shape[0]

    @property
    def Ginf(self):
        return self.Goo

    def update_state(self, time, dtime, temp, dtemp, statev, F, stress):

        cfac = zeros(2)

        # Get the shift factors (stored in statev)
        self.shiftfac(dtime, time, temp, dtemp, F, statev)

        # change reference state on sodev from configuration at end of current
        # time step to initial configuration
        F = F.reshape((3,3))
        C = asarray(dot(F.T, F), 6)
        sigo = stress.copy()
        pk2o = self.pull(F, sigo)
        pk2odev = self.dev(pk2o, C)

        # reduced time step
        dtred = dtime / statev[1] / statev[0]

        # loop over the prony terms
        I, J = (4, 14)
        for k in range(10):
            j = k * 6
            # compute needed viscoelastic factors
            ratio = dtred / self.params[J+k]
            e = exp(-ratio)

            if ratio > 1e-3:
              # explicit calculation of (1-exp(-ratio))/ratio
                s = (1. - e) / ratio
            else:
               # taylor series calculation of (1 - exp(-ratio))/ratio
                s = 1. - .5 * ratio + 1. / 6. * ratio ** 2

           # update the viscoelastic state variable history for kth prony term
            for l in range(6):
                statev[8+j+l] = (e * statev[8+j+l] +
                                 self.params[4+k] * (s - e) * statev[2+l] +
                                 self.params[4+k] * (1 - s) * pk2odev[l])
            cfac[0] += (1 - s) * self.params[4+k]

        # compute decaying deviatoric stress
        pk2dev = zeros(6)
        for l in range(6):
            for k in range(10):
                j = k * 6
                pk2dev[l] += statev[8+j+l]

        # change reference state on decaying portion of deviatoric stress from
        # initial configuration to configuration at end of current time step
        sdev = self.push(F, pk2dev)

        # eliminate the pressure arising from the reference state changes used
        # in computing the decaying portion of the deviatoric stress
        tr = sum(sdev[:3])
        if abs(tr) > 1e-16:
            sdev[0] -= tr / 3.
            sdev[1] -= tr / 3.
            sdev[2] = -(sdev[0] + sdev[1])

        # compute total deviatoric stress
        sodev = dev(sigo)
        sdev = sodev - sdev

        # total stress
        stress = sdev.copy()
        pressure = -sum(sigo[:3]) / 3.
        stress[:3] -= pressure

        # instantaneous deviatoric stress with original configuration as
        # reference state
        statev[2:8] = pk2odev

        return stress, cfac, statev

    def shiftfac(self, dtime, time, temp, dtemp, F, statev):

        # retrieve the WLF parameters - thermal analysis
        C1, C2, Tref   = self.params[:3]
        temp_new = temp + dtemp

        # Evaluate the WLF shift factor at the average temp of the step
        at = 1. # Default shift factor
        ts_flag = 1
        if ts_flag:
            tempavg = .5 * (temp + temp_new)
            tempdiff = tempavg - Tref
            if abs(C1) < 1e-10:
                at = 1.
            elif C2 + tempdiff <= 1e-30:
                at = 1e-30
            else:
                log_at = C1 * tempdiff / (C2 + tempdiff)
                if log_at > 30.:
                    at = 1e30
                elif log_at < -30.:
                    at = 1e-30
                else:
                    at = 10. ** log_at

        # Store the numerical shift factor for WLF
        statev[0] = 1. / at
        statev[1] = 1.

    def pull(self, F, A):
        return self.push(inv(F), A)

    def push(self, F, A):
        # Push transformation [A'] = 1/DET[F] [F] [A] [F]^T
        B = asmat(A)
        return asarray(dot(dot(F, B), F.T) / det(F), 6)

    def iso(self, A, C):
        W = array((1.,1.,1.,2.,2.,2.))
        Cinv = self.inv_6(C)
        return sum(W * A * C) * Cinv / 3.

    def dev(self, A, C):
        iso_A = self.iso(A, C)
        return A - iso_A

    def inv_6(self, A):
        # 3x3 MATRIX INVERSE (STORED AS 6x1 ARRAY)
        Ainv = zeros(6)

        # DET[A]
        det_A = A[0]*(A[1]*A[2] - A[4]*A[4]) \
              + A[3]*(A[4]*A[5] - A[3]*A[2]) \
              + A[5]*(A[3]*A[4] - A[1]*A[5])

        # [A]^-0
        Ainv[0] = (A[1]*A[2] - A[4]*A[4]) / det_A
        Ainv[1] = (A[0]*A[2] - A[5]*A[5]) / det_A
        Ainv[2] = (A[0]*A[1] - A[3]*A[3]) / det_A
        Ainv[3] = (A[4]*A[5] - A[2]*A[3]) / det_A
        Ainv[4] = (A[3]*A[5] - A[0]*A[4]) / det_A
        Ainv[5] = (A[3]*A[4] - A[1]*A[5]) / det_A
        return Ainv
