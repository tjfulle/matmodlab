import logging
import numpy as np

from matmodlab.mmd.material import MaterialModel
from matmodlab.constants import ROOT2, ROOT23, VOIGHT
from matmodlab.utils.parameters import Parameters

class VonMises(MaterialModel):
    name = 'vonmises'

    @classmethod
    def param_names(cls, n):
        return ['K',   # Linear elastic bulk modulus
                'G',   # Linear elastic shear modulus
                'Y0',  # yield stress in uniaxial tension
                       # (yield in tension)=sqrt(3)*(yield in shear)
                       #                   = sqrt(3)*sqrt(J2)
                'H',   # Hardening modulus
                'BETA',# isotropic/kinematic hardening parameter
                       #    BETA = 0 for isotropic hardening
                       #    0 < BETA < 1 for mixed hardening
                       #    BETA = 1 for kinematic hardening
                ]

    @staticmethod
    def completions_map():
        return {'K': 0, 'G': 1, 'YT': 2, 'HM': 3, 'HP': 4}

    @classmethod
    def from_other(cls, other_mat):
        iparray = np.zeros(5)
        iparray[0] = other_mat.completions['K'] or 0.
        iparray[1] = other_mat.completions['G'] or 0.
        iparray[2] = other_mat.completions['YT'] or 1.E+99
        iparray[3] = other_mat.completions['HM'] or 0.
        iparray[4] = other_mat.completions['HP'] or 0.
        if other_mat.completions['FRICTION_ANGLE']:
            logging.getLogger('matmodlab.mmd.simulator').warn(
                'model {0} cannot mimic {1} with '
                'pressure dependence'.format(cls.name, other_mat.name))
        return cls(iparray)

    def setup(self, **kwargs):
        '''Set up the von Mises material

        '''
        logger = logging.getLogger('matmodlab.mmd.simulator')
        # Check inputs
        K, G, Y0, H, BETA = self.params

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
            logger.error('Poisson\'s ratio > .5')
        if nu < -1.0:
            errors += 1
            logger.error('Poisson\'s ratio < -1.')
        if nu < 0.0:
            logger.warn('negative Poisson\'s ratio')
        if abs(Y0) <= 1.E-12:
            Y0 = 1.0e99
        if errors:
            raise ValueError('stopping due to previous errors')

        self.params[:] = [K, G, Y0, H, BETA]

        # Register State Variables
        sdv_keys = ['EQPS', 'Y',
                    'BS_XX', 'BS_YY', 'BS_ZZ', 'BS_XY', 'BS_XZ', 'BS_YZ',
                    'SIGE']
        sdv_vals = [0.0, Y0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        return sdv_keys, sdv_vals

    def update_state(self, time, dtime, temp, dtemp, energy, rho, F0, F,
        stran, d, elec_field, stress, statev, **kwargs):
        '''Compute updated stress given strain increment

        Parameters
        ----------
        dtime : float
            Time step

        d : array_like
            Deformation rate

        stress : array_like
            Stress at beginning of step

        statev : array_like
            State dependent variables

        Returns
        -------
        S : array_like
            Updated stress

        statev : array_like
            State dependent variables

        '''
        idx = lambda x: self.sdv_keys.index(x.upper())
        bs = np.array([statev[idx('BS_XX')],
                       statev[idx('BS_YY')],
                       statev[idx('BS_ZZ')],
                       statev[idx('BS_XY')],
                       statev[idx('BS_YZ')],
                       statev[idx('BS_XZ')]])
        yn = statev[idx('Y')]

        de = d / VOIGHT * dtime

        iso = de[:3].sum() / 3.0 * np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
        dev = de - iso

        stress_trial = stress + 3.0 * self.params['K'] * iso + 2.0 * self.params['G'] * dev

        xi_trial = stress_trial - bs
        xi_trial_eqv = self.eqv(xi_trial)

        if xi_trial_eqv <= yn:
            statev[idx('SIGE')] = xi_trial_eqv
            return stress_trial, statev, None
        else:
            N = xi_trial - xi_trial[:3].sum() / 3.0 * np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
            N = N / (ROOT23 * xi_trial_eqv)
            deqps = (xi_trial_eqv - yn) / (3.0 * self.params['G'] + self.params['H'])
            dps = 1. / ROOT23 * deqps * N

            stress_final = stress_trial - 2.0 * self.params['G'] / ROOT23 * deqps * N

            bs = bs + 2.0 / 3.0 * self.params['H'] * self.params['BETA'] * dps

            statev[idx('EQPS')] += deqps
            statev[idx('Y')] += self.params['H'] * (1.0 - self.params['BETA']) * deqps
            statev[idx('BS_XX')] = bs[0]
            statev[idx('BS_YY')] = bs[1]
            statev[idx('BS_ZZ')] = bs[2]
            statev[idx('BS_XY')] = bs[3]
            statev[idx('BS_YZ')] = bs[4]
            statev[idx('BS_XZ')] = bs[5]
            statev[idx('SIGE')] = self.eqv(stress_final - bs)
            return stress_final, statev, None


    def eqv(self, sig):
        # Returns sqrt(3 * rootj2) = sig_eqv = q
        s = sig - sig[:3].sum() / 3.0 * np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
        return 1. / ROOT23 * np.sqrt(np.dot(s[:3], s[:3]) + 2 * np.dot(s[3:], s[3:]))
