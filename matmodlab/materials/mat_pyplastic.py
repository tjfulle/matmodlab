import logging
import numpy as np
from matmodlab.mmd.material import MaterialModel
from matmodlab.utils.parameters import Parameters
from matmodlab.constants import ROOT2, ROOT3, TOOR2, TOOR3, I6, VOIGHT


class PyPlastic(MaterialModel):
    name = 'pyplastic'

    @classmethod
    def param_names(cls, n):
        return ['K',    # Linear elastic bulk modulus
                'G',    # Linear elastic shear modulus
                'A1',   # Intersection of the yield surface with the
                        #   sqrt(J2) axis (pure shear).
                        #     sqrt(J2) = r / sqrt(2); r = sqrt(2*J2)
                        #     sqrt(J2) = q / sqrt(3); q = sqrt(3*J2)
                'A4']   # Pressure dependence term.
                        #   A4 = -d(sqrt(J2)) / d(I1)
                        #         always positive

    @staticmethod
    def completions_map():
        return {'K': 0, 'G': 1, 'DPA': 2, 'DPB': 3}

    @classmethod
    def from_other(cls, other_mat):
        # other_mat is a material instance of the material model to mimic
        iparray = np.zeros(4)
        iparray[0] = other_mat.completions['K'] or 0.
        iparray[1] = other_mat.completions['G'] or 0.
        iparray[2] = other_mat.completions['DPA'] or 1.E+99
        iparray[3] = other_mat.completions['DPB'] or 0.
        if other_mat.completions['HARD_MOD']:
            logging.getLogger('matmodlab.mmd.simulator').warn(
                'model {0} cannot mimic {1} with '
                'hardening'.format(cls.name, other_mat.name))
        return cls(iparray)

    def setup(self, **kwargs):
        '''Set up the plastic material

        '''
        logger = logging.getLogger('matmodlab.mmd.simulator')
        K, G, A1, A4 = self.params
        if abs(A1) <= 1.E-12:
            A1 = 1.0e99

        # Check the input parameters
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
            logger.warn('#negative Poisson\'s ratio')
        if A1 <= 0.0:
            errors += 1
            logger.error('A1 must be positive nonzero')
        if A4 < 0.0:
            errors += 1
            logger.error('A4 must be non-negative')
        if errors:
            raise ValueError('stopping due to previous errors')

        # Save the new parameters
        self.params[:] = [K, G, A1, A4]

        # Register State Variables
        sdv_keys = ['EP_XX', 'EP_YY', 'EP_ZZ', 'EP_XY', 'EP_XZ', 'EP_YZ',
                    'I1', 'ROOTJ2', 'YROOTJ2', 'ISPLASTIC']
        sdv_vals = np.zeros(len(sdv_keys))
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
        sigsave = np.copy(stress)
        # Define helper functions and unload params/state vars
        A1 = self.params['A1']
        A4 = self.params['A4']
        idx = lambda x: self.sdv_keys.index(x.upper())
        ep = statev[idx('EP_XX'):idx('EP_YZ')+1]

        # Compute the trial stress and invariants
        stress = stress + self.dot_with_elastic_stiffness(d / VOIGHT * dtime)
        i1 = self.i1(stress)
        rootj2 = self.rootj2(stress)
        if rootj2 - (A1 - A4 * i1) <= 0.0:
            statev[idx('ISPLASTIC')] = 0.0
        else:
            statev[idx('ISPLASTIC')] = 1.0

            s = self.dev(stress)
            N = ROOT2 * A4 * I6 + s / self.tensor_mag(s)
            N = N / np.sqrt(6.0 * A4 ** 2 + 1.0)
            P = self.dot_with_elastic_stiffness(N)

            # 1) Check if linear drucker-prager
            # 2) Check if trial stress is beyond the vertex
            # 3) Check if trial stress is in the vertex
            if (A4 != 0.0 and
                    i1 > A1 / A4 and
                    rootj2 / (i1 - A1 / A4) < self.rootj2(P) / self.i1(P)):
                dstress = stress - A1 / A4 / 3.0 * I6
                # convert all of the extra strain into plastic strain
                ep += self.iso(dstress) / (3.0 * self.params['K'])
                ep += self.dev(dstress) / (2.0 * self.params['G'])
                stress = A1 / A4 / 3.0 * I6
            else:
                # not in vertex; do regular return
                lamb = ((rootj2 - A1 + A4 * i1) / (A4 * self.i1(P)
                        + self.rootj2(P)))
                stress = stress - lamb * P
                ep += lamb * N

            # Save the updated plastic strain
            statev[idx('EP_XX'):idx('EP_YZ')+1] = ep

        statev[idx('I1')] = self.i1(stress)
        statev[idx('ROOTJ2')] = self.rootj2(stress)
        statev[idx('YROOTJ2')] = A1 - A4 * self.i1(stress)

        return stress, statev, None

    def dot_with_elastic_stiffness(self, A):
        return (3.0 * self.params['K'] * self.iso(A) +
                2.0 * self.params['G'] * self.dev(A))

    def tensor_mag(self, A):
        return np.sqrt(np.dot(A[:3], A[:3]) + 2.0 * np.dot(A[3:], A[3:]))

    def iso(self, sig):
        return sig[:3].sum() / 3.0 * I6

    def dev(self, sig):
        return sig - self.iso(sig)

    def rootj2(self, sig):
        s = self.dev(sig)
        return self.tensor_mag(s) * TOOR2

    def i1(self, sig):
        return np.sum(sig[:3])
