import numpy as np

from core.runtime import opts
from core.material import MaterialModel
from utils.constants import ROOT2, ROOT23
from utils.data_containers import Parameters

class VonMises(MaterialModel):

    def __init__(self):
        self.name = "vonmises"
        self.param_names = ["K",    # Linear elastic bulk modulus
                            "G",    # Linear elastic shear modulus
                            "Y0",   # yield stress in uniaxial tension
                                    # (yield in tension)=sqrt(3)*(yield in shear)
                                    #                   = sqrt(3)*sqrt(J2)
                            "H",    # Hardening modulus
                            "BETA", # isotropic/kinematic hardening parameter
                                    #    BETA = 0 for isotropic hardening
                                    #    0 < BETA < 1 for mixed hardening
                                    #    BETA = 1 for kinematic hardening
                                    ]
        self.param_defaults = [0.0, 0.0, 1.0e30, 0.0, 0.0]

    def setup(self):
        """Set up the von Mises material

        """
        # Check inputs
        if opts.mimic == "elastic":
            self.logger.warn("model '{0}' mimicing '{1}'".format(
                self.name, "elastic"))
            K = self.params["K"]
            G = self.params["G"]
            Y0 = 1.0e99
            H = 0.0
            BETA = 0.0

        else:
            K = self.params["K"]
            G = self.params["G"]
            Y0 = self.params["Y0"]
            H = self.params["H"]
            BETA = self.params["BETA"]

            errors = 0
            if K <= 0.0:
                errors += 1
                self.logger.error("Bulk modulus K must be positive", raise_error=0)
            if G <= 0.0:
                errors += 1
                self.logger.error("Shear modulus G must be positive", raise_error=0)
            nu = (3.0 * K - 2.0 * G) / (6.0 * K + 2.0 * G)
            if nu > 0.5:
                errors += 1
                self.logger.error("Poisson's ratio > .5", raise_error=0)
            if nu < -1.0:
                errors += 1
                self.logger.error("Poisson's ratio < -1.", raise_error=0)
            if nu < 0.0:
                errors += 1
                self.logger.warn("negative Poisson's ratio", raise_error=0)
            if Y0 == 0.0:
                Y0 = 1.0e99
            if errors:
                self.logger.error("stopping due to previous errors")

        newparams = [K, G, Y0, H, BETA]
        newnames = ["K", "G", "Y0", "H", "BETA"]
        self.params = Parameters(newnames, newparams)

        self.bulk_modulus = self.params["K"]
        self.shear_modulus = self.params["G"]

        # Register State Variables
        self.sv_names = ["EQPS", "Y",
                         "BS_XX", "BS_YY", "BS_ZZ", "BS_XY", "BS_XZ", "BS_YZ",
                         "SIGE"]
        sv_values = [0.0, Y0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        self.register_xtra_variables(self.sv_names, sv_values)

    def update_state(self, time, dtime, temp, dtemp, energy, rho, F0, F,
        stran, d, elec_field, user_field, stress, xtra, **kwargs):
        """Compute updated stress given strain increment

        Parameters
        ----------
        dtime : float
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
        idx = lambda x: self.sv_names.index(x.upper())
        bs = np.array([xtra[idx('BS_XX')], xtra[idx('BS_YY')], xtra[idx('BS_ZZ')],
                       xtra[idx('BS_XY')], xtra[idx('BS_YZ')], xtra[idx('BS_XZ')]])
        yn = xtra[idx('Y')]

        de = d * dtime

        iso = de[:3].sum() / 3.0 * np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
        dev = de - iso

        stress_trial = stress + 3.0 * self.bulk_modulus * iso + 2.0 * self.shear_modulus * dev

        xi_trial = stress_trial - bs
        xi_trial_eqv = self.eqv(xi_trial)

        if xi_trial_eqv <= yn:
            xtra[idx('SIGE')] = xi_trial_eqv
            return stress_trial, xtra, self.constant_jacobian
        else:
            N = xi_trial - xi_trial[:3].sum() / 3.0 * np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
            N = N / (ROOT23 * xi_trial_eqv)
            deqps = (xi_trial_eqv - yn) / (3.0 * self.shear_modulus + self.params["H"])
            dps = 1. / ROOT23 * deqps * N

            stress_final = stress_trial - 2.0 * self.shear_modulus / ROOT23 * deqps * N

            bs = bs + 2.0 / 3.0 * self.params["H"] * self.params["BETA"] * dps

            xtra[idx('EQPS')] += deqps
            xtra[idx('Y')] += self.params["H"] * (1.0 - self.params["BETA"]) * deqps
            xtra[idx('BS_XX')] = bs[0]
            xtra[idx('BS_YY')] = bs[1]
            xtra[idx('BS_ZZ')] = bs[2]
            xtra[idx('BS_XY')] = bs[3]
            xtra[idx('BS_YZ')] = bs[4]
            xtra[idx('BS_XZ')] = bs[5]
            xtra[idx('SIGE')] = self.eqv(stress_final - bs)
            return stress_final, xtra, self.constant_jacobian


    def eqv(self, sig):
        # Returns sqrt(3 * rootj2) = sig_eqv = q
        s = sig - sig[:3].sum() / 3.0 * np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
        return 1. / ROOT23 * np.sqrt(np.dot(s[:3], s[:3]) + 2 * np.dot(s[3:], s[3:]))
