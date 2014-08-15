import numpy as np

from materials.parameters import Parameters
from materials.material import Material
from core.mmlio import Error1, log_error, log_message

class VonMises(Material):
    name = "vonmises"
    param_names = ["K",    # Linear elastic bulk modulus
                   "G",    # Linear elastic shear modulus
                   "Y0",   # yield stress in uniaxial tension
                           #    (yield in tension) = sqrt(3) * (yield in shear)
                           #                       = sqrt(3) * sqrt(J2)
                   "H",    # Hardening modulus
                   "BETA", # isotropic/kinematic hardening parameter
                           #    BETA = 0 for isotropic hardening
                           #    0 < BETA < 1 for mixed hardening
                           #    BETA = 1 for kinematic hardening
                  ]
    param_defaults = [0.0, 0.0, 1.0e30, 0.0, 0.0]

    def setup(self):
        """Set up the von Mises material

        """
        self.use_constant_jacobian = True
        # Check inputs
        if self.params.modelname == self.name:
            K = self.params["K"]
            G = self.params["G"]
            Y0 = self.params["Y0"]
            H = self.params["H"]
            BETA = self.params["BETA"]

            if K <= 0.0: log_error = "Bulk modulus K must be positive"
            if G <= 0.0: log_error = "Shear modulus G must be positive"
            nu = (3.0 * K - 2.0 * G) / (6.0 * K + 2.0 * G)
            if nu > 0.5: log_error = "Poisson's ratio > .5"
            if nu < -1.0: log_error = "Poisson's ratio < -1."
            if nu < 0.0: log_message = "#---- WARNING: negative Poisson's ratio"
            if Y0 == 0.0: Y0 = 1.0e99

        elif self.params.modelname == 'elastic':
            print("model '{0}' mimicing '{1}'".format(self.name, self.params.modelname))
            K = self.params["K"]
            G = self.params["G"]
            Y0 = 1.0e99
            H = 0.0
            BETA = 0.0

        newparams = [K, G, Y0, H, BETA]
        newnames = ["K", "G", "Y0", "H", "BETA"]
        self.params = Parameters(newnames, newparams, self.name)

        self.bulk_modulus = self.params["K"]
        self.shear_modulus = self.params["G"]

        # Register State Variables
        self.sv_names = ["EQPS", "Y",
                         "BS_XX", "BS_YY", "BS_ZZ", "BS_XY", "BS_XZ", "BS_YZ",
                         "SIGE"]
        sv_values = [0.0, Y0,
                     0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                     0.0]

        self.register_xtra_variables(self.sv_names)
        self.set_initial_state(sv_values)

    def update_state(self, dt, d, stress, xtra, *args, **kwargs):
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
        idx = lambda x: self.sv_names.index(x.upper())
        bs = np.array([xtra[idx('BS_XX')], xtra[idx('BS_YY')], xtra[idx('BS_ZZ')],
                       xtra[idx('BS_XY')], xtra[idx('BS_YZ')], xtra[idx('BS_XZ')]])
        yn = xtra[idx('Y')]

        de = d * dt

        iso = de[:3].sum() / 3.0 * np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
        dev = de - iso

        stress_trial = stress + 3.0 * self.bulk_modulus * iso + 2.0 * self.shear_modulus * dev

        xi_trial = stress_trial - bs
        xi_trial_eqv = self.eqv(xi_trial)

        if xi_trial_eqv <= yn:
            xtra[idx('SIGE')] = xi_trial_eqv
            return stress_trial, xtra
        else:
            N = xi_trial - xi_trial[:3].sum() / 3.0 * np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
            N = N / (np.sqrt(2.0 / 3.0) * xi_trial_eqv)
            deqps = (xi_trial_eqv - yn) / (3.0 * self.shear_modulus + self.params["H"])
            dps = np.sqrt(3.0 / 2.0) * deqps * N

            stress_final = stress_trial - 2.0 * self.shear_modulus * np.sqrt(3.0 / 2.0) * deqps * N

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
            return stress_final, xtra


    def eqv(self, sig):
        # Returns sqrt(3 * rootj2) = sig_eqv = q
        s = sig - sig[:3].sum() / 3.0 * np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
        return np.sqrt(3.0 / 2.0) * np.sqrt(np.dot(s[:3], s[:3]) + 2 * np.dot(s[3:], s[3:]))
