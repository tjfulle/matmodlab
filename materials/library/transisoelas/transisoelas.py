import numpy as np

from materials._material import Material
from core.mmlio import Error1, log_error, log_message

class TransIsoElas(Material):
    name = "transisoelas"
    param_names = ["A0", "A1", "A2", "A3",  # A0 = A1 = 0 for natural state
                   "B0", "B1", "C0", "C1",
                   "V1", "V2", "V3",
                   "K", "G"]

    def setup(self):
        """Set up the Transversely Isotropic Linear Elastic  material

        """
        # Check inputs
        self.ui = dict(zip(self.param_names, self.params))

        # If the user wants a linear elastic primitive:
        if self.ui["K"] > 0.0 and self.ui["G"] > 0.0:
            for key in self.ui.keys():
                if key in ['K', 'G']:
                    continue
                self.ui[key] = 0.0
            self.ui['B0'] = self.ui['K'] - 2.0 / 3.0 * self.ui['G']
            self.ui['A0'] = self.ui['G']

        vec = np.array([self.ui["V1"], self.ui["V2"], self.ui["V3"]])
        vmag = np.sqrt(np.dot(vec, vec))
        vec = vec / vmag if vmag > 0.0 else np.array([1, 0, 0])
        self.M = np.outer(vec, vec)

        self.eps = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.register_xtra_variables(["EPS_XX", "EPS_YY", "EPS_ZZ",
                                      "EPS_XY", "EPS_YZ", "EPS_XZ"])
        self.set_initial_state(list(self.eps))

        self.bulk_modulus = self.ui["B0"]
        self.shear_modulus = self.ui["A0"]

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

        # Handle strain-related tasks
        self.eps = np.array(xtra) + d * dt
        D = np.array([[self.eps[0], self.eps[3], self.eps[5]],
                      [self.eps[3], self.eps[1], self.eps[4]],
                      [self.eps[5], self.eps[4], self.eps[2]]])
        xtra = list(self.eps)

        # Calculate some helper functions
        trD = np.trace(D)
        trMD = np.trace(np.dot(self.M, D))
        alpha0 = self.ui["A0"] + self.ui["B0"] * trD + self.ui["C0"] * trMD
        alpha1 = self.ui["A1"] + self.ui["B1"] * trD + self.ui["C1"] * trMD
        alpha2 = self.ui["A2"]
        alpha3 = self.ui["A3"]

        # Actually calculate the stress
        stress = (alpha0 * np.eye(3, 3) + alpha1 * self.M + alpha2 * D
                 + alpha3 * (np.dot(self.M, D) + np.dot(D, self.M)))

        # Format it so that it plays nice with the model driver
        retstress = np.array([stress[0, 0], stress[1, 1], stress[2, 2],
                              stress[0, 1], stress[1, 2], stress[0, 2]])

        return retstress, xtra


    def jacobian(self, dt, d, stress, xtra, v, *args):
        """Return the constant stiffness
        dt : float
            time step

        d : array_like
            Deformation rate

        stress : array_like
            Stress at beginning of step

        """
        return self.constant_jacobian(v)
