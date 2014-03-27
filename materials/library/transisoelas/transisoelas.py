import numpy as np

from materials.material import Material
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
        #self.ui = dict(zip(self.param_names, self.params))

        # If the user wants a linear elastic primitive:
        if self.params["K"] > 0.0 and self.params["G"] > 0.0:
            for key in self.params.keys():
                if key in ['K', 'G']:
                    continue
                self.params[key] = 0.0
            self.params['B0'] = self.params['K'] - 2.0 / 3.0 * self.params['G']
            self.params['A0'] = self.params['G']

        vec = np.array([self.params["V1"], self.params["V2"], self.params["V3"]])
        vmag = np.sqrt(np.dot(vec, vec))
        vec = vec / vmag if vmag > 0.0 else np.array([1, 0, 0])
        self.M = np.outer(vec, vec)

        self.register_xtra_variables(["EPS_XX", "EPS_YY", "EPS_ZZ",
                                      "EPS_XY", "EPS_YZ", "EPS_XZ"])
        self.set_initial_state(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))

        self.bulk_modulus = self.params["B0"]
        self.shear_modulus = self.params["A0"]

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
        eps = np.array(xtra) + d * dt
        D = np.array([[eps[0], eps[3], eps[5]],
                      [eps[3], eps[1], eps[4]],
                      [eps[5], eps[4], eps[2]]])
        xtra = eps

        # Calculate some helper functions
        trD = np.trace(D)
        trMD = np.trace(np.dot(self.M, D))
        alpha0 = self.params["A0"] + self.params["B0"] * trD + self.params["C0"] * trMD
        alpha1 = self.params["A1"] + self.params["B1"] * trD + self.params["C1"] * trMD
        alpha2 = self.params["A2"]
        alpha3 = self.params["A3"]

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
