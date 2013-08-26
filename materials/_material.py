import numpy as np

from base.io import Error1

class Material(object):

    def __init__(self):
        self.ndata = 0
        self.nxtra = 0
        self.xtra = np.zeros(self.nxtra)
        self.xinit = np.zeros(self.nxtra)
        self.mtl_variables = []
        self.param_map = {}
        self.initialized = True

    def register_parameters(self, *parameters):
        self.nparam = len(parameters)
        for idx, name in enumerate(parameters):
            name = name.upper()
            self.param_map[name] = idx
            setattr(self, name, idx)

    def register_mtl_variable(self, var, vtype):
        self.mtl_variables.append((var, vtype))

    def register_xtra_variables(self, keys, mig=False):
        if self.nxtra:
            raise Error1("Register extra variables at most once")
        self.nxtra = len(keys)
        if mig:
            keys = [" ".join(x.split())
                    for x in "".join(keys).split("|") if x.split()]
        for key in keys:
            self.mtl_variables.append((key, "SCALAR"))

    def jacobian(self, dt, d, sig, xtra, v):
        """Numerically compute material Jacobian by a centered difference scheme.

        Returns
        -------
        Js : array_like
          Jacobian of the deformation J = dsig / dE

        Notes
        -----
        The submatrix returned is the one formed by the intersections of the
        rows and columns specified in the vector subscript array, v. That is,
        Js = J[v, v]. The physical array containing this submatrix is
        assumed to be dimensioned Js[nv, nv], where nv is the number of
        elements in v. Note that in the special case v = [1,2,3,4,5,6], with
        nv = 6, the matrix that is returned is the full Jacobian matrix, J.

        The components of Js are computed numerically using a centered
        differencing scheme which requires two calls to the material model
        subroutine for each element of v. The centering is about the point eps
        = epsold + d * dt, where d is the rate-of-strain array.

        History
        -------
        This subroutine is a python implementation of a routine by the same
        name in Tom Pucick's MMD driver.

        Authors
        -------
        Tom Pucick, original fortran implementation in the MMD driver
        Tim Fuller, Sandial National Laboratories, tjfulle@sandia.gov

        """

        # local variables
        nv = len(v)
        deps =  np.sqrt(np.finfo(np.float).eps)
        Jsub = np.zeros((nv, nv))
        dt = 1 if dt == 0. else dt

        for i in range(nv):
            # perturb forward
            dp = d.copy()
            dp[v[i]] = d[v[i]] + (deps / dt) / 2.
            sigp = sig.copy()
            xtrap = xtra.copy()
            sigp, xtrap = self.update_state(dt, dp, sigp, xtrap)

            # perturb backward
            dm = d.copy()
            dm[v[i]] = d[v[i]] - (deps / dt) / 2.
            sigm = sig.copy()
            xtram = xtra.copy()
            sigm, xtram = self.update_state(dt, dm, sigm, xtram)

            # compute component of jacobian
            Jsub[i, :] = (sigp[v] - sigm[v]) / deps

            continue

        return Jsub

    def isparam(self, param_name):
        return param_name.upper() in self.param_map

    def parameter_index(self, param_name):
        return self.param_map.get(param_name.upper())

    def setup(self, *args, **kwargs):
        raise Error1("setup must be provided by model")

    def update_state(self, *args, **kwargs):
        raise Error1("update_state must be provided by model")

    def initialize_material(self, *args, **kwargs):
        return

    def adjust_initial_state(self, xtra):
        return xtra

    def call_material_zero_state(self, stress, xtra, *args):
        dt = 1.
        d = np.zeros(6)
        return self.update_state(dt, d, stress, xtra, *args)

    def set_param_vals(self, param_vals):
        self._param_vals = np.array(param_vals)

    def set_initial_state(self, xtra):
        self.xinit = np.array(xtra)

    def initial_state(self):
        return self.xinit

    def material_variables(self):
        return self.mtl_variables

    def constant_jacobian(self, v=np.arange(6)):
        jac = np.zeros((6, 6))
        threek = 3. * self.bulk_modulus
        twog = 2. * self.shear_modulus
        nu = (threek - twog) / (2. * threek + twog)
        c1 = (1. - nu) / (1. + nu)
        c2 = nu / (1. + nu)

        # set diagonal
        for i in range(3):
            jac[i, i] = threek * c1
        for i in range(3, 6):
            jac[i, i] = twog

        # off diagonal
        (jac[0, 1], jac[0, 2],
         jac[1, 0], jac[1, 2],
         jac[2, 0], jac[2, 1]) = [threek * c2] * 6

        return jac[[[x] for x in v], v]

    def param_vals(self):
        return self._param_vals

    def params(self):
        return sorted(self.param_map, key=lambda x: self.param_map[x])
