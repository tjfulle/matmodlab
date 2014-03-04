import numpy as np

from core.mmlio import Error1, log_warning
try:
    from lib.mmlabpack import mmlabpack
except ImportError:
    import utils.mmlabpack as mmlabpack

class Material(object):
    def __init__(self):
        self.mtldb = None
        self.nparam = 0
        self.ndata = 0
        self.nxtra = 0
        self._constant_jacobian = False
        self.bulk_modulus = None
        self.shear_modulus = None
        self._jacobian = None
        self.xinit = np.zeros(self.nxtra)
        self.mtl_variables = []
        self._param_map = {}
        self.initialized = True
        if not hasattr(self, "param_names"):
            raise Error1("{0}: param_names not defined".format(self.name))
        self._register_parameters()

    @staticmethod
    def _fmt_param_name_aliases(s, mode=0):
        s = [n.upper() for n in s.split(":")]
        if mode == 0:
            return s[0], s[1:]
        return ":".join(s)

    def _register_parameters(self):
        self.nparam = len(self.param_names)
        for idx, name in enumerate(self.param_names):
            name, aliases = self._fmt_param_name_aliases(name)
            if name in self._param_map:
                raise Error1("{0}: param already registered".format(name))
            self._param_map[name] = idx
            setattr(self, name, idx)
            for alias in aliases:
                if alias in self._param_map:
                    raise Error1("{0}: non-unique param alias".format(alias))
                self._param_map[alias] = idx
                setattr(self, alias, idx)

    def set_options(self, **kwargs):
        for (k, v) in kwargs.items():
            setattr(self, "_{0}".format(k), v)

    def set_constant_jacobian(self):
        if not self.bulk_modulus:
            # raise Error1("{0}: bulk modulus not defined".format(self.name))
            log_warning("{0}: bulk modulus not defined".format(self.name))
            return
        if not self.shear_modulus:
            # raise Error1("{0}: shear modulus not defined".format(self.name))
            log_warning("{0}: shear modulus not defined".format(self.name))
            return

        self._jacobian = np.zeros((6, 6))
        threek = 3. * self.bulk_modulus
        twog = 2. * self.shear_modulus
        nu = (threek - twog) / (2. * threek + twog)
        c1 = (1. - nu) / (1. + nu)
        c2 = nu / (1. + nu)

        # set diagonal
        for i in range(3):
            self._jacobian[i, i] = threek * c1
        for i in range(3, 6):
            self._jacobian[i, i] = twog

        # off diagonal
        (self._jacobian[0, 1], self._jacobian[0, 2],
         self._jacobian[1, 0], self._jacobian[1, 2],
         self._jacobian[2, 0], self._jacobian[2, 1]) = [threek * c2] * 6
        return

    def register_mtl_variable(self, var, vtype, units=None):
        self.mtl_variables.append((var, vtype))

    def register_xtra_variables(self, keys, mig=False):
        if self.nxtra:
            raise Error1("Register extra variables at most once")
        if mig:
            keys = [" ".join(x.split())
                    for x in "".join(keys).split("|") if x.split()]
        self.nxtra = len(keys)
        for (i, key) in enumerate(keys):
            self.register_mtl_variable(key, "SCALAR")
            setattr(self, "_x{0}".format(key), i)

    def jacobian(self, dt, d, sig, xtra, v, *args):
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
        if self._constant_jacobian:
            return self.constant_jacobian(v)

        # local variables
        nv = len(v)
        deps =  np.sqrt(np.finfo(np.float).eps)
        Jsub = np.zeros((nv, nv))
        dt = 1 if dt == 0. else dt
        f = args[0]
        _a = [x for x in args[1:]]

        for i in range(nv):
            # perturb forward
            dp = d.copy()
            dp[v[i]] = d[v[i]] + (deps / dt) / 2.
            fp, _ = mmlabpack.update_deformation(dt, 0., f, dp)
            sigp = sig.copy()
            xtrap = xtra.copy()
            ap = [fp] + _a
            sigp, xtrap = self.update_state(dt, dp, sigp, xtrap, *ap)

            # perturb backward
            dm = d.copy()
            dm[v[i]] = d[v[i]] - (deps / dt) / 2.
            fm, _ = mmlabpack.update_deformation(dt, 0., f, dm)
            sigm = sig.copy()
            xtram = xtra.copy()
            am = [fm] + _a
            sigm, xtram = self.update_state(dt, dm, sigm, xtram, *am)

            # compute component of jacobian
            Jsub[i, :] = (sigp[v] - sigm[v]) / deps

            continue

        return Jsub

    def isparam(self, param_name):
        return param_name.upper() in self._param_map

    def parameters(self, ival=False, names=False):
        if names:
            return [self._fmt_param_name_aliases(p, mode=1)
                    for p in self.param_names]
        if ival:
            return self.iparams
        return self.params

    @classmethod
    def param_parse_table(cls):
        n_no_parse = 0
        parse_table = {}
        for (i, param) in enumerate(cls.param_names):
            for n in param.split(":"):
                n = n.strip().lower()
                if n.startswith("-"):
                    # not to be parsed
                    n = n[1:]
                    i = -i
                parse_table[n] = i
        return parse_table

    def setup_new_material(self, params):
        self.iparams = np.array(params)
        self.params = np.array(params)
        self.setup()

    def setup(self, *args, **kwargs):
        raise Error1("setup must be provided by model")

    def update_state(self, *args, **kwargs):
        raise Error1("update_state must be provided by model")

    def initialize_material(self, *args, **kwargs):
        return

    def adjust_initial_state(self, *args, **kwargs):
        return args[0]

    def call_material_zero_state(self, stress, xtra, *args):
        dt = 1.
        d = np.zeros(6)
        return self.update_state(dt, d, stress, xtra, *args)

    def set_initial_state(self, xtra):
        self.xinit = np.array(xtra)

    def initial_state(self):
        return self.xinit

    def material_variables(self):
        return self.mtl_variables

    def constant_jacobian(self, v=np.arange(6)):
        return self._jacobian[[[x] for x in v], v]
