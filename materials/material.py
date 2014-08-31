import numpy as np

from core.mmlio import Error1, log_warning, log_error, log_message
from materials.parameters import Parameters
import utils.mmlabpack as mmlabpack
try:
    from lib.visco import visco as ve
except ImportError:
    ve = None
try:
    from lib.thermomech import thermomech as tm
except ImportError:
    tm = None

class Material(object):
    def __init__(self):
        self.mtldb = None
        self.nparam = 0
        self.ndata = 0
        self.nxtra = 0
        self.nvisco = 0
        self._constant_jacobian = False
        self.bulk_modulus = None
        self.shear_modulus = None
        self.J0 = None
        self.xinit = np.zeros(self.nxtra)
        self.mtl_variables = []
        self.initialized = True
        self.use_constant_jacobian = False
        self.visco_params = None
        self.exp_params = None
        if not hasattr(self, "param_names"):
            raise Error1("{0}: param_names not defined".format(self.name))
        self._verify_param_names()

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

        if hasattr(cls, 'param_defaults'):
            param_defaults = np.array(cls.param_defaults)
            if len(set(parse_table.values())) != len(param_defaults):
                raise Error1("{0}: len(param_defaults) != len(param_names)".
                                                          format(self.name))
        else:
            param_defaults = np.zeros(len(set(parse_table.values())))

        return parse_table, param_defaults, cls.param_names

    @staticmethod
    def _fmt_param_name_aliases(s, mode=0):
        s = [n.upper() for n in s.split(":")]
        if mode == 0:
            return s[0], s[1:]
        if mode == -1:
            return s[0]
        return ":".join(s)

    def _verify_param_names(self):
        registered_params = []
        self.nparam = len(self.param_names)
        for idx, name in enumerate(self.param_names):
            name, aliases = self._fmt_param_name_aliases(name)
            if name in registered_params:
                raise Error1("{0}: param already registered".format(name))
            registered_params.append(name)
            for alias in aliases:
                if alias in registered_params:
                    raise Error1("{0}: non-unique param alias".format(alias))
                registered_params.append(name)

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

        self.J0 = np.zeros((6, 6))
        threek = 3. * self.bulk_modulus
        twog = 2. * self.shear_modulus
        nu = (threek - twog) / (2. * threek + twog)
        c1 = (1. - nu) / (1. + nu)
        c2 = nu / (1. + nu)

        # set diagonal
        for i in range(3):
            self.J0[i, i] = threek * c1
        for i in range(3, 6):
            self.J0[i, i] = twog

        # off diagonal
        (self.J0[0, 1], self.J0[0, 2],
         self.J0[1, 0], self.J0[1, 2],
         self.J0[2, 0], self.J0[2, 1]) = [threek * c2] * 6
        return

    def register_mtl_variable(self, var, vtype="SCALAR", units=None):
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

    def xidx(self, key):
        return getattr(self, "_x{0}".format(key), None)

    def get_initial_jacobian(self):
        """Get the initial Jacobian numerically

        """
        d = np.zeros(6)
        stress = np.zeros(6)
        time = 0.
        dtime = 1.
        F0 = np.eye(3).reshape(9,)
        F = np.eye(3).reshape(9,)
        stran = np.zeros(6)
        elec_field = np.zeros(3)
        temp = self._initial_temperature
        dtemp = 0.
        user_field = 0.
        return self.numerical_jacobian(time, dtime, temp, dtemp, F0, F, stran,
            d, elec_field, user_field, stress, self.xinit, range(6))

    def numerical_jacobian(self, time, dtime, temp, dtemp, F0, F, stran, d,
                           elec_field, user_field, stress, xtra, v):
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
            return self.constant_jacobian

        # local variables
        nv = len(v)
        deps =  np.sqrt(np.finfo(np.float64).eps)
        Jsub = np.zeros((nv, nv))
        dtime = 1 if dtime < 1.e-12 else dtime

        for i in range(nv):
            # perturb forward
            Dp = d.copy()
            Dp[v[i]] = d[v[i]] + (deps / dtime) / 2.
            Fp, Ep = mmlabpack.update_deformation(dtime, 0., F, Dp)
            sigp = stress.copy()
            xtrap = xtra.copy()
            sigp, xtrap, stif = self.compute_updated_state(time, dtime, temp,
                dtemp, F0, Fp, Ep, Dp, elec_field, user_field, sigp, xtrap)

            # perturb backward
            Dm = d.copy()
            Dm[v[i]] = d[v[i]] - (deps / dtime) / 2.
            Fm, Em = mmlabpack.update_deformation(dtime, 0., F, Dm)
            sigm = stress.copy()
            xtram = xtra.copy()
            sigm, xtram, stif = self.compute_updated_state( time, dtime, temp,
                dtemp, F0, Fm, Em, Dm, elec_field, user_field, sigm, xtram)

            # compute component of jacobian
            Jsub[i, :] = (sigp[v] - sigm[v]) / deps

            continue

        return Jsub

    def isparam(self, param_name):
        return getattr(self.params, param_name.upper(), False)

    def parameters(self, ival=False, names=False):
        if names:
            return [self._fmt_param_name_aliases(p, mode=1)
                    for p in self.param_names]
        if ival:
            return self.iparams
        return self.params

    def setup_new_material(self, params):
        # For some reason we need to clean the param name aliases.
        self.iparams = np.array(params)
        names = [self._fmt_param_name_aliases(p, mode=-1)
                 for p in params.names]
        self.params = Parameters(names, np.array(params), params.modelname)
        self.setup()

        if self._viscoelastic is not None:
            if ve is None:
                log_error("attempting visco analysis but visco.so not imported")

            # setup viscoelastic params
            self.visco_params = np.zeros(24)
            # starting location of G and T Prony terms
            self.viscopoint = (4, 14)
            n = self._viscoelastic.nprony
            I, J = self.viscopoint
            self.visco_params[I:I+n] = self._viscoelastic.data[:, 0]
            self.visco_params[J:J+n] = self._viscoelastic.data[:, 1]
            # Ginf
            self.visco_params[3] = self._viscoelastic.Ginf

            # allocate storage for shift factors
            for i in range(2):
                self.register_mtl_variable("SHIFT_{0}".format(i+1))

            m = {0: "XX", 1: "YY", 2: "ZZ", 3: "XY", 4: "YZ", 5: "XZ"}
            # register material variables
            for i in range(6):
                self.register_mtl_variable("TE_{0}".format(m[i]))

            # visco elastic model supports up to 10 Prony series terms,
            # allocate storage for stress corresponding to each
            for l in range(10):
                for i in range(6):
                    self.register_mtl_variable("H{0}_{1}".format(l+1, m[i]))

            # Now, allocate storage
            self.nvisco = len(self.material_variables) - self.nxtra
            xinit = np.append(self.initial_state, np.zeros(self.nvisco))
            self.adjust_initial_state(xinit)

        if self._trs is not None:
            self.visco_params[0] = self._trs.data[1] # C1
            self.visco_params[1] = self._trs.data[2] # C2
            self.visco_params[2] = self._trs.temp_ref # REF TEMP

        if self._expansion is not None:
            if tm is None:
                log_error("attempting thermal analysis but "
                          "thermomech.so not imported")

            self.exp_params = self._expansion.data

        if self.visco_params is not None:
            ve.propcheck(self.visco_params, log_error, log_message, log_warning)

    def setup(self, *args, **kwargs):
        raise Error1("setup must be provided by model")

    def update_state(self, *args, **kwargs):
        raise Error1("update_state must be provided by model")

    def compute_updated_state(self, time, dtime, temp, dtemp, F0, F, stran, d,
        elec_field, user_field, stress, xtra, disp=0, v=None, last=False):
        """Update the material state

        """
        N = self.nxtra
        comm = (log_error, log_message, log_warning)

        # Mechanical deformation
        Fm, Em = F, stran

        if self.exp_params is not None:
            # thermal expansion: get mechanical deformation
            Fm = tm.mechdef(self.exp_params, temp, dtemp, F.reshape(3,3), *comm)
            Fm = Fm.reshape(9,)
            Em = mmlabpack.update_strain(self._kappa, Fm)

        rho = 1.
        energy = 1.
        sig, xtra[:N], stif = self.update_state(time, dtime, temp, dtemp,
            energy, rho, F0, F, stran, d, elec_field, user_field, stress,
            xtra[:N], last=last, mode=0)

        if self.visco_params is not None:
            # get visco correction
            X = xtra[N:]
            I, J = self.viscopoint
            sig = ve.viscorelax(dtime, time, temp, dtemp, self.visco_params,
                                F.reshape(3,3), X, sig, *comm)
            xtra[N:] = X[:]

        if v is not None:
            stif = stif[[[i] for i in v], v]

        if disp == 2:
            return stif

        elif disp == 1:
            return sig, xtra

        return sig, xtra, stif

    def adjust_initial_state(self, *args, **kwargs):
        self.set_initial_state(args[0])

    def initialize(self, temp, user_field):
        """Call the material with initial state

        """
        N = self.nxtra
        xtra = self.initial_state
        if self.visco_params is not None:
            # initialize the visco variables
            x = xtra[N:]
            ve.viscoini(self.visco_params, x,
                        log_error, log_message, log_warning)
            xtra[N:] = x
        time = 0.
        dtime = 1.
        dtemp = 0.
        F0 = np.eye(3).reshape(9,)
        F = np.eye(3).reshape(9,)
        stran = np.zeros(6)
        d = np.zeros(6)
        stress = np.zeros(6)
        elec_field = np.zeros(3)
        stress, xtra, stif = self.compute_updated_state(time, dtime, temp, dtemp,
            F0, F, stran, d, elec_field, user_field, stress, xtra)
        return stress, xtra, stif

    def set_initial_state(self, xtra):
        self.xinit = np.array(xtra)

    @property
    def initial_state(self):
        return self.xinit

    @property
    def material_variables(self):
        return self.mtl_variables

    @property
    def constant_jacobian(self):
        return self.J0
