import numpy as np
import os
import numpy as np
from project import UMATS, PKG_D, MATLIB
from utils.misc import load_file
from utils.errors import *

import utils.mmlabpack as mmlabpack
from utils.variable import Variable, VAR_ARRAY, VAR_SCALAR
from utils.data_containers import Parameters
from utils.mmlio import log_message, log_warning, log_error
from materials.matdb import MATERIALS
from core.builder import Builder

try:
    from lib.visco import visco as ve
except ImportError:
    ve = None
try:
    from lib.thermomech import thermomech as tm
except ImportError:
    tm = None


MAT_D = [os.path.join(MATLIB, d) for d in os.listdir(MATLIB)
         if os.path.isdir(os.path.join(MATLIB, d))]
ABA = ("umat", "uhyper", "uanisohyper_inv")


class MaterialModel(object):
    """The base material class

    """
    _vars = []

    def __init__(self):
        self.nvisco = 0
        self._viscoelastic = None
        self._trs = None
        self._vars = []
        self._expansion = None
        self.bulk_modulus = None
        self.shear_modulus = None
        self.J0 = None
        self.nxtra = 0
        self.xinit = np.zeros(self.nxtra)
        self.xkeys = []
        self.initialized = True
        self.visco_params = None
        self.exp_params = None
        self.constant_j = False
        if not hasattr(self, "param_names"):
            raise AttributeError("{0}: param_names not defined".format(self.name))

    @classmethod
    def _parameter_names(self):
        """Return the parameter names for the model

        """
        return [n.split(":")[0].upper() for n in self.param_names]

    def setup_new_material(self, parameters):
        """Set up the new material

        """
        param_names = self._parameter_names()
        nprops = len(param_names)
        params = np.zeros(nprops)
        for (key, value) in parameters.items():
            if key.upper() not in param_names:
                raise ValueError("{0}: unrecognized parameter "
                                 "for model {1}".format(key, model))
            params[param_names.index(key)] = value

        self.iparams = Parameters(param_names, params)
        self.params = Parameters(param_names, params)
        self.setup()
        self.register_variable("XTRA", VAR_ARRAY, keys=self.xkeys, ivals=self.xinit)

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
                self.register_variable("SHIFT_{0}".format(i+1), VAR_SCALAR)

            m = {0: "XX", 1: "YY", 2: "ZZ", 3: "XY", 4: "YZ", 5: "XZ"}
            # register material variables
            for i in range(6):
                self.register_variable("TE_{0}".format(m[i]), VAR_SCALAR)

            # visco elastic model supports up to 10 Prony series terms,
            # allocate storage for stress corresponding to each
            for l in range(10):
                for i in range(6):
                    self.register_variable("H{0}_{1}".format(l+1, m[i]), VAR_SCALAR)

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

        self.set_constant_jacobian()

    def register_variable(self, var_name, var_type, keys=None, ivals=None):
        if keys is not None:
            length = len(ivals)
            self._vars.append(Variable(var_name, var_type, initial_value=ivals,
                                       length=length, keys=keys))

        else:
            self._vars.append(Variable(var_name, var_type))

    @property
    def variables(self):
        return self._vars

    @property
    def plot_keys(self):
        return [x for l in [v.keys for v in self.variables] for x in l]

    def set_constant_jacobian(self):
        if not self.bulk_modulus:
            log_warning("{0}: bulk modulus not defined".format(self.name))
            return
        if not self.shear_modulus:
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

    def register_xtra_variables(self, keys, values, mig=False):
        if self.nxtra:
            raise ValueError("Register extra variables at most once")
        if mig:
            keys = [" ".join(x.split())
                    for x in "".join(keys).split("|") if x.split()]
        self.nxtra = len(keys)
        if len(values) != len(keys):
            raise ValueError("len(values) != len(keys)")
        self.xkeys = keys
        self.xinit = np.array(values)

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
        if self.constant_j:
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

    @property
    def parameter_names(self):
        """Return the parameter names for the model

        """
        return [n.split(":")[0].upper() for n in self.param_names]

    @property
    def parameters(self):
        return self.params

    @property
    def initial_parameters(self):
        return self.iparams

    def setup(self, *args, **kwargs):
        raise NotImplementedError

    def update_state(self, *args, **kwargs):
        raise NotImplementedError

    def compute_updated_state(self, time, dtime, temp, dtemp, F0, F,
        stran, d, elec_field, user_field, stress, statev,
        disp=0, v=None, last=False):
        """Update the material state

        """
        if disp == 2 and self.constant_j:
            # only jacobian requested
            return self.constant_jacobian

        N = self.nxtra
        comm = (log_error, log_message, log_warning)

        sig = np.array(stress)
        xtra = np.array(statev)

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
            energy, rho, F0, F, stran, d, elec_field, user_field, sig,
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

    def initialize(self, stress, xtra, temp, user_field):
        """Call the material with initial state

        """
        N = self.nxtra
        if xtra is None:
            xtra = self.initial_state
        if stress is None:
            stress = np.zeros(6)

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
        elec_field = np.zeros(3)
        stress, xtra, stif = self.compute_updated_state(time, dtime, temp, dtemp,
            F0, F, stran, d, elec_field, user_field, stress, xtra)

        self.set_initial_state(stress, xtra)

    def set_initial_state(self, xtra, stress=None):
        self.xinit = np.array(xtra)
        self.sigini = np.zeros(6)
        if stress is not None:
            self.sigini[:] = np.array(stress)

    def adjust_initial_state(self, *args, **kwargs):
        self.xinit = np.array(args[0])

    @property
    def initial_state(self):
        return self.xinit

    @property
    def initial_stress(self):
        return self.sigini

    @property
    def constant_jacobian(self):
        return self.J0


# ----------------------------------------- Material Model Factory Method --- #
def Material(model, parameters=None, constants=None, depvar=None):
    """Material model factory method

    """
    m = model.lower()
    if m not in ABA and parameters is None:
        raise ValueError("{0}: required parameters not given".format(model))

    for lib in MATERIALS:
        if lib.lower() == m:
            break
    else:
        raise MatModelNotFoundError(model)

    # Check that the material is built
    lib_info = MATERIALS[lib]
    if lib_info.get("source_files"):
        so_lib = os.path.join(PKG_D, lib + ".so")
        if not os.path.isfile(so_lib):
            # try building it first
            Builder.build_material(lib, lib_info)
        if not os.path.isfile(so_lib):
            raise ModelLibNotFoundError(model)

    # Instantiate the material
    interface = load_file(lib_info["interface"])
    mat = getattr(interface, lib_info["class"])
    material = mat()
    material.setup_new_material(parameters)

    return material
