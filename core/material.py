import os
import sys

import numpy as np

from core.runtime import opts
from core.logger import Logger, ConsoleLogger
from utils.errors import *
from utils.constants import DEFAULT_TEMP
from utils.misc import load_file
from utils.fortran.mml_i import FIO
import utils.mmlabpack as mmlabpack
from utils.variable import Variable, VAR_ARRAY, VAR_SCALAR
from utils.data_containers import Parameters
from materials.aba.mml_i import ABAMATS
from matmodlab import UMATS, PKG_D, MATLIB, MML_MFILE

try:
    from lib.visco import visco as ve
except ImportError:
    ve = None
try:
    from lib.thermomech import thermomech as tm
except ImportError:
    tm = None


class MaterialModel(object):
    """The base material class

    """
    param_defaults = None

    def __init__(self, logger=None):
        self._vars = []
        self.nvisco = 0
        self._viscoelastic = None
        self._trs = None
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
        self.itemp = DEFAULT_TEMP
        if not hasattr(self, "param_names"):
            raise AttributeError("{0}: param_names not defined".format(self.name))
        if logger is None:
            logger = Logger()
        self.logger = logger

    @classmethod
    def _parameter_names(self, n):
        """Return the parameter names for the model

        """
        return [n.split(":")[0].upper() for n in self.param_names]

    @property
    def logger(self):
        return self._logger

    @logger.setter
    def logger(self, new_logger):
        try:
            new_logger.write
            new_logger.warn
            new_logger.error
        except AttributeError, TypeError:
            raise TypeError("attempting to assign a non logger "
                            "to the {0} material logger".format(self.name))
        self._logger = new_logger

    @property
    def initial_temp(self):
        return self.itemp

    def setup_new_material(self, parameters, depvar, initial_temp):
        """Set up the new material

        """
        param_names = self._parameter_names(len(parameters))
        nprops = len(param_names)
        self.itemp = initial_temp
        self.logger.write("setting up {0} material".format(self.name))

        if self.name in ABAMATS:
            # parameters are given as an array
            if not isinstance(parameters, (list, tuple, np.ndarray)):
                raise UserInputError("abaqus parameters must be an array")
            params = np.append(parameters, [0])

        else:
            if self.param_defaults:
                params = np.array(self.param_defaults)
            else:
                params = np.zeros(nprops)
            if not isinstance(parameters, dict):
                raise UserInputError("expected parameters to be a dict")
            for (key, value) in parameters.items():
                K = key.upper()
                if K not in param_names:
                    raise UserInputError("{0}: unrecognized parameter "
                                         "for model {1}".format(key, model))
                params[param_names.index(K)] = value

        self.iparams = Parameters(param_names, params)
        self.params = Parameters(param_names, params)
        try:
            self.setup(self.params, depvar)
        except TypeError:
            self.setup()

        if self._viscoelastic is not None:
            if ve is None:
                self.logger.error("attempting visco analysis but "
                                  "visco.so not imported")

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
                self.logger.error("attempting thermal analysis but "
                                  "thermomech.so not imported")

            self.exp_params = self._expansion.data

        if self.visco_params is not None:
            comm = (self.logger.error, self.logger.write, self.logger.warn)
            ve.propcheck(self.visco_params, *comm)

        self.register_variable("XTRA", VAR_ARRAY, keys=self.xkeys, ivals=self.xinit)

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
        print [x for l in [v.keys for v in self.variables] for x in l]
        sys.exit()
        return [x for l in [v.keys for v in self.variables] for x in l]

    def set_constant_jacobian(self):
        if not self.bulk_modulus:
            self.logger.warn("{0}: bulk modulus not defined".format(self.name))
            return
        if not self.shear_modulus:
            self.logger.warn("{0}: shear modulus not defined".format(self.name))
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
        temp = self.initial_temp
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
            sigm, xtram, stif = self.compute_updated_state(time, dtime, temp,
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

    def compute_updated_state(self, time, dtime, temp, dtemp, F0, F, stran, d,
            elec_field, user_field, stress, statev, disp=0, v=None, last=False):
        """Update the material state

        """
        if disp == 2 and self.constant_j:
            # only jacobian requested
            return self.constant_jacobian

        N = self.nxtra
        comm = (self.logger.error, self.logger.write, self.logger.warn)

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
            comm = (self.logger.error, self.logger.write, self.logger.warn)
            ve.viscoini(self.visco_params, x, *comm)
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

    @property
    def num_prop(self):
        return len(self.params)

    @property
    def num_xtra(self):
        return self.nxtra

# ----------------------------------------- Material Model Factory Method --- #
def Material(model, parameters=None, depvar=None, constants=None,
             source_files=None, source_directory=None, initial_temp=None,
             logger=None):
    """Material model factory method

    """
    # switch model, if requested
    if logger is None:
        logger = Logger()

    if opts.switch:
        logger.warn("switching {0} for {1}".format(model, opts.switch))
        model = opts.switch

    from core.builder import Builder
    if initial_temp is None:
        initial_temp = DEFAULT_TEMP

    m = model.lower()
    if parameters is None:
        raise InputError("{0}: required parameters not given".format(model))

    if m in ABAMATS:
        # Abaqus model
        lib = m

        # Check input
        if constants is None:
            raise UserInputError("abaqus material expected keyword constants")
        constants = int(constants)
        if len(parameters) != constants:
            raise UserInputError("len(parameters) != constants")
        parameters = np.array([float(x) for x in parameters])

        lib_info = ABAMATS[lib]

        # Check if model is already built
        so_lib = os.path.join(PKG_D, lib + ".so")
        source_files = get_aba_sources(source_files, source_directory)
        lib_info["source_files"].extend(source_files)
        if not os.path.isfile(so_lib):
            Builder.build_umat(lib, lib_info["source_files"],
                               verbosity=opts.verbosity)
        if not os.path.isfile(so_lib):
            raise ModelLibNotFoundError(model)

    else:
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
    material = mat(logger=logger)
    material.setup_new_material(parameters, depvar, initial_temp)

    return material


def get_aba_sources(source_files, source_directory):
    """Get the source files for the abaqus umat

    """
    if source_files:
        if source_directory:
            source_files = [os.path.join(source_directory, f)
                            for f in source_files]
        for (i, source_file) in enumerate(source_files):
            if not os.path.isfile(source_file):
                raise InputError("{0}: source file not "
                                 "found".format(source_file))
            source_files[i] = os.path.realpath(source_file)
    else:
        for ext in (".for", ".f", ".f90"):
            source_file = os.path.join(os.getcwd(), "umat" + ext)
            if os.path.isfile(source_file):
                break
            source_file = os.path.join(os.getcwd(), "umat" + ext.upper())
            if os.path.isfile(source_file):
                break
        else:
            raise InputError("umat.[f,for,f90] source file not found")
        source_files = [source_file]

    return source_files


def find_materials():
    from materials.builtin import BUILTIN
    mat_libs = BUILTIN
    errors = []
    for d in UMATS:
        f = os.path.join(d, MML_MFILE)
        if not os.path.isfile(f):
            ConsoleLogger.warn("{0} not found for {1}".format(MML_MFILE, d))
            continue
        info = load_file(info_file)
        try:
            libs = info.material_libraries()
        except AttributeError:
            continue

        for lib in libs:
            if lib in mat_libs:
                ConsoleLogger.error("{0}: duplicate material "
                                    "library".format(lib))
                errors.append(lib)
                continue
            mat_libs.update({lib: libs[lib]})
        del sys.modules[os.path.splitext(MML_MFILE)[0]]

    if errors:
        raise DuplicateExtModule(", ".join(errors))

    for lib in mat_libs:
        if not mat_libs[lib].get("source_files"):
            continue
        I = [mat_libs[lib]["source_directory"]]
        for d in mat_libs[lib].get("include_dirs", []):
            if d and d not in I:
                I.append(d)
        mat_libs[lib]["source_files"].append(FIO)
        mat_libs[lib]["include_dirs"] = I

    return mat_libs

MATERIALS = find_materials()
