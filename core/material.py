import os
import re
import sys
import numpy as np

from utils.errors import *
from materials.completion import *
from core.runtime import opts
import utils.xpyclbr as xpyclbr
import utils.mmlabpack as mmlabpack
from core.configurer import cfgparse
from utils.fortran.product import FIO
from utils.mtldb import read_params_from_db
from utils.misc import load_file, remove, who_is_calling
from utils.data_containers import Parameters
from core.logger import Logger
from utils.variable import Variable, VAR_ARRAY, VAR_SCALAR
from utils.constants import DEFAULT_TEMP, SET_AT_RUNTIME, ENGW
from core.product import MAT_LIB_DIRS, PKG_D, SUPPRESS_USER_ENV
from materials.product import ABA_MATS, USER_MAT, F_MTL_PARAM_DB
np.set_printoptions(precision=2)


class MetaClass(type):
    """metaclass which overrides the "__call__" function"""
    def __call__(cls, *args, **kwargs):
        """Called when you call Class() """
        obj = type.__call__(cls)
        obj.pre_init(*args, **kwargs)
        return obj

def Eye(n):
    # Specialized identity for tensors
    if n == 6:
        return np.array([1, 1, 1, 0, 0, 0], dtype=np.float64)
    if n == 9:
        return np.array([1, 0, 0, 0, 1, 0, 0, 0, 1], dtype=np.float64)
    raise MatModLabError("incorrect n")

class MaterialModel(object):
    """The base material class

    """
    __metaclass__ = MetaClass
    name = None
    lapack = None
    aux_files = None
    prop_names = None
    param_names = None
    completions = None
    source_files = None
    pre_initialized = False

    def __init__(self):
        raise MatModLabError("materials must define own __init__")

    def pre_init(self, *args, **kwargs):
        """Parses parameters from user input and allocates parameter array

        """
        # If no args are sent in, the caller must not want to do the material
        # pre-initialization
        if not args:
            return

        parameters = args[0]
        self._param_name_map = {}
        self._initial_temp = None

        if self.name is None:
            raise MatModLabError("material did not define name attribute")

        if self.param_names is None:
            raise MatModLabError("material did not define param_names attribute")

        mat_mimic = kwargs.pop("mimic", None)
        if mat_mimic is not None:
            # parameters have already been parsed by the model we are
            # mimicking, now we just need to modify them for this model
            if mat_mimic.completions is None:
                raise MatModLabError("{0}: model completion not done.\nthis is "
                                     "likely because the model did not provide "
                                     "a prop_names attribute".format(mimic.name))
            params = self.mimicking(mat_mimic)

        elif self.parameter_names == SET_AT_RUNTIME:
            # some models, like abaqus models, do not have the parameter names
            # set. The user must provide the entire UI array - as an array.

            if kwargs.get("param_names", None) is not None:
                # param_names passed in as argument. parse parameters as a
                # builtin model
                self.parameter_names = kwargs["param_names"]
                params = self._parse_params(parameters)

            else:
                # Check parameters -> must be an array
                if not isinstance(parameters, (list, tuple, np.ndarray)):
                    raise MatModLabError("{0}: parameters must be "
                                         "an array".format(self.name))
                params = np.array([float(x) for x in parameters])
                constants = len(params)

                # check if user sent param_names, or set default
                pnames = lambda n: ["PROP{0:02d}".format(i+1) for i in range(n)]
                self.parameter_names = pnames(constants)

        else:
            # default: param_names set be Material class
            params = self._parse_params(parameters)

        self.iparray = np.array(params)
        if self.prop_names is not None:
            self.completions = complete_properties(self.iparray, self.prop_names)
        self.pre_initialized = True

    def _parse_params(self, parameters):

        if not isinstance(parameters, dict):
            raise MatModLabError("expected parameters to be a dict")

        nprops = len(self.parameter_names)
        params = np.zeros(nprops)

        # populate the parameters array
        errors = 0
        for (key, value) in parameters.items():
            K = key.upper()
            idx = self.parameter_name_map(key)
            if idx is None:
                errors += 1
                Logger("console").error("{0}: unrecognized parameter "
                                        "for model {1}".format(key, self.name))
            try:
                params[idx] = float(value)
            except ValueError:
                errors += 1
                Logger("console").error("parameter {0} must be a float".format(key))

        if errors:
            raise MatModLabError("stopping due to previous errors")

        return params

    def initialize(self, initial_temp=None, file=None, trs=None,
                   expansion=None, viscoelastic=None, logger=None, **kwargs):
        if not self.pre_initialized:
            raise MatModLabError("material not pre-initialized")

        self._vars = []
        self.visco_model = viscoelastic
        self.expansion_model = expansion
        self.trs_model = trs
        self.J0 = None
        self.nxtra = 0
        self.xinit = np.zeros(self.nxtra)
        self.xkeys = []
        self.initialized = True
        self.initial_temp = initial_temp
        self.initial_stress = np.zeros(6)
        self.logger = logger or Logger(self.name, filename=None)
        self._file = file

        self._setup(**kwargs)
        self._initialize()

    def mimicking(self, mat_mimic):
        raise NotImplementedError("mimicking not supported by "
                                  "{0}".format(self.name))

    def _setup(self, **kwargs):
        """Set up the new material

        """
        self.logger.write("setting up {0} material".format(self.name))
        self.iparams = Parameters(self.parameter_names, self.iparray, self.name)
        self.params = Parameters(self.parameter_names, self.iparray, self.name)

        try:
            self.setup(self.params, **kwargs)
        except TypeError:
            self.setup()

        if self.visco_model is not None:
            vk, vd = self.visco_model.setup(self.logger, trs_model=self.trs_model)
            self.augment_xtra(vk, vd)

        if self.expansion_model is not None:
            self.expansion_model.setup()

        self.register_variable("XTRA", VAR_ARRAY, keys=self.xkeys,
                               ivals=self.initial_state)

        self.set_constant_jacobian()

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
            raise MatModLabError("attempting to assign a non logger "
                                 "to the {0} material logger".format(self.name))
        self._logger = new_logger

    @property
    def file(self):
        return self._file

    @file.setter
    def file(self, value):
        self._file = value

    @property
    def initial_temp(self):
        return self._initial_temp

    @initial_temp.setter
    def initial_temp(self, value):
        if value is not None:
            self._initial_temp = value

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

    def set_constant_jacobian(self):

        if self.completions is None:
            # compute stiffness, determine elastic properties, and perform the
            # completion
            self.J0 = self.get_initial_jacobian()
            C = isotropic_part(self.J0)
            lame = C[0,1]
            mu = C[5,5] / 2.
            a = np.array([mu, lame])
            b = [EC_LAME, EC_SHEAR]
            self.completions = complete_properties(a, b)
            return

        self.J0 = np.zeros((6, 6))
        threek = 3. * self.completions[EC_BULK]
        twog = 2. * self.completions[EC_SHEAR]
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
            raise MatModLabError("Register extra variables at most once")
        if mig:
            keys = [" ".join(x.split())
                    for x in "".join(keys).split("|") if x.split()]
        self.nxtra = len(keys)
        if len(values) != len(keys):
            raise MatModLabError("len(values) != len(keys)")
        self.xkeys = keys
        self.initial_state = np.array(values)

    def augment_xtra(self, keys, values):
        """Increase xkeys and initial state -> but not nxtra. Used by the
        visco model to tack on the extra visco variables to the end of the
        xtra array."""
        self.xkeys.extend(keys)
        if len(values) != len(keys):
            raise MatModLabError("len(values) != len(keys)")
        self.initial_state = np.append(self.initial_state, np.array(values))

    def get_initial_jacobian(self):
        """Get the initial Jacobian"""
        time, dtime = 0, 1
        temp, dtemp = self.initial_temp, 0.
        kappa = 0
        F0, F = Eye(9), Eye(9)
        stran, d, elec_field = np.zeros(6), np.zeros(6), np.zeros(3)
        user_field = 0
        ddsdde = self.compute_updated_state(time, dtime, temp, dtemp, kappa,
                       F0, F, stran, d, elec_field, user_field,
                       self.initial_stress, self.initial_state, disp=2)
        return ddsdde

    def numerical_jacobian(self, time, dtime, temp, dtemp, kappa, F0, F, stran, d,
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
            sigp = self.compute_updated_state(time, dtime, temp, dtemp, kappa,
                      F0, Fp, Ep, Dp, elec_field, user_field, sigp, xtrap, disp=3)

            # perturb backward
            Dm = d.copy()
            Dm[v[i]] = d[v[i]] - (deps / dtime) / 2.
            Fm, Em = mmlabpack.update_deformation(dtime, 0., F, Dm)
            sigm = stress.copy()
            xtram = xtra.copy()
            sigm = self.compute_updated_state(time, dtime, temp, dtemp, kappa,
                      F0, Fm, Em, Dm, elec_field, user_field, sigm, xtram, disp=3)

            # compute component of jacobian
            Jsub[i, :] = (sigp[v] - sigm[v]) / deps

            continue

        return Jsub

    @property
    def parameter_names(self):
        try: return [n.split(":")[0].upper() for n in self.param_names]
        except TypeError: return self.param_names

    @parameter_names.setter
    def parameter_names(self, param_names):
        """Set parameter names for models that set parameter names at run time

        """
        self.param_names = [str(p).strip().upper() for p in param_names]

    def parameter_name_map(self, name, default=None):
        """Maps name to the index in the UI array"""
        if name.upper() not in self._param_name_map:
            m = dict([(n.upper(), i) for (i, na) in enumerate(self.param_names)
                      for n in na.split(":")])
            self._param_name_map.update(m)
        return self._param_name_map.get(name.upper(), default)

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

    def compute_updated_state(self, time, dtime, temp, dtemp, kappa, F0, F,
            stran, d, elec_field, user_field, stress, statev,
            disp=0, v=None, last=False):
        """Update the material state

        """
        V = v if v is not None else range(6)
        N = self.nxtra

        sig = np.array(stress)
        xtra = np.array(statev)

        # Mechanical deformation
        Fm, Em = F, stran

        if self.expansion_model is not None:
            # thermal expansion: get mechanical deformation
            Fm, Em = self.expansion_model.update_state(
                self.logger, temp, dtemp, F, kappa)

        rho = 1.
        energy = 1.
        sig, xtra[:N], ddsdde = self.update_state(time, dtime, temp, dtemp,
            energy, rho, F0, Fm, Em, d, elec_field, user_field, sig,
            xtra[:N], last=last, mode=0)

        if self.visco_model is not None:
            # get visco correction
            sig, cfac, xtra[N:] = self.visco_model.update_state(
                self.logger, time, dtime, temp, dtemp, xtra[N:], F, sig)

        if disp == 3:
            return sig

        if ddsdde is None or self.visco_model is not None:
            # material models without an analytic jacobian send the Jacobian
            # back as None so that it is found numerically here. Likewise, we
            # find the numerical jacobian for visco materials - otherwise we
            # would have to convert the the stiffness to that corresponding to
            # the Truesdell rate, pull it back to the reference frame, apply
            # the visco correction, push it forward, and convert to Jaummann
            # rate. It's not as trivial as it sounds...
            ddsdde = self.numerical_jacobian(time, dtime, temp, dtemp, kappa, F0,
                        Fm, Em, d, elec_field, user_field, stress, xtra, V)

        if v is not None and len(v) != ddsdde.shape[0]:
            # if the numerical Jacobian was called, ddsdde is already the
            # sub-Jacobian
            ddsdde = ddsdde[[[i] for i in v], v]

        if last and opts.sqa_stiff:
            # check how close stiffness returned from material is to the numeric
            c = self.numerical_jacobian(time, dtime, temp, dtemp, kappa, F0,
                        Fm, Em, d, elec_field, user_field, stress, xtra, V)
            err = np.amax(np.abs(ddsdde - c)) / np.amax(ddsdde)
            if err > 5.E-03: # .5 percent error
                self.logger.warn("error in material stiffness: "
                                 "{0:.4E} ({1:.2f})".format(err, time), limit=False)

        if disp == 2:
            return ddsdde

        elif disp == 1:
            return sig, xtra

        return sig, xtra, ddsdde

    def _initialize(self):
        """Call the material with initial state

        """
        N = self.nxtra
        xtra = self.initial_state
        if self.visco_model is not None:
            # initialize the visco variables
            xtra[N:] = self.visco_model.initialize(self.logger, xtra[N:])

    @property
    def initial_state(self):
        return self.xinit

    @initial_state.setter
    def initial_state(self, xtra):
        self.xinit = np.array(xtra)

    @property
    def initial_stress(self):
        return self.sigini

    @initial_stress.setter
    def initial_stress(self, value):
        self.sigini = np.array(value)

    @property
    def constant_jacobian(self):
        return self.J0

    @property
    def num_prop(self):
        return len(self.params)

    @property
    def num_xtra(self):
        return self.nxtra


class AbaqusMaterial(MaterialModel):
    aba_model = True
    lapack = "lite"
    def setup(self, props, **kwargs):
        """Setup the material model.

        Checks properties, assigns storage for state variables, initializes
        all quantities.

        """
        try:
            self.import_model()
        except ImportError:
            raise ModelNotImportedError(self.name)

        xkeys = lambda n: ["SDV{0}".format(i+1) for i in range(n)]
        # depvar must be at least 1 (cannot pass reference to empty list)
        if hasattr(self, "depvar"):
            depvar = self.depvar
        else:
            depvar = kwargs.get("depvar") or 1

        # depvar allowed to be an integer (number of SDVs) or a list (names of
        # SDVs)
        try:
            depvar, sdv_keys = len(depvar), depvar
        except TypeError:
            sdv_keys = xkeys(depvar)

        # call model with zero state to get initial state variables
        statev = np.zeros(depvar)
        time, dtime = 0., 1.
        temp, dtemp = self.initial_temp, 0.
        stress = self.initial_stress
        F0 = Eye(9)
        F = Eye(9)
        stran = np.zeros(6)
        d = np.zeros(6)
        elec_field = np.zeros(3)
        user_field = None
        kappa = 0
        energy = 1.
        rho = 1.
        sigini, xinit, ddsdde = self.update_state(time, dtime, temp, dtemp,
            energy, rho, F0, F, stran, d, elec_field, user_field, stress,
            statev, nxtra=depvar)

        # Do the property completion
        C = isotropic_part(ddsdde)
        lame = C[0,1]
        mu = C[5,5] / 2.
        a = np.array([mu, lame])
        b = [EC_LAME, EC_SHEAR]
        self.completions = complete_properties(a, b)

        # set initial values of state variables
        self.initial_state = xinit
        self.register_xtra_variables(sdv_keys, xinit)

        # additional set up
        self.model_setup(**kwargs)

    def model_setup(self, *args, **kwargs):
        """Setup for specific model.  Only used by uanisohyper_inv"""
        pass

    def update_state(self, time, dtime, temp, dtemp, energy, rho, F0, F,
        stran, d, elec_field, user_field, stress, statev, nxtra=None, **kwargs):
        # abaqus defaults
        N = nxtra or self.nxtra
        w = np.array([1, 1, 1, 2, 2, 2], dtype=np.float64)
        cmname = "{0:8s}".format("umat")
        dfgrd0 = np.reshape(F0, (3, 3), order="F")
        dfgrd1 = np.reshape(F, (3, 3), order="F")
        dstran = d * dtime
        ddsdde = np.zeros((6, 6), order="F")
        ddsddt = np.zeros(6, order="F")
        drplde = np.zeros(6, order="F")
        predef = np.zeros(1, order="F")
        dpred = np.zeros(1, order="F")
        coords = np.zeros(3, order="F")
        drot = np.eye(3)
        ndi = nshr = 3
        spd = scd = rpl = drpldt = pnewdt = 0.
        noel = npt = layer = kspt = kinc = 1
        sse = mmlabpack.ddot(stress, stran) / rho
        celent = 1.
        kstep = 1
        time = np.array([time,time])
        # abaqus ordering
        stress = stress[[0,1,2,3,5,4]]
        # abaqus passes engineering strain
        dstran = dstran[[0,1,2,3,5,4]] * w
        stran = stran[[0,1,2,3,5,4]] * w
        stress, statev, ddsdde = self.update_state_umat(
            stress, statev, ddsdde, sse, spd, scd, rpl, ddsddt, drplde, drpldt,
            stran, dstran, time, dtime, temp, dtemp, predef, dpred, cmname,
            ndi, nshr, N, self.params, coords, drot, pnewdt, celent,
            dfgrd0, dfgrd1, noel, npt, layer, kspt, kstep, kinc)
        if np.any(np.isnan(stress)):
            self.logger.raise_error("umat stress contains nan's")
        stress = stress[[0,1,2,3,5,4]]
        stran = stran[[0,1,2,3,5,4]] / w
        return stress, statev, ddsdde


# ----------------------------------------- Material Model Factory Method --- #
def Material(model, parameters, logger=None, initial_temp=None,
             expansion=None, trs=None, viscoelastic=None, depvar=None,
             param_names=None, source_files=None, source_directory=None,
             fiber_dirs=None, rebuild=0, switch=None):
    """Factory method for subclasses of MaterialModel

    Parameters
    ----------
    model : str
        Material model name
    parameters : dict or ndarray
        Model parameters. For Abaqus umat models and matmodlab user models,
        parameters is a ndarray of model constants (specified in the order
        expected by the model). For other model types, parameters is a
        dictionary of name:value pairs.
    logger : instance or None
        A core.logger.Logger instance
    initial_temp : float or None
        Initial temperature. The initial temperature, if given, must be
        consistent with that of the simulation driver. Defaults to 298K if not
        specified.
    expansion : instance or None
         An instance of an Expansion model.
    trs : instance or None
         An instance of a time-temperature shift (TRS) model
    viscoelastic : instance or None
         An instance of a Viscoelastic model.
    depvar : int or None
        Number of state dependent variables*.
    param_names : list or None
        Parameter names*.  If given, then parameters must be a dict, otherwise,
        an array as described above.
    source_files : list of str or None
        List of model source files*. Each file name given in source_files must
        exist. If the optional source_directory is given, source files are
        looked for in it.
    source_directory : str or None
        Directory containing source files*. source_directory is optional, but
        allows giving source_files as a list of file names only - not fully
        qualified paths.
    rebuild : bool [False]
        Rebuild the material, or not.
    switch : list of tuple of form ("MATX", "MATY") or None
        A list of strings that tell us to use MATY instead of MATX whenever
        MATX is encountered

    Returns
    -------
    material : MaterialModel instance

    Notes
    -----
    *) Applicable only to Abaqus and matmodlab user material models

    """
    if parameters is None:
        raise MatModLabError("{0}: required parameters not given".format(model))
    logger = logger or Logger("console")

    # determine which model
    all_mats = find_materials()
    mat_model = "_".join(model.split()).lower()
    mat_info = all_mats.get(mat_model)
    if mat_info is None:
        raise MatModelNotFoundError(model)

    # determine if
    if "material" in parameters:
        mat_name = parameters.pop("material")
        mat_db = parameters.pop("mat_db", F_MTL_PARAM_DB)
        parameters.update(read_params_from_db(mat_name, mat_model, mat_db))

    errors = 0
    source_files = source_files or []
    if mat_model in ABA_MATS + USER_MAT:
        if not source_files:
            raise MatModLabError("{0}: requires source_files".format(model))
        if source_directory is not None:
            source_files = [os.path.join(source_directory, f)
                            for f in source_files]
        for f in source_files:
            if not os.path.isfile(f):
                errors += 1
                logger.error("{0}: file not found".format(f))
    if errors:
        raise MatModLabError("stopping due to previous errors")

    # default temperature
    if initial_temp is None:
        initial_temp = DEFAULT_TEMP

    # instantiate the material
    material = mat_info.mat_class(parameters, param_names=param_names)

    if material.aux_files:
        source_files.extend(material.aux_files)

    if material.source_files:
        source_files.extend(material.source_files)

    # Check if model is already built (if applicable)
    if source_files:
        so_lib = os.path.join(PKG_D, material.name + ".so")
        if rebuild or opts.rebuild_mat_lib:
            remove(so_lib)
            opts.rebuild_mat_lib = False
        if not os.path.isfile(so_lib):
            logger.write("{0}: rebuilding material library".format(material.name))
            import core.builder as bb
            bb.Builder.build_material(material.name, source_files,
                                      lapack=material.lapack,
                                      verbosity=opts.verbosity)
        if not os.path.isfile(so_lib):
            raise ModelLibNotFoundError(material.name)

        # reload model
        module = load_file(mat_info.file, reload=True)
        mat_class = getattr(module, mat_info.class_name)
        material = mat_class(parameters, param_names=param_names)

    # Check to see if any mimicing requests are made.
    # Switch options that are passed take precedence over the rcfile
    all_switch_opts = ((switch or []) + (opts.switch or []))
    for (old, new) in all_switch_opts:
        if old.lower() != material.name.lower():
            continue
        # a switch was requested, assign the previously instantiated mode to
        # mimic and instantiate the new model
        mimic, model = material, new

        # Make the request known.
        logger.warn("requesting that model {0} mimic model {1}".format(
            model, mimic.name))

        # Get the new material model
        mat_model = "_".join(model.split()).lower()
        mat_info = all_mats.get(mat_model)
        if mat_info is None:
            raise MatModelNotFoundError(model)

        # import and instantiate the new material
        # send the mimic keyword to the constructor this time around
        mat_class = mat_info.mat_class
        material = mat_class(None, None, mimic=mimic)
        break  # Only swap once

    # initialize and set up material
    material.initialize(initial_temp, file=mat_info.file, trs=trs,
                        expansion=expansion, viscoelastic=viscoelastic,
                        logger=logger, fiber_dirs=fiber_dirs, depvar=depvar)

    return material


def find_materials():
    """Find material models

    """
    logger = Logger("console")

    errors = []
    mat_libs = {}
    rx = re.compile(r"(?:^|[\\b_\\.-])[Mm]at")
    a = ["MaterialModel", "AbaqusMaterial"]
    # gather and verify all files
    search_dirs = [d for d in MAT_LIB_DIRS]
    if not SUPPRESS_USER_ENV:
        for user_mat in cfgparse("materials"):
            user_mat = os.path.realpath(user_mat)
            if user_mat not in search_dirs:
                search_dirs.append(user_mat)

    # go through each item in search_dirs and generate a list of material
    # interface files. if item is a directory gather all files that match rx;
    # if it's a file, add it to the list of material files
    for item in search_dirs:
        if os.path.isfile(item):
            d, files = os.path.split(os.path.realpath(item))
            files = [files]
        elif os.path.isdir(item):
            d = item
            files = [f for f in os.listdir(item) if rx.search(f)]
        else:
            logger.warn("{0} no such directory or file, skipping".format(d),
                               report_who=1)
            continue
        files = [f for f in files if f.endswith(".py")]

        if not files:
            logger.warn("{0}: no mat files found".format(d), report_who=1)

        # go through files and determine if it's an interface file. if it is,
        # load it and add it to mat_libs
        for f in files:
            module = f[:-3]
            try:
                libs = xpyclbr.readmodule(module, [d], ancestors=a)
            except AttributeError as e:
                errors.append(e.args[0])
                logger.error(e.args[0])
                continue
            for lib in libs:
                if lib in mat_libs:
                    logger.error("{0}: duplicate material".format(lib))
                    errors.append(lib)
                    continue
                module = load_file(libs[lib].file)
                mat_class = getattr(module, libs[lib].class_name)
                if not mat_class.name:
                    raise MatModLabError("{0}: material name attribute "
                                         "not defined".format(lib))
                libs[lib].mat_class = mat_class
                mat_libs.update({mat_class.name.lower(): libs[lib]})

    if errors:
        raise DuplicateExtModule(", ".join(errors))

    return mat_libs


def isotropic_part(A):
    alpha = np.sum(A[:3,:3])
    beta = np.trace(A)
    a = (2. * alpha - beta) / 15.
    b = (3. * beta - alpha) / 15.
    Aiso = b * np.eye(6)
    Aiso[:3,:3] += a * np.ones((3,3))
    return Aiso
