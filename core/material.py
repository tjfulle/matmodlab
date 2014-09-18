import os
import re
import sys
import numpy as np

from utils.errors import *
from core.runtime import opts
import utils.xpyclbr as xpyclbr
import utils.mmlabpack as mmlabpack
from utils.fortran.product import FIO
from utils.misc import load_file, remove
from core.product import MAT_LIB_DIRS, PKG_D
from utils.data_containers import Parameters
from core.logger import Logger, ConsoleLogger
from materials.product import ABA_MATS, USER_MAT
from utils.variable import Variable, VAR_ARRAY, VAR_SCALAR
from utils.constants import DEFAULT_TEMP, SET_AT_RUNTIME, ENGW
np.set_printoptions(precision=2)

try:
    from lib.visco import visco as visco
except ImportError:
    visco = None
try:
    from lib.expansion import expansion as xpansion
except ImportError:
    xpansion = None

class MaterialModel(object):
    """The base material class

    """
    W = np.ones(6)
    def __init__(self):
        raise Exception("materials must define own __init__")

    def assert_attr_exists(self, attr):
        noattr = -31
        if getattr(self, attr, noattr) == noattr:
            raise Exception("material missing attribute: {0}".format(attr))

    def init(self, logger=None, file=None):

        for attr in ("name", "param_names"):
            self.assert_attr_exists(attr)

        self._vars = []
        self.nvisco = 0
        self.visco_model = None
        self.trs_model = None
        self.expansion_model = None
        self.bulk_modulus = None
        self.shear_modulus = None
        self.J0 = None
        self.nxtra = 0
        self.initial_state = np.zeros(self.nxtra)
        self.xkeys = []
        self.initialized = True
        self.visco_params = None
        self.exp_params = None
        self.itemp = DEFAULT_TEMP
        self.param_defaults = getattr(self, "param_defaults", None)
        self.initial_stress = np.zeros(6)
        self.logger = logger or Logger()
        self._file = file
        self._param_name_map = {}

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
    def file(self):
        return self._file

    @file.setter
    def file(self, value):
        self._file = value

    @property
    def initial_temp(self):
        return self.itemp

    @initial_temp.setter
    def initial_temp(self, value):
        self.itemp = value

    def setup_new_material(self, parameters, depvar, initial_temp, trs=None,
                           expansion=None, viscoelastic=None):
        """Set up the new material

        """
        self.logger.write("setting up {0} material".format(self.name))
        self.initial_temp = initial_temp

        if self.parameter_names == SET_AT_RUNTIME:
            if hasattr(self, "aba_model"):
                self.parameter_names = len(parameters) + 1
                params = np.append(parameters, [0])
            else:
                self.parameter_names = len(parameters)
                params = np.array(parameters)

        else:
            nprops = len(self.parameter_names)
            if self.param_defaults:
                params = np.array(self.param_defaults)
            else:
                params = np.zeros(nprops)

            if not isinstance(parameters, dict):
                raise UserInputError("expected parameters to be a dict")

            # populate the parameters array
            for (key, value) in parameters.items():
                K = key.upper()
                idx = self.parameter_name_map(key)
                if idx is None:
                    raise UserInputError("{0}: unrecognized parameter "
                                         "for model {1}".format(key, self.name))
                params[idx] = value

        self.iparams = Parameters(self.parameter_names, params)
        self.params = Parameters(self.parameter_names, params)
        try:
            self.setup(self.params, depvar)
        except TypeError:
            self.setup()

        if viscoelastic is not None:
            if visco is None:
                self.logger.raise_error("attempting visco analysis but "
                                        "lib/visco.so not imported")
            self.visco_model = viscoelastic

            # setup viscoelastic params
            self.visco_params = np.zeros(24)
            # starting location of G and T Prony terms
            n = self.visco_model.nprony
            I, J = (4, 14)
            self.visco_params[I:I+n] = self.visco_model.data[:, 0]
            self.visco_params[J:J+n] = self.visco_model.data[:, 1]
            # Ginf
            self.visco_params[3] = self.visco_model.Ginf

            # Allocate storage for visco data
            visco_keys = []

            # Shift factors
            visco_keys.extend(["SHIFT_{0}".format(i+1) for i in range(2)])

            # Instantaneous deviatoric PK2
            m = {0: "XX", 1: "YY", 2: "ZZ", 3: "XY", 4: "YZ", 5: "XZ"}
            visco_keys.extend(["TE_{0}".format(m[i]) for i in range(6)])

            # Visco elastic model supports up to 10 Prony series terms,
            # allocate storage for stress corresponding to each
            nprony = 10
            for l in range(nprony):
                for i in range(6):
                    visco_keys.append("H{0}_{1}".format(l+1, m[i]))

            self.nvisco = len(visco_keys)
            visco_idata = np.zeros(self.nvisco)
            self.augment_xtra(visco_keys, visco_idata)

        if trs is not None:
            self.trs_model = trs
            self.visco_params[0] = self.trs_model.data[1] # C1
            self.visco_params[1] = self.trs_model.data[2] # C2
            self.visco_params[2] = self.trs_model.temp_ref # REF TEMP

        if expansion is not None:
            if xpansion is None:
                self.logger.error("attempting thermal expansion but "
                                  "lib/expansion.so not imported")
            self.expansion_model = expansion
            self.exp_params = self.expansion_model.data

        if self.visco_params is not None:
            comm = (self.logger.error, self.logger.write, self.logger.warn)
            visco.propcheck(self.visco_params, *comm)

        self.register_variable("XTRA", VAR_ARRAY, keys=self.xkeys,
                               ivals=self.initial_state)

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
        self.initial_state = np.array(values)

    def augment_xtra(self, keys, values):
        # increase xkeys and initial state -> but not nxtra
        self.xkeys.extend(keys)
        if len(values) != len(keys):
            raise ValueError("len(values) != len(keys)")
        self.initial_state = np.append(self.initial_state, np.array(values))

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
        kappa = 0
        v = range(6)
        c = self.numerical_jacobian(time, dtime, temp, dtemp, kappa, F0, F,
            stran, d, elec_field, user_field, stress, self.initial_state, v)
        return c

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
            Dp *= self.W
            Ep *= self.W
            sigp = stress.copy()
            xtrap = xtra.copy()
            sigp = self.compute_updated_state(time, dtime, temp, dtemp, kappa,
                      F0, Fp, Ep, Dp, elec_field, user_field, sigp, xtrap, disp=3)

            # perturb backward
            Dm = d.copy()
            Dm[v[i]] = d[v[i]] - (deps / dtime) / 2.
            Fm, Em = mmlabpack.update_deformation(dtime, 0., F, Dm)
            Dm *= self.W
            Em *= self.W
            sigm = stress.copy()
            xtram = xtra.copy()
            sigm = self.compute_updated_state(time, dtime, temp, dtemp, kappa,
                      F0, Fm, Em, Dm, elec_field, user_field, sigm, xtram, disp=3)

            # compute component of jacobian
            Jsub[i, :] = (sigp[v] - sigm[v]) / deps / self.W[v]

            continue

        return Jsub

    @property
    def parameter_names(self):
        try: return [n.split(":")[0].upper() for n in self.param_names]
        except TypeError: return self.param_names

    @parameter_names.setter
    def parameter_names(self, n):
        """Set parameter names for models that set parameter names at run time

        """
        self.param_names = ["PROP{0:02d}".format(i+1) for i in range(n)]

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
        comm = (self.logger.error, self.logger.write, self.logger.warn)

        sig = np.array(stress)
        xtra = np.array(statev)

        # Mechanical deformation
        Fm, Em = F, stran

        if self.exp_params is not None:
            # thermal expansion: get mechanical deformation
            Fm = xpansion.mechdef(self.exp_params, temp, dtemp,
                                  F.reshape(3,3), *comm)
            Fm = Fm.reshape(9,)
            Em = mmlabpack.update_strain(kappa, Fm)

        rho = 1.
        energy = 1.
        sig, xtra[:N], ddsdde = self.update_state(time, dtime, temp, dtemp,
            energy, rho, F0, Fm, Em, d, elec_field, user_field, sig,
            xtra[:N], last=last, mode=0)

        if self.visco_params is not None:
            # get visco correction
            X = np.array(xtra[N:], order="F")
            cfac = np.zeros(2)
            sig, cfac = visco.viscorelax(dtime, time, temp, dtemp,
                             self.visco_params, F.reshape(3,3), X, sig, *comm)
            xtra[N:] = X[:]

        if disp == 3:
            return sig

        if ddsdde is None or self.visco_params is not None:
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

        if last and opts.sqa:
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

    def initialize(self):
        """Call the material with initial state

        """
        N = self.nxtra
        xtra = self.initial_state
        if self.visco_params is not None:
            # initialize the visco variables
            x = xtra[N:]
            comm = (self.logger.error, self.logger.write, self.logger.warn)
            visco.viscoini(self.visco_params, x, *comm)
            xtra[N:] = x

        time = 0.
        dtime = 1.
        dtemp = 0.
        temp = self.initial_temp
        stress = self.initial_stress
        F0 = np.eye(3).reshape(9,)
        F = np.eye(3).reshape(9,)
        stran = np.zeros(6)
        d = np.zeros(6)
        elec_field = np.zeros(3)
        user_field = None
        kappa = 0
        sigini, xinit, ddsdde = self.compute_updated_state(time, dtime, temp,
            dtemp, kappa, F0, F, stran, d, elec_field, user_field, stress, xtra)

        self.initial_state = xinit
        self.initial_stress = sigini

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
    W = np.array([1., 1., 1., 2., 2., 2.])
    def setup(self, props, depvar):
        self.import_model()
        if not depvar:
            depvar = 1
        statev = np.zeros(depvar)
        xkeys = ["SDV{0}".format(i+1) for i in range(depvar)]
        self.register_xtra_variables(xkeys, statev)

        ddsdde = self.get_initial_jacobian()
        mu = ddsdde[3, 3]
        lam = ddsdde[0, 0] - 2. * mu

        self.bulk_modulus = lam + 2. / 3. * mu
        self.shear_modulus = mu

    def update_state(self, time, dtime, temp, dtemp, energy, rho, F0, F,
        stran, d, elec_field, user_field, stress, statev, **kwargs):
        # abaqus defaults
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
        sse = spd = scd = rpl = drpldt = celent = pnewdt = 0.
        noel = npt = layer = kspt = kstep = kinc = 1
        time = np.array([time,time])
        # abaqus ordering
        stress = stress[[0,1,2,3,5,4]]
        dstran = dstran[[0,1,2,3,5,4]]
        stress, statev, ddsdde = self.update_state_umat(
            stress, statev, ddsdde, sse, spd, scd, rpl, ddsddt, drplde, drpldt,
            stran, dstran, time, dtime, temp, dtemp, predef, dpred, cmname,
            ndi, nshr, self.nxtra, self.params, coords, drot, pnewdt, celent,
            dfgrd0, dfgrd1, noel, npt, layer, kspt, kstep, kinc)
        if np.any(np.isnan(stress)):
            self.logger.error("umat stress contains nan's")
        stress = stress[[0,1,2,3,5,4]]
        return stress, statev, ddsdde

    def set_constant_jacobian(self):
        time, dtime = 0, 0
        temp, dtemp = self.initial_temp, 0.
        kappa = 0
        F0, F = np.eye(3), np.eye(3)
        stran, d, elec_field = np.zeros(6), np.zeros(6), np.zeros(3)
        user_field = 0
        self.J0 = self.compute_updated_state(time, dtime, temp, dtemp, kappa,
                       F0, F, stran, d, elec_field, user_field,
                       self.initial_stress, self.initial_state, disp=2)
        return

    def get_initial_jacobian(self):
        """Get the initial Jacobian"""
        time, dtime = 0, 0
        temp, dtemp = self.initial_temp, 0.
        kappa = 0
        F0, F = np.eye(3), np.eye(3)
        stran, d, elec_field = np.zeros(6), np.zeros(6), np.zeros(3)
        user_field = 0
        ddsdde = self.compute_updated_state(time, dtime, temp, dtemp, kappa,
                       F0, F, stran, d, elec_field, user_field,
                       self.initial_stress, self.initial_state, disp=2)
        return ddsdde

# ----------------------------------------- Material Model Factory Method --- #
def Material(model, parameters=None, depvar=None, constants=None,
             source_files=None, source_directory=None, initial_temp=None,
             expansion=None, trs=None, viscoelastic=None, logger=None,
             rebuild=0):
    """Material model factory method

    """
    if parameters is None:
        raise InputError("{0}: required parameters not given".format(model))

    logger = logger or Logger()

    # switch model, if requested
    if opts.switch:
        logger.warn("switching {0} for {1}".format(model, opts.switch))
        model = opts.switch

    # determine which model
    m = "_".join(model.split()).lower()
    for (lib, libinfo) in find_materials().items():
        if lib.lower() == m:
            break
    else:
        raise MatModelNotFoundError(model)

    errors = 0
    source_files = source_files or []
    if m in ABA_MATS + USER_MAT:
        if not source_files:
            raise InputError("{0}: requires source_files".format(model))
        if source_directory is not None:
            source_files = [os.path.join(source_directory, f)
                            for f in source_files]
        for f in source_files:
            if not os.path.isfile(f):
                errors += 1
                ConsoleLogger.error("{0}: file not found".format(f))
    if errors:
        raise UserInputError("stopping due to previous errors")

    # default temperature
    if initial_temp is None:
        initial_temp = DEFAULT_TEMP

    # import the material
    module = load_file(libinfo.file)
    mat_class = getattr(module, libinfo.class_name)
    material = mat_class()

    if m in ABA_MATS or m in USER_MAT:
        material.source_files = material.aux_files

    # Check if model is already built (if applicable)
    if hasattr(material, "source_files"):
        so_lib = os.path.join(PKG_D, lib + ".so")
        if rebuild: remove(so_lib)
        if not os.path.isfile(so_lib):
            logger.write("{0}: rebuilding material library".format(material.name))
            material.source_files.extend(source_files)
            lapack = getattr(material, "lapack", None)
            import core.builder as bb
            bb.Builder.build_material(lib, material.source_files,
                                      lapack=lapack,
                                      verbosity=opts.verbosity)
        if not os.path.isfile(so_lib):
            raise ModelLibNotFoundError(lib)
        # reload model
        if libinfo.module in sys.modules:
            del sys.modules[libinfo.module]
        module = load_file(libinfo.file, reload=True)
        mat_class = getattr(module, libinfo.class_name)
        material = mat_class()

    # initialize and set up material
    material.init(logger=logger, file=libinfo.file)

    if material.parameter_names == SET_AT_RUNTIME:
        # some models, like abaqus models, do not have the parameter names
        # set. The user must provide the entire UI array - as an array.

        # Check for number of constants
        if constants is None:
            raise UserInputError("{0}: expected keyword constants".format(lib))
        constants = int(constants)

        # Check parameters
        if not isinstance(parameters, (list, tuple, np.ndarray)):
            raise UserInputError("{0}: parameters must be an array".format(lib))
        if len(parameters) != constants:
            raise UserInputError("len(parameters) != constants")
        parameters = np.array([float(x) for x in parameters])

    # Do the actual setup
    material.setup_new_material(parameters, depvar, initial_temp, trs=trs,
                                expansion=expansion, viscoelastic=viscoelastic)
    material.initialize()

    return material


def find_materials():
    """Find material models

    Notes
    -----
    Looks for function product.material_libraries in directories specified in
    environ["MMLMTLS"].

    """
    errors = []
    mat_libs = {}
    rx = re.compile(r"(?:^|[\\b_\\.-])[Mm]at")
    a = ["MaterialModel", "AbaqusMaterial"]
    n = ["name"]
    # gather and verify all files
    for item in MAT_LIB_DIRS:
        if os.path.isfile(item):
            d, files = os.path.split(os.path.realpath(item))
            files = [files]
        elif os.path.isdir(item):
            d = item
            files = [f for f in os.listdir(item) if rx.search(f)]
        else:
            ConsoleLogger.warn("{0} no such directory or file, skipping".format(d),
                               report_who=1, beg="*** WARNING: ")
            continue
        files = [f for f in files if f.endswith(".py")]

        if not files:
            ConsoleLogger.write("{0}: no mat files found".format(d),
                                report_who=1, beg="*** WARNING: ")

        for f in files:
            module = f[:-3]
            try:
                libs = xpyclbr.readmodule(module, [d], ancestors=a, reqattrs=n)
            except AttributeError as e:
                errors.append(e.args[0])
                ConsoleLogger.error(e.args[0])
                continue
            for lib in libs:
                if lib in mat_libs:
                    ConsoleLogger.error("{0}: duplicate material".format(lib))
                    errors.append(lib)
                    continue
                mat_libs.update({libs[lib].name: libs[lib]})

    if errors:
        raise DuplicateExtModule(", ".join(errors))

    return mat_libs
