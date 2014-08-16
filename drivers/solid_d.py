import os
import re
import sys
import numpy as np
from numpy.linalg import solve, lstsq

from mml import RESTART
from core.runtime import opts
import utils.xmltools as xmltools
from drivers.driver import Driver
from core.solvers import sig2d
from core.mmlio import (fatal_inp_error, input_errors,
                        log_message, log_warning, Error1)
try:
    from lib.mmlabpack import mmlabpack
except ImportError:
    import utils.mmlabpack as mmlabpack
try:
    from mml_user_sub import eval_at_step
except ImportError:
    eval_at_step = None

NSYMM = 6
NTENS = 9
I9 = np.eye(3).reshape(9)
Z6 = np.zeros(6)
Z3 = np.zeros(3)
I6 = np.array([1, 1, 1, 0, 0, 0], dtype=np.float64)
W = np.array([1, 1, 1, 2, 2, 2], dtype=np.float64)
DI3 = [[0, 1, 2], [0, 1, 2]]
DEFAULT_TMPR = 298.
CONTROL_FLAGS = {"D": 1,  # strain rate
                 "E": 2,  # strain
                 "R": 3,  # stress rate
                 "S": 4,  # stress
                 "F": 5,  # deformation gradient
                 "P": 6,  # electric field
                 "T": 7,  # temperature
                 "U": 8,  # displacement
                 "X": 9}  # user defined field
dev = mmlabpack.dev
mag = mmlabpack.mag

np.set_printoptions(precision=4)

class SolidDriver(Driver):
    name = "solid"
    def __init__(self, path, opts, material):
        super(SolidDriver, self).__init__()
        self.opts = opts
        self.kappa = opts[1]
        self.proportional = bool(opts[2])
        self.path = path

        # Create material
        itmpr = path[0][18]
        ifield = path[0][19:]
        mtl, params, options, istate = material
        options["initial_temperature"] = itmpr
        self._mtl_istate = istate
        self.material = mtl.instantiate_material(params, options)

        # register variables
        self.register_glob_variable("TIME_STEP")
        self.register_glob_variable("STEP_NUM")
        self.register_glob_variable("LEG_NUM")
        self.register_variable("STRESS", vtype="SYMTENS")
        self.register_variable("STRAIN", vtype="SYMTENS")
        self.register_variable("DEFGRAD", vtype="TENS")
        self.register_variable("SYMM_L", vtype="SYMTENS")
        self.register_variable("EFIELD", vtype="VECTOR")
        self.register_variable("VSTRAIN", vtype="SCALAR")
        self.register_variable("EQSTRAIN", vtype="SCALAR")
        self.register_variable("PRESSURE", vtype="SCALAR")
        self.register_variable("SMISES", vtype="SCALAR")
        self.register_variable("DSTRESS", vtype="SYMTENS")
        self.register_variable("TMPR", vtype="SCALAR")

        # register material variables
        self.xtra_start = self.ndata
        for (var, vtype) in self.material.material_variables:
            self.register_variable(var, vtype=vtype)

        nxtra = self.ndata - self.xtra_start
        self.xtra_end = self.xtra_start + nxtra
        setattr(self, "xtra_slice", slice(self.xtra_start, self.xtra_end))

        # allocate storage
        self.allocd()

        if opts[0] == RESTART:
            # restart info given
            start_leg, time, glob_data, elem_data = opts[3]
            self._glob_data[:] = glob_data
            self._data[:] = elem_data
            self.start_leg = start_leg
            self.step_num = glob_data[1]
            self.time = time

        else:
            # initialize nonzero data
            self._data[self.defgrad_slice] = I9

            if len(self._mtl_istate):
                # --- initial state is given
                sig = self._mtl_istate[:NSYMM]
                xtra = self._mtl_istate[NSYMM:]
                if len(xtra) != self.material.nxtra:
                    raise Error1("incorrect len(InitialState)")
            else:
                # initialize material
                sig, xtra = self.material.initialize(itmpr, ifield)
                mml_user_sub_eval(0., np.zeros(NSYMM), sig, xtra)

            pres = -np.sum(sig[:3]) / 3.
            self.setvars(stress=sig, pressure=pres, xtra=xtra, tmpr=itmpr)

            self.start_leg = 0
            self.step_num = 0
            self.time = 0.

        return

    def process_paths_and_surfaces(self, iomgr, *args):
        """Process the deformation path

        Parameters
        ----------

        Returns
        -------

        """
        legs = self.path[self.start_leg:]
        kappa = self.kappa

        # initial leg
        glob_step_num = self.step_num
        xtra = self.elem_var_vals("XTRA")
        sig = self.elem_var_vals("STRESS")
        tmpr = np.zeros(3)
        tmpr[1] = self.elem_var_vals("TMPR")
        tleg = np.zeros(2)
        d = np.zeros(NSYMM)
        dt = 0.
        eps = np.zeros(NSYMM)
        f0 = np.reshape(np.eye(3), (NTENS, 1))
        f = np.reshape(np.eye(3), (NTENS, 1))
        depsdt = np.zeros(NSYMM)
        sigdum = np.zeros((2, NSYMM))

        # compute the initial jacobian
        J0 = self.material.constant_jacobian()

        # v array is an array of integers that contains the rows and columns of
        # the slice needed in the jacobian subroutine.
        nv = 0
        vdum = np.zeros(NSYMM, dtype=np.int)

        # Process each leg
        nlegs = len(legs)
        lsl = len(str(nlegs))
        for (ileg, leg) in enumerate(legs):
            leg_num = self.start_leg + ileg
            tleg[0] = tleg[1]
            tmpr[0] = tmpr[1]
            sigdum[0] = sig[:]
            if nv:
                sigdum[0, v] = sigspec[1]

            tleg[1] = leg[0]
            nsteps = leg[1]
            control = leg[2:8]
            c = leg[8:14]
            ndumps = leg[14]
            ef = leg[15:18]
            tmpr[1] = leg[18]
            ufield = leg[19:]

            delt = tleg[1] - tleg[0]
            if delt == 0.:
                continue

            # ndumps_per_leg is the number of times to write to the output
            # file in this leg
            dump_interval = max(1, int(nsteps / ndumps))
            lsn = len(str(nsteps))
            consfmt = ("leg {{0:{0}d}}, step {{1:{1}d}}, time {{2:.4E}}, "
                       "dt {{3:.4E}}".format(lsl, lsn))

            nv = 0
            for i, cij in enumerate(c):
                if control[i] == 1:                            # -- strain rate
                    depsdt[i] = cij

                elif control[i] == 2:                          # -- strain
                    depsdt[i] = (cij - eps[i]) / delt

                elif control[i] == 3:                          # -- stress rate
                    sigdum[1, i] = sigdum[0, i] + cij * delt
                    vdum[nv] = i
                    nv += 1

                elif control[i] == 4:                          # -- stress
                    sigdum[1, i] = cij
                    vdum[nv] = i
                    nv += 1

                continue

            sigspec = np.empty((3, nv))
            v = vdum[:nv]
            sigspec[:2] = sigdum[:2, v]
            Jsub = J0[[[x] for x in v], v]

            t = tleg[0]
            dt = delt / nsteps
            dtmpr = (tmpr[1] - tmpr[0]) / nsteps

            if not nv:
                # strain or strain rate prescribed and d is constant over
                # entire leg
                d = mmlabpack.deps2d(dt, kappa, eps, depsdt)

                if opts.sqa and kappa == 0.:
                    if not np.allclose(d, depsdt):
                        log_message("sqa: d != depsdt (k=0, leg"
                                    "={0})".format(leg_num))

            else:
                # Initial guess for d[v]
                try:
                    depsdt[v] = solve(Jsub, (sigspec[1] - sigspec[0]) / delt)
                except:
                    depsdt[v] -= lstsq(Jsub, (sigspec[1] - sigspec[0]) / delt)[0]

            warned = False
            # process this leg
            for n in range(int(nsteps)):

                # increment time
                t += dt
                self.time = t

                # interpolate values to the target values for this step
                a1 = float(nsteps - (n + 1)) / nsteps
                a2 = float(n + 1) / nsteps
                sigspec[2] = a1 * sigspec[0] + a2 * sigspec[1]
                tmpr[2] = a1 * tmpr[0] + a2 * tmpr[1]
                temp = tmpr[2] - dtmpr

                # --- find current value of d: sym(velocity gradient)
                if nv:
                    # One or more stresses prescribed
                    # get just the prescribed stress components
                    d = sig2d(self.material, t, dt, temp, dtmpr,
                              f0, f, eps, depsdt, sig, xtra, ef, ufield,
                              v, sigspec[2], self.proportional)

                # compute the current deformation gradient and strain from
                # previous values and the deformation rate
                f, eps = mmlabpack.update_deformation(dt, kappa, f0, d)

                # update material state
                sigsave = np.array(sig)
                xtrasave = np.array(xtra)
                sig, xtra = self.material.compute_updated_state(t, dt, temp, dtmpr,
                    f0, f, eps, d, sig, xtra, ef, ufield, last=True)

                # -------------------------- quantities derived from final state
                eqeps = np.sqrt(2. / 3. * (np.sum(eps[:3] ** 2)
                                           + 2. * np.sum(eps[3:] ** 2)))
                epsv = np.sum(eps[:3])

                pres = -np.sum(sig[:3]) / 3.
                dstress = (sig - sigsave) / dt
                smises = np.sqrt(3./2.) * mag(dev(sig))
                f0 = f

                # advance all data after updating state
                glob_step_num += 1
                self.setglobvars(leg_num=leg_num,
                                 step_num=glob_step_num, time_step=dt)

                self.setvars(stress=sig, strain=eps, defgrad=f,
                             symm_l=d, efield=ef, eqstrain=eqeps,
                             vstrain=epsv, pressure=pres,
                             dstress=dstress, xtra=xtra, tmpr=tmpr[2],
                             smises=smises)
                mml_user_sub_eval(t, d, sig, xtra)

                # --- write state to file
                endstep = abs(t - tleg[1]) / tleg[1] < 1.E-12
                if (nsteps - n) % dump_interval == 0 or endstep:
                    iomgr(t)

                if n == 0 or round((nsteps - 1) / 2.) == n or endstep:
                    log_message(consfmt.format(leg_num, n + 1, t, dt))

                if n > 1 and nv and not warned:
                    absmax = lambda a: np.max(np.abs(a))
                    sigerr = np.sqrt(np.sum((sig[v] - sigspec[2]) ** 2))
                    warned = True
                    _tol = np.amax(np.abs(sig[v])) / self.material.bulk_modulus
                    if sigerr > _tol:
                        log_warning("leg: {0}, prescribed stress error: "
                                    "{1: .3f}. consider increasing number of "
                                    "steps".format(ileg, sigerr))

                continue  # continue to next step

            continue # continue to next leg

        self._paths_and_surfaces_processed = True
        return 0

    # --------------------------------------------------------- Parsing methods
    @staticmethod
    def format_path_and_opts(pathdict, functions, tterm):
        """Parse the Path elements of the input file and register the formatted
        paths to the class

        """
        path = pPrdef(pathdict, functions, tterm)
        if input_errors():
            return
        return path, (0, pathdict["kappa"], pathdict["proportional"])

def mybool(a):
    if str(a).lower().strip() in ("false", "no", "0", "none"):
        return 0
    else:
        return 1


def format_termination_time(leg_num, termination_time, final_time):
    try:
        termination_time = float(termination_time)
    except ValueError:
        fatal_inp_error("Path: expected float for termination time of "
                        "leg {0} got {1}".format(leg_num, termination_time))
        return

    if termination_time < 0.:
        fatal_inp_error("Path: expected positive termination time leg {0} "
                        "got {1}".format(leg_num, termination_time))
        return

    if termination_time < final_time:
        fatal_inp_error("Path: expected time to increase monotonically in "
                        "leg {0}".format(leg_num))
        return

    return termination_time


def format_num_steps(leg_num, num_steps):
    try:
        num_steps = int(num_steps)
    except ValueError:
        fatal_inp_error("Path: expected integer number of steps in "
                        "leg {0} got {1}".format(leg_num, num_steps))
        return
    if num_steps < 0:
        fatal_inp_error("Path: expected positive integer number of "
                        "steps in leg {0} got {1}".format(
                            leg_num, num_steps))
        return

    return num_steps

def pPrdef(pathdict, functions, tterm):
    """Parse the Path block and set defaults

    """
    lines = [line.split() for line in pathdict.pop("Content") if line.split()]

    # parse the Path depending on type
    pformat = pathdict["format"]
    if pformat == "default":
        path = parse_path_default(lines)

    elif pformat == "table":
        path = parse_path_table(lines, pathdict["tfmt"],
                                pathdict["cols"],
                                pathdict["cfmt"],
                                pathdict["lineskip"])

    elif pformat == "fcnspec":
        path = parse_path_cijfcn(lines, functions,
                                 pathdict["nfac"],
                                 pathdict["cfmt"])
        pathdict["nfac"] = 1

    else:
        fatal_inp_error("Path: {0}: invalid format".format(pformat))
        return

    if input_errors():
        return

    # store relevant info to the class
    path = _format_path(path, pathdict, tterm)

    return path

def parse_path_default(lines):
    """Parse the individual path

    """
    path = []
    final_time = 0.
    leg_num = 1
    for line in lines:
        if not line:
            continue
        termination_time, num_steps, control_hold = line[:3]
        Cij_hold = line[3:]

        # check entries
        # --- termination time
        termination_time = format_termination_time(
            leg_num, termination_time, final_time)
        if termination_time is None:
            termination_time = 1e99
        final_time = termination_time

        # --- number of steps
        num_steps = format_num_steps(leg_num, num_steps)
        if num_steps is None:
            num_steps = 10000

        # --- control
        control = format_path_control(control_hold, leg_num=leg_num)

        # --- Cij
        Cij = []
        for (i, comp) in enumerate(Cij_hold):
            try:
                comp = float(comp)
            except ValueError:
                fatal_inp_error("Path: Component {0} of leg {1} must be a "
                                "float, got {2}".format(i+1, leg_num, comp))
            Cij.append(comp)

        Cij = np.array(Cij)

        # --- Check lengths of Cij and control are consistent
        if len(Cij) != len(control):
            fatal_inp_error("Path: len(Cij) != len(control) in leg {0}"
                            .format(leg_num))
            continue

        path.append([termination_time, num_steps, control, Cij])
        leg_num += 1
        continue

    return path


def parse_path_table(lines, tfmt, cols, cfmt, lineskip):
    """Parse the path table

    """
    path = []
    final_time = 0.
    termination_time = 0.
    leg_num = 1

    # check the control
    control = format_path_control(cfmt)

    tbl = []
    for idx, line in enumerate(lines):
        if idx < lineskip or not line:
            continue
        if line[0].strip().startswith("#"):
            continue
        try:
            line = [float(x) for x in line]
        except ValueError:
            fatal_inp_error("Expected floats in leg {0}, got {1}".format(
                leg_num, line))
            continue
        tbl.append(line)
    tbl = np.array(tbl)

    # if cols was not specified, must want all
    if not cols:
        columns = range(tbl.shape[1])
    else:
        columns = format_tbl_cols(cols)

    for line in tbl:
        try:
            line = line[columns]
        except IndexError:
            fatal_inp_error("Requested column not found in leg "
                            "{0}".format(leg_num))
            continue

        if tfmt == "dt":
            termination_time += line[0]
        else:
            termination_time = line[0]

        Cij = line[1:]

        # check entries
        # --- termination time
        termination_time = format_termination_time(
            leg_num, termination_time, final_time)
        if termination_time is None:
            continue
        final_time = termination_time

        # --- number of steps
        num_steps = 1

        # --- Check lengths of Cij and control are consistent
        if len(Cij) != len(control):
            fatal_inp_error("Path: len(Cij) != len(control) in leg {0}"
                            .format(leg_num))

        path.append([termination_time, num_steps, control, Cij])
        leg_num += 1
        continue

    return path


def parse_path_cijfcn(lines, functions, num_steps, cfmt):
    """Parse the path given by functions

    """
    start_time = 0.
    leg_num = 1
    if not lines:
        fatal_inp_error("Empty path encountered")
        return
    elif len(lines) > 1:
        fatal_inp_error("Only one line of table functions allowed, "
                        "got {0}".format(len(lines)))
        return

    termination_time = lines[0][0]
    cijfcns = lines[0][1:]

    # check entries
    # --- termination time
    termination_time = format_termination_time(1, termination_time, -1)
    if termination_time is None:
        # place holder, just to check rest of input
        termination_time = 1.e99
    final_time = termination_time

    # --- control
    control = format_path_control(cfmt, leg_num=leg_num)

    # --- get the actual functions
    Cij = []
    for icij, cijfcn in enumerate(cijfcns):
        cijfcn = cijfcn.split(":")
        try:
            fid, scale = cijfcn
        except ValueError:
            fid, scale = cijfcn[0], 1
        try:
            fid = int(float(fid))
        except ValueError:
            fatal_inp_error("expected integer function ID, got {0}".format(fid))
            continue
        try:
            scale = float(scale)
        except ValueError:
            fatal_inp_error("expected real function scale for function {0}"
                            ", got {1}".format(fid, scale))
            continue

        fcn = functions.get(fid)
        if fcn is None:
            fatal_inp_error("{0}: function not defined".format(fid))
            continue
        Cij.append((scale, fcn))

    # --- Check lengths of Cij and control are consistent
    if len(Cij) != len(control):
        fatal_inp_error("Path: len(Cij) != len(control) in leg {0}"
                        .format(leg_num))

    path = []
    vals = np.zeros(len(control))
    if 7 in control:
        # check for nonzero initial values of temperature
        idx = np.where(control == 7)[0][0]
        s, f = Cij[idx]
        vals[idx] = s * f(start_time)
    path.append([start_time, 1, control, vals])
    for time in np.linspace(start_time, final_time, num_steps-1):
        if time == start_time:
            continue
        leg = [time, 1, control]
        leg.append(np.array([s * f(time) for (s, f) in Cij]))
        path.append(leg)
    return path

def format_path_control(cfmt, leg_num=None):
    leg = "" if leg_num is None else "(leg {0})".format(leg_num)

    _cfmt = [CONTROL_FLAGS.get(s.upper(), s) for s in cfmt]

    control = []
    for (i, flag) in enumerate(_cfmt):
        try:
            flag = int(flag)
        except ValueError:
            fatal_inp_error("Path: unexpected control flag {0}".format(flag))
            continue

        if flag not in CONTROL_FLAGS.values():
            valid = ", ".join(xmltools.stringify(x)
                              for x in CONTROL_FLAGS.values())
            fatal_inp_error("Path: expected control flag to be one of {0}, "
                            "got {1} {2}".format(valid, flag, leg))
            continue

        control.append(flag)

    if control.count(7) > 1:
            fatal_inp_error("Path: multiple temperature fields in "
                            "leg {0}".format(leg))

    if 5 in control:
        if any(flag != 5 and flag not in (6, 9) for flag in control):
            fatal_inp_error("Path: mixed mode deformation not allowed with "
                            "deformation gradient control {0}".format(leg))

        # must specify all components
        elif len(control) < 9:
            fatal_inp_error("all 9 components of deformation gradient must "
                            "be specified {0}".format(leg))

    if 8 in control:
        # like deformation gradient control, if displacement is specified
        # for one, it must be for all
        if any(flag != 8 and flag not in (6, 9) for flag in control):
            fatal_inp_error("Path: mixed mode deformation not allowed with "
                            "displacement control {0}".format(leg))

        # must specify all components
        elif len(control) < 3:
            fatal_inp_error("all 3 components of displacement must "
                            "be specified {0}".format(leg))

    return np.array(control, dtype=np.int)


def format_tbl_cols(cols):
    columns = []
    for item in [x.split(":")
                 for x in xmltools.str2list(
                         re.sub(r"\s*:\s*", ":", cols), dtype=str)]:
        try:
            item = [int(x) for x in item]
        except ValueError:
            fatal_inp_error("Path: expected integer cols, got "
                            "{0}".format(cols))
            continue
        item[0] -= 1

        if len(item) == 1:
            columns.append(item[0])

        elif len(item) not in (2, 3):
            fatal_inp_error("Path: expected cfmt range to be specified as "
                            "start:end:[step], got {0}".format(
                                ":".join(str(x) for x in item)))
            continue

        if len(item) == 2:
            columns.extend(range(item[0], item[1]))

        elif len(item) == 3:
            columns.extend(range(item[0], item[1], item[2]))

    return columns


def _format_path(path, pathdict, tterm):
    """Format the path by applying multipliers

    """
    # stress control if any of the control types are 3 or 4
    stress_control = any(c in (3, 4) for leg in path for c in leg[2])
    kappa = pathdict["kappa"]
    if stress_control and kappa != 0.:
        fatal_inp_error("kappa must be 0 with stress control option")

    # From these formulas, note that AMPL may be used to increase or
    # decrease the peak strain without changing the strain rate. ratfac is
    # the multiplier on strain rate and stress rate.
    amplitude = pathdict["amplitude"]
    ratfac = pathdict["ratfac"]
    nfac = pathdict["nfac"]
    ndumps = pathdict["ndumps"]
    if ndumps == "all":
        ndumps = 100000000
    ndumps= int(ndumps)

    # factors to be applied to deformation types
    efac = amplitude * pathdict["estar"]
    tfac = abs(amplitude) * pathdict["tstar"] / ratfac
    sfac = amplitude * pathdict["sstar"]
    ffac = amplitude * pathdict["fstar"]
    effac = amplitude * pathdict["efstar"]
    dfac = amplitude * pathdict["dstar"]

    # for now unit tensor for rotation
    Rij = np.reshape(np.eye(3), (NTENS,))

    # format each leg
    if not tterm:
        tterm = 1.e80

    for ileg, (termination_time, num_steps, control, Cij) in enumerate(path):

        leg_num = ileg + 1

        num_steps = int(nfac * num_steps)
        termination_time = tfac * termination_time

        if len(control) != len(Cij):
            fatal_inp_error("len(cij) != len(control) in leg "
                            "{0}".format(leg_num))
            continue

        # pull out electric field from other deformation specifications
        tmpr = DEFAULT_TMPR
        efcomp = np.zeros(3)
        user_field = []
        trtbl = np.array([True] * len(control))
        j = 0
        for i, c in enumerate(control):
            if c in (6, 7, 9):
                trtbl[i] = False
                if c == 6:
                    efcomp[j] = effac * Cij[i]
                    j += 1
                elif c == 7:
                    tmpr = Cij[i]
                else:
                    user_field.append(Cij[i])

        Cij = Cij[trtbl]
        control = control[trtbl]

        if 5 in control:
            # check for valid deformation
            defgrad = np.reshape(ffac * Cij, (3, 3))
            jac = np.linalg.det(defgrad)
            if jac <= 0:
                fatal_inp_error("Inadmissible deformation gradient in "
                                "leg {0} gave a Jacobian of "
                                "{1:f}".format(leg_num, jac))

            # convert defgrad to strain E with associated rotation given by
            # axis of rotation x and angle of rotation theta
            Rij, Vij = np.linalg.qr(defgrad)
            if np.max(np.abs(Rij - np.eye(3))) > np.finfo(np.float).eps:
                fatal_inp_error("Rotation encountered in leg {0}. "
                                "Rotations are not supported".format(leg_num))
            Uij = np.dot(Rij.T, np.dot(Vij, Rij))
            Cij = mmlabpack.u2e(Uij, kappa)
            Rij = np.reshape(Rij, (NTENS,))

            # deformation gradient now converted to strains
            control = np.array([2] * NSYMM, dtype=np.int)

        elif 8 in control:
            # displacement control check
            # convert displacments to strains
            Uij = np.zeros((3, 3))
            Uij[DI3] = dfac * Cij[:3] + 1.
            Cij = mmlabpack.u2e(Uij, kappa, 1)

            # displacements now converted to strains
            control = np.array([2] * NSYMM, dtype=np.int)

        elif 2 in control and len(control) == 1:
            # only one strain value given -> volumetric strain
            evol = Cij[0]
            if kappa * evol + 1. < 0.:
                fatal_inp_error("1 + kappa * ev must be positive in leg "
                                "{0}".format(leg_num))

            if kappa == 0.:
                eij = evol / 3.

            else:
                eij = ((kappa * evol + 1.) ** (1. / 3.) - 1.)
                eij = eij / kappa

            control = np.array([2] * NSYMM, dtype=np.int)
            Cij = np.array([eij, eij, eij, 0., 0., 0.])

        elif 4 in control and len(control) == 1:
            # only one stress value given -> pressure
            Sij = -Cij[0]
            control = np.array([4, 4, 4, 2, 2, 2], dtype=np.int)
            Cij = np.array([Sij, Sij, Sij, 0., 0., 0.])

        control = np.append(control, [2] * (NSYMM - len(control)))
        Cij = np.append(Cij, [0.] * (NSYMM - len(Cij)))

        # adjust components based on user input
        for idx, ctype in enumerate(control):
            if ctype in (1, 3):
                # adjust rates
                Cij[idx] *= ratfac

            elif ctype == 2:
                # adjust strain
                Cij[idx] *= efac

                if kappa * Cij[idx] + 1. < 0.:
                    fatal_inp_error("1 + kappa*E[{0}] must be positive in "
                                    "leg {1}".format(idx, leg_num))

            elif ctype == 4:
                # adjust stress
                Cij[idx] *= sfac

            continue

        # initial stress check
        if termination_time == 0.:
            if 3 in control:
                fatal_inp_error("initial stress rate ambiguous")

            elif 4 in control and any(x != 0. for x in Cij):
                fatal_inp_error("nonzero initial stress not yet supported")

        # Replace leg with modfied values
        leg = [termination_time, num_steps]
        leg.extend(control)
        leg.extend(Cij)
        leg.append(ndumps)
        leg.extend(efcomp)
        leg.append(tmpr)
        leg.extend(user_field)
        path[ileg] = leg

        if termination_time > tterm:
            del path[ileg+1:]
            break

        continue

    return np.array(path)

def mml_user_sub_eval(t, d, sig, xtra):
    """Evaluate a user subroutine

    """
    if eval_at_step:
        eval_at_step(opts.runid, t, d, sig, xtra)
