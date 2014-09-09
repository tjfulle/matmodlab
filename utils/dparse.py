import numpy as np

from utils.errors import GenericError, UserInputError
from utils.constants import DEFAULT_TEMP, NTENS, NSYMM
import utils.mmlabpack as mmlabpack
from core.functions import Function, DEFAULT_FUNCTIONS

CONTROL_FLAGS = {"D": 1,  # strain rate
                 "E": 2,  # strain
                 "R": 3,  # stress rate
                 "S": 4,  # stress
                 "F": 5,  # deformation gradient
                 "P": 6,  # electric field
                 "T": 7,  # temperature
                 "U": 8,  # displacement
                 "X": 9}  # user defined field

def parse_default_path(lines):
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
                raise ValueError("Path: Component {0} of leg {1} must be a "
                                 "float, got {2}".format(i+1, leg_num, comp))
            Cij.append(comp)

        Cij = np.array(Cij)

        # --- Check lengths of Cij and control are consistent
        if len(Cij) != len(control):
            raise ValueError("Path: len(Cij) != len(control) in leg {0}"
                             .format(leg_num))
            continue

        path.append([termination_time, num_steps, control, Cij])
        leg_num += 1
        continue

    return path

def format_termination_time(leg_num, termination_time, final_time):
    try:
        termination_time = float(termination_time)
    except ValueError:
        raise UserInputError("Path: expected float for termination time of "
                             "leg {0} got {1}".format(leg_num, termination_time))

    if termination_time < 0.:
        raise UserInputError("Path: expected positive termination time leg {0} "
                             "got {1}".format(leg_num, termination_time))

    if termination_time < final_time:
        raise UserInputError("Path: expected time to increase monotonically in "
                             "leg {0}".format(leg_num))

    return termination_time

def format_num_steps(leg_num, num_steps):
    try:
        num_steps = int(num_steps)
    except ValueError:
        raise ValueError("Path: expected integer number of steps in "
                         "leg {0} got {1}".format(leg_num, num_steps))
        return
    if num_steps < 0:
        raise ValueError("Path: expected positive integer number of "
                         "steps in leg {0} got {1}".format(leg_num, num_steps))
        return

    return num_steps

def format_path_control(cfmt, leg_num=None):
    leg = "" if leg_num is None else "(leg {0})".format(leg_num)

    _cfmt = [CONTROL_FLAGS.get(s.upper(), s) for s in cfmt]

    control = []
    for (i, flag) in enumerate(_cfmt):
        try:
            flag = int(flag)
        except ValueError:
            raise ValueError("Path: unexpected control flag {0}".format(flag))
            continue

        if flag not in CONTROL_FLAGS.values():
            valid = ", ".join(xmltools.stringify(x)
                              for x in CONTROL_FLAGS.values())
            raise ValueError("Path: expected control flag to be one of {0}, "
                             "got {1} {2}".format(valid, flag, leg))
            continue

        control.append(flag)

    if control.count(7) > 1:
            raise ValueError("Path: multiple temperature fields in "
                             "leg {0}".format(leg))

    if 5 in control:
        if any(flag != 5 and flag not in (6, 9) for flag in control):
            raise ValueError("Path: mixed mode deformation not allowed with "
                             "deformation gradient control {0}".format(leg))

        # must specify all components
        elif len(control) < 9:
            raise ValueError("all 9 components of deformation gradient must "
                             "be specified {0}".format(leg))

    if 8 in control:
        # like deformation gradient control, if displacement is specified
        # for one, it must be for all
        if any(flag != 8 and flag not in (6, 9) for flag in control):
            raise ValueError("Path: mixed mode deformation not allowed with "
                             "displacement control {0}".format(leg))

        # must specify all components
        elif len(control) < 3:
            raise ValueError("all 3 components of displacement must "
                             "be specified {0}".format(leg))

    return np.array(control, dtype=np.int)

def format_continuum_path(path, kappa, amplitude, ratfac, nfac, ndumps,
                          estar, tstar, sstar, fstar, efstar, dstar, tterm):
    """Format the path by applying multipliers

    """
    # stress control if any of the control types are 3 or 4
    stress_control = any(c in (3, 4) for leg in path for c in leg[2])
    if stress_control and kappa != 0.:
        raise ValueError("kappa must be 0 with stress control option")

    # From these formulas, note that AMPL may be used to increase or
    # decrease the peak strain without changing the strain rate. ratfac is
    # the multiplier on strain rate and stress rate.
    if ndumps == "all":
        ndumps = 100000000
    ndumps= int(ndumps)

    # factors to be applied to deformation types
    efac = amplitude * estar
    tfac = abs(amplitude) * tstar / ratfac
    sfac = amplitude * sstar
    ffac = amplitude * fstar
    effac = amplitude * efstar
    dfac = amplitude * dstar

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
            raise ValueError("len(cij) != len(control) in leg "
                             "{0}".format(leg_num))
            continue

        # pull out electric field from other deformation specifications
        temp = DEFAULT_TEMP
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
                    temp = Cij[i]
                else:
                    user_field.append(Cij[i])

        Cij = Cij[trtbl]
        control = control[trtbl]

        if 5 in control:
            # check for valid deformation
            defgrad = np.reshape(ffac * Cij, (3, 3))
            jac = np.linalg.det(defgrad)
            if jac <= 0:
                raise ValueError("Inadmissible deformation gradient in "
                                 "leg {0} gave a Jacobian of "
                                 "{1:f}".format(leg_num, jac))

            # convert defgrad to strain E with associated rotation given by
            # axis of rotation x and angle of rotation theta
            Rij, Vij = np.linalg.qr(defgrad)
            if np.max(np.abs(Rij - np.eye(3))) > np.finfo(np.float).eps:
                raise ValueError("Rotation encountered in leg {0}. "
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
                raise ValueError("1 + kappa * ev must be positive in leg "
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
                    raise ValueError("1 + kappa*E[{0}] must be positive in "
                                     "leg {1}".format(idx, leg_num))

            elif ctype == 4:
                # adjust stress
                Cij[idx] *= sfac

            continue

        # initial stress check
        if termination_time == 0.:
            if 3 in control:
                raise ValueError("initial stress rate ambiguous")

            elif 4 in control and any(x != 0. for x in Cij):
                raise ValueError("nonzero initial stress not yet supported")

        # Replace leg with modfied values
        leg = [termination_time, num_steps]
        leg.extend(control)
        leg.extend(Cij)
        leg.append(ndumps)
        leg.extend(efcomp)
        leg.append(temp)
        leg.extend(user_field)
        path[ileg] = leg

        if termination_time > tterm:
            del path[ileg+1:]
            break

        continue

    return np.array(path)

def parse_function_path(lines, functions, num_steps, cfmt):
    """Parse the path given by functions

    """
    start_time = 0.
    leg_num = 1
    if not lines:
        raise GenericError("Empty path encountered")
        return
    elif len(lines) > 1:
        raise GenericError("Only one line of table functions allowed, "
                           "got {0}".format(len(lines)))
        return

    # format functions
    functions = format_functions(functions)

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
            raise GenericError("expected integer function ID, got {0}".format(fid))
            continue
        try:
            scale = float(scale)
        except ValueError:
            raise GenericError("expected real function scale for function {0}"
                               ", got {1}".format(fid, scale))
            continue

        fcn = functions.get(fid)
        if fcn is None:
            raise GenericError("{0}: function not defined".format(fid))
            continue
        Cij.append((scale, fcn))

    # --- Check lengths of Cij and control are consistent
    if len(Cij) != len(control):
        raise UserInputError("Path: len(Cij) != len(control) in leg {0}"
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

def format_functions(funcs):
    functions = dict(DEFAULT_FUNCTIONS)
    if isinstance(funcs, Function):
        functions[funcs.func_id] = funcs
    else:
        for func in funcs:
            if not isinstance(func, Function):
                raise GenericError("functions must be instances "
                                   "of utils.functions.Function")
            functions[func.func_id] = func
    return functions

def parse_table_path(lines, tfmt, cols, cfmt, lineskip):
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
            raise UserInputError("Expected floats in leg {0}, got {1}".format(
                leg_num, line))
            continue
        tbl.append(line)
    tbl = np.array(tbl)

    # if cols was not specified, must want all
    if not cols:
        columns = list(range(tbl.shape[1]))
    else:
        columns = cols

    for line in tbl:
        try:
            line = line[columns]
        except IndexError:
            raise UserInputError("Requested column not found in leg "
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
            raise UserInputError("Path: len(Cij) != len(control) in leg {0}"
                                 .format(leg_num))

        path.append([termination_time, num_steps, control, Cij])
        leg_num += 1
        continue

    return path
