import numpy as np
from utils.errors import MatModLabError
from core.legs import SingleLeg
from core.functions import _Function as Function, DEFAULT_FUNCTIONS
from utils.constants import DEFAULT_TEMP, NTENS, NSYMM
import utils.mmlabpack as mmlabpack

def cflags(key, s=0, r=0):
    d = {"D": 1,  # strain rate
         "E": 2,  # strain
         "R": 3,  # stress rate
         "S": 4,  # stress
         "F": 5,  # deformation gradient
         "P": 6,  # electric field
         "T": 7,  # temperature
         "U": 8,  # displacement
         "X": 9}  # user defined field
    if r:
        return dict([(v,k) for (k,v) in d.items()]).get(key)

    try:
        value = int(key)
        if value not in d.values():
            value = None
    except ValueError:
        try:
            value = d[key.upper()]
        except KeyError:
            # bad key
            value = None
    if s == 1 and (value is None or value not in (1, 2, 3, 4, 5)):
        message = "unexpected control flag {0}".format(key)
        raise MatModLabError(message)
    elif s == 2 and (value is None or value not in (1, 2, 3, 4, 5, 8)):
        message = "unexpected control flag {0}".format(key)
        raise MatModLabError(message)
    return value

def continuum_legs(path_input, path, num_steps, amplitude,
                   rate_multiplier, step_multiplier, num_io_dumps,
                   termination_time, tfmt, cols, cfmt, skiprows, functions,
                   kappa, estar, tstar, sstar, fstar, efstar, dstar):

    p = path_input.lower()
    if p[:4] not in ("defa", "func", "tabl"):
        raise MatModLabError("{0}: path_input not "
                             "recognized".format(path_input))

    if p == "default":
        path = _parse_default_path(path)

    elif p == "function":
        num_steps = num_steps or 1
        if cfmt is None:
            raise MatModLabError("function path: expected keyword cfmt")
        num_steps = int(num_steps * step_multiplier)
        path = _parse_function_path(path, functions, num_steps, cfmt)

    elif p == "table":
        if cfmt is None:
            raise MatModLabError("table path: expected keyword cfmt")
        if cols is None:
            raise MatModLabError("table path: expected keyword cols")
        if not isinstance(cols, (list, tuple)):
            raise MatModLabError("table path: expected cols to be a list")
        path = _parse_table_path(path, tfmt, cols, cfmt, skiprows)

    legs = []
    for (t, dt, c, n, T, ef, uf) in path:
        if termination_time is not None and t + dt > termination_time:
            break
        leg = continuum_leg(t, dt, c, num_steps=n, num_io_dumps=num_io_dumps,
                            elec_field=ef, temp=T, user_field=uf, kappa=kappa,
                            amplitude=amplitude, rate_multiplier=rate_multiplier,
                            step_multiplier=step_multiplier,
                            estar=estar, tstar=tstar, sstar=sstar, fstar=fstar,
                            efstar=efstar, dstar=dstar)
        legs.append(leg)

    return legs

def continuum_leg(start_time, time_step, components, num_steps=1,
                  num_io_dumps=1000000, elec_field=None, temp=DEFAULT_TEMP,
                  user_field=None, kappa=0., amplitude=1.,
                  rate_multiplier=1., step_multiplier=1.,
                  estar=1., tstar=1., sstar=1., fstar=1., efstar=1., dstar=1.):
    """Constructor for a continuum leg"""

    # set defaults
    num_steps = int(step_multiplier * num_steps)

    if elec_field is None:
        elec_field = np.zeros(3)
    elec_field = np.asarray(elec_field)

    if user_field is None:
        user_field = []
    user_field = np.asarray(user_field)

    # Separate the identifiers and components
    control = np.array([cflags(c[0], s=2) for c in components], dtype=np.int)
    components = np.array([float(c[1]) for c in components], dtype=np.float64)

    # multiplication factors
    efac = amplitude * estar
    tfac = abs(amplitude) * tstar / rate_multiplier
    sfac = amplitude * sstar
    ffac = amplitude * fstar
    effac = amplitude * efstar
    dfac = amplitude * dstar

    termination_time = tfac * (start_time + time_step)
    time_step = termination_time - start_time
    elec_field *= effac

    # for now unit tensor for rotation
    Rij = np.reshape(np.eye(3), (NTENS,))

    # the goal is to convert all input deformations to either stresses or
    # strains
    if 5 in control:

        # deformation gradient control
        if np.any(np.abs(control - 5) != 0):
            raise MatModLabError("mixed mode deformation not allowed "
                                 "with deformation gradient control")
        if len(control) != 9:
            raise MatModLabError("must specify all components of "
                                 "deformation gradient")

        defgrad = np.reshape(ffac * components, (3, 3))
        jac = np.linalg.det(defgrad)
        if jac <= 0:
            raise MatModLabError("negative Jacobian ({0:f})".format(jac))

        # convert deformation gradient to strain E with associated
        # rotation given by axis of rotation x and angle of rotation theta
        Rij, Vij = np.linalg.qr(defgrad)
        if np.max(np.abs(Rij - np.eye(3))) > np.finfo(np.float).eps:
            raise MatModLabError("QR decomposition of deformation gradient "
                                 "gave unexpected rotations (rotations are "
                                 "not yet supported)")
        Uij = np.dot(Rij.T, np.dot(Vij, Rij))
        components = mmlabpack.u2e(Uij, kappa)
        Rij = np.reshape(Rij, (NTENS,))

        # deformation gradient now converted to strains
        control = np.array([2] * NSYMM, dtype=np.int)

    elif 8 in control:
        # displacement control check
        if np.any(np.abs(control - 8) != 0):
            raise MatModLabError("mixed mode deformation not allowed "
                                 "with displacement control")
        if len(control) > 3:
            raise MatModLabError("must specify only components of "
                                 "displacement with displacement control")
        elif len(control) < 3:
            components = np.append(components, [0.] * 3 - len(control))
            control = np.array([8, 8, 8], dtype=np.int)

        # convert displacments to strains
        Uij = np.zeros((3, 3))
        Uij[DI3] = dfac * components[:3] + 1.
        components = mmlabpack.u2e(Uij, kappa, 1)

        # displacements now converted to strains
        control = np.array([2] * NSYMM, dtype=np.int)

    elif 2 in control and len(control) == 1:
        # only one strain value given -> volumetric strain
        evol = components[0]
        if kappa * evol + 1. < 0.:
            raise MatModLabError("1 + kappa * ev must be positive")

        if kappa == 0.:
            eij = evol / 3.

        else:
            eij = ((kappa * evol + 1.) ** (1. / 3.) - 1.)
            eij = eij / kappa

        control = np.array([2] * NSYMM, dtype=np.int)
        components = np.array([eij, eij, eij, 0., 0., 0.], dtype=np.float64)

    elif 4 in control and len(control) == 1:
        # only one stress value given -> pressure
        Sij = -components[0]
        control = np.array([4, 4, 4, 2, 2, 2], dtype=np.int)
        components = np.array([Sij, Sij, Sij, 0., 0., 0.], dtype=np.float64)

    N = len(control)
    control = np.append(control, [2] * (NSYMM - N))
    components = np.append(components, [0.] * (NSYMM - N))

    # At this point, components has only strains and stresses
    components[np.where((control==1) | (control==3))] *= rate_multiplier
    components[np.where(control==2)] *= efac
    components[np.where(control==4)] *= sfac

    bad = np.where(kappa * components[np.where(control==2)] + 1. < 0.)
    if np.any(bad):
        idx = str(bad[0])
        raise MatModLabError("1 + kappa*E[{0}] must be positive".format(idx))

    # initial stress check
    if abs(termination_time) < 1.e-16:
        if 3 in control:
            raise MatModLabError("initial stress rate ambiguous")

        elif 4 in control and any(x != 0. for x in components):
            raise MatModLabError("nonzero initial stress not yet supported")

    return SingleLeg(start_time, time_step, NSYMM, control, components,
                     num_steps, num_io_dumps, elec_field, temp, user_field)

def _parse_default_path(lines):
    """Parse the individual path"""
    path = []
    final_time = 0.
    leg_num = 1
    start_time = 0.
    for line in lines:
        if not line:
            continue

        termination_time, num_steps, control = line[:3]

        # check entries
        # --- termination time
        termination_time = fmtterm(leg_num, termination_time, final_time)
        final_time = termination_time

        # --- components of deformation
        components = fmtcomps(line[3:])

        # --- number of steps
        num_steps = fmtnsteps(num_steps, leg_num)

        # --- control
        control = fmtctrl(control)

        # remove all non-deformations from control
        temp = components[np.where(control==7)]
        elec_field = components[np.where(control==6)]
        user_field = components[np.where(control==9)]

        ic = np.where((control!=6)&(control!=7)&(control!=9))
        components = components[ic]
        control = control[ic]

        ntemp = temp.shape[0]
        if ntemp > 1:
            raise MatModLabError("multiple temperature fields in "
                                 "leg {0}".format(leg_num))
        try:
            temp = temp[0]
        except IndexError:
            temp = None

        # --- Check lengths of Cij and control are consistent
        if len(components) != len(control):
            raise MatModLabError("len(components) != len(control) in "
                                 "leg {0}".format(leg_num))

        time_step = termination_time - start_time
        path.append([start_time, time_step, zip(control, components),
                     num_steps, temp, elec_field, user_field])
        leg_num += 1
        start_time = termination_time
        continue

    return path

def _parse_function_path(lines, funcs, num_steps, control):
    """Parse the path given by functions

    """
    leg_num = 1
    if not lines:
        raise MatModLabError("Empty path encountered")
    elif len(lines) > 1:
        raise MatModLabError("Only one line of table functions allowed, "
                             "got {0}".format(len(lines)))

    # format functions
    functions = dict(DEFAULT_FUNCTIONS)
    if isinstance(funcs, Function):
        functions[funcs.func_id] = funcs
    else:
        for func in funcs:
            if not isinstance(func, Function):
                raise MatModLabError("functions must be instances "
                                     "of utils.functions.Function")
            functions[func.func_id] = func

    termination_time = lines[0][0]
    cijfcns = lines[0][1:]

    # check entries
    # --- termination time
    termination_time = fmtterm(1, termination_time, -1)
    final_time = termination_time

    # --- control
    control = fmtctrl(control)

    # --- get the actual functions
    components = []
    for icij, cijfcn in enumerate(cijfcns):
        cijfcn = cijfcn.split(":")
        try:
            fid, scale = cijfcn
        except ValueError:
            fid, scale = cijfcn[0], 1
        try:
            fid = int(float(fid))
        except ValueError:
            raise MatModLabError("expected integer function ID, "
                                 "got {0}".format(fid))
            continue
        try:
            scale = float(scale)
        except ValueError:
            raise MatModLabError("expected real function scale for function {0}"
                                 ", got {1}".format(fid, scale))

        fcn = functions.get(fid)
        if fcn is None:
            raise MatModLabError("{0}: function not defined".format(fid))
        components.append((scale, fcn))
    components = np.array(components)

    # --- Check lengths of components and control are consistent
    if len(components) != len(control):
        raise MatModLabError("len(Cij) != len(control) in "
                             "leg {0}".format(leg_num))

    # temperature field
    temp = components[np.where(control==7)]
    ntemp = temp.shape[0]
    if ntemp > 1:
        raise MatModLabError("multiple temperature fields in "
                             "leg {0}".format(leg_num))
    try:
        s, f = temp[0]
        temp = lambda t: s * f(t)
    except IndexError:
        temp = lambda t: None

    # electric field
    ef = components[np.where(control==6)]
    if ef.shape[0] > 0:
        f = np.array(ef)
        ef = lambda t, f=f: np.array([f[i,0]*f[i,1](t) for i in range(len(f))])
    else:
        ef = lambda t: None

    # user field
    uf = components[np.where(control==9)]
    if uf.shape[0] > 0:
        f = np.array(uf)
        uf = lambda t, f=f: np.array([f[i,0]*f[i,1](t) for i in range(len(f))])
    else:
        uf = lambda t: None

    ic = np.where((control!=6)&(control!=7)&(control!=9))
    components = components[ic]
    control = control[ic]

    path = []
    start_time = 0.
    cij = np.array([s*f(0) for (s, f) in components], dtype=np.float64)
    path.append([0, 0, zip(control, cij), 1, temp(0), ef(0), uf(0)])
    for time in np.linspace(start_time, final_time, num_steps-1):
        if abs(time - start_time) < 1.e-16:
            continue
        cij = np.array([s*f(time) for (s, f) in components], dtype=np.float64)
        time_step = time - start_time
        leg = [start_time, time_step, zip(control, cij), 1,
               temp(time), ef(time), uf(time)]
        path.append(leg)
        start_time = time

    return path

def _parse_table_path(lines, tfmt, cols, control, lineskip):
    """Parse the path table

    """
    path = []
    final_time = 0.
    termination_time = 0.

    # --- control
    control = fmtctrl(control)

    if isinstance(lines, np.ndarray):
        table = np.array(lines)
    else:
        table = []
        leg_num = 1
        for idx, line in enumerate(lines):
            if idx < lineskip or not line:
                continue
            if line[0].strip().startswith("#"):
                continue
            try:
                line = [float(x) for x in line]
            except ValueError:
                raise MatModLabError("expected floats in leg {0}, "
                                     "got {1}".format(leg_num, line))
            table.append(line)
            leg_num += 1
        table = np.array(table)

    # if cols was not specified, must want all
    if not cols:
        columns = list(range(table.shape[1]))
    else:
        columns = cols

    start_time = 0.
    for (iline, line) in enumerate(table):
        leg_num = iline + 1
        try:
            line = line[columns]
        except IndexError:
            raise MatModLabError("requested column not found in leg "
                                 "{0}".format(leg_num))

        if tfmt == "dt":
            termination_time += line[0]
        else:
            termination_time = line[0]
        time_step = termination_time - start_time

        components = line[1:]

        # check entries
        # --- termination time
        termination_time = fmtterm(leg_num, termination_time, final_time)
        if termination_time is None:
            continue
        final_time = termination_time

        # --- number of steps
        num_steps = 1

        # --- Check lengths of components and control are consistent
        if len(components) != len(control):
            raise MatModLabError("len(components) != len(control) "
                                 "in leg {0}".format(leg_num))

        # remove all non-deformations from control
        temp = components[np.where(control==7)]
        elec_field = components[np.where(control==6)]
        user_field = components[np.where(control==9)]

        ic = np.where((control!=6)&(control!=7)&(control!=9))
        components = components[ic]
        control = control[ic]

        ntemp = temp.shape[0]
        if ntemp > 1:
            raise MatModLabError("multiple temperature fields in "
                                 "leg {0}".format(leg_num))
        try:
            temp = temp[0]
        except IndexError:
            temp = None

        # determine start time
        time_step = termination_time - start_time
        leg = [start_time, time_step, zip(control, components), num_steps,
               temp, elec_field, user_field]
        path.append(leg)

        leg_num += 1
        start_time = termination_time
        continue

    return path

def fmtterm(leg_num, tterm, tfinal):
    try:
        tterm = float(tterm)
    except ValueError:
        raise MatModLabError("expected float for termination time of "
                             "leg {0} got {1}".format(leg_num,
                                                      tterm))
    if tterm < 0.:
        raise MatModLabError("expected positive termination time leg {0} "
                             "got {1}".format(leg_num, tterm))
    if tterm < tfinal:
        raise MatModLabError("expected time to increase monotonically in "
                             "leg {0}".format(leg_num))

    return tterm

def fmtnsteps(num_steps, leg_num=None):
    try:
        num_steps = int(num_steps)
    except ValueError:
        raise MatModLabError("expected integer number of steps in "
                             "leg {0} got {1}".format(leg_num, num_steps))
    if num_steps < 0:
        raise MatModLabError("expected positive integer number of "
                             "steps in leg {0} got {1}".format(
                                 leg_num, num_steps))
    return num_steps

def fmtcomps(x):
    try:
        components = np.array(x, dtype=np.float64)
    except ValueError:
        string = ",".join("{0}".format(_) for _ in x)
        raise MatModLabError("expected components of deformation "
                             "to be floats, got {0}".format(string))
    return components

def fmtctrl(control):
    control = [cflags(s) for s in control]
    bad = [i for (i, c) in enumerate(control) if c is None]
    if bad:
        bad = ",".join(bad)
        raise MatModLabError("unexpected control flag[s] "
                             "{0} (leg {1})".format(bad, leg_num))
    control = np.array(control, dtype=np.int)
    return control

Leg = continuum_leg
