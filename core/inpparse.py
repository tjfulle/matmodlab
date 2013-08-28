import os
import re
import sys
import math
import numpy as np
import xml.dom.minidom as xdom

if __name__ == "__main__":
    D = os.path.dirname(os.path.realpath(__file__))
    sys.path.insert(0, os.path.join(D, "../"))

from __config__ import cfg
import utils.tensor as tensor
import utils.xmltools as xmltools
from core.io import Error1
from drivers.driver import isdriver
from utils.namespace import Namespace
from utils.pprepro import find_and_make_subs, find_and_fill_includes
from utils.fcnbldr import build_lambda, build_interpolating_function
from utils.opthold import OptionHolder
from materials.material import get_material_from_db


S_PHYSICS = "Physics"
S_PERMUTATION = "Permutation"
S_OPT = "Optimization"

S_AUX_FILES = "Auxiliary Files"
S_METHOD = "Method"
S_PARAMS = "Parameters"
S_OBJ_FCN = "Objective Function"
S_MITER = "Maximum Iterations"
S_TOL = "Tolerance"
S_DISP = "Disp"
S_TTERM = "Termination Time"

INP_ERRORS = 0
def fatal_inp_error(message):
    global INP_ERRORS
    INP_ERRORS += 1
    sys.stderr.write("*** error: {0}\n".format(message))


def parse_input(filepath):
    """Parse input file contents

    Parameters
    ----------
    user_input : str
    The user input

    """
    # find all "include" files, and preprocess the input
    user_input = find_and_fill_includes(open(filepath, "r").read())
    user_input, nsubs = find_and_make_subs(user_input, disp=1)
    if nsubs:
        with open(filepath + ".preprocessed", "w") as fobj:
            fobj.write(user_input)

    # Parse the xml document, the root element is always "GMDSpec"
    doc = xdom.parseString(user_input)
    try:
        gmdspec = doc.getElementsByTagName("GMDSpec")[0]
    except IndexError:
        raise Error1("Expected Root Element 'GMDSpec'")

    # ------------------------------------------ get and parse blocks --- #
    gmdblks = {}
    #           (blkname, required, remove)
    rootblks = ((S_OPT, 0, 1),
                (S_PERMUTATION, 0, 1),
                (S_PHYSICS, 1, 0))
    # find all functions first
    functions = pFunctions(gmdspec.getElementsByTagName("Function"))
    args = (functions,)

    for (rootblk, reqd, rem) in rootblks:
        rootlmns = gmdspec.getElementsByTagName(rootblk)
        if not rootlmns:
            if reqd:
                raise Error1("GMDSpec: {0}: block missing".format(rootblk))
            continue
        if len(rootlmns) > 1:
            raise Error1("Expected 1 {0} block, got {1}".format(
                rootblk, len(rootlmns)))
        rootlmn = rootlmns[0]
        gmdblks[rootblk] = rootlmn
        if rem:
            p = rootlmn.parentNode
            p.removeChild(rootlmn)

    if S_OPT in gmdblks and S_PERMUTATION in gmdblks:
        raise Error1("Incompatible blocks: [Optimzation, Permutation]")

    if S_OPT in gmdblks:
        return optimization_namespace(gmdblks[S_OPT], gmdspec.toxml())

    elif S_PERMUTATION in gmdblks:
        return permutation_namespace(gmdblks[S_PERMUTATION], gmdspec.toxml())

    else:
        return physics_namespace(gmdblks[S_PHYSICS], *args)


# ------------------------------------------------- Parsing functions --- #
# Each XML block is parsed with a corresponding function 'pBlockName'
# where BlockName is the name of the block as entered by the user in the
# input file
def pLegs(leglmn, *args):
    """Parse the Legs block and set defaults

    """
    functions = args[0]

    # Set up options for legs
    options = OptionHolder()
    options.addopt("kappa", 0.)
    options.addopt("amplitude", 1.)
    options.addopt("ratfac", 1.)
    options.addopt("nfac", 1.)
    options.addopt("tstar", 1., test=lambda x: x > 0.)
    options.addopt("estar", 1.)
    options.addopt("sstar", 1.)
    options.addopt("fstar", 1.)
    options.addopt("efstar", 1.)
    options.addopt("dstar", 1.)
    options.addopt("format", "default", dtype=str,
                   choices=("default", "table", "fcnspec"))
    options.addopt("proportional", 0, dtype=mybool)
    options.addopt("ndumps", "20", dtype=str)

    # the following options are for table formatted legs
    options.addopt("tblcols", "1:7", dtype=str)
    options.addopt("tbltfmt", "time", dtype=str, choices=("time", "dt"))
    options.addopt("tblcfmt", "222222", dtype=str)

    # Get control terms
    for i in range(leglmn.attributes.length):
        options.setopt(*xmltools.get_name_value(leglmn.attributes.item(i)))

    # Read in the actual legs - splitting them in to lists
    lines = []
    for node in leglmn.childNodes:
        if node.nodeType == node.COMMENT_NODE:
            continue
        lines.extend([" ".join(xmltools.uni2str(item).split())
                      for item in node.nodeValue.splitlines() if item.split()])
    lines = [xmltools.str2list(line) for line in lines]

    # parse the legs depending on type
    if options.getopt("format") == "default":
        legs = parse_legs_default(lines)

    elif options.getopt("format") == "table":
        legs = parse_legs_table(lines, options.getopt("tbltfmt"),
                                options.getopt("tblcols"),
                                options.getopt("tblcfmt"))

    elif options.getopt("format") == "fcnspec":
        legs = parse_legs_cijfcn(lines, functions)

    else:
        raise Error1("Legs: {0}: invalid format".format(options.getopt("format")))

    legs = format_legs(legs, options)

    proportional = options.getopt("proportional")

    return legs, options.getopt("kappa"), proportional


def parse_legs_default(lines):
    """Parse the individual legs

    """
    legs = []
    final_time = 0.
    leg_num = 1
    for line in lines:
        if not line:
            continue
        termination_time, num_steps, control_hold = line[:3]
        Cij_hold = line[3:]

        # check entries
        # --- termination time
        try:
            termination_time = float(termination_time)
        except ValueError:
            raise Error1("Legs: termination time of leg {0} must be a float, "
                         "got {1}".format(leg_num, termination_time))
        if termination_time < 0.:
            raise Error1("Legs: termination time {0} of leg {1} must be "
                         "positive".format(termination_time, leg_num))
        elif termination_time < final_time:
            raise Error("Legs: time must increase monitonically at leg "
                        "{0}".format(leg_num))
        final_time = termination_time

        # --- number of steps
        try:
            num_steps = int(num_steps)
        except ValueError:
            raise Error1("Legs: number of steps of leg {0} must be an integer, "
                         "got {1}".format(leg_num, num_steps))
        if num_steps < 0:
            raise Error1("Legs: number of steps {0} of leg {1} must be "
                         "positive".format(num_steps, leg_num))

        # --- control
        control = format_leg_control(control_hold, leg_num=leg_num)

        # --- Cij
        Cij = []
        for (i, comp) in enumerate(Cij_hold):
            try:
                comp = float(comp)
            except ValueError:
                raise Error1("Legs: Component {0} of leg {1} must be a "
                             "float, got {2}".format(i+1, leg_num, comp))
            Cij.append(comp)

        Cij = np.array(Cij)

        # --- Check lengths of Cij and control are consistent
        if len(Cij) != len(control):
            raise Error1("Legs: len(Cij) != len(control) in leg {0}"
                         .format(leg_num))

        legs.append([termination_time, num_steps, control, Cij])
        leg_num += 1
        continue

    return legs


def parse_legs_cijfcn(lines, functions):
    """Parse the individual legs

    """
    start_time = 0.
    leg_num = 1

    if not lines:
        raise Error1("No table functions defined")
    elif len(lines) > 1:
        raise Error1("Only one line of table functions allowed, "
                     "got {0}".format(len(lines)))

    termination_time, num_steps, control_hold = lines[0][:3]
    cijfcns = lines[0][3:]

    # check entries
    # --- termination time
    try:
        termination_time = float(termination_time)
    except ValueError:
        raise Error1("Legs: termination time must be a float, "
                     "got {0}".format(termination_time))
    if termination_time < 0.:
        raise Error1("Legs: termination time {0} must be "
                     "positive".format(termination_time))
    final_time = termination_time

    # --- number of steps
    try:
        num_steps = int(num_steps)
    except ValueError:
        raise Error1("Legs: number of steps must be an integer, "
                     "got {0}".format(num_steps))
    if num_steps < 0:
        raise Error1("Legs: number of steps {0} must be "
                     "positive".format(num_steps))

    # --- control
    control = format_leg_control(control_hold, leg_num=leg_num)

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
            raise Error1("Function ID must be an integer, got {0}".format(fid))
        try:
            scale = float(scale)
        except ValueError:
            raise Error1("Function scale must be a float, got {0}".format(scale))

        fcn = functions.get(fid)
        if fcn is None:
            raise Error1("{0}: function not defined".format(fid))
        Cij.append((scale, fcn))

    # --- Check lengths of Cij and control are consistent
    if len(Cij) != len(control):
        raise Error1("Legs: len(Cij) != len(control) in leg {0}"
                     .format(leg_num))

    legs = []
    for time in np.linspace(start_time, final_time, num_steps):
        leg = [time, 1, control]
        leg.append(np.array([s * f(time) for (s, f) in Cij]))
        legs.append(leg)

    return legs

def parse_legs_table(lines, tbltfmt, tblcols, tblcfmt):
    """Parse the legs table

    """
    legs = []
    final_time = 0.
    termination_time = 0.
    leg_num = 1

    # Convert tblcols to a list
    columns = format_tbl_cols(tblcols)

    # check the control
    control = format_leg_control(tblcfmt)

    for line in lines:
        if not line:
            continue
        try:
            line = np.array([float(x) for x in line])
        except ValueError:
            raise Error1("Expected floats in leg {0}, got {1}".format(
                leg_num, line))
        try:
            line = line[columns]
        except IndexError:
            raise Error1("Requested column not found in leg {0}".format(leg_num))

        if tbltfmt == "dt":
            termination_time += line[0]
        else:
            termination_time = line[0]

        Cij = line[1:]

        # check entries
        # --- termination time
        if termination_time < 0.:
            raise Error1("Legs: termination time {0} of leg {1} must be "
                         "positive".format(termination_time, leg_num))
        elif termination_time < final_time:
            raise Error("Legs: time must increase monitonically at leg "
                        "{0}".format(leg_num))
        final_time = termination_time

        # --- number of steps
        num_steps = 1

        # --- Check lengths of Cij and control are consistent
        if len(Cij) != len(control):
            raise Error1("Legs: len(Cij) != len(control) in leg {0}"
                         .format(leg_num))

        legs.append([termination_time, num_steps, control, Cij])
        leg_num += 1
        continue

    return legs


def format_leg_control(cfmt, leg_num=None):
    leg = "" if leg_num is None else "(leg {0})".format(leg_num)
    valid_control_flags = [1, 2, 3, 4, 5, 6, 8, 9]
    control = []
    for (i, flag) in enumerate(cfmt):
        try:
            flag = int(flag)
        except ValueError:
            raise Error1("Legs: control flag {0} must be an "
                         "integer, got {1} {2}".format(i+1, flag, leg))

        if flag not in valid_control_flags:
            valid = ", ".join(xmltools.stringify(x)
                              for x in valid_control_flags)
            raise Error1("Legs: {0}: invalid control flag choose from "
                         "{1} {2}".format(flag, valid, leg))

        control.append(flag)

    if 5 in control:
        if any(flag != 5 and flag not in (6, 9) for flag in control):
            raise Error1("Legs: mixed mode deformation not allowed with "
                         "deformation gradient control {0}".format(leg))

        # must specify all components
        elif len(control) != 9:
            raise Error1("all 9 components of deformation gradient must "
                         "be specified {0}".format(leg))

    if 8 in control:
        # like deformation gradient control, if displacement is specified
        # for one, it must be for all
        if any(flag != 8 and flag not in (6, 9) for flag in control):
            raise Error1("Legs: mixed mode deformation not allowed with "
                         "displacement control {0}".format(leg))

        # must specify all components
        elif len(control) != 3:
            raise Error1("all 3 components of displacement must "
                         "be specified {0}".format(leg))

    return np.array(control, dtype=np.int)


def format_tbl_cols(tblcols):
    columns = []
    for item in [x.split(":")
                 for x in xmltools.str2list(re.sub(r"\s*:\s*", ":", tblcols))]:
        try:
            item = [int(x) for x in item]
        except ValueError:
            raise Error1("Legs: tblcols items must be int, got "
                         "{0}".format(tblcols))
        item[0] -= 1

        if len(item) == 1:
            columns.append(item[0])
        elif len(item) not in (2, 3):
            raise Error1("Legs: tblcfmt range must be specified as "
                         "start:end:[step], got {0}".format(
                             ":".join(str(x) for x in item)))
        if len(item) == 2:
            columns.extend(range(item[0], item[1]))
        elif len(item) == 3:
            columns.extend(range(item[0], item[1], item[2]))
    return columns


def format_legs(legs, options):
    """Format the legs by applying multipliers

    """
    # stress control if any of the control types are 3 or 4
    stress_control = any(c in (3, 4) for leg in legs for c in leg[2])
    kappa = options.getopt("kappa")
    if stress_control and kappa != 0.:
        raise Error1("kappa must be 0 with stress control option")

    # From these formulas, note that AMPL may be used to increase or
    # decrease the peak strain without changing the strain rate. ratfac is
    # the multiplier on strain rate and stress rate.
    amplitude = options.getopt("amplitude")
    ratfac = options.getopt("ratfac")
    nfac = options.getopt("nfac")
    ndumps = options.getopt("ndumps")
    if ndumps == "all":
        ndumps = 100000000
    ndumps= int(ndumps)

    # factors to be applied to deformation types
    efac = amplitude * options.getopt("estar")
    tfac = abs(amplitude) * options.getopt("tstar") / ratfac
    sfac = amplitude * options.getopt("sstar")
    ffac = amplitude * options.getopt("fstar")
    effac = amplitude * options.getopt("efstar")
    dfac = amplitude * options.getopt("dstar")

    # for now unit tensor for rotation
    Rij = np.reshape(np.eye(3), (9,))

    # format each leg
    for ileg, (termination_time, num_steps, control, Cij) in enumerate(legs):

        leg_num = ileg + 1

        num_steps = int(nfac * num_steps)
        termination_time = tfac * termination_time

        # pull out electric field from other deformation specifications
        efcomp = np.zeros(3)
        trtbl = np.array([True] * len(control))
        j = 0
        for i, c in enumerate(control):
            if c == 6:
                efcomp[j] = effac * Cij[i]
                trtbl[i] = False
                j += 1
        Cij = Cij[trtbl]
        control = control[trtbl]

        if 5 in control:
            # check for valid deformation
            defgrad = np.reshape(ffac * Cij, (3, 3))
            jac = np.linalg.det(defgrad)
            if jac <= 0:
                raise Error1("Inadmissible deformation gradient in leg "
                             "{0} gave a Jacobian of {1:f}".format(leg_num, jac))

            # convert defgrad to strain E with associated rotation given by
            # axis of rotation x and angle of rotation theta
            Rij, Vij = np.linalg.qr(defgrad)
            if np.max(np.abs(Rij - np.eye(3))) > np.finfo(np.float).eps:
                raise Error1("Rotation encountered in leg {0}. "
                             "Rotations are not yet supported".format(leg_num))
            Uij = tensor.asarray(np.dot(Rij.T, np.dot(Vij, Rij)))
            Rij = np.reshape(Rij, (9,))
            Cij = tensor.u2e(Uij, kappa)

            # deformation gradient now converted to strains
            control = np.array([2] * 6, dtype=np.int)

        elif 8 in control:
            # displacement control check
            # convert displacments to strains
            Uij = np.zeros(6)
            Uij[:3] = dfac * Cij[:3] + 1.
            Cij = tensor.u2e(Uij, kappa)

            # displacements now converted to strains
            control = np.array([2] * 6, dtype=np.int)

        elif 2 in control and len(control) == 1:
            # only one strain value given -> volumetric strain
            evol = Cij[0]
            if kappa * evol + 1. < 0.:
                raise Error1("1 + kappa * ev must be positive in leg "
                             "{0}".format(leg_num))

            if kappa == 0.:
                eij = evol / 3.

            else:
                eij = ((kappa * evol + 1.) ** (1. / 3.) - 1.)
                eij = eij / kappa

            control = np.array([2] * 6, dtype=np.int)
            Cij = np.array([eij, eij, eij, 0., 0., 0.])

        elif 4 in control and len(control) == 1:
            # only one stress value given -> pressure
            Sij = -Cij[0]
            control = np.array([4, 4, 4, 2, 2, 2], dtype=np.int)
            Cij = np.array([Sij, Sij, Sij, 0., 0., 0.])

        if len(control) != len(Cij):
            raise Error1("len(cij) != len(control) in leg {0}".format(leg_num))

        control = np.append(control, [2] * (6 - len(control)))
        Cij = np.append(Cij, [0.] * (6 - len(Cij)))

        # adjust components based on user input
        for idx, ctype in enumerate(control):
            if ctype in (1, 3):
                # adjust rates
                Cij[idx] *= ratfac

            elif ctype == 2:
                # adjust strain
                Cij[idx] *= efac

                if kappa * Cij[idx] + 1. < 0.:
                    raise Error("1 + kappa*E[{0}] must be positive in "
                                "leg {1}".format(idx, leg_num))

            elif ctype == 4:
                # adjust stress
                Cij[idx] *= sfac

            continue

        # initial stress check
        if termination_time == 0.:
            if 3 in control:
                raise Error1("initial stress rate ambiguous")
            elif 4 in control and any(x != 4 for x in control):
                raise Error1("Mixed initial state not allowed")

        # Replace leg with modfied values
        legs[ileg][0] = termination_time
        legs[ileg][1] = num_steps
        legs[ileg][2] = control
        legs[ileg][3] = Cij
        legs[ileg].append(ndumps)

        # legs[ileg].append(Rij)
        legs[ileg].append(efcomp)

        continue

    return legs


def pOptimization(optlmn, *args):
    """Parse the optimization block

    """
    odict = {}

    # Set up options for permutation
    options = OptionHolder()
    options.addopt("method", "simplex", dtype=str,
                   choices=("simplex", "powell", "cobyla", "slsqp"))
    options.addopt("maxiter", 25, dtype=int)
    options.addopt("tolerance", 1.e-6, dtype=float)
    options.addopt("disp", 0, dtype=int)

    # Get control terms
    for i in range(optlmn.attributes.length):
        options.setopt(*xmltools.get_name_value(optlmn.attributes.item(i)))

    # objective function
    objfcn = optlmn.getElementsByTagName("ObjectiveFunction")
    if not objfcn:
        raise Error1("ObjectiveFunction not found")
    elif len(objfcn) > 1:
        raise Error1("Only one ObjectiveFunction tag supported")
    objfcn = objfcn[0]
    objfile = objfcn.getAttribute("href")
    if not objfile:
        raise Error1("Expected href attribute to ObjectiveFunction")
    elif not os.path.isfile(objfile):
        raise Error1("{0}: no such file".format(objfile))
    objfile = os.path.realpath(objfile)

    # auxiliary files
    auxfiles = []
    for item in optlmn.getElementsByTagName("AuxiliaryFile"):
        auxfile = item.getAttribute("href")
        if not auxfile:
            raise Error1("Expected href attribute to AuxiliaryFile")
        elif not os.path.isfile(auxfile):
            raise Error1("{0}: no such file".format(auxfile))
        auxfiles.append(os.path.realpath(auxfile))

    # read in optimization parameters
    p = []
    for items in optlmn.getElementsByTagName("Optimize"):
        var = str(items.attributes.get("var").value)

        ivalue = items.attributes.get("initial_value")
        if not ivalue:
            raise Error1("{0}: no initial value given".format(var))
        ivalue = float(ivalue.value)

        bounds = items.attributes.get("bounds")
        if not bounds:
            bounds = [None, None]
        else:
            bounds = str2list(bounds.value, dtype=float)
        if len(bounds) != 2:
            raise Error1("{0}: incorrect bounds, must give upper "
                         "and lower bound".format(var))
        p.append([var, ivalue, bounds])

    odict[S_METHOD] = options.getopt("method")
    odict[S_MITER] = options.getopt("maxiter")
    odict[S_TOL] = options.getopt("tolerance")
    odict[S_DISP] = options.getopt("disp")

    odict[S_PARAMS] = p
    odict[S_AUX_FILES] = auxfiles
    odict[S_OBJ_FCN] = objfile

    return odict


def pPermutation(permlmn, *args):
    """Parse the permutation block

    """
    pdict = {}

    # Set up options for permutation
    options = OptionHolder()
    options.addopt("method", "zip", dtype=str, choices=("zip", "combine"))
    options.addopt("seed", 12, dtype=int)

    # Get control terms
    for i in range(permlmn.attributes.length):
        options.setopt(*xmltools.get_name_value(permlmn.attributes.item(i)))

    rstate = np.random.RandomState(options.getopt("seed"))
    gdict = {"__builtins__": None}
    N_default = 10
    safe = {"range": lambda a, b, N=N_default: np.linspace(a, b, N),
            "sequence": lambda a: np.array(a),
            "weibull": lambda a, b, N=N_default: a * rstate.weibull(b, N),
            "uniform": lambda a, b, N=N_default: rstate.uniform(a, b, N),
            "normal": lambda a, b, N=N_default: rstate.normal(a, b, N),
            "percentage": lambda a, b, N=N_default: (
                np.linspace(a-(b/100.)*a, a+(b/100.)* a, N))}

    # read in permutated values
    p = []
    for items in permlmn.getElementsByTagName("Permutate"):
        var = str(items.attributes.get("var").value)
        values = str(items.attributes.get("values").value)
        try:
            p.append([var, eval(values, gdict, safe)])
        except:
            raise Error1("{0}: invalid extression".format(values))

    pdict[S_PARAMS] = p
    pdict[S_METHOD] = options.getopt("method")

    return pdict


def pExtract(extlmn, *args):

    options = OptionHolder()
    options.addopt("format", "ascii", dtype=str, choices=("ascii", "mathematica"))
    options.addopt("step", 1, dtype=int)
    options.addopt("ffmt", ".18f", dtype=str)

    # Get control terms
    for i in range(extlmn.attributes.length):
        options.setopt(*xmltools.get_name_value(extlmn.attributes.item(i)))

    variables = []
    for item in extlmn.getElementsByTagName("Variables"):
        data = item.firstChild.data.split("\n")
        data = [xmltools.stringify(x, "upper")
                for sub in data for x in sub.split()]
        if "ALL" in data:
            variables = "ALL"
            break
        variables.extend(data)
    return (options.getopt("format"), options.getopt("step"),
            options.getopt("ffmt"), variables)


def physics_namespace(physlmn, *args):
    simblk = pPhysics(physlmn, *args)

    # set up the namespace to return
    ns = Namespace()

    ns.stype = S_PHYSICS

    ns.ttermination = simblk.get(S_TTERM)

    ns.mtlmdl = simblk["Material"][0]
    ns.mtlprops = simblk["Material"][1]
    ns.density = simblk["Material"][2]

    ns.legs = simblk["Legs"][0]
    ns.kappa = simblk["Legs"][1]
    ns.proportional = simblk["Legs"][2]

    ns.extract = simblk.get("Extract")

    options = simblk.get("Options")
    ns.error = options["error"]
    ns.driver = options["driver"]

    return ns


def optimization_namespace(optlmn, basexml):
    optblk = pOptimization(optlmn)
    # set up the namespace to return
    ns = Namespace()
    ns.stype = S_OPT
    ns.method = optblk[S_METHOD]
    ns.parameters = optblk[S_PARAMS]
    ns.auxiliary_files = optblk[S_AUX_FILES]
    ns.objective_function = optblk[S_OBJ_FCN]
    ns.tolerance = optblk[S_TOL]
    ns.maxiter = optblk[S_MITER]
    ns.disp = optblk[S_DISP]
    ns.basexml = basexml
    return ns


def permutation_namespace(permlmn, basexml):
    permblk = pPermutation(permlmn)
    # set up the namespace to return
    ns = Namespace()
    ns.stype = S_PERMUTATION
    ns.method = permblk[S_METHOD]
    ns.parameters = permblk[S_PARAMS]
    ns.basexml = basexml
    return ns


def pPhysics(physlmn, *args):
    subblks = (("Material", 1), ("Legs", 1), ("Extract", 0))
    simblk = {}
    for (subblk, reqd) in subblks:
        sublmns = physlmn.getElementsByTagName(subblk)
        if not sublmns:
            if reqd:
                raise Error1("Physics: {0}: block missing".format(subblk))
            continue
        if len(sublmns) > 1:
            raise Error("Expected 1 {0} block, got {1}".format(
                subblk, len(sublmns)))
        sublmn = sublmns[0]
        parsefcn = getattr(sys.modules[__name__],
                           "p{0}".format(sublmn.nodeName))
        simblk[subblk] = parsefcn(sublmn, *args)
        p = sublmn.parentNode
        p.removeChild(sublmn)
    tlmn = physlmn.getElementsByTagName(S_TTERM)
    if tlmn:
        term_time = float(tlmn[0].firstChild.data)
        simblk[S_TTERM] = term_time
        p = tlmn[0].parentNode
        p.removeChild(tlmn[0])

    # Get options
    simblk["Options"] = {}
    options = OptionHolder()
    options.addopt("error", "all", dtype=str)
    options.addopt("driver", "solid", dtype=str)
    for i in range(physlmn.attributes.length):
        options.setopt(*xmltools.get_name_value(physlmn.attributes.item(i)))
    simblk["Options"]["error"] = options.getopt("error")
    driver = options.getopt("driver")
    if not isdriver(driver):
        raise Error1("{0}: unrecognized driver".format(driver))
    simblk["Options"]["driver"] = driver

    return simblk


def pMaterial(mtllmn, *args):
    """Parse the material block

    """
    model = mtllmn.attributes.get("model")
    if model is None:
        raise Error1("Material: model not found")
    model = str(model.value.lower())

    density = mtllmn.attributes.get("density")
    if density is None:
        density = 1.
    else:
        density = float(density.value)

    mtlmdl = get_material_from_db(model)
    if mtlmdl is None:
        raise Error1("{0}: material not in database".format(model))

    # mtlmdl.parameters is a comma separated list of parameters
    pdict = dict([(xmltools.stringify(n, "lower"), i)
                  for i, n in enumerate(mtlmdl.parameters.split(","))])
    params = np.zeros(len(pdict))
    for node in mtllmn.childNodes:
        if node.nodeType != mtllmn.ELEMENT_NODE:
            continue
        name = node.nodeName
        idx = pdict.get(name.lower())
        if idx is None:
            fatal_inp_error("Material: {0}: invalid parameter".format(name))
            continue
        try:
            val = float(node.firstChild.data)
        except ValueError:
            fatal_inp_error("Material: {0}: invalid value "
                            "{1}".format(name, node.firstChild))
        params[idx] = val

    if INP_ERRORS:
        raise Error1("Stopping due to previous errors")

    return model, params, density


def pFunctions(element_list, *args):
    """Parse the functions block

    """
    __ae__ = "ANALYTIC EXPRESSION"
    __pwl__ = "PIECEWISE LINEAR"
    const_fcn_id = 1
    functions = {const_fcn_id: lambda x: 1.}
    if not element_list:
        return functions

    for function in element_list:

        fid = function.attributes.get("id")
        if fid is None:
            raise Error1("Function: id not found")

        fid = int(fid.value)

        if fid == const_fcn_id:
            raise Error1("Function id {0} is reserved".format(fid))

        if fid in functions:
            raise Error1("{0}: duplicate function definition".format(fid))

        ftype = function.attributes.get("type")
        if ftype is None:
            raise Error1("Functions.Function: type not found")

        ftype = " ".join(ftype.value.split()).upper()

        if ftype not in (__ae__, __pwl__):
            raise Error1("{0}: invalid function type".format(ftype))

        expr = function.firstChild.data.strip()

        if ftype == __ae__:
            var = function.attributes.get("var")
            if not var:
                var = "x"
            else:
                var = var.value.strip()
            func, err = build_lambda(expr, var=var, disp=1)
            if err:
                raise Error1("{0}: in analytic expression in "
                             "function {1}".format(err, fid))

        elif ftype == __pwl__:
            # parse the table in expr

            try:
                columns = str2list(function.attributes.get("columns").value,
                                   dtype=str)
            except AttributeError:
                columns = ["x", "y"]

            except TypeError:
                columns = ["x", "y"]

            table = []
            ncol = len(columns)
            for line in expr.split("\n"):
                line = [float(x) for x in line.split()]
                if not line:
                    continue
                if len(line) != ncol:
                    nl = len(line)
                    raise Error1("Expected {0} columns in function "
                                 "{1}, got {2}".format(ncol, fid, nl))

                table.append(line)

            func, err = build_interpolating_function(np.array(table), disp=1)
            if err:
                raise Error1("{0}: in piecwise linear table in "
                             "function {1}".format(err, fid))

        functions[fid] = func
        continue

    return functions


def str2list(string, dtype=int):
    string = " ".join(string.split())
    string = re.sub(r"^[\(\[\{]", " ", string.strip())
    string = re.sub(r"[\)\]\}]$", " ", string.strip())
    string = re.sub(r"[, ]", " ", string)
    return [dtype(x) for x in string.split()]


def mybool(a):
    if str(a).lower().strip() in ("false", "no", "0"):
        return 0
    else:
        return 1
