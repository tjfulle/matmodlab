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
from utils.errors import Error1
from utils.namespace import Namespace
from utils.pprepro import find_and_make_subs, find_and_fill_includes
from materials.material import get_material_from_db


class OptionHolder(object):
    def __init__(self):
        pass

    def addopt(self, name, default, test=lambda x: True, dtype=float, choices=None):
        ns = Namespace()
        ns.name = name
        ns.value = default
        ns.test = test
        ns.dtype = dtype
        ns.choices = choices
        setattr(self, name, ns)

    def setopt(self, name, value):
        opt = self.getopt(name, getval=False)
        try:
            value = opt.dtype(value)
        except ValueError:
            raise Error1("{0}: invalid type for {1}".format(value, name))
        if not opt.test(value):
            raise Error1("{0}: invalid value for {1}".format(value, name))
        if opt.choices is not None:
            if value not in opt.choices:
                raise Error1("{0}: must be one of {1}, got {2}".format(
                    name, ", ".join(opt.choices), value))
        opt.value = value

    def getopt(self, name, getval=True):
        try: opt = getattr(self, name)
        except AttributeError: raise Error1("{0}: no such option".format(name))
        if getval:
            return opt.value
        return opt


def parse_input(user_input):
    """Parse input file contents

    Parameters
    ----------
    user_input : str
    The user input

    """
    user_input = find_and_fill_includes(user_input)
    user_input = find_and_make_subs(user_input)
    dom = xdom.parseString(user_input)

    # Get the root element (Should always be "GMDSpec")
    try:
        gmdspec = dom.getElementsByTagName("GMDSpec")[0]
    except IndexError:
        raise Error1("Expected Root Element 'GMDSpec'")

    # ------------------------------------------ get and parse blocks --- #
    gmdblks = {}
    rootblks = (("Optimization", 0, 1),
                ("Permutation", 0, 1),
                ("Simulation", 1, 0))
    for (rootblk, reqd, rem) in rootblks:
        rootlmns = gmdspec.getElementsByTagName(rootblk)
        if not rootlmns:
            if reqd:
                raise Error1("GMDSpec: {0}: block missing".format(rootblk))
            continue
        if len(rootlmns) > 1:
            raise Error("Expected 1 {0} block, got {1}".format(
                rootblk, len(rootlmns)))
        rootlmn = rootlmns[0]
        gmdblks[rootblk] = rootlmn
        if rem:
            p = rootlmn.parentNode
            p.removeChild(rootlmn)

    if "Optimzation" in gmdblks and "Permutation" in gmdblks:
        raise Error1("Incompatible blocks: [Optimzation, Permutation]")

    if "Optimization" in gmdblks:
        raise Error1("optimization parsing not done")

    elif "Permutation" in gmdblks:
        return permutation_namespace(gmdblks["Permutation"], gmdspec.toxml())
    else:
        return simulation_namespace(gmdblks["Simulation"])


# ------------------------------------------------- Parsing functions --- #
# Each XML block is parsed with a corresponding function 'pBlockName'
# where BlockName is the name of the block as entered by the user in the
# input file
def pLegs(leglmn):
    """Parse the Legs block and set defaults

    """
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
    options.addopt("format", "default", dtype=str)
    options.addopt("proportional", "0", dtype=mybool)

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
            control = np.array([4] * 6, dtype=np.int)
            Cij = np.array([Sij, Sij, Sij, 0., 0., 0.])

        if len(control) != 6:
            raise Error1("Bad control length in leg {0}".format(leg_num))

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
        # legs[ileg].append(Rij)
        legs[ileg].append(efcomp)

        continue

    return legs


def pOptimization(optlmn):
    raise Error1("Optimization coding not done")


# safe values to be used in eval
RAND = np.random.RandomState(17)
GDICT = {"__builtins__": None}
SAFE = {"sqrt": math.sqrt, "sin": math.sin, "cos": math.cos,
        "tan": math.tan, "asin": math.asin, "acos": math.acos,
        "atan": math.atan, "atan2": math.atan2, "pi": math.pi,
        "log": math.log, "exp": math.exp, "floor": math.floor,
        "ceil": math.ceil, "abs": math.fabs, "random": RAND.random_sample, }


def pPermutation(permlmn):
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
    safe = {"range": lambda a, b, N: np.linspace(a, b, N),
            "sequence": lambda a: np.array(a),
            "weibull": lambda a, b, N: a * rstate.weibull(b, N),
            "uniform": lambda a, b, N: rstate.uniform(a, b, N),
            "normal": lambda a, b, N: rstate.normal(a, b, N),
            "percentage": lambda a, b: np.array([a-(b/100.)*a, a, a+(b/100.)* a])}

    # read in permutated values
    p = {}
    for items in permlmn.getElementsByTagName("Permutate"):
        var = str(items.attributes.get("var").value)
        method = str(items.attributes.get("method").value)
        p[var] = eval(method, gdict, safe)

    pdict["parameters"] = p
    pdict["method"] = options.getopt("method")

    return pdict


def pExtract(extlmn):

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


def simulation_namespace(simlmn):
    simblk = pSimulation(simlmn)

    # set up the namespace to return
    ns = Namespace()

    ns.stype = "simulation"

    ns.mtlmdl = simblk["Material"][0]
    ns.mtlprops = simblk["Material"][1]
    ns.driver = simblk["Material"][2]
    ns.density = simblk["Material"][3]

    ns.legs = simblk["Legs"][0]
    ns.kappa = simblk["Legs"][1]
    ns.proportional = simblk["Legs"][2]

    ns.extract = simblk.get("Extract")

    return ns


def permutation_namespace(permlmn, basexml):

    permblk = pPermutation(permlmn)

    # set up the namespace to return
    ns = Namespace()

    ns.stype = "permutation"
    ns.method = permblk["method"]
    ns.parameters = permblk["parameters"]
    ns.basexml = basexml

    return ns


def pSimulation(simlmn):
    subblks = (("Material", 1), ("Legs", 1), ("Extract", 0))
    simblk = {}
    for (subblk, reqd) in subblks:
        sublmns = simlmn.getElementsByTagName(subblk)
        if not sublmns:
            if reqd:
                raise Error1("Simulation: {0}: block missing".format(subblk))
            continue
        if len(sublmns) > 1:
            raise Error("Expected 1 {0} block, got {1}".format(
                subblk, len(sublmns)))
        sublmn = sublmns[0]
        parsefcn = getattr(sys.modules[__name__],
                           "p{0}".format(sublmn.nodeName))
        simblk[subblk] = parsefcn(sublmn)
        p = sublmn.parentNode
        p.removeChild(sublmn)
    return simblk


def pMaterial(mtllmn):
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
            raise Error1("Material: {0}: invalid parameter".format(name))
        try:
            val = float(node.firstChild.data)
        except ValueError:
            raise Error1("Material: {0}: invalid value "
                         "{1}".format(name, node.firstChild))
        params[idx] = val

    return model, params, mtlmdl.driver, density


def mybool(a):
    if str(a).lower().strip() in ("false", "no", "0"):
        return 0
    else:
        return 1
