import os
import re
import sys
import math
import numpy as np
import xml.dom.minidom as xdom

if __name__ == "__main__":
    D = os.path.dirname(os.path.realpath(__file__))
    sys.path.insert(0, os.path.join(D, "../"))

from __config__ import cfg, F_MTL_PARAM_DB
import utils.tensor as tensor
import utils.xmltools as xmltools
from core.io import fatal_inp_error, input_errors
from drivers.driver import isdriver, create_driver
from utils.namespace import Namespace
from utils.pprepro import find_and_make_subs, find_and_fill_includes
from utils.fcnbldr import build_lambda, build_interpolating_function
from utils.opthold import OptionHolder, OptionHolderError as OptionHolderError
from utils.mtldb import read_material_params_from_db
from materials.material import get_material_from_db


S_PHYSICS = "Physics"
S_PERMUTATION = "Permutation"
S_OPT = "Optimization"

S_AUX_FILE = "AuxiliaryFile"
S_METHOD = "Method"
S_PARAMS = "Parameters"
S_OBJ_FCN = "ObjectiveFunction"
S_MITER = "Maximum Iterations"
S_TOL = "Tolerance"
S_DISP = "Disp"
S_TTERM = "TerminationTime"
S_MATERIAL = "Material"
S_EXTRACT = "Extract"
S_PATH = "Path"
S_SURFACE = "Surface"
S_DRIVER = "driver"


class UserInputError(Exception):
    def __init__(self, message):
        sys.stderr.write("*** {0} ***\n".format(message))
        sys.exit(2)


def parse_input(filepath):
    """Parse input file contents

    Parameters
    ----------
    user_input : str
    The user input

    """
    # find all "Include" files, and preprocess the input
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
        raise UserInputError("Expected Root Element 'GMDSpec'")

    # ------------------------------------------ get and parse blocks --- #
    gmdblks = {}
    #           (blkname, required, remove)
    rootblks = ((S_OPT, 0, 1), (S_PERMUTATION, 0, 1), (S_PHYSICS, 1, 0))

    # find all functions first
    functions = pFunctions(gmdspec.getElementsByTagName("Function"))

    for (rootblk, reqd, rem) in rootblks:
        rootlmns = gmdspec.getElementsByTagName(rootblk)
        if not rootlmns:
            if reqd:
                fatal_inp_error("GMDSpec: {0}: block missing".format(rootblk))
            continue

        if len(rootlmns) > 1:
            fatal_inp_error("Expected 1 {0} block, got {1}".format(
                rootblk, len(rootlmns)))
            continue

        rootlmn = rootlmns[0]
        gmdblks[rootblk] = rootlmn

        if rem:
            p = rootlmn.parentNode
            p.removeChild(rootlmn)

    if input_errors():
        raise UserInputError("stopping due to previous Errors")

    if S_OPT in gmdblks and S_PERMUTATION in gmdblks:
        raise UserInputError("Incompatible blocks: [Optimzation, Permutation]")

    if S_OPT in gmdblks:
        ns = optimization_namespace(gmdblks[S_OPT], gmdspec.toxml())

    elif S_PERMUTATION in gmdblks:
        ns = permutation_namespace(gmdblks[S_PERMUTATION], gmdspec.toxml())

    else:
        ns = physics_namespace(gmdblks[S_PHYSICS], functions)

    if input_errors():
        raise UserInputError("stopping due to previous Errors")

    return ns


# ------------------------------------------------- Parsing functions --- #
# Each XML block is parsed with a corresponding function 'pBlockName'
# where BlockName is the name of the block as entered by the user in the
# input file
def pOptimization(optlmn):
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
        try:
            options.setopt(*xmltools.get_name_value(optlmn.attributes.item(i)))
        except OptionHolderError, e:
            fatal_inp_error(e.message)

    # objective function
    objfcn = optlmn.getElementsByTagName(S_OBJ_FCN)
    if not objfcn:
        fatal_inp_error("{0} not found".format(S_OBJ_FCN))
    elif len(objfcn) > 1:
        fatal_inp_error("Only one {0} tag supported".format(S_OBJ_FCN))
    else:
        objfcn = objfcn[0]

    objfile = objfcn.getAttribute("href")
    if not objfile:
        fatal_inp_error("Expected href attribute to {0}".format(S_OBJ_FCN))
    elif not os.path.isfile(objfile):
        fatal_inp_error("{0}: no such file".format(objfile))
    else:
        objfile = os.path.realpath(objfile)

    # auxiliary files
    auxfiles = []
    for item in optlmn.getElementsByTagName(S_AUX_FILE):
        auxfile = item.getAttribute("href")
        if not auxfile:
            fatal_inp_error("Expected href attribute to {0}".format(S_AUX_FILE))
        elif not os.path.isfile(auxfile):
            fatal_inp_error("{0}: no such file".format(auxfile))
        else:
            auxfiles.append(os.path.realpath(auxfile))

    # read in optimization parameters
    p = []
    for items in optlmn.getElementsByTagName("Optimize"):
        var = str(items.attributes.get("var").value)

        ivalue = items.attributes.get("initial_value")
        if not ivalue:
            fatal_inp_error("{0}: no initial value given".format(var))
            continue
        ivalue = float(ivalue.value)

        bounds = items.attributes.get("bounds")
        if not bounds:
            bounds = [None, None]
        else:
            bounds = xmltools.str2list(bounds.value, dtype=float)
        if len(bounds) != 2:
            fatal_inp_error("{0}: incorrect bounds, must give upper "
                            "and lower bound".format(var))
            continue
        p.append([var, ivalue, bounds])

    odict[S_METHOD] = options.getopt("method")
    odict[S_MITER] = options.getopt("maxiter")
    odict[S_TOL] = options.getopt("tolerance")
    odict[S_DISP] = options.getopt("disp")

    odict[S_PARAMS] = p
    odict[S_AUX_FILE] = auxfiles
    odict[S_OBJ_FCN] = objfile

    return odict


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
        try:
            options.setopt(*xmltools.get_name_value(permlmn.attributes.item(i)))
        except OptionHolderError, e:
            fatal_inp_error(e.message)

    rstate = np.random.RandomState(options.getopt("seed"))
    gdict = {"__builtins__": None}
    N_default = 10
    safe = {"range": lambda a, b, N=N_default: np.linspace(a, b, N),
            "list": lambda a: np.array(a),
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
            fatal_inp_error("{0}: invalid expression".format(values))
            continue

    pdict[S_PARAMS] = p
    pdict[S_METHOD] = options.getopt("method")

    return pdict


def pExtract(extlmns):
    extlmn = extlmns[-1]
    options = OptionHolder()
    options.addopt("format", "ascii", dtype=str, choices=("ascii", "mathematica"))
    options.addopt("step", 1, dtype=int)
    options.addopt("ffmt", ".18f", dtype=str)

    # Get control terms
    for i in range(extlmn.attributes.length):
        try:
            options.setopt(*xmltools.get_name_value(extlmn.attributes.item(i)))
        except OptionHolderError, e:
            fatal_inp_error(e.message)

    # get requested variables to extract
    variables = []
    for varlmn in  extlmn.getElementsByTagName("Variables"):
        data = varlmn.firstChild.data.split("\n")
        variables.extend([xmltools.stringify(x, "upper")
                          for sub in data for x in sub.split()])
    if "ALL" in variables:
        variables = "ALL"

    # get Paths to extract -> further parsing is handled by drivers that
    # support extracting paths
    paths = extlmn.getElementsByTagName("Path")

    return (options.getopt("format"), options.getopt("step"),
            options.getopt("ffmt"), variables, paths)


def physics_namespace(physlmn, functions):

    simblk = pPhysics(physlmn, functions)
    if input_errors():
        raise UserInputError("stopping due to previous errors")

    # set up the namespace to return
    ns = Namespace()

    ns.stype = S_PHYSICS

    ns.ttermination = simblk.get(S_TTERM)

    ns.mtlmdl = simblk[S_MATERIAL][0]
    ns.mtlprops = simblk[S_MATERIAL][1]
    ns.density = simblk[S_MATERIAL][2]

    ns.extract = simblk.get(S_EXTRACT)

    ns.driver = simblk[S_DRIVER]

    return ns


def optimization_namespace(optlmn, basexml):
    optblk = pOptimization(optlmn)
    # set up the namespace to return
    ns = Namespace()
    ns.stype = S_OPT
    ns.method = optblk[S_METHOD]
    ns.parameters = optblk[S_PARAMS]
    ns.auxiliary_files = optblk[S_AUX_FILE]
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


def pPhysics(physlmn, functions):
    """Parse the physics tag

    """
    simblk = {}

    # Get the driver first
    driver = physlmn.getAttribute("driver")
    driver = "solid" if not driver else driver
    if not isdriver(driver):
        fatal_inp_error("{0}: unrecognized driver".format(driver))
        return

    driver = create_driver(driver)
    simblk[S_DRIVER] = driver

    # parse the sub blocks
    subblks = ((S_MATERIAL, 1), (S_EXTRACT, 0), (S_TTERM, 0))
    for (subblk, reqd) in subblks:
        sublmns = physlmn.getElementsByTagName(subblk)
        if not sublmns:
            if reqd:
                fatal_inp_error("Physics: {0}: block missing".format(subblk))
            continue
        parsefcn = getattr(sys.modules[__name__],
                           "p{0}".format(sublmns[0].nodeName))
        simblk[subblk] = parsefcn(sublmns)
        for sublmn in sublmns:
            p = sublmn.parentNode
            p.removeChild(sublmn)

    # Finally, parse the paths and surfaces
    pathlmns = physlmn.getElementsByTagName(S_PATH)
    surflmns = physlmn.getElementsByTagName(S_SURFACE)
    if not pathlmns and not surflmns:
        fatal_inp_error("Physics: must specify at least one surface or path")
        return

    driver.parse_and_register_paths_and_surfaces(pathlmns, surflmns, functions)

    return simblk


def pTerminationTime(ttermlmns):
    tlmn = physlmn.getElementsByTagName(S_TTERM)
    if tlmn:
        return float(tlmn[0].firstChild.data)
    return None


def pMaterial(mtllmns):
    """Parse the material block

    """
    mtllmn = mtllmns[-1]
    model = mtllmn.attributes.get("model")
    if model is None:
        fatal_inp_error("Material: model not found")
        return
    model = str(model.value.lower())

    density = mtllmn.attributes.get("density")
    if density is None:
        density = 1.
    else:
        density = float(density.value)

    mtlmdl = get_material_from_db(model)
    if mtlmdl is None:
        fatal_inp_error("{0}: material not in database".format(model))
        return

    # mtlmdl.parameters is a comma separated list of parameters
    pdict = dict([(xmltools.stringify(n, "lower"), i)
                  for i, n in enumerate(mtlmdl.parameters.split(","))])
    params = parse_mtl_params(mtllmn, pdict, model)

    return model, params, density


def parse_mtl_params(mtllmn, pdict, model):
    # create a mapping of (name, value) pairs
    param_map = {}
    for node in mtllmn.childNodes:
        if node.nodeType != node.ELEMENT_NODE:
            continue
        name = node.nodeName
        if name.lower() == "matlabel":
            mat = node.getAttribute("material")
            if not mat:
                fatal_inp_error("Matlabel: expected material attribute")
                continue
            dbfile = node.getAttribute("db")
            if not dbfile:
                dbfile = F_MTL_PARAM_DB
            if not os.path.isfile(dbfile):
                if not os.path.isfile(os.path.join(cfg.I, dbfile)):
                    fatal_inp_error("{0}: no such file".format(dbfile))
                    continue
                dbfile = os.path.join(cfg.I, dbfile)
            mtl_db_params = read_material_params_from_db(mat, model, dbfile)
            if mtl_db_params is None:
                fatal_inp_error("Material: error reading parameters for "
                                "{0} from database".format(mat))
                continue
            param_map.update(mtl_db_params)

        else:
            val = node.firstChild.data.strip()
            param_map[name] = val

    # put the parameters in an array
    params = np.zeros(len(pdict))
    for (name, val) in param_map.items():
        idx = pdict.get(name.lower())
        if idx is None:
            fatal_inp_error("Material: {0}: invalid parameter".format(name))
            continue
        try:
            val = float(val)
        except ValueError:
            fatal_inp_error("Material: {0}: invalid value "
                            "{1}".format(name, val))
            continue
        params[idx] = val

    return params


def pFunctions(element_list):
    """Parse the functions block

    """
    __ae__ = "ANALYTIC EXPRESSION"
    __pwl__ = "PIECEWISE LINEAR"
    zero_fcn_id = 0
    const_fcn_id = 1
    functions = {zero_fcn_id: lambda x: 0., const_fcn_id: lambda x: 1.}
    if not element_list:
        return functions

    for function in element_list:

        fid = function.attributes.get("id")
        if fid is None:
            fatal_inp_error("Function: id not found")
            continue
        fid = int(fid.value)
        if fid in (zero_fcn_id, const_fcn_id):
            fatal_inp_error("Function id {0} is reserved".format(fid))
            continue
        if fid in functions:
            fatal_inp_error("{0}: duplicate function definition".format(fid))
            continue

        ftype = function.attributes.get("type")
        if ftype is None:
            fatal_inp_error("Functions.Function: type not found")
            continue
        ftype = " ".join(ftype.value.split()).upper()
        if ftype not in (__ae__, __pwl__):
            fatal_inp_error("{0}: invalid function type".format(ftype))
            continue

        href = function.getAttribute("href")
        if href:
            if ftype == __ae__:
                fatal_inp_error("function file support only for piecewise linear")
                continue
            if not os.path.isfile(href):
                if not os.path.isfile(os.path.join(cfg.I, href)):
                    fatal_inp_error("{0}: no such file".format(href))
                    continue
                href = os.path.join(cfg.I, href)
            expr = open(href, "r").read()

        else:
            expr = function.firstChild.data.strip()

        if ftype == __ae__:
            var = function.getAttribute("var")
            if not var:
                var = "x"
            func, err = build_lambda(expr, var=var, disp=1)
            if err:
                fatal_inp_error("{0}: in analytic expression in "
                                "function {1}".format(err, fid))
                continue

        elif ftype == __pwl__:
            # parse the table in expr
            cols = function.getAttribute("cols")
            if not cols:
                cols = np.arange(2)
            else:
                cols = np.array(xmltools.str2list(cols, dtype=int)) - 1
                if len(cols) != 2:
                    fatal_inp_error("len(cols) != 2")
                    continue

            table = []
            nc = 0
            for line in expr.split("\n"):
                line = line.split("#", 1)[0].split()
                if not line:
                    continue
                line = [float(x) for x in line]
                if not nc: nc = len(line)
                if len(line) != nc:
                    fatal_inp_error("Inconsistent table data")
                    continue
                if len(line) < np.amax(cols):
                    fatal_inp_error("Note enought columns in table data")
                    continue
                table.append(line)
            table = np.array(table)[cols]

            func, err = build_interpolating_function(table, disp=1)
            if err:
                fatal_inp_error("{0}: in piecwise linear table in "
                                "function {1}".format(err, fid))
                continue

        functions[fid] = func
        continue

    return functions
