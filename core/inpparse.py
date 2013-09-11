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
import utils.pprepro as pp
import utils.tensor as tensor
import utils.xmltools as xmltools
from core.io import fatal_inp_error, input_errors
from core.respfcn import check_response_function_element
from drivers.driver import isdriver, create_driver
from utils.namespace import Namespace
from utils.fcnbldr import build_lambda, build_interpolating_function
from utils.opthold import OptionHolder, OptionHolderError as OptionHolderError
from utils.mtldb import read_material_params_from_db
from materials.material import get_material_from_db


S_PHYSICS = "Physics"
S_TTERM = "termination_time"
S_DRIVER = "driver"
T_DRIVER_TYPES = ("solid", "eos")
S_MATERIAL = "Material"
S_MODEL = "model"
S_RHO = "density"
S_PARAMS = "Parameters"

S_EXTRACT = "Extract"
S_EXFORMAT = "format"
T_EXFORMATS = ("ascii", "mathematica")
S_STEP = "step"
S_FFMT = "ffmt"

S_PATH = "Path"
S_SURFACE = "Surface"

S_HREF = "href"

S_PERMUTATION = "Permutation"
S_PERMUTATE = "Permutate"
T_PERM_METHODS = ("zip", "combine", "shotgun")

S_OPT = "Optimization"
T_OPT_METHODS = ("simplex", "cobyla", "powell")
S_MITER = "maxiter"
S_TOL = "tolerance"
S_DISP = "disp"
S_RESP_FCN = "ResponseFunction"
S_RESP_DESC = "descriptor"
S_AUX_FILE = "AuxiliaryFile"
S_METHOD = "method"
S_CORR = "correlation"
S_SEED = "seed"


class UserInputError(Exception):
    def __init__(self, message):
        sys.stderr.write("*** {0} ***\n".format(message))
        raise SystemExit(2)


def parse_input(filepath):
    """Parse input file contents

    Parameters
    ----------
    user_input : str
    The user input

    """
    # remove the shebang, if any
    lines = open(filepath, "r").readlines()
    if re.search("#!/", lines[0]):
        lines = lines[1:]

    # find all "Include" files, and preprocess the input
    user_input = pp.find_and_fill_includes("\n".join(lines))
    user_input, nsubs = pp.find_and_make_subs(user_input, disp=1)
    if nsubs:
        with open(filepath + ".preprocessed", "w") as fobj:
            fobj.write(user_input)

    # Parse the xml document, the root element is always "GMDSpec"
    doc = xdom.parseString(user_input)
    try:
        gmdspec = doc.getElementsByTagName("GMDSpec")[0]
    except IndexError:
        raise UserInputError("expected Root Element 'GMDSpec'")

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
            fatal_inp_error("expected 1 {0} block, got {1}".format(
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
    options.addopt(S_METHOD, T_OPT_METHODS[0], dtype=str, choices=T_OPT_METHODS)
    options.addopt(S_MITER, 25, dtype=int)
    options.addopt(S_TOL, 1.e-6, dtype=float)
    options.addopt(S_DISP, 0, dtype=int)

    # Get control terms
    for i in range(optlmn.attributes.length):
        try:
            options.setopt(*xmltools.get_name_value(optlmn.attributes.item(i)))
        except OptionHolderError, e:
            fatal_inp_error(e.message)

    # response function
    respfcn = optlmn.getElementsByTagName(S_RESP_FCN)
    if not respfcn:
        fatal_inp_error("{0} not found".format(S_RESP_FCN))
    elif len(respfcn) > 1:
        fatal_inp_error("Only one {0} tag supported".format(S_RESP_FCN))
    else:
        respfcn = respfcn[0]

    response_fcn = check_response_function_element(respfcn)

    # auxiliary files
    auxfiles = []
    for item in optlmn.getElementsByTagName(S_AUX_FILE):
        auxfile = item.getAttribute(S_HREF)
        if not auxfile:
            fatal_inp_error("pOptimization: expected {0} attribute to {1}".format(
                S_HREF, S_AUX_FILE))
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

    odict[S_METHOD] = options.getopt(S_METHOD)
    odict[S_MITER] = options.getopt(S_MITER)
    odict[S_TOL] = options.getopt(S_TOL)
    odict[S_DISP] = options.getopt(S_DISP)

    odict[S_PARAMS] = p
    odict[S_AUX_FILE] = auxfiles
    odict.update(response_fcn)

    return odict


def pPermutation(permlmn):
    """Parse the permutation block

    """
    pdict = {}

    # Set up options for permutation
    options = OptionHolder()
    options.addopt(S_METHOD, T_PERM_METHODS[0], dtype=str, choices=T_PERM_METHODS)
    options.addopt(S_SEED, None, dtype=int)
    options.addopt(S_CORR, "none", dtype=str)

    # Get control terms
    for i in range(permlmn.attributes.length):
        try:
            options.setopt(*xmltools.get_name_value(permlmn.attributes.item(i)))
        except OptionHolderError, e:
            fatal_inp_error(e.message)

    seed = options.getopt("seed")
    if seed is not None:
        pp.set_random_seed(seed)
    N_default = 10
    pp.update_safe({"range": lambda a, b, N=N_default: np.linspace(a, b, N),
                    "list": lambda a: np.array(a),
                    "weibull": lambda a, b, N=N_default: a * pp.RAND.weibull(b, N),
                    "uniform": lambda a, b, N=N_default: pp.RAND.uniform(a, b, N),
                    "normal": lambda a, b, N=N_default: pp.RAND.normal(a, b, N),
                    "percentage": lambda a, b, N=N_default: (
                        np.linspace(a-(b/100.)*a, a+(b/100.)* a, N))})

    # response function
    respfcn = permlmn.getElementsByTagName(S_RESP_FCN)
    if len(respfcn) > 1:
        fatal_inp_error("Only one {0} tag supported".format(S_RESP_FCN))
    elif respfcn:
        respfcn = respfcn[0]
    response_fcn = check_response_function_element(respfcn)

    # read in permutated values
    p = []
    for items in permlmn.getElementsByTagName(S_PERMUTATE):
        var = str(items.attributes.get("var").value)
        values = str(items.attributes.get("values").value)
        s = re.search(r"(?P<meth>\w+)\s*\(", values)
        if s: meth = s.group("meth")
        else: meth = ""
        if options.getopt(S_METHOD) == "shotgun":
            if meth != "range":
                fatal_inp_error("shotgun: expected values=range(...")
                continue
            values = re.sub(meth, "uniform", values)
        try:
            p.append([var, eval(values, pp.GDICT, pp.SAFE)])
        except:
            fatal_inp_error("{0}: invalid expression".format(values))
            continue

    if options.getopt(S_CORR) is not None:
        corr = xmltools.str2list(options.getopt(S_CORR), dtype=str)
        for (i, c) in enumerate(corr):
            if c.lower() not in ("table", "plot", "none"):
                fatal_inp_error("{0}: unrecognized correlation option".format(c))
            corr[i] = c.lower()
        if "none" in corr:
            corr = None
        options.setopt(S_CORR, corr)

    pdict[S_CORR] = options.getopt(S_CORR)
    pdict[S_PARAMS] = p
    pdict[S_METHOD] = options.getopt(S_METHOD)

    pdict.update(response_fcn)

    return pdict


def pExtract(extlmns):
    extlmn = extlmns[-1]
    options = OptionHolder()
    options.addopt(S_EXFORMAT, T_EXFORMATS[0], dtype=str, choices=T_EXFORMATS)
    options.addopt(S_STEP, 1, dtype=int)
    options.addopt(S_FFMT, ".18f", dtype=str)

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

    return (options.getopt(S_EXFORMAT), options.getopt(S_STEP),
            options.getopt(S_FFMT), variables, paths)


def physics_namespace(physlmn, functions):

    simblk = pPhysics(physlmn, functions)
    if input_errors():
        raise UserInputError("stopping due to previous errors")

    # set up the namespace to return
    ns = Namespace()

    ns.stype = S_PHYSICS

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
    ns.response_function = optblk[S_HREF]
    ns.response_descriptor = optblk[S_RESP_DESC]
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
    ns.response_function = permblk[S_HREF]
    ns.response_descriptor = permblk[S_RESP_DESC]
    ns.correlation = permblk[S_CORR]
    ns.basexml = basexml
    return ns


def pPhysics(physlmn, functions):
    """Parse the physics tag

    """
    simblk = {}

    # Get the driver first
    options = OptionHolder()
    options.addopt(S_DRIVER, T_DRIVER_TYPES[0], dtype=str, choices=T_DRIVER_TYPES)
    options.addopt(S_TTERM, None, dtype=float)
    for i in range(physlmn.attributes.length):
        try:
            options.setopt(*xmltools.get_name_value(physlmn.attributes.item(i)))
        except OptionHolderError, e:
            fatal_inp_error(e.message)
    driver = options.getopt(S_DRIVER)
    if not isdriver(driver):
        fatal_inp_error("{0}: unrecognized driver".format(driver))
        return

    driver = create_driver(driver)
    simblk[S_DRIVER] = driver

    # parse the sub blocks
    subblks = ((S_MATERIAL, 1), (S_EXTRACT, 0))
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

    driver.parse_and_register_paths_and_surfaces(pathlmns, surflmns, functions,
                                                 options.getopt(S_TTERM))

    return simblk


def pMaterial(mtllmns):
    """Parse the material block

    """
    mtllmn = mtllmns[-1]
    model = mtllmn.attributes.get(S_MODEL)
    if model is None:
        fatal_inp_error("expeced 'model' Material attribute")
        return
    model = str(model.value.lower())

    density = mtllmn.attributes.get(S_RHO)
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
            dbfile = node.getAttribute(S_HREF)
            if not dbfile:
                dbfile = F_MTL_PARAM_DB

            found = False
            if not os.path.isfile(dbfile):
                if os.path.isfile(os.path.join(cfg.I, dbfile)):
                    found = True
                    dbfile = os.path.join(cfg.I, dbfile)
                elif os.path.isfile(os.path.join(cfg.MTL_LIB, dbfile)):
                    found = True
                    dbfile = os.path.join(cfg.MTL_LIB, dbfile)
                if not found:
                    fatal_inp_error("{0}: no such file".format(dbfile))
                    continue
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

        href = function.getAttribute(S_HREF)
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
