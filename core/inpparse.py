import os
import re
import sys
import numpy as np
from xml.etree.ElementTree import iterparse
import xml.dom.minidom as xdom

if __name__ == "__main__":
    _D = os.path.dirname(os.path.realpath(__file__))
    sys.path.insert(0, os.path.dirname(_D))
from __config__ import cfg, F_MTL_PARAM_DB
import utils.pprepro as pp
from utils.mtldb import read_material_params_from_db
from utils.fcnbldr import build_lambda, build_interpolating_function
from utils.xmltools import stringify
from drivers.driver import isdriver, getdrvcls
from materials.material import get_material_from_db

_D = os.path.dirname(os.path.realpath(__file__))
NOT_SPECIFIED = -64023
F_INP_DFNS = os.path.join(_D, "inpdfn.xml")
assert os.path.exists(F_INP_DFNS)
ZERO_FCN_ID = 0
CONST_FCN_ID = 1


class UserInputError(Exception):
    def __init__(self, message):
        sys.stderr.write("*** user input error: {0}\n".format(message))
        raise SystemExit(2)


class ElementError(Exception):
    pass


class Element(object):
    def __init__(self, **kwargs):
        valid = ("name", "n", "content", "conflict")
        for (key, val) in kwargs.items():
            if key not in valid:
                raise ElementError("{0}: invalid attribute key".format(key))
            self.__dict__.update({key: val})
        if not self.__dict__.get("name"):
            raise ElementError("expected name kwarg")
        self.conflict = aslist(self.__dict__.get("conflict"))

        # number of elements
        n = self.__dict__.pop("n", "*")
        if n.isdigit():
            n = int(n)
            self.more_than_one = n > 1
        else:
            if n not in ("?", "+", "*"):
                raise ElementError("expected kw n to be int or one of ?, +, *")
            self.more_than_one = n in ("+", "*")
        self.__dict__["n"] = n

        # content to read in
        self.content = [x.strip() for x in
                        self.__dict__.pop("content", "").split("then") if x]
        for item in self.content:
            if item not in ("href", "text", "children"):
                raise ElementError("{0}: bad content type".format(item))

        self.attribs = {}
        self.elmnts = {}

    def copy(self):
        b = object.__new__(Element)
        b.__dict__ = self.__dict__.copy()
        return b

    def regattrib(self, attrib):
        """Register an Attribute instance to the Element

        """
        self.attribs[attrib.name] = [attrib, attrib.default]

    def setattrib(self, attrib, value):
        """Set an attribute value

        """
        myattrib = self.attribs.get(attrib)
        if myattrib is None:
            raise UserInputError("{0}: {1}: unexpected "
                                 "attribute".format(self.name, attrib))
        err, value = myattrib[0].test(value)
        if err != 0:
            fatal_inp_error("{0}: {1}".format(self.name, err))
        else:
            value = myattrib[0].dtype(value)
            l = self.getattrib("len")
            if l is not None:
                if len(val) != l:
                    err = "expected {0} to be length {1}".format(attrib, l)
                    fatal_inp_error("{0}: {1}".format(self.name, err))
            self.attribs[attrib][1] = myattrib[0].dtype(value)

    def getattrib(self, attrib):
        theattrib = self.attribs.get(attrib)
        if theattrib is None:
            return
        return theattrib[1]

    def setelmnt(self, elmnt):
        self.elmnts[elmnt.name] = elmnt

    def getelmnt(self, elmnt, default=None):
        return self.elmnts.get(elmnt, default)

    def elements(self):
        return self.elmnts.values()

    def attributes(self):
        return dict((attrib[0].name, attrib[0].dtype(attrib[1]))
                    for _, attrib in self.attribs.items())

    def getcontent(self, dom):
        for item in self.content:
            if item == "children":
                content = []
                for node in dom.childNodes:
                    if node.nodeName.lower() == "matlabel":
                        p = read_matlabel(node, self.getattrib("model"))
                        if p:
                            content.extend(p)
                        continue
                    if node.nodeType != node.ELEMENT_NODE:
                        continue
                    name = node.nodeName.strip()
                    val = node.firstChild.data.strip()
                    content.append("{0} = {1}".format(name, val))
                if content:
                    return content

            elif item == "href":
                # read from file, returning list
                if self.getattrib("href"):
                    return [x.strip() for x in
                            open(self.getattrib("href"), "r").readlines()]

            else:
                content = []
                for node in dom.childNodes:
                    if node.nodeType == node.COMMENT_NODE:
                        continue
                    content.extend([" ".join(s.split()) for s in
                                    node.data.split("\n") if s.strip()])
                return content


        return


class Attribute(object):
    def __init__(self, **kwargs):
        valid = ("name", "type", "default", "choices", "test", "len")
        for (key, val) in kwargs.items():
            if key not in valid:
                raise AttributeError("{0}: invalid attribute key".format(key))
            self.__dict__.update({key: val})
        if not self.__dict__.get("name"):
            raise AttributeError("expected name kwarg")
        self.__dict__.setdefault("default", NOT_SPECIFIED)
        if self.default == "none": self.default = None
        self.choices = aslist(self.__dict__.get("choices"))
        self.testmeth = self.__dict__.pop("test", "always")
        self.typeconv = self.__dict__.pop("type", "string")

    def __repr__(self):
        string = ", ".join("{0}={1}".format(k, repr(v)) for (k, v) in
                           self.__dict__.items())
        return "Attribute({0})".format(string)

    def items(self):
        return self.__dict__.items()

    def test(self, a):
        """Test that input a is valid for this attribute

        Returns
        -------
        0 : pass
        errmsg : fail

        """
        if self.choices and a not in self.choices:
            choices = ", ".join("'{0}'".format(c) for c in self.choices)
            errmsg = "{0}: invalid choice: '{1}' (choose from {2})".format(
                self.name, a, choices)
            return errmsg, a
        passed = eval("{0}('{1}')".format(self.testmeth, a),
                      {"__builtins__": None}, TESTS)
        try:
            passed, a = passed
        except:
            pass

        if passed is True:
            return 0, a
        if passed:
            return passed, a
        return "{0}: {1}: invalid value".format(self.name, a), a

    def dtype(self, a):
        """Convert a to proper data type

        """
        try:
            return eval("{0}('{1}')".format(self.typeconv, a),
                        {"__builtins__": None}, DTYPES)
        except:
            return NOT_SPECIFIED


def defntree(filepath=None):
    """Parse the input definition template

    Returns
    -------
    Tree : list of Element instances

    """
    if filepath is None:
        filepath = F_INP_DFNS
    stack = []
    for (event, node) in iterparse(filepath, ["start", "end"]):
        if event == "end" and node.tag == "Element":
            # pop last element on stack to its parent
            stack[-2].setelmnt(stack.pop())

        if event == 'start':
            if node.tag == "Root":
                # The root element
                stack.append(Element(**node.attrib))

            if node.tag == "Element":
                # stack an element
                stack.append(Element(**node.attrib))

            elif node.tag == "Attribute":
                # set attribute to its parent element
                stack[-1].regattrib(Attribute(**node.attrib))

        continue

    Tree = stack.pop()
    if stack:
        raise SystemExit("*** error - inconsistent input stack")
    return Tree


# recognized types
def boolean(a):
    if re.search(r"no|0|none|false", a.lower()): return False
    return True
def real(a): return float(a)
def integer(a): return int(a)
def string(a): return str(a)
def indices(a):
    a = aslist(a, dtype=int)
    return np.array(a, dtype=int) - 1
def array(a):
    a = aslist(a, dtype=float)
    return np.array(a, dtype=np.float64)
DTYPES = {"boolean": boolean, "real": real, "integer": integer, "string": string,
          "array": array, "indices": indices}

# tests
def always(a): return True
def ispositive(a): return float(a) > 0.
def isnonnegative(a): return float(a) >= 0.
def reservedfid(a): return int(a) in (ZERO_FCN_ID, CONST_FCN_ID)
def isfile(a):
    if not os.path.isfile(a):
        if not os.path.isfile(os.path.join(cfg.I, a)):
            return "{0}: no such file".format(a)
        a = os.path.join(cfg.I, href)
    return True, a
def isint(a): return a.isdigit()
def isnumber(a):
    try:
        float(a)
        return True
    except ValueError:
        return False
TESTS = {"always": always, "ispositive": ispositive,
         "isnonnegative": isnonnegative, "reservedfid": reservedfid,
         "isfile": isfile, "isint": isint, "isnumber": isnumber}

# conversions
def aslist(string, dtype=string):
    if string is None:
        return []
    string = " ".join(string.split())
    string = re.sub(r"^[\(\[\{]", " ", string.strip())
    string = re.sub(r"[\)\]\}]$", " ", string.strip())
    string = re.sub(r"[, ]", " ", string)
    return [dtype(x) for x in string.split()]

def gmdfcn(a):
    pass


def inp2dict(parent, dom):

    child_elements = {}
    for child in parent.elements():

        # get els only one deep, except for the special case 'Function'
        els = dom.getElementsByTagName(child.name)
        if child.name != "Function":
            els = [el for el in els if el.parentNode.nodeName == parent.name]

        # check for correct number of elements
        if child.n == "*":
            pass

        elif child.n == "?":
            if len(els) not in (0, 1):
                raise UserInputError("{0}: expected 0 or 1 {1} "
                                     "elements".format(parent.name, child.name))
        elif child.n == "+":
            if len(els) < 1:
                raise UserInputError("{0}: expected at least 1 {1} "
                                     "element".format(parent.name, child.name))
        elif len(els) != child.n:
            raise UserInputError("{0}: expected exactly {1} {2} element(s)"
                                 .format(parent.name, child.n, child.name))

        # check for conflicts
        for name in child_elements:
            if child_elements[name] and els and name in child.conflict:
                raise UserInputError("conflicting elements: "
                                     "{0}/{1}".format(child.name, name))

        # Set attributes and children
        if not els:
            child_element = None if not child.more_than_one else []
            child_elements[child.name] = child_element
            continue

        child = [child.copy() for i in range(len(els))]
        name = child[0].name
        child_elements[name] = []
        for (i, el) in enumerate(els):

            child_elements[name].append({})

            for a in range(el.attributes.length):
                n, v = nandv(el.attributes.item(a))
                child[i].setattrib(n, v)

            child_elements[name][-1].update(child[i].attributes())

            content = child[i].getcontent(el)
            if content is not None:
                child_elements[name][-1].update({"Content": content})

            grandchildren = inp2dict(child[i], el)
            if grandchildren:
                child_elements[name][-1].update({"Elements": grandchildren})

        if not child[0].more_than_one:
            child_elements[name] = child_elements[name][0]

    return child_elements


def nandv(item):
    name = item.name.encode("utf-8").strip()
    value = " ".join(item.value.split())
    return name, value


INP_ERRORS = 0
def fatal_inp_error(message):
    global INP_ERRORS
    INP_ERRORS += 1
    sys.stderr.write("*** error: {0}\n".format(message))
    if INP_ERRORS > 5:
        raise SystemExit("*** error: maximum number of input errors exceeded")


def get_content_from_children(lmn):
    for node in lmn.childNodes:
        if node.nodeType != node.ELEMENT_NODE:
            continue
        name = node.nodeName
        val = node.firstChild.data.strip()
        param_map[name] = val


def read_matlabel(dom, model):
    """Read in the material parameters from a Matlabel

    """
    dbfile = F_MTL_PARAM_DB
    material = None
    for a in range(dom.attributes.length):
        n, v = nandv(dom.attributes.item(a))
        if n == "href":
            dbfile = v
        elif n == "material":
            material = v
        else:
            fatal_inp_error("Matlabel: {0}: unrecognized attribute".format(n))
            return
    if material is None:
        fatal_inp_error("Matlabel: expected material attribute")
        return

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
            return

    mtl_db_params = read_material_params_from_db(material, model, dbfile)
    if mtl_db_params is None:
        fatal_inp_error("Material: error reading parameters for "
                        "{0} from database".format(material))
        return
    return ["{0} = {1}".format(k, v) for k, v in mtl_db_params.items()]

def parse_input(filepath):
    """Parse the input

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

    Tree = defntree()
    dom = xdom.parseString(user_input)
    root = dom.getElementsByTagName(Tree.name)
    if len(root) != 1:
        raise UserInputError("expected root element {0}".format(Tree.name))
    els = inp2dict(Tree, root[0])

    # get functions first
    functions = pFunction(els.pop("Function"))

    prmargs = pPermutation(els.pop("Permutation"), functions)
    if prmargs:
        return prmargs
    optargs = pOptimization(els.pop("Optimization"), functions)
    if optargs:
        return optargs
    return pPhysics(els.pop("Physics"), functions)


def pFunction(flist):
    """Parse the functions block

    """
    functions = {ZERO_FCN_ID: lambda x: 0., CONST_FCN_ID: lambda x: 1.}
    if not flist:
        return functions

    for function in flist:
        fid = function["id"]
        if fid == NOT_SPECIFIED:
            fatal_inp_error("Function: id not found")
            continue

        if fid in functions:
            fatal_inp_error("{0}: duplicate function definition".format(fid))
            continue

        if function["type"] == NOT_SPECIFIED:
            fatal_inp_error("Functions.Function: type not found")
            continue

        expr = function["Content"]
        if function["type"] == "analytic_expression":
            var = function["var"]
            func, err = build_lambda(function["Content"][-1], var=var, disp=1)
            if err:
                fatal_inp_error("{0}: in analytic expression in "
                                "function {1}".format(err, fid))
                continue

        elif function["type"] == "piecewise_linear":
            # parse the table in expr
            if len(function["cols"]) != 2:
                fatal_inp_error("len(cols) != 2")
                continue

            table = []
            nc = 0
            for line in expr:
                line = line.split("#", 1)[0].split()
                if not line:
                    continue
                line = [float(x) for x in line]
                if not nc: nc = len(line)
                if len(line) != nc:
                    fatal_inp_error("Inconsistent table data")
                    continue
                if len(line) < np.amax(function["cols"]):
                    fatal_inp_error("Note enought columns in table data")
                    continue
                table.append(line)
            table = np.array(table)[function["cols"]]
            func, err = build_interpolating_function(table, disp=1)
            if err:
                fatal_inp_error("{0}: in piecwise linear table in "
                                "function {1}".format(err, fid))
                continue

        functions[fid] = func
        continue

    return functions


def pPhysics(physdict, functions):
    """Parse the physics tag

    """
    material = pMaterial(physdict["Elements"].pop("Material"))
    dcls = getdrvcls(physdict["driver"])
    driver = [physdict["driver"]]
    driver.extend(dcls.format_path(physdict["Elements"]["Path"], None, functions,
                                   physdict["termination_time"]))
    extract = pExtract(physdict["Elements"].pop("Extract"), dcls)

    # Return the physics dictionary
    return ("Physics", driver, material, extract)


def pMaterial(mtldict):
    """Parse the material block

    """
    model = mtldict["model"]
    if model == NOT_SPECIFIED:
        fatal_inp_error("expeced 'model' Material attribute")
        return

    mtlmdl = get_material_from_db(model)
    if mtlmdl is None:
        fatal_inp_error("{0}: material not in database".format(model))
        return

    # mtlmdl.parameters is a comma separated list of parameters
    pdict = dict([(n.lower(), i)
                  for i, n in enumerate(mtlmdl.parameters.split(","))])
    # put the parameters in an array
    params = np.zeros(len(pdict))

    # get the user give parameters
    ui = mtldict.pop("Content")

    for p in ui:
        name, val = [x.strip() for x in re.split(r"[= ]", p) if x.strip()]
        idx = pdict.get(name.lower())
        if idx is None:
            fatal_inp_error("Material: {0}: invalid parameter for the {1} "
                            "material model".format(name, model))
            continue
        try:
            val = float(val)
        except ValueError:
            fatal_inp_error("Material: {0}: invalid value "
                            "{1}".format(name, val))
            continue
        params[idx] = val

    return [model, params]


def pOptimization(optdict, *args):
    if not optdict: return


def pPermutation(permdict, *args):
    """Parse the permutation block

    """
    return
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

    seed = permdict["seed"]
    if seed is not None:
        pp.set_random_seed(seed)
    N_default = 10
    pp.update_safe({S_RANGE: lambda a, b, N=N_default: np.linspace(a, b, N),
                    S_LIST: lambda a: np.array(a),
                    S_WEIBULL: lambda a, b, N=N_default: a * pp.RAND.weibull(b, N),
                    S_UNIFORM: lambda a, b, N=N_default: pp.RAND.uniform(a, b, N),
                    S_NORMAL: lambda a, b, N=N_default: pp.RAND.normal(a, b, N),
                    S_PERC: lambda a, b, N=N_default: (
                        np.linspace(a-(b/100.)*a, a+(b/100.)* a, N))})

    # response function
    respfcn = permdict["ResponseFunction"]
    if len(respfcn) > 1:
        fatal_inp_error("Only one {0} tag supported".format(S_RESP_FCN))
    elif respfcn:
        respfcn = respfcn[0]
    response_fcn = check_response_function_element(respfcn)

    # read in permutated values
    p = []
    for items in permdict["Permutate"]:
        var = str(items.attributes.get(S_VAR).value)
        values = str(items.attributes.get(S_VALS).value)
        s = re.search(r"(?P<meth>\w+)\s*\(", values)
        if s: meth = s.group("meth")
        else: meth = ""
        if permdict["method"] == "shotgun":
            if meth != S_RANGE:
                fatal_inp_error("{0}: expected values=range(...".format(S_SHOTGUN))
                continue
            values = re.sub(meth, S_UNIFORM, values)
        try:
            p.append([var, eval(values, pp.GDICT, pp.SAFE)])
        except:
            fatal_inp_error("{0}: invalid expression".format(values))
            continue

    if permdict["correlations"] is not None:
        corr = xmltools.str2list(permdict["correlations"], dtype=str)
        for (i, c) in enumerate(corr):
            if c.lower() not in (S_TBL, S_PLOT, S_NONE):
                fatal_inp_error("{0}: unrecognized correlation option".format(c))
            corr[i] = c.lower()
        if "none" in corr:
            corr = None
        permdict["correlation"] = corr

    pdict["correlation"] = permdict["correlation"]
    pdict["parameters"] = p
    pdict["method"] = permdict["method"]

    pdict.update(response_fcn)

    return pdict


def pExtract(extdict, driver):
    """Set up the extraction request

    """
    if not extdict:
        return None

    elements = extdict.pop("Elements")

    # --- get requested variables to extract
    # extdict["Elements"]["Variables"]["Content"] is a list of the form
    # [[line1], [line2], ..., [linen]]
    # where line1, line2, ..., linen are the lines in the Variabes element of the
    # input file
    variables = []
    if elements["Variables"]:
        for _vars in elements["Variables"]["Content"]:
            if not _vars:
                continue
            _vars = _vars.split()
            variables.extend([stringify(var, "upper") for var in _vars])
        if "ALL" in variables:
            variables = "ALL"

    paths = driver.format_path_extraction(elements.pop("Path"))
    # get Paths to extract -> further parsing is handled by drivers that
    # support extracting paths
    return extdict["format"], extdict["step"], extdict["ffmt"], variables, paths


if __name__ == "__main__":
    test = """\
<GMDSpec>
    <Function id="2" type="analytic_expression" var="t">
      sin(t)
    </Function>
    <Function id="3" type="piecewise_linear" var="t" href="test.fcn"/>
    <Function id="4" type="piecewise_linear" var="t" href="test.fcn">
      2 4
      4 6
    </Function>
    <Physics driver="solid">
        <Material model="elastic">
            <K> 10 </K>
            <G> 8 </G>
        </Material>
        <Path type="prdef">
            0 10 222222 0 0 0 0 0 0
            1 10 222222 1 0 0 0 0 0
        </Path>
    </Physics>
    <Extract>
    </Extract>
<!--    <Optimization>
        <AuxiliaryFile>
        </AuxiliaryFile>
        <Optimize var="B" initial_value="23"/>
    </Optimization> -->
</GMDSpec>
"""
    parsed_input = parse_input(test)
    print parsed_input
