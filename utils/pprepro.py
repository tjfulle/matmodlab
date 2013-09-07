import os
import re
import sys
import math
import numpy as np
import xml.dom.minidom as xdom


# safe values to be used in eval
RAND = np.random.RandomState(17)
GDICT = {"__builtins__": None}
SAFE = {"np": np,
        "sqrt": np.sqrt, "max": np.amax, "min": np.amin,
        "stdev": np.std, "std": np.std,
        "abs": np.abs, "ave": np.average, "average": np.average,
        "sin": np.sin, "cos": np.cos, "tan": np.tan,
        "asin": np.arcsin, "acos": np.arccos,
        "atan": np.arctan, "atan2": np.arctan2,
        "log": np.log, "exp": np.exp,
        "floor": np.floor, "ceil": np.ceil,
        "pi": math.pi, "G": 9.80665, "inf": np.inf, "nan": np.nan,
        "random": RAND.random_sample, }
DEBUG = False
WARNINGS = 0


def find_vars_to_sub(lines):
    """Find vars to be preprocessed

    Parameters
    ----------
    lines : str
        user input

    Returns
    -------
    vars_to_sub : dict
        dictionary of found variables

    """
    global SAFE
    vars_to_sub = {}
    hold = []
    regex = r"(?i)\{\s*[\w]+[\w\d]*\s*=.*?\}"
    variables = re.findall(regex, lines)
    for variable in variables:
        vsplit = re.sub(r"[\{\}]", "", variable).split("=")
        key = vsplit[0].strip()
        hold.append(key)
        expr = vsplit[1].strip()
        var_to_sub = eval(expr, GDICT, SAFE)
        SAFE[key] = var_to_sub
        vars_to_sub[key] = repr(var_to_sub)

    # replace value
    for (i, variable) in enumerate(variables):
        lines = re.sub(re.escape(variable), vars_to_sub[hold[i]], lines)

    return lines, vars_to_sub


def find_subs_to_make(lines):
    """Find all substitutions that need to be made

    """
    regex = re.compile(r"{(?P<var>.*)}")
    subs_to_make = [k for m in regex.findall(lines) for k in m.split()]
    return subs_to_make


def find_and_make_subs(lines, prepro=None, disp=0):
    """Preprocess the input file

    Parameters
    ----------
    lines : str
        user input

    Returns
    -------
    lines : str
        preprocessed user input

    """
    global SAFE
    if prepro is not None:
        SAFE.update(prepro)
    lines, vars_to_sub = find_vars_to_sub(lines)
    if not vars_to_sub and prepro is None:
        if disp:
            return lines, 0
        return lines
    return make_var_subs(lines, vars_to_sub, disp=disp)


def make_var_subs(lines, vars_to_sub, disp=0):
    global WARNING
    # the regular expression that defines the preprocessing
    pregex = r"(?i){{.*\b{0:s}\b.*}}"

    if vars_to_sub and DEBUG:
        # Print out preprocessed values for debugging
        sys.stdout.write("Preprocessor values:\n")
        name_len = max([len(x) for x in vars_to_sub])
        for pat, repl in vars_to_sub.items():
            sys.stdout.write("    {0:<{1}s} {2}\n".format(
                pat + ":", name_len + 2, repl))

        for pat, repl in vars_to_sub.items():
            # Check that each preprocessor variable is used somewhere else in
            # the file. Issue a warning if not. It is done here outside of the
            # replacement loop to be sure that we can check all variables.
            # Because variables in the preprocessor block were put in the SAFE
            # it is possible for, say, var_2 to be replaced when replacing
            # var_1 if the user specified something like
            #          param = {var_1 * var_2}
            # So, since we want to check that all preprocessed variables are
            # used, we do it here. Checking is important for permutate and
            # optimization jobs to make sure that all permutated and optimized
            # variables are used properly in the file before the job actually
            # begins.
            if pat not in SAFE and not re.search(pregex.format(pat), lines):
                sys.stderr.write("*** prepro: {0}: not found in "
                                 "input\n".format(pat))
                WARNINGS += 1
                continue

    # Replace '{ var }' with '{ (var_value) }'
    regex = r"{{.*\b{0}\b.*}}"
    nsubs = 0
    for pat, repl in vars_to_sub.items():
        repl = "({0})".format(re.sub(r"[\{\}]", " ", repl).strip())
        matches = re.findall(regex.format(pat), lines)
        for match in matches:
            mrepl, n = re.subn(r"\b{0}\b".format(pat), repl, match)
            lines = re.sub(re.escape(match), mrepl, lines)
            nsubs += n
            continue
        continue

    # Evaluate substitutions
    regex = r"(?i){.*?}"
    matches = re.findall(regex, lines)
    for pat in matches:
        repl = re.sub(r"[\{\}]", "", pat)
        repl = repr(eval(repl, GDICT, SAFE))
        lines = re.sub(re.escape(pat), repl, lines)

    if disp:
        return lines, nsubs
    return lines


def find_and_fill_includes(lines):
    """Look for 'include' commands in lines and insert then contents in place

    Parameters
    ----------
    lines : str
        User input

    Returns
    -------
    lines : str
        User input, modified in place, with inserts inserted

    """
    #doc = xdom.parseString(lines)
    #includes = doc.getElementsByTagName("Include")
    #print dir(includes[0])
    #print includes
    #sys.exit('check include')
    regex = r"(?i)<include\s(?P<include>.*)/>"
    _lines = []
    for line in lines.split("\n"):
        if not line.split():
            _lines.append(line.strip())
            continue
        include = re.search(regex, line)
        if include is None:
            _lines.append(line)
            continue

        href = re.search(r"""href=["'](?P<href>.*?)["']""",
                         include.group("include"))
        if not href:
            raise SystemExit("expected href='...'")
        name = href.group("href").strip()
        fpath = os.path.realpath(os.path.expanduser(name))
        try:
            fill = open(fpath, "r").read()
        except IOError:
            raise SystemExit("{0}: include not found".format(repr(name)))
        _lines.extend(find_and_fill_includes(fill).split("\n"))
        continue
    return "\n".join(_lines)
