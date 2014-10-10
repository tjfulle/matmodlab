#!/usr/bin/env python
import os
import re
import sys
import glob
import stat
import argparse
import textwrap
import xml.dom.minidom as xdom

from utils.pprepro import preprocess_input

def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    p = argparse.ArgumentParser()
    p.add_argument("sources", nargs="*")
    args = p.parse_args(argv)

    if not args.sources:
        args.sources = glob.glob("*.xml")

    inp_files = []
    test_files = []
    for f in args.sources:
        assert os.path.isfile(f), "%s: no such file" % f
        filepath = os.path.realpath(f)
        if f.endswith(".xml"):
            inp_files.append(filepath)
        elif f.endswith(".rxml"):
            test_files.append(filepath)
        else:
            raise ValueError("%s: expected [r]xml file" % f)

    convert_xml_input(inp_files)


def format_item(item):
    item = item.strip()
    try: return "{0:d}".format(int(item))
    except ValueError: pass
    try: return "{0:.6e}".format(float(item))
    except ValueError: pass
    return '"{0}"'.format(item)


def path_repl(item):
    item = item.lower()
    d = {"ndumps": "num_io_dumps", "ratfac": "rate_multiplier",
         "nfac": "step_multiplier", "format": "path_input",
         "type": "", "ampl": "amplitude"}
    return d.get(item, item)


def is_default_val(key, val):
    small = 1.E-12
    key = key.lower().strip()
    try: val = val.lower().strip()
    except AttributeError: pass
    eq = lambda x, y: abs(float(x) - y) < small
    if re.search(r"[a-z]star", key) and eq(val, 1.):
        return True
    if key.strip() == "kappa" and eq(val, 0.):
        return True
    if "multiplier" in key and eq(val, 1.):
        return True
    if "num_io" in key and "all" in val:
        return True
    if "ampl" in key and eq(val, 1.):
        return True
    return False


def child_nodes_to_list(el):
    it = []
    for node in el.childNodes:
        if node.nodeType == node.COMMENT_NODE:
            continue
        try:
            it.extend([" ".join(s.split()) for s in
                         node.data.split("\n") if s.strip()])
        except AttributeError as E:
            raise E
    return it

def format_path(path_el):
    path = child_nodes_to_list(path_el)
    return "\n    ".join([" ".join(l.split())
                          for l in path if l.split()])


def upcase(item):
    return item.strip().upper()


def format_mat_params(paramel):
    params = []
    for node in paramel.childNodes:
        if node.nodeType != node.ELEMENT_NODE:
            continue
        if not node.nodeName:
            continue
        kw = upcase(node.nodeName)
        v = format_item(node.firstChild.data)
        try:
            if abs(v) < 1.E-14:
                continue
        except TypeError:
            pass
        params.append('"{0}":{1}'.format(kw, v))
    params = "{{{0}}}".format(", ".join(params))
    params = textwrap.fill(params, width=70, subsequent_indent=" "*18)
    return params.replace(":", ": ")


def fix_mat_model(model):
    return {"mnrv": "mooney rivlin"}.get(model.lower(), model.lower())


def convert_rxml_input(f, disp=1):
    filedir, filename = os.path.split(f)
    fileroot = os.path.splitext(filename)[0]
    runid = fileroot.replace("-", "_")

    lines = open(f).read()

    # start processing file
    doc = xdom.parseString(lines)

    # get the root
    root = doc.getElementsByTagName("rtest")[0]

    # get the keywords
    keywords = root.getElementsByTagName("keywords")[0]
    keywords = [str(s).strip() for s in keywords.firstChild.data.split()]

    name = "".join(x[0].upper() + x[1:].lower() for x in runid.split("_"))
    inp = """
class Test{0}(TestBase):
    def __init__(self):
        self.runid = runid
        self.keywords = {1}
    def run_job(self):
        runner(d=self.test_dir, v=0)
""".format(name, keywords)
    return inp


def convert_xml_input(files):

    for f in files:
        filedir, filename = os.path.split(f)
        fileroot = os.path.splitext(filename)[0]
        runid = fileroot.replace("-", "_")

        lines = open(f).read()

        # preprocess all lines
        lines = preprocess_input(lines)

        # start processing file
        doc = xdom.parseString(lines)

        # get the root
        root = doc.getElementsByTagName("MMLSpec")[0]

        # get the driver
        path_el = root.getElementsByTagName("Path")[0]
        path_op = dict([(path_repl(kw), format_item(v))
                        for (kw, v) in path_el.attributes.items()
                        if path_repl(kw)])
        dopts = ", ".join(['{0}={1}'.format(kw, v) for (kw,v) in path_op.items()
                           if not is_default_val(kw, v)])
        path = format_path(path_el)
        dspec = ('driver = Driver("Continuum", path, '
                 'logger=logger, {0})'.format(dopts))
        dspec = textwrap.fill(dspec, width=78, subsequent_indent=" "*20)

        # get the material
        mat_el = root.getElementsByTagName("Material")[0]
        mat_mod = fix_mat_model(mat_el.getAttribute("model"))
        params = format_mat_params(mat_el)

        ft = os.path.join(filedir, fileroot + ".rxml")
        if os.path.isfile(ft):
            test_inp = convert_rxml_input(ft, disp=0)
        else:
            test_inp = ""

        inp = '''#!/usr/bin/env mmd
from matmodlab import *

runid = "{0}"
{5}
@matmodlab
def runner(d=None, v=1):
    d = d or os.getcwd()
    logfile = os.path.join(d, runid + ".log")
    logger = Logger(logfile=logfile, verbosity=v)

    # set up path
    path = """
    {1}
    """
    {2}

    # set up material
    parameters = {3}
    material = Material("{4}", parameters, logger=logger)

    # setup simulation
    mps = MaterialPointSimulator(runid, driver, material, logger=logger, d=d)
    mps.run()

    return 0

if __name__ == "__main__":
    sys.exit(runner())
'''.format(runid, path, dspec, params, mat_mod, test_inp)#, subbed_vars=vars_to_sub)

        if test_inp:
            pyf = os.path.join(filedir, "test_" + runid + ".py")
        else:
            pyf = os.path.join(filedir, runid + ".py")

        with open(pyf, "w") as fh:
            fh.write(inp)


if __name__ == "__main__":
    main()
