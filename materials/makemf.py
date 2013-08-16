import os
import sys
import imp
import subprocess
import shutil
from numpy.f2py import main as f2py

from utils.namespace import Namespace

D = os.path.dirname(os.path.realpath(__file__))
F90_MODELS = {"elastic": {"signature": os.path.join(D, "elastic.pyf"),
                          "interface": os.path.join(D, "elastic_interface.py"),
                          "class": "Elastic",
                          "files": [os.path.join(D, f) for f in
                                    ("elastic.f90", "elastic_interface.f90")]}}

def makemf(*args, **kwargs):

    fc = kwargs.get("FC", "gfortran")
    destd = kwargs.get("DESTD", D)
    materials = kwargs.get("MATERIALS")

    mtldict = {}

    if materials is None:
        materials = F90_MODELS.keys()

    argv = [x for x in sys.argv]
    for (name, items) in F90_MODELS.items():

        if name not in materials:
            continue

        source_files = items["files"]
        signature = os.path.realpath(os.path.join(D, items["signature"]))

        # f2py pulls its arguments from sys.argv
        # f2py returns none if successful, it is an exception if not
        # successful
        fflags = "-fPIC -shared"
        sys.argv = ["f2py", "-c", "-I{0}".format(D), "-m", name,
                    "--f90flags={0}".format(fflags),
                    "--f77flags={0}".format(fflags),
                    "--f77exec={0}".format(fc), "--f90exec={0}".format(fc),
                    signature, "--quiet"]
        sys.argv.extend(source_files)

        f = os.devnull
        with open(f, "w") as sys.stdout:
            with open(f, "a") as sys.stderr:
                try:
                    built = not f2py()
                except BaseException as e:
                    msg = re.sub(r"error: ", "", e.message)
                    built = None
                except:
                    msg = "failed to build kayenta with f2py"
                    built = None

        sys.stdout, sys.stderr = sys.__stdout__, sys.__stderr__
        sys.stdout.flush()
        sys.stderr.flush()

        if built is None:
            print msg
            continue

        interface = items["interface"]
        py_mod = os.path.splitext(os.path.basename(interface))[0]
        module = imp.load_source(py_mod, interface)
        material = getattr(module, items["class"])()

        ns = Namespace()
        ns.filepath = module.__file__.rstrip("c")
        ns.mclass = material.__class__.__name__
        ns.name = material.name
        ns.parameters = ", ".join(material.parameters()).lower()
        ns.driver = material.driver

        mtldict[material.name] = ns

        if destd != D:
            shutil.move(name + ".so", os.path.join(destd, name + ".so"))

    sys.argv = [x for x in argv]

    return mtldict

if __name__ == "__main__":
    makemf()
