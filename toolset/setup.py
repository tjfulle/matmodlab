#!/usr/bin/env python
"""Configure gmp

"""
import os
import re
import sys
import imp
import shutil
import argparse
import subprocess
from distutils import sysconfig

version = "0.0.0"

def logmes(message, end="\n"):
    sys.stdout.write("{0}{1}".format(message, end))
    sys.stdout.flush()


def logerr(message=None, end="\n", errors=[0]):
    if message is None:
        return errors[0]
    sys.stdout.write("*** setup: error: {0}{1}".format(message, end))
    errors[0] += 1


def stop(msg=""):
    sys.exit("setup: error: Stopping due to previous errors. {0}".format(msg))


def main(argv=None):
    """Setup the gmd executable

    """
    if argv is None:
        argv = sys.argv[1:]
    parser = argparse.ArgumentParser()
    parser.add_argument("-B", default=False,
        action="store_true",
        help="Suppress python byte code generation [default: %(default)s]")
    parser.add_argument("--Ntpl", default=False, action="store_true",
        help="Do not build TPLs [default: %(default)s]")
    parser.add_argument("--Rtpl", default=False, action="store_true",
        help="Force rebuild of TPLs [default: %(default)s]")
    parser.add_argument("--mtldirs", default=[], action="append",
        help="Additional directories to find makemf.py files [default: None]")
    parser.add_argument("--testdirs", default=[], action="append",
        help="Additional directories to find test files [default: None]")
    args = parser.parse_args(argv)

    build_tpls = not args.Ntpl
    make_exowrap = True

    # module level variables
    fpath = os.path.realpath(__file__)
    fdir, fnam = os.path.split(fpath)
    root = os.path.dirname(fdir)
    tpl = os.path.join(root, "tpl")
    libd = os.path.join(root, "lib")
    mtld = os.path.join(root, "materials")
    utld = os.path.join(root, "utils")
    pypath = [root]

    tools = os.path.join(root, "toolset")
    core = os.path.join(root, "core")

    path = os.getenv("PATH", "").split(os.pathsep)
    logmes("setup: gmd {0}".format(version))

    mtldirs = []
    for d in args.mtldirs:
        d = os.path.realpath(d)
        if not os.path.isdir(d):
            logerr("{0}: no such directory".format(d))
            continue
        mtldirs.append(d)
    mtldirs = os.pathsep.join(mtldirs)

    testdirs = []
    for d in args.testdirs:
        d = os.path.realpath(d)
        if not os.path.isdir(d):
            logerr("{0}: no such directory".format(d))
            continue
        testdirs.append(d)
    testdirs = os.pathsep.join(testdirs)

    # --- system
    logmes("checking host platform", end="... ")
    platform = sys.platform
    logmes(platform)
    sys.dont_write_bytecode = args.B

    # --- python
    logmes("setup: checking python interpreter")
    logmes("path to python executable", end="... ")
    pyexe = os.path.realpath(sys.executable)
    logmes(pyexe)

    # --- python version
    logmes("checking python version", end="... ")
    (major, minor, micro, relev, ser) = sys.version_info
    logmes("python {0}.{1}.{2}.{3}".format(*sys.version_info))
    if (major != 3 and major != 2) or (major == 2 and minor < 6):
        logerr("python >= 2.6 required")

    # --- 64 bit python?
    logmes("checking for 64 bit python", end="... ")
    if sys.maxsize < 2 ** 32:
        logmes("no")
        logerr("gmd requires 64 bit python (due to exowrap)")
    else: logmes("yes")

    # --- numpy
    logmes("checking whether numpy is importable", end="... ")
    try:
        import numpy
        logmes("yes")
    except ImportError:
        logerr("no")

    # --- scipy
    logmes("checking whether scipy is importable", end="... ")
    try:
        import scipy
        logmes("yes")
    except ImportError:
        logerr("no")

    # find f2py
    logmes("setup: checking fortran compiler")
    f2py = os.path.join(os.path.dirname(pyexe), "f2py")
    logmes("checking for compatible f2py", end="... ")
    if not os.path.isfile(f2py) and sys.platform == "darwin":
        f2py = os.path.join(pyexe.split("Resources", 1)[0], "bin/f2py")
    if not os.path.isfile(f2py):
        logmes("no")
        logerr("compatible f2py required for building exowrap")
        make_exowrap = False
    else: logmes("yes")

    logmes("checking for gfortran", end="... ")
    gfortran = None
    for p in path:
        if os.path.isfile(os.path.join(p, "gfortran")):
            gfortran = os.path.join(p, "gfortran")
            logmes("yes")
            break
    else:
        logmes("no")
        logerr("gfortran required for building tpl libraries")

    if logerr():
        stop("Resolve before continuing")

    # build TPLs
    logmes("setup: looking for tpl.py files")
    for (d, dirs, files) in os.walk(tpl):
        if "tpl.py" in files:
            f = os.path.join(d, "tpl.py")
            dd = d.replace(root, ".")
            logmes("building tpl in {0}".format(dd), end="... ")
            tplpy = imp.load_source("tpl", os.path.join(d, "tpl.py"))
            info = tplpy.build_tpl(ROOT=root, SKIPTPL=args.Ntpl)
            if info is None:
                logerr("tpl failed to build")
            else:
                logmes("yes")
                pypath.append(info.get("PYTHONPATH"))
    if logerr():
        stop("Resolve before continuing")

    pypath = os.pathsep.join(x for x in pypath if x)
    for path in pypath:
        if path not in sys.path:
            sys.path.insert(0, path)

    # --- executables
    logmes("setup: writing executable scripts")
    pyopts = "" if not sys.dont_write_bytecode else "-B"

    write_exe("gmd", tools, os.path.join(root, "main.py"),
              pypath, pyexe, pyopts, "")

    write_exe("buildmtls", tools, os.path.join(utld, "building.py"),
              pypath, pyexe, pyopts, mtldirs)

    write_exe("runtests", tools, os.path.join(utld, "testing.py"),
              pypath, pyexe, pyopts, testdirs)

    logmes("setup: Setup complete")
    if build_tpls:
        logmes("setup: To finish installation, "
               "add: \n          {0}\n"
               "       to your PATH environment variable".format(tools))
    return


def remove(paths):
    """Remove paths"""
    if not isinstance(paths, (list, tuple)):
        paths = [paths]

    for path in paths:
        pyc = path + ".c" if path.endswith(".py") else None
        try: os.remove(path)
        except OSError: pass
        try: os.remove(pyc)
        except OSError: pass
        except TypeError: pass
        continue
    return


def write_exe(name, destd, pyfile, pypath, pyexe, pyopts, env):
    exe = os.path.join(destd, name)
    remove(exe)
    logmes("writing {0}".format(os.path.basename(exe)), end="...  ")
    if not os.path.isfile(pyfile):
        logerr("{0}: no such file".format(pyfile))
        return
    with open(exe, "w") as fobj:
        fobj.write("#!/bin/sh -f\n")
        fobj.write("export PYTHONPATH={0}\n".format(pypath))
        fobj.write("export PY_EXE_ENV={0}\n".format(env))
        fobj.write("PYTHON={0}\n".format(pyexe))
        fobj.write("PYFILE={0}\n".format(pyfile))
        fobj.write('$PYTHON {0} $PYFILE "$@"\n'.format(pyopts))
    os.chmod(exe, 0o750)
    logmes("done")
    return

if __name__ == "__main__":
    main()
