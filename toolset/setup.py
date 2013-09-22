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

fpath = os.path.realpath(__file__)
fdir, fnam = os.path.split(fpath)
root = os.path.dirname(fdir)
sys.path.insert(0, root)
from __config__ import __version__, SPLASH
import utils.makeut as makeut

version = ".".join(str(x) for x in __version__)

def log_message(message, end="\n"):
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

    log_message(SPLASH)

    build_tpls = not args.Ntpl
    make_exowrap = True

    # module level variables
    tpl = os.path.join(root, "tpl")
    libd = os.path.join(root, "lib")
    mtld = os.path.join(root, "materials")
    utld = os.path.join(root, "utils")
    vizd = os.path.join(root, "viz")
    tools = os.path.join(root, "toolset")
    core = os.path.join(root, "core")

    pypath = [root]

    path = os.getenv("PATH", "").split(os.pathsep)
    log_message("setup: gmd {0}".format(version))

    mtldirs = [os.path.realpath(d) for d in args.mtldirs]
    for d in mtldirs:
        if not os.path.isdir(d):
            logerr("{0}: no such material directory".format(d))
            continue
    mtldirs = os.pathsep.join(list(set(mtldirs)))

    testdirs = [os.path.realpath(d) for d in args.testdirs]
    for d in testdirs:
        if not os.path.isdir(d):
            logerr("{0}: no such test directory".format(d))
            continue
    testdirs = os.pathsep.join(list(set(testdirs)))

    # --- system
    log_message("checking host platform", end="... ")
    platform = sys.platform
    log_message(platform)
    sys.dont_write_bytecode = args.B

    # --- python
    log_message("setup: checking python interpreter")
    log_message("path to python executable", end="... ")
    pyexe = os.path.realpath(sys.executable)
    log_message(pyexe)

    # --- python version
    log_message("checking python version", end="... ")
    (major, minor, micro, relev, ser) = sys.version_info
    log_message("python {0}.{1}.{2}.{3}".format(*sys.version_info))
    if (major != 3 and major != 2) or (major == 2 and minor < 6):
        logerr("python >= 2.6 required")

    # --- 64 bit python?
    log_message("checking for 64 bit python", end="... ")
    if sys.maxsize < 2 ** 32:
        log_message("no")
        logerr("gmd requires 64 bit python (due to exowrap)")
    else: log_message("yes")

    # --- numpy
    log_message("checking whether numpy is importable", end="... ")
    try:
        import numpy
        log_message("yes")
    except ImportError:
        logerr("no")

    # --- scipy
    log_message("checking whether scipy is importable", end="... ")
    try:
        import scipy
        log_message("yes")
    except ImportError:
        logerr("no")

    # find f2py
    log_message("setup: checking fortran compiler")
    f2py = os.path.join(os.path.dirname(pyexe), "f2py")
    log_message("checking for compatible f2py", end="... ")
    if not os.path.isfile(f2py) and sys.platform == "darwin":
        f2py = os.path.join(pyexe.split("Resources", 1)[0], "bin/f2py")
    if not os.path.isfile(f2py):
        log_message("no")
        logerr("compatible f2py required for building exowrap")
        make_exowrap = False
    else: log_message("yes")

    log_message("checking for gfortran", end="... ")
    gfortran = None
    for p in path:
        if os.path.isfile(os.path.join(p, "gfortran")):
            gfortran = os.path.join(p, "gfortran")
            log_message("yes")
            break
    else:
        log_message("no")
        logerr("gfortran required for building tpl libraries")

    if logerr():
        stop("Resolve before continuing")

    # build TPLs
    log_message("setup: looking for tpl.py files")
    for (d, dirs, files) in os.walk(tpl):
        if "tpl.py" in files:
            f = os.path.join(d, "tpl.py")
            dd = d.replace(root, ".")
            log_message("building tpl in {0}".format(dd), end="... ")
            tplpy = imp.load_source("tpl", os.path.join(d, "tpl.py"))
            info = tplpy.build_tpl(ROOT=root, SKIPTPL=args.Ntpl,
                                   REBUILD=args.Rtpl)
            if info is None:
                logerr("tpl failed to build")
            else:
                log_message("yes")
                pypath.append(info.get("PYTHONPATH"))
    if logerr():
        stop("Resolve before continuing")

    # build the fortran based utilities
    log_message("building fortran based utils", end="... ")
    stat = makeut.makeut(libd, gfortran)
    if stat != 0:
        log_message("no")
        logerr("failed to build fortran based utils")
    else:
        log_message("yes")

    pypath = os.pathsep.join(x for x in pypath if x)
    for path in pypath:
        if path not in sys.path:
            sys.path.insert(0, path)

    # --- executables
    log_message("setup: writing executable scripts")
    pyopts = "" if not sys.dont_write_bytecode else "-B"

    write_exe("gmd", tools, os.path.join(root, "main.py"),
              pyexe, pyopts, {"PYTHONPATH": pypath, "FC": gfortran},
              {"GMDMTLS": mtldirs})

    write_exe("buildmtls", tools, os.path.join(core, "build.py"),
              pyexe, pyopts, {"PYTHONPATH": pypath, "FC": gfortran},
              {"GMDMTLS": mtldirs})

    write_exe("runtests", tools, os.path.join(core, "test.py"),
              pyexe, pyopts, {"PYTHONPATH": pypath}, {"GMDTESTS": testdirs})

    write_exe("gmddump", tools, os.path.join(utld, "exodump.py"),
              pyexe, pyopts, {"PYTHONPATH": pypath})

    write_exe("gmdviz", tools, os.path.join(vizd, "plot2d.py"),
              pyexe, pyopts, {"PYTHONPATH": pypath})

    write_exe("gmddiff", tools, os.path.join(utld, "gmddiff.py"),
              pyexe, pyopts, {"PYTHONPATH": pypath})

    log_message("setup: Setup complete")
    if build_tpls:
        log_message("setup: To finish installation, "
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


def write_exe(name, destd, pyfile, pyexe, pyopts, env, gmdvars=None):
    exe = os.path.join(destd, name)
    remove(exe)
    log_message("writing {0}".format(os.path.basename(exe)), end="...  ")
    if not os.path.isfile(pyfile):
        logerr("{0}: no such file".format(pyfile))
        return

    # set environment variables in scripts
    env = ["export {0}={1}".format(k, v) for (k, v) in env.items()]
    if gmdvars:
        # if user set gmd variable on command line (gmdvar), add it to the
        # already recognized gmd environment variable
        for (k, v) in gmdvars.items():
            if not v.split():
                continue
            env.append("export {0}={1}{2}${0}".format(k, v, os.pathsep))
    env = "\n".join(env)

    with open(exe, "w") as fobj:
        fobj.write("#!/bin/sh -f\n")
        fobj.write("{0}\n".format(env))
        fobj.write("export PYTHON={0}\n".format(pyexe))
        fobj.write("PYFILE={0}\n".format(pyfile))
        fobj.write('$PYTHON {0} $PYFILE "$@"\n'.format(pyopts))
    os.chmod(exe, 0o750)
    log_message("done")
    return

if __name__ == "__main__":
    main()
