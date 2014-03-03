#!/usr/bin/env python
import os
import re
import sys
import glob
import shutil
import tempfile
from subprocess import call, STDOUT
from argparse import ArgumentParser, SUPPRESS

from utils.impmod import load_file
from utils.int2str import int2str
from materials.material import write_mtldb, gather_materials
from utils.fortran.extbuilder import FortranExtBuilder, FortranNotFoundError
from __config__ import *


QUIET = False


def main(argv=None):
    """Setup and build components of the Material Model Laboratory

    Parameters
    ----------
    argv : list, optional
      command line arguments

    """
    global QUIET
    if argv is None:
        argv = sys.argv[1:]

    # --- input parser
    parser = ArgumentParser(usage="setup.py <subcommand> [options] [args]")

    # global options
    parser.add_argument("--with-fortran", default=1, type=int,
        help="Verbosity [default: %(default)s]")
    parser.add_argument("--fc", default=None,
        help="Path to fortran compiler [default: gfortran]")
    parser.add_argument("-v", default=1, type=int,
        help="Verbosity [default: %(default)s]")
    parser.set_defaults(rebuild_all=False)

    # --- subparsers
    subdesc = """Most subcommands take additional options/arguments.  To see a
                 detailed list, execute setup.py subcommand --help"""
    subparsers = parser.add_subparsers(title="subcommands",
                                       dest="subparser_name",
                                       description=subdesc,
                                       help="sub-command help")

    # --- 'all' subparser
    parser_all = subparsers.add_parser("build_all", help="Build all components")
    parser_all.add_argument("--rebuild", dest="rebuild_all",
        default=False, action="store_true",
        help="Rebuild all components [default: %(default)s]")

    # --- 'material' subparser
    parser_mtl = subparsers.add_parser("build_mtl",
        help="Build only the material models")
    parser_mtl.add_argument("-m", dest="mats_to_build",
        action="append", default=[],
        help="Material to build [default: all]")
    parser_mtl.add_argument("-w", dest="wipe_database",
        action="store_true", default=False,
        help="Wipe material database before building [default: all]")
    parser_mtl.add_argument("-v", default=1, type=int, help=SUPPRESS)

    # --- 'clean' subparser
    parser_cln = subparsers.add_parser("clean", help="Clean all directories")

    # parse the command line
    args = parser.parse_args(argv)
    command = args.subparser_name

    if command == "clean":
        sys.exit(clean_matmodlab())

    # run everything in the matmodlab directory
    cwd = os.getcwd()
    os.chdir(ROOT_D)

    if args.v == 0:
        QUIET = True

    cout(SPLASH)

    if command == "build_all" and args.rebuild_all:
        remove(BLD_D)
        os.makedirs(BLD_D)

    # --- check prerequisites
    if command == "build_all":
        errors = check_prereqs()
        if errors:
            raise SystemExit("System does not satisfy prerequisites")

    # python passes mutables by reference. The env dictionary initialized here
    # is updated in each function it is passed to below
    env = {}

    # base configuration
    try:
        fb = FortranExtBuilder("matmodlab", fc=args.fc, verbosity=args.v)
    except FortranNotFoundError:
        fb = None
        cout("fortran compiler not found, skipping fortran tools")

    if command == "build_all" and fb:
        # build optional fortran utilities
        ext = "mmlabpack"
        mmlabpack = os.path.join(PKG_D, ext + ".so")
        remove(mmlabpack)
        sources = [os.path.join(ROOT_D, "utils/fortran/mmlabpack.f90"),
                   os.path.join(ROOT_D, "utils/fortran/dgpadm.f")]
        fb.add_extension(ext, sources, requires_lapack=True)

    if command in ("build_mtl", "build_all"):
        try:
            # args.mats_to_build and args.wipe_database only defined for
            # build_mtl parser
            mats_to_build = [m.lower() for m in args.mats_to_build]
            if not mats_to_build:
                mats_to_build = "all"
            wipe = args.wipe_database
        except AttributeError:
            mats_to_build = "all"
            wipe = True

        cout("Gathering material[s] to be built")
        material_info = gather_materials(mats_to_build)
        cout("{0} material[s] found: {1}".format(
                int2str(len(material_info), c=True),
                ", ".join(x for x in material_info)))

        for (name, conf) in material_info.items():
            sources = conf.get("source_files")
            if sources and not fb:
                cerr("*** warning: fortran based models will not be built")
                break
            if sources:
                # assume fortran model if source files are given
                conf["source_files"].append(FIO)
                ifile = conf["interface_file"]
                include_dirs = [os.path.dirname(ifile)]
                d = conf.get("include_dir")
                if d and d not in include_dirs:
                    include_dirs.append(conf["include_dir"])
                stat = fb.add_extension(name, sources, include_dirs=include_dirs,
                                        requires_lapack=conf.get("requires_lapack"))
                if stat:
                    # failed to add extension
                    del material_info[name]

    if fb and fb.exts_to_build:
        fb.build_extension_modules()
        for mtl in [x for x in material_info if x not in fb.built_ext_modules]:
            # remove from the material_info list any modules with source files that
            # didn't finish building
            if material_info[mtl]["source_files"]:
                cout("*** warning: {0}: failed to build".format(mtl))
                del material_info[mtl]

        # build the PYTHONPATH environment variable
        pypath = [ROOT_D]
        for (key, val) in material_info.items():
            pypath.append(os.path.dirname(val["interface_file"]))
        env.setdefault("PYTHONPATH", []).extend(pypath)

        write_mtldb(material_info, wipe)

    if command != "build_all":
        os.chdir(cwd)
        return 0

    # convert python lists in the env dict to : separated lists for the
    # shell
    for (k, v) in env.items():
        if not isinstance(v, (basestring, str)):
            v = os.pathsep.join(x for x in v if x.split())
        env[k] = v
    if fb:
        env["FC"] = fb.fc

    # write executables in ./toolset
    write_executables(env)

    # test the build, explicitly using the env dict
    stat = test_build(env)
    if stat != 0:
        cout("\n{0}\n"
             "Problems encountered during test.  To diagnose, execute\n"
             "  % runtests {1} -kfast -j4\n"
             "{0}\n".format("*" * 78, os.path.join(ROOT_D, "tests")))
    else:
        if TLS_D not in PATH:
            cout("To complete build, add {0} to the PATH "
                 "environment variable".format(TLS_D))
        else:
            cout("Build complete")

    os.chdir(cwd)
    return stat


def cout(message, end="\n"):
    """Write message to stdout """
    if not QUIET:
        sys.__stdout__.write(message + end)
        sys.__stdout__.flush()


def cerr(message):
    """Write message to stderr """
    sys.__stderr__.write(message + "\n")
    sys.__stderr__.flush()


def check_prereqs():
    """Check prequisits for the Material Model Laboratory

    """
    # --- system
    cout("Checking host platform", end="... ")
    platform = sys.platform
    cout(platform)

    # --- python
    cout("Checking python interpreter", end="... ")
    cout(PYEXE)

    errors  = 0
    # --- python version
    cout("Checking python version", end="... ")
    (major, minor, micro, relev, ser) = sys.version_info
    cout("Python {0}.{1}.{2}.{3}".format(*sys.version_info))
    if (major != 3 and major != 2) or (major == 2 and minor < 7):
        errors += 1
        cerr("*** error: python >= 2.7 required")

    # --- numpy
    cout("Checking for numpy", end="... ")
    try:
        import numpy
        cout("yes")
    except ImportError:
        errors += 1
        cerr("no")

    # --- scipy
    cout("Checking for scipy", end="... ")
    try:
        import scipy
        cout("yes")
    except ImportError:
        errors += 1
        cerr("no")

    return errors


def write_executables(env):
    """Write the executable scripts

    Parameters
    ----------
    env : dict
      dictionary of environment variables needed for scripts

    """
    # --- executables
    cout("Writing executable scripts")
    _write_exe("mmd", os.path.join(CORE, "main.py"), env)
    _write_exe("mml", os.path.join(VIZ_D, "main.py"), env)
    _write_exe("runtests", os.path.join(CORE, "test.py"), env)
#    _write_exe("gmddump", os.path.join(UTL_D, "exo/exodump.py"), env)
#    _write_exe("exdump", os.path.join(UTL_D, "exo/exodump.py"), env)
    _write_exe("mmv", os.path.join(VIZ_D, "plot2d.py"), env)
#    _write_exe("exdiff", os.path.join(UTL_D, "exo/exodiff.py"), env)
    _write_exe("buildmtls", os.path.join(ROOT_D, "setup.py"), env, parg="build_mtl")
    cout("Executable scripts written")
    return


def remove(path):
    """Remove file or directory -- dangerous!

    """
    if not os.path.exists(path): return
    try: os.remove(path)
    except OSError: shutil.rmtree(path)
    return


def _write_exe(name, pyfile, env, parg=""):
    """Write executable script to the toolset directory

    Parameters
    ----------
    name : str
      name of executable
    pyfile : str
      path to python file executable will execute
    env : dict
      dictionary of environment variables to be written to the executable
    parg : str
      optional arguments to be sent to python script

    """

    # executable path -- remove if exists
    exe = os.path.join(TLS_D, name)
    remove(exe)
    if not os.path.isfile(pyfile):
        cerr("*** warning: {0}: no such file".format(pyfile))
        return

    # set environment variables in scripts
    _env = []
    for (k, v) in env.items():
        _env.append("export {0}={1}".format(k, v))
    _env = "\n".join(_env)

    cout("  {0}".format(os.path.basename(exe)), end="... ")
    with open(exe, "w") as fobj:
        fobj.write("#!/bin/sh -f\n")
        fobj.write("{0}\n".format(_env))
        fobj.write("export PYTHON={0}\n".format(PYEXE))
        fobj.write("PYFILE={0}\n".format(pyfile))
        fobj.write('$PYTHON $PYFILE {0} "$@"\n'.format(parg))
    os.chmod(exe, 0o750)
    cout("done")
    return


def clean_matmodlab():
    """Recursively clean the project

    """
    exts = (".pyc", ".o", ".a", ".con", ".so")
    cout("cleaning matmodlab", end="... ")
    for (dirname, dirs, files) in os.walk(ROOT_D):
        d = os.path.basename(dirname)
        if d == ".git":
            del dirs[:]
            continue
        if d == "build" or d.startswith("TestResults."):
            remove(dirname)
            del dirs[:]
            continue
        [remove(os.path.join(dirname, f)) for f in files if f.endswith(exts)]
    remove(F_MTL_MODEL_DB)
    cout("yes")


def test_build(env):
    """Test if the installation worked

    """
    # run tests in temporary directory
    d = tempfile.mkdtemp()
    os.chdir(d)

    # add . and ./toolset to path
    env["PATH"] = os.pathsep.join([ROOT_D, TLS_D] + PATH)

    cout("Testing installation", end="... ")
    exe = os.path.join(ROOT_D, "toolset/runtests")
    testd = os.path.join(ROOT_D, "tests")
    cmd = "{0} {1} -kfast -j4".format(exe, testd)
    con = open(os.devnull, "w")
    stat = call(cmd.split(), env=env, stdout=con, stderr=STDOUT)
    if stat != 0: cout("fail")
    else: cout("pass")
    os.chdir(ROOT_D)
    shutil.rmtree(d)
    return stat


def build_material(materials, verbosity=0):
    """Build a single material

    Parameters
    ----------
    material : str
      The name of the material to build

    Notes
    -----
    This function is meant to be imported by mmd to build a material before
    running the job.

    """
    cout("building {0}".format(", ".join(materials)))
    hold = [x for x in sys.argv]
    cmd = "-v {0} build_mtl {1}".format(
        verbosity, " ".join(["-m {0}".format(m) for m in materials]))
    stat = main(cmd.split())
    if stat == 0:
        cout("built {0}".format(",".join(materials)))
    sys.argv = [x for x in hold]
    return stat


if __name__ == "__main__":
    main()
