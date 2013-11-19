#!/usr/bin/env python
import os
import re
import sys
import glob
import shutil
import tempfile
from subprocess import call, STDOUT
from argparse import ArgumentParser, SUPPRESS

# distutils
from numpy.distutils.misc_util import Configuration
from numpy.distutils.system_info import get_info
from numpy.distutils.core import setup

from utils.impmod import load_file
from utils.int2str import int2str
from materials.material import write_mtldb
from __config__ import SPLASH

UMATS = [d for d in os.getenv("MMLMTLS", "").split(os.pathsep) if d.split()]
D = os.path.dirname(os.path.realpath(__file__))
QUIET = False
PYEXE = os.path.realpath(sys.executable)
CORE = os.path.join(D, "core")
VIZD = os.path.join(D, "viz")
UTLD = os.path.join(D, "utils")
TOOLS = os.path.join(D, "toolset")
FIO = os.path.join(D, "utils/fortran/mmlfio.f90")
LIB = os.path.join(D, "lib")
BLD = os.path.join(D, "build")
PATH = os.getenv("PATH").split(os.pathsep)


def which(exe):
    """Python implementation of unix which command

    """
    for d in PATH:
        x = os.path.join(d, exe)
        if os.path.isfile(x):
            return x


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
    parser.add_argument("--fc", default=which("gfortran"),
        help="Path to fortran compiler [default: %(default)s]")
    parser.add_argument("-v", default=1, type=int,
        help="Verbosity [default: %(default)s]")
    parser.set_defaults(rebuild_tpl=False, rebuild_utl=False, rebuild_all=False)

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

    # --- 'tpl' subparser
    parser_tpl = subparsers.add_parser("build_tpl", help="Build only the TPLs")
    parser_tpl.add_argument("--rebuild", dest="rebuild_tpl",
        default=False, action="store_true",
        help="Rebuild TPLs [default: %(default)s]")

    # --- 'util' subparser
    parser_utl= subparsers.add_parser("build_utl",
        help="Build only the fortran utilities")
    parser_utl.add_argument("--rebuild", dest="rebuild_utl",
        default=False, action="store_true",
        help="Rebuild utilities [default: %(default)s]")

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
    os.chdir(D)

    if args.v == 0:
        QUIET = True

    cout(SPLASH)

    if command == "build_all" and args.rebuild_all:
        remove(BLD)
        os.makedirs(BLD)

    # --- check prerequisites
    if command == "build_all":
        check_prereqs()

    # python passes mutables by reference. The env dictionary initialized here
    # is updated in each function it is passed to below
    env = {}
    # --- build TPLs
    if command in ("build_tpl", "build_all"):
        build_tpls(args.fc, args.rebuild_tpl or args.rebuild_all, env)
        if command != "build_all":
            os.chdir(cwd)
            return 0

    # # base configuration
    config, lapack = base_configuration()

    if command in ("build_utl", "build_all"):
        utl_config(config, lapack, args.rebuild_utl or args.rebuild_all)

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

        mtlinfo = {}
        fflags = []
        mtl_config(config, lapack, mats_to_build, mtlinfo, fflags)

    skip_build = command != "build_all" and not mtlinfo
    if skip_build:
        cout("No materials to build")
    else:
        build_extension_modules(config, mtlinfo, env, args.fc, fflags, args.v)
        write_mtldb(mtlinfo, wipe)

    if command != "build_all":
        os.chdir(cwd)
        return 0


    # convert python lists in the env dict to : separated lists for the
    # shell
    for (k, v) in env.items():
        if not isinstance(v, (basestring, str)):
            v = os.pathsep.join(x for x in v if x.split())
        env[k] = v
    env["FC"] = args.fc

    # write executables in ./toolset
    write_executables(env)

    # test the build, explicitly using the env dict
    stat = test_build(env)
    if stat != 0:
        cout("\n{0}\n"
             "Problems encountered during test.  To diagnose, execute\n"
             "  % runtests {1} -kfast -j4\n"
             "{0}\n".format("*" * 78, os.path.join(D, "tests")))
    else:
        if TOOLS not in PATH:
            cout("To complete build, add {0} to the PATH "
                 "environment variable".format(TOOLS))
        else:
            cout("Build complete")

    os.chdir(cwd)
    return stat


def build_tpls(fc, rebuild, env):
    """Look for and build TPLs

    Parameters
    ----------
    fc : str
      path to fortran executable
    rebuild : bool
      rebuild the TPLs if already built
    env : dict
      dictionary of environment variables required by TPLs to be set when
      running any of the Material Model Laboratory executables.

    Notes
    -----
    env is modified in place and passed by reference

    """
    rel_d = lambda d: d.replace(D + os.path.sep, "")

    # look for TPLs
    errors = 0
    TPL = os.path.join(D, "tpl")
    cout("Building TPLs")
    for (dirname, dirs, files) in os.walk(TPL):

        if "tpl.py" not in files:
            continue

        cout("Building TPL in {0}".format(rel_d(dirname)), end="... ")

        # load the tpl file and execute its build_tpl command
        tpl = load_file(os.path.join(dirname, "tpl.py"))
        info = tpl.build_tpl(fc, rebuild)
        retcode = info.pop("retcode")

        if retcode == 0:
            cout("yes")
        elif retcode > 0:
            errors += 1
            cout("no")
        else:
            cout("previously built, --rebuild to rebuild")

        # add environment to env
        for (key, value) in info.items():
            env.setdefault(key, []).append(value)

    if errors:
        raise SystemExit("Failed to build TPLs")

    cout("TPLs built")

    return


def utl_config(config, lapack, rebuild):
    """Set up the fortran utilities distutils configuration

    Parameters
    ----------
    config : instance
      distutils configuration instance
    lapack : dict
      distutils lapack_opt information
    rebuild : bool
      rebuild if already built

    Notes
    -----
    the config object is modified in place and passed by reference

    """

    if os.path.isfile(os.path.join(LIB, "mmlabpack.so")) and not rebuild:
        return

    # fortran tools
    sources = [os.path.join(D, "utils/fortran/mmlabpack.f90"),
               os.path.join(D, "utils/fortran/dgpadm.f")]
    config.add_extension("mmlabpack", sources=sources, **lapack)
    return


def base_configuration():
    """Setup the base numpy distutils configuration, including setting sys.argv

    """
    config = Configuration("matmodlab", parent_package="", top_path="")
    lapack = get_info("lapack_opt", notfound_action=2)
    lapack.setdefault("extra_compile_args", []).extend(["-fPIC", "-shared"])

    return config, lapack


def build_extension_modules(config, mtlinfo, env, fc, fflags, verbosity):
    """Build all extension modules in config

    Parameters
    ----------
    config : instance
      distutils configuration instance
    mtlinf : dict
      information about each material model
    env : dict
      dictionary of environment variables needed for scripts
    verbosity : int
      level of verbosity

    Notes
    -----
    env is modified in place and passed by reference

    """

    if verbosity < 2:
        # redirect stderr and stdout
        sys.stdout = open(os.devnull, "w")
        sys.stderr = open(os.devnull, "a")

    # change sys.argv for distutils
    fexec = "--f77exec={0} --f90exec={0}".format(fc).split()
    if fflags:
        fflags = " ".join(fflags)
        fflags = "--f77flags='{0}' --f90flags='{0}'".format(fflags).split()
    sys.argv = ["./setup.py", "config_fc"] + fexec + fflags + ["build_ext", "-i"]

    # build the extension modules with distutils setup
    cout("Building extension module[s]", end="... ")
    setup(**config.todict())
    cout("done")
    sys.stdout, sys.stderr = sys.__stdout__, sys.__stderr__

    # move files
    cout("Staging extension module[s]")
    built = []
    for src in glob.glob("*.so"):
        dst = os.path.join(LIB, os.path.basename(src))
        built.append(os.path.basename(os.path.splitext(src)[0]))
        if os.path.isfile(dst):
            os.remove(dst)
        shutil.move(src, dst)

    for mtl in [x for x in mtlinfo if x not in built]:
        # remove from the mtlinfo list any modules with source files that
        # didn't finish building
        if mtlinfo[mtl]["source_files"]:
            cout("*** warning: {0}: failed to build".format(mtl))
            del mtlinfo[mtl]

    # build the PYTHONPATH environment variable
    pypath = [D]
    for (key, val) in mtlinfo.items():
        pypath.append(os.path.dirname(val["interface_file"]))
    env.setdefault("PYTHONPATH", []).extend(pypath)

    return


def mtl_config(config, lapack, mats_to_build, mtlinfo, fflags):
    """Set up the material model distutils configuration

    Parameters
    ----------
    config : instance
      distutils configuration instance
    lapack : dict
      distutils lapack_opt information
    mats_to_build : list or str
      list of materials to build, or 'all' if all materials are to be built
    mtlinfo : dict
      dict of information for each material
    fflags : list
      fortran compile flags

    Notes
    -----
    the config and mtlinfo objects are modified in place and passed by
    reference

    """
    build_all = mats_to_build == "all"

    # --- builtin materials are all described in the mmats file
    cout("Gathering material[s] to be built")
    mats = load_file(os.path.join(D, "materials/library/mmats.py"))
    cout("  built in materials", end="... ")
    builtin = []
    for (name, conf) in mats.conf().items():
        if not build_all and name not in mats_to_build:
            continue
        kwargs = dict(lapack)
        incd = conf.get("include_dir", os.path.dirname(conf["interface_file"]))
        kwargs.setdefault("include_dirs", []).append(incd)
        mtlinfo[name] = conf
        builtin.append(name)

        # add extension to configuration if there are source files to build
        sources = conf["source_files"]
        if sources:
            sources.append(FIO)
            config.add_extension(name, sources=sources, **kwargs)
    cout(",".join(builtin))

    # --- user materials
    umats = []
    if UMATS:
        cout("  user material[s]", end="... ")
    for dirname in UMATS:
        if not os.path.isdir(dirname):
            cout("  *** warning: {0}: no such directory".format(dirname))
            continue
        if "umat.py" not in os.listdir(dirname):
            cout("  *** warning: umat.py not found in {0}".format(dirname))
            continue
        filename = os.path.join(dirname, "umat.py")
        umat = load_file(filename)
        try:
            name, conf = umat.conf()
        except ValueError:
            cout("  ***error: {0}: failed to gather information".format(filename))
            continue
        except AttributeError:
            cout("  ***error: {0}: conf function not defined".format(filename))
            continue
        if not build_all and name not in mats_to_build:
            continue

        kwargs = dict(lapack)
        incd = conf.get("include_dir", os.path.dirname(conf["interface_file"]))
        kwargs.setdefault("include_dirs", []).append(incd)
        if conf.get("FFLAGS"):
            fflags.extend(conf.get("FFLAGS"))
        mtlinfo[name] = conf
        umats.append(name)
        sources = conf["source_files"]
        if sources:
            sources.append(FIO)
            config.add_extension(name, sources=sources, **kwargs)

        continue
    if umats:
        cout(",".join(umats))

    cout("{0} materials found: {1}".format(int2str(len(mtlinfo), c=True),
                                           ", ".join(x for x in mtlinfo)))
    return


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

    # --- 64 bit python?
    cout("Checking for 64 bit python", end="... ")
    if sys.maxsize < 2 ** 32:
        cout("no")
        errors += 1
        cerr("*** error: matmodlab requires 64 bit python (due to exowrap)")
    else: cout("yes")

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
    _write_exe("mml", os.path.join(VIZD, "main.py"), env)
    _write_exe("runtests", os.path.join(CORE, "test.py"), env)
    _write_exe("gmddump", os.path.join(UTLD, "exodump.py"), env)
    _write_exe("exdump", os.path.join(UTLD, "exodump.py"), env)
    _write_exe("mmv", os.path.join(VIZD, "plot2d.py"), env)
    _write_exe("exdiff", os.path.join(UTLD, "exodiff.py"), env)
    _write_exe("buildmtls", os.path.join(D, "setup.py"), env, parg="build_mtl")
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
    exe = os.path.join(TOOLS, name)
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
    for (dirname, dirs, files) in os.walk(D):
        d = os.path.basename(dirname)
        if d == ".git":
            del dirs[:]
            continue
        if d == "build" or d.startswith("TestResults."):
            remove(dirname)
            del dirs[:]
            continue
        [remove(os.path.join(dirname, f)) for f in files if f.endswith(exts)]
    cout("yes")


def test_build(env):
    """Test if the installation worked

    """
    # run tests in temporary directory
    d = tempfile.mkdtemp()
    os.chdir(d)

    # add . and ./toolset to path
    env["PATH"] = os.pathsep.join([D, TOOLS] + PATH)

    cout("Testing installation", end="... ")
    exe = os.path.join(D, "toolset/runtests")
    testd = os.path.join(D, "tests")
    cmd = "{0} {1} -kfast -j4".format(exe, testd)
    con = open(os.devnull, "w")
    stat = call(cmd.split(), env=env, stdout=con, stderr=STDOUT)
    if stat != 0: cout("fail")
    else: cout("pass")
    os.chdir(D)
    shutil.rmtree(d)
    return stat


def build_material(material, verbosity=0):
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
    cout("building {0}".format(material))
    hold = [x for x in sys.argv]
    cmd = "-v {0} build_mtl -m {1}".format(verbosity, material)
    stat = main(cmd.split())
    if stat == 0:
        cout("built {0}".format(material))
    sys.argv = [x for x in hold]
    return stat


if __name__ == "__main__":
    main()
