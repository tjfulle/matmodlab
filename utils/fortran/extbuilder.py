#!/usr/bin/env python
import os
import re
import sys
import glob
import shutil
import warnings

# distutils
from numpy.distutils.misc_util import Configuration
from numpy.distutils.system_info import get_info
from numpy.distutils.core import setup

from __config__ import FFLAGS, PKG_D, PATH, cout

D = os.path.dirname(os.path.realpath(__file__))


class ExtModuleNotBuilt(Exception): pass
class FortranNotFoundError(Exception): pass
class FortranExtBuilder(object):
    """Interface with numpy distutils to build fortran extension modules in
    place

    """
    def __init__(self, name, fc=None, verbosity=1):
        # find fortran compiler
        if fc is None:
            fc = which(os.getenv("FC", "gfortran"))
        if not fc:
            raise FortranNotFoundError("no fortran compiler found")
        fc = os.path.realpath(fc)
        if not os.path.isfile(fc):
            raise FortranNotFoundError("{0}: fortran compiler "
                                       "not found".format(fc))

        self.fc = fc
        self.config = Configuration(name, parent_package="", top_path="",
                                    package_path=PKG_D)
        self.quiet = verbosity < 2
        self.silent = verbosity < 1
        self.exts_built = []
        self.exts_failed = []
        self.exts_to_build = []
        self.ext_modules_built = False

    def add_extension(self, name, sources, **kwargs):
        """Add an extension module to build"""
        options = {}
        if kwargs.get("requires_lapack"):
            lapack = self._find_lapack()
            if not lapack:
                cout("*** warning: {0}: required lapack package "
                     "not found, skipping".format(name))
                return -1
            options.update(lapack)
        idirs = kwargs.get("include_dirs")
        if idirs:
            options["include_dirs"] = idirs
        self.config.add_extension(name, sources=sources, **options)
        self.exts_to_build.append(name)
        return

    def build_extension_modules(self, verbosity=1):
        """Build all extension modules in config"""
        if not self.exts_to_build:
            return
        if self.quiet:
            # redirect stderr and stdout
            sys.stdout = open(os.devnull, "w")
            sys.stderr = open(os.devnull, "a")

        cwd = os.getcwd()
        os.chdir(PKG_D)
        # change sys.argv for distutils
        hold = [x for x in sys.argv]
        fexec = "--f77exec={0} --f90exec={0}".format(self.fc)
        sys.argv = "./setup.py config_fc {0}".format(fexec).split()
        if FFLAGS:
            fflags = " ".join(FFLAGS)
            fflags = "--f77flags='{0}' --f90flags='{0}'".format(fflags).split()
            sys.argv.extend(fflags)
        sys.argv.extend("build_ext -i".split())

        # build the extension modules with distutils setup
        self.logmes("Building extension module[s]", end="... ")
        setup(**self.config.todict())
        self.logmes("done")
        sys.stdout, sys.stderr = sys.__stdout__, sys.__stderr__
        sys.argv = hold

        # move files
        self.logmes("Staging extension module[s]", end="... ")
        d = self.config.package_dir[self.config.name]
        for mod in glob.glob(d + "/*.so"):
            self.exts_built.append(module_name(mod))
        self.exts_failed = [n for n in self.exts_to_build
                            if n not in self.exts_built]
        self.ext_modules_built = True
        self.exts_to_build = []
        self.logmes("done")
        if self.exts_failed:
            raise ExtModuleNotBuilt("{0}: failed to build".format(
                    ", ".join(self.exts_failed)))
        os.chdir(cwd)
        return

    def logmes(self, message, end="\n"):
        """Write message to stdout """
        if not self.silent:
            cout(message, end=end)

    @staticmethod
    def _find_lapack():
        """Setup the base numpy distutils configuration"""
        warnings.simplefilter("ignore")
        for item in ("lapack_opt", "lapack_mkl", "lapack"):
            lapack = get_info(item, notfound_action=0)
            if lapack:
                lapack.setdefault("extra_compile_args", []).extend(
                    ["-fPIC", "-shared"])
                return lapack


def module_name(filepath):
    return os.path.splitext(os.path.basename(filepath))[0]


def which(exe):
    """Python implementation of unix which command

    """
    if os.path.isfile(exe):
        return exe
    for d in PATH:
        x = os.path.join(d, exe)
        if os.path.isfile(x):
            return x
