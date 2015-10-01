#!/usr/bin/env python
import os
import re
import sys
import glob
import shutil
import logging
import warnings
import subprocess

from os.path import isfile, realpath, dirname, join, splitext, basename, isdir

# distutils
import numpy as np
from numpy.distutils.misc_util import Configuration
from numpy.distutils.system_info import get_info
from numpy.distutils.core import setup

from .product import LAPACK, LAPACK_OBJ, MMLABPACK
from ..misc import remove, stdout_redirected, merged_stderr_stdout
from ...mml_siteenv import environ
from ...product import PKG_D, PYEXE
from ...materials.product import ABA_UTL

FORT_COMPILER = environ.fc

class ExtModuleNotBuilt(Exception): pass
class FortranNotFoundError(Exception): pass
class FortranExtBuilder(object):
    """Interface with numpy distutils to build fortran extension modules in
    place

    """
    def __init__(self, name, fc=None, verbosity=1):
        # find fortran compiler
        global FORT_COMPILER
        if fc is None:
            fc = environ.fc
        if not fc:
            raise FortranNotFoundError("no fortran compiler found")
        fc = realpath(fc)
        if not isfile(fc):
            raise FortranNotFoundError("{0}: fortran compiler "
                                       "not found".format(fc))

        self.fc = fc
        FORT_COMPILER = fc
        self.name = name
        self.chatty = verbosity > 4
        self.exts_built = []
        self.exts_failed = []
        self.exts_to_build = []
        self.ext_modules_built = False
        self._build_blas_lapack = False

    def add_extension(self, name, sources, **kwargs):
        """Add an extension module to build"""
        options = {}
        lapack = kwargs.get("lapack")
        mmlabpack = kwargs.get("mmlabpack")

        errors = 0
        for source in sources:
            if not os.path.isfile(source):
                message = '{0!r}: file not found'.format(source)
                logging.getLogger('matmodlab.mmd.builder').error(message)
                errors += 1
        if errors:
            raise ExtModuleNotBuilt

        if mmlabpack:
            lapack = "lite"
            sources = MMLABPACK + sources

        if ABA_UTL in sources:
            lapack = 'lite'

        if lapack:
            if lapack == "lite":
                self._build_blas_lapack = True
                options["extra_objects"] = [LAPACK_OBJ]
                options["extra_compile_args"] = ["-fPIC", "-shared"]
            else:
                lapack = self._find_lapack()
                if not lapack:
                    logging.getLogger('matmodlab.mmd.builder').warn(
                        '{0}: required lapack package '
                        'not found, skipping'.format(name))
                    return -1
                options.update(lapack)

        idirs = kwargs.get("include_dirs")
        if idirs:
            options["include_dirs"] = idirs

        # Explicitly add this python distributions lib directory. This
        # shouldn't be necessary, but on some RHEL systems I have found that
        # it is
        d = realpath(join(dirname(PYEXE), "../lib"))
        assert isdir(d)
        options["library_dirs"] = [d]

        self.exts_to_build.append((name, sources, options))
        return

    def build_extension_modules(self, verbosity=None):
        """Build all extension modules in config"""
        if not self.exts_to_build:
            return

        if verbosity is not None:
            chatty = verbosity > 4
        else:
            chatty = self.chatty

        to_build = [x[0] for x in self.exts_to_build]
        if self._build_blas_lapack and not isfile(LAPACK_OBJ):
            to_build.insert(0, "blas_lapack-lite")
        logging.getLogger('matmodlab.mmd.builder').info(
            'The following fortran extension modules will be built:\n'
            '    {0}'.format(','.join(to_build)))

        if self._build_blas_lapack:
            if not isfile(LAPACK_OBJ):
                stat = build_blas_lapack()
                if stat != 0:
                    logging.getLogger('matmodlab.mmd.builder').error(
                        'failed to build blas_lapack, dependent '
                        'libraries will not be importable')

        config = Configuration(self.name, parent_package="", top_path="",
                               package_path=PKG_D)
        for (name, sources, options) in self.exts_to_build:
            config.add_extension(name, sources=sources, **options)

        cwd = os.getcwd()
        os.chdir(PKG_D)

        # change sys.argv for distutils
        hold = [x for x in sys.argv]

        fexec = "--f77exec={0} --f90exec={0}".format(self.fc)
        argv = "./setup.py config_fc {0}".format(fexec).split()
        fflags = ["-Wno-unused-dummy-argument"]
        if environ.fflags:
            fflags.extend(environ.fflags)
        fflags = " ".join(fflags)
        fflags = "--f77flags='{0}' --f90flags='{0}'".format(fflags).split()
        argv.extend(fflags)
        argv.extend("build_ext -i".split())

        # build the extension modules with distutils setup
        logging.getLogger('matmodlab.mmd.builder').info(
            'building extension module[s]... ', extra={'continued':1})
        f = join(PKG_D, "build.log") if not chatty else sys.stdout
        failed = 0
        try:
            sys.argv = [x for x in argv]
            with stdout_redirected(to=f), merged_stderr_stdout():
                setup(**config.todict())
            logging.getLogger('matmodlab.mmd.builder').info('done')
        except:
            logging.getLogger('matmodlab.mmd.builder').error('failed')
            failed = 1
        finally:
            sys.stdout, sys.stderr = sys.__stdout__, sys.__stderr__
            sys.argv = [x for x in hold]

        # move files
        d = config.package_dir[config.name]
        for mod in glob.glob(d + "/*.so"):
            self.exts_built.append(module_name(mod))

        logging.getLogger('matmodlab.mmd.builder').info(
            'staging extension module[s]... ', extra={'continued':1})

        self.exts_failed = [n[0] for n in self.exts_to_build
                            if n[0] not in self.exts_built]
        self.ext_modules_built = True
        self.exts_to_build = []
        if self.exts_failed:
            logging.getLogger('matmodlab.mmd.builder').info('failed')
            raise ExtModuleNotBuilt("{0}: failed to build".format(
                    ", ".join(self.exts_failed)))
        else:
            logging.getLogger('matmodlab.mmd.builder').info('done')
        os.chdir(cwd)
        return

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
    return splitext(basename(filepath))[0]


def build_blas_lapack():
    """Build the blas_lapack-lite object

    """
    logging.getLogger('matmodlab.mmd.builder').info(
        'building blas_lapack-lite... ', extra={'continued':1})
    cmd = [FORT_COMPILER, "-fPIC", "-shared", "-O3", LAPACK, "-o" + LAPACK_OBJ]
    build = subprocess.Popen(cmd, stdout=open(os.devnull, "a"),
                             stderr=subprocess.STDOUT)
    build.wait()
    if build.returncode == 0:
        logging.getLogger('matmodlab.mmd.builder').info('done')
    else:
        logging.getLogger('matmodlab.mmd.builder').info('no')
    return build.returncode
