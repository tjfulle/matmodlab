#!/usr/bin/env python
import os
import sys
import glob
import shutil
import argparse

from matmodlab import ROOT_D, MML_IFILE
import utils.conlog as logger
from utils.misc import load_file, int2str
from utils.fortran.extbuilder import FortranExtBuilder
from utils.errors import DuplicateExtModule
from core.material import MATERIALS


class BuilderError(Exception):
    pass


class Builder(object):
    def __init__(self, name, fc=None, verbosity=1):
        self.fb = FortranExtBuilder(name, fc=fc, verbosity=verbosity)
        pass

    def build_materials(self, mats_to_build="all"):
        self.fetch_fort_libs_to_build(mats_to_fetch=mats_to_build)
        self._build_extension_modules()

    def build_utils(self):
        self.fetch_fort_libs_to_build(mats_to_fetch=None)
        self._build_extension_modules()

    def build_all(self, mats_to_build="all"):
        self.fetch_fort_libs_to_build(mats_to_fetch=mats_to_build)
        self._build_extension_modules()

    @staticmethod
    def build_umat(name, source_files, verbosity=0):
        fb = FortranExtBuilder(name, verbosity=verbosity)
        logger.write("building {0}".format(name))
        fb.add_extension(name, source_files, requires_lapack="lite")
        fb.build_extension_modules()
        pass

    @staticmethod
    def build_material(name, info=None, verbosity=0):
        """Build a single material

        Parameters
        ----------
        name : str
          The name of the material to build

        """
        fb = FortranExtBuilder(name, verbosity=verbosity)
        logger.write("building {0}".format(name))
        if info is None:
            info = MATERIALS[name]
        fb.add_extension(name, info["source_files"],
                         requires_lapack=info.get("requires_lapack"))
        fb.build_extension_modules(verbosity=verbosity)
        return

    def fetch_fort_libs_to_build(self, mats_to_fetch="all"):
        """Add the fortran utilities to items to be built

        """
        fort_libs = {}
        for (dirname, dirs, files) in os.walk(ROOT_D):
            if MML_IFILE not in files:
                continue
            libs = {}
            info = load_file(os.path.join(dirname, MML_IFILE))
            if hasattr(info, "fortran_libraries"):
                libs.update(info.fortran_libraries())

            if not libs:
                continue

            for name in libs:
                if name in fort_libs:
                    raise DuplicateExtModule(name)
                fort_libs.update({name: libs[name]})
            del sys.modules[os.path.splitext(MML_IFILE)[0]]

        if mats_to_fetch is not None:
            for name in MATERIALS:
                if name in fort_libs:
                    raise DuplicateExtModule(name)
                if mats_to_fetch != "all" and name not in mats_to_fetch:
                    continue
                if not MATERIALS[name].get("source_files"):
                    continue
                fort_libs.update({name: MATERIALS[name]})

        for ext in fort_libs:
            s = fort_libs[ext]["source_files"]
            l = fort_libs[ext].get("lapack", False)
            I = fort_libs[ext].get("include_dirs", [])
            self.fb.add_extension(ext, s, include_dirs=I, requires_lapack=l)

        return

    def _build_extension_modules(self):
        """Build the extension modules

        """
        self.fb.build_extension_modules()
        for ext in self.fb.exts_failed:
            logger.write("*** warning: {0}: failed to build".format(ext))


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    parser = argparse.ArgumentParser(prog="mmd")
    parser.add_argument("-v", default=1, type=int,
       help="Verbosity [default: %(default)s]")
    parser.add_argument("-w", action="store_true", default=False,
       help="Wipe before building [default: %(default)s]")
    parser.add_argument("-W", action="store_true", default=False,
       help="Wipe and exit [default: %(default)s]")
    parser.add_argument("-m", nargs="+",
       help="Materials to build [default: all]")
    parser.add_argument("-u", action="store_true", default=False,
       help="Build auxiliary support files only [default: all]")
    args = parser.parse_args(argv)

    builder = Builder("matmodlab", verbosity=args.v)
    if args.w or args.W:
        for f in glob.glob(os.path.join(PKG_D, "*.so")):
            os.remove(f)
        bld_d = os.path.join(PKG_D, "build")
        if os.path.isdir(bld_d):
            shutil.rmtree(bld_d)
        if args.W:
            return 0
        args.u = True

    if args.u and args.m:
        builder.build_all(mats_to_build=args.m)

    elif args.u:
        builder.build_utils()

    elif args.m:
        builder.build_materials(args.m)

    else:
        builder.build_all()

    return 0

if __name__ == "__main__":
    main()
