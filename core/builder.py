#!/usr/bin/env python
import os
import sys
import glob
import shutil
import argparse
import importlib

from utils.fortran.product import FIO
from materials.product import ABA_MATS
from utils.misc import load_file, remove
from core.product import ROOT_D, F_PRODUCT, PKG_D
from utils.errors import DuplicateExtModule
from core.logger import ConsoleLogger as logger
from utils.fortran.extbuilder import FortranExtBuilder


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
    def build_material(name, source_files, verbosity=0, lapack=False):
        """Build a single material

        Parameters
        ----------
        name : str
          The name of the material to build

        """
        fb = FortranExtBuilder(name, verbosity=verbosity)
        logger.write("building {0}".format(name))
        if name not in ABA_MATS:
            source_files.append(FIO)
        fb.add_extension(name, source_files, lapack=lapack)
        fb.build_extension_modules(verbosity=verbosity)
        return

    def fetch_fort_libs_to_build(self, mats_to_fetch="all"):
        """Add the fortran utilities to items to be built

        """
        fort_libs = {}
        module = os.path.splitext(F_PRODUCT)[0]
        for (dirname, dirs, files) in os.walk(ROOT_D):
            if F_PRODUCT not in files:
                continue
            libs = {}
            info = load_file(os.path.join(dirname, F_PRODUCT))
            if hasattr(info, "fortran_libraries"):
                libs.update(info.fortran_libraries())

            if not libs:
                continue

            for name in libs:
                if name in fort_libs:
                    raise DuplicateExtModule(name)
                fort_libs.update({name: libs[name]})

        if mats_to_fetch is not None:
            mats_fetched = []
            # find materials and filter out those to build
            from core.material import find_materials
            for (name, info) in find_materials().items():
                if name in fort_libs:
                    raise DuplicateExtModule(name)
                if mats_to_fetch != "all" and name not in mats_to_fetch:
                    continue
                mats_fetched.append(name)
                material = info.mat_class
                if material.source_files is None:
                    continue
                d = os.path.dirname(info.file)
                source_files = [x for x in material.source_files]
                if name not in ABA_MATS:
                    source_files.append(FIO)
                I = getattr(material, "include_dirs", [d])
                fort_libs.update({name: {"source_files": source_files,
                                         "lapack": material.lapack,
                                         "include_dirs": I}})

        if mats_to_fetch != "all" and mats_to_fetch is not None:
            for mat in mats_to_fetch:
                if mat not in mats_fetched:
                    logger.error("{0}: material not found".format(mat))
            if logger.errors:
                logger.raise_error("stopping due to previous errors")

        for ext in fort_libs:
            s = fort_libs[ext]["source_files"]
            l = fort_libs[ext].get("lapack", False)
            I = fort_libs[ext].get("include_dirs", [])
            m = fort_libs[ext].get("mmlabpack", False)
            self.fb.add_extension(ext, s, include_dirs=I, lapack=l, mmlabpack=m)

        return

    def _build_extension_modules(self):
        """Build the extension modules

        """
        self.fb.build_extension_modules()
        for ext in self.fb.exts_failed:
            logger.warn("{0}: failed to build".format(ext))


def wipe_built_libs():
    for f in glob.glob(os.path.join(PKG_D, "*.so")):
        remove(f)
    for f in glob.glob(os.path.join(PKG_D, "*.o")):
        remove(f)
    for f in glob.glob(os.path.join(PKG_D, "*.pyc")):
        remove(f)
    bld_d = os.path.join(PKG_D, "build")
    remove(bld_d)


def build(what_to_build, wipe_and_build=False, verbosity=1):

    builder = Builder("matmodlab", verbosity=verbosity)

    if wipe_and_build:
        wipe_built_libs()

    what, more = what_to_build[:2]
    recognized = ("utils", "all", "material")
    if what not in recognized:
        raise SystemExit("builder.build: expected what_to_build[0] to be one "
                         "of {0}, got {1}".format(", ".join(recognized), what))

    if what == "utils":
        builder.build_utils()

    elif what == "all":
        builder.build_all()

    elif what == "material":
        builder.build_materials(more)

    return 0

def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    parser = argparse.ArgumentParser(prog="mml build",
                description="%(prog)s: build fortran utilities and materials.")
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

    if args.W:
        sys.exit(wipe_built_libs())

    if args.u and args.m:
        sys.exit("***error: mml build: -m and -u are mutually exclusive options")

    what_to_build = ["all", None]
    if args.u:
        what_to_build[0] = "utils"
    elif args.m:
        what_to_build = ("material", args.m)

    return build(what_to_build, wipe_and_build=args.w, verbosity=args.v)

if __name__ == "__main__":
    main()
