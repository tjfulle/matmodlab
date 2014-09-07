#!/usr/bin/env python
import os
import sys
import glob
import shutil
import argparse

from utils.fortran.product import FIO
from materials.product import ABA_MATS
from utils.misc import load_file, int2str
from core.product import ROOT_D, F_PRODUCT
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
            del sys.modules[os.path.splitext(F_PRODUCT)[0]]

        if mats_to_fetch is not None:
            # find materials and filter out those to build
            from core.material import find_materials
            for (name, info) in find_materials().items():
                if name in fort_libs:
                    raise DuplicateExtModule(name)
                if mats_to_fetch != "all" and name not in mats_to_fetch:
                    continue
                # meta information, such as source files, are stored as
                # instance attributes of the material. Load it to get the
                # info.
                loaded = load_file(info.file)
                material = getattr(loaded, info.class_name)()
                if not hasattr(material, "source_files"):
                    del sys.modules[info.module]
                    continue
                d = os.path.dirname(info.file)
                source_files = material.source_files
                if name not in ABA_MATS:
                    source_files.append(FIO)
                l = getattr(material, "lapack", None)
                I = getattr(material, "include_dirs", [d])
                fort_libs.update({name: {"source_files": source_files,
                                         "lapack": l, "include_dirs": I}})
                del sys.modules[info.module]

        for ext in fort_libs:
            s = fort_libs[ext]["source_files"]
            l = fort_libs[ext].get("lapack", False)
            I = fort_libs[ext].get("include_dirs", [])
            self.fb.add_extension(ext, s, include_dirs=I, lapack=l)

        return

    def _build_extension_modules(self):
        """Build the extension modules

        """
        self.fb.build_extension_modules()
        for ext in self.fb.exts_failed:
            logger.write("*** warning: {0}: failed to build".format(ext))


def build(wipe=False, wipe_and_build=False, mat_to_build=None,
          build_utils=False, verbosity=1):

    builder = Builder("matmodlab", verbosity=verbosity)

    if wipe or wipe_and_build:
        for f in glob.glob(os.path.join(PKG_D, "*.so")):
            os.remove(f)
        bld_d = os.path.join(PKG_D, "build")
        if os.path.isdir(bld_d):
            shutil.rmtree(bld_d)
        if wipe:
            return 0
        build_utils = True

    if build_utils and mat_to_build:
        builder.build_all(mats_to_build=mat_to_build)

    elif build_utils:
        builder.build_utils()

    elif mat_to_build:
        builder.build_materials(mat_to_build)

    else:
        builder.build_all()

    return 0

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

    return build(wipe=args.W, wipe_and_build=args.w, mat_to_build=args.m,
                 build_utils=args.u, verbosity=args.v)

if __name__ == "__main__":
    main()
