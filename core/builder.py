#!/usr/bin/env python
import os
import sys
import glob
import shutil
import argparse

from utils.misc import load_file, int2str
from utils.fortran.extbuilder import FortranExtBuilder
from materials.materialdb import _Material
from __config__ import ROOT_D, PKG_D, SO_EXT, FIO, ABQIO, MTL_DB, cout


class BuilderError(Exception):
    pass


class Builder(object):
    def __init__(self, name, fc=None, verbosity=1):
        self.fb = FortranExtBuilder(name, fc=fc, verbosity=verbosity)
        pass

    def build_materials(self, mats_to_build="all"):
        self._add_mtls(mats_to_build)
        self._build_extension_modules()

    def build_utils(self):
        self._add_utils()
        self._build_extension_modules()

    def build_all(self, mats_to_build="all"):
        self._add_utils()
        self._add_mtls(mats_to_build)
        self._build_extension_modules()

    @property
    def path(self):
        if not MTL_DB:
            return []
        return MTL_DB.path


    @staticmethod
    def build_material(material, verbosity=0):
        """Build a single material

        Parameters
        ----------
        material : str
          The name of the material to build

        """
        if not isinstance(material, _Material):
            material = MTL_DB.material_from_name(material)

        fb = FortranExtBuilder(material.name, verbosity=verbosity)
        cout("building {0}".format(material.name))
        material.source_files.append(FIO)
        fb.add_extension(material.name, material.source_files,
                         requires_lapack=material.requires_lapack)
        fb.build_extension_modules()
        return

    def _add_utils(self):
        """Add the fortran utilities to items to be built

        """
        ext = "mmlabpack"
        sources = [os.path.join(ROOT_D, "utils/fortran/mmlabpack.f90"),
                   os.path.join(ROOT_D, "utils/fortran/dgpadm.f")]
        self.fb.add_extension(ext, sources, requires_lapack="lite")
        return

    def _add_mtls(self, mats_to_build):
        """Add fortran material models

        """
        if mats_to_build == "all":
            mats_to_build = [m.name for m in MTL_DB]

        cout("Material[s] to be built: {0}".format(", ".join(mats_to_build)))

        for material in MTL_DB:
            if material.name not in mats_to_build:
                continue
            if material.python_model:
                continue

            # assume fortran model if source files are given
            if material.abaqus_umat:
                material.source_files.append(ABQIO)
            else:
                material.source_files.append(FIO)
            include_dirs = [material.dirname]
            d = material.include_dir
            if d and d not in include_dirs:
                include_dirs.append(d)
            stat = self.fb.add_extension(
                material.name, material.source_files,
                include_dirs=include_dirs,
                requires_lapack=material.requires_lapack)
            if stat:
                # failed to add extension
                MTL_DB.remove(material)

        return

    def _build_extension_modules(self):
        """Build the extension modules

        """
        self.fb.build_extension_modules()
        for ext in self.fb.exts_failed:
            cout("*** warning: {0}: failed to build".format(ext))


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

    builder = Builder("matmodlab")
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
