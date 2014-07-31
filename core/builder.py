#!/usr/bin/env python
import os
import sys
import glob
import shutil
import argparse

from utils.misc import load_file, int2str
from utils.fortran.extbuilder import FortranExtBuilder
import __config__ as cfg


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
        if not cfg.MTL_DB:
            return []
        return cfg.MTL_DB.path

    @staticmethod
    def build_umat(source_files, verbosity=0):
        name = "umat"
        fb = FortranExtBuilder(name, verbosity=verbosity)
        cfg.cout("building {0}".format(name))
        source_files += [cfg.ABQIO, cfg.ABQUMAT]
        fb.add_extension(name, source_files)
        fb.build_extension_modules()
        pass

    @staticmethod
    def build_material(material, verbosity=0):
        """Build a single material

        Parameters
        ----------
        material : str
          The name of the material to build

        """
        material = cfg.MTL_DB[material]  #.material_from_name(material)
        fb = FortranExtBuilder(material.name, verbosity=verbosity)
        cfg.cout("building {0}".format(material.name))
        fb.add_extension(material.name, material.source_files,
                         requires_lapack=material.requires_lapack)
        fb.build_extension_modules(verbosity=verbosity)
        return

    def _add_utils(self):
        """Add the fortran utilities to items to be built

        """
        ext = "mmlabpack"
        sources = [os.path.join(cfg.ROOT_D, "utils/fortran/mmlabpack.f90"),
                   os.path.join(cfg.ROOT_D, "utils/fortran/dgpadm.f")]
        self.fb.add_extension(ext, sources, requires_lapack="lite")
        return

    def _add_mtls(self, mats_to_build):
        """Add fortran material models

        """
        if mats_to_build == "all":
            mats_to_build = [m.name for m in cfg.MTL_DB]

        cfg.cout("Material[s] to be built: {0}".format(", ".join(mats_to_build)))

        for material in cfg.MTL_DB:

            if material.name not in mats_to_build:
                continue

            if material.python_model:
                continue

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
                cfg.MTL_DB.remove(material)

        return

    def _build_extension_modules(self):
        """Build the extension modules

        """
        self.fb.build_extension_modules()
        for ext in self.fb.exts_failed:
            cfg.cout("*** warning: {0}: failed to build".format(ext))


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
        for f in glob.glob(os.path.join(cfg.PKG_D, "*.so")):
            os.remove(f)
        bld_d = os.path.join(cfg.PKG_D, "build")
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
