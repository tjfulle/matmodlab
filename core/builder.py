#!/usr/bin/env python
import os
import re
import sys
import glob
import shutil

from utils.misc import load_file, int2str
from utils.fortran.extbuilder import FortranExtBuilder
from __config__ import ROOT_D, PKG_D, SO_EXT, FIO, MTL_DB, cout


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
        self.fb.add_extension(ext, sources, requires_lapack=True)
        return

    def _add_mtls(self, mats_to_build):
        """Add fortran material models

        """
        cout("Gathering material[s] to be built")
        cout("{0} material[s] found: {1}".format(
                int2str(len(MTL_DB), c=True),
                ", ".join(x for x in MTL_DB.materials)))

        if mats_to_build == "all":
            mats_to_build = [m.name for m in MTL_DB]

        for material in MTL_DB:
            if material.name not in mats_to_build:
                continue
            if material.python_model:
                continue

            # assume fortran model if source files are given
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
