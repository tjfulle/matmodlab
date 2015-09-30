#!/usr/bin/env python
import os
import re
import sys
import glob
import shutil
import logging
import argparse
import importlib

from ..constants import *
from ..materials.product import *
from ..utils.misc import load_file, remove
from ..product import ROOT_D, PKG_D
from ..utils.errors import MatModLabError
from ..utils.fortran.extbuilder import FortranExtBuilder
from ..utils.fortran.product import IO_F90
from ..utils.logio import setup_logger

logger = setup_logger('matmodlab.mmd.builder')

class Builder(object):
    def __init__(self, name, fc=None, verbosity=1):
        self.fb = FortranExtBuilder(name, fc=fc, verbosity=verbosity)
        pass

    def build_materials(self, mats_to_build='all'):
        self.fetch_fort_libs_to_build(mats_to_fetch=mats_to_build)
        self._build_extension_modules()

    def build_utils(self):
        self.fetch_fort_libs_to_build(mats_to_fetch=None)
        self._build_extension_modules()

    def build_all(self, mats_to_build='all', user_env=0):
        self.fetch_fort_libs_to_build(mats_to_fetch=mats_to_build,
                                      user_env=user_env)
        self._build_extension_modules()

    @staticmethod
    def build_material(name, source_files, verbosity=0, lapack=False):
        '''Build a single material

        Parameters
        ----------
        name : str
          The name of the material to build

        '''
        fb = FortranExtBuilder(name, verbosity=verbosity)
        logger.info('building {0}'.format(name))
        source_files.append(IO_F90)
        fb.add_extension(name, source_files, lapack=lapack)
        fb.build_extension_modules(verbosity=verbosity)
        return

    def fetch_fort_libs_to_build(self, mats_to_fetch='all', user_env=0, force=0):
        '''Add the fortran utilities to items to be built

        '''
        fort_libs = {}
        module = os.path.splitext('product.py')[0]
        if user_env <= 1:
            for (dirname, dirs, files) in os.walk(ROOT_D):
                if 'product.py' not in files:
                    continue
                libs = {}
                info = load_file(os.path.join(dirname, 'product.py'))
                if hasattr(info, 'fortran_libraries'):
                    libs.update(info.fortran_libraries())

                if not libs:
                    continue

                for name in libs:
                    if name in fort_libs:
                        raise MatModLabError('duplicate extension '
                                             'module {0}'.format(name))
                    if (os.path.isfile(os.path.join(PKG_D, name + '.so'))
                        and not force):
                        continue
                    fort_libs.update({name: libs[name]})

        from .loader import MaterialLoader
        all_mats = MaterialLoader.load_materials()
        if user_env <= 1 and mats_to_fetch is not None:
            mats_fetched = []
            # find materials and filter out those to build
            for (name, info) in all_mats.std_libs.items():
                if name in fort_libs:
                    raise MatModLabError('duplicate extension '
                                         'module {0}'.format(name))
                if mats_to_fetch != 'all' and name not in mats_to_fetch:
                    continue
                mats_fetched.append(name)
                if isinstance(info, UserMaterial):
                    continue
                material = info.mat_class
                if not material.source_files():
                    continue
                d = os.path.dirname(info.file)
                source_files = [x for x in material.source_files()]
                source_files.append(IO_F90)
                I = getattr(material, 'include_dirs', [d])
                fort_libs.update({name: {'source_files': source_files,
                                         'lapack': material.lapack,
                                         'include_dirs': I}})

        # user materials
        for (name, mat) in all_mats.user_libs.items():

            if user_env >= 2 and mat.builtin:
                continue

            if mats_to_fetch != 'all' and name not in mats_to_fetch:
                continue
            mats_fetched.append(name)

            if mat.libname is not None:
                libname = mat.libname
                # need to build with new libname
                for (i, f) in enumerate(mat.source_files):
                    if f.endswith('.pyf'):
                        signature = f
                        break
                else:
                    raise MatModLabError('signature file not found')
                lines = open(signature, 'r').read()
                new_signature = os.path.join(PKG_D, libname + '.pyf')
                libname_ = getattr(mat.mat_class, 'libname', mat.mat_class.name)
                pat = r'(?is)python\s+module\s+{0}'.format(libname_)
                repl = r'python module {0}'.format(libname)
                lines = re.sub(pat, repl, lines)
                with open(new_signature, 'w') as fh:
                    fh.write(lines)
                mat.source_files[i] = new_signature
            else:
                libname = getattr(mat.mat_class, 'libname', mat.mat_class.name)

            I = list(set([os.path.dirname(f) for f in mat.source_files]))
            if IO_F90 not in mat.source_files:
                mat.source_files.append(IO_F90)

            for f in mat.source_files:
                if not os.path.isfile(f):
                    raise MatModLabError('{0!r}: no such file'.format(f))

            fort_libs.update({libname: {'source_files': mat.source_files,
                                        'lapack': 'lite',
                                        'include_dirs': I}})

        errors = 0
        if mats_to_fetch != 'all' and mats_to_fetch is not None:
            for mat in mats_to_fetch:
                if mat not in mats_fetched:
                    errors += 1
                    logger.error('{0}: material not found'.format(mat))
            if errors:
                raise ValueError('stopping due to previous errors')

        for ext in fort_libs:
            s = fort_libs[ext]['source_files']
            l = fort_libs[ext].get('lapack', False)
            I = fort_libs[ext].get('include_dirs', [])
            m = fort_libs[ext].get('mmlabpack', False)
            self.fb.add_extension(ext, s, include_dirs=I, lapack=l, mmlabpack=m)

        return

    def _build_extension_modules(self):
        '''Build the extension modules

        '''
        self.fb.build_extension_modules()
        for ext in self.fb.exts_failed:
            logger.warn('{0}: failed to build'.format(ext))

def wipe_built_libs():
    for f in glob.glob(os.path.join(PKG_D, '*.so')):
        remove(f)
    for f in glob.glob(os.path.join(PKG_D, '*.o')):
        remove(f)
    for f in glob.glob(os.path.join(PKG_D, '*.pyc')):
        remove(f)
    bld_d = os.path.join(PKG_D, 'build')
    remove(bld_d)

def build(what_to_build, wipe_and_build=False, verbosity=1, user_env=0):

    builder = Builder('matmodlab', verbosity=verbosity)

    if wipe_and_build:
        wipe_built_libs()

    what, more = what_to_build[:2]
    recognized = ('utils', 'all', 'material')
    if what not in recognized:
        raise SystemExit('builder.build: expected what_to_build[0] to be one '
                         'of {0}, got {1}'.format(', '.join(recognized), what))

    if what == 'utils':
        builder.build_utils()

    elif what == 'all':
        builder.build_all(user_env=user_env)

    elif what == 'material':
        builder.build_materials(more)

    return 0

def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    parser = argparse.ArgumentParser(prog='mml build',
                description='%(prog)s: build fortran utilities and materials.')
    parser.add_argument('-v', default=1, type=int,
       help='Verbosity [default: %(default)s]')
    parser.add_argument('-w', action='store_true', default=False,
       help='Wipe before building [default: %(default)s]')
    parser.add_argument('-W', action='store_true', default=False,
       help='Wipe and exit [default: %(default)s]')
    parser.add_argument('-m', nargs='+',
       help='Materials to build [default: all]')
    parser.add_argument('-u', action='store_true', default=False,
       help='Build auxiliary support files only [default: all]')
    parser.add_argument('-e', nargs='?', default=0, const=1, type=int,
       help='Build materials in user environment file [default: all]')
    args = parser.parse_args(argv)

    if args.W:
        sys.exit(wipe_built_libs())

    if args.u and args.m:
        sys.exit('***error: mml build: -m and -u are mutually exclusive options')

    what_to_build = ['all', None]
    if args.u:
        what_to_build[0] = 'utils'
    elif args.m:
        what_to_build = ('material', args.m)

    return build(what_to_build, wipe_and_build=args.w,
                 verbosity=args.v, user_env=args.e)

if __name__ == '__main__':
    main()
