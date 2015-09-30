import re
import os
import logging
from ..materials.product import *
from ..utils import xpyclbr
from ..mml_siteenv import environ
from ..utils.errors import MatModLabError
from ..utils.misc import load_file, rand_id

class MaterialLoader:
    def __init__(self, std_libs, user_libs):
        self.std_libs = std_libs
        self.user_libs = user_libs

    def get(self, name, response=None):
        name = name.strip()
        i = is_user_model(name)
        if i:
            name = get_user_interface(name, response=response)
            if name is not None:
                return self.std_libs[name]
            return None
        for (k, m) in self.std_libs.items():
            if is_user_model(k):
                continue
            if name.lower() == k.strip().lower():
                return m

        return None

    def __getitem__(self, name):
        mat = self.get(name)
        if mat is None:
            raise KeyError(name)

    @classmethod
    def load_materials(cls):
        '''Find material models

        '''
        errors = []
        std_libs = {}
        rx = re.compile(r'(?:^|[\\b_\\.-])[Mm]at')
        a = ['MaterialModel']

        # gather and verify all files
        # go through each item in std_materials and generate a list of material
        # interface files. if item is a directory gather all files that match rx;
        # if it's a file, add it to the list of material files
        for item in environ.std_materials:
            if os.path.isfile(item):
                d, files = os.path.split(os.path.realpath(item))
                files = [files]
            elif os.path.isdir(item):
                d = item
                files = [f for f in os.listdir(item) if rx.search(f)]
            else:
                logging.warn('{0} no such directory or file, skipping'.format(d))
                continue
            files = [f for f in files if f.endswith('.py')]

            if not files:
                logging.warn('{0}: no mat files found'.format(d))

            # go through files and determine if it's an interface file. if it is,
            # load it and add it to std_libs
            for f in files:
                module = f[:-3]
                try:
                    libs = xpyclbr.readmodule(module, [d], ancestors=a)
                except AttributeError as e:
                    errors.append(e.args[0])
                    logging.error(e.args[0])
                    continue
                for lib in libs:
                    if lib in std_libs:
                        logging.error('{0}: duplicate material'.format(lib))
                        errors.append(lib)
                        continue
                    module = load_file(libs[lib].file)
                    mat_class = getattr(module, libs[lib].class_name, None)
                    if mat_class.name is None:
                        raise MatModLabError('{0}: material name attribute '
                                             'not defined'.format(lib))
                    libs[lib].mat_class = mat_class
                    std_libs.update({mat_class.name: libs[lib]})

        # load materials in the materials dict
        user_libs = {}
        for (name, info) in environ.materials.items():
            if name in user_libs:
                logging.error('{0}: duplicate material'.format(name))
                errors.append(name)

            source_files = info.get('source_files')
            if not source_files:
                raise MatModLabError('{0}: requires source_files'.format(name))

            model = info.get('model', USER)
            response = info.get('response')
            user_i = get_user_interface(model, response=response)
            if user_i is None:
                raise MatModLabError('non-user material in materials dict')
            ordering = get_default_ordering(model)

            mat = UserMaterial(std_libs[user_i],
                               info.get('libname', user_i+rand_id(4)),
                               source_files, info.get('param_names'),
                               info.get('depvar'), info.get('user_ics', 0),
                               info.get('ordering', ordering), info.get('builtin'))
            user_libs[name] = mat

        if errors:
            raise MatModLabError('duplicate extension modules: '
                                 '{0}'.format(', '.join(errors)))

        return cls(std_libs, user_libs)
