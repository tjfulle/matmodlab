import os
import logging
from os.path import join
from matmodlab.constants import *
from matmodlab.utils.fortran.product import DGPADM_F, IO_F90
from matmodlab.product import MAT_D
D = join(MAT_D, 'src')

# Auxiliary files
TENSALG_F90 = join(D, 'tensalg.f90')
SDVINI = join(D, 'sdvini.f90')

ABA_UANISOHYPER_JAC_F90 = join(D, 'uanisohyper_inv_jac.f90')
ABA_UHYPER_JAC_F90 = join(D, 'uhyper_jac.f90')

# Signature files
ABA_UANISOHYPER_PYF = join(D, 'uanisohyper_inv.pyf')
ABA_UHYPER_PYF = join(D, 'uhyper.pyf')
ABA_UMAT_PYF = join(D, 'umat.pyf')

# standar abaqus include
ABA_UTL = join(D, 'abaqus.f90')

class UserMaterial:
    def __init__(self, mat_info, libname, source_files, param_names,
                 depvar, user_ics, ordering, builtin):
        self.libname = libname
        self.mat_class = mat_info.mat_class
        self.source_files = source_files
        self.source_files.extend(self.mat_class.aux_files())
        if not user_ics:
            self.source_files.append(SDVINI)
        self.param_names = param_names
        self.depvar = depvar
        self.ordering = ordering
        self.file = mat_info.file
        self.builtin = bool(builtin)

        errors = 0
        for (i, f) in enumerate(self.source_files):
            filename = os.path.realpath(f)
            if not os.path.isfile(filename):
                errors += 1
                logging.error('{0}: file not found'.format(f))
            self.source_files[i] = filename

        if errors:
            raise ValueError('stopping due to previous errors')

def is_user_model(model):
    if model not in (USER, UMAT, UHYPER, UANISOHYPER_INV):
        return 0
    if model in (UMAT, UHYPER, UANISOHYPER_INV):
        return 2
    return 1

def get_default_ordering(model):
    if model in (UMAT, UHYPER, UANISOHYPER_INV):
        return [XX, YY, ZZ, XY, XZ, YZ]
    return [XX, YY, ZZ, XY, YZ, XZ]

def get_user_interface(name, response=None):
    i = is_user_model(name)
    if not i or i == 2:
        return name
    if i:
        if response is None or response == MECHANICAL:
            name = UMAT
        elif response == HYPERELASTIC:
            name = UHYPER
        elif response == ANISOHYPER:
            name = UANISOHYPER_INV
        return name
    return None


def fortran_libraries_x():
    # expansion and viscoelastic models now fortran
    libs = {}

    visco_f90 = join(D, 'visco.f90')
    visco_pyf = join(D, 'visco.pyf')
    libs['visco'] = {'source_files': [visco_f90, visco_pyf, IO_F90]}

    expansion_f90 = join(D, 'expansion.f90')
    expansion_pyf = join(D, 'expansion.pyf')
    libs['expansion'] = {'source_files': [expansion_f90, expansion_pyf],
                         'mmlabpack': True}

    return libs
