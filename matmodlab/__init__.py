import os
import sys
import inspect
import logging
import warnings
from math import *
from product import *
from StringIO import StringIO

from os.path import realpath, isfile, isdir, join, splitext, dirname, basename

__version__ = "3.0.5"

errors = []
(major, minor, micro, relev, ser) = sys.version_info
if (major != 3 and major != 2) or (major == 2 and minor < 7):
    errors.append('python >= 2.7 required')
    errors.append('  {0} provides {1}.{2}.{3}'.format(
        sys.executable, major, minor, micro))

try:
    from traits.etsconfig.api import ETSConfig
    toolkit = os.getenv('ETS_TOOLKIT', 'qt4')
    ETSConfig.toolkit = toolkit
    os.environ['ETS_TOOLKIT'] = toolkit
except ImportError:
    pass

# --- numpy
try: import numpy as np
except ImportError: errors.append('numpy not found')

# --- scipy
try: import scipy
except ImportError: errors.append('scipy not found')

# check prerequisites
if errors:
    raise SystemExit('*** error: matmodlab could not run due to the '
                     'following errors:\n  {0}'.format('\n  '.join(errors)))

# --- ADD CWD TO sys.path
sys.path.insert(0, os.getcwd())

# ------------------------ FACTORY METHODS TO SET UP AND RUN A SIMULATION --- #
from numpy import array, float64
from .mmd.simulator import *
from .mml_siteenv import environ
from .mmd.material import build_material
from .mmd.permutator import Permutator, PermutateVariable
from .mmd.optimizer import Optimizer, OptimizeVariable
from .constants import *
from .materials.product import *
from .utils.elas import elas
RAND = np.random.RandomState()

def genrand():
    return RAND.random_sample()
randreal = genrand()

def requires(major, minor, micro=None):
    M, m, _m = VERSION
    if M != major and m != minor:
        raise SystemExit('input requires matmodlab version '
                         '{0}.{1}'.format(major, minor))

def matmodlab(func):
    warnings.warn('deprecated', DeprecationWarning)

def gen_runid():
    stack = inspect.stack()[1]
    return splitext(basename(stack[1]))[0]

def get_my_directory():
    '''return the directory of the calling function'''
    stack = inspect.stack()[1]
    d = dirname(realpath(stack[1]))
    return d

def init_from_matmodlab_magic(p):
    if p == BOKEH:
        from bokeh.plotting import output_notebook
        output_notebook()
        environ.plotter = BOKEH
        i = 2
    elif p == MATPLOTLIB:
        environ.plotter = MATPLOTLIB
        i = 1

    environ.notebook = i
    environ.log_level = logging.WARNING
    try:
        from sympy import init_printing
        init_printing()
    except ImportError:
        pass

def load_interactive_material(std_material=None, user_material=None, **kwds):

    def isstr(s):
        try:
            s + ''
            return True
        except TypeError:
            return False

    if user_material is not None:
        name = kwds.get('name')
        if name is None:
            raise ValueError("interactive material is missing the 'name' keyword")
        if not os.path.isfile(user_material):
            raise IOError('{0:!r}: no such file'.format(user_material))
        exts = ('.f', 'F', '.f90', '.for', '.F90', '.FOR')
        if not user_material.endswith(exts):
            exts = ', '.join(exts)
            raise ValueError('expected extension to be one of {0}'.format(exts))

        d = {}
        d['filename'] = user_material
        d['model'] = kwds.get('model', UMAT)
        d['response'] = kwds.get('response', MECHANICAL)
        environ.interactive_usr_materials[name] = d

        root = os.path.splitext(os.path.basename(user_material))[0]
        so_file = os.path.join(LIB_D, root + '.so')
        if os.path.isfile(so_file):
            os.remove(so_file)

    elif std_material is not None:
        try:
            environ.interactive_std_materials[std_material.name] = std_material
        except AttributeError:
            raise AttributeError("interactive material is missing attribute 'name'")

    else:
        raise ValueError('expected one of std_material or user_material')

load_material = load_interactive_material
