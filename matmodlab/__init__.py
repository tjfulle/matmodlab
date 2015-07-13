import os
import sys
import inspect
import logging
import warnings
from math import *
from product import *
from StringIO import StringIO

from os.path import realpath, isfile, isdir, join, splitext, dirname, basename

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

# Monkey path the logging stream handler emit function
logging.basicConfig(level=logging.DEBUG, format='%(message)s')
def emit(self, record):
    '''Monkey-patch the logging StreamHandler emit function. Allows omiting
    trailing newline when not wanted'''
    if hasattr(self, 'baseFilename'):
        fs = '%s\n'
    else:
        fs = '%s' if getattr(record, 'continued', False) else '%s\n'
    self.stream.write(fs % self.format(record))
    self.flush()
logging.StreamHandler.emit = emit

# ------------------------ FACTORY METHODS TO SET UP AND RUN A SIMULATION --- #
from numpy import array, float64
from mmd.mdb import mdb
from mmd.simulator import *
from mml_siteenv import environ
from mmd.permutator import Permutator, PermutateVariable
from mmd.optimizer import Optimizer, OptimizeVariable
from constants import *
from materials.product import *
from utils.elas import elas
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
    if p == 'bokeh':
        from bokeh.plotting import output_notebook
        output_notebook()
        i = 2
    elif p == 'matplotlib':
        i = 1

    environ.notebook = i
    environ.log_level = logging.WARNING
    try:
        from sympy import init_printing
        init_printing()
    except ImportError:
        pass
