import os
import logging
from .constants import *
from .product import *
from .materials.product import *
from distutils.spawn import find_executable as which
__all__ = ['environ']

class Environment(object):
    # --- IO type variables
    _log_level = logging.DEBUG
    warn = WARN
    Wall = False
    Werror = False
    Wlimit = 10

    # --- Debug and sqa
    raise_e = False
    sqa = False
    debug = False
    sqa_stiff = False

    # --- Performance
    nprocs = 1

    # --- IPython notebook
    notebook = 0

    # --- Material switching
    switch = []

    # --- Material models
    materials = {}
    materials['hyperelastic:neo hooke'] = {
        'model': USER, 'response': HYPERELASTIC,
        'source_files': [join(MAT_D, 'src/uhyper_neohooke.f90')],
        'libname': 'uhyper_neohooke',
        'param_names': ['C10', 'D1'], 'builtin': True,
        'ordering': [XX, YY, ZZ, XY, XZ, YZ]}
    materials['hyperelastic:polynomial'] = {
        'model': UHYPER, 'libname': 'uhyper_polynomial', 'builtin': True,
        'param_names': ['C10', 'C01', 'C20', 'C11', 'C02', 'C30',
                        'C21', 'C12', 'C03', 'D1', 'D2', 'D3'],
        'source_files': [join(MAT_D, 'src/uhyper_poly.f90')]}

    std_materials = [MAT_D]
    interactive_std_materials = {}
    interactive_usr_materials = {}

    rebuild_mat_lib = []

    simulation_dir = None

    plotter = MATPLOTLIB
    output_format = REC

    # For the gui
    gui_mode = False
    do_not_fork = False

    parent_process = 0

    # Fortran compiling
    fflags = [x for x in os.getenv('FFLAGS', '').split() if x.split()]
    fc = which(os.getenv('FC', 'gfortran'))

    # --- Test search directories
    test_dirs = [TEST_D]

    @property
    def log_level(self):
        return self._log_level

    @property
    def verbosity(self):
        return {logging.CRITICAL: 0,
                logging.ERROR: 0,
                logging.WARNING: 0,
                logging.INFO: 1,
                logging.DEBUG: 2}[self._log_level]

    @log_level.setter
    def log_level(self, value):
        choices = {-2: logging.CRITICAL,
                   -1: logging.ERROR,
                    0: logging.WARNING,
                    1: logging.INFO,
                    2: logging.DEBUG}
        if value in choices.values():
            log_level = value
        else:
            log_level = choices[value]
        self._log_level = log_level

    def __contains__(self, item):
        try:
            getattr(self, item)
            return True
        except AttributeError:
            return False

    def __getitem__(self, name):
        return getattr(self, '{0}'.format(name))

    def __setattr__(self, name, item):
        if name in self and item is None:
            return
        return super(Environment, self).__setattr__(name, item)

    def __str__(self):
        string = ', '.join('{0}={1!r}'.format(k, self[k])
                           for k in dir(self)
                           if not k.startswith('_'))
        return 'Environment({0})'.format(string)

    def _update(self, d):
        for (k, v) in d.items():
            setattr(self, k, v)

environ = Environment()

# --- Read in the user environment
__f1 = [os.path.expanduser('~/mml_userenv.py')]
__f2 = os.getenv('MML_USERENV', '').split(os.pathsep)
__f3 = [os.path.join(os.getcwd(), 'mml_userenv.py')]
for filename in __f1 + __f2 + __f3:
    if not os.path.isfile(filename):
        continue
    THIS_D = dirname(realpath(filename))
    with open(filename) as fh:
        directory, basename = os.path.split(filename)
        namespace = {}
        code = compile(fh.read(), basename, 'exec')
        exec(code, globals(), namespace)
        for (k, v) in namespace.items():
            if k.startswith('_'):
                continue
            if k not in environ:
                raise ValueError('unexpected environment variable %r' % k)
            if environ[k] is not None and type(environ[k]) != type(v):
                t = repr(type(environ[k])).split()[1].rstrip('>')
                raise ValueError('%s environment variable must be %s' % (k, t))
            if isinstance(environ[k], dict):
                for (kk, vv) in v.items():
                    if kk in environ:
                        raise ValueError('%s %s variable exists' % (kk, k))
                environ[k].update(v)
            elif isinstance(environ[k], list):
                environ[k].extend(v)
            else:
                environ._update({k: v})
