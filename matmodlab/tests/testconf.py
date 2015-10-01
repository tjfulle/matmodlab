import pytest
import shutil
from os.path import splitext, realpath, dirname, join, isfile
from numpy import allclose, argmax, prod, reshape, amax, abs, trace, dot, eye, \
    array, exp

from matmodlab import *
from matmodlab.constants import *
from matmodlab.utils.misc import remove
from matmodlab.utils.fileio import filediff

this_directory = dirname(realpath(__file__))
control_file = join(this_directory, 'base.diff')

def newton(xn, func, fprime, tol=1.e-7):
    for iter in range(25):
        fder = fprime(xn)
        if fder == 0:
            message = 'derivative was zero'
            warnings.warn(message, RuntimeWarning)
            return xn
        fval = func(xn)
        x = xn - fval / fder
        if abs(x - xn) < tol:
            return x
        xn = x
    msg = "Failed to converge after %d iterations, value is %s" % (25, x)
    raise RuntimeError(msg)

class StandardMatmodlabTest(object):
    '''Defines setup and teardown methods for standard test'''

    @classmethod
    def setup_class(self):
        self.completed_jobs = []

    @classmethod
    def teardown_class(self):
        '''Removes all test generated files'''
        exts = ['.' + x for x in DB_FMTS]
        exts = []
        exts += ['.difflog', '.con', '.log', '.eval', '.dat']
        for job in self.completed_jobs:
            for ext in exts:
                remove(join(this_directory, job + ext))

    @staticmethod
    def compare_with_baseline(job, base=None, cf=control_file, interp=0, adjust_n=0):
        if base is None:
            for ext in ('.base_rpk', '.base_dat'):
                base = join(this_directory, job.job + ext)
                if isfile(base):
                    break
            else:
                raise OSError('no base file found for {0}'.format(job.job))
        f = splitext(job.filename)[0] + '.difflog'
        with open(f, 'w') as fh:
            return filediff(job.filename, base, control_file=cf, stream=fh,
                            interp=interp, adjust_n=adjust_n)
