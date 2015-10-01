import os
import re
import sys
import time
import shutil
import logging
import datetime
import traceback
import subprocess
import numpy as np
import multiprocessing as mp
from random import shuffle
from itertools import izip, product
from collections import OrderedDict

from ..constants import *
from ..product import SPLASH
from ..utils.misc import backup
from ..mml_siteenv import environ
from ..utils.logio import setup_logger
from ..utils.errors import MatModLabError
from ..utils.mmltab import MMLTabularWriter, correlations, plot_correlations

RAND = np.random.RandomState()

class PermutatorState:
    pass
ps = PermutatorState()
ps.num_jobs = 0
ps.job_num = 0


class Permutator(object):
    def __init__(self, job, func, xinit, method=ZIP, correlations=False,
                 verbosity=None, descriptors=None, nprocs=1, funcargs=[], d=None,
                 shotgun=False, bu=0):

        self.job = job

        self.func = func
        self.nprocs = nprocs
        self.correlations = correlations
        self.shotgun = shotgun

        d = os.path.realpath(d or os.getcwd())
        self.directory = d
        self.rootd = os.path.join(d, job + ".eval")
        self.output = os.path.join(self.rootd, job + '.edb')

        if descriptors is None:
            self.descriptors = None
            self.nresp = 0
        else:
            if not isinstance(descriptors, (list, tuple)):
                descriptors = [descriptors]
            self.descriptors = ["_".join(x.split()) for x in descriptors]
            self.nresp = len(descriptors)

        # funcargs sent to every evaluation
        if not isinstance(funcargs, (list, tuple)):
            funcargs = [funcargs]
        self.funcargs = [x for x in funcargs]

        # set up logger
        if os.path.isdir(self.rootd):
            if bu:
                # back up the directory
                backup(self.rootd, mv=1)
            else:
                shutil.rmtree(self.rootd)
        os.makedirs(self.rootd)

        # basic logger
        logfile = os.path.join(self.rootd, self.job + '.log')
        logger = setup_logger('matmodlab.mmd.permutator', logfile,
                              verbosity=verbosity)

        # individual sims only log to file and not the console
        environ.parent_process = 1

        # check method
        if method not in (ZIP, COMBINATION):
            raise MatModLabError('unkown permutation method')
        self.method = method

        # check xinit
        self.names = []
        idata = []
        for x in xinit:
            try:
                self.names.append(x.name)
                idata.append(x.data)
            except AttributeError:
                raise MatModLabError("each xinit must be PermutateVariable")

        # set up the jobs
        if self.shotgun:
            # randomize order of data
            for (i, item) in enumerate(idata):
                shuffle(item)
                idata[i] = item

        if self.method in (ZIP,):
            if not all(len(x) == len(idata[0]) for x in idata):
                msg = ("Number of permutations must be the same for all "
                       "permutated parameters when using method: {0}".format(
                           self.method))
                raise MatModLabError(msg)
            self.data = zip(*idata)

        else:
            self.data = list(product(*idata))

        ps.num_jobs = len(self.data)
        self.timing = {}

        # setup the mml-evaldb file
        self.tabular = MMLTabularWriter(self.output, self.job)

        # write summary to the log file
        varz = "\n    ".join("{0}={1}".format(x.name, repr(x)) for x in xinit)
        summary = """
Summary of permutation job input
------- -- ----------- --- -----
Job:    {0}
Method: {1}
Number of realizations: {2}
Number of Variables: {3:d}
Variables:
    {4}
""".format(self.job, self.method, ps.num_jobs, len(self.names), varz)
        logger.info(summary)

    def run(self):

        logger = logging.getLogger('matmodlab.mmd.permutator')
        self.timing["start"] = time.time()
        logger.info("{0}: Starting permutation jobs...".format(self.job))
        args = [(self.func, x, self.funcargs, i, self.rootd, self.job,
                 self.names, self.descriptors, self.tabular)
                 for (i, x) in enumerate(self.data)]
        nprocs = max(self.nprocs, environ.nprocs)
        nprocs = min(min(mp.cpu_count(), nprocs), len(self.data)-1)

        # run the first job to see if it fails or not, rebuild material (if
        # requested), etc.
        self.statuses = [run_job(args[0])]
        if self.statuses[0] != 0:
            resp = raw_input("First job failed, continue? Y/N [N]  ")
            resp = "N" or resp.upper()
            if resp[0] == "N":
                self.finish()
                return

        if nprocs == 1:
            for arg in args[1:]:
                self.statuses.append(run_job(arg))
        else:
            pool = mp.Pool(processes=nprocs)
            out = pool.map(run_job, args[1:])
            pool.close()
            pool.join()
            self.statuses.extend(out)
        logger.info("\nPermutation jobs complete")

        self.finish()

        return

    def finish(self):

        self.timing["end"] = time.time()

        logger = logging.getLogger('matmodlab.mmd.permutator')

        # write the summary
        self.tabular.close()

        if not [x for x in self.statuses if x == 0]:
            logger.info("All calculations failed")
        else:
            dtime = self.timing["end"] - self.timing["start"]
            logger.info("Calculations completed ({0:.4f}s)".format(dtime))

        if self.correlations and [x for x in self.statuses if x == 0]:
            logger.info("Creating correlation matrix... ", extra={'continued':1})
            correlations(self.tabular.filename)
            if not environ.do_not_fork:
                plot_correlations(self.tabular.filename, pdf=1)
            logger.info("done")

        environ.parent_process = 0

        if environ.notebook:
            print '\nDone'

    def plot_correlations(self):
        return plot_correlations(self.tabular.filename)

    @staticmethod
    def set_random_seed(seed, seedset=[0]):
        if seedset[0]:
            logging.warn("random seed already set")
        global RAND
        RAND = np.random.RandomState(seed)
        seedset[0] = 1

class _PermutateVariable(object):

    def __init__(self, name, method, ival, data, srep):
        self.name = name
        self._m = method
        self.srep = srep
        self.ival = ival
        self._data = data

    def __repr__(self):
        return self.srep

    @property
    def data(self):
        return self._data

    @property
    def initial_value(self):
        return self.ival

    @property
    def method(self):
        return self._m

def PermutateVariable(name, init, b=None, N=10, method=LIST):
    """PermutateVariable factory method

    """
    lspc = np.linspace
    weib = RAND.weibull
    unif = RAND.uniform
    nrml = RAND.normal
    funcs = {
        RANGE: lambda a, b, N: ls(a, b, N),
        LIST: lambda *a: np.array(a),
        WEIBULL: lambda a, b, N: a * weib(b, N),
        UNIFORM: lambda a, b, N: unif(a, b, N),
        NORMAL: lambda a, b, N: nrml(a, b, N),
        PERCENTAGE: lambda a, b, N: lspc(a-(b/100.)*a, a+(b/100.)* a, N),
        UPERCENTAGE: lambda a, b, N: unif(a-(b/100.)*a, a+(b/100.)* a, N),
        NPERCENTAGE: lambda a, b, N: nrml(a-(b/100.)*a, a+(b/100.)* a, N)
        }
    try:
        func = funcs[method]
    except KeyError:
        raise MatModLabError('unkown PermutateVariable method')

    if method == LIST:
        fun_args = [x for x in init]
        srep = "[{0}]".format(", ".join("{0}".format(x) for x in init))

    else:
        try:
            init = float(init)
            b = float(b)
        except TypeError:
            raise MatModLabError('b keyword required for PermutateVariable method')
        fun_args = [init, b, int(N)]
        srep = "{0}({1}, {2}, {3})".format(method, init, b, N)

    ival = fun_args[0]
    data = func(*fun_args)
    return _PermutateVariable(name, method, ival, data, srep)

def catd(d, i):
    N = max(len(str(ps.num_jobs)), 2)
    return os.path.join(d, "eval_{0:0{1}d}".format(i, N))

def run_job(args):
    """Run the single permutation job

    """
    logger = logging.getLogger('matmodlab.mmd.permutator')
    (func, x, funcargs, i, rootd, job, names, descriptors, tabular) = args
    #func = getattr(sys.modules[func[0]], func[1])

    job_num = i + 1
    ps.job_num = i + 1
    evald = catd(rootd, ps.job_num)
    os.makedirs(evald)
    cwd = os.getcwd()
    os.chdir(evald)
    environ.simulation_dir = evald
    nresp = 0 if descriptors is None else len(descriptors)

    # write the params.in for this run
    parameters = zip(names, x)
    with open(os.path.join(evald, "params.in"), "w") as fobj:
        for name, param in parameters:
            fobj.write("{0} = {1: .18f}\n".format(name, param))

    s = ",".join("{0}={1:.2g}".format(n, p) for n, p in parameters)
    line = "Starting job {0}/{1} with {2}".format(ps.job_num, ps.num_jobs, s)
    if len(line) > 84:
        line = line[:84].split(",")
        line = ",".join(line[:-1]) + ",..."
    logger.info(line)

    if environ.notebook:
        print '\rRunning job {0}'.format(ps.job_num),

    try:
        resp = func(x, names, evald, job, *funcargs)
        logger.info("Finished job {0}".format(ps.job_num))
        stat = 0
    except:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        logger.error("\nRun {0} failed with the following "
                     "exception:\n".format(ps.job_num))
        traceback.print_exception(exc_type, exc_value, exc_traceback)
        stat = 1
        resp = [np.nan for _ in range(nresp)] or None

    responses = None
    if descriptors is not None:
        if not isinstance(resp, (tuple, list)):
            resp = (resp,)
        if nresp != len(resp):
            logger.error("job {0}: number of responses does not match number "
                         "of response descriptors".format(ps.job_num))
        else:
            responses = zip(descriptors, resp)
    tabular.write_eval_info(ps.job_num, stat, evald, parameters, responses)
    os.chdir(cwd)

    return stat
