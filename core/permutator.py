import os
import re
import sys
import time
import shutil
import datetime
import subprocess
import numpy as np
import multiprocessing as mp
from itertools import izip, product

from matmodlab import MML_ENV
from core.logger import Logger
from core.runtime import opts
import utils.mmltab as mmltab
from utils.errors import UserInputError, GenericError


PERM_METHODS = ("zip", "combination", "shotgun", )
NJOBS = 0
RAND = np.random.RandomState()
logger = Logger()


class Permutator(object):
    def __init__(self, func, xinit, runid, method="zip", correlations=False,
                 verbosity=1, descriptor=None, nprocs=1, funcargs=[], d=None):
        global NJOBS

        self.runid = runid
        #self.func = (func.__module__, func.func_name)
        self.func = func
        self.nprocs = max(nprocs, opts.nprocs)
        self.correlations = correlations

        if not isinstance(descriptor, (list, tuple)):
            descriptor = [descriptor]
        self.descriptor = descriptor
        self.nresp = len(descriptor)

        if not isinstance(funcargs, (list, tuple)):
            funcargs = [funcargs]
        self.funcargs = [x for x in funcargs]
        # funcargs sent to every evaluation with first argument
        # the evaluation directory
        self.funcargs.insert(0, None)

        # set up logger
        self.rootd = os.path.join(d or os.getcwd(), runid + ".eval")
        if os.path.isdir(self.rootd):
            shutil.rmtree(self.rootd)
        os.makedirs(self.rootd)
        logfile = os.path.join(self.rootd, runid + ".log")
        logger.logfile = logfile
        logger.verbosity = verbosity

        # check method
        m = method.lower()
        if m not in PERM_METHODS:
            raise UserInputError("{0}: unrecognized method".format(method))
        self.method = m

        # check xinit
        self.names = []
        idata = []
        for x in xinit:
            if not isinstance(x, PermutateVariable):
                raise UserInputError("all xinit must be of type PermutateVariable")
            self.names.append(x.name)
            idata.append(x.data)

        # set up the jobs
        if self.method in ("zip", "shotgun"):
            if not all(len(x) == len(idata[0]) for x in idata):
                msg = ("Number of permutations must be the same for all "
                       "permutated parameters when using method: {0}".format(
                           self.method))
                raise GenericError(msg)
            self.data = zip(*idata)

        elif self.method == "combination":
            self.data = list(product(*idata))

        NJOBS = len(self.data)
        self.timing = {}

        # setup the mml-evaldb file
        self.tabular = mmltab.MMLTabularWriter(self.runid, d=self.rootd)

        # write summary to the log file
        str_pars = "\n".join("    {0}={1:.2g}".format(name, idata[i][0])
                             for (i, name) in enumerate(self.names))
        summary = """
summary of permutation job input
------- -- ----------- --- -----
runid: {0}
method: {1}
number of realizations: {2}
variables: {3:d}
starting values:
{4}
""".format(self.runid, self.method, NJOBS, len(self.names), str_pars)
        logger.write(summary)

    def run(self):

        self.timing["start"] = time.time()
        logger.write("{0}: starting permutation jobs...".format(self.runid))
        args = [(self.func, x, self.funcargs, i, self.rootd, self.names,
                 self.descriptor, self.tabular)
                 for (i, x) in enumerate(self.data)]
        nprocs = min(min(mp.cpu_count(), self.nprocs), len(self.data))
        if nprocs == 1:
            stats = [run_job(arg) for arg in args]
        else:
            pool = mp.Pool(processes=nprocs)
            stats = pool.map(run_job, args)
            pool.close()
            pool.join()

        logger.write("\npermutation jobs complete")
        self.timing["end"] = time.time()

        self.finish()

        return

    def finish(self):
        # write the summary
        self.tabular.close()

        logger.write("{0}: calculations completed ({1:.4f}s)".format(
            self.runid, self.timing["end"] - self.timing["start"]))

        if self.correlations:
            logger.write("{0}: creating correlation matrix".format(self.runid))
            mmltab.correlations(self.tabular._filepath)
            mmltab.plot_correlations(self.tabular._filepath)

        # close the log
        logger.finish()

    @property
    def output(self):
        return self.tabular._filepath

    @staticmethod
    def set_random_seed(seed, seedset=[0]):
        if seedset[0]:
            logger.warn("random seed already set")
        global RAND
        RAND = np.random.RandomState(seed)
        seedset[0] = 1


class PermutateVariable(object):

    def __init__(self, name, *args, **kwargs):
        N = int(kwargs.get("N", 10))
        arg = kwargs.get("arg")
        method = kwargs.get("method", "list")

        l = np.linspace
        s = {"range": lambda a, b, N: l(a, b, N),
             "list": lambda *a: np.array(a),
             "weibull": lambda a, b, N: a * RAND.weibull(b, N),
             "uniform": lambda a, b, N: RAND.uniform(a, b, N),
             "normal": lambda a, b, N: RAND.normal(a, b, N),
             "percentage": lambda a, b, N: (l(a-(b/100.)*a, a+(b/100.)* a, N))}
        m = method.lower()
        func = s.get(m)
        if func is None:
            raise UserInputError("{0} unrecognized method".format(method))

        if m != "list":
            if arg is None:
                raise UserInputError("{0}: arg keyword required".format(method))
            if len(args) > 1:
                n = len(args) + 1
                raise UserInputError("{0}: expected only 2 positional "
                                     "argument, got {1}".format(method, n))
            args = [args[0], arg, N]
            srep = "{0}({1}, {2}, {3})".format(m, args[0], arg, N)
        else:
            if len(args) == 1:
                args = args[0]
            srep = "{0}({1},...,{2})".format(m, args[0], args[-1])

        self.name = name
        self.srep = srep
        self.ival = args[0]
        self._data = func(*args)
        self._m = m

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

def catd(d, i):
    N = max(len(str(NJOBS)), 2)
    return os.path.join(d, "eval_{0:0{1}d}".format(i, N))

def run_job(args):
    """Run the single permutation job

    """
    (func, x, funcargs, i, rootd, names, descriptor, tabular) = args
    #func = getattr(sys.modules[func[0]], func[1])

    job_num = i + 1
    evald = catd(rootd, job_num)
    os.makedirs(evald)
    funcargs[0] = evald

    # write the params.in for this run
    parameters = zip(names, x)
    with open(os.path.join(evald, "params.in"), "w") as fobj:
        for name, param in parameters:
            fobj.write("{0} = {1: .18f}\n".format(name, param))

    logger.write("starting job {0}/{1} with {2}".format(job_num, NJOBS,
        ",".join("{0}={1:.2g}".format(n, p) for n, p in parameters)))

    try:
        resp = func(x, *funcargs)
        logger.write("finished job {0}".format(job_num))
        stat = 0
    except:
        logger.error("job {0} failed".format(job_num))
        stat = 1
        resp = [np.nan for _ in range(len(descriptor))]

    response = None
    if descriptor is not None:
        if not isinstance(resp, tuple):
            resp = resp,
        if len(descriptor) != len(resp):
            logger.error("job {0}: number of responses does not match number "
                         "of response descriptors".format(job_num))
        else:
            response = [(n, resp[i]) for (i, n) in enumerate(descriptor)]

    tabular.write_eval_info(job_num, stat, evald, parameters, response)

    return stat
