import os
import sys
import time
import subprocess
import numpy as np
import shutil
import datetime

from runtime import opts
from core.logger import Logger
from utils.mmltab import MMLTabularWriter
from utils.errors import MatModLabError

IOPT = 0
BIGNUM = 1.E+20
MAXITER = 50
TOL = 1.E-06
OPT_METHODS = ("simplex", "powell", "cobyla",)
logger = Logger()

class Optimizer(object):
    def __init__(self, func, xinit, runid, method="simplex", verbosity=1, d=None,
                 maxiter=MAXITER, tolerance=TOL, descriptor=None, funcargs=[]):
        opts.raise_e = True
        self.runid = runid
        self.func = func

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

        # check method
        m = method.lower()
        if m not in OPT_METHODS:
            raise MatModLabError("{0}: unrecognized method".format(method))
        self.method = m

        # set up logger
        d = d or os.getcwd()
        self.rootd = os.path.join(d, runid + ".eval")
        if os.path.isdir(self.rootd):
            shutil.rmtree(self.rootd)
        os.makedirs(self.rootd)
        logfile = os.path.join(self.rootd, runid + ".log")
        logger.assign_logfile(logfile)
        logger.verbosity = verbosity

        # check xinit
        self.names = []
        self.idata = []
        self.bounds = []
        for x in xinit:
            if not isinstance(x, OptimizeVariable):
                raise MatModLabError("all xinit must be of type OptimizeVariable")

            self.names.append(x.name)
            self.idata.append(x.initial_value)

            if x.bounds is not None:
                if self.method in ("simplex", "powell"):
                    logger.warn("{0}: method does not support bounds".format(m))
                    x.bounds = None
            self.bounds.append(x.bounds)

        if self.method in ("simplex", "powell"):
            self.bounds = None

        if maxiter <= 0:
            logger.warn("maxiter < 0, setting to default value")
            maxiter = MAXITER
        self.maxiter = maxiter

        if tolerance <= 0:
            logger.warn("tolerance < 0, setting to default value")
            tolerance = TOL
        self.tolerance = tolerance

        self.tabular = MMLTabularWriter(runid, self.rootd)
        self.timing = {}

        # write summary to the log file
        str_pars = "\n".join("  {0}={1:.2g}".format(name, self.idata[i])
                             for (i, name) in enumerate(self.names))
        resp = "\n".join("  {0}".format(it) for it in self.descriptor)
        summary = """
summary of optimization job input
------- -- ------------ --- -----
runid: {0}
method: {1}
variables: {2:d}
{3}
response descriptors:
{4}
""".format(self.runid, self.method, len(self.names), str_pars, resp)
        logger.write(summary)

    def run(self):
        """Run the optimization job

        Set up directory to run the optimization job and call the minimizer

        """
        import scipy.optimize

        self.timing["start"] = time.time()
        logger.write("{0}: starting optimization jobs...".format(self.runid))

        # optimization methods work best with number around 1, here we
        # normalize the optimization variables and save the multiplier to be
        # used when the function gets called by the optimizer.
        xfac = []
        for ival in self.idata:
            mag = eval("1.e" + "{0:12.6E}".format(ival).split("E")[1])
            xfac.append(mag)
            continue
        xfac = np.array(xfac)
        x0 = self.idata / xfac

        if self.bounds is not None:
            # user has specified bounds on the parameters to be optimized. Here,
            # we convert the bounds to inequality constraints
            lcons, ucons = [], []
            for ibnd, bound in enumerate(self.bounds):
                lbnd, ubnd = bound
                lcons.append(lambda z, idx=ibnd, bnd=lbnd: z[idx]-bnd/xfac[idx])
                ucons.append(lambda z, idx=ibnd, bnd=ubnd: bnd/xfac[idx]-z[idx])
                self.bounds[ibnd] = (lbnd, ubnd)
                continue
            cons = lcons + ucons

        args = (self.func, self.funcargs, self.rootd, self.names,
                self.descriptor, self.tabular, xfac)

        if self.method == "simplex":
            xopt = scipy.optimize.fmin(
                run_job, x0, xtol=self.tolerance, ftol=self.tolerance,
                maxiter=self.maxiter, args=args, disp=0)

        elif self.method == "powell":
            xopt = scipy.optimize.fmin_powell(
                run_job, x0, xtol=self.tolerance, ftol=self.tolerance,
                maxiter=self.maxiter, args=args, disp=0)

        elif self.method == "cobyla":
            xopt = scipy.optimize.fmin_cobyla(
                run_job, x0, cons, consargs=(), args=args, disp=0)

        self.xopt = xopt * xfac

        self.timing["end"] = time.time()

        logger.write("\noptimization jobs complete")

        self.finish()

        return

    def finish(self):
        """ finish up the optimization job """
        self.tabular.close()
        opt_pars = "\n".join("  {0}={1:12.6E}".format(name, self.xopt[i])
                             for (i, name) in enumerate(self.names))
        opt_time = self.timing["end"] - self.timing["start"]
        summary = """
summary of optimization results
------- -- ------------ -------
{0}: calculations completed ({1:.4f}s.)
iterations: {2}
optimized parameters
{3}
""".format(self.runid, opt_time, IOPT, opt_pars)
        logger.write(summary)

        # close the log
        logger.finish()

        # write out optimized params
        with open(os.path.join(self.rootd, "params.opt"), "w") as fobj:
            for (i, name) in enumerate(self.names):
                fobj.write("{0} = {1: .18f}\n".format(name, self.xopt[i]))

    @property
    def output(self):
        return self.tabular._filepath

def catd(d, i):
    N = 3
    return os.path.join(d, "eval_{0:0{1}d}".format(i, N))

def run_job(xcall, *args):
    """Objective function

    Creates a directory to run the current job, runs the job, returns the
    value of the objective function determined.

    Returns
    -------
    error : float
        Error in job

    """
    global IOPT
    func, funcargs, rootd, xnames, desc, tabular, xfac = args

    IOPT += 1
    evald = catd(rootd, IOPT)
    os.mkdir(evald)
    funcargs[0] = evald

    # write the params.in for this run
    x = xcall * xfac
    parameters = zip(xnames, x)
    with open(os.path.join(evald, "params.in"), "w") as fobj:
        for name, param in parameters:
            fobj.write("{0} = {1: .18f}\n".format(name, param))

    logger.write("starting job {0} with {1}".format(
        IOPT, ",".join("{0}={1:.2g}".format(n, p) for n, p in parameters)),
        end="... ")

    try:
        err = func(x, *funcargs)
        logger.write("done (error={0:.4e})".format(err))
        stat = 0
    except BaseException as e:
        message = " ".join("{0}".format(_) for _ in e.args)
        if hasattr(e, "filename"):
            message = e.filename + ": " + message[1:]
        logger.error("\nrun {0} failed with the following exception:\n"
                     "   {1}".format(IOPT, message))
        stat = 1
        err = np.nan

    tabular.write_eval_info(IOPT, stat, evald, parameters, ((desc[0], err),))

    return err

class OptimizeVariable(object):

    def __init__(self, name, initial_value, bounds=None):
        self.name = name
        self.ival = initial_value
        self.cval = initial_value
        self.bounds = bounds

        errors = 0
        # check bounds
        if bounds is not None:
            if not isinstance(bounds, (list, tuple, np.ndarray)):
                raise MatModLabError("expected bounds to be a tuple of length 2")
            if len(bounds) != 2:
                raise MatModLabError("expected bounds to be a tuple of length 2")
            if bounds[0] is None: bounds[0] = -BIGNUM
            if bounds[1] is None: bounds[1] =  BIGNUM

            if bounds[0] > bounds[1]:
                errors += 1
                logger.error("{0}: upper bound < lower bound".format(name))

            if bounds[1] < initial_value < bounds[0]:
                errors += 1
                logger.error("{0}: initial value not bracketed "
                             "by bounds".format(name))
            if errors:
                raise MatModLabError("stopping due to previous errors")

            self.bounds = np.array(bounds)

    def __repr__(self):
        return "opt{0}({1})".format(self.name, self.initial_value)

    @property
    def current_value(self):
        return self.cval

    @property
    def initial_value(self):
        return self.ival
