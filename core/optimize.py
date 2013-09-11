import os
import sys
import time
import subprocess
import numpy as np
import shutil
import scipy.optimize
import datetime

from __config__ import cfg
import core.io as io
from core.respfcn import evaluate_response_function
from utils.gmdtab import GMDTabularWriter
import utils.pprepro as pprepro

IOPT = -1
HUGE = 1.e80
OPT_METHODS = {"simplex": "fmin", "powell": "fmin_powell",
               "cobyla": "fmin_cobyla"}

class OptimizationHandler(object):
    def __init__(self, runid, verbosity, method, exe, script, descriptor,
                 parameters, tolerance, maxiter, disp,
                 basexml, auxiliary, *opts):

        # root directory to run the problem
        self.rootd = os.path.join(os.getcwd(), runid + ".eval")
        if os.path.isdir(self.rootd):
            shutil.rmtree(self.rootd)
        os.makedirs(self.rootd)

        # logger
        io.setup_logger(runid, verbosity, d=self.rootd)

        # check inputs
        self.method = OPT_METHODS.get(method.lower())
        if self.method is None:
            io.log_warning("{0}: unrecognized optimization method".format(method))
        for x in exe.split():
            if not os.path.isfile(x):
                io.log_warning("{0}: no such file".format(x))
        if not os.path.isfile(script):
            io.log_warning("{0}: no such file".format(script))
        if maxiter <= 0:
            io.log_warning("maxiter must be greater than zero")
        if tolerance <= 0:
            io.log_warning("tolerance must be greater than zero")

        # check parameters to be optimized
        self.ivals = []
        self.bounds = []
        self.names = []
        inp_subs = pprepro.find_subs_to_make(basexml)
        for i, (name, ival, bounds) in enumerate(parameters):
            if name not in inp_subs:
                io.log_warning("{0}: not in xml input".format(name))
            if any(b is not None for b in bounds):
                if self.method in ("fmin", "fmin_powell"):
                    io.log_warning("{0}: bounds not supported".format(method))

                if bounds[0] is None: bounds[0] = -HUGE
                if bounds[1] is None: bounds[1] = HUGE
                if bounds[0] > bounds[1]:
                    io.log_warning("{0}: upper bound must be greater than "
                                   "lower".format(name))
                if bounds[1] < ival < bounds[0]:
                    io.log_warning("{0}: initial value out of "
                                   "bounds".format(name))

            self.bounds.append(bounds)
            self.names.append(name)
            self.ivals.append(ival)

        if io.WARNINGS_LOGGED:
            raise io.Error1("Stopping due to previous errors")

        self.runid = runid
        self.exe = exe
        self.script = script
        self.descriptor = descriptor if descriptor else "ERR"
        self.tolerance = tolerance
        self.maxiter = maxiter
        self.disp = disp
        self.basexml = basexml
        self.auxiliary_files = auxiliary
        self.tabular = GMDTabularWriter(runid, self.rootd)
        self.timing = {}

    def setup(self):
        pass

    def run(self):
        """Run the optimization job

        Set up directory to run the optimization job and call the minimizer

        """
        os.chdir(self.rootd)
        cwd = os.getcwd()
        io.log_message("{0}: starting jobs".format(self.runid))
        self.timing["start"] = time.time()

        # optimization methods work best with number around 1, here we
        # normalize the optimization variables and save the multiplier to be
        # used when the function gets called by the optimizer.
        xfac = []
        for ival in self.ivals:
            mag = eval("1.e" + "{0:12.6E}".format(ival).split("E")[1])
            xfac.append(mag)
            continue
        xfac = np.array(xfac)
        x0 = self.ivals / xfac

        if any(b is not None for bound in self.bounds for b in bound):
            # user has specified bounds on the parameters to be optimized. Here,
            # we convert the bounds to inequality constraints
            lcons, ucons = [], []
            for ibnd, bound in enumerate(self.bounds):
                lbnd, ubnd = bound
                if lbnd is None:
                    lbnd = -1.e20
                if ubnd is None:
                    ubnd = 1.e20

                lcons.append(lambda z, idx=ibnd, bnd=lbnd: z[idx] - bnd / xfac[idx])
                ucons.append(lambda z, idx=ibnd, bnd=ubnd: bnd / xfac[idx] - z[idx])

                self.bounds[ibnd] = (lbnd, ubnd)

                continue

            cons = lcons + ucons

        fargs = (self.rootd, self.runid, self.names, self.basexml, self.exe,
                 self.script, self.descriptor, self.auxiliary_files,
                 self.tabular, xfac,)

        if self.method == OPT_METHODS["simplex"]:
            xopt = scipy.optimize.fmin(
                func, x0, xtol=self.tolerance, ftol=self.tolerance,
                maxiter=self.maxiter, disp=self.disp, args=fargs)

        elif self.method == OPT_METHODS["powell"]:
            xopt = scipy.optimize.fmin_powell(
                func, x0, xtol=self.tolerance, ftol=self.tolerance,
                maxiter=self.maxiter, disp=self.disp, args=fargs)

        elif self.method == OPT_METHODS["cobyla"]:
            xopt = scipy.optimize.fmin_cobyla(
                func, x0, cons, consargs=(), disp=self.disp, args=fargs)

        self.xopt = xopt * xfac

        self.timing["end"] = time.time()

        return 0

    def finish(self):
        """ finish up the optimization job """
        self.tabular.close()
        io.log_message("{0}: calculations completed ({1:.4f}s)".format(
            self.runid, self.timing["end"] - self.timing["start"]))
        io.log_message("optimized parameters found in {0} iterations".format(IOPT))
        io.log_message("optimized parameters:")
        for (i, name) in enumerate(self.names):
            io.log_message("\t{0} = {1:12.6E}".format(name, self.xopt[i]))

        # close the log
        io.close_and_reset_logger()

    def output(self):
        return self.tabular._filepath


def func(xcall, *args):
    """Objective function

    Creates a directory to run the current job, runs the job through Payette
    and then gets the average normalized root mean squared error between the
    output and the gold file.

    Parameters
    ----------

    Returns
    -------
    error : float
        Average root mean squared error between the out file and gold file

    """
    global IOPT
    (rootd, runid, xnames, basexml, exe, script, desc, aux, tabular, xfac) = args

    IOPT += 1
    evald = os.path.join(rootd, "eval_{0}".format(IOPT))
    os.mkdir(evald)
    os.chdir(evald)

    # write the params.in for this run
    parameters = zip(xnames, xcall * xfac)
    with open("params.in", "w") as fobj:
        for name, param in parameters:
            fobj.write("{0} = {1: .18f}\n".format(name, param))

    io.log_message("starting job {0} with {1}".format(
        IOPT + 1, ",".join("{0}={1:.2g}".format(n, p) for n, p in parameters)))

    # Preprocess the input
    xmlinp = pprepro.find_and_make_subs(basexml, prepro=dict(parameters))
    xmlf = os.path.join(evald, runid + ".xml.preprocessed")
    with open(xmlf, "w") as fobj:
        fobj.write(xmlinp)

    # Run the job
    cmd = "{0} -I{1} {2}".format(exe, cfg.I, xmlf)
    out = open(os.path.join(evald, runid + ".con"), "w")
    job = subprocess.Popen(cmd.split(), stdout=out,
                           stderr=subprocess.STDOUT)
    job.wait()

    if job.returncode != 0:
        tabular.write_eval_info(IOPT, job.returncode, evald,
                                parameters, ((desc, np.nan),))
        io.log_message("**** error: job {0} failed".format(IOPT))
        return np.nan

    # Now the response function
    io.log_message("analyzing results of job {0}".format(IOPT + 1))
    outf = os.path.join(evald, runid + ".exo")
    opterr = evaluate_response_function(script, outf, aux)
    if opterr == np.nan:
        io.log_message("*** error: job {0} response function "
                       "failed".format(IOPT + 1))
    tabular.write_eval_info(IOPT, job.returncode, evald,
                            parameters, ((desc, opterr),))

    io.log_message("finished with job {0}".format(IOPT))


    # go back to the rootd
    os.chdir(rootd)

    return opterr