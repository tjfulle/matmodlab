import os
import re
import sys
import time
import shutil
import subprocess
import datetime
import numpy as np
import multiprocessing as mp
from itertools import izip, product

import core.io as io
from __config__ import cfg
from utils.gmdtab import GMDTabularWriter
from utils.pprepro import find_and_make_subs
from core.respfcn import evaluate_response_function, GMD_RESP_FCN_RE
import utils.gmdtab as gmdtab


PERM_METHODS = ("zip", "combine", )
NJOBS = 0


class PermutationHandler(object):
    def __init__(self, runid, verbosity, exe, nproc, method, respfcn,
                 parameters, basexml, correlation):
        global NJOBS

        self.rootd = os.path.join(os.getcwd(), runid + ".eval")
        if os.path.isdir(self.rootd):
            shutil.rmtree(self.rootd)
        os.makedirs(self.rootd)
        io.setup_logger(runid, verbosity, d=self.rootd)

        self.runid = runid
        self.method = method
        self.exe = exe
        self.basexml = basexml
        self.nproc = nproc

        self.names = []
        self.timing = {}
        self.ivalues = []
        for (name, ivalue) in parameters:
            self.names.append(name)
            self.ivalues.append(ivalue)

        if respfcn:
            self.respdesc, self.respfcn = respfcn
            s = re.search(GMD_RESP_FCN_RE, self.respfcn)
            self.respdesc = s.group("var")
        else:
            self.respfcn = self.respdesc = None
        self.correlation = correlation

        # set up the jobs
        if self.method in ("zip", "shotgun"):
            if not all(len(x) == len(self.ivalues[0]) for x in self.ivalues):
                raise io.Error1("Number of permutations must be the same for "
                                "all permutated parameters when using method: {0}"
                                .format(self.method))
            self.ranges = zip(*self.ivalues)

        elif self.method == "combine":
            self.ranges = list(product(*self.ivalues))

        NJOBS = len(self.ranges)

        # setup the gmd-evaldb file
        self.tabular = GMDTabularWriter(self.runid, d=self.rootd)

        # write to the log file the
        io.log_debug("Permutated parameters:")
        for name in self.names:
            io.log_debug("  {0}".format(name))

        pass

    def run(self):

        os.chdir(self.rootd)
        cwd = os.getcwd()
        io.log_message("{0}: starting {1} jobs".format(self.runid, NJOBS))
        self.timing["start"] = time.time()

        job_inp = ((i, self.exe, self.runid, self.names, self.basexml,
                    params, self.respfcn, self.respdesc, self.rootd, self.tabular)
                   for (i, params) in enumerate(self.ranges))

        nproc = min(min(mp.cpu_count(), self.nproc), len(self.ranges))
        if nproc == 1:
            self.statuses = [run_single_job(job) for job in job_inp]

        else:
            pool = mp.Pool(processes=nproc)
            self.statuses = pool.map(run_single_job, job_inp)
            pool.close()
            pool.join()

        io.log_message("finished permutation jobs")
        self.timing["end"] = time.time()
        return

    def finish(self):
        # write the summary
        self.tabular.close()

        io.log_message("{0}: calculations completed ({1:.4f}s)".format(
            self.runid, self.timing["end"] - self.timing["start"]))

        if self.respfcn and self.correlation:
            io.log_message("{0}: creating correlation matrix".format(self.runid))
            if "table" in self.correlation:
                gmdtab.correlations(self.tabular._filepath)
            if "plot" in self.correlation:
                gmdtab.plot_correlations(self.tabular._filepath)

        # close the log
        io.close_and_reset_logger()

    def output(self):
        return self.tabular._filepath


def run_single_job(args):
    (job_num, exe, runid, names, basexml, params, respfcn,
     respdesc, rootd, tabular) = args
    # make and move in to the evaluation directory
    evald = os.path.join(rootd, "eval_{0}".format(job_num))
    os.makedirs(evald)
    os.chdir(evald)

    # write the params.in for this run
    parameters = zip(names, params)
    with open("params.in", "w") as fobj:
        for name, param in parameters:
            fobj.write("{0} = {1: .18f}\n".format(name, param))

    # Preprocess the input
    xmlinp = find_and_make_subs(basexml, prepro=dict(parameters))
    xmlf = os.path.join(evald, runid + ".xml.preprocessed")
    with open(xmlf, "w") as fobj:
        fobj.write(xmlinp)

    cmd = "{0} -I{1} {2}".format(exe, cfg.I, xmlf)
    out = open(os.path.join(evald, runid + ".con"), "w")
    io.log_message("starting job {0}/{1} with {2}".format(
        job_num + 1, NJOBS,
        ",".join("{0}={1:.2g}".format(n, p) for n, p in parameters)))
    job = subprocess.Popen(cmd.split(), stdout=out, stderr=subprocess.STDOUT)
    job.wait()
    if job.returncode != 0:
        io.log_message("*** error: job {0} failed".format(job_num + 1))
    else:
        io.log_message("finished with job {0}".format(job_num + 1))

    response = None
    if respfcn is not None:
        io.log_message("analyzing results of job {0}".format(job_num + 1))
        outf = os.path.join(evald, runid + ".exo")
        response = evaluate_response_function(respfcn, outf)
        if response == np.nan:
            io.log_message("*** error: job {0} response function "
                           "failed".format(job_num + 1))
        response = ((respdesc, response),)

    tabular.write_eval_info(job_num, job.returncode, evald, parameters, response)

    return job.returncode
