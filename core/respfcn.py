import os
import re
import sys
import subprocess
import numpy as np
from utils.exodump import read_vars_from_exofile
from core.io import fatal_inp_error

MML_RESP_FCNS = {"max": np.amax, "min": np.amin, "mean": np.mean,
                 "ave": np.average,
                 "absmax": lambda a: np.amax(np.abs(a)),
                 "absmin": lambda a: np.amax(np.abs(a))}
MML_RESP_FCN_RE = r"mml\.(?P<fcn>\w+)\s*\(\s*(?P<var>\w+)\s*\)"


def evaluate_response_function(respfcn, outfile, auxfiles=[]):
    """Evaluate the response function

    """
    if respfcn.startswith("mml."):
        s = re.search(MML_RESP_FCN_RE, respfcn)
        fcn = s.group("fcn")
        var = s.group("var")

        data = read_vars_from_exofile(outfile, var, h=0)[:, 1]
        respfcn = "{0}({1})".format(fcn, data.tolist())
        return eval(respfcn, {"__builtins__": None}, MML_RESP_FCNS)

    cmd = "{0} {1} {2}".format(respfcn, outfile, " ".join(auxfiles))
    job = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT)
    job.wait()
    if job.returncode != 0:
        return None

    out, err = job.communicate()
    response = float(out)

    return response


def check_response_function(respfcn):
    """Check the ResponseFunction element of the input file and return a
    formatted dictionary

    """
    if not respfcn:
        return

    # determine if response function is a file, or mml function descriptor
    if respfcn.startswith("mml."):
        s = re.search(MML_RESP_FCN_RE, respfcn)
        if not s:
            fatal_inp_error("expected builtin in form mml.fcn(VAR), "
                            "got {1}".format(respfcn))
            return
        fcn = s.group("fcn")
        var = s.group("var")
        if fcn.lower() not in MML_RESP_FCNS:
            fatal_inp_error("{0}: not a valid mml function, choose "
                            "from {1}".format(fcn, ", ".join(MML_RESP_FCNS)))
        respfcn = "mml.{0}({1})".format(fcn.lower(), var.upper())

    elif not os.path.isfile(respfcn):
        fatal_inp_error("{0}: no such file".format(respfcn))

    else:
        respfcn = os.path.realpath(respfcn)

    return respfcn
