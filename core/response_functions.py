import os
import re
import subprocess
import numpy as np
from utils.exodump import read_vars_from_exofile
from core.io import fatal_inp_error
import utils.xmltools as xmltools
from utils.opthold import OptionHolder, OptionHolderError as OptionHolderError

S_HREF = "href"
S_RESP_DESC = "descriptor"
S_RESP_FCN = "ResponseFunction"
GMD_RESP_FCNS = {"max": np.amax, "min": np.amin, "mean": np.mean,
                 "ave": np.average}
GMD_RESP_FCN_RE = r"gmd\.(?P<fcn>\w+)\s*\(\s*(?P<var>\w+)\s*\)"


def evaluate_response_function(respfcn, outfile, auxfiles=[]):
    """Evaluate the response function

    """
    if respfcn.startswith("gmd."):
        s = re.search(GMD_RESP_FCN_RE, respfcn)
        fcn = s.group("fcn")
        var = s.group("var")

        data = read_vars_from_exofile(outfile, var, h=0)[:, 1]
        respfcn = "{0}({1})".format(fcn, data.tolist())
        return eval(respfcn, {"__builtins__": None}, GMD_RESP_FCNS)

    cmd = "{0} {1} {2}".format(respfcn, outfile, " ".join(auxfiles))
    job = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT)
    job.wait()
    if job.returncode != 0:
        io.log_message("*** error: job {0} script failed".format(IOPT + 1))
        response = np.nan

    else:
        out, err = job.communicate()
        response = float(out)

    return response


def check_response_function_element(element):
    """Check the ResponseFunction element of the input file and return a
    formatted dictionary

    """
    response_fcn = {S_HREF: None, S_RESP_DESC: None}
    if not element:
        return response_fcn

    options = OptionHolder()
    options.addopt(S_HREF, None, dtype=str)
    options.addopt(S_RESP_DESC, None, dtype=str)

    for i in range(element.attributes.length):
        try:
            options.setopt(*xmltools.get_name_value(element.attributes.item(i)))
        except OptionHolderError, e:
            fatal_inp_error(e.message)
    if options.getopt(S_HREF):
        response_fcn[S_HREF] = options.getopt(S_HREF)
    if options.getopt(S_RESP_DESC):
        response_fcn[S_RESP_DESC] = options.getopt(S_RESP_DESC)

    if not response_fcn[S_HREF]:
        fatal_inp_error("expected {0} attribute to {1}".format(
            S_HREF, S_RESP_FCN))
        return response_fcn

    # determine if S_HREF is a file, or gmd function descriptor
    href = response_fcn[S_HREF]
    if href.startswith("gmd."):
        s = re.search(GMD_RESP_FCN_RE, href)
        if not s:
            fatal_inp_error("expected {0} in form gmd.fcn(VAR), "
                            "got {1}".format(S_HREF, href))
            return response_fcn
        fcn = s.group("fcn")
        var = s.group("var")
        if fcn.lower() not in GMD_RESP_FCNS:
            fatal_inp_error("{0}: not a valid gmd function, choose "
                            "from {1}".format(fcn, ", ".join(GMD_RESP_FCNS)))
        response_fcn[S_HREF] = "gmd.{0}({1})".format(fcn.lower(), var.upper())

    elif not os.path.isfile(response_fcn[S_HREF]):
        fatal_inp_error("{0}: no such file".format(response_fcn[S_HREF]))

    else:
        response_fcn[S_HREF] = os.path.realpath(response_fcn[S_HREF])

    return response_fcn
