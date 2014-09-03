import sys
from utils.misc import who_is_calling
from core.runtime import opts

def warn(string, limit=False, warnings=[0]):
    warnings[0] += 1
    if limit and warnings[0] > 10:
        return
    who = who_is_calling()
    sys.__stderr__.write("*** WARNING: {0} ({1})\n".format(string.upper(), who))
    sys.__stderr__.flush()

def error(string, r=1):
    """Write error message to stderr and stop"""
    who = who_is_calling()
    sys.__stderr__.write("*** ERROR: {0} ({1})\n".format(string.upper(), who))
    sys.__stderr__.flush()
    if r:
        raise SystemExit(1)

def write(string, end="\n"):
    """Write message to stdout """
    who = who_is_calling()
    if opts.verbosity:
        sys.__stdout__.write("{0} ({1}){2}".format(string.upper(), who, end))
        sys.__stdout__.flush()
