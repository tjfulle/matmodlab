import sys
from utils.exomgr import ExodusIIManager
from utils.logger import Logger
from core.runtime import opts

exo = ExodusIIManager()
logger = Logger()

def log_warning(msg):
    logger.warn(msg)

def log_error(msg):
    logger.error(msg)

def log_message(msg):
    logger.write(msg)

def cout(message, end="\n"):
    """Write message to stdout """
    if opts.verbosity:
        sys.__stdout__.write(message + end)
        sys.__stdout__.flush()


def cerr(message):
    """Write message to stderr """
    sys.__stderr__.write(message + "\n")
    sys.__stderr__.flush()
