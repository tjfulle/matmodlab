import logging
from mml_siteenv import environ
from matmodlab.mml_siteenv import environ
from misc import who_is_calling

class MatModLabError(Exception):
    def __init__(self, message):
        who = who_is_calling()
        message = "{0}: ({1})".format(message, who).lstrip()
        if 'matmodlab.mmd.optimizer' in logging.Logger.manager.loggerDict:
            key = 'matmodlab.mmd.optimizer'
        elif 'matmodlab.mmd.permutator' in logging.Logger.manager.loggerDict:
            key = 'matmodlab.mmd.permutator'
        else:
            key = 'matmodlab.mmd.simulator'
        logging.getLogger(key).error(message)
        if environ.raise_e or environ.notebook:
            super(MatModLabError, self).__init__(message)
        else:
            raise SystemExit('*** Error: ' + message)

def StopFortran(message):
    raise MatModLabError(message)
