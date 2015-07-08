import logging
from matmodlab.mml_siteenv import environ
from misc import who_is_calling

def adderr(string):
    return "*** error: {0}".format(string.lstrip())

class MatModLabError(Exception):
    def __init__(self, message):
        who = who_is_calling()
        message = adderr("{0}: ({1})".format(message, who))
        if environ.raise_e:
            super(MatModLabError, self).__init__(message)
        else:
            raise SystemExit(message)

def StopFortran(message):
    logging.getLogger('mps')
    logging.error(message)
    raise SystemExit(message)
