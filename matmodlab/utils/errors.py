import logging
from matmodlab.mml_siteenv import environ
from misc import who_is_calling

class MatModLabError(Exception):
    def __init__(self, message):
        who = who_is_calling()
        message = "{0}: ({1})".format(message, who).lstrip()
        if environ.raise_e or environ.notebook:
            super(MatModLabError, self).__init__(message)
        else:
            raise SystemExit('*** Error: ' + message)

def StopFortran(message):
    logging.getLogger('mps')
    logging.error(message)
    raise SystemExit(message)
