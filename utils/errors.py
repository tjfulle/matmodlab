from project import PKG_D
from core.runtime import opts
from utils.mmlio import logger
from utils.misc import who_is_calling

def adderr(string):
    return "***ERROR: {0}".format(string.lstrip())

class FileNotFoundError(Exception):
    def __init__(self, filename):
        message = adderr("{0}: file not found".format(filename))
        logger.error(message, r=0, caller=who_is_calling())
        if opts.debug:
            super(FileNotFoundError, self).__init__(message)
        else:
            raise SystemExit(message)

class ModelNotImportedError(Exception):
    def __init__(self, model):
        message = adderr("{0}: model not imported".format(model))
        logger.error(message, r=0, caller=who_is_calling())
        if opts.debug:
            super(ModelNotImportedError, self).__init__(message)
        else:
            raise SystemExit(message)

class ModelLibNotFoundError(Exception):
    def __init__(self, model):
        message = adderr("{0}.so shared object library "
                         "not found in {1}".format(model, PKG_D))
        logger.error(message, r=0, caller=who_is_calling())
        if opts.debug:
            super(ModelLibNotFoundError, self).__init__(message)
        else:
            raise SystemExit(message)

class MatModelNotFoundError(Exception):
    def __init__(self, model):
        message = adderr("{0}: material model not found".format(model))
        logger.error(message, r=0, caller=who_is_calling())
        if opts.debug:
            super(ModelLibNotFoundError, self).__init__(message)
        else:
            raise SystemExit(message)

class DuplicateExtModule(Exception):
    def __init__(self, name):
        message = adderr("{0}: duplicate extension module encountered".format(name))
        logger.error(message, r=0, caller=who_is_calling())
        if opts.debug:
            super(DuplicateExtModule, self).__init__(message)
        else:
            raise SystemExit(message)

class GenericError(Exception):
    def __init__(self, message):
        logger.error(message, r=0, caller=who_is_calling())
        if opts.debug:
            super(GenericError, self).__init__(message)
        else:
            raise SystemExit(message)

class UserInputError(Exception):
    def __init__(self, message):
        logger.error(message, r=0, caller=who_is_calling())
        if opts.debug:
            super(InputError, self).__init__(message)
        else:
            raise SystemExit(message)
