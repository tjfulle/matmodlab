from core.product import PKG_D
from core.runtime import opts
from utils.misc import who_is_calling

def adderr(string):
    return "*** error: {0}".format(string.lstrip())

class FileNotFoundError(Exception):
    def __init__(self, filename):
        who = who_is_calling()
        message = adderr("{0}: file not found ({1})".format(filename, who))
        if opts.raise_e: super(FileNotFoundError, self).__init__(message)
        else: raise SystemExit(message)

class ModelNotImportedError(Exception):
    def __init__(self, model):
        who = who_is_calling()
        message = adderr("{0}: model not imported ({1})".format(model, who))
        if opts.raise_e: super(ModelNotImportedError, self).__init__(message)
        else: raise SystemExit(message)

class ModelLibNotFoundError(Exception):
    def __init__(self, model):
        who = who_is_calling()
        message = adderr("{0}.so shared object library "
                         "not found in {1} ({2})".format(model, pkg_d, who))
        if opts.raise_e: super(ModelLibNotFoundError, self).__init__(message)
        else: raise SystemExit(message)

class MatModelNotFoundError(Exception):
    def __init__(self, model):
        who = who_is_calling()
        message = adderr("{0}: material model not found ({1})".format(model, who))
        if opts.raise_e: super(ModelLibNotFoundError, self).__init__(message)
        else: raise SystemExit(message)

class DuplicateExtModule(Exception):
    def __init__(self, name):
        who = who_is_calling()
        message = adderr("{0}: duplicate extension module "
                         "encountered ({1})".format(name, who))
        if opts.raise_e: super(DuplicateExtModule, self).__init__(message)
        else: raise SystemExit(message)

class MatModLabError(Exception):
    def __init__(self, message):
        who = who_is_calling()
        message = adderr("{0}: ({1})".format(message, who))
        if opts.raise_e: super(MatModLabError, self).__init__(message)
        else: raise SystemExit(message)
