from matmodlab import PKG_D
from core.runtime import opts
from utils.misc import who_is_calling

def adderr(string):
    return "*** ERROR: {0}".format(string.lstrip())

class FileNotFoundError(Exception):
    def __init__(self, filename):
        who = who_is_calling()
        message = adderr("{0}: FILE NOT FOUND ({1})".format(filename, who))
        if opts.raise_e: super(FileNotFoundError, self).__init__(message)
        else: raise SystemExit(message)

class ModelNotImportedError(Exception):
    def __init__(self, model):
        who = who_is_calling()
        message = adderr("{0}: MODEL NOT IMPORTED ({1})".format(model, who))
        if opts.raise_e: super(ModelNotImportedError, self).__init__(message)
        else: raise SystemExit(message)

class ModelLibNotFoundError(Exception):
    def __init__(self, model):
        who = who_is_calling()
        message = adderr("{0}.so SHARED OBJECT LIBRARY "
                         "NOT FOUND IN {1} ({2})".format(model, PKG_D, who))
        if opts.raise_e: super(ModelLibNotFoundError, self).__init__(message)
        else: raise SystemExit(message)

class MatModelNotFoundError(Exception):
    def __init__(self, model):
        who = who_is_calling()
        message = adderr("{0}: MATERIAL MODEL NOT FOUND ({1})".format(model, who))
        if opts.raise_e: super(ModelLibNotFoundError, self).__init__(message)
        else: raise SystemExit(message)

class DuplicateExtModule(Exception):
    def __init__(self, name):
        who = who_is_calling()
        message = adderr("{0}: DUPLICATE EXTENSION MODULE "
                         "ENCOUNTERED ({1})".format(name, who))
        if opts.raise_e: super(DuplicateExtModule, self).__init__(message)
        else: raise SystemExit(message)

class GenericError(Exception):
    def __init__(self, message):
        who = who_is_calling()
        message = adderr("{0}: ({1})".format(message.upper(), who))
        if opts.raise_e: super(GenericError, self).__init__(message)
        else: raise SystemExit(message)

class UserInputError(Exception):
    def __init__(self, message):
        who = who_is_calling()
        message = adderr("{0}: ({1})".format(message.upper(), who))
        if opts.raise_e: super(InputError, self).__init__(message)
        else: raise SystemExit(message)
