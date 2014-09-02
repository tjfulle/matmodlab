"""Runtime options for simulations

"""
from utils.namespace import Namespace
opts = Namespace()
opts.debug = False
opts.sqa = False
opts.I = None
opts.verbosity = 0
opts.runid = None
opts.Werror = False
opts.switch = None
opts.mimic = None

def set_runtime_opt(opt, val):
    setattr(opts, opt, val)
