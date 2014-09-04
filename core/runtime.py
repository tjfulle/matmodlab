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
opts.nprocs = 1
opts.viz_on_completion = False
opts.raise_e = False

def set_runtime_opt(opt, val):
    setattr(opts, opt, val)
    if opts.debug:
        opts.raise_e = True
