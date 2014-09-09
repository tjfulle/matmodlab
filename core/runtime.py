"""Runtime options for simulations

"""
from utils.namespace import Namespace
opts = Namespace()
opts.debug = False
opts.sqa = False
opts.I = None
opts.verbosity = 1
opts.runid = None
opts.Werror = False
opts.Wall = False
opts.Wlimit = 10
opts.switch = None
opts.mimic = None
opts.nprocs = 1
opts.viz_on_completion = False
opts.raise_e = False
opts.rebuild_material = False

def set_runtime_opt(opt, val):
    setattr(opts, opt, val)
    if opts.debug:
        opts.raise_e = True
    if opts.Wall:
        opts.Wlimit = 10000
