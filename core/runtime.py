"""Runtime options for simulations

"""
from core.configurer import cfgparse
from utils.namespace import Namespace
from core.product import SUPRESS_USER_ENV

opts = Namespace()

opts.I = None
opts.runid = None

opts.Wall = False
opts.Werror = False
opts.Wlimit = 10
opts.raise_e = False

opts.viz_on_completion = False

# user config file configurable options
opts.sqa = False
opts.warn = "std"
opts.debug = False
opts.mimic = []
opts.nprocs = 1
opts.switch = None
opts.verbosity = 1
if not SUPRESS_USER_ENV:
    opts.sqa = cfgparse("sqa", default=opts.sqa)
    opts.warn = cfgparse("warn", default=opts.warn)
    opts.debug = cfgparse("debug", default=opts.debug)
    opts.mimic = cfgparse("mimic", default=opts.mimic)
    opts.switch = cfgparse("switch", default=opts.switch)
    opts.nprocs = cfgparse("nprocs", default=opts.nprocs)
    opts.verbosity = cfgparse("verbosity", default=opts.verbosity)

def set_runtime_opt(opt, val):
    if opt not in opts:
        raise AttributeError("attempting to set invalid runtime "
                             "option [{0}]".format(opt))
    setattr(opts, opt, val)
    if opts.debug:
        opts.raise_e = True
    if opts.Wall:
        opts.Wlimit = 10000
