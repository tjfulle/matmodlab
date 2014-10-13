"""Runtime options for simulations

"""
import os
import multiprocessing as mp
from core.configurer import cfgparse
from core.product import SUPPRESS_USER_ENV

class RuntimeOptions(object):
    def __init__(self):
        # warning level
        self._warn = "std"
        self.Wall = False
        self.Werror = False
        self.Wlimit = 10
        self.raise_e = False

        self._sqa = False
        self._debug = False
        self.sqa_stiff = False

        self._v = 1

        self._nprocs = 1
        self.do_not_fork = False

        self.viz_on_completion = False
        self._switch = []
        self.rebuild_mat_lib = False

        self._sim_dir = os.getcwd()

        # user config file configurable options
        if not SUPPRESS_USER_ENV:
            self.sqa = cfgparse("sqa", default=self._sqa)
            self.warn = cfgparse("warn", default=self._warn)
            self.debug = cfgparse("debug", default=self._debug)
            self.switch = cfgparse("switch", default=self._switch)
            self.nprocs = cfgparse("nprocs", default=self._nprocs)
            self.verbosity = cfgparse("verbosity", default=self._v)

    @property
    def warn(self):
        return self._warn
    @warn.setter
    def warn(self, x):
        v = x.lower()
        self._warn = v
        if v == "all":
            self.Wall = True
            self.Wlimit = 10000
        elif v == "error":
            self.Wall = self.Werror = self.raise_e = True

    @property
    def switch(self):
        return self._switch
    @switch.setter
    def switch(self, x):
        self._switch = []
        if x is None:
            return
        for (i, pair) in enumerate(x):
            # check to be sure switching is set correctly
            try:
                a, b = pair
            except ValueError:
                raise ValueError("invalid switching directive")
            self._switch.append((a, b))

    @property
    def debug(self):
        return self._debug
    @debug.setter
    def debug(self, x):
        self._debug = bool(x)
        if self._debug:
            self.warn = "error"

    @property
    def sqa(self):
        return self._sqa
    @sqa.setter
    def sqa(self, x):
        self._sqa = bool(x)

    @property
    def nprocs(self):
        num_procs = 1 if self.do_not_fork else self._nprocs
        return min(mp.cpu_count(), num_procs)

    @nprocs.setter
    def nprocs(self, n):
        self._nprocs = n

    @property
    def verbosity(self):
        return self._v
    @verbosity.setter
    def verbosity(self, x):
        self._v = int(x)

    @property
    def simulation_dir(self):
        return self._sim_dir
    @simulation_dir.setter
    def simulation_dir(self, x):
        assert os.path.isdir(x)
        self._sim_dir = x


opts = RuntimeOptions()
