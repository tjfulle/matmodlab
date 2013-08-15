import numpy as np

from utils.errors import Error1

class Material(object):

    def __init__(self):
        self.driver = "constitutive model"
        self.ndata = 0
        self.nxtra = 0
        self.xtra_var_keys = []
        self.xtra = np.zeros(self.nxtra)
        self.param_map = {}

    def register_parameters(self, *parameters):
        self.nparam = len(parameters)
        for idx, name in enumerate(parameters):
            name = name.upper()
            self.param_map[name] = idx
            setattr(self, name, idx)

    def register_xtra_variables(self, keys, mig=False):
        if self.nxtra:
            raise Error1("Register extra variables at most once")
        self.nxtra = len(keys)
        if mig:
            keys = [" ".join(x.split())
                    for x in "".join(keys).split("|") if x.split()]
        self.xtra_var_keys = keys

    def isparam(self, param_name):
        return param_name.upper() in self.param_map

    def parameter_index(self, param_name):
        return self.param_map.get(param_name.upper())

    def setup(self, *args, **kwargs):
        raise Error1("setup must be provided by model")

    def update_state(self, *args, **kwargs):
        raise Error1("update_state must be provided by model")

    def initialize(self, *args, **kwargs):
        return

    def set_initial_state(self, xtra):
        self.xtra = np.array(xtra)

    def variables(self):
        return self.xtra_var_keys

    def initial_state(self):
        return self.xtra
