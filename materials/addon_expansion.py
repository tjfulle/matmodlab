import numpy as np
from utils.errors import MatModLabError

xpansion = None

class Expansion(object):
    def __init__(self, exp_type, data):
        data = np.array(data)
        self._type = exp_type.upper()
        if self._type == "ISOTROPIC":
            if len(data) != 1:
                raise MatModLabError("expected on value for isotropic expansion")
        else:
            raise MatModLabError("{0}: unknown expansion type".format(exp_type))
        self._data = data

    def setup(self):
        global xpansion
        try:
            from lib.expansion import expansion as xpansion
        except ImportError:
            raise MatModLabError("attempting thermal expansion but "
                                 "lib/expansion.so not imported")

    def update_state(self, logger, temp, dtemp, F, kappa):
        comm = (logger.write, logger.warn, logger.raise_error)
        Fm, Em = xpansion.mechdef(self.data, temp, dtemp, kappa, F, *comm)
        return Fm, Em

    @property
    def data(self):
        return self._data
