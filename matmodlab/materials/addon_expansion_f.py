import logging
import numpy as np
from matmodlab.utils.errors import StopFortran
from matmodlab.materials.product import ISOTROPIC

xpansion = None

class Expansion(object):
    def __init__(self, exp_type, data):
        data = np.array(data)
        self.exp_type = exp_type
        if self.exp_type == ISOTROPIC:
            if len(data) != 1:
                raise ValueError("unexpected value for isotropic expansion")
        else:
            raise ValueError("{0}: unknown expansion type".format(exp_type))
        self._data = data

        global xpansion
        try:
            from matmodlab.lib.expansion import expansion as xpansion
        except ImportError:
            raise ImportError('lib.expansion.so not imported')

    def update_state(self, temp, dtemp, F, kappa):
        log = logging.getLogger('matmodlab.mmd.simulator')
        Fm, Em = xpansion.mechdef(self.data, temp, dtemp, kappa, F,
                                  log.info, log.warn, StopFortran)
        return Fm, Em

    @property
    def data(self):
        return self._data
