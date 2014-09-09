import numpy as np
from utils.errors import UserInputError
class Expansion(object):
    def __init__(self, exp_type, data):
        data = np.array(data)
        self._type = exp_type.upper()
        if self._type == "ISOTROPIC":
            if len(data) != 1:
                raise UserInputError("expected on value for isotropic expansion")
        else:
            raise UserInputError("{0}: unknown expansion type".format(exp_type))
        self._data = data

    @property
    def data(self):
        return self._data
