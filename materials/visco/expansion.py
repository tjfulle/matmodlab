from core.mmlio import fatal_inp_error
class Expansion(object):
    def __init__(self, exp_type, data):
        self._type = exp_type.upper()
        if self._type == "ISOTROPIC":
            if len(data) != 1:
                fatal_inp_error("expected on value for isotropic expansion")
        else:
            fatal_inp_error("{0}: unknown expansion type".format(exp_type))
        self._data = data

    @property
    def data(self):
        return self._data
