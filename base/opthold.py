from base.io import Error1
from base.namespace import Namespace


class OptionHolder(object):
    def __init__(self):
        pass

    def addopt(self, name, default, test=lambda x: True,
               dtype=float, choices=None):
        ns = Namespace()
        ns.name = name
        ns.value = default
        ns.test = test
        ns.dtype = dtype
        ns.choices = choices
        setattr(self, name, ns)

    def setopt(self, name, value):
        opt = self.getopt(name, getval=False)
        if opt is None:
            raise Error1("{0}: setopt: no such option".format(name))
        try:
            value = opt.dtype(value)
        except ValueError:
            raise Error1("{0}: invalid type for {1}".format(value, name))
        if not opt.test(value):
            raise Error1("{0}: invalid value for {1}".format(value, name))
        if opt.choices is not None:
            if value not in opt.choices:
                raise Error1("{0}: must be one of {1}, got {2}".format(
                    name, ", ".join(opt.choices), value))
        opt.value = value

    def getopt(self, name, getval=True):
        try: opt = getattr(self, name)
        except AttributeError: return None
        if getval:
            return opt.value
        return opt
