import numpy as np

class DataContainer(object):
    """Array like object to hold simulation data. Data are
    accessible by either index of name, i.e. data[0] or data["NAME"] (assuming
    NAME has index 0)

    """
    def __init__(self, arg):
        self._IX = {}
        data = []
        I = 0
        for (name, keys, vals) in arg:
            L = len(keys)
            data.extend(vals)
            name = name.upper()
            if name == "XTRA":
                if L == 0:
                    continue
                self._IX[name] = slice(I,I+L)
                for (J, key) in enumerate(keys, I):
                    self._IX[key] = J

            elif L == 1:
                self._IX[name] = I

            else:
                self._IX[name] = slice(I, I+L)
                for (J, key) in enumerate(keys, I):
                    self._IX[key] = J
            I += L
        self.data = np.array(data)

    def __getitem__(self, key):
        try:
            return self.data[self._IX[key.upper()]]
        except KeyError:
            return np.array([])

    def __setitem__(self, ix, val):
        self.data[ix] = val

    def __contains__(self, name):
        return name.upper() in self._IX

    def update(self, **kwargs):
        for (kw, val) in kwargs.items():
            kw = kw.upper()
            try:
                self.data[self._IX[kw]] = val
            except KeyError:
                continue

    def reshape(self, shape):
        return self.data.reshape(shape)

    def todict(self):
        d = {}
        for (kw, slc) in self._IX.items():
            d[kw] = self.data[slc]
        return d

    def get(self, name):
        return self.data[self._IX[name.upper()]]

    def summary(self, indent=""):
        summ = []
        for name in sorted(self._IX):
            value = self.get(name)
            summ.append("{0}{1} = {2}".format(indent, name, value))
        summ = "\n".join(summ)
        return summ

class _Parameter(object):
    def __init__(self, name, value, index):
        self.name = name
        self.value = value
        self.index = index
    def __index__(self): return self.index
    def __repr__(self): return str(self.value)
    def __mul__(self, other): return self.value * other
    def __rmul__(self, other): return other * self.value
    def __div__(self, other): return self.value / other
    def __rdiv__(self, other): return other / self.value
    def __add__(self, other): return self.value + other
    def __radd__(self, other): return other + self.value
    def __sub__(self, other): return self.value - other
    def __rsub__(self, other): return other - self.value


class Parameters(np.ndarray):
    """Array like object to hold material parameters. Parameters are
    accessible by either index of name, i.e. param[0] or param["A0"] (assuming
    parameter A0 has index 0)

    """
    def __new__(cls, names, values, model_name):
        if len(names) != len(values):
            raise ValueError("mismatch key,val pairs")
        obj = np.asarray(values).view(cls)
        obj.model_name = model_name
        obj.names = [s.upper() for s in names]
        obj.named_idx = dict((s, i) for (i, s) in enumerate(obj.names))
        for (name, i) in obj.named_idx.items():
            try:
                setattr(obj, name, _Parameter(name, obj[i], i))
            except:
                raise ValueError("{0}: parameter name reserved for "
                                 "internal use, rename".format(name))
        return obj

    def __str__(self):
        string = ", ".join("{0}={1:.2f}".format(n, self[i]) for (i, n) in
                          enumerate(self.names))
        return "Parameters({0})".format(string)

    def getidx(self, key):
        if isinstance(key, (str, basestring)):
            idx = self.named_idx.get(key.upper(), key)
        else:
            idx = key
        return idx

    def __getitem__(self, key):
        idx = self.getidx(key)
        return super(Parameters, self).__getitem__(idx)

    def __setitem__(self, key, value):
        idx = self.getidx(key)
        super(Parameters, self).__setitem__(idx, value)

    def __array_finalize__(self, obj):
        self.names = getattr(obj, "names", None)
        self.named_idx = getattr(obj, "named_idx", None)
        if self.named_idx:
            for (name, i) in self.named_idx.items():
                setattr(self, name, _Parameter(name, obj[i], i))
