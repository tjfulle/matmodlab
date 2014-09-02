import numpy as np

def catstr(a, b): return "{0}_{1}".format(a, b)

class Array(np.ndarray):
    """Array like object to hold simulation data. Data are
    accessible by either index of name, i.e. data[0] or data["NAME"] (assuming
    NAME has index 0)

    """
    def __new__(cls, arg):
        values, keys, names, lengths = [], [], [], []
        for (name, akeys, avals) in arg:
            if name == "XTRA" and not len(avals):
                continue
            values.extend(avals)
            keys.extend(akeys)
            names.append(name)
            lengths.append(len(avals))
        obj = np.asarray(values).view(cls)
        obj.names = names
        obj.keys = keys
        obj.lengths = lengths
        obj._IX = {}

        I = 0
        for (i, name) in enumerate(names):
            L = lengths[i]
            ikeys = keys[I:I+L]
            obj._IX[i] = i
            if L != 1:
                obj._IX[name] = range(I, I+L)
                for (J, key) in enumerate(ikeys, I):
                    obj._IX[key] = J
            else:
                obj._IX[name] = I
            I += L
        return obj

    def __getitem__(self, key):
        if key == "XTRA" and "XTRA" not in self._IX:
            return np.array([])
        return super(Array, self).__getitem__(self._IX.get(key,key))

    def __setitem__(self, key, value):
        if key == "XTRA" and "XTRA" not in self._IX:
            return
        super(Array, self).__setitem__(self._IX.get(key,key), value)

    def __array_finalize__(self, obj):
        self.names = getattr(obj, "names", None)
        self.lengths = getattr(obj, "lengths", None)
        self.keys = getattr(obj, "keys", None)
        self._IX = {}
        if self.names:
            I = 0
            for (i, name) in enumerate(self.names):
                L = self.lengths[i]
                ikeys = self.keys[I:I+L]
                self._IX[i] = i
                if L != 1:
                    self._IX[name] = range(I, I+L)
                    for (J, key) in enumerate(ikeys, I):
                        self._IX[key] = J
                else:
                    self._IX[name] = I
                I += L
