import numpy as np

from utils.io import Error1

class Driver(object):
    _variables = []
    ndata = 0
    _data = np.zeros(ndata)

    def register_variable(self, var, vtype="SCALAR"):
        """Register material variable

        """
        vtype = vtype.upper()
        name = var.upper()
        if vtype == "SCALAR":
            var = [name]

        elif vtype == "TENS":
            var = ["{0}_{1}".format(name, x) for x in ("XX", "XY", "XZ",
                                                       "YX", "YY", "YZ",
                                                       "ZX", "ZY", "ZZ")]
        elif vtype == "SYMTENS":
            var = ["{0}_{1}".format(name, x)
                   for x in ("XX", "YY", "ZZ", "XY", "YZ", "XZ")]

        elif vtype == "SKEWTENS":
            var = ["{0}_{1}".format(name, x) for x in ("XY", "YZ", "XZ")]

        elif vtype == "VECTOR":
            var = ["{0}_{1}".format(name, x) for x in ("X", "Y", "Z")]

        else:
            raise Error1("{0}: unrecognized vtype".format(vtype))

        start = self.ndata
        self.ndata += len(var)
        end = self.ndata
        self._variables.extend(var)
        setattr(self, "{0}_slice".format(name.lower()), slice(start, end))

    def data(self, name=None):
        """Return the current material data

        Returns
        -------
        data : array_like
            Material data

        """
        return self._data[self.getslice(name)]

    def getslice(self, name=None):
        if name is None:
            return slice(0, self.ndata)
        return getattr(self, "{0}_slice".format(name.lower()))

    def allocd(self):
        """Allocate space for material data

        Notes
        -----
        This must be called after each consititutive model's setup method so
        that the number of xtra variables is known.

        """
        # Model data array.  See comments above.
        self._data = np.zeros(self.ndata)

    def setvars(self, **kwargs):
        for kw, arg in kwargs.items():
            self._data[self.getslice(kw)] = arg

    def variables(self):
        return self._variables
