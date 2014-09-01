import numpy as np
from varinc import *

def catstr(a, b): return "{0}_{1}".format(a, b)

class Variable(object):
    """Variable class

    """
    def __init__(self, var_name, var_type, initial_value=None,
                 keys=None, length=None):

        if var_type not in VAR_TYPES:
            raise Exception("{0}: unknown variable type".format(var_type))

        if var_type == VAR_ARRAY:
            if length is None:
                raise ValueError("array variables must define a length")
            if keys is None:
                keys = [catstr(var_name, CMP_ARRAY(i)) for i in range(length)]

        elif length is not None:
            raise ValueError("{0}: attempting to assign length".format(var_name))

        elif var_type == VAR_SCALAR:
            length = DIM_SCALAR
            keys = [var_name]

        elif var_type == VAR_VECTOR:
            length = DIM_VECTOR
            keys = [catstr(var_name, CMP_VECTOR(i)) for i in range(length)]

        elif var_type == VAR_TENSOR:
            length = DIM_TENSOR
            keys = [catstr(var_name, CMP_TENSOR(i)) for i in range(length)]

        elif var_type == VAR_SYMTENSOR:
            length = DIM_SYMTENSOR
            keys = [catstr(var_name, CMP_SYMTENSOR(i)) for i in range(length)]

        elif var_type == VAR_SKEWTENSOR:
            length = DIM_SKEWTENSOR
            keys = [catstr(var_name, CMP_SKEWTENSOR(i)) for i in range(length)]

        else:
            raise Exception("{0}: unexpected variable type".format(var_type))

        # set initial value
        if initial_value is None:
            initial_value = np.zeros(length)

        elif isscalar(initial_value):
            initial_value = np.ones(length) * initial_value

        elif len(initial_value) != length:
            raise Exception("{0}: initial_value must have "
                            "length {1}".format(var_name, length))


        self.name = var_name
        self.vtype = var_type
        self.length = length
        self.initial_value = initial_value
        self.keys = keys

        return


class VariableContainer(np.ndarray):
    """Array like object to hold simulation data. Data are
    accessible by either index of name, i.e. data[0] or data["NAME"] (assuming
    NAME has index 0)

    """
    def __new__(cls, *args):
        values = []
        keys = []
        names = []
        lengths = []
        for arg in args:
            for item in arg:
                values.extend(item.initial_value)
                keys.extend(item.keys)
                names.append(item.name)
                lengths.append(item.length)

        obj = np.asarray(values).view(cls)
        obj.names = names
        obj.lengths = lengths
        obj.keys = keys

        I = 0
        for (i, name) in enumerate(names):
            l = lengths[i]
            ikeys = keys[I:I+l]
            setattr(obj, name, (I, l))
            if len(ikeys) > 1:
                J = I
                for key in ikeys:
                    setattr(obj, key, (J, 1))
                    J += 1
            I += l
        return obj

    def getidx(self, key):
        if isinstance(key, (str, basestring)):
            try:
                I, stride = getattr(self, key.upper())
            except AttributeError:
                return key
            if stride == 1:
                idx = I
            else:
                idx = slice(I, I+stride)
        else:
            idx = key
        return idx

    def __getitem__(self, key):
        idx = self.getidx(key)
        print idx
        print key
        print len(self)
        print
        if idx == len(self):
            idx == -1
        return super(VariableContainer, self).__getitem__(idx)

    def __setitem__(self, key, value):
        idx = self.getidx(key)
        super(VariableContainer, self).__setitem__(idx, value)

    def __array_finalize__(self, obj):
        self.names = getattr(obj, "names", None)
        self.lengths = getattr(obj, "lengths", None)
        self.keys = getattr(obj, "keys", None)
        if self.names:
            I = 0
            for (i, name) in enumerate(self.names):
                l = self.lengths[i]
                ikeys = self.keys[I:I+l]
                setattr(self, name, (I, l))
                if len(ikeys) > 1:
                    J = I
                    for key in ikeys:
                        setattr(self, key, J)
                        J += 1
                I += l

def isscalar(a):
    try:
        return not [i for i in a]
    except TypeError:
        return True
