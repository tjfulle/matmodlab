import numpy as np

VAR_TYPES = []
VAR_SCALAR = 0
DIM_SCALAR = 1
CMP_SCALAR = lambda i: ""
VAR_TYPES.append(VAR_SCALAR)

VAR_VECTOR = 1
DIM_VECTOR = 3
CMP_VECTOR = lambda i: ["X", "Y", "Z"][i]
VAR_TYPES.append(VAR_VECTOR)

VAR_TENSOR = 2
DIM_TENSOR = 9
CMP_TENSOR = lambda i: ["XX", "XY", "XZ",
                        "YX", "YY", "YZ",
                        "ZX", "ZY", "ZZ"][i]
VAR_TYPES.append(VAR_TENSOR)

VAR_SYMTENSOR = 3
DIM_SYMTENSOR = 6
CMP_SYMTENSOR = lambda i: ["XX", "YY", "ZZ", "XY", "YZ", "XZ"][i]
VAR_TYPES.append(VAR_SYMTENSOR)

VAR_SKEWTENSOR = 4
DIM_SKEWTENSOR = 3
CMP_SKEWTENSOR = lambda i: ["XY", "YZ", "XZ"][i]
VAR_TYPES.append(VAR_SKEWTENSOR)

VAR_ARRAY = 5
DIM_ARRAY = None
CMP_ARRAY = lambda i: "{0}".format(i+1)
VAR_TYPES.append(VAR_ARRAY)

LOC_GLOB = 0
LOC_NODE = 1
LOC_ELEM = 2

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

def isscalar(a):
    try:
        return not [i for i in a]
    except TypeError:
        return True
