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
