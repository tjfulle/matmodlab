import numpy as np

# Field types
SCALAR = 0
VECTOR = 3
TENSOR_3D = 6
TENSOR_3D_FULL = 9
SDV = -1

def COMPONENT_LABELS(dtype):
    if dtype == VECTOR:
        return ('X', 'Y', 'Z')
    elif dtype == TENSOR_3D:
        return ('XX', 'YY', 'ZZ', 'XY', 'YZ', 'XZ')
    elif dtype == TENSOR_3D_FULL:
        return ('XX', 'XY', 'XZ', 'YX', 'YY', 'YZ', 'ZX', 'ZY', 'ZZ')
def COMPONENT(name, n):
    if n == VECTOR:
        x = ('X', 'Y', 'Z')
    elif n == TENSOR_3D:
        x = ('XX', 'YY', 'ZZ', 'XY', 'YZ', 'XZ')
    elif n == TENSOR_3D_FULL:
        x = ('XX', 'XY', 'XZ', 'YX', 'YY', 'YZ', 'ZX', 'ZY', 'ZZ')
    return x.index(name.upper())


# Valid invariants
MAGNITUDE = 0
MISES = 1
PRES = 2
EQ = 3
V = 4

# Misc. constant arrays and scalars
Z6 = np.zeros(6)
I6 = np.array([1., 1., 1., 0., 0., 0.])
I9 = np.array([1., 0., 0., 0., 1., 0., 0., 0., 1.])
VOIGHT = np.array([1, 1, 1, 2, 2, 2], dtype=np.float64)

DEFAULT_TEMP = 298.

ROOT2 = np.sqrt(2.0)
ROOT3 = np.sqrt(3.0)
TOOR2 = 1.0 / ROOT2
TOOR3 = 1.0 / ROOT3
ROOT23 = ROOT2 / ROOT3
TOLER = 1.E-06

# Tensor ordering
XX, YY, ZZ, XY, YZ, XZ = range(6)

# --- These are the standard outputs, any other requests are handled by tabfileio
REC = 'rpk'
TXT = 'out'
CSV = 'csv'
DB_FMTS = (REC, TXT, CSV)

# --- Permutate symbolic constants
ZIP = 'Zip'
COMBINATION = 'Combination'

RANGE = 'Range'
LIST = 'List'
WEIBULL = 'Weibull'
UNIFORM = 'Uniform'
NORMAL = 'Normal'
PERCENTAGE = 'Percentage'
UPERCENTAGE = 'UPercentage'
NPERCENTAGE = 'NPercentage'

# --- Optimization symbolic constants
SIMPLEX = 'Simplex'
POWELL = 'Powell'
COBYLA = 'Cobyla'
BRUTE = 'Brute'

# --- Warning levels
IGNORE = 'Ignore'
WARN = 'Warn'
ERROR = 'Error'

# --- User materials
# behaviors
MECHANICAL = 'Mechanical'
HYPERELASTIC = 'Hyperelastic'
ANISOHYPER = 'Anisohyper'

# types
USER = 'User'
UMAT = 'Umat'
UHYPER = 'Uhyper'
UANISOHYPER_INV = 'Uanisohyper_inv'

# add-ons
ISOTROPIC = 'Isotropic'
WLF = 'WLF'
PRONY = 'Prony'

# plotting
BOKEH = 'bokeh'
MATPLOTLIB = 'matplotlib'
