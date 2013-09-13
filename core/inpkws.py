"""Input keywords

"""
S_GMD = "GMDSpec"

S_PHYSICS = "Physics"
S_TTERM = "termination_time"
S_DRIVER = "driver"
T_DRIVER_TYPES = ("solid", "eos")
S_MATERIAL = "Material"
S_MODEL = "model"
S_RHO = "density"
S_PARAMS = "Parameters"

S_EXTRACT = "Extract"
S_EXFORMAT = "format"
T_EXFORMATS = ("ascii", "mathematica")
S_STEP = "step"
S_FFMT = "ffmt"

S_PATH = "Path"
S_SURFACE = "Surface"

S_HREF = "href"
S_MTL = "material"

S_PERMUTATION = "Permutation"
S_PERMUTATE = "Permutate"
T_PERM_METHODS = ("zip", "combine", "shotgun")

S_OPTIMIZATION = "Optimization"
S_OPTMZ = "Optimize"
T_OPT_METHODS = ("simplex", "cobyla", "powell")
S_MITER = "maxiter"
S_TOL = "tolerance"
S_DISP = "disp"
S_RESP_FCN = "ResponseFunction"
S_RESP_DESC = "descriptor"
S_AUX_FILE = "AuxiliaryFile"
S_METHOD = "method"
S_CORR = "correlation"
S_PLOT = "plot"
S_NONE = "none"
S_TBL = "table"
S_SEED = "seed"
S_IVAL = "initial_value"
S_VAR = "var"
S_VALS = "values"
S_SHOTGUN = "shotgun"

S_UNIFORM = "uniform"
S_RANGE = "range"
S_LIST = "list"
S_WEIBULL = "weibull"
S_NORMAL = "normal"
S_PERC = "percentage"

S_FCN = "Function"
S_ANAEXPR = "ANALYTIC EXPRESSION"
S_PWLIN = "PIECEWISE LINEAR"
ZERO_FCN_ID = 0
CONST_FCN_ID = 1

# driver
K_PRDEF = 0
S_PRDEF = "prdef"
S_KAPPA = "kappa"
S_AMPLITUDE = "amplitude"
S_RATFAC = "ratfac"
S_NFAC = "nfac"
S_TSTAR = "tstar"
S_ESTAR = "estar"
S_SSTAR = "sstar"
S_FSTAR = "fstar"
S_EFSTAR = "efstar"
S_DSTAR = "dstar"
S_FORMAT = "format"
S_PROPORTIONAL = "proportional"
S_NDUMPS = "ndumps"
S_COLS = "cols"
S_TFMT = "tfmt"
S_CFMT = "cfmt"
S_DEFAULT = "default"
S_TIME = "time"
S_DT = "dt"

# eos driver
K_RTSPC = 0
S_RSTAR = "rstar"
S_INCREMENTS = "increments"
S_TYPE = "type"
S_DENSITY_RANGE = "density_range"
S_INITIAL_TEMPERATURE = "initial_temperature"
S_HUGONIOT = "hugoniot"
S_ISOTHERM = "isotherm"
