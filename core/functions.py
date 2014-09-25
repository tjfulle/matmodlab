import sys
import math
import numpy as np
from utils.errors import MatModLabError


GDICT = {"__builtins__": None}
SAFE = {"np": np,
        "sqrt": np.sqrt, "max": np.amax, "min": np.amin,
        "stdev": np.std, "std": np.std,
        "abs": np.abs, "ave": np.average, "average": np.average,
        "sin": np.sin, "cos": np.cos, "tan": np.tan,
        "asin": np.arcsin, "acos": np.arccos,
        "atan": np.arctan, "atan2": np.arctan2,
        "log": np.log, "exp": np.exp,
        "floor": np.floor, "ceil": np.ceil,
        "pi": math.pi, "G": 9.80665, "inf": np.inf, "nan": np.nan}


DEFAULT_FUNCTIONS = {0: lambda x: 0., 1: lambda x: 1.}


class _Function(object):
    def __init__(self, function_id, function_type, function_defn):

	if function_id in DEFAULT_FUNCTIONS:
	    raise MatModLabError("function_id must not be "
                                 "one of {0}".format(", ".join(DEFAULT_FUNCTIONS)))

        func_type = "_".join(function_type.lower().split())
	if func_type == "analytic_expression":
	    self.func = function_defn
        elif func_type == "piecewise_linear":
            a = np.array(function_defn)
            self.func = _build_interpolating_function(a)
	else:
	    raise MatModLabError("only analytic expression supported right now")

        try:
            self.func(0)
        except BaseException, e:
            err = e.message if not hasattr(e, "msg") else e.msg
            raise MatModLabError(err)
	self.func_id = function_id

    def __call__(self, x):
	return self.func(x)

def _build_interpolating_function(a, monotonic=True):
    """Build an interpolating function from 2D numpy array a"""
    xp, fp = a[:, 0], a[:, 1]

    func = lambda x=np.amin(xp): np.interp(float(x), xp, fp)
    err = None

    if monotonic and np.any(np.diff(xp) < 0.):
        # check monotonicity
        func = None
        err = "Non-monotonic data detected"

    if func:
        try:
            func()
        except BaseException, e:
            message = e.message if not hasattr(e, "msg") else e.msg
            raise MatModLabError(message)

    return func


def _build_lambda(expr, var="x", default=[0.], disp=0):
    """Build a lambda function"""
    if not isinstance(var, (list, tuple)):
        var = [var]
    if not isinstance(default, (list, tuple)):
        default = [default]
    default = [float(x) for x in default]
    n = len(var) - len(default)
    if n > 0:
        default += [0.] * n

    # Build var to be of the form: var[0]=d[1], var[1]=d[1], ...
    # where d is the default
    var = ", ".join(["{0}={1}".format(x, default[i]) for i, x in enumerate(var)])
    try:
        func = eval("lambda {0}: {1}".format(var, expr), SAFE)
        err = None
    except BaseException as e:
        err = e.message if not hasattr(e, "msg") else e.msg
        func = None

    # Check the function
    if func:
        try:
            func()
        except BaseException, e:
            func = None
            err = e.message if not hasattr(e, "msg") else e.msg

    if not disp:
        return func

    return func, err


def Function(func_id, func_type, func_defn):
    return _Function(func_id, func_type, func_defn)
