import sys
import math
import numpy as np


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


def build_interpolating_function(a, disp=0, monotonic=True):
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
            func = None
            err = e.message if not hasattr(e, "msg") else e.msg

    if not disp:
        return func

    return func, err


def build_lambda(expr, var="x", default=[0.], disp=0):
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
