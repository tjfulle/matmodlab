"""Module containing methods for computing kinematic quantities"""

import sys
import numpy as np
from numpy.linalg import inv, solve, lstsq

from core.io import Error1
from core.solvers import newton, simplex
from lib.mmlabpack import mmlabpack

I = np.eye(3)

def sig2d(material, dt, d, sig, xtra, v, sigspec, proportional, *args):
    """Determine the symmetric part of the velocity gradient given stress

    Parameters
    ----------

    Returns
    -------

    Approach
    --------
    Seek to determine the unknown components of the symmetric part of
    velocity gradient d[v] satisfying

                               P(d[v]) = Ppres[:]                      (1)

    where P is the current stress, d the symmetric part of the velocity
    gradient, v is a vector subscript array containing the components for
    which stresses (or stress rates) are prescribed, and Ppres[:] are the
    prescribed values at the current time.

    Solution is found iteratively in (up to) 3 steps
      1) Call newton to solve 1, return stress, xtra, d if converged
      2) Call newton with d[v] = 0. to solve 1, return stress, xtra, d
         if converged
      3) Call simplex with d[v] = 0. to solve 1, return stress, xtra, d

    """
    dsave = d.copy()

    if not proportional:
        d = newton(material, dt, d, sig, xtra, v, sigspec, *args)
        if d is not None:
            return d

        # --- didn't converge, try Newton's method with initial
        # --- d[v]=0.
        d = dsave.copy()
        d[v] = np.zeros(len(v))
        d = newton(material, dt, d, sig, xtra, v, sigspec, *args)
        if d is not None:
            return d

    # --- Still didn't converge. Try downhill simplex method and accept
    #     whatever answer it returns:
    d = dsave.copy()
    return simplex(material, dt, d, sig, xtra, v, sigspec, proportional,
                   *args)
