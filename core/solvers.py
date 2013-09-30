"""Module containing utilities for Payette's iterative stress solver"""

import math
import numpy as np
import scipy.optimize
import sys

import core.io as io

EPS = np.finfo(np.float).eps


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
        d = _newton(material, dt, d, sig, xtra, v, sigspec, *args)
        if d is not None:
            return d

        # --- didn't converge, try Newton's method with initial
        # --- d[v]=0.
        d = dsave.copy()
        d[v] = np.zeros(len(v))
        d = _newton(material, dt, d, sig, xtra, v, sigspec, *args)
        if d is not None:
            return d

    # --- Still didn't converge. Try downhill simplex method and accept
    #     whatever answer it returns:
    d = dsave.copy()
    return _simplex(material, dt, d, sig, xtra, v, sigspec, proportional,
                    *args)


def _newton(material, dt, darg, sigarg, xtraarg, v, sigspec, *args):
    """Seek to determine the unknown components of the symmetric part of velocity
    gradient d[v] satisfying

                               sig(d[v]) = sigspec

    where sig is the current stress, d the symmetric part of the velocity
    gradient, v is a vector subscript array containing the components for
    which stresses (or stress rates) are prescribed, and sigspec are the
    prescribed values at the current time.

    Parameters
    ----------
    material : instance
        constiutive model instance
    dt : float
        time step
    sig : ndarray
        stress at beginning of step
    xtra : ndarray
        extra variables at beginning of step
    v : ndarray
        vector subscript array containing the components for which
        stresses (or stress rates) are specified
    sigspec : ndarray
        Prescribed stress

    Returns
    -------
    d : ndarray || None
        If converged, the symmetric part of the velocity gradient, else None

    Notes
    -----
    The approach is an iterative scheme employing a multidimensional Newton's
    method. Each iteration begins with a call to subroutine jacobian, which
    numerically computes the Jacobian submatrix

                                  Js = J[v, v]

    where J[:,;] is the full Jacobian matrix J = dsig/deps. The value of
    d[v] is then updated according to

                d[v] = d[v] - Jsi*sigerr(d[v])/dt

    where

                   sigerr(d[v]) = sig(d[v]) - sigspec

    The process is repeated until a convergence critierion is satisfied. The
    argument converged is a flag indicat- ing whether or not the procedure
    converged:

    """
    depsmag = lambda a: math.sqrt(sum(a[:3] ** 2) + 2. * sum(a[3:] ** 2)) * dt

    # Initialize
    tol1, tol2 = EPS, math.sqrt(EPS)
    maxit1, maxit2, depsmax = 20, 30, .2

    sig = sigarg.copy()
    d = darg.copy()
    xtra = xtraarg.copy()

    sigsave = sig.copy()
    xtrasave = xtra.copy()

    # --- Check if strain increment is too large
    if (depsmag(d) > depsmax):
        return None

    # update the material state to get the first guess at the new stress
    sig, xtra = material.update_state(dt, d, sig, xtra, *args)
    sigerr = sig[v] - sigspec

    # --- Perform Newton iteration
    for i in range(maxit2):
        sig = sigsave.copy()
        xtra = xtrasave.copy()
        Jsub = material.jacobian(dt, d, sig, xtra, v)
        try:
            d[v] -= np.linalg.solve(Jsub, sigerr) / dt

        except:
            d[v] -= np.linalg.lstsq(Jsub, sigerr)[0] / dt
            io.log_warning("newton: using least squares approximation "
                           "to matrix inverse", limit=True)

        if (depsmag(d) > depsmax):
            # increment too large
            return None

        sig, xtra = material.update_state(dt, d, sig, xtra, *args)
        sigerr = sig[v] - sigspec
        dnom = max(np.amax(np.abs(sigspec)), 1.)
        relerr = np.amax(np.abs(sigerr) / dnom)

        if i <= maxit1 and relerr < tol1:
            return d

        elif i > maxit1 and relerr < tol2:
            return d

        continue

    # didn't converge, restore restore data and exit
    io.log_warning("newton: failed to find converged solution")
    return None


def _simplex(material, dt, darg, sigarg, xtraarg, v, sigspec, proportional,
             *args):
    """Perform a downhill simplex search to find sym_velgrad[v] such that

                        sig(sym_velgrad[v]) = sigspec[v]

    Parameters
    ----------
    material : instance
        constiutive model instance
    dt : float
        time step
    sig : ndarray
        stress at beginning of step
    xtra : ndarray
        extra variables at beginning of step
    v : ndarray
        vector subscript array containing the components for which
        stresses (or stress rates) are specified
    sigspec : ndarray
        Prescribed stress

    Returns
    -------
    d : ndarray
        the symmetric part of the velocity gradient

    """
    # --- Perform the simplex search
    d = darg.copy()
    sig = sigarg.copy()
    xtra = xtraarg.copy()
    args = (material, dt, d, sig, xtra, v, sigspec, proportional)
    d[v] = scipy.optimize.fmin(func, d[v], args=args, maxiter=20, disp=False)
    return d


def func(x, material, dt, darg, sigarg, xtraarg, v, sigspec, proportional,
         *args):
    """Objective function to be optimized by simplex

    """
    d = darg.copy()
    sig = sigarg.copy()
    xtra = xtraarg.copy()

    # initialize
    d[v] = x

    # store the best guesses
    sig, xtra = material.update_state(dt, d, sig, xtra, *args)

    # check the error
    error = 0.
    if not proportional:
        for i, j in enumerate(v):
            error += (sig[j] - sigspec[i]) ** 2
            continue

    else:
        stress_v, stress_u = [], []
        for i, j in enumerate(v):
            stress_u.append(sigspec[i])
            stress_v.append(sig[j])
            continue
        stress_v = np.array(stress_v)
        stress_u = np.array(stress_u)

        stress_u_norm = np.linalg.norm(stress_u)
        if stress_u_norm != 0.0:
            dum = (np.dot(stress_u / stress_u_norm, stress_v) *
                   stress_u / stress_u_norm)
            error = np.linalg.norm(dum) + np.linalg.norm(stress_v - dum) ** 2
        else:
            error = np.linalg.norm(stress_v)

    return error
