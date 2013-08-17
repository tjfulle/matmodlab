"""Module containing methods for computing kinematic quantities"""

import sys
import numpy as np
from numpy.linalg import inv, solve, lstsq

import utils.tensor as tensor
from utils.tensor import expm, powm, logm, sqrtm
import core.solvers as solvers
from utils.errors import Error1

I = np.eye(3)

def deps2d(dt, k, eps, depsdt):
    """Compute symmetric part of velocity gradient given depsdt

    Parameters
    ----------
    dt : float
       time step

    k : float
        Seth-Hill parameter

    eps : ndarray
        Strain

    depsdt : ndarray
        Strain rate

    Returns
    -------
    d : ndarray
        Symmetric part of velocity gradient

    Theory
    ------
    Velocity gradient L is given by

                 L = dFdt * Finv
                   = dRdt*I*Rinv + R*dUdt*Uinv*Rinv

    where F, I, R, U are the deformation gradient, identity, rotation, and
    right stretch tensor, respectively. d*dt and *inv are the rate and inverse
    or *, respectively,

    The stretch U is given by

                 if k != 0:
                     U = (k*E + I)**(1/k)

                 else:
                     U = exp(E)

    and its rate

                     dUdt = 1/k*(k*E + I)**(1/k - 1)*k*dEdt
                          = (k*E + I)**(1/k)*(k*E + I)**(-1)*dEdt
                          = U*X*dEdt

                     where X = (kE + I)**(-1)

       Then

                 d = sym(L)
                 w = skew(L)

    """
    # convert 1x6 arrays to 3x3 matrices for easier processing

    # strain
    eps = as3x3(eps)
    depsdt = as3x3(depsdt)
    epsf = eps + depsdt * dt

    # stretch and its rate
    if k == 0.:
        u = expm(epsf)
    else:
        u = powm(k * epsf + I, 1. / k)

    # center X on half step
    x = 0.5 * (inv(k * epsf + I) + inv(k * eps + I))
    du = np.dot(np.dot(u, x), depsdt)

    # velocity gradient
    L = np.dot(du, inv(u))
    d = .5 * (L + L.T)

    return as6x1(d)


def sig2d(material, dt, d, sig, xtra, v, sigspec, proportional):
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
        d = solvers.newton(material, dt, d, sig, xtra, v, sigspec)
        if d is not None:
            return d

        # --- didn't converge, try Newton's method with initial
        # --- d[v]=0.
        d = dsave.copy()
        d[v] = np.zeros(len(nv))
        d = solvers.newton(material, dt, d, sig, xtra, v, sigspec)
        if d is not None:
            return d

    # --- Still didn't converge. Try downhill simplex method and accept
    #     whatever answer it returns:
    d = dsave.copy()
    return solvers.simplex(material, dt, d, sig, xtra, v, sigspec, proportional)


def update_deformation(dt, k, f0, d):
    """From the value of the Seth-Hill parameter kappa, current strain E,
    deformation gradient F, and symmetric part of the velocit gradient d,
    update the strain and deformation gradient.

    Parameters
    ----------
    dt : float
        Time step

    k : float
        Seth-Hill parameter

    f0 : ndarray
        Deformation gradient

    d : ndarray
        Symmetric part of velocity gradient

    Returns
    -------
    f, eps : ndarray
        Deformation gradient and strain

    Theory
    ------
    Deformation gradient is given by

                 dFdt = L*F                                             (1)

    where F and L are the deformation and velocity gradients, respectively.
    The solution to (1) is

                 F = F0*exp(Lt)

    Solving incrementally,

                 Fc = Fp*exp(Lp*dt)

    where the c and p refer to the current and previous values, respectively.

    With the updated F, Fc, known, the updated stretch is found by

                 U = (trans(Fc)*Fc)**(1/2)

    Then, the updated strain is found by

                 E = 1/k * (U**k - I)

    where k is the Seth-Hill strain parameter.

    """
    # convert arrays to matrices for upcoming operations
    f0 = np.reshape(f0, (3, 3))
    d = as3x3(d)

    ff = np.dot(expm(d * dt), f0)
    u = sqrtm(np.dot(ff.T, ff))
    if k == 0:
        eps = logm(u)
    else:
        Ef = 1. / k * (powm(u, k) - I)
    if np.linalg.det(ff) <= 0.:
        raise Error1("negative Jacobian encountered")
    return np.reshape(ff, (9,)), as6x1(eps)


def as3x3(a):
    return np.array([[a[0],a[3],a[5]], [a[3],a[1],a[4]], [a[5],a[4],a[2]]])


def as6x1(a):
    return .5*(a+a.T)[[[0,1,2,0,1,0],[0,1,2,1,2,2]]]
