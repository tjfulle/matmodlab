import sys
import numpy as np
import scipy.interpolate as interpolate

TOL = 1.E-09

def extract_isotherm(rhorange, itmpr, surface):
    """Extract the isotherm from the surface

    Parameters
    ----------
    rhorange : ndarray of floats
        Density range to determine isotherm

    itmpr : float
        Temperature of isotherm

    surface : ndarray of float, shape(N, 6)
        surface[:, 0] -> density
        surface[:, 1] -> temperature
        surface[:, 2] -> energy
        surface[:, 3] -> pressure
        surface[:, 4] -> dpdt
        surface[:, 5] -> dedt

    Returns
    -------
    enrgy, pres : ndarray of floats
        Energy and pressure on isotherm

    """
    if not inrange(rhorange, surface[:, 0]):
        sys.stderr.write("extract_isotherm: density not in range "
                         "defined by surface\n")
        return
    if not inrange(itmpr, surface[:, 1]):
        sys.stderr.write("extract_isotherm: {0}: temperature not in range "
                         "defined by surface\n".format(options.getopt("type")))
        return

    # grid to interpolate the isotherm
    grid = np.column_stack((rhorange, itmpr * np.ones_like(rhorange)))

    # energy on isotherm
    enrgy = interpolate.griddata(surface[:, [0, 1]], surface[:, 2], grid)

    # pressure on isotherm
    pres = interpolate.griddata(surface[:, [0, 1]], surface[:, 3], grid)

    return enrgy, pres

def extract_hugoniot(rhorange, itmpr, surface):
    """Extract the Hugoniot from the surface

    Parameters
    ----------
    rhorange : ndarray of floats
        Density range to determine isotherm

    itmpr : float
        Initial temperature

    surface : ndarray of float, shape(N, 6)
        surface[:, 0] -> density
        surface[:, 1] -> temperature
        surface[:, 2] -> energy
        surface[:, 3] -> pressure
        surface[:, 4] -> dpdt
        surface[:, 5] -> dedt

    Returns
    -------
    enrgy, pres : ndarray of floats
        Energy and pressure on isotherm

    """
    if not inrange(rhorange, surface[:, 0]):
        sys.stderr.write("extract_hugoniot: {0}: density not in range "
                         "defined by surface\n".format(options.getopt("type")))
        return
    if not inrange(itmpr, surface[:, 1]):
        sys.stderr.write("extract_hugoniot: {0}: temperature not in range "
                         "defined by surface\n".format(options.getopt("type")))
        return

    step = np.sqrt(surface[:, 0].shape[0])

    # initial densit
    ri = rhorange[0]
    grid = np.array([[ri, itmpr]])
    ei = interpolate.griddata(surface[:, [0, 1]], surface[:, 2], grid)[0]
    pi = interpolate.griddata(surface[:, [0, 1]], surface[:, 3], grid)[0]

    # density and energy
    r = surface[::step, 0]
    e = surface[:step, 2]

    # interpolate pressure as function of density and energy
    z = surface[:, 3].reshape((step, step))
    f_p = interpolate.RectBivariateSpline(r, e, z, kx=1, ky=1, s=0)

    # interpolate dpdt as function of density and energy
    z = surface[:, 4].reshape((step, step))
    f_dpdt = interpolate.RectBivariateSpline(r, e, z, kx=1, ky=1, s=0)

    # interpolate dedt as function of density and energy
    z = surface[:, 5].reshape((step, step))
    f_dedt = interpolate.RectBivariateSpline(r, e, z, kx=1, ky=1, s=0)
    del r, z

    e = ei
    enrgy = []
    pres = []
    for rho in rhorange:
        # Here we solve the Rankine-Hugoniot equation as
        # a function of energy with constant density:
        #
        # E-E0 == 0.5*[P(E,V)+P0]*(V0-V)
        #
        # Where V0 = 1/rho0 and V = 1/rho. We rewrite it as:
        #
        # 0.5*[P(E,V)+P0]*(V0-V)-E+E0 == 0.0 = f(E)
        #
        # The derivative is given by:
        #
        # df(E)/dE = 0.5*(dP/dE)*(1/rho0 - 1/rho) - 1
        #
        # The solution to the first equation is found by a simple
        # application of newton's method:
        #
        # x_n+1 = x_n - f(E)/(df(E)/dE)

        a = (1. / ri - 1. / rho) / 2.
        maxiter = 100
        for it in range(maxiter):
            p = f_p(rho, e)[0, 0]
            f = (p + pi) * a - e + ei
            dpdt = f_dpdt(rho, e)[0, 0]
            dedt = f_dedt(rho, e)[0, 0]
            df = dpdt / dedt * a - 1.0

            e = e - f / df

            errval = abs(f / ei)
            if errval < TOL:
                break

        else:
            sys.stderr.write(
                "Max iterations reached "
                "(tol={0:14.10e}).\n".format(TOL) +
                "rel error   = {0:14.10e}\n".format(float(errval)) +
                "abs error   = {0:14.10e}\n".format(float(f)) +
                "func val    = {0:14.10e}\n".format(float(f)) +
                "ei = {0:14.10e}\n".format(float(ei)))
            return

        enrgy.append(e)
        pres.append(p)

    return np.array(enrgy), np.array(pres)


def inrange(a, b):
    """Test if the array a is in the range of b

    """
    if isinstance(a, (int, float)):
        mina = maxa = a
    else:
        mina = np.amin(a)
        maxa = np.amax(a)
    return mina >= np.amin(b) and maxa <= np.amax(b)
