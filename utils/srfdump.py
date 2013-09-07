import os
import sys
import argparse
import numpy as np
from utils.exodump import read_vars_from_exofile
from scipy.interpolate import RectBivariateSpline, SmoothBivariateSpline

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
        surface[:, 4] -> dedt
        surface[:, 5] -> dpdt

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
        sys.stderr.write("extract_isotherm: temperature not in range "
                         "defined by surface\n")
        return

    step = np.sqrt(surface[:, 0].shape[0])
    rho, tmpr = surface[::step, 0], surface[:step, 1]

    # energy on isotherm
    z = surface[:, 2].reshape(step, step)
    f = RectBivariateSpline(rho, tmpr, z.T)
    enrgy = np.array([f.ev(r, itmpr)[0] for r in rhorange])

    # pressure on isotherm
    z = surface[:, 3].reshape(step, step)
    f = RectBivariateSpline(rho, tmpr, z.T)
    pres = np.array([f.ev(r, itmpr)[0] for r in rhorange])

    return enrgy, pres


def extract_hugoniot(rhorange, itmpr, srf):
    """Extract the Hugoniot from the surface

    Parameters
    ----------
    rhorange : ndarray of floats
        Density range to determine isotherm

    itmpr : float
        Initial temperature

    srf : ndarray of float, shape(N, 6)
        srf[:, 0] -> density
        srf[:, 1] -> temperature
        srf[:, 2] -> energy
        srf[:, 3] -> pressure
        srf[:, 4] -> dedt
        srf[:, 5] -> dpdt

    Returns
    -------
    enrgy, pres : ndarray of floats
        Energy and pressure on isotherm

    """
    if not inrange(rhorange, srf[:, 0]):
        sys.stderr.write("extract_hugoniot: density not in range "
                         "defined by surface\n")
        return
    if not inrange(itmpr, srf[:, 1]):
        sys.stderr.write("extract_hugoniot: temperature not in range "
                         "defined by surface\n")
        return

    s = np.sqrt(srf[:, 0].shape[0])

    # initial energy and pressure on grid
    f = RectBivariateSpline(srf[::s, 0], srf[:s, 1], srf[:, 2].reshape(s, s).T)
    ei = f.ev(rhorange[0], itmpr)[0]
    f = RectBivariateSpline(srf[::s, 0], srf[:s, 1], srf[:, 3].reshape(s, s).T)
    pi = f.ev(rhorange[0], itmpr)[0]
    del f

    # t, p, dedt, dpdt as functions of density and energy
    x, y = srf[:, 0], srf[:, 2]
    f_t = SmoothBivariateSpline(x, y, srf[:, 1])
    f_p = SmoothBivariateSpline(x, y, srf[:, 3])
    f_dedt = SmoothBivariateSpline(x, y, srf[:, 4])
    f_dpdt = SmoothBivariateSpline(x, y, srf[:, 5])

    e = ei
    enrgy = []
    pres = []
    tmpr = []
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

        a = (1. / rhorange[0] - 1. / rho) / 2.
        for it in range(100):
            p = f_p.ev(rho, e)[0]
            f = (p + pi) * a - e + ei
            dpdt = f_dpdt.ev(rho, e)[0]
            dedt = f_dedt.ev(rho, e)[0]
            df = dpdt / dedt * a - 1.0

            e = e - f / df

            if abs(f) < TOL:
                break

        else:
            sys.stderr.write(
                "Max iterations reached "
                "(tol={0:14.10e}).\n".format(TOL) +
                "rel error   = {0:14.10e}\n".format(float(f / ei)) +
                "abs error   = {0:14.10e}\n".format(float(f)) +
                "func val    = {0:14.10e}\n".format(float(f)) +
                "ei          = {0:14.10e}\n".format(float(ei)))
            return

        enrgy.append(e)
        pres.append(p)
        tmpr.append(f_t.ev(rho, e)[0])

    return np.array(enrgy), np.array(pres), np.array(tmpr)


def inrange(a, b):
    """Test if the array a is in the range of b

    """
    if isinstance(a, (int, float)):
        mina = maxa = a
    else:
        mina = np.amin(a)
        maxa = np.amax(a)
    return mina >= np.amin(b) and maxa <= np.amax(b)


def read_from_command_line(argv=None):
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import cm
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.ticker import LinearLocator, FormatStrFormatter

    if argv is None:
        argv = sys.argv[1:]
    parser = argparse.ArgumentParser()
    parser.add_argument("source")
    args = parser.parse_args(argv)
    assert os.path.isfile(args.source)

    variables=["RHO", "TMPR", "ENRGY", "PRES", "DEDT", "DPDT"]
    surface = read_vars_from_exofile(args.source, variables=variables, h=0)[:, 1:]
    step = np.sqrt(surface[:, 0].shape[0])
    r = surface[::step, 0]
    t = surface[:step, 1]

    x, y = np.meshgrid(r, t)
    es = surface[:, 2].reshape((step, step))
    ps = surface[:, 3].reshape((step, step))

    fig = plt.figure()
    axs = fig.gca(projection='3d')
    axs.plot_wireframe(x, y, ps, rstride=10, cstride=10)

    axs.zaxis.set_major_locator(LinearLocator(10))
    axs.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    # isotherm
#    enrgy, pres = extract_isotherm(r, np.mean(t), surface)
#    axl = fig.gca(projection="3d")
#    axl.plot(r, np.mean(t) * np.ones_like(r), pres, c="r")
#    plt.show()

    # hugoniot
    enrgy, pres, tmpr = extract_hugoniot(r, np.mean(t), surface)
    axl = fig.gca(projection="3d")
    axl.plot(r, tmpr, pres, c="r")
    plt.show()


if __name__ == "__main__":
    read_from_command_line()
