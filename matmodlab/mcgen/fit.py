import os
import re
import sys
import logging
import numpy as np
try:
    import scipy.optimize as sciopt
except ImportError:
    sciopt = None

from _const import *

class CurveFitter(object):
    """CurveFitter base class"""
    name = None
    key = None
    plot_label = 'Curve fit'
    requires_opt = False
    def fit_points(self, *args, **kwargs):
        raise NotImplementedError
    def eval(self, *args, **kwargs):
        raise NotImplementedError
    def dump_info(self, *args, **kwargs):
        raise NotImplementedError

class PronyFit(CurveFitter):
    name = 'Prony'
    key = PRONY
    plot_label = r'$\sum_{i=1}^{n} y_i e^{\frac{t/a_T}{\tau_i}}$'
    def __init__(self, *args, **kwargs):
        optprony = kwargs.pop('optprony', False)
        self.optprony = optprony and sciopt is not None

    def fit_points(self, xp, yp):
        """Retuns the best fits for a Prony series

        Parameters
        ----------
        xp : ndarray
            x points (log(t/aT))
        yp : ndarray
            y points

        Returns
        -------
        fit : ndarray
            (tau_i, Y_i) for Prony series fit (last point is ( ,Y_inf))

        Notes
        -----
        Fits
                       ---
                       \        -(t/aT) / tau_i
            Y = Y_0 +  /   Y_i e
                       ---

        with a least squares fit.

        xp should be given in ascending order.

        """
        n = len(xp)
        mn = np.amin(xp)
        mx = np.amax(xp)

        # number of decade points
        ndp = int(mx - mn + 1)

        # decades
        nn = ndp + 1
        tau = np.empty(nn)
        tau[:-1] = [BASE ** (round(mn) + i) for i in range(ndp)]

        d = np.zeros((n, nn))
        d[:, -1] = 1.
        for i in range(n):
            d[i, :ndp] = np.exp(-BASE ** xp[i] / tau[:ndp])
        try:
            dtdi = np.linalg.inv(np.dot(d.T, d))
        except np.linalg.LinAlgError:
            raise ValueError('adjust initial WLF coefficients')
        ddd = np.dot(dtdi, d.T)
        coeffs = np.dot(ddd, yp)
        if self.optprony:
            # finish off the optimization.  The following optimizes both
            # tau and coeffs
            def func(xarg, *args):
                ni, xp, yp = args
                ci = xarg[:ni]
                ti = xarg[ni:]
                y = [self._eval(ti, ci, x) for x in xp]
                err = np.sqrt(np.mean((yp - y) ** 2))
                return err
            xarg = np.append(coeffs, tau)
            xopt = sciopt.fmin(func, xarg, args=(nn, xp, yp), disp=0)
            coeffs = xopt[:nn]
            tau = xopt[nn:]

        return np.column_stack((tau, coeffs))

    def _eval(self, ti, ci, z):
        s = np.sum(ci[:-1] * np.exp(-z / ti[:-1]))
        return ci[-1] + s

    def eval(self, fit, z):
        """Determine the value on the curve defined by a Prony series at z = t / aT

        Parameters
        ----------
        fit : ndarray
            Array returned by fit_points

        Returns
        -------
        val : real
            The value of the Prony series at z

        """
        ti, ci = fit[:, 0], fit[:, 1]
        return self._eval(ti, ci, BASE ** z)

    def dump_info(self, fit, ffmt='.18f', delimiter=','):
        line = []
        ti, ci = fit[:, 0], fit[:, 1]
        line.append(['tau_{0:d}'.format(i+1) for i in range(len(ti)-1)])
        line.append(['{0:{1}}'.format(r, ffmt) for r in ti[:-1]])
        line.append(['y_{0:d}'.format(i+1) for i in range(len(ci)-1)])
        line[-1].append('y_inf')
        line.append(['{0:{1}}'.format(r, ffmt) for r in ci])
        line = joinn(line, sep=delimiter)
        return line

class ModifiedPowerFit(CurveFitter):
    name = 'Modified Power'
    key = MODIFIED_POWER
    requires_opt = True
    plot_label = r'$y_0 + y_1 \left(\frac{t}{a_T}\right) ^ a$'
    def __init__(self, *args, **kwargs):
        pass

    def fit_points(self, xp, yp):
        """Retuns the best fits for a modified power curve

        Parameters
        ----------
        xp : ndarray
            x points (log(t/aT))
        yp : ndarray
            y points

        Returns
        -------
        fit : ndarray
            The fit parameters [Ee, E1, a]

        Notes
        -----
        Fits

                                  a
               Er = Ee + E1 (t/aT)

        """
        def func(p, x, y):
            return y - self._eval(p[0], p[1], p[2], x)
        out, success = sciopt.leastsq(func, [200., 100., -.1], args=(xp, yp))
        return out[:3]

    def eval(self, fit, x):
        Ee, E1, a = fit
        return self._eval(Ee, E1, a, x)

    @staticmethod
    def _eval(Ee, E1, a, x):
        return Ee + E1 * (BASE ** x) ** a

    def dump_info(self, fit, ffmt='.18f', delimiter=','):
        line = []
        line.append(['Ee', 'E1', 'a'])
        line.append(['{0:{1}}'.format(r, ffmt) for r in fit])
        line = joinn(line, sep=delimiter)
        return line

class PowerFit(CurveFitter):
    name = 'Power'
    key = POWER
    requires_opt = True
    plot_label = r'$y_0\left(\frac{t}{a_T}\right) ^ a$'
    def __init__(self, *args, **kwargs):
        pass
    def fit_points(self, xp, yp):
        """Retuns the best fits for a power curve

        Parameters
        ----------
        xp : ndarray
            x points (log(t/aT))
        yp : ndarray
            y points

        Returns
        -------
        fit : ndarray
            The fit parameters [E0, a]

        Notes
        -----
        Fits

                                  a
               Er = E0 (t/aT)

        """
        def func(p, x, y):
            return y - self._eval(p[0], p[1], x)
        out, success = sciopt.leastsq(func, [100., -.1], args=(xp, yp))
        return out[:2]

    def eval(self, fit, x):
        E0, a = fit
        return self._eval(E0, a, x)

    @staticmethod
    def _eval(E0, a, x):
        return E0 * (BASE ** x) ** a

    def dump_info(self, fit, ffmt='.18f', delimiter=','):
        line = []
        line.append(['E0', 'a'])
        line.append(['{0:{1}}'.format(r, ffmt) for r in fit])
        line = joinn(line, sep=delimiter)
        return line

class PolyFit(CurveFitter):
    name = 'Polynomial'
    key = POLYNOMIAL
    def __init__(self, *args, **kwargs):
        self.order = kwargs.get('order', 2)
        self.p = None
        pass
    @property
    def plot_label(self):
        if self.p is None:
            return r'$c_0 + c_1 x + ... + c_n x^n$'
        np = self.p.shape[0]
        l = ['c_0']
        if np > 1:
            l.append('c_1 x')
        if np > 2:
            l.extend([r'\cdots', 'c_{0} x^{0}'.format(np-1)])
        return r'${0}$'.format(' + '.join(l))
    def fit_points(self, xp, yp):
        """Retuns the best fits for a polynomial curve

        Parameters
        ----------
        xp : ndarray
            x points (log(t/aT))
        yp : ndarray
            y points

        """
        self.p = np.polyfit(xp, yp, self.order)
        return self.p

    def eval(self, fit, x):
        return np.poly1d(fit)(x)

    def dump_info(self, fit, ffmt='.18f', delimiter=','):
        line = []
        line.append(['p_{0}'.format(i) for i in range(fit.shape[0])])
        line.append(['{0:{1}}'.format(r, ffmt) for r in fit[::-1]])
        line = joinn(line, sep=delimiter)
        return line

# curve fit types
def get_fitter(key):
    fitters = CurveFitter.__subclasses__()
    for f in fitters:
        if f.key == key:
            return f
    else:
        raise ValueError('{0}: unrecognized fitter'.format(key))
    if f.requires_opt and not sciopt:
        raise ValueError('{0}: requires scipy.optimize'.format(key))
    return f
