import os
import re
import sys
import logging
import numpy as np
from pandas import DataFrame

try: import scipy.optimize as sciopt
except ImportError: sciopt = None
try: import matplotlib.pyplot as plt
except ImportError: plt = None
try: import bokeh.plotting as bp
except ImportError: bp = None

from fit import *
from _const import *

class Environment:
    pass
environ = Environment()
environ.notebook = 0

class MasterCurve(object):
    fiterr = None
    def __init__(self, txy, ref_temp=75., apply_log=False, xfac=1., yfac=1.,
                 skip_temps=None, wlf_coeffs=None,
                 xvar='Time', xunits='min', yvar='Er', yunits='psi',
                 optimizer=FMIN, fitter=PRONY, optwlf=sciopt is not None,
                 **kwargs):
        """Initialize the master curve object

        Parameters
        ----------
        txy : array_like (n, 3)
            txy[i] is the ith [Temp, X, Y]
        ref_temp : real
            Reference temperature
        optimizer : str [FMIN]
            The scipy.optimize optimizing procedure
        optwlf : bool [True]
            Optimize the WLF parameters
        fitter : str [prony]
            Name of CurveFitter
        skip_temps : list [None]
            Temperatures to skip
        wlf_coeffs : list [None]
            Initial guesses for C1 and C2

        kwargs : dict
            keywords [optional] to pass to fitter

        """
        columns = ('Temp', 'X', 'Log[X]', 'Y')
        txy = np.asarray(txy)
        txy[:, -1] *= yfac
        if apply_log:
            txy[:, 1] *= xfac
            logx = log(txy[:, 1])
            txy = np.insert(txy, 2, logx, axis=1)
        else:
            x = (BASE ** txy[:, 1]) * xfac
            logx = log(x)
            txy = np.insert(txy, 1, x, axis=1)
            txy = np.insert(txy, 2, logx, axis=1)

        self.ref_temp = ref_temp
        self.skip_temps = aslist(skip_temps)
        self.df = DataFrame(txy, columns=columns)

        self.wlf_coeffs = wlf_coeffs
        self.optwlf = optwlf
        self.optimizer = optimizer
        cf = get_fitter(fitter)
        self.cf = cf(**kwargs)

        self.xvar = xvar
        self.yvar = yvar
        self.xunits = xvar
        self.yunits = yunits

        self._fit = 0

    def fit(self, wlf_coeffs=None, skip_temps=None, ref_temp=None, optimize=None):

        skip_temps = aslist(skip_temps)
        skip_temps.extend(skip_temps)

        # make skip temps a list, if not already
        df = self.df.copy()
        for temp in skip_temps:
            df = df[~(np.abs(df['Temp'] - temp) < EPS)]

        ref_temp = self.ref_temp if ref_temp is None else ref_temp
        wlf_coeffs = self.wlf_coeffs if wlf_coeffs is None else wlf_coeffs

        if not any(np.abs(df['Temp'] - ref_temp) < EPS):
            raise ValueError('Reference temperature {0} not '
                             'found in data'.format(ref_temp))

        wlf_opt = self.opt_wlf_coeffs(df, ref_temp, wlf_coeffs, optimize)
        self.dfm = self.shift_data(df, ref_temp, wlf_opt)
        self.mc_fit = self.fit_shifted_data(self.dfm)
        self.wlf_opt = wlf_opt
        self._fit = 1

    def opt_wlf_coeffs(self, df, ref_temp, wlf_coeffs, opt):
        """Generate the optimized master curve"""

        temps = np.unique(df['Temp'])
        if len(temps) == 1:
            # only one data set
            return np.zeros(2)

        if wlf_coeffs is None:
            wlf_coeffs = self.get_wlf_coeffs(df, ref_temp)

        if opt is None:
            opt = self.optwlf

        if not opt:
            return wlf_coeffs

        def func(xopt, *args):
            """Objective function returning the area between the fitted curve
            and shifted data

            """
            if np.any(np.abs(xopt[1] + temps - ref_temp) < EPS):
                self.fiterr = 1000.
                return self.fiterr

            df1 = self.shift_data(df.copy(), ref_temp, xopt)
            fit = self.fit_shifted_data(df1)

            # determine error between fitted curve and master curve
            yvals = []
            for logx in df1['Log[X/aT]']:
                yvals.append(self.cf.eval(fit, logx))
            yvals = np.asarray(yvals)
            error = np.sqrt(np.mean((yvals - df1['Y']) ** 2))
            self.fiterr = error  # / area(data[:,0],data[:,1])
            return self.fiterr

        if self.optimizer == COBYLA:
            cons = [lambda x: 1 if abs(x[1]+temp-ref_temp) > EPS else -1
                    for temp in temps]
            wlf_coeffs = sciopt.fmin_cobyla(func, wlf_coeffs, cons, disp=0)
        elif self.optimizer == POWELL:
            wlf_coeffs = sciopt.fmin_powell(func, wlf_coeffs, disp=0)
        else:
            wlf_coeffs = sciopt.fmin(func, wlf_coeffs, disp=0)

        return wlf_coeffs

    def shift_data(self, df, ref_temp, wlf):
        """Compute the master curve for data series"""
        # reference temperature curve
        def f(x):
            temp = np.asarray(x['Temp'])[0]
            shift = -wlf[0] * (temp - ref_temp) / (wlf[1] + temp - ref_temp)
            x['Log[X/aT]'] = np.asarray(x['Log[X]']) - shift
            return x
        df = df.groupby('Temp').apply(f)
        return df

    def fit_shifted_data(self, df):
        """Fit the master curve

        """
        t = np.asarray(df['Log[X/aT]'])
        d = np.asarray(df['Y'])
        return self.cf.fit_points(t, d)

    @staticmethod
    def x_shift(df1, df2):
        """

        Parameters
        ----------
        df1 : ndarray
            Base curve to shift to consisting of list of x,y pairs
        df2 : ndarray
            Curve to shift consisting of list of x,y pairs

         Returns
         -------
         shift : real
             A scalar shift value

        Notes
        -----
        Single curves must be monotonically increasing or decreasing. Composite
        curve can be more complex and x values should generally be increasing
        (noise is okay)

        shift value returned is such that x points should be shifted by
        subtracting the shift value

        """
        ref_curve = np.asarray(df1[['Log[X]', 'Y']])
        curve = np.asarray(df2[['Log[X]', 'Y']])

        ref_bnds = bounding_box(ref_curve)
        crv_bnds = bounding_box(curve)

        if (crv_bnds[1, 1] > ref_bnds[1, 1]):
            # y values for curve larger than ref_curve, check for overlap
            if (crv_bnds[0, 1] < ref_bnds[1, 1]):
                ypt = (ref_bnds[1, 1] + crv_bnds[0, 1]) / 2.
                x_crv = interp1d(curve, ypt, findx=True)
                x_ref = interp1d(ref_curve, ypt, findx=True)
            else:
                # no overlap
                x_ref, x_crv = ref_curve[0, 0], curve[-1, 0]
        else:
            # y values for ref_curve larger than curve, check for overlap
            if (ref_bnds[0, 1] < crv_bnds[1, 1]):
                ypt = (ref_bnds[0, 1] + crv_bnds[1, 1]) / 2.
                x_crv = interp1d(curve, ypt, findx=True)
                x_ref = interp1d(ref_curve, ypt, findx=True)
            else:
                # no overlap
                x_ref, x_crv = ref_curve[-1, 0], curve[0, 0]

        return -(x_ref - x_crv)

    def get_wlf_coeffs(self, df, T0):
        """Defines the WLF shift

        Notes
        -----
        returns a tuple containing best fits for C1 and C2:

         log(aT) = -C1(T-ref_temp)/(C2+T-ref_temp)

         The coefficients are determined by the least squares fit

              x = (A^TA)^-1 A^Tb  (T is matrix transpose)

         where A is nx2 matrix (n is number of T and logaT pairs) of rows or
         [(T-Tr) logaT] for each T,logaT pair b is a nx1 vector where each row is
         -logaT*(T-Tr) for each pair x is a 2x1 vector of C1 and C2

        """
        # shift the data to the reference curve
        dg = df.groupby('Temp')
        rc = dg.get_group(T0)

        # Shift each data set.  xs is [T, logaT]
        xs = np.array([[g, self.x_shift(rc, df)] for (g, df) in dg])
        if all([abs(x) < 1.e-6 for x in xs[:,1]]):
            raise WLFError('initial WLF coefficients required for data set')

        # Setup A Matrix
        A = np.zeros((xs.shape[0], 2))
        A[:, 0] = xs[:,0] - T0
        A[:, 1] = xs[:,1]

        # Setup b Vector
        b = -A[:, 0] * A[:, 1]

        # Calculate WLF Coefficients
        ATA = np.dot(A.T, A)
        ATb = np.dot(A.T, b)
        try:
            wlf_coeffs = np.linalg.solve(ATA, ATb)
        except np.linalg.LinAlgError:
            logging.warn('using least squares to find wlf coefficients')
            wlf_coeffs = np.linalg.lstsq(ATA, ATb)[0]

        return wlf_coeffs

    def plot(self, **kwargs):
        if environ.notebook == 2:
            return self._bp_plot(**kwargs)
        self._mp_plot(**kwargs)

    def _bp_plot(self, raw=False, show_fit=False):
        if bp is None:
            raise ImportError('bokeh')

        x_label = 'Log[{0}/aT] ({1})'.format(self.xvar, self.xunits)
        y_label = '{0} ({1})'.format(self.yvar, self.yunits)
        plot = bp.figure(x_axis_label=x_label, y_axis_label=y_label)
        if raw:
            plot.title = 'Raw, unshifted data'
            dg = self.df.groupby('Temp')
            colors = gen_colors([str(temp) for temp in dg.groups.keys()])
            for (temp, df) in dg:
                plot.scatter(df['Log[X]'], df['Y'], legend='{0}'.format(temp),
                             color=colors[str(temp)])
            return plot

        if not self._fit:
            self.fit()

        dg = self.dfm.groupby('Temp')
        colors = gen_colors([str(temp) for temp in dg.groups.keys()])
        plot.title = 'Data shift: {0}'.format(self.cf.name)
        for (temp, df) in dg:
            plot.scatter(df['Log[X/aT]'], df['Y'],
                         legend='{0}'.format(temp), color=colors[str(temp)])
        if show_fit:
            xp, yp = self._mc_points()
            plot.line(xp, yp, color='black', line_width=1.5,
                      legend='Master Curve Fit')
        return plot

    def _mp_plot(self, raw=False, show_fit=False, filename=None,
                 legend_loc='best', legend_ncol=1):
        """Plot the given data or shifted data and/or the fit """
        if plt is None:
            raise ImportError('matplotlib')

        plt.clf()
        plt.cla()
        xl, xu = self.xvar, self.xunits
        yl, yu = self.yvar, self.yunits
        plt.xlabel(r'$\log\left(\frac{%s}{aT}\right)$ (%s)' % (xl, xu))
        ylabel = r'$%s$ (%s)' % (yl, yu)
        plt.ylabel(ylabel)

        if raw:
            dg = self.df.groupby('Temp')
            colors = gen_colors([str(temp) for temp in dg.groups.keys()])
            for (temp, df) in dg:
                plt.scatter(df['Log[X]'], df['Y'], label='{0}'.format(temp),
                            color=colors[str(temp)])

        else:
            if not self._fit:
                self.fit()
            dg = self.dfm.groupby('Temp')
            colors = gen_colors([str(temp) for temp in dg.groups.keys()])
            for (temp, df) in dg:
                plt.scatter(df['Log[X/aT]'], df['Y'],
                            label='{0}'.format(temp), color=colors[str(temp)])

            if show_fit:
                xp, yp = self._mc_points()
                plt.plot(xp, yp, 'k-', lw=1.5)

        if legend_loc is not None:
            kwds = {'loc': legend_loc, 'ncol': legend_ncol,
                    'scatterpoints': 1}
            if show_fit:
                kwds['title'] = '${0}$ = {1}'.format(self.yvar, self.cf.plot_label)
                kwds['fancybox'] = True
            plt.legend(**kwds)

        if environ.notebook:
            plt.show()
        elif filename is None:
            plt.show()
        if filename is not None:
            plt.savefig(filename, transparent=True)

        return

    def to_csv(self, filename):
        """Dump data to a file

        Parameters
        ----------
        filename : str
            file path

        Notes
        -----
        Writes to each row of filename

        """
        if not self._fit:
            self.fit()

        fh = open(filename, 'w')
        fh.write('mcgen\nVersion\n1.2\n')
        fh.write('Curve Fitter\n{0}\n'.format(self.cf.name))
        fh.write('Curve Fitting Information\n')
        # write out the fit info
        line = self.cf.dump_info(self.mc_fit, delimiter=',')
        fh.write(line)
        try:
            fh.write('Fit Error\n{0:.18f}\n'.format(self.fiterr))
        except ValueError:
            pass
        fh.write(joinn(['WLF C1', 'WLF C2'], sep=','))
        fh.write(joinn([self.wlf_opt[0], self.wlf_opt[1]], sep=',', num=True))

        fh.write('Data\n')
        self.dfm.to_csv(fh, float_format='%.18f', index=False)
        fh.close()

    def _mc_points(self, n=50):
        xmin = np.amin(self.dfm['Log[X/aT]'])
        xmax = np.amax(self.dfm['Log[X/aT]'])
        xvals = np.linspace(xmin, xmax, n)
        yvals = np.array([self.cf.eval(self.mc_fit, x) for x in xvals])
        return xvals, yvals

def read_csv(filename, apply_log=True, ref_temp=75., cols=[0,1,2],
             xvar='Time', xunits='min', yvar='Er', yunits='psi',
             skip_temps=None, xfac=1., yfac=1., **kwargs):
    """Read data from filename

    Parameters
    ----------
    filename : str
        File in which data is found

    Returns
    -------
    all_data : tuple of (real, ndarray) pairs
        (temp, data) pairs where temp is the temperature corresponding
        to the (log(x), y) curves in data

    Notes
    -----
    Each line of filename must be formatted as

      Temperature, x, y

    """
    if not os.path.isfile(filename):
        raise OSError('***error: {0}: no such file'.format(filename))
    lines = open(filename, 'r').readlines()
    data = []
    for line in lines:
        line = [float(x) for x in re.split(r'[ \,]', line.split('#', 1)[0])
                if x.split()]
        if not line:
            continue
        data.append(line)
    data = np.array(data)[:,cols]
    data = np.asarray(sorted(data, key=lambda x: (x[0], x[1])))
    return MasterCurve(data, ref_temp=ref_temp, apply_log=apply_log,
                       skip_temps=skip_temps, xfac=xfac, yfac=yfac,
                       xvar=xvar, xunits=xunits, yvar=yvar, yunits=yunits,
                       **kwargs)

def show(plot):
    if environ.notebook == 2:
        bp.show(plot)

def init_notebook(plot_lib='bokeh'):
    if plot_lib == 'bokeh':
        if bp is None:
            raise ImportError('bokeh')
        from bokeh.io import output_notebook
        output_notebook()
        environ.notebook = 2
    elif plot_lib == 'matplotlib':
        if plt is None:
            raise ImportError('matplotlib')
        plt.rcParams['figure.figsize'] = (15, 12)
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.size'] = 20
        plt.rcParams['font.serif'] = 'Times New Roman'
        plt.rcParams['legend.scatterpoints'] = 1
        plt.rcParams['legend.handlelength'] = 0
        environ.notebook = 1
    else:
        raise ValueError('expected bokeh or matplotlib, got {0!r}'.format(plot_lib))
