import numpy as np
from itertools import cycle
MODIFIED_POWER = 1
POWER = 2
PRONY = 3
POLYNOMIAL = 4

COBYLA = 1
POWELL = 2
FMIN = 3

PHONY = 123456.789
EPS = np.finfo(float).eps
BASE = 10.

def loadcsv(filename):
    """Load the csv file written out by MasterCurve.dump"""
    class CSVData:
        data = {}
        def get(self, temp):
            return self.data[temp]
        def __iter__(self):
            return iter(self.data.items())
    dtype = np.float64
    array = np.array
    cstack = np.column_stack

    assert os.path.isfile(filename)
    lines = open(filename).readlines()

    for (i, line) in enumerate(lines):
        if re.search('Version', line):
            version = float(lines[i+1])
            assert version > 1.
        if re.search('Curve Fitter', line):
            fitter = lines[i+1]
        if re.search('Fit Error', line):
            fiterr = float(lines[i+1])
        if re.search('WLF', line):
            wlf_coeffs = [float(x) for x in lines[i+1].split(',')]
            assert len(wlf_coeffs) == 2

        if re.search('Data', line):
            j = i + 1

        if re.search('Master Curve', line):
            k = i + 1

    d = CSVData()

    desc = lines[j].split(',')
    temps = array([float(x) for x in desc[2:]])
    data = array([[None if not x.split() else float(x) for x in y.split(',')]
                  for y in lines[j+1:k-1]])
    d.master = array([[float(x) for x in y.split(',')] for y in lines[k+1:]])

    d.toat = data[:,0]
    d.logtoat = data[:,1]

    d.raw_master = cstack((d.toat, d.logtoat, array(data[:,2], dtype=dtype)))
    d.ref_temp = temps[0]

    d.temps = temps[1:]
    for (i, temp) in enumerate(temps[1:]):
        td = data[:,i+3]
        idx = [j for (j,x) in enumerate(td) if x is not None]
        a = array(td[idx], dtype=dtype)
        if len(a) == 0:
            continue
        a = cstack((d.toat[idx], d.logtoat[idx], a))
        d.data[temp] = a

    return d

def log(x, base=BASE):
    e = 2.718281828459045
    if abs(base - 10.) < EPS:
        return np.log10(x)
    elif abs(base - 2.) < EPS:
        return np.log2(x)
    elif abs(base - e) < EPS:
        return np.log(x)
    else:
        return np.log(x) / np.log(BASE)

def interp1d(xy, x, findx=False, clip=False):
    """Wrapper around numpy's interp

    """
    xp = xy[:, 0]
    yp = xy[:, 1]
    if findx:
        xp, yp = yp, xp
    xd = np.diff(xp)
    if np.allclose(-1, np.sign(np.diff(xp))):
        # descending curve, reverse it
        xp, yp = xp[::-1], yp[::-1]
    if not clip:
        return np.interp(x, xp, yp)

    yval = np.interp(x, xp, yp, left=PHONY, right=PHONY)
    if abs(yval - PHONY) < 1.e-12:
        return None
    return yval

def islist(a):
    return isinstance(a, (list, tuple, np.ndarray))

def multidim(a):
    try:
        if islist(a) and islist(a[0]):
            return True
    except IndexError:
        return False

def joinn(l, sep=',', num=False, end='\n', ffmt='.18f'):
    if num:
        realfmt = lambda r: '{0:{1}}'.format(float(r), ffmt)
        l = [realfmt(x) if x is not None else '' for x in l]
    if not multidim(l):
        line = sep.join(l)
    else:
        line = '\n'.join(sep.join(s) for s in l)
    return line + end

def bounding_box(curve):
    """Determine the box that bounds curve

    Parameters
    ----------
    curve : ndarray
        curve[i, 0] is the x coordinate of the ith data point
        curve[i, 1] is the y coordinate of the ith data point

    Returns
    -------
    box : ndarray
        box[0, i] is xm[in,ax]
        box[1, i] is ym[in,ax]

    """
    if curve[0, 1] > curve[-1, 1]:
        # y values are decreasing from left to right
        xmin, ymin = curve[-1, 0], curve[-1, 1]
        xmax, ymax = curve[0, 0], curve[0, 1]
    else:
        xmin, ymin = curve[0, 0], curve[0, 1]
        xmax, ymax = curve[-1, 0], curve[-1, 1]
    return np.array([[xmin, ymin], [xmax, ymax]])

def area(x, y, yaxis=False):
    if not yaxis:
        return np.trapz(y, x)
    return np.trapz(x, y)

COLORS = ['Blue', 'Red', 'Purple', 'Green', 'Orange', 'HotPink', 'Cyan',
          'Magenta', 'Chocolate', 'Yellow', 'Black', 'DodgerBlue', 'DarkRed',
          'DarkViolet', 'DarkGreen', 'OrangeRed', 'Teal', 'DarkSlateGray',
          'RoyalBlue', 'Crimson', 'SeaGreen', 'Plum', 'DarkGoldenRod',
          'MidnightBlue', 'DarkOliveGreen', 'DarkMagenta', 'DarkOrchid',
          'DarkTurquoise', 'Lime', 'Turquoise', 'DarkCyan', 'Maroon']

def gen_colors(keys):
    colors = {}
    c = cycle(COLORS)
    for key in keys:
        colors[key] = next(c).lower()
    return colors

def aslist(item):
    if item is None:
        return []
    if not isinstance(item, (list, tuple, np.ndarray)):
        item = [skip_temps]
    return [x for x in item]
