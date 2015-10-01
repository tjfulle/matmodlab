import numpy as np
from .nonmonotonic import calculate_bounded_area

__all__ = ['SAME', 'DIFF', 'NOT_SAME', 'ERRORS', 'DIFFTOL', 'FAILTOL', 'FLOOR',
           'afloor', 'amag', 'rms_error', 'interp_rms_error', 'diff_data_sets',
           'calculate_bounded_area']
SAME = 0
DIFF = 1
NOT_SAME = 2
ERRORS = 2
DIFFTOL = 1.5E-06
FAILTOL = 1.E-04
FLOOR = 1.E-12

def afloor(a, floor):
    a[np.where(np.abs(a) <= floor)] = 0.
    return a

def amag(a):
    return np.sqrt(np.sum(a * a))

def rms_error(t1, d1, t2, d2, disp=1):
    """Compute the RMS and normalized RMS error

    """
    t1 = np.asarray(t1)
    d1 = np.asarray(d1)
    t2 = np.asarray(t2)
    d2 = np.asarray(d2)

    if t1.shape[0] == t2.shape[0]:
        rms = np.sqrt(np.mean((d1 - d2) ** 2))
    else:
        rms = interp_rms_error(t1, d1, t2, d2)
    dnom = np.amax(np.abs(d1))
    if dnom < 1.e-12: dnom = 1.
    if disp:
        return rms, rms / dnom
    return rms / dnom


def interp_rms_error(t1, d1, t2, d2):
    """Compute RMS error by interpolation

    """
    ti = max(np.amin(t1), np.amin(t2))
    tf = min(np.amax(t1), np.amax(t2))
    n = t1.shape[0]
    f1 = lambda x: np.interp(x, t1, d1)
    f2 = lambda x: np.interp(x, t2, d2)
    rms = np.sqrt(np.mean([(f1(t) - f2(t)) ** 2
                           for t in np.linspace(ti, tf, n)]))
    return rms

def diff_data_sets(head1, data1, head2, data2, vars_to_compare,
                   stream, interp=False):
    """Diff the files

    """
    head1 = [s.upper() for s in head1]
    head2 = [s.upper() for s in head2]

    warn = lambda s: stream.write('***warning: {0}\n'.format(s))
    error = lambda s: stream.write('***error: {0}\n'.format(s))
    info = lambda s, end='\n': stream.write('{0}{1}'.format(s, end))

    # Compare times first
    try:
        t1 = data1[:, head1.index("TIME")]
    except:
        error("TIME not in File1")
        return NOT_SAME
    try:
        t2 = data2[:, head2.index("TIME")]
    except:
        error("TIME not in File2")
        return NOT_SAME

    if not interp:
        # interpolation will not be used when comparing values, so the
        # timesteps must be equal
        if t1.shape[0] != t2.shape[0]:
            error("Number of timesteps in File1({0:d}) and "
                  "File2({1:d}) differ".format(t1.shape[0], t2.shape[0]))
            return NOT_SAME

        if not np.allclose(t1, t2, atol=FAILTOL, rtol=FAILTOL):
            error("Timestep size in File1 and File2 differ")
            return NOT_SAME

    status = []
    bad = [[], []]
    for (var, dtol, ftol, floor) in vars_to_compare:

        if var == "TIME":
            continue

        try:
            i1 = head1.index(var)
        except ValueError:
            warn("{0}: not in File1\n".format(var))
            continue

        try:
            i2 = head2.index(var)
        except ValueError:
            warn("{0}: not in File2\n".format(var))
            continue

        d1 = afloor(data1[:, i1], floor)
        d2 = afloor(data2[:, i2], floor)

        info("Comparing {0}".format(var), end="." * (40 - len(var)))

        if not interp:
            if np.allclose(d1, d2, atol=ftol, rtol=ftol):
                info(" pass")
                info("File1.{0} := File2.{0}\n".format(var))
                status.append(SAME)
                continue

        if amag(d1) < 1.e-10 and amag(d2) < 1.e-10:
            info(" pass")
            info("File1.{0} = File2.{0} = 0\n".format(var))
            status.append(SAME)
            continue

        rms, nrms = rms_error(t1, d1, t2, d2)
        if nrms < dtol:
            info(" pass")
            info("File1.{0} == File2.{0}".format(var))
            status.append(SAME)

        elif nrms < ftol:
            info(" diff")
            warn("File1.{0} ~= File2.{0}".format(var))
            status.append(DIFF)
            bad[1].append(var)

        else:
            info(" fail")
            error("File1.{0} != File2.{0}".format(var))
            status.append(NOT_SAME)
            bad[0].append(var)

        info("NRMS(File.{0}, File2.{0}) = {1: 12.6E}\n".format(var, nrms))
        continue

    failed = ", ".join("{0}".format(f) for f in bad[0])
    diffed = ", ".join("{0}".format(f) for f in bad[1])
    if failed:
        info("Variabes that failed: {0}".format(failed))
    if diffed:
        info("Variabes that diffed: {0}".format(diffed))

    return max(status)
