import os
import sys
import time
import numpy as np
import argparse
import xml.dom.minidom as xdom

from exoreader import ExodusIIReader

class Logger(object):
    def __init__(self):
        self.log = sys.stdout
    def info(self, message):
        self.log.write(message + "\n")
    def warning(self, message):
        self.log.write("*** warning: {0}\n".format(message))
    def error(self, message):
        self.log.write("*** error: {0}\n".format(message))
LOG = Logger()
DIFFTOL = 1.E-06
FAILTOL = 1.E-04
FLOOR = 1.E-12


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    parser = argparse.ArgumentParser()
    parser.add_argument("-f",
        help=("Use the given file to specify the variables to be considered "
              "and to what tolerances [default: %(default)s]."))
    parser.add_argument("source1")
    parser.add_argument("source2")
    args = parser.parse_args(argv)

    if not os.path.isfile(args.source1):
        error("{0}: no such file".format(args.source1))
    if not os.path.isfile(args.source2):
        error("{0}: no such file".format(args.source2))

    H1, D1 = loadcontents(args.source1)
    H2, D2 = loadcontents(args.source2)

    if args.f is not None:
        if not os.path.isfile(args.f):
            LOG.error("{0}: no such file".format(args.f))
            return 2
        variables = read_diff_file(args.f)
    else:
        variables = zip(H1, [DIFFTOL] * len(H1), [FAILTOL] * len(H1),
                        [FLOOR] * len(H1))


    status = diff_files(H1, D1, H2, D2, variables)

    if status == 0:
        LOG.info("Files are the same")
    elif status == 1:
        LOG.info("Files diffed")
    else:
        LOG.info("Files are different")
    return status


def loadcontents(filepath):
    if filepath.endswith((".exo", ".e", ".base_exo")):
        return loadexo(filepath)
    return loadascii(filepath)


def loadexo(filepath):
    LOG.info("Reading {0}".format(filepath))
    exof = ExodusIIReader.new_from_exofile(filepath)
    glob_var_names = exof.glob_var_names()
    elem_var_names = exof.elem_var_names()
    data = [exof.get_all_times()]
    for glob_var_name in glob_var_names:
        data.append(exof.get_glob_var_time(glob_var_name))
    for elem_var_name in elem_var_names:
        data.append(exof.get_elem_var_time(elem_var_name, 0))
    data = np.transpose(np.array(data))
    head = ["TIME"] + glob_var_names + elem_var_names
    exof.close()
    return head, data


def loadascii(filepath):
    LOG.info("Reading {0}".format(filepath))
    head = loadhead(filepath)
    data = loadtxt(filepath, skiprows=1)
    return head, data


def loadhead(filepath, comments="#"):
    """Get the file header

    """
    line = " ".join(x.strip() for x in linecache.getline(filepath, 1).split())
    if line.startswith(comments):
        line = line[1:]
    return line.split()


def loadtxt(f, skiprows=0, comments="#"):
    """Load text from output files

    """
    lines = []
    for (iline, line) in enumerate(open(f, "r").readlines()[skiprows:]):
        try:
            line = [float(x) for x in line.split(comments, 1)[0].split()]
        except ValueError:
            break
        if not lines:
            ncols = len(line)
        if len(line) < ncols:
            break
        if len(line) > ncols:
            stop("*** {0}: error: {1}: inconsistent data in row {1}".format(
                EXE, os.path.basename(f), iline))
        lines.append(line)
    return np.array(lines)


def diff_files(head1, data1, head2, data2, vars_to_compare):
    """Diff the files

    """
    # Compare times first
    try:
        t1 = data1[:, head1.index("TIME")]
    except:
        LOG.error("TIME not in File1")
        return 2
    try:
        t2 = data2[:, head2.index("TIME")]
    except:
        LOG.error("TIME not in File2")
        return 2

    if t1.shape[0] != t2.shape[0]:
        LOG.error("Number of timesteps in File1 and File2 differ")
        return 2

    if not np.allclose(t1, t2, atol=FAILTOL, rtol=FAILTOL):
        LOG.error("Timestep size in File1 and File2 differ")
        return 2

    status = []
    for (var, dtol, ftol, floor) in vars_to_compare:

        if var == "TIME":
            continue

        try:
            i1 = head1.index(var)
        except IndexError:
            LOG.warning("{0}: not in File1")
            continue

        try:
            i2 = head2.index(var)
        except IndexError:
            LOG.warning("{0}: not in File2")
            continue

        d1 = afloor(data1[:, i1], floor)
        d2 = afloor(data2[:, i2], floor)

        LOG.info("Comparing {0}".format(var))
        if np.allclose(d1, d2, atol=ftol, rtol=ftol):
            LOG.info("File1.{0} == File2.{0}\n".format(var))
            status.append(0)
            continue

        rms, nrms = rms_error(t1, d1, t2, d2)
        if nrms < dtol:
            LOG.info("File1.{0} == File2.{0}".format(var))
            status.append(0)

        elif nrms < ftol:
            LOG.info("File1.{0} ~= File2.{0}".format(var))
            status.append(1)

        else:
            LOG.info("File1.{0} != File2.{0}".format(var))
            status.append(2)

        LOG.info("NRMS(File.{0}, File2.{0}) = {1: 12.6E}\n".format(var, nrms))
        continue

    return max(status)


def rms_error(t1, d1, t2, d2):
    if t1.shape[0] == t2.shape[0]:
        rms = np.sqrt(np.mean((d1 - d2) ** 2))
    else:
        rms = interp_rms_error(t1, d1, t2, d2)
    dnom = np.amax(np.abs(d1))
    if dnom < 1.e-12: dnom = 1.
    return rms, rms / dnom


def interp_rms_error(t1, d1, t2, d2):
    ti = max(np.amin(t1), np.amin(t2))
    tf = min(np.amax(t1), np.amax(t2))
    n = t1.shape[0]
    f1 = lambda x: np.interp(x, t1, d1)
    f2 = lambda x: np.interp(x, t2, d2)
    rms = np.sqrt(np.mean([(f1(t) - f2(t)) ** 2
                           for t in np.linspace(ti, tf, n)]))
    return rms


def read_diff_file(filepath):
    doc = xdom.parse(filepath)
    try:
        gmddiff = doc.getElementsByTagName("GMDDiff")[0]
    except IndexError:
        LOG.error("{0}: expected root element 'GMDDiff'".format(filepath))
        sys.exit(2)
    ftol = gmddiff.getAttribute("ftol")
    if ftol: ftol = float(ftol)
    else: ftol = FAILTOL
    dtol = gmddiff.getAttribute("dtol")
    if dtol: dtol = float(dtol)
    else: dtol = DIFFTOL
    floor = gmddiff.getAttribute("floor")
    if floor: floor = float(floor)
    else: floor = FLOOR

    variables = []
    for var in gmddiff.getElementsByTagName("Variable"):
        name = var.getAttribute("name")
        vftol = gmddiff.getAttribute("ftol")
        if vftol: vftol = float(vftol)
        else: vftol = ftol
        vdtol = gmddiff.getAttribute("dtol")
        if vdtol: vdtol = float(vdtol)
        else: vdtol = dtol
        vfloor = gmddiff.getAttribute("floor")
        if vfloor: vfloor = float(vfloor)
        else: vfloor = floor
        variables.append((name, vdtol, vftol, floor))

    return variables


def afloor(a, floor):
    a[np.where(np.abs(a) <= floor)] = 0.
    return a


if __name__ == "__main__":
    sys.exit(main())
