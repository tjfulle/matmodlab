import os
import sys
import argparse
import warnings
import numpy as np
from numpy.compat import asbytes
import xml.dom.minidom as xdom
from os.path import isfile, splitext, basename, join

from .numerix import *
from ..constants import *

def savefile(filename, names, data):
    """Save the file using tabfileio"""
    try:
        import tabfileio
    except ImportError:
        raise ValueError('external tabfileio package required to load file type')
    tabfileio.write_file(filename, names, data)

def loadfile(filename, disp=1, skiprows=0, sheetname="MML", columns=None,
             comments='#', variables=None, at_step=0, upcase=1):
    """Load the file"""
    if columns is not None and variables is not None:
        raise ValueError('columns and variables keywords are exclusive')
    columns = columns if columns is not None else variables

    if not is_string_like(filename):
        # assume filename is a stream
        names, data = loadstream(filename, upcase=upcase, disp=1,
                                 comments=comments, skiprows=skiprows)

    elif filename.endswith(tuple('.%s'%ext for ext in (CSV, TXT))):
        # standard Matmodlab formats
        names, data = loadtxt(filename, upcase=upcase, disp=1,
                              comments=comments, skiprows=skiprows)

    elif filename.endswith(('.exo', '.base_exo', '.dbx', '.base_dbx')):
        # legacy finite element database formats
        try:
            from femlib.fileio import loaddb_single_element
        except ImportError:
            raise ValueError('external femlib package required to load file type')
        names, data = loaddb_single_element(filename, disp=1, blk_num=1,
                                            elem_num=1, at_step=at_step,
                                            upcase=upcase)

    elif filename.endswith(('.rpk', '.base_rpk')):
        # Matmodlab record array pickle
        names, data = loadrec(filename, upcase=upcase, disp=1, at_step=at_step)

    else:
        # ??? -> let tabfileio deal with this extension
        try:
            import tabfileio
        except ImportError:
            raise ValueError('external tabfileio package '
                             'required to load file type')

        names, data = tabfileio.read_file(filename, disp=1, sheetname=sheetname)

    if columns:
        columns = tolist(columns)
        for (i, item) in enumerate(columns):
            if is_string_like(item):
                # determine the integer column index
                if names is None:
                    raise ValueError('cannot determine column numbers '
                                     'of named variables')
                j = index(names, item)
                if j is None:
                    raise ValueError('%r not in file' % item)
                columns[i] = j
        data = data[:, columns]
        if names:
            names = [names[i] for i in columns]

    if disp:
        return names, data
    return data

def loadrec(filename, upcase=0, disp=1, at_step=0):
    """Load a numpy record array stored as a pickle"""

    # load the data
    data = np.load(filename)

    # get the names -> specific to how Matmodlab stores the data
    names = []
    for item in data.dtype.descr:
        # Get the names of each field, expanded to include component
        try:
            key, dtype, shape = item
            keys = ['%s.%s' % (key, ext) for ext in COMPONENT_LABELS(shape[0])]
        except ValueError:
            key, dtype = item
            if key.startswith('SDV_'):
                key = key.replace('SDV_', 'SDV.')
            keys = [key]
        names.extend(keys)

    data = rec2arr(data)

    if at_step:
        # Get only the data at the end of the step
        d = {}
        for (i, x) in enumerate(data['Step']):
            d.setdefault(int(x), []).append(i)
        rows = [x[-1] for x in sorted(d.values())]
        data = data[rows]

    if disp:
        if upcase:
            names = [x.upper() for x in names]
        return names, data

    return data

def loadstream(stream, comments='#', skiprows=0, upcase=False, disp=1):
    """Load data contained in the stream"""

    for i in range(skiprows):
        next(stream)

    names = None
    if disp:
        # find the header
        names = find_header(stream, comments=comments, upcase=upcase)
        if names is None:
            warnings.warn('loadstream: could not find header')

    data = np.loadtxt(stream)
    if disp:
        return names, data
    return data

def loadtxt(filename, comments='#', skiprows=0, upcase=False,
            delimiter=' ', disp=1):

    if filename.endswith('.csv'):
        delimiter = ','

    # open the stream and load the data
    stream = open(filename, 'r')
    names, data = loadstream(stream, comments=comments, delimiter=delimiter,
                             disp=1, skiprows=skiprows, upcase=upcase)
    stream.close()

    if disp:
        return names, data
    return data

def find_header(stream, comments='#', delimiter=' ', upcase=False):
    """Find the header in the first line of the stream"""

    comments = asbytes(comments)
    terminator = asbytes('\r\n')
    pos = stream.tell()

    def _split(line):
        """Look for a header in the line"""
        try:
            pre, post = [s.strip(terminator)
                         for s in asbytes(line).split(comments, 1)]
        except ValueError:
            return None
        if pre.split():
            # Line of form: a b c # e f g
            # -> not a header
            return None
        post = [x.strip() for x in post.split(delimiter)] or None
        return post

    names = None
    try:
        while not names:
            first_line = next(stream)
            names = _split(first_line)
    except StopIteration:
        names = None

    if names is None:
        # no header found, rewind
        stream.seek(pos)

    return names

def flatten(a):
    flat = []
    for x in a:
        try: flat.extend(x)
        except TypeError: flat.append(x)
    return flat

def rec2arr(recarr):
    arr = []
    for row in recarr:
        arr.append(flatten(row.tolist()))
    return np.array(arr)

def filediff_entry(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    parser = argparse.ArgumentParser()
    parser.add_argument('-f',
        help=('Use the given file to specify the variables to be considered '
              'and to what tolerances [default: %(default)s].'))
    parser.add_argument('--interp', default=False, action='store_true',
        help=('Interpolate variabes through time to compute error '
              '[default: %(default)s].'))
    parser.add_argument('--plot', default=False, action='store_true',
        help=('Plot file variables that diff [default: %(default)s].'))
    parser.add_argument('source1')
    parser.add_argument('source2')
    args = parser.parse_args(argv)
    if args.plot:
        return plot_files(args.source1, args.source2)

    return filediff(args.source1, args.source2, control_file=args.f,
                    interp=args.interp)

def filediff(source1, source2, control_file=None, interp=False, stream=sys.stdout,
             adjust_n=0):

    errors = 0
    if not isfile(source1):
        errors += 1
        stream.write('***error: {0}: no such file\n'.format(source1))
    if not isfile(source2):
        errors += 1
        stream.write('***error: {0}: no such file\n'.format(source2))
    if errors:
        return ERRORS

    stream.write('reading {0}... '.format(source1))
    H1, D1 = loadfile(source1)
    stream.write('done\nreading {0}... \n'.format(source2))
    H2, D2 = loadfile(source2)
    stream.write('done\n'.format(source1))

    if adjust_n:
        m = min(D1.shape[0], D2.shape[0])
        D1 = D1[:m]
        D2 = D2[:m]

    if control_file is not None:
        if not isfile(control_file):
            stream.write('***error: {0}: no such file'.format(control_file))
            return ERRORS
        variables = read_diff_file(control_file, stream)
        if variables == ERRORS:
            return ERRORS
    else:
        variables = zip(H1, [DIFFTOL] * len(H1), [FAILTOL] * len(H1),
                        [FLOOR] * len(H1))

    status = diff_data_sets(H1, D1, H2, D2, variables, stream, interp=interp)

    if status == 0:
        stream.write('\nFiles are the same\n')

    elif status == 1:
        stream.write('\nFiles diffed\n')

    else:
        stream.write('\nFiles are different\n')

    return status

def read_diff_file(filepath, stream):
    '''Read the diff instruction file

    Parameters
    ----------
    filepath : str
        Path to diff instruction file

    Notes
    -----
    The diff instruction file has the following format

    <ExDiff [ftol="real"] [dtol="real"] [floor="real"]>
      <Variable name="string" [ftol="real"] [dtol="real"] [floor="real"]/>
    </ExDiff>

    It lets you specify:
      global failure tolerance (ExDiff ftol attribute)
      global diff tolerance (ExDiff dtol attribute)
      global floor (ExDiff floor attribute)

      individual variables to specify (Variable tags)
      individual failure tolerance (Variable ftol attribute)
      individual diff tolerance (Variable dtol attribute)
      individual floor (Variable floor attribute)

    '''
    doc = xdom.parse(filepath)
    try:
        exdiff = doc.getElementsByTagName('ExDiff')[0]
    except IndexError:
        stream.write('***error: {0}: expected root '
                     'element "ExDiff"'.format(filepath))
        return ERRORS
    ftol = exdiff.getAttribute('ftol')
    if ftol: ftol = float(ftol)
    else: ftol = FAILTOL
    dtol = exdiff.getAttribute('dtol')
    if dtol: dtol = float(dtol)
    else: dtol = DIFFTOL
    floor = exdiff.getAttribute('floor')
    if floor: floor = float(floor)
    else: floor = FLOOR

    variables = []
    for var in exdiff.getElementsByTagName('Variable'):
        name = var.getAttribute('name')
        vftol = var.getAttribute('ftol')
        if vftol: vftol = float(vftol)
        else: vftol = ftol
        vdtol = var.getAttribute('dtol')
        if vdtol: vdtol = float(vdtol)
        else: vdtol = dtol
        vfloor = var.getAttribute('floor')
        if vfloor: vfloor = float(vfloor)
        else: vfloor = floor
        variables.append((name, vdtol, vftol, floor))

    return variables

def plot_files(source1, source2):
    import time
    import matplotlib.pyplot as plt
    cwd = os.getcwd()
    head1, data1 = loadfile(source1)
    head2, data2 = loadfile(source2)
    label1 = splitext(basename(source1))[0]
    label2 = splitext(basename(source2))[0]

    # Compare times first
    try:
        time1 = data1[:, head1.index("TIME")]
    except:
        sys.stdout.write("***error: TIME not in File1\n")
        return NOT_SAME
    try:
        time2 = data2[:, head2.index("TIME")]
    except:
        sys.stdout.write("***error: TIME not in File2\n")
        return NOT_SAME

    head2 = dict([(v, i) for (i, v) in enumerate(head2)])

    ti = time.time()

    aspect_ratio = 4. / 3.
    plots = []
    for (col, yvar) in enumerate(head1):
        if yvar == "TIME":
            continue
        col2 = head2.get(yvar)
        if col2 is None:
            continue

        name = "{0}_vs_TIME.png".format(yvar)
        filename = join(cwd, name)

        y1 = data1[:, col]
        y2 = data2[:, col2]

        plt.clf()
        plt.cla()
        plt.xlabel("TIME")
        plt.ylabel(yvar)
        plt.plot(time2, y2, ls="-", lw=4, c="orange", label=label2)
        plt.plot(time1, y1, ls="-", lw=2, c="green", label=label1)
        plt.legend(loc="best")
        plt.gcf().set_size_inches(aspect_ratio * 5, 5.)
        plt.savefig(filename, dpi=100)
        plots.append(filename)

    return

def filedump_entry(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    parser = argparse.ArgumentParser()
    parser.add_argument('-o',
        help='Output file name [default: root(source).out]')
    parser.add_argument('--list', default=False, action='store_true',
        help='List variable names and exit')
    parser.add_argument('--ofmt',
        help='Output format, one of math, numpy')
    parser.add_argument('--ffmt', default='%.18f',
        help='Output floating point format [default: %.18f]')
    parser.add_argument('source')
    parser.add_argument('variables', nargs='*', default=None,
        help='Variables to dump [default: ALL]')
    args = parser.parse_args(argv)

    ofmts = {'math': '.math', 'numpy': '.npy'}

    # setup output stream
    if args.ofmt is not None:
        args.o = splitext(args.source)[0] + ofmts[args.ofmt]

    if args.o is None:
        args.o = splitext(args.source)[0] + '.out'
    elif args.o in ('1', 'stdout'):
        args.o = sys.stdout
    elif args.o in ('2', 'stderr'):
        args.o = sys.stderr

    return filedump(args.source, args.o, variables=args.variables,
                    listvars=args.list, ffmt=args.ffmt)

def filedump(infile, outfile, variables=None, listvars=False, ffmt='%.18f'):
    '''Read the exodus file in filepath and dump the contents to a columnar data
    file

    '''
    ofmts = {'.math': 'mathematica', '.npy': 'ndarray'}
    if not isinstance(outfile, basestring):
        fown = 0
        ofmt = 'ascii'
        stream = outfile
    else:
        fown = 1
        if not isfile(infile):
            raise IOError('{0}: no such file'.format(filepath))
        root, ext = splitext(outfile)
        ofmt = ofmts.get(ext, 'ascii')
        stream = open(outfile, 'w')

    # read the data
    head, data = loadfile(infile, variables=variables)

    if listvars:
        print('\n'.join(head))
        return

    if ofmt == 'ascii':
        ffmt = ' '.join(ffmt for i in range(data.shape[1]))
        stream.write(' '.join(head) + '\n')
        stream.write('\n'.join(ffmt % tuple(row) for row in data))

    elif ofmt == 'mathematica':
        ffmt = ','.join(ffmt for i in range(data.shape[0]))
        stream.write('\n'.join('{0}={{{1}}}'.format(name, ffmt % tuple(data[:, i]))
                               for (i, name) in enumerate(head)))

    elif ofmt == 'ndarray':
        np.save(stream, data)

    if not fown:
        return stream

    if fown:
        stream.close()

def is_string_like(obj):
    # from numpy/lib/_iotools.py
    try:
        obj + ''
    except (TypeError, ValueError):
        return False
    return True

def is_number_like(obj):
    # from numpy/lib/_iotools.py
    try:
        int(obj)
    except (ValueError, TypeError):
        return False
    return True

def tolist(obj):
    if is_number_like(obj):
        return [obj]
    elif is_string_like(obj):
        return [obj]
    return list(obj)

def index(lst, x):
    if not lst:
        return None
    try:
        return [s.upper() for s in lst].index(x.upper())
    except ValueError:
        return None
