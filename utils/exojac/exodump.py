#!/usr/bin/python
import os
import re
import sys
import argparse
import numpy as np

from exoread import ExodusIIReader

OFMTS = {"ascii": ".out", "mathematica": ".math", "ndarray": ".npy"}

class ExoDumpError(Exception):
    def __init__(self, message):
        sys.stderr.write("*** exodump: error: {0}\n".format(message))
        self.message = message
        raise SystemExit(1)


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    parser = argparse.ArgumentParser()
    parser.add_argument("source")
    parser.add_argument("-o",
        help="Output file name [default: basename(source).out]")
    parser.add_argument("--variables", action="append",
        help="Variables to dump [default: ALL]")
    parser.add_argument("--list", default=False, action="store_true",
        help="List variable names and exit")
    parser.add_argument("--ffmt",
        help="Output floating point format [default: .18f]")
    parser.add_argument("--ofmt", default="ascii", choices=OFMTS.keys(),
        help="Output format [default: %(default)s]")
    parser.add_argument("--step", default=1, type=int,
        help="Step [default: %(default)s]")
    parser.add_argument("--block", default=1, type=int,
        help="Block number [default: %(default)s]")
    parser.add_argument("--element", default=1, type=int,
        help="Element number [default: %(default)s]")
    args = parser.parse_args(argv)
    return exodump(args.source, outfile=args.o,
                   variables=args.variables, listvars=args.list,
                   ffmt=args.ffmt, ofmt=args.ofmt, step=args.step,
                   elem_blk=args.block, elem_num=args.element)


def exodump(filepath, outfile=None, variables=None, listvars=False, step=1,
            ffmt=None, ofmt="ascii", elem_blk=1, elem_num=1):
    """Read the exodus file in filepath and dump the contents to a columnar data
    file

    """
    ofmt = ofmt.lower()
    if ofmt not in OFMTS:
        raise ExoDumpError("{0}: unrecognized output format\n".format(ofmt))

    if not os.path.isfile(filepath):
        raise ExoDumpError("{0}: no such file".format(filepath))

    # setup output stream
    if outfile is None:
        ext = OFMTS[ofmt]
        stream = open(os.path.splitext(filepath)[0] + ext, "w")
    elif outfile in ("1", "stdout"):
        stream = sys.stdout
    elif outfile in ("2", "stderr"):
        stream = sys.stderr
    elif outfile is "return":
        stream = None
    else:
        stream = open(outfile, "w")

    # setup variables
    if variables is None:
        variables = ["ALL"]
    else:
        if not isinstance(variables, (list, tuple)):
            variables = [variables]
        variables = [v.strip() for v in variables]
        if "all" in [v.lower() for v in variables]:
            variables = ["ALL"]

    # read the data
    header, data = read_vars_from_exofile(filepath, variables=variables,
                                          step=step, elem_blk=elem_blk,
                                          elem_num=elem_num)
    if listvars:
        print("\n".join(header))
        return 0

    # Floating point format for numbers
    if ffmt is None: ffmt = ".18f"
    fmt = "{0: " + ffmt + "}"
    ffmt = lambda a, fmt=fmt: fmt.format(float(a))

    if ofmt == "ascii":
        asciidump(stream, ffmt, header, data)

    elif ofmt == "mathematica":
        mathdump(stream, ffmt, header, data)

    elif ofmt == "ndarray":
        nddump(stream, ffmt, header, data)

    stream.close()

    return 0


def read_vars_from_exofile(filepath, variables=None, step=1, h=1,
                           elem_blk=1, elem_num=1):
    """Read the specified variables from the exodus file in filepath

    """
    # setup variables
    if variables is None:
        variables = ["ALL"]
    else:
        if not isinstance(variables, (list, tuple)):
            variables = [variables]
        variables = [v.strip() for v in variables]
        if "all" in [v.lower() for v in variables]:
            variables = ["ALL"]

    if not os.path.isfile(filepath):
        raise ExoDumpError("{0}: no such file".format(filepath))

    exof = ExodusIIReader(filepath)
    return_time = True
    glob_var_names = exof.glob_var_names
    elem_var_names = exof.elem_var_names

    if variables[0] != "ALL":
        return_time = "time" in [x.lower() for x in variables]
        glob_var_names = expand_var_names(glob_var_names, variables)
        elem_var_names = expand_var_names(elem_var_names, variables)
        bad = [x for x in variables if x is not None]
        if bad:
            raise ExoDumpError("{0}: variables not in "
                               "{1}".format(", ".join(bad), filepath))

    # retrieve the data from the database
    header = ["TIME"]
    header.extend([H.upper() for H in glob_var_names])
    header.extend([H.upper() for H in elem_var_names])
    data = []
    for i in myrange(0, exof.num_time_steps, step):
        row = [exof.get_time(i)]
        glob_vars_vals = exof.get_glob_vars(i, disp=1)
        for var in glob_var_names:
            try: row.append(glob_vars_vals[var])
            except KeyError: continue
        for var in elem_var_names:
            row.append(exof.get_elem_var_time(var, elem_num, elem_blk)[i])
        data.append(row)
    exof.close()
    data = np.array(data)

    if len(header) != data.shape[1]:
        raise ExoDumpError("inconsistent data")

    if not return_time:
        data = data[:, 1:]
        header = header[1:]

    if h:
        return header, data

    return data


def asciidump(stream, ffmt, header, data):
    stream.write("{0}\n".format(" ".join(header)))
    stream.write("\n".join(" ".join(ffmt(d) for d in row) for row in data))
    return


def nddump(stream, ffmt, header, data):
    np.save(stream, data)
    return


def mathdump(stream, ffmt, header, data):
    for (i, name) in enumerate(header):
        col = data[:, i]
        stream.write("{0}={{{1}}}\n".format(
            name, ",".join(ffmt(d) for d in data[:, i])))
    return


def myrange(start, end, step):
    r = [i for i in range(start, end, step)]
    if end - 1 not in r:
        r.append(end - 1)
    return r


def expand_var_names(master, slave):
    mstring = " ".join(master)
    matches = []
    v = []
    def endsort(item):
        endings = {"_XX": 0, "_YY": 1, "_ZZ": 2,
                   "_XY": 3, "_YZ": 4, "_XZ": 5,
                   "_YX": 6, "_ZY": 7, "_ZX": 8,
                   "_X": 0, "_Y": 1, "_Z": 2}
        for (ending, order) in endings.items():
            if item.endswith(ending):
                return order
        return 9

    for i, name in enumerate(slave):
        if name in master:
            matches.append(name)
            slave[i] = None
            continue
        vt = []
        for match in re.findall(r"(?i)\b{0}_[XYZ]+".format(name), mstring):
            vt.append(match.strip())
            slave[i] = None
            continue
        vt = sorted(vt, key=lambda x: endsort(x))
        matches.extend(vt)
    return matches


if __name__ == '__main__':
    sys.exit(main())
