import os
import re
import sys
import numpy as np
import argparse

import utils.io as io
from utils.errors import Error1
from exoreader import ExodusIIReader


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    parser = argparse.ArgumentParser()
    parser.add_argument("source")
    parser.add_argument("-o",
        help="Output file name [default: basename(source).out]")
    parser.add_argument("--variables", action="append",
        help="Variables to dump [default: ALL]")
    parser.add_argument("--ffmt",
        help="Output floating point format [default: .18f]")
    args = parser.parse_args(argv)
    exodump(args.source, outfile=args.o, variables=args.variables,
            ffmt=args.ffmt)


def exodump(filepath, outfile=None, variables=None, step=1, ffmt=None,
            ofmt="ascii"):
    """Read the exodus file in filepath and dump the contents to a columnar data
    file

    """
    if ofmt != "ascii":
        io.logmes("exodump: {0}: unrecognized format".format(ofmt))
        return

    if not os.path.isfile(filepath):
        raise Error1("{0}: no such file".format(filepath))

    # setup output stream
    if outfile is None:
        stream = open(os.path.splitext(filepath)[0] + ".out", "w")
    elif outfile in ("1", "stdout"):
        stream = sys.stdout
    elif outfile in ("2", "stderr"):
        stream = sys.stderr
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

    # Floating point format for numbers
    if ffmt is None: ffmt = ".18f"
    fmt = "{0: " + ffmt + "} "
    ffmt = lambda a, fmt=fmt: fmt.format(float(a))

    exof = ExodusIIReader.new_from_exofile(filepath)
    glob_var_names = exof.glob_var_names()
    elem_var_names = exof.elem_var_names()

    if variables[0] != "ALL":
        glob_var_names = find_matches(glob_var_names, variables)
        elem_var_names = find_matches(elem_var_names, variables)
        bad = [x for x in variables if x is not None]
        if bad:
            raise Error1("{0}: variables not in "
                         "{1}".format(", ".join(bad), filepath))

    def myrange(start, end, step):
        r = [i for i in range(start, end, step)]
        if end - 1 not in r:
            r.append(end - 1)
        return r

    stream.write("TIME {0} {1}\n".format(" ".join(glob_var_names).upper(),
                                         " ".join(elem_var_names).upper()))
    for i in myrange(0, exof.num_time_steps, step):
        time = exof.get_time(i)
        stream.write(ffmt(time))
        glob_vars_vals = exof.get_glob_vars(i, disp=1)
        for var in glob_var_names:
            try: stream.write(ffmt(glob_vars_vals[var]))
            except KeyError: continue
        for var in elem_var_names:
            val = exof.get_elem_var(i, var)[0]
            stream.write(ffmt(val))
        stream.write("\n")


def find_matches(master, slave):
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

if __name__ == "__main__":
    main()
