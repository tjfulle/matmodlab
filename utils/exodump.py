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
    parser.add_argument("--outfile")
    parser.add_argument("--variables", action="append")
    parser.add_argument("--ffmt")
    args = parser.parse_args(argv)
    exodump(args.source, outfile=args.outfile, variables=args.variables,
            ffmt=args.ffmt)


def exodump(filepath, outfile=None, variables="ALL", step=1, ffmt=None,
            ofmt="ascii"):
    """Read the exodus file in filepath and dump the contents to a columnar data
    file

    """
    if ofmt != "ascii":
        io.logmes("exodump: {0}: unrecognized format".format(ofmt))
        return

    if not os.path.isfile(filepath):
        raise Error1("{0}: no such file".format(filepath))

    if outfile is None:
        outfile = os.path.splitext(filepath)[0] + ".out"

    # Floating point format for numbers
    if ffmt is None: ffmt = ".18f"
    fmt = "{0: " + ffmt + "} "
    ffmt = lambda a, fmt=fmt: fmt.format(float(a))

    exof = ExodusIIReader.new_from_exofile(filepath)
    glob_var_names = exof.glob_var_names()
    elem_var_names = exof.elem_var_names()

    if variables != "ALL":
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

    with open(outfile, "w") as fobj:
        fobj.write("TIME {0} {1}\n".format(" ".join(glob_var_names).upper(),
                                             " ".join(elem_var_names).upper()))
        for i in myrange(0, exof.num_time_steps, step):
            time = exof.get_time(i)
            fobj.write(ffmt(time))
            glob_vars_vals = exof.get_glob_vars(i, disp=1)
            for var in glob_var_names:
                try: fobj.write(ffmt(glob_vars_vals[var]))
                except KeyError: continue
            for var in elem_var_names:
                val = exof.get_elem_var(i, var)[0]
                fobj.write(ffmt(val))
            fobj.write("\n")


def find_matches(master, slave):
    mstring = " ".join(master)
    matches = []
    v = []
    def endsort(item):
        endings = {"-XX": 0, "-YY": 1, "-ZZ": 2,
                   "-XY": 3, "-YZ": 4, "-XZ": 5,
                   "-X": 0, "-Y": 1, "-Z": 2}
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
        for match in re.findall(r"(?i)\b{0}-[XYZ]+".format(name), mstring):
            vt.append(match.strip())
            slave[i] = None
            continue
        vt = sorted(vt, key=lambda x: endsort(x))
        matches.extend(vt)
    return matches

if __name__ == "__main__":
    main()
