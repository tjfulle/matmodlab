import os
import sys
import argparse

from __config__ import cfg
import core.gmd as gmd
import utils.inpparse as inpparse
from utils.errors import Error1


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    parser = argparse.ArgumentParser()
    parser.add_argument("source", help="Source file path")
    parser.add_argument("-v", default=1, type=int,
       help="Verbosity [default: %(default)s]")
    parser.add_argument("--dbg", default=False, action="store_true",
       help="Debug mode [default: %(default)s]")
    args = parser.parse_args(argv)
    cfg.verbosity = args.v
    cfg.debug = args.dbg

    # parse the user input
    try:
        lines = open(args.source, "r").read()
    except IOError:
        raise Error1("{0}: no such file".format(args.source))
    runid = os.path.splitext(os.path.basename(args.source))[0]
    mm_input = inpparse.parse_input(lines)

    if mm_input.stype == "simulation":
        opts = (mm_input.kappa, mm_input.density, mm_input.proportional)
        model = gmd.ModelDriver(runid, mm_input.driver, mm_input.mtlmdl,
                                mm_input.mtlprops, mm_input.legs,
                                mm_input.extract, *opts)

    else:
        sys.exit("{0}: simulation type not known".format(mm_input.stype))

    # Setup and run the problem
    model.setup()
    model.run()
    model.finish()


if __name__ == "__main__":
    main()
