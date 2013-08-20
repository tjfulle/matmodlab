import os
import sys
import argparse

from __config__ import cfg
import core.gmd as gmd
import core.permutate as perm
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
    parser.add_argument("--sqa", default=False, action="store_true",
       help="SQA mode [default: %(default)s]")
    parser.add_argument("-j", default=1, type=int,
       help="Number of simultaneous jobs to run [default: %(default)s]")
    args = parser.parse_args(argv)
    cfg.verbosity = args.v
    cfg.debug = args.dbg
    cfg.sqa = args.sqa

    # parse the user input
    source = args.source
    if not os.path.isfile(source):
        source += ".xml"
        if not os.path.isfile(source):
            sys.exit("gmd: {0}: no such file".format(args.source))

    lines = open(source, "r").read()

    basename = os.path.basename(args.source).rstrip(".preprocessed")
    runid = os.path.splitext(basename)[0]
    mm_input = inpparse.parse_input(lines)

    if mm_input.stype == "simulation":
        opts = (mm_input.kappa, mm_input.density, mm_input.proportional,
                mm_input.ndumps)
        model = gmd.ModelDriver(runid, args.v, mm_input.driver, mm_input.mtlmdl,
                                mm_input.mtlprops, mm_input.legs,
                                mm_input.ttermination, mm_input.extract, opts)

    elif mm_input.stype == "permutation":
        f = os.path.realpath(__file__)
        opts = (args.j,)
        exe = "{0} {1}".format(sys.executable, f)
        model = perm.PermutationDriver(runid, exe, mm_input.method,
                                       mm_input.parameters, mm_input.basexml,
                                       *opts)

    else:
        sys.exit("{0}: unrecognized simulation type".format(mm_input.stype))

    # Setup and run the problem
    model.setup()
    model.run()
    model.finish()


if __name__ == "__main__":
    main()
