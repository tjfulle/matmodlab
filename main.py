import os
import sys
import argparse

import __config__ as cfg
import core.gmd as gmd
import core.parser as parser


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    aparser = argparse.ArgumentParser()
    aparser.add_argument("source", help="Source file path")
    aparser.add_argument("-v", default=1, type=int,
       help="Verbosity [default: %(default)s]")
    args = aparser.parse_args(argv)
    cfg.verbosity = args.v

    # parse the user input
    try:
        lines = open(args.source, "r").read()
    except OSError:
        raise errors.Error1("{0}: no such file".format(args.source))
    runid = os.path.splitext(os.path.basename(args.source))[0]
    mm_input = parser.parse_input(lines)

    if mm_input.stype == "simulation":
        model = gmd.ModelDriver(runid, mm_input.driver, mm_input.mtlmdl,
                                mm_input.mtlprops, mm_input.legs, mm_input.kappa,
                                mm_input.density)

    else:
        sys.exit("{0}: simulation type not known".format(mm_input.stype))

    # Setup and run the problem
    model.setup()
    model.run()
    model.finish()


if __name__ == "__main__":
    main()
