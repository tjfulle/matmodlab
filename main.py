import os
import sys
import argparse

from __config__ import cfg
import core.gmd as gmd
import core.permutate as perm
import core.optimize as opt
import base.inpparse as inpparse
from base.inpparse import S_PHYSICS, S_OPT, S_PERMUTATION
from base.io import Error1

FILE = os.path.realpath(__file__)


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
    parser.add_argument("-V", default=False, action="store_true",
       help="Launch simulation visualizer on completion [default: %(default)s]")
    args = parser.parse_args(argv)
    cfg.debug = args.dbg
    cfg.sqa = args.sqa

    # parse the user input
    source = args.source
    if not os.path.isfile(source):
        source += ".xml"
        if not os.path.isfile(source):
            raise OSError("{0}: no such file".format(args.source))

    basename = os.path.basename(source).rstrip(".preprocessed")
    runid = os.path.splitext(basename)[0]
    mm_input = inpparse.parse_input(source)

    if mm_input.stype == S_PHYSICS:
        opts = (mm_input.kappa, mm_input.density, mm_input.proportional,
                mm_input.ndumps)
        # set up the logger
        model = gmd.PhysicsDriver(runid, args.v, mm_input.driver, mm_input.mtlmdl,
                                  mm_input.mtlprops, mm_input.legs,
                                  mm_input.ttermination, mm_input.extract, opts)

    elif mm_input.stype == S_PERMUTATION:
        opts = (args.j,)
        exe = "{0} {1}".format(sys.executable, FILE)
        model = perm.PermutationDriver(runid, args.v, mm_input.method,
                                       mm_input.parameters, exe,
                                       mm_input.basexml, *opts)

    elif mm_input.stype == S_OPT:
        exe = "{0} {1}".format(sys.executable, FILE)
        model = opt.OptimizationDriver(runid, args.v, mm_input.method,
                                       exe, mm_input.objective_function,
                                       mm_input.parameters,
                                       mm_input.tolerance, mm_input.maxiter,
                                       mm_input.disp, mm_input.basexml,
                                       mm_input.auxiliary_files)

    else:
        sys.exit("{0}: unrecognized simulation type".format(mm_input.stype))

    # Setup and run the problem
    model.setup()
    model.run()
    model.finish()

    if args.V:
        from viz.plot2d import create_model_plot
        create_model_plot(model.output())


if __name__ == "__main__":
    main()
