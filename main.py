import os
import sys
import argparse

from __config__ import cfg, SPLASH
from core.physics import PhysicsHandler
from core.permutate import PermutationHandler
from core.optimize import OptimizationHandler
from core.inpparse import parse_input, S_PHYSICS, S_OPT, S_PERMUTATION
from core.io import Error1 as Error1

FILE = os.path.realpath(__file__)


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", default=1, type=int,
       help="Verbosity [default: %(default)s]")
    parser.add_argument("--dbg", default=False, action="store_true",
       help="Debug mode [default: %(default)s]")
    parser.add_argument("--sqa", default=False, action="store_true",
       help="SQA mode [default: %(default)s]")
    parser.add_argument("-j", default=1, type=int,
       help=("Number of simultaneous jobs to run (permutation only) "
             "[default: %(default)s]"))
    parser.add_argument("-V", default=False, action="store_true",
       help="Launch simulation visualizer on completion [default: %(default)s]")
    parser.add_argument("-I", default=os.getcwd(), help=argparse.SUPPRESS)
    parser.add_argument("sources", nargs="*", help="Source file paths")
    args = parser.parse_args(argv)
    cfg.debug = args.dbg
    cfg.sqa = args.sqa
    # directory to look for hrefs and other files
    cfg.I = args.I

    if not args.sources:
        sys.exit("GUI not yet functional")
        import viz.select as vs
        window = vs.MaterialModelSelector(model_type="any")
        sys.exit(window.configure_traits())

    output = []
    for (i, source) in enumerate(args.sources):
        # parse the user input
        if not os.path.isfile(source):
            source += ".xml"
            if not os.path.isfile(source):
                logerr("{0}: no such file".format(args.source))
                continue

        basename = os.path.basename(source).rstrip(".preprocessed")
        if not basename.endswith(".xml"):
            logerr("*** gmd: expected .xml file extension")
            continue
        runid = os.path.splitext(basename)[0]
        mm_input = parse_input(source)

        if args.v:
            sys.stdout.write(SPLASH)

        if mm_input.stype == S_PHYSICS:
            opts = (mm_input.density,)
            model = PhysicsHandler(runid, args.v, mm_input.driver, mm_input.mtlmdl,
                                   mm_input.mtlprops, mm_input.ttermination,
                                   mm_input.extract, opts)

        elif mm_input.stype == S_PERMUTATION:
            opts = (args.j,)
            exe = "{0} {1}".format(sys.executable, FILE)
            model = PermutationHandler(runid, args.v, mm_input.method,
                                       mm_input.response_function,
                                       mm_input.response_descriptor,
                                       mm_input.parameters, exe,
                                       mm_input.basexml, *opts)

        elif mm_input.stype == S_OPT:
            exe = "{0} {1}".format(sys.executable, FILE)
            model = OptimizationHandler(runid, args.v, mm_input.method,
                                        exe, mm_input.response_function,
                                        mm_input.response_descriptor,
                                        mm_input.parameters,
                                        mm_input.tolerance, mm_input.maxiter,
                                        mm_input.disp, mm_input.basexml,
                                        mm_input.auxiliary_files)

        else:
            logerr("{0}: unrecognized simulation type".format(mm_input.stype))
            continue

        # Setup and run the problem
        model.setup()
        try:
            model.run()
        except Error1, e:
            logerr("{0}: failed to run with message: {1}".format(runid, e.message))

        model.finish()
        output.append(model.output())

        del model

    if args.V and output:
        from viz.plot2d import create_model_plot
        create_model_plot(output)


def logerr(message=None, errors=[0]):
    if message is None:
        return errors[0]
    sys.stderr.write("*** gmd: error: {0}\n".format(message))
    errors[0] += 1


def stop(message):
    sys.stderr.write("*** gmd: error: {0}\n".format(message))
    raise SystemExit(2)


if __name__ == "__main__":
    main()
