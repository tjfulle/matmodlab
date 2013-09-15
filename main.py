import os
import re
import sys
import random
import argparse

from __config__ import cfg, SPLASH, ROOT_D, LIB_D
import core.inpparse as inp
from core.physics import PhysicsHandler
from core.permutate import PermutationHandler
from core.optimize import OptimizationHandler
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
    parser.add_argument("-B", metavar="material",
        help="Build material model before running [default: %(default)s]")
    parser.add_argument("--restart", const=-1, default=False, nargs="?",
        help=argparse.SUPPRESS)
    parser.add_argument("sources", nargs="*", help="Source file paths")
    args = parser.parse_args(argv)
    cfg.debug = args.dbg
    cfg.sqa = args.sqa
    # directory to look for hrefs and other files
    cfg.I = args.I

    if args.B:
        import core.build as build
        try: os.remove(os.path.join(LIB_D, "{0}.so".format(args.B)))
        except OSError: pass
        b = build.main("-m {0}".format(args.B).split())
        if b != 0:
            raise SystemExit("failed to build")

    if not args.sources:
        raise SystemExit("GUI not yet functional")
        import viz.select as vs
        window = vs.MaterialModelSelector(model_type="any")
        sys.exit(window.configure_traits())

    output = []
    for (i, source) in enumerate(args.sources):

        # parse the user input
        basename = re.sub(".preprocessed$", "", os.path.basename(source))
        runid = os.path.splitext(basename)[0]

        if args.restart:
            source = runid + ".exo"
            mm_input = parse_exo_input(source, time=float(args.restart))
            restart_info = [mm_input.kappa, mm_input.leg_num, mm_input.time,
                            mm_input.glob_data, mm_input.elem_data]

        else:
            if not os.path.isfile(source):
                source += ".xml"
                if not os.path.isfile(source):
                    logerr("{0}: no such file".format(args.source))
                    continue

            if os.path.splitext(basename)[1] != ".xml":
                logerr("*** gmd: expected .xml file extension")
                continue
            mm_input = inp.parse_input(source)
            restart_info = None

        if args.v:
            sys.stdout.write(SPLASH)
            sys.stdout.flush()

        if inp.INP_ERRORS:
            raise SystemExit("stopping due to input errors")

        stype = mm_input[0]
        if stype == "Physics":
            model = PhysicsHandler(runid, args.v, *mm_input[1:])

        elif stype == "Permutation":
            exe = "{0} {1}".format(sys.executable, FILE)
            model = PermutationHandler(runid, args.v, exe, args.j, *mm_input[1:])

        elif stype == "Optimization":
            exe = "{0} {1}".format(sys.executable, FILE)
            model = OptimizationHandler(runid, args.v, exe, *mm_input[1:])

        else:
            logerr("{0}: unrecognized simulation type".format(mm_input.stype))
            continue

        # Run the problem
        try:
            model.run()
        except Error1, e:
            logerr("{0}: failed to run with message: {1}".format(runid, e.message))

        model.finish()
        output.append(model.output())

        if args.v:
            # a fun quote to end with
            ol = open(os.path.join(ROOT_D, "utils/zol")).readlines()
            sys.stdout.write("\n" + ol[random.randint(0, len(ol)-1)])

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
