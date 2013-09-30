import os
import re
import sys
import shutil
import random
import argparse
from os.path import splitext

from __config__ import cfg, SPLASH, ROOT_D, LIB_D
import core.inpparse as inp
from core.physics import PhysicsHandler
from core.permutate import PermutationHandler
from core.optimize import OptimizationHandler
from core.io import Error1 as Error1, input_errors

FILE = os.path.realpath(__file__)


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", default=1, type=int,
       help="Verbosity [default: %(default)s]")
    parser.add_argument("-p", action="append",
       help="pprepro variables [default: %(default)s]")
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
    parser.add_argument("--clean", default=False, action="store_true",
        help=argparse.SUPPRESS)
    parser.add_argument("--restart", const=-1, default=False, nargs="?",
        help=argparse.SUPPRESS)
    parser.add_argument("sources", nargs="+", help="Source file paths")
    args = parser.parse_args(argv)
    cfg.debug = args.dbg
    cfg.sqa = args.sqa
    # directory to look for hrefs and other files
    cfg.I = args.I

    # add the working directory to the Python path
    sys.path.insert(0, os.getcwd())

    if args.B:
        import core.build as build
        try: os.remove(os.path.join(LIB_D, "{0}.so".format(args.B)))
        except OSError: pass
        b = build.main("-m {0}".format(args.B).split())
        if b != 0:
            raise SystemExit("failed to build")

    if args.p:
        tup = lambda a: (a[0].strip(), a[1].strip())
        args.p = dict(tup(x.split("=")) for x in args.p)

    if args.v:
        sys.stdout.write(SPLASH)
        sys.stdout.flush()

    # --- gather all input.
    # this is done in a separate loop from running the input
    # so that we can gather all inputs from all files (a single file can
    # specify multiple Physics inputs).
    all_input = []
    for (i, source) in enumerate(args.sources):

        # parse the user input
        basename = re.sub(".preprocessed$", "", os.path.basename(source))
        runid = splitext(basename)[0]

        if args.clean:
            clean_all_output(runid)
            continue

        elif args.restart:
            source = runid + ".exo"
            mm_input = inp.parse_exo_input(source, time=float(args.restart))

        else:
            if not os.path.isfile(source):
                source += ".xml"
                if not os.path.isfile(source):
                    logerr("{0}: no such file".format(source))
                    continue

            if splitext(basename)[1] != ".xml":
                logerr("*** mmd: expected .xml file extension")
                continue
            mm_input = inp.parse_input(source, argp=args.p)

        if input_errors():
            raise SystemExit("stopping due to input errors")

        all_input.extend([(runid, _) for _ in mm_input])

        continue

    # --- run all input
    output = []
    ninp = len(all_input)
    for (iinp, (runid, mm_input)) in enumerate(all_input):
        stype = mm_input[0]
        if stype in ("Optimization", "Permutation"):
            exe = "{0} {1}".format(sys.executable, FILE)
            if stype == "Permutation":
                model = PermutationHandler(runid, args.v, exe, args.j,
                                           *mm_input[1:])
            else:
                model = OptimizationHandler(runid, args.v, exe, *mm_input[1:])

        elif stype == "Physics":
            runid_ = mm_input[1]
            if not runid_:
                runid_ = runid
                if ninp > 1:
                    runid_ += "-" + str(iinp)
            model = PhysicsHandler(runid_, args.v, *mm_input[2:])

        else:
            logerr("{0}: unrecognized simulation type".format(mm_input.stype))
            continue

        # Run the problems
        try:
            model.run()
        except Error1, e:
            logerr("{0}: failed to run with message: {1}".format(
                model.runid, e.message))

        model.finish()
        output.append(model.output())
        del model

        if args.v and iinp + 1 != ninp:
            # separate screen output of different runs
            write_newline(n=2)

    if args.v:
        # a fun quote to end with
        ol = open(os.path.join(ROOT_D, "utils/zol")).readlines()
        sys.stdout.write("\n" + ol[random.randint(0, len(ol)-1)])

    if args.V and output:
        from viz.plot2d import create_model_plot
        create_model_plot(output)


def write_newline(n=1):
    sys.stdout.write("\n" * n)


def logerr(message=None, errors=[0]):
    if message is None:
        return errors[0]
    sys.stderr.write("*** mmd: error: {0}\n".format(message))
    errors[0] += 1


def stop(message):
    sys.stderr.write("*** mmd: error: {0}\n".format(message))
    raise SystemExit(2)


def clean_all_output(runid):
    if runid == "all":
        # collect all runids
        cwd = os.getcwd()
        files = os.listdir(cwd)
        runids = [splitext(f)[0] for f in files if f.endswith(".xml") or
                  all(splitext(f)[0] + e in files for e in (".exo", ".log"))]
        for runid in runids:
            if "all" in runid: continue
            clean_all_output(runid)

    for ext in (".exo", ".log", ".out", ".xml.preprocessed"):
        try: os.remove(runid + ext)
        except OSError: pass
    for ext in (".eval",):
        try: shutil.rmtree(runid + ext)
        except OSError: pass

if __name__ == "__main__":
    main()
