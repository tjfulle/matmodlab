#!/opt/epd/bin/python2.7
import os
import sys
import imp
import argparse

D = os.path.dirname(os.path.realpath(__file__))
R = os.path.realpath(os.path.join(D, "../"))
MTLDIRS = [os.path.join(R, "materials")]
MTLDIRS.extend(os.getenv("PY_EXE_ENV", "").split(os.pathsep))
FC = os.getenv("FC", "gfortran")
LIBD = os.path.join(R, "lib")
VERSION = "0.0.0"

from materials.material import write_mtldb


def logmes(message, end="\n"):
    sys.stdout.write("build-mtl: {0}{1}".format(message, end))
    sys.stdout.flush()


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", action="append",
        help="Material to build [default: all]")
    args = parser.parse_args(argv)

    logmes("gmd {0}".format(VERSION))
    logmes("looking for makemf files")

    kwargs = {"FC": FC, "DESTD": LIBD, "MATERIALS": args.m}
    mtldict = {}
    allfailed = []
    allbuilt = []
    for dirpath in MTLDIRS:
        for (d, dirs, files) in os.walk(dirpath):
            if "makemf.py" in files:
                f = os.path.join(d, "makemf.py")
                logmes("building makemf in {0}".format(d), end="... ")
                makemf = imp.load_source("makemf", os.path.join(d, "makemf.py"))
                made = makemf.makemf(**kwargs)
                failed = made.get("FAILED")
                built = made.get("BUILT")
                skipped = made.get("SKIPPED")

                if failed:
                    logmes("no")
                    allfailed.extend(failed)

                if skipped:
                    if not failed and not built:
                        logmes("skipped")

                if built:
                    if not failed:
                        logmes("yes")
                    if built:
                        mtldict.update(built)
                        allbuilt.extend(built.keys())

    if allfailed:
        logmes("the following materials failed to build: "
               "{0}".format(", ".join(allfailed)))

    if allbuilt:
        logmes("the following materials were built: "
               "{0}".format(", ".join(allbuilt)))

    if mtldict:
        write_mtldb(mtldict)

    return
if __name__ == "__main__":
    main()
