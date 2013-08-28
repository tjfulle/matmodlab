#!/opt/epd/bin/python2.7
import os
import sys
import imp
import argparse

from __config__ import __version__

D = os.path.dirname(os.path.realpath(__file__))
R = os.path.realpath(os.path.join(D, "../"))
MTLDIRS = [os.path.join(R, "materials")]
MTLDIRS.extend([x for x in os.getenv("GMDSETUPMTLDIR", "").split(os.pathsep) if x])
FC = os.getenv("FC", "gfortran")
LIBD = os.path.join(R, "lib")
UTLD = os.path.join(R, "utils")
FIO = os.path.join(UTLD, "gmdfio.f90")
VERSION = ".".join(str(x) for x in __version__)

from materials.material import write_mtldb


def log_message(message, end="\n", pre="build-mtl: "):
    sys.stdout.write("{0}{1}{2}".format(pre, message, end))
    sys.stdout.flush()


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", action="append",
        help="Material to build [default: all]")
    parser.add_argument("-w", action="store_true", default=False,
        help="Wipe material database before building [default: all]")
    args = parser.parse_args(argv)

    log_message("gmd {0}".format(VERSION))
    log_message("looking for makemf files")

    kwargs = {"FC": FC, "DESTD": LIBD, "MATERIALS": args.m, "FIO": FIO}
    mtldict = {}
    allfailed = []
    allbuilt = []
    for dirpath in MTLDIRS:
        for (d, dirs, files) in os.walk(dirpath):
            if "makemf.py" in files:
                f = os.path.join(d, "makemf.py")
                log_message("building makemf in {0}".format(d), end="... ")
                makemf = imp.load_source("makemf", os.path.join(d, "makemf.py"))
                made = makemf.makemf(**kwargs)
                failed = made.get("FAILED")
                built = made.get("BUILT")
                skipped = made.get("SKIPPED")

                if failed:
                    log_message("no", pre="")
                    allfailed.extend(failed)

                if skipped:
                    if not failed and not built:
                        log_message("skipped", pre="")

                if built:
                    if not failed:
                        log_message("yes", pre="")
                    if built:
                        mtldict.update(built)
                        allbuilt.extend(built.keys())

    if allfailed:
        log_message("the following materials failed to build: "
               "{0}".format(", ".join(allfailed)))

    if allbuilt:
        log_message("the following materials were built: "
               "{0}".format(", ".join(allbuilt)))

    if mtldict:
        write_mtldb(mtldict, wipe=args.w)

    return
if __name__ == "__main__":
    main()
