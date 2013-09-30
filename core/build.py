#!/opt/epd/bin/python2.7
import os
import sys
import imp
import argparse

from __config__ import __version__, SPLASH

D = os.path.dirname(os.path.realpath(__file__))
R = os.path.realpath(os.path.join(D, "../"))
MTLDIRS = [os.path.join(R, "materials/library")]
MTLDIRS.extend([x for x in os.getenv("GMDMTLS", "").split(os.pathsep) if x])
FC = os.getenv("FC", "gfortran")
LIBD = os.path.join(R, "lib")
UTLD = os.path.join(R, "utils")
FIO = os.path.join(UTLD, "mmlfio.f90")
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

    sys.stdout.write(SPLASH)
    log_message("Material Model Laboratory {0}".format(VERSION))
    log_message("looking for makemf files")

    allbuilt = []
    allfailed = []
    allskipped = []
    retcode = []
    for dirpath in MTLDIRS:
        for (d, dirs, files) in os.walk(dirpath):
            if "makemf.py" not in files:
                continue

            f = os.path.join(d, "makemf.py")
            log_message("building makemf in {0}".format(d), end="... ")
            makemf = imp.load_source("makemf", os.path.join(d, "makemf.py"))
            built, failed, skipped = makemf.makemf(
                LIBD, FC, FIO, materials=args.m)
            if failed:
                log_message("no", pre="")
            elif skipped:
                log_message("skipped", pre="")
            else:
                log_message("yes", pre="")

            if built:
                if len(built) == 4 and isinstance(built[0], basestring):
                    allbuilt.append(built)
                else:
                    allbuilt.extend(built)
            allfailed.extend(failed)
            allskipped.extend(skipped)
            retcode.append({True: 1, False: 0}[any(failed)])

    if allfailed:
        log_message("the following materials failed to build: "
                    "{0}".format(", ".join(allfailed)))

    if allskipped:
        log_message("the following materials were skipped: "
                    "{0}".format(", ".join(allskipped)))

    if allbuilt:
        log_message("the following materials were built: "
                    "{0}".format(", ".join(b[0] for b in allbuilt)))
        write_mtldb(allbuilt, wipe=args.w)

    return max(retcode)


if __name__ == "__main__":
    main()
