import os
import re
import sys
import time
import shutil
import argparse
import subprocess
import multiprocessing as mp
import xml.dom.minidom as xdom

import utils.xmltools as xmltools

D = os.path.dirname(os.path.realpath(__file__))
R = os.path.realpath(os.path.join(D, "../"))
TESTS = [os.path.join(R, "tests")]
TESTS.extend([x for x in os.getenv("GMDSETUPTSTDIR", "").split(os.pathsep) if x])
PATH = os.getenv("PATH", "").split(os.pathsep)
PLATFORM = sys.platform.lower()
PATH.append(os.path.join(R, "tpl/exowrap/Build_{0}/bin".format(PLATFORM)))
PASSSTATUS = 0
DIFFSTATUS = 1
FAILSTATUS = 2


class Error(Exception):
    def __init__(self, message):
        sys.stderr.write(message + "\n")
        raise SystemExit(2)


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    parser = argparse.ArgumentParser()
    parser.add_argument("-k", action="append", default=[],
        help="Keywords of tests to include [default: %(default)s]")
    parser.add_argument("-K", action="append", default=[],
        help="Keywords of tests to exclude [default: %(default)s]")
    parser.add_argument("-j", default=1, type=int,
        help="Number of simultaneous tests [default: %(default)s]")
    parser.add_argument("--list", action="store_true", default=False,
        dest="list_and_exit",
        help="List matching tests and exit [default: %(default)s]")
    parser.add_argument("--testdirs", action="append", default=[],
        help="Additional directories to find tests [default: %(default)s]")
    parser.add_argument("tests", nargs="*",
        help="Specific tests to run [default: %(default)s]")
    args = parser.parse_args(argv)

    # Directory to find tests
    dirs = TESTS
    for d in args.testdirs:
        if not os.path.isdir(d):
            logwrn("{0}: no such directory".format(d))
            continue
        dirs.append(d)

    rtests = find_rtests(dirs, args.k, args.K, args.tests)

    if args.list_and_exit:
        sys.exit(list_rtests(rtests))

    status = run_rtests(rtests, args.j)

    if status != PASSSTATUS:
        print "A test did not pass :("
    else:
        print "All tests passed!"
    return


def logmes(message):
    sys.stdout.write("runtest: {0}\n".format(message))
    sys.stdout.flush()


def logwrn(message=None, warnings=[0]):
    if message is None:
        return warnings[0]
    sys.stderr.write("*** runtest: warning: {0}\n".format(message))
    sys.stderr.flush()
    warnings[0] += 1


def rtest_statuses(status):
    return {PASSSTATUS: "PASS", DIFFSTATUS: "DIFF", FAILSTATUS: "FAIL"}.get(
        status, "FAIL")


def find_rtests(search_dirs, include, exclude, tests=None):
    """Find all regression tests in search_dirs

    """
    if not tests:
        # get a list of all test files (files with .rxml extension)
        test_files = []
        for d in search_dirs:
            for (dirname, dirs, files) in os.walk(d):
                test_files.extend([os.path.join(dirname, f) for f in files
                                   if f.endswith(".rxml")])
    else:
        # user supplied test file
        if not isinstance(tests, (tuple, list)):
            tests = [tests]
        test_files = [os.path.realpath(f) for f in tests]
        for test in test_files:
            if not os.path.isfile(test):
                raise Error("{0}: no such file".format(test))

    # put all found tests in the rtests dictionary
    rtests = {}
    for test_file in test_files:
        test_file_d = os.path.dirname(test_file)

        doc = xdom.parse(test_file)
        try:
            rtest = doc.getElementsByTagName("rtest")[0]
        except IndexError:
            raise Error("Expected root element rtest in {0}".format(test_file))

        # --- name
        name = rtest.attributes.get("name")
        if name is None:
            raise Error("{0}: rtest: name attribute required".format(
                os.path.basename(test_file)))
        name = str(name.value.strip())

        # --- keywords
        keywords = rtest.getElementsByTagName("keywords")
        if keywords is None:
            raise Error("{0}: rtest: keyword element required".format(name))
        keywords = xmltools.child2list(keywords, "lower")

        # --- link_files
        link_files = rtest.getElementsByTagName("link_files")
        if link_files is None:
            raise Error("{0}: rtest: link_files element "
                         "required".format(name))
        link_files = [os.path.join(test_file_d, f).format(NAME=name)
                      for f in xmltools.child2list(link_files)]
        for link_file in link_files:
            if not os.path.isfile(link_file):
                raise Error("{0}: no such file".format(link_file))

        # --- execute
        execute = []
        exct = rtest.getElementsByTagName("execute")
        if exct is None:
            raise Error("{0}: rtest: execute element "
                        "required".format(name))
        for item in exct:
            exe = item.attributes.get("name")
            if exe is None:
                raise Error("{0}: execute: name attribute "
                            "required".format(name))
            exe = exe.value.strip()
            x = which(exe)
            if x is None:
                raise Error("{0}: {1}: executable not found".format(name, exe))
            opts = [s.format(NAME=name) for s in xmltools.child2list([item])]
            if exe == "exodiff":
                opts.insert(0, "-status")
            execute.append([x] + opts)

        rtests[name] = {"Execute": execute, "Link Files": link_files,
                        "Keywords": keywords}
        doc.unlink()

    return filter_rtests(rtests, include, exclude)


def filter_rtests(rtests, include, exclude):
    """filter rtests based on keywords

    """
    skip = []
    for key, val in rtests.items():
        keywords = val["Keywords"]
        if any(kw in exclude for kw in keywords):
            skip.append(key)
        if include and not all(kw in keywords for kw in include):
            skip.append(key)
    for key in list(set(skip)):
        del rtests[key]
    return rtests


def run_rtests(rtests, nproc):
    """Run all of the rtests

    """
    testd = os.path.join(os.getcwd(), "TestResults.{0}".format(PLATFORM))

    if not os.path.isdir(testd):
        os.makedirs(testd)

    test_inp = ((testd, rtest, details) for (rtest, details) in rtests.items())
    nproc = min(min(mp.cpu_count(), nproc), len(rtests))
    if nproc == 1:
        statuses = [run_rtest(job) for job in test_inp]

    else:
        pool = mp.Pool(processes=nproc)
        statuses = pool.map(run_rtest, test_inp)
        pool.close()
        pool.join()

    return max(statuses)


def run_rtest(args):
    """Run the rtest

    """
    (testd, rtest, details) = args
    times = [time.time()]
    l = 50
    logmes("{0:{1}s} start".format(rtest, l))
    # make the test directory
    testd = os.path.join(testd, rtest)
    if os.path.isdir(testd):
        shutil.rmtree(testd)
    os.makedirs(testd)
    os.chdir(testd)

    # link the files
    for src in details["Link Files"]:
        dst = os.path.join(testd, os.path.basename(src))
        os.symlink(src, dst)

    # run each command
    status = []
    for (i, cmd) in enumerate(details["Execute"]):
        exe = os.path.basename(cmd[0])
        outf = exe + ".con"
        out = open(outf, "w")
        status.append(subprocess.call(" ".join(cmd), shell=True,
                                      stdout=out, stderr=subprocess.STDOUT))
        out.close()
        times.append(time.time())
        if status[-1] != 0:
            break

    status = max(status)

    stat = rtest_statuses(status)
    t = times[-1] - times[0]
    logmes("{0:{1}s} done({2:.2f}s) [{3}]".format(rtest, l, t, stat))

    return status


def list_rtests(rtests):
    for (rtest, details) in rtests.items():
        print "{0}: {1}".format(rtest, " ".join(details["Keywords"]))
    return 0


def which(exe):
    for d in PATH:
        x = os.path.join(d, exe)
        if os.path.isfile(x):
            return x
    return


if __name__ == "__main__":
    main()
