import os
import re
import sys
import imp
import time
import random
import string
import argparse
from matmodlab import SPLASH
from core.runtime import set_runtime_opt
from core.test import TestBase, PASSED, DIFFED, FAILED, FAILED_TO_RUN

TESTRE = re.compile(r"(?:^|[\\b_\\.-])[Tt]est")
STR_RESULTS = {PASSED: "PASS", DIFFED: "DIFF", FAILED: "FAIL",
               FAILED_TO_RUN: "FAILED TO RUN"}
TIMING = []
WIDTH = 80

class Logger(object):
    def __init__(self, d):
        self.ch = sys.__stdout__
        self.fh = open(os.path.join(d, "testing.log"), "w")
    def write(self, string, end="\n"):
        self.ch.write(string + end)
        self.fh.write(string + end)
    def warn(self, string, end="\n"):
        string = "*** WARNING: {0}".format(string)
        self.write(string, end)
    def error(self, string, end="\n"):
        string = "*** ERROR: {0}".format(string)
        self.write(string, end)
    def finish(self):
        self.ch.flush()
        self.fh.flush()
        self.fh.close()

root_dir = os.path.join(os.getcwd(), "TestResults.{0}".format(sys.platform))
if not os.path.isdir(root_dir):
    os.makedirs(root_dir)
logger = Logger(root_dir)


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    p = argparse.ArgumentParser()
    p.add_argument("-k", action="append", default=[],
        help="Keywords to include [default: ]")
    p.add_argument("-K", action="append", default=[],
        help="Keywords to exclude [default: ]")
    p.add_argument("sources", nargs="+")

    # parse out known arguments and reset sys.argv
    args, sys.argv[1:] = p.parse_known_args(argv)

    # suppress logging from other products
    sys.argv.extend(["-v", "0"])

    gather_and_run_tests(args.sources, args.k, args.K)


def gather_and_run_tests(sources, include, exclude):
    """Gather and run all tests

    Parameters
    ----------
    sources : list
        files or directories to scan for tests
    include : list
        list of test keywords to include
    exclude : list
        list of test keywords to exclude

    """
    logger.write(SPLASH)

    sources = [os.path.realpath(s) for s in sources]

    s = "\n".join("    {0}".format(x) for x in sources)
    kw = ", ".join("{0}".format(x) for x in include)
    KW = ", ".join("{0}".format(x) for x in exclude)
    logger.write("\nGATHERING TESTS FROM\n{0}".format(s))
    logger.write("KEYWORDS TO INCLUDE\n    {0}".format(kw))
    logger.write("KEYWORDS TO EXCLUDE\n    {0}".format(KW))

    # gather the tests
    TIMING.append(time.time())
    tests = gather_and_filter_tests(sources, root_dir, include, exclude)

    # write information
    TIMING.append(time.time())
    ntests = sum(len(tests[m]["filtered"]) for m in tests)
    logger.write("FOUND {0:d} TESTS IN {1:.2}s".format(
        ntests, TIMING[-1]-TIMING[0]))

    # run the tests
    logger.write("\nRUNNING TESTS")
    results = run_tests(tests)
    logger.write("ALL TESTS COMPLETED")

    # write out some information
    TIMING.append(time.time())
    logger.write("\nSUMMARY OF TESTS\nRAN {0:d} TESTS "
                 "IN {1:.4f}s".format(ntests, TIMING[-1]-TIMING[0]))

    npass, nfail, nftr, ndiff = 0, 0, 0, 0
    S = []
    for (module, stats) in results.items():
        if not tests[module]["filtered"]:
            continue
        npass += len([i for i in stats if i == PASSED])
        nfail += len([i for i in stats if i == FAILED])
        nftr += len([i for i in stats if i == FAILED_TO_RUN])
        ndiff += len([i for i in stats if i == DIFFED])
        s = "[{0}]".format(" ".join(STR_RESULTS[i] for i in stats))
        S.append("    {0}: {1}".format(module, s))

    logger.write("   {0: 3d} PASSED\n"
                 "   {1: 3d} FAILED\n"
                 "   {2: 3d} DIFFED\n"
                 "   {3: 3d} FAILED TO RUN\nDETAILS".format(npass,
                                                           nfail, ndiff, nftr))
    logger.write("\n".join(S))

    logger.finish()
    return


def fillwithdots(a, b):
    dots = "." * (WIDTH - len(a) - len(b))
    return "{0}{1}{2}".format(a, dots, b)

def isclass(item):
    return type(item) == type(object)


def load_test_file(test_file):
    d, name = os.path.split(test_file)
    module = os.path.splitext(name)[0]
    if module in sys.modules:
        module = module + "".join(random.sample(string.ascii_letters, 4))
    if d not in sys.path:
        sys.path.insert(0, d)
    loaded = imp.load_source(module, test_file)

    tests = []
    reprs = []
    for item in dir(loaded):
        attr = getattr(loaded, item)
        if not isclass(attr): continue
        if issubclass(attr, TestBase) and attr != TestBase:
            tests.append(attr)
            reprs.append(re.sub(r"[<\'>]|(class)", " ", repr(attr)).strip())
    return module, d, name, tests, reprs


def gather_and_filter_tests(sources, root_dir, include, exclude):
    """Gather all tests

    Parameters
    ----------
    sources : list
        files or directories to scan for tests
    root_dir : str
        root directory for running tests
    include : list
        list of test keywords to include
    exclude : list
        list of test keywords to exclude

    Returns
    -------
    all_tests : dict
        test = all_tests[M] contains test information for module M
    u          test["file_dir"]: directory where test file resides
               test["file_name"]: file name of test
               test["all_tests"]: all tests in M
               tests["filtered"]: the tests to run

    """
    if not isinstance(sources, (list, tuple)):
        sources = [sources]

    # gather tests
    test_files = []
    for source in sources:
        item = os.path.realpath(source)
        if not os.path.exists(item):
            logger.warn("{0}: no such file or directory".format(source))
            continue
        if os.path.isfile(item):
            test_files.append(item)
            continue
        for (dirname, dirs, files) in os.walk(item):
            test_files.extend([os.path.join(dirname, f) for f in files
                               if TESTRE.search(f) and f.endswith(".py")])

    # load tests
    all_tests = {}
    for test_file in test_files:
        module, file_dir, file_name, tests, reprs = load_test_file(test_file)
        d = file_dir.replace(os.getcwd() + os.path.sep, "")
        test_dir = os.path.join(root_dir, d)
        if not os.path.isdir(test_dir):
            os.makedirs(test_dir)
        all_tests[module] = {"tests": tests, "repr": reprs,
                             "test_dir": test_dir, "file_dir": file_dir,
                             "file_name": file_name}

    # validate and filter tests
    for (module, info) in all_tests.items():
        all_tests[module]["filtered"] = []
        test_dir = info["test_dir"]
        file_dir = info["file_dir"]
        for (i, test_cls) in enumerate(info["tests"]):
            test_instance = test_cls()
            validated = test_instance.validate(file_dir, test_dir, module, logger)
            if not validated:
                logger.error("{0}: SKIPPING UNVALIDATED TEST".format(module))
                continue

            disabled = getattr(test_instance, "disabled", False)
            if disabled:
                continue

            # filter tests to be excluded
            if any([kw in test_instance.keywords for kw in exclude]):
                continue

            # keep only tests wanted
            if include and not any([kw in test_instance.keywords
                                    for kw in include]):
                continue

            all_tests[module]["filtered"].append(test_instance)

    return all_tests


def run_tests(tests):
    results = {}
    for (module, info) in tests.items():
        d = info["file_dir"]
        results[module] = []
        for (i, the_test) in enumerate(info["filtered"]):
            test_repr = info["repr"][i]
            ti = time.time()
            logger.write(fillwithdots(test_repr, "RUNNING"))
            stat = the_test.setup(logger)
            if stat:
                logger.error("{0}: FAILED TO SETUP".format(module))
                results[module].append(self.stat)
                continue
            the_test.run(logger)
            results[module].append(the_test.stat)
            if the_test.stat == the_test.passed:
                the_test.tear_down()
            dt = time.time() - ti
            line = fillwithdots(test_repr, "FINISHED")
            s = " [{1}] ({0:.1f}s)".format(dt, STR_RESULTS[the_test.stat])
            logger.write(line + s)

    return results


if __name__ == "__main__":
    main()
