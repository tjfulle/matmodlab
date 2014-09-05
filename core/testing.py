import os
import re
import sys
import imp
import time
import shutil
import random
import string
import argparse
from matmodlab import SPLASH
from core.logger import Logger
from core.test import TestBase, PASSED, DIFFED, FAILED, FAILED_TO_RUN

TESTRE = re.compile(r"(?:^|[\\b_\\.-])[Tt]est")
STR_RESULTS = {PASSED: "PASS", DIFFED: "DIFF", FAILED: "FAIL",
               FAILED_TO_RUN: "FAILED TO RUN"}
TIMING = []
WIDTH = 80

ROOT_DIR = os.path.join(os.getcwd(), "TestResults.{0}".format(sys.platform))
if not os.path.isdir(ROOT_DIR):
    os.makedirs(ROOT_DIR)
logfile = os.path.join(ROOT_DIR, "testing.log")
logger = Logger(logfile=logfile, ignore_opts=1)


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    p = argparse.ArgumentParser()
    p.add_argument("-k", action="append", default=[],
        help="Keywords to include [default: ]")
    p.add_argument("-K", action="append", default=[],
        help="Keywords to exclude [default: ]")
    p.add_argument("--keep-passed", action="store_true", default=False,
        help="Do not tear down passed tests on completion [default: ]")
    p.add_argument("sources", nargs="+")

    # parse out known arguments and reset sys.argv
    args, sys.argv[1:] = p.parse_known_args(argv)

    # suppress logging from other products
    sys.argv.extend(["-v", "0"])

    gather_and_run_tests(args.sources, args.k, args.K,
                         tear_down=not args.keep_passed)


def gather_and_run_tests(sources, include, exclude, tear_down=True):
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
    logger.write("\nGATHERING TESTS FROM\n{0}"
                 "\nKEYWORDS TO INCLUDE\n    {1}"
                 "\nKEYWORDS TO EXCLUDE\n    {2}".format(s, kw, KW), transform=str)

    # gather the tests
    TIMING.append(time.time())
    tests = gather_and_filter_tests(sources, ROOT_DIR, include, exclude)

    # write information
    TIMING.append(time.time())
    ntests = sum(len(tests[m]["filtered"]) for m in tests)
    logger.write("FOUND {0:d} TESTS IN {1:.2}s".format(
        ntests, TIMING[-1]-TIMING[0]))

    # run the tests
    logger.write("\nRUNNING TESTS")
    results = run_tests(tests)
    logger.write("ALL TESTS COMPLETED")

    # collect results
    for (module, info) in tests.items():
        tests[module]["results"] = []
        for (i, the_test) in enumerate(info["filtered"]):
            tests[module]["results"].append(results[module][i])

    # write out some information
    TIMING.append(time.time())
    logger.write("\nsummary of tests\nran {0:d} tests "
                 "in {1:.4f}s".format(ntests, TIMING[-1]-TIMING[0]))

    S = []
    npass, nfail, nftr, ndiff = 0, 0, 0, 0
    for (module, info) in tests.items():

        if not info["filtered"]:
            continue

        my_results = dict([(I, 0) for I in STR_RESULTS])
        for (i, test) in enumerate(info["filtered"]):
            result = info["results"][i]
            my_results[result] += 1
            if result == PASSED and tear_down:
                test.tear_down()

        npass += my_results[PASSED]
        nfail += my_results[FAILED]
        nftr += my_results[FAILED_TO_RUN]
        ndiff += my_results[DIFFED]
        s = "[{0}]".format(", ".join(STR_RESULTS[i] for i in info["results"]))
        S.append("    {0}: {1}".format(module, s))

    logger.write("   {0: 3d} passed\n"
                 "   {1: 3d} failed\n"
                 "   {2: 3d} diffed\n"
                 "   {3: 3d} failed to run\n"
                 "details".format(npass, nfail, ndiff, nftr))
    logger.write("\n".join(S), transform=str)

    if tear_down:
        logger.write("\ntearing down passed tests")
        # tear down indivdual tests
        torn_down = 0
        for (module, info) in tests.items():
            for (i, test) in enumerate(info["filtered"]):
                if info["results"][i] == PASSED:
                    test.tear_down()
                    torn_down += test.torn_down

        if torn_down == ntests:
            logger.write("all tests passed, removing results directory", end=" ")
            logger.write("(--keep-passed SUPPRESSES TEAR DOWN)", transform=str)
            shutil.rmtree(ROOT_DIR)

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
        d = os.path.basename(file_dir)
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
            test_instance.logger = logger
            validated = test_instance.validate(file_dir, test_dir, module)
            if not validated:
                logger.error("{0}: skipping unvalidated test".format(module))
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
        results[module] = []
        for (i, the_test) in enumerate(info["filtered"]):
            test_repr = info["repr"][i]
            ti = time.time()
            logger.write(fillwithdots(test_repr, "RUNNING"), transform=str)
            stat = the_test.setup()
            if stat:
                logger.error("{0}: failed to setup".format(module))
                results[module].append(self.stat)
                continue
            the_test.run()
            results[module].append(the_test.stat)
            dt = time.time() - ti
            line = fillwithdots(test_repr, "FINISHED")
            s = " [{1}] ({0:.1f}s)".format(dt, STR_RESULTS[the_test.stat])
            logger.write(line + s, transform=str)

    return results


if __name__ == "__main__":
    main()
