import os
import re
import sys
import imp
import time
import shutil
import random
import string
import argparse
import datetime
import numpy as np
import multiprocessing as mp

from core.logger import Logger
from utils.namespace import Namespace
from utils.misc import fillwithdots, remove, load_file
from core.product import SPLASH, TEST_DIRS, TEST_CONS_WIDTH
from core.test import PASSED, DIFFED, FAILED, FAILED_TO_RUN, NOT_RUN
from core.test import TestBase, TestError as TestError



TIMING = []
E_POST = ".post"
F_POST = "graphics.html"
TESTRE = re.compile(r"(?:^|[\\b_\\.-])[Tt]est")
RES_MAP = {PASSED: "PASS", DIFFED: "DIFF", FAILED: "FAIL",
           FAILED_TO_RUN: "FAILED TO RUN", NOT_RUN: "NOT RUN"}
ROOT_RES_D = os.path.join(os.getcwd(), "TestResults.{0}".format(sys.platform))
if not os.path.isdir(ROOT_RES_D):
    os.makedirs(ROOT_RES_D)
logfile = os.path.join(ROOT_RES_D, "testing.log")
logger = Logger(logfile=logfile, ignore_opts=1)
INI, TTR = 0, 1


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    p = argparse.ArgumentParser()
    p.add_argument("-k", action="append", default=[],
        help="Keywords to include [default: ]")
    p.add_argument("-K", action="append", default=[],
        help="Keywords to exclude [default: ]")
    p.add_argument("-j", type=int, default=1,
        help="Number of simutaneous tests to run [default: ]")
    p.add_argument("--no-tear-down", action="store_true", default=False,
        help="Do not tear down passed tests on completion [default: ]")
    p.add_argument("--html", action="store_true", default=False,
        help="Write html summary of results (negates tear down) [default: ]")
    p.add_argument("--overlay", action="store_true", default=False,
        help=("Create overlays of failed tests with baseline (negates tear "
              "down) [default: ]"))
    p.add_argument("sources", nargs="*")

    # parse out known arguments and reset sys.argv
    args, sys.argv[1:] = p.parse_known_args(argv)

    # suppress logging from other products
    sys.argv.extend(["-v", "0"])

    sources = args.sources
    if not sources:
        sources.extend(TEST_DIRS)
    gather_and_run_tests(args.sources, args.k, args.K, nprocs=args.j,
        tear_down=not args.no_tear_down, html_summary=args.html,
        overlay=args.overlay)


def result_str(i):
    return RES_MAP.get(i, "UNKNOWN")

def sort_by_status(test):
    return test.status

def sort_by_time(test):
    if not test.instance:
        return 4
    if "long" in test.instance.keywords:
        return 0
    if "medium" in test.instance.keywords:
        return 1
    if "fast" in test.instance.keywords:
        return 2
    return 3

def sort_by_run_stat(test):
    if test.instance:
        # test will run
        return 3
    if test.disabled:
        return 2
    return 1

def gather_and_run_tests(sources, include, exclude, tear_down=True,
                         html_summary=False, nprocs=1, overlay=False):
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

    if not tear_down:
        html_summary = True
    elif html_summary:
        tear_down = False
    if overlay:
        tear_down = False
        html_summary = True

    sources = [os.path.realpath(s) for s in sources]

    logger.write("summary of user input")
    s = "\n                ".join("{0}".format(x) for x in sources)
    kw = ", ".join("{0}".format(x) for x in include)
    KW = ", ".join("{0}".format(x) for x in exclude)
    logger.write(  "  TEST OUTPUT DIRECTORY: {4}"
                 "\n  TEST SOURCES: {0}"
                 "\n  KEYWORDS TO INCLUDE: {1}"
                 "\n  KEYWORDS TO EXCLUDE: {2}"
                 "\n  NUMBER OF SIMULTANEOUS JOBS: {3}"
                 "\n  TEAR DOWN OF PASSED TESTS: {5}"
                 "\n  CREATION OF HTML SUMMARY: {6}".format(
                     s, kw, KW, nprocs, ROOT_RES_D, tear_down, html_summary),
                 transform=str)

    # gather the tests
    TIMING.append(time.time())
    logger.write("\ngathering tests")
    opts = {"overlay": overlay}
    tests = gather_and_filter_tests(sources, ROOT_RES_D, include, exclude, **opts)

    for init_file in tests[INI]:
        init, module = load_file(init_file, disp=1)
        try:
            init.run(ROOT_RES_D)
        except AttributeError:
            logger.error("{0}: missing run attribute".format(init_file))
        del sys.modules[module]

    # write information
    TIMING.append(time.time())
    ntests = len(tests[TTR])
    ntests_to_run = len([t for t in tests[TTR] if t.instance])
    ndisabled = len([t for t in tests[TTR] if t.disabled])
    n_notinc = ntests - ntests_to_run - ndisabled
    n1, n2, n3 = False, False, False

    logger.write("FOUND {0:d} TESTS IN {1:.2}s".format(
        ntests, TIMING[-1]-TIMING[0]), transform=str)
    for test in sorted(tests[TTR], key=sort_by_run_stat):
        if test.instance and not n1:
            n1 = True
            logger.write("  tests to be run ({0:d})".format(ntests_to_run))
        elif test.disabled and not n2:
            n2 = True
            logger.write("  disabled tests ({0:d})".format(ndisabled))
        elif not n3:
            n3 = True
            logger.write("  tests filtered out by keyword "
                         "request ({0:d})".format(n_notinc))
        logger.write("    {0}".format(test.str_repr), transform=str)

    if not ntests_to_run:
        logger.write("nothing to do")
        return

    nprocs = min(min(mp.cpu_count(), nprocs), len(tests[TTR]))

    # run the tests
    logger.write("\nRUNNING TESTS")
    if nprocs == 1:
        p = []
        for test in tests[TTR]:
            p.append(run_test(test))
    else:
        pool = mp.Pool(processes=nprocs)
        try:
            p = pool.map(run_test, tests[TTR])
            pool.close()
            pool.join()
        except KeyboardInterrupt:
            logger.error("keyboard interrupt")
            raise SystemExit("KeyboardInterrupt intercepted")

    # transfer info back to the actual test. this is done because the changed
    # state of the test is not persistent with multiprocessing
    if len(p) != ntests:
        logger.error("an error during testing is preventing proper diagnostics")
        p = [(-12, np.nan)] * ntests

    for i, (status, dtime) in enumerate(p):
        tests[TTR][i].status = status
        tests[TTR][i].dtime = dtime
        if tests[TTR][i].instance:
            tests[TTR][i].instance.status = status


    # write out some information
    TIMING.append(time.time())
    dtf = TIMING[-1] - TIMING[0]
    logger.write(fillwithdots("ALL TESTS COMPLETED",
                              "({0:.4f}s)".format(dtf),
                              TEST_CONS_WIDTH),
                 transform=str)

    # determine number of passed, failed, etc.
    npass = len([test for test in tests[TTR] if test.status == PASSED])
    nfail = len([test for test in tests[TTR] if test.status == FAILED])
    nftr = len([test for test in tests[TTR] if test.status == FAILED_TO_RUN])
    ndiff = len([test for test in tests[TTR] if test.status == DIFFED])
    nnr = len([test for test in tests[TTR] if test.status == NOT_RUN])
    nunkn = ntests - npass - nfail - nftr - ndiff
    logger.write("\nsummary of completed tests\nran {0:d} tests "
                 "in {1:.4f}s".format(ntests_to_run, TIMING[-1]-TIMING[0]))
    logger.write(
        "  {0: 3d} passed\n"
        "  {1: 3d} failed\n"
        "  {2: 3d} diffed\n"
        "  {3: 3d} failed to run\n"
        "  {4: 3d} not run\n"
        "  {5: 3d} unknown".format(npass, nfail, ndiff, nftr, nnr, nunkn))

    # collect results for pretty logging
    results_by_module = {}
    results_by_status = {}
    for test in tests[TTR]:
        if test.status == NOT_RUN:
            continue
        S = result_str(test.status)
        s = "{0}: {1}".format(test.str_repr.split(".")[1], S)
        results_by_module.setdefault(test.module, []).append(s)
        results_by_status.setdefault(S, []).append(test.str_repr)

    logger.write("\ndetail by test module")
    for (module, statuses) in results_by_module.items():
        logger.write("  " + module, transform=str)
        s = "\n".join("    {0}".format(x) for x in statuses)
        logger.write(s, transform=str)

    logger.write("\ndetail by test status")
    for (status, str_reprs) in results_by_status.items():
        logger.write("  " + status)
        s = "\n".join("    {0}".format(x) for x in str_reprs)
        logger.write(s, transform=str)

    if tear_down:
        logger.write("\ntearing down passed tests", end=" ")
        logger.write("(--no-tear-down SUPPRESSES TEAR DOWN)", transform=str)
        # tear down indivdual tests
        torn_down = 0
        for test in tests[TTR]:
            if test.status == PASSED:
                test.instance.tear_down()
                torn_down += test.instance.torn_down

        if torn_down == ntests_to_run:
            logger.write("all tests passed, removing entire results directory")
            shutil.rmtree(ROOT_RES_D)
        else:
            logger.write("failed to tear down all tests, generating html summary")
            html_summary = True

    if html_summary:
        logger.write("\nWRITING HTML SUMMARY TO {0}".format(ROOT_RES_D),
                     transform=str)
        write_html_summary(ROOT_RES_D, tests[TTR])

    logger.finish()

    return


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


def gather_and_filter_tests(sources, root_dir, include, exclude, **opts):
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
    all_tests : list of Namespace instances
        each test is its own Namespace with the following attributes
        test.file_dir : directory where test file resides
        test.test_dir : directory where test will be run
        test.file_name: file name of test
        test.all_tests: all tests in M
        test.instance : the test class instance - if it is to be run - else None
        test.str_repr : string representation for the test

    """
    all_tests = {INI: [], TTR: []}
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
            i_file = os.path.join(dirname, "test_init.py")
            if os.path.isfile(i_file):
                all_tests[INI].append(i_file)
            test_files.extend([os.path.join(dirname, f) for f in files
                               if TESTRE.search(f) and f.endswith(".py")])

    for test_file in test_files:

        # load the test file and extract tests from it
        module, file_dir, file_name, tests, reprs = load_test_file(test_file)

        # create directory to run the test
        for (i, test_cls) in enumerate(tests):
            kwargs = {"str_repr": reprs[i],
                      "file_dir": file_dir,
                      "file_name": file_name,
                      "instance": None,
                      "status": NOT_RUN,
                      "module": module,
                      "disabled": False}

            test_space = Namespace(**kwargs)
            all_tests[TTR].append(test_space)

            # validate and filter tests
            # instantiate test and set defaults
            # this is
            the_test = test_cls()
            try:
                the_test.init_and_check(root_dir, file_dir, module,
                                        reprs[i], **opts)
            except TestError as e:
                logger.error("THE FOLLOWING ERRORS WERE ENCOUNTERED WHILE "
                             "INITIALIZING TEST {0}".format(reprs[i]),
                             transform=str)
                logger.write(e.args[0])
                logger.error("skipping test")
                continue

            if the_test.disabled:
                all_tests[TTR][-1].disabled = True
                continue

            # filter tests to be excluded
            if any([kw in the_test.keywords for kw in exclude]):
                continue

            # keep only tests wanted
            if include and not all([kw in the_test.keywords for kw in include]):
                continue

            # if we got to here, the test will be run, store the instance
            all_tests[TTR][-1].instance = the_test
            all_tests[TTR][-1].test_dir = the_test.test_dir

    all_tests[TTR] = sorted(all_tests[TTR], key=sort_by_time)

    return all_tests


def run_test(test):
    """Run a single test

    Parameters
    ----------
    test : Namespace object
        A Namespace object as described in gather_and_filter_tests

    Notes
    -----
    Nothing is returned, the Namespace is modified with any information

    """
    W = TEST_CONS_WIDTH
    if not test.instance:
        return NOT_RUN, np.nan

    ti = time.time()
    logger.write(fillwithdots(test.str_repr, "RUNNING", W),
                 transform=str)
    try:
        test.instance.setup()
    except TestError as e:
        logger.error("{0}: failed to setup with the following "
                     "errors".format(test.str_repr))
        logger.write(e.args[0])
        dtime = np.nan
    else:
        test.instance.run()
        logger.write(fillwithdots(test.str_repr, "POST HOOK", W), transform=str)
        dtime = time.time() - ti

    status = test.instance.status
    dtime = dtime
    line = fillwithdots(test.str_repr, "FINISHED", W)
    s = " [{1}] ({0:.1f}s)".format(dtime, result_str(status))
    logger.write(line + s, transform=str)

    return status, dtime


def write_html_summary(root_d, tests):
    """write summary of the results to an html file

    """
    # html header
    html = []
    html.append("<html>\n<head>\n<title>Test Results</title>\n</head>\n")
    html.append("<body>\n<h1>Summary</h1>\n")

    now = datetime.datetime.now()
    html.append("<ul>\n")

    # meta information for all tests
    html.append("<li> Directory: {0} </li>\n".format(root_d))
    html.append("<li> Date: {0} </li>\n".format(now.ctime()))
    options = " ".join(arg for arg in sys.argv[1:])
    html.append("<li> Options: {0} </li>\n".format(options))

    # summary of statuses
    groups = {}
    for test in tests:
        S = result_str(test.status)
        groups.setdefault(S, []).append(1)
    groups = ["{0} {1}".format(sum(v), k) for (k,v) in groups.items()]
    html.append("<li> {0} </li>\n".format(", ".join(groups)))
    html.append("</ul>\n")

    # write out tests by test status
    HF = "<h1>Tests that showed '{0}'</h1>\n"
    results_by_status = {}
    for test in tests:
        S = result_str(test.status)
        results_by_status.setdefault(S, []).append(test)

    for (status, the_tests) in results_by_status.items():
        html.append(HF.format(status))
        for test in the_tests:
            html_summary = test_html_summary(test, root_d)
            if not html_summary:
                continue
            html.append(html_summary)
    html.append("</body>")

    filename = os.path.join(root_d, "summary.html")
    with open(filename, "w") as fh:
        fh.write("\n".join(html))

    return

def test_html_summary(test, root_d):
    # get info from details
    if not test.instance:
        return "<ul> <li> {0} </li> </ul>".format(test.str_repr)

    test_dir = test.instance.test_dir

    if not os.path.isdir(test_dir):
        return "<ul> <li> {0} </li> </ul>".format(test.str_repr)

    status = result_str(test.status)
    keywords = ", ".join(test.instance.keywords)
    tcompletion = "{0:.2f}s".format(test.dtime)

    # look for post processing link
    html_link = os.path.join(test_dir, "png", "graphics.html")
    if not os.path.isfile(html_link):
        html_link = None

    html = []
    html.append("<ul>\n")
    html.append("<li>{0}</li>".format(test.str_repr))
    html.append("<ul>")

    html.append("<li>Files: ")
    files = [f for f in os.listdir(test_dir)
             if os.path.isfile(os.path.join(test_dir, f))]
    for f in files:
        fpath = os.path.join(test_dir, f).replace(root_d, ".")
        html.append("<a href='{0}' type='text/plain'>{1}</a> ".format(fpath, f))
        continue
    if html_link:
        html.append("<li><a href='{0}'>Plots</a>".format(os.path.join(html_link)))
    html.append("<li>Keywords: {0}".format(keywords))
    html.append("<li>Status: {0}".format(test.status))
    html.append("<li>Completion time: {0}".format(test.dtime))
    html.append("</ul>")
    html.append("</ul>\n")
    return "\n".join(html)


if __name__ == "__main__":
    main()
