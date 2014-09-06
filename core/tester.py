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

from core.product import SPLASH
from core.logger import Logger
from core.test import TestBase, PASSED, DIFFED, FAILED, FAILED_TO_RUN, NOT_RUN
from utils.namespace import Namespace


TIMING = []
WIDTH = 80
E_POST = ".post"
F_POST = "graphics.html"
TESTRE = re.compile(r"(?:^|[\\b_\\.-])[Tt]est")
RES_MAP = {PASSED: "PASS", DIFFED: "DIFF", FAILED: "FAIL",
           FAILED_TO_RUN: "FAILED TO RUN", NOT_RUN: "NOT RUN"}
RES_D = os.path.join(os.getcwd(), "TestResults.{0}".format(sys.platform))
if not os.path.isdir(RES_D):
    os.makedirs(RES_D)
logfile = os.path.join(RES_D, "testing.log")
logger = Logger(logfile=logfile, ignore_opts=1)


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
    p.add_argument("sources", nargs="+")

    # parse out known arguments and reset sys.argv
    args, sys.argv[1:] = p.parse_known_args(argv)

    # suppress logging from other products
    sys.argv.extend(["-v", "0"])

    gather_and_run_tests(args.sources, args.k, args.K, nprocs=args.j,
        tear_down=not args.no_tear_down, html_summary=args.html)


def result_str(i):
    return RES_MAP.get(i, "UNKOWN")


def sort_by_status(test):
    return test.status

def sort_by_time(test):
    if not test.instance:
        return 3
    if "long" in test.instance.keywords:
        return 0
    if "fast" in test.instance.keywords:
        return 1
    return 2

def gather_and_run_tests(sources, include, exclude, tear_down=True,
                         html_summary=False, nprocs=1):
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

    if html_summary:
        tear_down = False

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
                     s, kw, KW, nprocs, RES_D, tear_down, html_summary),
                 transform=str)

    # gather the tests
    TIMING.append(time.time())
    logger.write("\ngathering tests")
    tests = gather_and_filter_tests(sources, RES_D, include, exclude)

    # write information
    TIMING.append(time.time())
    ntests = len(tests)
    ntests_to_run = len([t for t in tests if t.instance])
    ndisabled = len([t for t in tests if t.disabled])
    n_notinc = ntests - ntests_to_run - ndisabled

    logger.write("FOUND {0:d} TESTS IN {1:.2}s".format(
        ntests, TIMING[-1]-TIMING[0]), transform=str)
    for test in tests:
        if test.instance: star = " +++"
        elif test.disabled: star = "**"
        else: star = "*"
        logger.write("    {0}{1}".format(test.str_repr, star), transform=str)
    logger.write("(+++) tests to be run ({0:d})".format(ntests_to_run))
    if ndisabled:
        logger.write(" (**) disabled tests ({0:d})".format(ndisabled))
    if n_notinc:
        logger.write("  (*) tests filtered out by keyword "
                     "request ({0:d})".format(n_notinc))

    if not ntests_to_run:
        logger.write("nothing to do")
        return

    nprocs = min(min(mp.cpu_count(), nprocs), len(tests))

    # run the tests
    logger.write("\nRUNNING TESTS")
    output = []
    if nprocs == 1:
        for test in tests:
            output.append(run_test(test))
    else:
        pool = mp.Pool(processes=nprocs)
        try:
            p = pool.map_async(run_test, tests, callback=output.extend)
            p.wait()
            pool.close()
            pool.join()
        except KeyboardInterrupt:
            logger.error("keyboard interrupt")
            raise SystemExit("KeyboardInterrupt intercepted")

        # when multiprocessing, the results from run_test are saved.  why?
        for (i, test) in enumerate(tests):
            test.status, test.dtime = output[i]
            if test.instance:
                test.instance.status = output[i][0]

    logger.write("ALL TESTS COMPLETED")

    # write out some information
    TIMING.append(time.time())

    # determine number of passed, failed, etc.
    npass = len([test for test in tests if test.status == PASSED])
    nfail = len([test for test in tests if test.status == FAILED])
    nftr = len([test for test in tests if test.status == FAILED_TO_RUN])
    ndiff = len([test for test in tests if test.status == DIFFED])
    logger.write("\nsummary of completed tests\nran {0:d} tests "
                 "in {1:.4f}s".format(ntests_to_run, TIMING[-1]-TIMING[0]))
    logger.write("  {0: 3d} passed\n"
                 "  {1: 3d} failed\n"
                 "  {2: 3d} diffed\n"
                 "  {3: 3d} failed to run".format(npass, nfail, ndiff, nftr))

    # collect results for pretty logging
    results_by_module = {}
    results_by_status = {}
    for test in tests:
        if test.status == NOT_RUN: continue
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
        for test in tests:
            if test.status == PASSED:
                test.instance.tear_down()
                torn_down += test.instance.torn_down

        if torn_down == ntests_to_run:
            logger.write("all tests passed, removing results directory")
            shutil.rmtree(RES_D)

    if html_summary:
        logger.write("\nWRITING HTML SUMMARY TO {0}".format(RES_D), transform=str)
        write_html_summary(RES_D, tests)

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
    all_tests : list of Namespace instances
        each test is its own Namespace with the following attributes
        test.file_dir : directory where test file resides
        test.test_dir : directory where test will be run
        test.file_name: file name of test
        test.all_tests: all tests in M
        test.instance : the test class instance - if it is to be run - else None
        test.str_repr : string representation for the test
        test.test_cls : uninstanteated test class

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

    all_tests = []
    for test_file in test_files:

        # load the test file and extract tests from it
        module, file_dir, file_name, tests, reprs = load_test_file(test_file)

        # create directory to run the test
        d = os.path.basename(file_dir)
        test_dir = os.path.join(root_dir, d, module)
        if not os.path.isdir(test_dir):
            os.makedirs(test_dir)

        for (i, test_cls) in enumerate(tests):
            kwargs = {"str_repr": reprs[i],
                      "test_dir": test_dir,
                      "file_dir": file_dir,
                      "file_name": file_name,
                      "instance": None,
                      "status": NOT_RUN,
                      "module": module,
                      "disabled": False,
                      "test_cls": test_cls}

            test_space = Namespace(**kwargs)
            all_tests.append(test_space)

            # validate and filter tests
            # instantiate test and set defaults
            # this is
            the_test = test_cls()
            the_test.init(file_dir, test_dir, module, logger)
            validated = the_test.validate()
            if not validated:
                logger.error("{0}: skipping unvalidated test".format(module))
                continue

            if the_test.disabled:
                all_tests[-1].disabled = True
                continue

            # filter tests to be excluded
            if any([kw in the_test.keywords for kw in exclude]):
                continue

            # keep only tests wanted
            if include and not any([kw in the_test.keywords for kw in include]):
                continue

            # if we got to here, the test will be run, store the instance
            all_tests[-1].instance = the_test

    return sorted(all_tests, key=sort_by_time)


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
    if not test.instance:
        return NOT_RUN, np.nan
    ti = time.time()
    logger.write(fillwithdots(test.str_repr, "RUNNING"), transform=str)
    status = test.instance.setup()
    if status:
        logger.error("{0}: failed to setup".format(test.str_repr))
        return test.status, np.nan
    test.instance.run()
    test.instance.post_hook()
    dtime = time.time() - ti
    test.status = test.instance.status
    test.dtime = dtime
    line = fillwithdots(test.str_repr, "FINISHED")
    s = " [{1}] ({0:.1f}s)".format(dtime, result_str(test.status))
    logger.write(line + s, transform=str)

    return test.status, dtime


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
    status = result_str(test.status)
    keywords = ", ".join(test.instance.keywords)
    tcompletion = "{0:.2f}s".format(test.dtime)

    # look for post processing link
    html_link = os.path.join(test_dir, test.instance.runid + E_POST, F_POST)
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
