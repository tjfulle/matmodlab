import os
import re
import sys
import time
import shutil
import argparse
import datetime
import subprocess
import cPickle as pickle
import multiprocessing as mp
import xml.dom.minidom as xdom

import utils.xmltools as xmltools
from utils.namespace import Namespace

D = os.path.dirname(os.path.realpath(__file__))
R = os.path.realpath(os.path.join(D, "../"))
TESTS = [os.path.join(R, "tests")]
TESTS.extend([x for x in os.getenv("GMDSETUPTSTDIR", "").split(os.pathsep) if x])
PATH = os.getenv("PATH", "").split(os.pathsep)
PLATFORM = sys.platform.lower()
PATH.append(os.path.join(R, "tpl/exowrap/Build_{0}/bin".format(PLATFORM)))
NOTRUN_STATUS = -1
PASS_STATUS = 0
DIFF_STATUS = 1
FAIL_STATUS = 2

S_STAT = "Status"
S_PSTAT = "Previous Status"
S_LNFL = "Link Files"
S_EXEC = "Execute"
S_KWS = "Keywords"
S_TESTD = "Test Directory"
S_TIME = "Completion Time"

F_SUMMARY = "summary.html"
F_POST = "graphics.html"
F_DUMP = "completed_tests.db"

E_POST = ".post"
E_BASE = ".base_exo"
E_EXO = ".exo"

WIDTH = 70


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
    parser.add_argument("-F", action="store_true", default=False,
        help="Force tests previously run to rerun [default: %(default)s]")
    parser.add_argument("--plot", action="store_true", default=False,
        help="Create overlay plots for failed tests [default: %(default)s]")
    parser.add_argument("--list", action="store_true", default=False,
        dest="list_and_exit",
        help="List matching tests and exit [default: %(default)s]")
    parser.add_argument("--testdirs", action="append", default=[],
        help="Additional directories to find tests [default: %(default)s]")
    parser.add_argument("tests", nargs="*",
        help="Specific tests to run [default: %(default)s]")
    args = parser.parse_args(argv)

    log = sys.stdout

    # Directory to find tests
    dirs = TESTS
    for d in args.testdirs:
        if not os.path.isdir(d):
            log_warning("{0}: no such directory".format(d))
            continue
        dirs.append(d)

    # --- root directory to run tests
    testd = os.path.join(os.getcwd(), "TestResults.{0}".format(PLATFORM))

    # --- timer
    timing = Namespace()
    timing.start = time.time()

    # find the rtests
    rtests = find_rtests(dirs, args.k, args.K, args.tests)
    timing.tests_found = time.time()

    # list them if that is all that is wanted
    if args.list_and_exit:
        sys.exit(list_rtests(rtests))

    # how many did we find?
    log_message("Found {0} tests in {1:.2f}s".format(
        len(rtests), timing.tests_found - timing.start))

    completed_rtests = get_completed_rtests(testd)
    for rtest in completed_rtests:
        if rtest in rtests:
            if not args.F:
                log_message("{0}: test previously run.  use -F to "
                            "force a rerun".format(rtest))
                del rtests[rtest]
        cur_stat = completed_rtests[rtest][S_STAT]
        prev_stat = completed_rtests[rtest].get(S_PSTAT, cur_stat)
        completed_rtests[rtest][S_PSTAT] = prev_stat
        completed_rtests[rtest][S_STAT] = NOTRUN_STATUS
    for rtest in rtests:
        if rtest in completed_rtests:
            del completed_rtests[rtest]

    # run all of the tests
    if not rtests:
        log_message("No tests found matching criteria")
        if completed_rtests:
            dump_rtests_to_file(testd, completed_rtests)
            write_html_summary(testd, completed_rtests)
        return

    log_message("Running {0} tests".format(len(rtests)))
    rtests = run_rtests(testd, rtests, args.j)
    timing.tests_finished = time.time()
    log_message("All tests ran in {0:.2f}s".format(
        timing.tests_finished - timing.start))

    statuses = [details[S_STAT] for (rtest, details) in rtests.items()]
    status = max(statuses)
    if status != PASS_STATUS:
        log_message("1 or more tests did not pass")
    else:
        log_message("All tests passed")

    if args.plot:
        failed = [rtest for (rtest, details) in rtests.items()
                  if details[S_STAT] in (DIFF_STATUS, FAIL_STATUS)]
        if failed:
            log_message("Postprocessing {0} tests".format(len(failed)))

        for rtest in failed:
            postprocess_rtest(rtest, rtests[rtest])

    rtests.update(completed_rtests)
    dump_rtests_to_file(testd, rtests)
    write_html_summary(testd, rtests)

    return


def rtest_statuses(status=None):
    statuses = {PASS_STATUS: "PASS",
                DIFF_STATUS: "DIFF",
                FAIL_STATUS: "FAIL",
                NOTRUN_STATUS: "NOT RUN"}
    if status is None:
        return statuses
    return statuses.get(status, "FAIL")


def read_exo_file(filepath):
    import numpy as np
    from exoreader import ExodusIIReader
    exof = ExodusIIReader.new_from_exofile(filepath)
    glob_var_names = exof.glob_var_names()
    elem_var_names = exof.elem_var_names()
    data = [exof.get_all_times()]
    for glob_var_name in glob_var_names:
        data.append(exof.get_glob_var_time(glob_var_name))
    for elem_var_name in elem_var_names:
        data.append(exof.get_elem_var_time(elem_var_name, 0))
    data = np.transpose(np.array(data))
    head = ["TIME"] + glob_var_names + elem_var_names
    exof.close()
    ndumps = data.shape[0]
    nsteps = min(300, ndumps)
    step = int(ndumps / nsteps)
    return head, data[::step]


def postprocess_rtest(rtest, details):
    rtestd = details[S_TESTD]
    destd = os.path.join(rtestd, rtest + E_POST)
    file1 = os.path.join(rtestd, rtest + E_EXO)
    file2 = os.path.join(rtestd, rtest + E_BASE)
    if os.path.isfile(file1):
        if not os.path.isfile(file2):
            file2 = None
        create_overlay_plots(rtest, destd, file1, file2)
    return


def create_overlay_plots(rtest, destd, file1, file2=None):
    """Create overlays of variables common to files 1 and 2

    """
    exe = "postproc"
    import matplotlib.pyplot as plt
    # Make output directory for plots
    try:
        shutil.rmtree(destd)
    except OSError:
        pass
    os.mkdir(destd)

    ti = time.time()
    xvar = "TIME"
    head1, data1 = read_exo_file(file1)
    x1 = data1[:, head1.index(xvar)]
    label1 = os.path.basename(file1)

    if file2 is not None:
        head2, data2 = read_exo_file(file2)
        x2 = data2[:, head2.index(xvar)]
        label2 = os.path.basename(file2)

    aspect_ratio = 4. / 3.
    plots = []
    log_message("{0:{1}s} starting plots".format(rtest + ":", WIDTH - 9), exe)
    for yvar in [n for n in head1 if n != xvar]:
        name = yvar + ".png"
        f = os.path.join(destd, name)
        y1 = data1[:, head1.index(yvar)]
        plt.clf()
        plt.cla()
        plt.xlabel("TIME")
        plt.ylabel(yvar)
        if file2 is not None:
            if yvar not in head2:
                continue
            y2 = data1[:, head2.index(yvar)]
            plt.plot(x2, y2, ls="-", lw=4, c="orange", label=label2)
        plt.plot(x1, y1, ls="-", lw=2, c="green", label=label1)
        plt.legend(loc="best")
        plt.gcf().set_size_inches(aspect_ratio * 5, 5.)
        plt.savefig(f, dpi=100)
        plots.append(f)
    msg = "plots complete ({0:.2f}s)".format(time.time() - ti)
    log_message("{0:{1}s} {2}".format(rtest + ":", WIDTH - len(msg) + 5, msg), exe)

    # write an html summary
    with open(os.path.join(destd, F_POST), "w") as fobj:
        fobj.write("<html>\n<head>\n<title>{0}</title>\n</head>\n"
                   "<body>\n<table>\n<tr>\n".format(rtest))
        for i, plot in enumerate(plots):
            name = os.path.basename(plot)
            if i % 3 == 0 and i != 0:
                fobj.write("</tr>\n<tr>\n")
            width = str(int(aspect_ratio * 300))
            height = str(int(300))
            fobj.write("<td>{0}<a href='{1}'><img src='{1}' width='{2}' "
                       "height='{3}'></a></td>\n".format(name, plot, width, height))
        fobj.write("</tr>\n</table>\n</body>\n</html>")

    return


def log_message(message, exe="runtests"):
    sys.stdout.write("{0}: {1}\n".format(exe, message))
    sys.stdout.flush()


def log_warning(message=None, warnings=[0]):
    if message is None:
        return warnings[0]
    sys.stderr.write("*** runtests: warning: {0}\n".format(message))
    sys.stderr.flush()
    warnings[0] += 1


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

        rtests[name] = {S_EXEC: execute, S_LNFL: link_files,
                        S_KWS: keywords}
        doc.unlink()

    return filter_rtests(rtests, include, exclude)


def filter_rtests(rtests, include, exclude):
    """filter rtests based on keywords

    """
    skip = []
    for key, val in rtests.items():
        keywords = val[S_KWS]
        if any(kw in exclude for kw in keywords):
            skip.append(key)
        if include and not all(kw in keywords for kw in include):
            skip.append(key)
    for key in list(set(skip)):
        del rtests[key]
    return rtests


def run_rtests(testd, rtests, nproc):
    """Run all of the rtests

    """
    if not os.path.isdir(testd):
        os.makedirs(testd)

    test_inp = ((testd, rtest, details)
                for (rtest, details) in rtests.items())
    nproc = min(min(mp.cpu_count(), nproc), len(rtests))
    if nproc == 1:
        statuses = [run_rtest(job) for job in test_inp]

    else:
        pool = mp.Pool(processes=nproc)
        statuses = pool.map(run_rtest, test_inp)
        pool.close()
        pool.join()

    [statuses[0].update(d) for d in statuses[1:]]

    return statuses[0]


def run_rtest(args):
    """Run the rtest

    """
    (testd, rtest, details) = args[:3]
    times = [time.time()]
    log_message("{0:{1}s} start".format(rtest + ":", WIDTH))
    # make the test directory
    rtestd = os.path.join(testd, rtest)
    if os.path.isdir(rtestd):
        shutil.rmtree(rtestd)
    os.makedirs(rtestd)
    os.chdir(rtestd)

    # link the files
    for src in details[S_LNFL]:
        dst = os.path.join(rtestd, os.path.basename(src))
        os.symlink(src, dst)

    # run each command
    status = []
    for (i, cmd) in enumerate(details[S_EXEC]):
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
    msg = "done({0:.2f}s) [{1}]".format(t, stat)
    log_message("{0:{1}s} {2}".format(rtest + ":", WIDTH - len(msg) + 12, msg))

    details[S_TESTD] = rtestd
    details[S_STAT] = status
    details[S_TIME] = t

    return {rtest: details}


def list_rtests(rtests):
    for (rtest, details) in rtests.items():
        log_message("{0}: {1}".format(rtest, " ".join(details[S_KWS])))
    return 0


def which(exe):
    if exe.startswith("./"):
        return exe
    for d in PATH:
        x = os.path.join(d, exe)
        if os.path.isfile(x):
            return x
    return


def group_tests_by_status(tests_to_group):
    grouped = []
    rtests = tests_to_group.copy()
    for (code, status) in rtest_statuses().items():
        d = [{rtest: details} for (rtest, details) in rtests.items()
             if details[S_STAT] == code]
        [d[0].update(_) for _ in d[1:]]
        d = {} if not d else d[0]

        grouped.append([code, d])
        for rtest in d:
            del rtests[rtest]
        continue
    return sorted(grouped, key=lambda x: x[0])


def dump_rtests_to_file(testd, rtests):
    """Dump the rtests dictionary to a file

    """
    rtests_dump = os.path.join(testd, F_DUMP)
    with open(rtests_dump, "w") as fobj:
        pickle.dump(rtests, fobj)


def get_completed_rtests(testd):
    """Get rtests previously dumped to a file

    """
    rtests_dump = os.path.join(testd, F_DUMP)
    if not os.path.isfile(rtests_dump):
        return {}
    with open(rtests_dump, "r") as fobj:
        dumped_rtests = pickle.load(fobj)
    return dumped_rtests


def write_html_summary(testd, tests_to_summarize):
    """write summary of the results dictionary to html file

    """
    filename = os.path.join(testd, F_SUMMARY)

    # group tests by return status
    rtests = group_tests_by_status(tests_to_summarize)

    fobj = open(filename, "w")
    # write header
    fobj.write("<html>\n<head>\n<title>Test Results</title>\n</head>\n")
    fobj.write("<body>\n<h1>Summary</h1>\n")

    now = datetime.datetime.now()
    fobj.write("<ul>\n")
    fobj.write("<li> Directory: {0} </li>\n".format(testd))
    fobj.write("<li> Date: {0} </li>\n".format(now.ctime()))
    options = " ".join(arg for arg in sys.argv[1:] if not arg.endswith(".rxml"))
    fobj.write("<li> Options: {0} </li>\n".format(options))
    fobj.write("<li> ")
    for (code, group) in rtests:
        if group:
            fobj.write("{0} {1}".format(len(group), rtest_statuses(code)))
    fobj.write(" </li>\n")
    fobj.write("</ul>\n")

    # write out tests by test status
    HF = "<h1>Tests that showed '{0}'</h1>\n"
    for (code, group) in rtests:
        status = rtest_statuses(code)
        fobj.write(HF.format(status))
        for (rtest, details) in group.items():
            rtest_html_summary = generate_rtest_html_summary(
                rtest, details, testd)
            fobj.write(rtest_html_summary)

    fobj.write("</body>")
    fobj.flush()
    fobj.close()
    return


def generate_rtest_html_summary(rtest, details, testd):
    # get info from details
    rtestd = os.path.join(testd, rtest)
    status = rtest_statuses(details[S_STAT])
    prev_stat = details.get(S_PSTAT)
    if prev_stat is not None:
        status += " (previously: {0})".format(rtest_statuses(prev_stat))
    keywords = "  ".join(details[S_KWS])
    tcompletion = "{0:.2f}s".format(details[S_TIME])

    # look for post processing link
    try:
        plotd = [d for d in os.listdir(rtestd)
                 if os.path.isdir(d) and d.endswith(E_POST)][0]
        html_link = os.path.join(rtestd, plotd, F_POST)
    except IndexError:
        html_link = None

    rtest_html_summary = []
    rtest_html_summary.append("<ul>\n")
    rtest_html_summary.append("<li>{0}</li>".format(rtest))
    rtest_html_summary.append("<ul>")

    rtest_html_summary.append("<li>Files: ")
    files = [f for f in os.listdir(rtestd)
             if os.path.isfile(os.path.join(rtestd, f))]
    for f in files:
        fpath = os.path.join(rtestd, f).replace(testd, ".")
        rtest_html_summary.append("<a href='{0}' type='text/plain'>{1}</a> "
                   .format(fpath, f))
        continue
    if html_link:
        rtest_html_summary.append("<li><a href='{0}'>Plots</a>"
                                  .format(os.path.join(html_link)))
    rtest_html_summary.append("<li>Keywords: {0}".format(keywords))
    rtest_html_summary.append("<li>Status: {0}".format(status))
    rtest_html_summary.append("<li>Completion time: {0}".format(tcompletion))
    rtest_html_summary.append("</ul>")
    rtest_html_summary.append("</ul>\n")
    return "\n".join(rtest_html_summary)

if __name__ == "__main__":
    main()
