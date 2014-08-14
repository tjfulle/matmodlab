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
from mml import SPLASH, TLS_D, TESTS_D, ROOT_D, MML_ENV, PYEXE

D = os.path.dirname(os.path.realpath(__file__))
R = os.path.realpath(os.path.join(D, "../"))
TESTS = os.path.join(R, "tests")
PLATFORM = sys.platform.lower()
D_TESTS =  os.path.join(os.getcwd(), "TestResults.{0}".format(PLATFORM))
NOTRUN_STATUS = -1
PASS_STATUS = 0
DIFF_STATUS = 1
FAIL_STATUS = 2

S_STAT = "Status"
S_PSTAT = "Previous Status"
S_LNFL = "Link Files"
S_EXEC = "Execute"
S_BDIR = "Base Directory"
S_KWS = "Keywords"
S_TESTD = "Test Directory"
S_TIME = "Completion Time"

F_RTEST_EXT = ".rxml"
F_RTEST_STAT = ".rtest-status"
F_SUMMARY = "summary.html"
F_POST = "graphics.html"
F_DUMP = "completed_tests.db"

E_POST = ".post"
E_BASE = ".base_exo"
E_EXO = ".exo"

WIDTH = 70

RAISE_ON_ERROR = True

class Error(Exception):
    def __init__(self, message, pre=""):
        self.message = message.rstrip()
        if RAISE_ON_ERROR:
            super(Error, self).__init__(message)
        else:
            sys.stderr.write("{0}*** error: runtests: {1}\n".format(pre, message))
            raise SystemExit(2)


class Environ(object):
    _env = dict(os.environ)
    def put(self, key, val):
        self._env[key] = val
    @property
    def env(self):
        return dict(self._env)


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    parser = argparse.ArgumentParser(fromfile_prefix_chars="@")
    parser.add_argument("-k", action="append", default=[],
        help="Keywords of tests to include [default: %(default)s]")
    parser.add_argument("-K", action="append", default=[],
        help="Keywords of tests to exclude [default: %(default)s]")
    parser.add_argument("-j", default=1, type=int,
        help="Number of simultaneous tests [default: %(default)s]")
    parser.add_argument("-s", metavar="\"X:Y\"", action="append", default=[],
        help=("Run simulation with model-Y instead of "
              "model-X [default: %(default)s]"))
    parser.add_argument("-i", action="store_true", default=False,
        help="Run in place [default: %(default)s]")
    parser.add_argument("-F", action="store_true", default=False,
        help="Force tests previously run to rerun [default: %(default)s]")
    parser.add_argument("--plot-failed", action="store_true", default=False,
        help="Create overlay plots for failed tests [default: %(default)s]")
    parser.add_argument("--plot-all", action="store_true", default=False,
        help="Create overlay plots for completed tests [default: %(default)s]")
    parser.add_argument("--list", action="store_true", default=False,
        dest="list_and_exit",
        help="List matching tests and exit [default: %(default)s]")
    parser.add_argument("-D", default=D_TESTS,
        help="Directory to run tests [default: %(default)s]")
    parser.add_argument("-w", default=False, action="store_true",
        help="Wipe test directory before running tests [default: %(default)s]")
    parser.add_argument("--rebaseline", default=False, action="store_true",
        help="Rebaseline test in current directory [default: %(default)s]")
    parser.add_argument("--run-failed", default=False, action="store_true",
        help="Run tests that previously had failed [default: %(default)s]")
    parser.add_argument("path", nargs="*",
        help="""Path[s] to directory[ies] to find tests, or to test
                specific test file[s].  Specify a file name with a newline
                separated list of paths by @filename. [default: %(default)s]""")
    args = parser.parse_args(argv)

    if args.rebaseline:
        sys.exit(_rebaseline_tests(d=os.getcwd(), files=args.path))

    log = sys.stdout

    sys.stdout.write(SPLASH)

    # Directory to find tests
    dirs, tests = [], []
    if not args.path:
        if os.path.isfile(F_RTEST_STAT):
            sys.exit(run_rtest_in_cwd())
        args.path = [TESTS_D]
        #parser.print_help()
        #raise Error("too few arguments: path to tests not specified", pre="\n")

    for p in args.path:
        p = os.path.realpath(p)
        if os.path.isdir(p):
            dirs.append(p)
        elif os.path.isfile and p.endswith(F_RTEST_EXT):
            tests.append(p)
        else:
            f, _ = os.path.splitext(p)
            if os.path.isfile(f + F_RTEST_EXT):
                tests.append(f + F_RTEST_EXT)
            else:
                log_warning("{0}: no such file or directory".format(p))

    if not dirs and not tests:
        raise Error("no test directories or files found")

    if not dirs and len(tests) == 1:
        # run inplace
        args.i = True

    # --- root directory to run tests
    testd = args.D

    if args.w and testd != os.getcwd():
        try: shutil.rmtree(testd)
        except OSError: pass

    if not os.path.isdir(testd):
        os.makedirs(testd)

    # --- timer
    timing = Namespace()
    timing.start = time.time()

    # perform any initialization
    if dirs:
        find_and_run_init(dirs, testd)

    # find the rtests
    if args.run_failed:
        rtests = get_completed_rtests(testd, failed_only=True)
        args.F = True
    else:
        rtests = find_rtests(dirs, args.k, args.K, tests)
    ntests = len(rtests)
    timing.tests_found = time.time()

    # list them if that is all that is wanted
    if args.list_and_exit:
        sys.exit(list_rtests(rtests))

    # how many did we find?
    log_message("Found {0} tests in {1:.2f}s".format(
        len(rtests), timing.tests_found - timing.start))

    statuses = []
    completed_rtests = get_completed_rtests(testd)
    for rtest in completed_rtests:
        if rtest in rtests:
            if not args.F:
                log_message("{0}: test previously run.  use -F to "
                            "force a rerun".format(rtest))
                del rtests[rtest]
                statuses.append(NOTRUN_STATUS)
        cur_stat = completed_rtests[rtest][S_STAT]
        prev_stat = completed_rtests[rtest].get(S_PSTAT, cur_stat)
        completed_rtests[rtest][S_PSTAT] = prev_stat
        completed_rtests[rtest][S_STAT] = NOTRUN_STATUS
    for rtest in rtests:
        if rtest in completed_rtests:
            del completed_rtests[rtest]

    # run all of the tests
    if not rtests:
        log_message("No tests to run")
        if not completed_rtests:
            return
        dump_rtests_to_file(testd, completed_rtests)
        write_html_summary(testd, completed_rtests)
        return

    else:
        log_message("Running {0} tests".format(len(rtests)))
        rtests = run_rtests(testd, rtests, args.j, args.i, args.s)
        timing.tests_finished = time.time()
        statuses.extend([details[S_STAT] for (rtest, details) in rtests.items()])
        log_message("{0} tests ran in {1:.2f}s".format(
            ntests, timing.tests_finished - timing.start))


    npass = statuses.count(PASS_STATUS)
    nfail = statuses.count(FAIL_STATUS)
    ndiff = statuses.count(DIFF_STATUS)
    nnrun = statuses.count(NOTRUN_STATUS)

    if (nfail + ndiff + nnrun) > 0:
        log_message("===== Summary of unsatisfactory tests")

    if nnrun > 0:
        log_message("not run summary:")
        for (rtest, details) in rtests.items():
            if details[S_STAT] == NOTRUN_STATUS:
                log_message("  {0}".format(rtest))

    if ndiff > 0:
        log_message("diff summary:")
        for (rtest, details) in rtests.items():
            if details[S_STAT] == DIFF_STATUS:
                log_message("  {0}".format(rtest))

    if nfail > 0:
        log_message("fail summary:")
        for (rtest, details) in rtests.items():
            if details[S_STAT] == FAIL_STATUS:
                log_message("  {0}".format(rtest))

    if npass: log_message("{0}/{1} tests passed".format(npass, ntests))
    if ndiff: log_message("{0}/{1} tests diffed".format(ndiff, ntests))
    if nfail: log_message("{0}/{1} tests failed".format(nfail, ntests))
    if nnrun: log_message("{0}/{1} tests not run".format(nnrun, ntests))

    if args.plot_all or args.plot_failed:
        if args.plot_all:
            to_plot = [rtest for (rtest, details) in rtests.items()]
        else:
            to_plot = [rtest for (rtest, details) in rtests.items()
                       if details[S_STAT] in (DIFF_STATUS, FAIL_STATUS)]
        if to_plot:
            log_message("Postprocessing {0} tests".format(len(to_plot)))

        for rtest in to_plot:
            postprocess_rtest(rtest, rtests[rtest])

    rtests.update(completed_rtests)
    dump_rtests_to_file(testd, rtests)
    write_html_summary(testd, rtests)

    return max(statuses)


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
    from utils.exo import ExodusIIFile
    exof = ExodusIIFile(filepath, "r")
    data = [exof.get_all_times()]
    for glob_var_name in exof.glob_var_names:
        data.append(exof.get_glob_var_time(glob_var_name))
    for elem_var_name in exof.elem_var_names:
        data.append(exof.get_elem_var_time(elem_var_name, 0))
    data = np.transpose(np.array(data))
    head = ["TIME"] + exof.glob_var_names + exof.elem_var_names
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
            y2 = data2[:, head2.index(yvar)]
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


def log_message(message, exe="runtests", pre=None):
    if pre is None:
        pre = "{0}: ".format(exe)
    sys.stdout.write("{0}{1}\n".format(pre, message))
    sys.stdout.flush()


def log_warning(message=None, warnings=[0]):
    if message is None:
        return warnings[0]
    sys.stderr.write("*** runtests: warning: {0}\n".format(message))
    sys.stderr.flush()
    warnings[0] += 1


def find_and_run_init(search_dirs, testd):
    """Find and run initialization files

    """
    xenv = Environ()
    xenv.put("PATH", MML_ENV["PATH"])
    xenv.put("PYTHONPATH", MML_ENV["PYTHONPATH"])
    # find all init.rxml files
    init_files = []
    for d in search_dirs:
        for (dirname, dirs, files) in os.walk(d):
            init_files.extend([os.path.join(dirname, f) for f in files
                               if f == "init.rxml"])
    # local variables
    known_vars = {"TESTDIR": testd,
                  "PYTHON": PYEXE}

    # run the initialization
    for init_file in init_files:
        dirname, filename = os.path.split(init_file)
        known_vars["DIRNAME"] = dirname

        # --- expand variables
        lines = open(init_file).read()
        lines = lines.format(**known_vars)
        # environment variables
        envars = re.findall(r"(?P<envar>\$\w+)", lines)
        for envar in envars:
            lines = lines.replace(envar, os.environ[envar.lstrip("$")])

        # --- parse the xml file
        doc = xdom.parseString(lines)
        try:
            init = doc.getElementsByTagName("init")[0]
        except IndexError:
            raise Error("expected root element init in {0}".format(init_file))

        # --- execute
        exe_stmnts = find_and_format_exes(init)
        if not exe_stmnts:
            continue
        status = []
        for cmd in exe_stmnts:
            exe = os.path.basename(cmd[0])
            if cmd[0] == PYEXE:
                exe += " " + cmd[1]
            outf = "_".join(exe.split()) + ".con"
            out = open(outf, "w")
            job = subprocess.Popen(cmd, env=xenv.env,
                                   stdout=out, stderr=subprocess.STDOUT)
            job.wait()
            status.append(job.returncode)
            out.close()
        status = max(status)

        if status:
            raise Error("failed to run: {0}".format(init_file))

        doc.unlink()
        continue


    return


def find_and_format_exes(element):
    """Find and format the contents of an execute element

    """
    exe_els = element.getElementsByTagName("execute")
    if not exe_els:
        return

    name = element.attributes.get("name")
    exe_stmnts = []
    for exe_el in exe_els:
        exe = exe_el.attributes.get("name")
        if exe is None:
            raise Error("execute: name attribute required")
        exe = exe.value.strip()
        x = which(exe)
        if x is None:
            raise Error("execute: {0}: executable not found".format(exe))
        opts = [s for s in xmltools.child2list([exe_el])]
        if exe == "exodiff":
            opts = ["-status", "-allow_name_mismatch"] + opts
        exe_stmnts.append(x.split() + opts)
    return exe_stmnts


def find_rtests(search_dirs, include, exclude, tests=None):
    """Find all regression tests in search_dirs

    """
    if not tests:
        # get a list of all test files (files with .rxml extension)
        test_files = []
        for d in search_dirs:
            for (dirname, dirs, files) in os.walk(d):
                test_files.extend([os.path.join(dirname, f) for f in files
                                   if f.endswith(F_RTEST_EXT) and f != "init.rxml"])
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
        try:
            parsed_file = parse_rxml(test_file)
            rtests.update(parsed_file)
        except Error as e:
            log_warning("unable to parse {0} due to the following error:\n"
                        "\t{1}".format(os.path.basename(test_file), e.message))

    return filter_rtests(rtests, include, exclude)


def parse_rxml(test_file):
    """Parse the xml test file

    """
    details = {}
    test_file_d = os.path.dirname(test_file)

    # read the root element - for now only to get the test name
    doc = xdom.parse(test_file)
    try:
        rtest = doc.getElementsByTagName("rtest")[0]
    except IndexError:
        raise Error("expected root element rtest in {0}".format(test_file))

    # --- name
    name = rtest.attributes.get("name")
    if name is None:
        raise Error("{0}: rtest: name attribute required".format(
            os.path.basename(test_file)))
    try:
        bdir, name = os.path.split(str(name.value.strip()))
    except AttributeError:
        bdir = "orphaned"
        name = str(name.value.strip())

    # --- reread the xml file, expanding known variables
    del doc
    lines = open(test_file).read()

    # local variables
    known_vars = {"NAME": name,
                  "DIRNAME": os.path.dirname(test_file),
                  "PYTHON": PYEXE}
    lines = lines.format(**known_vars)

    # environment variables
    envars = re.findall(r"(?P<envar>\$\w+)", lines)
    for envar in envars:
        lines = lines.replace(envar, os.environ[envar.lstrip("$")])

    # --- re-parse the xml file
    doc = xdom.parseString(lines)
    rtest = doc.getElementsByTagName("rtest")[0]

    # --- keywords
    keywords = rtest.getElementsByTagName("keywords")
    if keywords is None:
        raise Error("{0}: rtest: keyword element required".format(name))
    keywords = xmltools.child2list(keywords, "lower")

    # --- repeat test
    repeat = rtest.attributes.get("repeat")
    if repeat is None:
        Nrepeat = 1
    else:
        try:
            Nrepeat = int(repeat.value)
        except ValueError:
            raise Error("{0}: rtest: invalid value for 'repeat'".format(name))

    # --- link_files
    link_files = rtest.getElementsByTagName("link_files")
    if link_files:
        link_files = [os.path.join(test_file_d, f)
                      for f in xmltools.child2list(link_files)]
        for link_file in link_files:
            if not os.path.isfile(link_file):
                raise Error("{0}: no such file".format(link_file))

    # --- execute
    exe_stmnts = find_and_format_exes(rtest)
    if not exe_stmnts:
        raise Error("{0}: rtest: execute element required".format(name))

    if Nrepeat == 1:
        details[name] = {S_BDIR: bdir, S_EXEC: exe_stmnts, S_LNFL: link_files,
                         S_KWS: keywords}
    else:
        # Add multiple instances of the same test (for use with random inputs)
        Ndigits = len("{0:d}".format(Nrepeat))
        for idx in range(1, Nrepeat + 1):
            suffix = "_{0:0{1}d}".format(idx, Ndigits)
            details[name + suffix] = {S_BDIR: bdir,
                                      S_EXEC: exe_stmnts, S_LNFL: link_files,
                                      S_KWS: keywords}
    doc.unlink()
    return details


def filter_rtests(rtests, include, exclude):
    """filter rtests based on keywords

    """
    skip = []
    #        long, medium, short
    order = [[], [], []]
    for key, val in rtests.items():
        keywords = val[S_KWS]
        if any(kw in exclude for kw in keywords):
            skip.append(key)
        if include and not all(kw in keywords for kw in include):
            skip.append(key)
        # set the order to run the tests.  put long tests first
        if "long" in keywords:
            order[0].append(key)
        elif "medium" in keywords:
            order[1].append(key)
        elif "fast" in keywords:
            order[2].append(key)
        else:
            log_warning("{0}: expected one of [long, medium, fast] speed keywords")
    for key in list(set(skip)):
        del rtests[key]
    order = [key for speedlist in order for key in speedlist]
    for key in rtests:
        rtests[key]["order"] = order.index(key)

    return rtests


def run_rtests(testd, rtests, nproc, inplace, mtlswaplist=None):
    """Run all of the rtests

    """
    if mtlswaplist is None:
        mtlswaplist = []

    test_inp = ((testd, rtest, rtests[rtest], inplace, mtlswaplist)
                for rtest in sorted(rtests, key=lambda x: rtests[x]["order"]))
    nproc = min(min(mp.cpu_count(), nproc), len(rtests))
    statuses = []
    if nproc == 1:
        statuses.extend([run_rtest(job) for job in test_inp])

    else:
        pool = mp.Pool(processes=nproc)
        try:
            p = pool.map_async(run_rtest, test_inp, callback=statuses.extend)
            p.wait()
            pool.close()
            pool.join()
        except KeyboardInterrupt:
            raise SystemExit("KeyboardInterrupt caught")

    [statuses[0].update(d) for d in statuses[1:]]

    return statuses[0]


def run_rtest_in_cwd():
    try:
        rxml = [f for f in os.listdir(os.getcwd()) if f.endswith(F_RTEST_EXT)][-1]
    except IndexError:
        raise Error("no test found in PWD")
    details = parse_rxml(rxml)
    for (rtest, details) in details.items():
        break
    run_rtest((None, rtest, details, True, None))

def run_rtest(args):
    """Run the rtest

    """
    xenv = Environ()
    xenv.put("PATH", MML_ENV["PATH"])
    xenv.put("PYTHONPATH", MML_ENV["PYTHONPATH"])
    try:
        (testd, rtest, details, inplace, mtlswaplist) = args[:5]
        bdir = details[S_BDIR]
        times = [time.time()]
        log_message("{0:{1}s} running".format(rtest + ":", WIDTH))
        # make the test directory
        if inplace:
            rtestd = os.getcwd()
        else:
            rtestd = os.path.join(testd, details[S_BDIR], rtest)
            if os.path.isdir(rtestd):
                shutil.rmtree(rtestd)
            os.makedirs(rtestd)
        os.chdir(rtestd)

        # link the files
        if not inplace:
            for src in details[S_LNFL]:
                dst = os.path.join(rtestd, os.path.basename(src))
                os.symlink(src, dst)

        # Generate the material swap commands
        mtlswapcmd = []
        if mtlswaplist:
            for pair in mtlswaplist:
                mtlswapcmd.append("-s")
                mtlswapcmd.append('{0}'.format(pair))

        # run each command
        status = []
        for (i, cmd) in enumerate(details[S_EXEC]):
            exe = os.path.basename(cmd[0])
            if cmd[0] == PYEXE:
                exe += " " + cmd[1]
            outf = "_".join(exe.split()) + ".con"
            out = open(outf, "w")
            if exe == "mmd" and len(mtlswapcmd) > 0:
                cmd = cmd[:1] + mtlswapcmd + cmd[1:]
#                cmd.insert(1, mtlswapstr)
            job = subprocess.Popen(cmd, env=xenv.env,
                                   stdout=out, stderr=subprocess.STDOUT)
            job.wait()
            status.append(job.returncode)

            out.close()
            times.append(time.time())
            if status[-1] != 0:
                if exe == "mmd":
                    status[-1] = 2
                break

        status = max(status)

        stat = rtest_statuses(status)
        t = times[-1] - times[0]
        msg = "done({0:.2f}s) [{1}]".format(t, stat)
        log_message("{0:{1}s} {2}".format(rtest + ":", WIDTH - len(msg) + 14, msg))

        details[S_TESTD] = rtestd
        details[S_STAT] = status
        details[S_TIME] = t

        now = datetime.datetime.now()
        with open(F_RTEST_STAT, "a") as fobj:
            fobj.write("name: {0}\ndate: {1}\ntiming: {2}\nstatus: {3}".format(
                rtest, now, t, stat))

        return {rtest: details}

    except KeyboardInterrupt:
        return


def list_rtests(rtests):
    pre = ""
    log_message("found {0} tests".format(len(rtests)), pre=pre)
    i = max(len(k) for k in rtests)
    fmt = lambda t, k: "{0:{1}s} {2}".format(t, i, k)
    hline = "{0} {1}".format("-" * i, "-" * (80 - i))
    log_message(hline, pre=pre)
    log_message(fmt("name", "keywords"), pre=pre)
    log_message(hline, pre=pre)
    for (rtest, details) in rtests.items():
        kws = details[S_KWS]
        log_message(fmt(rtest, ", ".join(details[S_KWS])), pre=pre)
    log_message(hline, pre=pre)
    return 0


def which(exe):
    # Allow for exe = '/path/to/python pyfile.py --opts' and not choke.
    tmpexe = exe.split(None, 1)
    opts = "" if len(tmpexe) == 1 else " " + tmpexe[1]
    if tmpexe[0].startswith("./"):
        return tmpexe[0] + opts
    for d in MML_ENV["PATH"].split(os.pathsep):
        x = os.path.join(d, tmpexe[0])
        if os.path.isfile(x):
            return x + opts
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


def get_completed_rtests(testd, failed_only=False):
    """Get rtests previously dumped to a file

    """
    rtests_dump = os.path.join(testd, F_DUMP)
    if not os.path.isfile(rtests_dump):
        return {}
    with open(rtests_dump, "r") as fobj:
        dumped_rtests = pickle.load(fobj)
    if failed_only:
        for (rtest, info) in dumped_rtests.items():
            if info[S_STAT] == PASS_STATUS:
                del dumped_rtests[rtest]
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
    options = " ".join(arg for arg in sys.argv[1:] if not arg.endswith(F_RTEST_EXT))
    fobj.write("<li> Options: {0} </li>\n".format(options))
    groups = []
    for (code, group) in rtests:
        if group:
            groups.append("{0} {1}".format(len(group), rtest_statuses(code)))
    fobj.write("<li> {0} </li>\n".format(", ".join(groups)))
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
    rtestd = details[S_TESTD]
    status = rtest_statuses(details[S_STAT])
    prev_stat = details.get(S_PSTAT)
    if prev_stat is not None:
        status += " (previously: {0})".format(rtest_statuses(prev_stat))
    keywords = "  ".join(details[S_KWS])
    tcompletion = "{0:.2f}s".format(details[S_TIME])

    # look for post processing link
    html_link = os.path.join(rtestd, rtest + E_POST, F_POST)
    if not os.path.isfile(html_link): html_link = None

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


def _rebaseline_tests(d=None, files=None):
    from os.path import isfile, join, splitext, realpath, basename, exists
    if d is None:
        d = os.getcwd()
    log_message("finding tests to rebaseline")

    # find tests to rebaseline
    if files:
        tests = [splitext(realpath(f))[0] for f in files]
    else:
        tests = [splitext(realpath(f))[0] for f in os.listdir(d)
                 if f.endswith(".xml")]

    for test in tests:
        print test
        runid = basename(test)
        base = realpath(runid + ".base_exo")
        exo = realpath(runid + ".exo")
        if not exists(base):
            continue
        if not exists(exo):
            continue
        log_message("{0}: rebaselining".format(runid))
        shutil.copyfile(exo, base)



if __name__ == "__main__":
    sys.exit(main())
