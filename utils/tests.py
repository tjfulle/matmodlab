import os
import re
import sys
import time
import shutil
import subprocess
import xml.dom.minidom as xdom

from utils.errors import Error1
from utils.io import logmes, logwrn


D = os.path.dirname(os.path.realpath(__file__))
R = os.path.realpath(os.path.join(D, "../"))
TESTS = os.path.join(R, "tests")
PATH = os.getenv("PATH", "").split(os.pathsep)
PLATFORM = sys.platform.lower()
PATH.append(os.path.join(R, "tpl/exowrap/Build_{0}/bin".format(PLATFORM)))

PASSSTATUS = 0
DIFFSTATUS = 1
FAILSTATUS = 2


def rtest_statuses(status):
    return {PASSSTATUS: "PASS", DIFFSTATUS: "DIFF", FAILSTATUS: "FAIL"}.get(
        status, "FAIL")

def find_rtests(search_dirs, include, exclude):
    """Find all regression tests in search_dirs

    """
    # get a list of all test files (files with .rxml extension)
    test_files = []
    for d in search_dirs:
        for (dirname, dirs, files) in os.walk(d):
            test_files.extend([os.path.join(dirname, f) for f in files
                               if f.endswith(".rxml")])

    # put all found tests in the rtests dictionary
    rtests = {}
    for test_file in test_files:
        test_file_d = os.path.dirname(test_file)

        doc = xdom.parse(test_file)
        try:
            rtest = doc.getElementsByTagName("rtest")[0]
        except IndexError:
            raise Error1("Expected root element rtest in {0}".format(test_file))

        # --- name
        name = rtest.attributes.get("name")
        if name is None:
            raise Error1("{0}: rtest: name attribute required".format(
                os.path.basename(test_file)))
        name = str(name.value.strip())

        # --- keywords
        keywords = rtest.getElementsByTagName("keywords")
        if keywords is None:
            raise Error1("{0}: rtest: keyword element required".format(name))
        keywords = child2list(keywords, "lower")

        # --- link_files
        link_files = rtest.getElementsByTagName("link_files")
        if link_files is None:
            raise Error1("{0}: rtest: link_files element "
                         "required".format(name))
        link_files = [os.path.join(test_file_d, f).format(NAME=name)
                      for f in child2list(link_files)]
        for link_file in link_files:
            if not os.path.isfile(link_file):
                raise Error1("{0}: no such file".format(link_file))

        # --- execute
        execute = []
        exct = rtest.getElementsByTagName("execute")
        if exct is None:
            raise Error1("{0}: rtest: execute element "
                         "required".format(name))
        for item in exct:
            exe = item.attributes.get("name")
            if exe is None:
                raise Error1("{0}: execute: name attribute "
                             "required".format(name))
            exe = exe.value.strip()
            x = which(exe)
            if x is None:
                raise Error1("{0}: {1}: executable not found".format(name, exe))
            opts = [s.format(NAME=name) for s in child2list([item])]
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
        if include and not any(kw in keywords for kw in include):
            skip.append(key)
    for key in list(set(skip)):
        del rtests[key]
    return rtests


def run_rtests(rtests):
    """Run all of the rtests

    """
    testd = os.path.join(os.getcwd(), "TestResults.{0}".format(PLATFORM))
    if not os.path.isdir(testd):
        os.makedirs(testd)

    statuses = [run_rtest(testd, rtest, details)
                for (rtest, details) in rtests.items()]

    return max(statuses)


def run_rtest(testd, rtest, details):
    """Run the rtest

    """
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


def list_all_rtests(dirs):
    rtests = find_rtests(dirs, [], [])
    for (rtest, details) in rtests.items():
        print rtest


def which(exe):
    for d in PATH:
        x = os.path.join(d, exe)
        if os.path.isfile(x):
            return x
    return


def str2list(string, dtype=str):
    string = re.sub(r"[, ]", " ", string)
    return [dtype(x) for x in string.split()]


def child2list(item_list, action=None):
    child_list = []
    for item in item_list:
        child = item.firstChild.data.split("\n")
        for data in child:
            child_list.extend([str(s.strip()) for s in data.split()])
    if action == "lower":
        child_list = [s.lower() for s in child_list]
    return child_list


def main(dirs):
    rtests = find_rtests(dirs, ["fast"], [])
    status = run_rtests(rtests)
    if status != 0:
        print "A test did not pass :("
    else:
        print "All tests passed!"


if __name__ == "__main__":
    main([TESTS])
