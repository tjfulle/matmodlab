import os
import sys
import shutil
from core.logger import ConsoleLogger
from utils.exojac import exodiff
from core.product import TEST_CONS_WIDTH
from utils.misc import fillwithdots, remove

PASSED = 0
DIFFED = 1
FAILED = 2
FAILED_TO_RUN = -2
NOT_RUN = -5
DIFFTOL = 1.E-06
FAILTOL = 1.E-04

NOATTR = -31


class TestError(Exception):
    pass


class TestBase(object):
    _is_mml_test = True
    passed = PASSED
    diffed = DIFFED
    failed = FAILED
    failed_to_run = FAILED_TO_RUN
    not_run = NOT_RUN

    def init_and_check(self, root_dir, test_file_dir, module, str_repr, **opts):
        """Initialize the test.

        This is done in the parent class after any subclasses have been
        initialized

        """
        self.test_file_dir = test_file_dir
        self.module = module
        self.initialized = True
        self.torn_down = 0
        self.status = self.not_run
        self.disabled = getattr(self, "disabled", False)
        self.gen_overlay = getattr(self, "gen_overlay", False)
        self.gen_overlay_if_fail = getattr(self, "gen_overlay_if_fail", False)
        self._no_teardown = False
        self.str_repr = str_repr
        self.interp = getattr(self, "interpolate_diff", False)
        self.validated = False
        self.force_overlay = opts.get("overlay", False)

        self.exofile = None
        self.setup_by_class = False
        self.status = self.not_run

        # check requirements
        errors = []
        self.runid = getattr(self, "runid", "UNKNOWN")
        if self.runid == "UNKOWN":
            errors.append("{0}: missing runid attribute".format(self.str_repr))

        self.keywords = getattr(self, "keywords", [])
        if not self.keywords:
            errors.append("{0}: missing keywords attribute".format(self.str_repr))

        elif not isinstance(self.keywords, (list, tuple)):
            errors.append("{0}: expected keywords to be a "
                          "list".format(self.str_repr))

        if all(n not in self.keywords for n in ("long", "fast", "medium")):
            errors.append("{0}: expected long, fast, or medium "
                          "keyword".format(self.str_repr))

        self.validated = not errors

        d = os.path.basename(self.test_file_dir)
        sub_dir = getattr(self, "test_dir_base", d)
        self.test_dir = os.path.join(root_dir, sub_dir, self.runid)

        if errors:
            raise TestError("\n"+"\n".join("   {0}".format(x) for x in errors))

    def make_test_dir(self):
        """create test directory to run tests"""
        if os.path.isdir(self.test_dir):
            remove(self.test_dir)
        os.makedirs(self.test_dir)
        if getattr(self, "base_res", NOATTR) != NOATTR:
            dest = os.path.join(self.test_dir, os.path.basename(self.base_res))
            os.symlink(self.base_res, dest)

    def setup(self, *args, **kwargs):
        """The standard setup

        """
        from core.product import TEST_D

        # Look for standard files
        errors = []
        self.exofile = os.path.join(self.test_dir, self.runid + ".exo")

        if getattr(self, "base_res", NOATTR) == NOATTR:
            self.base_res = os.path.join(self.test_file_dir,
                                         self.runid + ".base_exo")
        if not os.path.isfile(self.base_res):
            errors.append("{0}: base_res file not found".format(self.str_repr))

        if getattr(self, "exodiff", NOATTR) == NOATTR:
            f = "base.exodiff"
            if os.path.isfile(os.path.join(self.test_file_dir, f)):
                self.exodiff = os.path.join(self.test_file_dir, f)
            else:
                self.exodiff = os.path.join(TEST_D, f)
        if not os.path.isfile(self.exodiff):
            errors.append("{0}: exodiff file not found".format(self.str_repr))

        self.setup_by_class = True
        self.status = self.failed_to_run

        if errors:
            raise TestError("\n"+"\n".join("   {0}".format(x) for x in errors))

        self.make_test_dir()
        return 0

    def pre_hook(self, *args, **kwargs):
        pass

    def run(self):
        """The standard test

        """
        self.status = self.failed_to_run

        if not getattr(self, "setup_by_class", False):
            raise TestError("{0}: running standard test requires "
                            "calling super's setup method".format(self.str_repr))

        try:
            self.run_job()
        except BaseException as e:
            raise TestError("{0}: failed with the following "
                            "exception: {1}".format(self.str_repr, e.args[0]))

        if not os.path.isfile(self.exofile):
            raise TestError("{0}: file not found".format(self.exofile))

        exodiff_log = os.path.join(self.test_dir, self.runid + ".exodiff.log")
        self.status = exodiff.exodiff(self.exofile, self.base_res, f=exodiff_log,
                                      v=0, control_file=self.exodiff,
                                      interp=self.interp)
        return

    def tear_down(self, force=0):

        if force:
            remove(self.test_dir)
            self.torn_down = 1
            return

        if self._no_teardown:
            return

        if self.status != self.passed:
            return

        remove(self.test_dir)
        self.torn_down = 1

    def post_hook(self, *args, **kwargs):
        if self.force_overlay:
            self._create_overlays()
        elif self.gen_overlay:
            self._create_overlays()
        elif self.gen_overlay_if_fail and self.status==self.failed:
            self._create_overlays()
        pass

    def _create_overlays(self):
        """Create overlays of variables common to sim file and baseline

        """
        if not all([self.exofile, self.base_res]):
            ConsoleLogger.warn("overlays only created for tests setup by class")
            return

        import time
        import matplotlib.pyplot as plt
        from utils.exojac.exodump import load_data

        # we created plots for a reason -> so don't tear down results!
        self._no_teardown = True

        # Make output directory for plots
        destd = os.path.join(self.test_dir, "png")
        try: shutil.rmtree(destd)
        except OSError: pass
        os.makedirs(destd)

        # get the data
        head1, data1 = load_data(self.exofile)
        head2, data2 = load_data(self.base_res)

        # TIME is always first column
        time1, data1 = data1[:, 0], data1[:, 1:]
        time2, data2 = data2[:, 0], data2[:, 1:]
        head1 = head1[1:]
        head2 = dict([(v, i) for (i, v) in enumerate(head2[1:])])

        ti = time.time()

        label1 = self.runid
        label2 = self.runid + "_base"

        aspect_ratio = 4. / 3.
        plots = []
        msg = fillwithdots(self.str_repr, "POST PROCESSING", TEST_CONS_WIDTH)
        for (col, yvar) in enumerate(head1):
            name = yvar + ".png"
            filename = os.path.join(destd, name)
            y1 = data1[:, col]
            plt.clf()
            plt.cla()
            plt.xlabel("TIME")
            plt.ylabel(yvar)

            col2 = head2.get(yvar)
            if col2 is None:
                continue
            y2 = data2[:, col2]
            plt.plot(time2, y2, ls="-", lw=4, c="orange", label=label2)
            plt.plot(time1, y1, ls="-", lw=2, c="green", label=label1)
            plt.legend(loc="best")
            plt.gcf().set_size_inches(aspect_ratio * 5, 5.)
            plt.savefig(filename, dpi=100)
            plots.append(filename)
        dtime = time.time() - ti
        msg = fillwithdots(self.str_repr,
                           "POST PROCESSING COMPLETE",
                           TEST_CONS_WIDTH)
        msg +=  " ({0:.0f}s)".format(dtime)

        # write an html summary
        fh = open(os.path.join(destd, "graphics.html"), "w")
        fh.write("<html>\n<head>\n<title>{0}</title>\n</head>\n"
                   "<body>\n<table>\n<tr>\n".format(self.str_repr))
        for i, plot in enumerate(plots):
            name = os.path.basename(plot)
            if i % 3 == 0 and i != 0:
                fh.write("</tr>\n<tr>\n")
            width = str(int(aspect_ratio * 300))
            height = str(int(300))
            fh.write("<td>{0}<a href='{1}'><img src='{1}' width='{2}' "
                     "height='{3}'></a></td>\n".format(name, plot, width, height))
        fh.write("</tr>\n</table>\n</body>\n</html>")
        fh.close()

        return
