import os
import sys
import shutil
import traceback
import numpy as np
from utils.exojac import exodiff
from core.logger import ConsoleLogger
from utils.misc import fillwithdots, remove
from core.product import TEST_CONS_WIDTH, TEST_D

PASSED = 0
DIFFED = 1
FAILED = 2
FAILED_TO_RUN = -2
NOT_RUN = -5
DIFFTOL = 5.E-08
FAILTOL = 1.E-07

NOATTR = -31


def reqa(it, attr):
    return "    {0}: required attribute '{1}' not defined".format(it, attr)


class TestError(Exception):
    pass


class MetaSuperInit(type):
    """metaclass which overrides the "__call__" function"""
    def __call__(cls, *args, **kwargs):
        """Called when you call Class() """
        obj = type.__call__(cls)
        obj.super_init(*args, **kwargs)
        return obj


class TestBase(object):
    __metaclass__ = MetaSuperInit
    passed = PASSED
    diffed = DIFFED
    failed = FAILED
    failed_to_run = FAILED_TO_RUN
    not_run = NOT_RUN

    def super_init(self, *args, **kwargs):
        """Initialize the test.

        This is done in the parent class after any subclasses have been
        initialized

        """
        root_dir = args[0]

        module = self.__module__
        self.name = '{0}.{1}'.format(module, self.__class__.__name__)
        self.file = os.path.realpath(sys.modules[module].__file__)
        self.d = os.path.dirname(self.file)
        speeds = ("fast", "medium", "long")
        speed = [x for x in speeds if x in self.keywords]
        if not speed:
            raise TestError("{0}: must define one of {1} "
                            "keyword".format(self.name, ", ".join(speeds)))
        elif len(speed) > 1:
            raise TestError("{0}: must define only one of {1} "
                            "keyword".format(self.name, ", ".join(speeds)))
        self.speed = speed[0]

        self.module = module
        self.torn_down = 0
        self.status = self.not_run
        self.disabled = getattr(self, "disabled", False)
        self.gen_overlay = getattr(self, "gen_overlay", False)
        self.gen_overlay_if_fail = getattr(self, "gen_overlay_if_fail", False)
        self._no_teardown = False
        self.interp = getattr(self, "interpolate_diff", False)
        self.force_overlay = kwargs.get("overlay", False)
        self.dtime = np.nan
        self.ready = False

        self.exofile = None
        self.status = self.not_run

        if all(n not in self.keywords for n in ("long", "fast", "medium")):
            raise TestError("{0}: expected long, fast, or medium "
                            "keyword".format(self.name))

        d = os.path.basename(self.d)
        sub_dir = getattr(self, "test_dir_base", d)
        self.test_dir = os.path.join(root_dir, sub_dir, self.name)


    @property
    def keywords(self):
        try:
            return self._keywords
        except AttributeError:
            raise TestError(reqa(self.name, "keywords"))

    @keywords.setter
    def keywords(self, keywords):
        self._keywords = keywords

    @property
    def runid(self):
        try:
            return self._runid
        except AttributeError:
            raise TestError(reqa(self.name, "runid"))

    @runid.setter
    def runid(self, runid):
        self._runid = runid

    @property
    def base_res(self):
        try:
            return self._base_res
        except AttributeError:
            return None

    @base_res.setter
    def base_res(self, base_res):
        if not os.path.isfile(base_res):
            raise TestError("{0}: base_res file not found".format(self.runid))
        self._base_res = base_res

    @property
    def exodiff(self):
        try:
            return self._exodiff
        except AttributeError:
            f = "base.exodiff"
            if os.path.isfile(os.path.join(self.d, f)):
                self.exodiff = os.path.join(self.d, f)
            else:
                self.exodiff = os.path.join(TEST_D, f)
        return self.exodiff

    @exodiff.setter
    def exodiff(self, exodiff):
        if not os.path.isfile(exodiff):
            raise TestError("{0}: exodiff file not found".format(self.name))
        self._exodiff = exodiff

    @property
    def dtime(self):
        return self._dtime
    @dtime.setter
    def dtime(self, value):
        self._dtime = value

    def make_test_dir(self):
        """create test directory to run tests"""
        if os.path.isdir(self.test_dir):
            remove(self.test_dir)
        os.makedirs(self.test_dir)
        if self.base_res:
            dest = os.path.join(self.test_dir, os.path.basename(self.base_res))
            os.symlink(self.base_res, dest)

    def setup(self, *args, **kwargs):
        """The standard setup

        """
        self.status = self.failed_to_run

        # standard files
        self.exofile = os.path.join(self.test_dir, self.runid + ".exo")
        if not self.base_res:
            self.base_res = os.path.join(self.d, self.runid + ".base_exo")
        self.make_test_dir()
        self.ready = True

    def pre_hook(self, *args, **kwargs):
        pass

    def run(self):
        """The standard test

        """
        if not self.ready:
            raise TestError("{0}: test must be setup first".format(self.name))

        try:
            self.run_job()
        except BaseException as e:
            tb = sys.exc_info()[2]
            tb_list = traceback.extract_tb(tb)
            s = " ".join("{0}".format(x) for x in e.args)
            tb_str = " ".join(traceback.format_list(tb_list)) + "\n" + s
            raise TestError(tb_str)

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
        msg = fillwithdots(self.name, "POST PROCESSING", TEST_CONS_WIDTH)
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
        msg = fillwithdots(self.name,
                           "POST PROCESSING COMPLETE",
                           TEST_CONS_WIDTH)
        msg +=  " ({0:.0f}s)".format(dtime)
        self.dtime = dtime

        # write an html summary
        fh = open(os.path.join(destd, "graphics.html"), "w")
        fh.write("<html>\n<head>\n<title>{0}</title>\n</head>\n"
                   "<body>\n<table>\n<tr>\n".format(self.name))
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
