import os
import sys
import shutil
from utils.exojac import exodiff

PASSED = 0
DIFFED = 1
FAILED = 2
FAILED_TO_RUN = -2

class TestBase(object):
    _is_mml_test = True
    passed = PASSED
    diffed = DIFFED
    failed = FAILED
    failed_to_run = FAILED_TO_RUN

    def validate(self, src_dir, test_dir, module, logger):

        self.test_dir = test_dir
        self.src_dir = src_dir
        self.module = module
        self.src_dir

        errors = 0

        self.runid = getattr(self, "runid", None)
        if not self.runid:
            self.runid = "unkown_test"
            errors += 1
            logger.error("{0}: MISSING RUNID ATTRIBUTE".format(self.runid))

        self.keywords = getattr(self, "keywords", [])
        if not self.keywords:
            errors += 1
            logger.error("{0}: MISSING KEYWORDS ATTRIBUTE".format(self.runid))

        elif not isinstance(self.keywords, (list, tuple)):
            errors += 1
            logger.error("{0}: EXPECTED KEYWORDS TO BE A LIST".format(self.runid))

        if all(n not in self.keywords for n in ("long", "fast")):
            errors += 1
            logger.error("{0}: EXPECTED LONG OR FAST KEYWORD".format(self.runid))

        if not os.path.isdir(self.test_dir):
            errors += 1
            logger.error("{0}: {1} DIRECTORY DOES NOT "
                         "EXIST".format(self.runid, self.test_dir))

        return not errors

    def setup(self, *args, **kwargs):
        """The standard setup

        """
        from matmodlab import TEST_D
        logger = args[0]

        # Look for standard files
        errors = 0
        self.exofile = os.path.join(self.test_dir, self.runid + ".exo")

        self.base_exo = getattr(self, "base_exo",
            os.path.join(self.src_dir, self.runid + ".base_exo"))
        if not os.path.isfile(self.base_exo):
            errors += 1
            logger.error("{0}: base_exo FILE NOT FOUND".format(self.runid))

        self.exodiff = getattr(self, "exodiff",
                               os.path.join(TEST_D, "base.exodiff"))
        if not os.path.isfile(self.exodiff):
            errors += 1
            logger.error("{0}: exodiff FILE NOT FOUND".format(self.runid))

        self.setup_by_class = True
        self.stat = self.failed_to_run

        return errors

    def run(self, logger):
        """The standard test

        """
        self.stat = self.failed_to_run

        if not getattr(self, "setup_by_class", False):
            logger.error("{0}: RUNNING STANDARD TEST REQUIRES "
                         "CALLING SUPER'S SETUP METHOD".format(self.runid))
            return

        try:
            self.run_job()
        except BaseException as e:
            logger.error("{0}: FAILED WITH THE FOLLOWING "
                         "EXCEPTION: {1}".format(self.runid, e.message))
            return

        if not os.path.isfile(self.exofile):
            logger.error("{0}: FILE NOT FOUND".format(self.exofile))
            return

        exodiff_log = os.path.join(self.test_dir, self.runid + ".exodiff.log")
        self.stat = exodiff.exodiff(self.exofile, self.base_exo, f=exodiff_log,
                                    v=0, control_file=self.exodiff)
        return

    def tear_down(self):
        if self.stat != self.passed:
            return
        for f in os.listdir(self.test_dir):
            if self.module in f or self.runid in f:
                if f.endswith((".log", ".exo", ".pyc", ".con", ".eval")):
                    remove(os.path.join(self.test_dir, f))

def remove(f):
    if os.path.isdir(f):
        rm = shutil.rmtree
    else:
        rm = os.remove
    try: rm(f)
    except OSError: pass
