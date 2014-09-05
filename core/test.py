import os
import sys
import shutil
from core.logger import Logger
from utils.exojac import exodiff

PASSED = 0
DIFFED = 1
FAILED = 2
FAILED_TO_RUN = -2
NOT_RUN = -5

class TestBase(object):
    _is_mml_test = True
    passed = PASSED
    diffed = DIFFED
    failed = FAILED
    failed_to_run = FAILED_TO_RUN
    not_run = NOT_RUN

    @property
    def logger(self):
        return self._logger

    @logger.setter
    def logger(self, new_logger):
        try:
            new_logger.write
            new_logger.warn
            new_logger.error
        except AttributeError, TypeError:
            raise TypeError("attempting to assign a non logger "
                            "to the {0} Driver logger".format(self.kind))
        self._logger = new_logger

    def init(self, src_dir, test_dir, module, logger):
        """Initialize the test.

        This is done in the parent class after any subclasses have been
        initialized

        """
        self.src_dir = src_dir
        self.test_dir = test_dir
        self.module = module
        self.logger = logger
        self.initialized = True
        self.torn_down = 0
        self.status = self.not_run
        self.disabled = getattr(self, "disabled", False)

    def validate(self):

        initialized = getattr(self, "initialized", False)
        if not initialized:
            self.logger.raise_error("test suite must be initialized")

        errors = 0
        self.runid = getattr(self, "runid", None)
        if not self.runid:
            self.runid = "unkown_test"
            errors += 1
            self.logger.error("{0}: missing runid attribute".format(self.runid))

        self.keywords = getattr(self, "keywords", [])
        if not self.keywords:
            errors += 1
            self.logger.error("{0}: missing keywords attribute".format(self.runid))

        elif not isinstance(self.keywords, (list, tuple)):
            errors += 1
            self.logger.error("{0}: expected keywords to be a "
                              "list".format(self.runid))

        if all(n not in self.keywords for n in ("long", "fast")):
            errors += 1
            self.logger.error("{0}: expected long or fast "
                              "keyword".format(self.runid))

        if not os.path.isdir(self.test_dir):
            errors += 1
            self.logger.error("{0}: {1} directory does not "
                              "exist".format(self.runid, self.test_dir))

        return not errors

    def setup(self, *args, **kwargs):
        """The standard setup

        """
        from matmodlab import TEST_D

        # Look for standard files
        errors = 0
        self.exofile = os.path.join(self.test_dir, self.runid + ".exo")

        self.base_exo = getattr(self, "base_exo",
            os.path.join(self.src_dir, self.runid + ".base_exo"))
        if not os.path.isfile(self.base_exo):
            errors += 1
            self.logger.error("{0}: base_exo file not found".format(self.runid))

        self.exodiff = getattr(self, "exodiff",
                               os.path.join(TEST_D, "base.exodiff"))
        if not os.path.isfile(self.exodiff):
            errors += 1
            self.logger.error("{0}: exodiff file not found".format(self.runid))

        self.setup_by_class = True
        self.status = self.failed_to_run

        return errors

    def run(self):
        """The standard test

        """
        self.status = self.failed_to_run

        if not getattr(self, "setup_by_class", False):
            self.logger.error("{0}: running standard test requires "
                              "calling super's setup method".format(self.runid))
            return

        try:
            self.run_job()
        except BaseException as e:
            self.logger.error("{0}: failed with the following "
                              "exception: {1}".format(self.runid, e.message))
            return

        if not os.path.isfile(self.exofile):
            self.logger.error("{0}: file not found".format(self.exofile))
            return

        exodiff_log = os.path.join(self.test_dir, self.runid + ".exodiff.log")
        self.status = exodiff.exodiff(self.exofile, self.base_exo, f=exodiff_log,
                                      v=0, control_file=self.exodiff)
        return

    def tear_down(self):
        if self.status != self.passed:
            return
        for f in os.listdir(self.test_dir):
            if self.module in f or self.runid in f:
                if f.endswith((".log", ".exo", ".pyc", ".con", ".eval")):
                    remove(os.path.join(self.test_dir, f))
        self.torn_down = 1

    def post_hook(self, *args, **kwargs):
        pass

def remove(f):
    if os.path.isdir(f):
        rm = shutil.rmtree
    else:
        rm = os.remove
    try: rm(f)
    except OSError: pass
