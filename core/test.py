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
    keywords = None
    runid = None
    exodiff = None
    pass_if_ran = 0

    @classmethod
    def validate(cls, module, logger):
        errors = 0
        if cls.runid is None:
            errors += 1
            line = "{0}: test missing runid class attribute".format(module)
            logger.error(line)
        if cls.keywords is None:
            errors += 1
            line = "{0}: test missing keywords class attribute".format(module)
            logger.error(line)
        elif not isinstance(cls.keywords, (list, tuple)):
            errors += 1
            line = "{0}: expected keywords to be a list".format(module)
            logger.error(line)
        elif all(n not in cls.keywords for n in ("long", "fast")):
            errors += 1
            line = "{0}: expected long or fast keyword".format(module)
            logger.error(line)

        return not errors

    def __init__(self):
        pass

    def setup(self, *args, **kwargs):
        pass

    def run(self, d, logger):
        """The standard test

        """
        from matmodlab import TEST_D
        d = d or os.getcwd()

        def check_file_existence(f):
            if os.path.isfile(f):
                return 0
            logger.error("{0}: file not found".format(f))
            return 1

        # Hijack console output
        confile = os.path.join(d, self.runid + ".con")
        #sys.stdout = sys.stderr = open(confile, "w")
        try:
            self.run_job(d=d)
            self.stat = PASSED
        except:
            self.stat = FAILED_TO_RUN
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__

        if self.stat == FAILED_TO_RUN:
            return self.stat

        if self.pass_if_ran:
            return self.stat

        # compare output to base exodus file
        errors = 0
        exofile = os.path.join(d, self.runid + ".exo")
        errors += check_file_existence(exofile)
        exobase = os.path.join(d, self.runid + ".base_exo")
        errors += check_file_existence(exobase)

        if self.exodiff:
            control_file = self.exodiff
        control_file = os.path.join(TEST_D, "base.exdiff")
        errors += check_file_existence(control_file)

        if errors:
            self.stat = FAILED
            logger.error("STOPPING TEST DUE TO PREVIOUS ERRORS")
            return self.stat

        exlfile = os.path.join(d, self.runid + ".exodiff.log")
        self.stat = exodiff.exodiff(exofile, exobase, f=exlfile, v=0,
                                    control_file=control_file)
        return self.stat

    def tear_down(self, module, d):
        if self.stat != PASSED:
            return
        for f in os.listdir(d):
            if module in f or self.runid in f:
                if f.endswith((".log", ".exo", ".pyc", ".con", ".eval")):
                    remove(os.path.join(d,f))

def remove(f):
    if os.path.isdir(f):
        rm = shutil.rmtree
    else:
        rm = os.remove
    try: rm(f)
    except OSError: pass
