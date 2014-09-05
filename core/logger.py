import os
import sys
import string
from utils.misc import who_is_calling
from core.runtime import opts


upper = string.upper

class Logger(object):
    def __init__(self, logfile=None, verbosity=1, no_fh=0, ignore_opts=0):
        self.ch = sys.stdout
        self.eh = sys.stderr
        self.no_fh = no_fh
        self._fh = open(os.devnull, "a")
        self.verbosity = verbosity
        self.logfile = logfile
        self.ignore_opts = ignore_opts

    @property
    def verbosity(self):
        return self._v

    @verbosity.setter
    def verbosity(self, v):
        self._v = v
        if not self._v:
            self.ch = open(os.devnull, "a")

    @property
    def logfile(self):
        return self._logfile

    @logfile.setter
    def logfile(self, filepath):
        self._logfile = filepath
        if filepath is not None:
            self.fh = open(filepath, "w")

    @property
    def fh(self):
        return self._fh

    @fh.setter
    def fh(self, filehandler):
        if self.no_fh:
            raise Exception("attempting to add file handler to console logger")
        try:
            filehandler.write
        except AttributeError:
            raise TypeError("attempting to assign non file handler "
                            "to logger filehandler")
        self._fh = filehandler

    def log(self, message, beg="", end="\n"):
        self.write(message, beg=beg, end=end)

    def write(self, message, beg="", end="\n", log_to_eh=0, write_to_console=1,
              transform=None):
        if transform is None: transform = upper
        message = "{0}{1}{2}".format(beg, transform(message), end)
        v = opts.verbosity if not self.ignore_opts else 1
        if write_to_console and v:
            # write to console
            if log_to_eh:
                self.ch.flush()
                self.eh.write(message)
            else:
                self.ch.write(message)
        # always write to file
        self.fh.write(message)

    def warn(self, message, limit=False, warnings=[0]):
        message = "*** WARNING: {0}".format(message)
        if limit and warnings[0] > opts.Wlimit:
            return
        self.write(message, log_to_eh=1)
        warnings[0] += 1

    def raise_error(self, message, raise_error=1, report_caller=1, caller=None):
        if caller is None:
            caller = who_is_calling()
        if report_caller:
            conmsg = "*** ERROR: {0} ({1})\n"
        else:
            conmsg = "*** ERROR: {0}\n"
        conmsg = conmsg.format(upper(message), caller)
        self.write(conmsg, log_to_eh=1, transform=str)
        if raise_error > 0:
            raise Exception(message)
        elif raise_error == 0:
            return
        sys.exit(1)

    def error(self, message, raise_error=0, report_caller=0):
        caller = None
        if report_caller:
            caller = who_is_calling()
        self.raise_error(message, raise_error=raise_error, caller=caller)

    def debug(self, message):
        self.write(message, write_to_console=0)

    def finish(self):
        self.eh.flush()
        self.ch.flush()
        self.fh.flush()
        self.fh.close()
        self.fh = open(os.devnull, "w")


ConsoleLogger = Logger(no_fh=1)
