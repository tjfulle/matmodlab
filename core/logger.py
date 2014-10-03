import os
import sys
import string
from core.runtime import opts
from utils.misc import who_is_calling


std_transform = string.upper

class Logger(object):
    def __init__(self, logfile=None, verbosity=None, no_fh=0, chatty=0, mode="w"):
        self.ch = sys.stdout
        self.eh = sys.stderr
        self.no_fh = no_fh
        self.chatty = chatty
        self._fh = open(os.devnull, "a")
        self.verbosity = verbosity or opts.verbosity
        self.assign_logfile(logfile, mode=mode)
        self.errors = 0

    @property
    def verbosity(self):
        return self._v

    @verbosity.setter
    def verbosity(self, v):
        if v is None:
            v = 1
        self._v = v
        if not self._v:
            self.ch = open(os.devnull, "a")

    @property
    def logfile(self):
        return self._logfile

    def assign_logfile(self, filepath, mode="w"):
        self._logfile = filepath
        if filepath is not None:
            self.fh = open(filepath, mode=mode)

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
              transform=None, report_who=False, who=None):
        transform = transform or std_transform
        # look for paths in the message and do not transform therm
        if report_who:
            who = who_is_calling()
        if who:
            beg = "{0}{1}: ".format(beg, who)
        message = message.rstrip()
        message = "{0}{1}{2}".format(beg, transform_str(message, transform), end)

        # always write to file
        self.fh.write(message)

        if opts.parent_process_running and not self.chatty:
            return

        if write_to_console and self.verbosity:
            # write to console
            if log_to_eh:
                self.ch.flush()
                self.eh.write(message)
            else:
                self.ch.write(message)

    def warn(self, message, limit=False, warnings=[0], report_who=None):
        who = None if not report_who else who_is_calling()
        message = "*** WARNING: {0}".format(message)
        if limit and warnings[0] > opts.Wlimit:
            return
        self.write(message, log_to_eh=1, who=who)
        warnings[0] += 1

    def raise_error(self, message, raise_error=1, report_caller=1, caller=None,
                    **kwargs):
        if caller is None:
            caller = who_is_calling()
        beg = kwargs.get("beg", "*** ERROR: ")
        if report_caller:
            conmsg = "{2}{0} ({1})\n"
        else:
            conmsg = "{2}{0}\n"
        transform = kwargs.pop("transform", std_transform)
        conmsg = conmsg.format(transform_str(message, transform), caller, beg)
        self.write(conmsg, log_to_eh=1, transform=str, **kwargs)
        if raise_error > 0:
            if opts.raise_e:
                raise Exception(conmsg)
            else:
                sys.exit(1)

    def error(self, message, raise_error=0, **kwargs):
        self.errors += 1
        caller = who_is_calling()
        self.raise_error(message.rstrip(), raise_error=raise_error,
                         caller=caller, **kwargs)

    def debug(self, message):
        self.write(message, write_to_console=0)

    def finish(self):
        self.eh.flush()
        self.ch.flush()
        self.fh.flush()
        self.fh.close()

    def close(self):
        self.finish()


def transform_str(s, transform):
    return " ".join(x if os.path.sep in x else transform(x) for x in s.split(" "))



ConsoleLogger = Logger(no_fh=1)
