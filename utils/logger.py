import os
import sys
from utils.misc import who_is_calling
from matmodlab import SPLASH


class Logger(object):
    def __init__(self, runid=None, verbosity=1, d=None):
        self.ch = sys.stdout
        self.eh = sys.stderr
        self.fh = None
        self.set_verbosity(verbosity)
        self.write_intro()

        # add file handler
        if runid is not None:
            d = d or os.getcwd()
            filepath = os.path.join(d, runid + ".log")
            self.add_file_handler(filepath)

    def write(self, string, beg="", end="\n", q=0):
        if q: return
        string = "{0}{1}{2}".format(beg, string, end).upper()
        if self.ch:
            self.ch.write(string)
            self.ch.flush()
        if self.fh:
            self.fh.write(string)

    def warn(self, string):
        string = "*** WARNING: {0}".format(string)
        self.write(string)

    def error(self, string, r=1, c=1, caller=None):
        if caller is None:
            caller = who_is_calling()
        if c: message = "*** ERROR: {0}: {1}\n"
        else: message = "*** ERROR: {1}\n"
        message = message.format(caller, string).upper()
        self.eh.write(message)
        if self.fh: self.fh.write(message)
        if r:
            raise Exception(message)
        elif r < 0:
            return
        else:
            sys.exit(1)

    def debug(self, string):
        if self.fh:
            self.fh.write(string.upper() + "\n")

    def write_formatted(self, *args, **kwargs):
        fmt = kwargs["fmt"]
        message = fmt.format(*args)
        self.write(message)

    def add_file_handler(self, filename):
        if self.fh:
            self.close_file_handler()
        self.fh = open(filename, "w")
        self.fh.write(SPLASH)

    def set_verbosity(self, v):
        if not v:
            self.ch = None

    def write_intro(self):
        self.write(SPLASH)

    def finish(self):
        self.eh.flush()
        if self.ch:
            self.ch.flush()
        if self.fh:
            self.fh.flush()
            self.fh.close()
            self.fh = None
