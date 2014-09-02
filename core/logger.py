import os
import sys
from utils.misc import who_is_calling
from project import SPLASH


class Logger(object):
    def __init__(self):
        self.ch = sys.stdout
        self.eh = sys.stderr
        self.fh = None

    def write(self, string, beg="", end="\n", q=0):
        if q: return
        string = "{0}{1}{2}".format(beg, string, end)
        if self.ch:
            self.ch.write(string)
            self.ch.flush()
        if self.fh:
            self.fh.write(string)

    def warn(self, string, beg="", end=""):
        string = "{0}*** WARNING: {1}: {2}".format(beg, string, end)
        self.write(string)

    def error(self, string, r=1, c=1):
        caller = who_is_calling()
        if c: message = "*** ERROR: {0}: {1}\n"
        else: message = "*** ERROR: {1}\n"
        message = message.format(caller, string)
        self.eh.write(message)
        if self.fh: self.fh.write(message)
        if r:
            raise SystemExit(message)
        else: sys.exit(1)

    def debug(self, string):
        if self.fh:
            self.fh.write(string + "\n")

    def write_formatted(self, *args, **kwargs):
        fmt = kwargs["fmt"]
        message = fmt.format(*args)
        self.write(message)

    def add_file_handler(self, filename):
        self.fh = open(filename, "w")

    def set_verbosity(self, v):
        if not v: self.ch = None

    def write_intro(self):
        self.write(SPLASH)
