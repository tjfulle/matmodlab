import os
import sys
import string
import logging

from core.runtime import opts
from utils.misc import who_is_calling
from core.product import SPLASH

_LIST_ = -123
_SPLASHED_ = -124
CONSOLE = "console"

def loggers():
    return Logger(_LIST_)

def Logger(logger=None, _cache={}, **kwargs):

    if logger is None:
        logger = CONSOLE

    if logger == _LIST_:
        return _cache.keys()

    remove = kwargs.pop("remove_from_cache", None)
    if remove:
        # remove logger
        del _cache[logger]
        return

    # when running tests, permutation, optimization, etc. a parent process
    # will run many child processes (simulations) and we don't want every
    # child to log to the console. Parent processes can send in the kwarg
    # parent_process which will suppress logging to the console (by setting
    # verbosity=-1) for all loggers set up after it. Logging to file is
    # unaffected.
    parent_process = kwargs.pop("parent_process", None)
    if parent_process is not None:
        _cache["parent_process"] = 1
    elif _cache.get("parent_process"):
        kwargs["verbosity"] = -1

    try:
        instance = _cache[logger]
    except KeyError:
        instance = _Logger(logger, **kwargs)
        _cache[logger] = instance
    return instance

class _Logger(object):
    _splashed = [False]
    def __init__(self, name, filename=1, verbosity=1, mode="w", splash=True):

        self.logger_id = name
        self.errors = 0

        # set the logging level
        fhlev = logging.DEBUG
        chlev = {0: logging.CRITICAL,
                 1: logging.INFO,
                 2: logging.DEBUG}.get(abs(verbosity), logging.NOTSET)

        # basic logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(chlev)

        # console handler
        if verbosity < 0:
            # verbosity less than 0 flags a parent process, we still want to
            # log, so we send it to a 'console' file
            ch = logging.FileHandler(name + ".con", mode="w")
        else:
            ch = logging.StreamHandler()
        ch.setLevel(chlev)
        self.logger.addHandler(ch)

        # file handler.  by default, we add a file handler
        if filename == 1:
            filename = name + ".log"

        if filename:
            fh = logging.FileHandler(filename, mode="w")
            fh.setLevel(fhlev)
            self.logger.addHandler(fh)

        for handler in self.logger.handlers:
            # do our best to only splash to screen once
            if "stderr" in str(handler.stream) and not splash:
                continue
            elif "stderr" in str(handler.stream) and self._splashed[0]:
                continue
            handler.stream.write(SPLASH + "\n")
            self._splashed[0] = True

    def info(self, message, beg="", end=None, report_who=False, who=None):
        if report_who:
            who = who_is_calling()
        if who:
            beg = "{0}{1}: ".format(beg, who)
        message = message.rstrip()
        message = "{0}{1}".format(beg, message)
        if end is not None:
            message = "{0}{1}".format(message, end)
        c = True if end is not None else False
        continued = {"continued": c}
        self.logger.info(message, extra=continued)

    def warn(self, message, limit=False, warnings=[0], report_who=None, who=None):
        if report_who:
            who = who_is_calling()
        if who:
            message = "{0}: {1} ".format(who, message)
        message = "*** warning: {0}".format(message)
        if limit and warnings[0] > opts.Wlimit:
            return
        continued = {"continued": False}
        self.logger.warn(message, extra=continued)
        warnings[0] += 1

    def write(self, *args, **kwargs):
        self.info(*args, **kwargs)

    def exception(self, message, caller=None):
        if caller is None:
            caller = who_is_calling()
        self.raise_error(message, caller=caller)

    def raise_error(self, message, caller=None):
        if caller is None:
            caller = who_is_calling()
        self.error(message, caller=caller)
        if opts.raise_e:
            raise Exception(message)
        else:
            sys.exit(1)

    def error(self, message, caller=None):
        self.errors += 1
        if caller is None:
            caller = who_is_calling()
        message = "*** error: {0}: {1}".format(caller, message)
        continued = {"continued": False}
        self.logger.error(message.rstrip(), extra=continued)

    def debug(self, message):
        continued = {"continued": False}
        self.logger.debug(message, extra=continued)

    def finish(self):
        Logger(self.logger_id, remove_from_cache=1)

    def close(self):
        self.finish()


def emit(self, record):
    """Monkey-patch the logging StreamHandler emit function. Allows omiting
    trailing newline when not wanted"""
    msg = self.format(record)
    fs = "%s" if getattr(record, "continued", False) else "%s\n"
    self.stream.write(fs % msg)
    self.flush()

logging.StreamHandler.emit = emit
ConsoleLogger = Logger(CONSOLE, filename=None, splash=False)
