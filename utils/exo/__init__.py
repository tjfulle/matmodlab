import os

import exowrite as w
import exoread as r

__all__ = ["ExodusIIFile", ]

from exoinc import ExodusIIFileError

class ExodusIIFile(object):
    def __new__(cls, runid, mode="r", d=None):
        if mode not in "rw":
            raise ExodusIIFileError("{0}: bad read/write mode".format(mode))
        if mode == "w":
            return w.ExodusIIWriter(runid, d=d)
        filepath = runid
        if not os.path.isfile(filepath):
            filepath = runid + ".exo"
        return r.ExodusIIReader(filepath)
