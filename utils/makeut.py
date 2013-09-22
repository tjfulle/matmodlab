import os
import sys
import shutil
from utils.gmdf2py import f2py
D = os.path.dirname(os.path.realpath(__file__))
def makeut(destd, fc):
    """Make fortran based utils

    """
    utils = [("linalg", os.path.join(D, "linalg.f90")),]
    stats = 0
    for (name, filepath) in utils:
        stat = f2py(name, [filepath], None, fc, None)
        if stat == 0:
            shutil.move(name + ".so", os.path.join(destd, name + ".so"))
        stats += stat
    return stats
