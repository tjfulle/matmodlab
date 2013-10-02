import os
import sys
import shutil
from utils.if2py import f2py
D = os.path.dirname(os.path.realpath(__file__))
def makeut(destd, fc):
    """Make fortran based utils

    """
    utils = [("mmlabpack", [os.path.join(D, "mmlabpack.f90"),
                            os.path.join(D, "dgpadm.f"),]),]
    stats = 0
    for (name, source_files) in utils:
        stat = f2py(name, source_files, None, fc, None)
        if stat == 0:
            shutil.move(name + ".so", os.path.join(destd, name + ".so"))
        stats += stat
    return stats


if __name__ == "__main__":
    for d in ("/usr/local/bin", "/usr/bin", "/opt/local/bin"): 
        gfortran = os.path.join(d, "gfortran")
        if os.path.isfile(gfortran): break
    else:
        sys.exit("gfortran not found")
    makeut(os.path.join(D, "../lib"), gfortran)
