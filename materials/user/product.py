import os
from utils.fortran.product import FIO
D = os.path.dirname(os.path.realpath(__file__))
# Auxiliary files
USER_PYF = os.path.join(D, "user.pyf")
USER_PY = os.path.join(D, "user.py")
def material_libraries():
    return {"user": {"source_files": [FIO, USER_PYF],
                     "interface": USER_PY, "class": "UserMat"}}
