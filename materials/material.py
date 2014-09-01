import os
import numpy as np
from project import UMATS, PKG_D
from utils.misc import load_file


D = os.path.dirname(os.path.realpath(__file__))
LIB = os.path.join(D, "Library")
MATS = [os.path.join(LIB, d) for d in os.listdir(LIB)
        if os.path.isdir(os.path.join(LIB, d))]
ABA = ("umat", "uhyper", "uanisohyper_inv")


def Material(model, parameters=None, constants=None, depvar=None):
    """Material model factory method

    """
    m = model.lower()
    if m not in ABA and parameters is None:
        raise ValueError("{0}: required parameters not given".format(model))

    for d in MATS + UMATS:
        try:
            meta = load_file(os.path.join(d, "mmlmat.py"))
        except OSError:
            continue
        if meta.NAME.lower() == m:
            break
    else:
        raise ValueError("{0}: model not found".format(model))

    # Check that the material is built
    if hasattr(meta, "SOURCE_FILES"):
        so_lib = os.path.join(PKG_D, meta.NAME + ".so")
        if not os.path.isfile(so_lib):
            raise ValueError("{0}: shared object library not found".format(model))

    # Instantiate the material
    interface = load_file(meta.INTERFACE)
    mat = getattr(interface, meta.CLASS)
    material = mat()
    material.setup_new_material(parameters)

    return material
