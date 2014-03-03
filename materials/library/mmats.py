"""All built in materials register there interface and source files here

This file is used only during the setup phase to build and install materials.

"""
import os
from materials.material import _Material

D = os.path.dirname(os.path.realpath(__file__))

d = os.path.join(D, "idealgas")
material = {"source_files": None,
            "interface_file": os.path.join(d, "idealgas.py"),
            "class_name": "IdealGas"}
idealgas = _Material("idealgas", **material)

d = os.path.join(D, "MooneyRivlin")
source_files = [os.path.join(d, f) for f in ("mnrv.f90", "mnrv.pyf")]
material = {"source_files": source_files,
            "interface_file": os.path.join(d, "mnrv.py"),
            "requires_lapack": True, "class_name": "MooneyRivlin"}
mnrv = _Material("mnrv", **material)

d = os.path.join(D, "plastic")
source_files = [os.path.join(d, f) for f in
                ("plastic_interface.f90", "plastic.f90", "plastic.pyf")]
material = {"source_files": source_files,
            "interface_file": os.path.join(d, "plastic.py"),
            "class_name": "Plastic"}
plastic = _Material("plastic", **material)

# Elastic
d = os.path.join(D, "elastic")
source_files = [os.path.join(d, f) for f in
                ("elastic_interface.f90", "elastic.f90", "elastic.pyf")]
material = {"source_files": source_files,
            "interface_file": os.path.join(d, "elastic.py"),
            "class_name": "Elastic"}
elastic = _Material("elastic", **material)

d = os.path.join(D, "pyelastic")
material = {"source_files": None,
            "interface_file": os.path.join(d, "pyelastic.py"),
            "class_name": "PyElastic"}
pyelastic = _Material("pyelastic", **material)


NAMES = {"idealgas": idealgas, "mnrv": mnrv, "plastic": plastic,
         "elastic": elastic, "pyelastic": pyelastic}


def conf(name=None):
    """Return the material configurations for building

    Parameters
    ----------
    name : name of material configuration to return

    Returns
    -------
    conf : dict

    """
    if name is None:
        print "{0}: name argument required".format("mmats.conf")
        return

    try:
        return NAMES[name.lower()]
    except ValueError:
        print "{0}: unknown material".format(name)
        return
