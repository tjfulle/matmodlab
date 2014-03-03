"""All built in materials register there interface and source files here

This file is used only during the setup phase to build and install materials.

"""
import os
D = os.path.dirname(os.path.realpath(__file__))

NAMES = ("idealgas", "mnrv", "plastic", "elastic", "pyelastic", 'vonmises')

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
        nidx = [n.lower() for n in NAMES].index(name.lower())
    except ValueError:
        print "{0}: unknown material".format(name)
        return

    if nidx == 0:
        # Ideal Gas
        d = os.path.join(D, "idealgas")
        material = {"source_files": None, "include_dir": d,
                    "interface_file": os.path.join(d, "idealgas.py"),
                    "class": "IdealGas"}


    elif nidx == 1:
        # Mooney Rivlin
        d = os.path.join(D, "MooneyRivlin")
        source_files = [os.path.join(d, f) for f in ("mnrv.f90", "mnrv.pyf")]
        material = {"source_files": source_files, "include_dir": d,
                    "interface_file": os.path.join(d, "mnrv.py"),
                    "requires_lapack": True, "class": "MooneyRivlin"}

    elif nidx == 2:
        # Plastic
        d = os.path.join(D, "plastic")
        source_files = [os.path.join(d, f) for f in
                        ("plastic_interface.f90", "plastic.f90", "plastic.pyf")]
        material = {"source_files": source_files, "include_dir": d,
                    "interface_file": os.path.join(d, "plastic.py"),
                    "class": "Plastic"}
    elif nidx == 3:
        # Elastic
        d = os.path.join(D, "elastic")
        source_files = [os.path.join(d, f) for f in
                        ("elastic_interface.f90", "elastic.f90", "elastic.pyf")]
        material = {"source_files": source_files, "include_dir": d,
                    "interface_file": os.path.join(d, "elastic.py"),
                    "class": "Elastic"}
    elif nidx == 4:
        # python Elastic
        d = os.path.join(D, "pyelastic")
        material = {"source_files": None, "include_dir": d,
                    "interface_file": os.path.join(d, "pyelastic.py"),
                    "class": "PyElastic"}
    elif nidx == 5:
        # python von mises with combined hardening
        d = os.path.join(D, "vonmises")
        material = {"source_files": None, "include_dir": d,
                    "interface_file": os.path.join(d, "vonmises.py"),
                    "class": "VonMises"}

    return material
