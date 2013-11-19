"""All built in materials register there interface and source files here

This file is used only during the setup phase to build and install materials.

"""
import os
D = os.path.dirname(os.path.realpath(__file__))

def conf(*args):
    """Return the material configurations for building

    Parameters
    ----------
    args : not used

    Returns
    -------
    conf : dict

    """
    materials = {}

    # Ideal Gas
    d = os.path.join(D, "idealgas")
    materials["idealgas"] = {"source_files": None, "include_dir": d,
                             "interface_file": os.path.join(d, "idealgas.py"),
                             "class": "IdealGas"}

    # Mooney Rivlin
    d = os.path.join(D, "MooneyRivlin")
    source_files = [os.path.join(d, f) for f in ("mnrv.f90", "mnrv.pyf")]
    materials["mnrv"] = {"source_files": source_files, "include_dir": d,
                         "interface_file": os.path.join(d, "mnrv.py"),
                         "class": "MooneyRivlin"}

    # Plastic
    d = os.path.join(D, "plastic")
    source_files = [os.path.join(d, f) for f in
                    ("plastic_interface.f90", "plastic.f90", "plastic.pyf")]
    materials["plastic"] = {"source_files": source_files, "include_dir": d,
                            "interface_file": os.path.join(d, "plastic.py"),
                            "class": "Plastic"}
    # Elastic
    d = os.path.join(D, "elastic")
    source_files = [os.path.join(d, f) for f in
                    ("elastic_interface.f90", "elastic.f90", "elastic.pyf")]
    materials["elastic"] = {"source_files": source_files, "include_dir": d,
                            "interface_file": os.path.join(d, "elastic.py"),
                            "class": "Elastic"}

    return materials
