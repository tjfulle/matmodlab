"""All built in materials register there interface and source files here

This file is used only during the setup phase to build and install materials.

"""
import os
from matmodlab import MATLIB

BUILTIN = {}

# --- Ideal Gase
d = os.path.join(MATLIB, "idealgas")
BUILTIN["idealgas"]= {"interface": os.path.join(d, "idealgas.py"),
                      "class": "IdealGas"}

# --- Mooney Rivlin
d = os.path.join(MATLIB, "MooneyRivlin")
source_files = [os.path.join(d, f) for f in ("mnrv.f90", "mnrv.pyf")]
BUILTIN["mnrv"] = {"source_files": source_files, "source_directory": d,
                   "interface": os.path.join(d, "mnrv.py"),
                   "lapack": "lite", "class": "MooneyRivlin"}

# --- Plastic (non functional at the moment)
d = os.path.join(MATLIB, "plastic")
source_files = [os.path.join(d, f) for f in
                ("plastic_interface.f90", "plastic.f90", "plastic.pyf")]
BUILTIN["plastic"] = {"source_files": source_files, "source_directory": d,
                      "interface": os.path.join(d, "plastic.py"),
                      "class": "Plastic"}

# --- Pure python elastic model
d = os.path.join(MATLIB, "pyelastic")
BUILTIN["pyelastic"] = {"interface": os.path.join(d, "pyelastic.py"),
                        "class": "PyElastic"}

# --- Elastic
d = os.path.join(MATLIB, "elastic")
source_files = [os.path.join(d, f) for f in
                ("elastic_interface.f90", "elastic.f90", "elastic.pyf")]
BUILTIN["elastic"] = {"source_files": source_files, "source_directory": d,
                      "interface": os.path.join(d, "elastic.py"),
                      "class": "Elastic"}

# --- Pure python plastic model
d = os.path.join(MATLIB, "vonmises")
BUILTIN["vonmises"] = {"interface": os.path.join(d, "vonmises.py"),
                       "class": "VonMises"}

# --- Pure python plastic model
d = os.path.join(MATLIB, "pyplastic")
BUILTIN["pyplastic"] = {"interface": os.path.join(d, "pyplastic.py"),
                        "class": "Pyplastic"}

# --- Pure python transversely isotropic model
d = os.path.join(MATLIB, "transisoelas")
BUILTIN["transisoelas"] = {"interface": os.path.join(d, "transisoelas.py"),
                           "class": "TransIsoElas"}
