import os
import sys
import numpy as np
import xml.dom.minidom as xdom

from core.io import Error1
from utils.impmod import load_file
from utils.namespace import Namespace

D = os.path.dirname(os.path.realpath(__file__))
DRIVER_DB = None


class Driver(object):
    name = None
    paths = None
    surfaces = None
    kappa = 0
    def __init__(self):
        self._elem_variables = []
        self._glob_variables = []
        self.ndata = 0
        self._data = np.zeros(self.ndata)
        self.nglobdata = 0
        self._glob_data = np.zeros(self.nglobdata)
        self._paths_and_surfaces_processed = False

    def register_variable(self, var, vtype="SCALAR", units=None):
        """Register material variable

        """
        vtype = vtype.upper()
        name = var.upper()
        if vtype == "SCALAR":
            var = [name]

        elif vtype == "TENS":
            var = ["{0}_{1}".format(name, x) for x in ("XX", "XY", "XZ",
                                                       "YX", "YY", "YZ",
                                                       "ZX", "ZY", "ZZ")]
        elif vtype == "SYMTENS":
            var = ["{0}_{1}".format(name, x)
                   for x in ("XX", "YY", "ZZ", "XY", "YZ", "XZ")]

        elif vtype == "SKEWTENS":
            var = ["{0}_{1}".format(name, x) for x in ("XY", "YZ", "XZ")]

        elif vtype == "VECTOR":
            var = ["{0}_{1}".format(name, x) for x in ("X", "Y", "Z")]

        else:
            raise Error1("{0}: unrecognized vtype".format(vtype))

        start = self.ndata
        self.ndata += len(var)
        end = self.ndata
        self._elem_variables.extend(var)
        setattr(self, "{0}_slice".format(name.lower()), slice(start, end))

    def register_glob_variable(self, var):
        """Register global variable

        All global variables are scalars (so far)

        """
        name = var.upper()
        var = [name]
        start = self.nglobdata
        self.nglobdata += len(var)
        end = self.nglobdata
        self._glob_variables.extend(var)
        setattr(self, "{0}_slice".format(name.lower()), slice(start, end))

    def elem_vars(self):
        return self._elem_variables

    def elem_var_vals(self, name=None):
        """Return the current material data

        Returns
        -------
        data : array_like
            Material data

        """
        return self._data[self.getslice(name)]

    def glob_vars(self):
        return self._glob_variables

    def glob_var_vals(self, name=None):
        """Return the current material data

        Returns
        -------
        data : array_like
            Material data

        """
        if name is None:
            _slice = slice(0, self.nglobdata)
        else:
            _slice = self.getslice(name)
        return self._glob_data[_slice]

    def getslice(self, name=None):
        if name is None:
            return slice(0, self.ndata)
        return getattr(self, "{0}_slice".format(name.lower()))

    def allocd(self):
        """Allocate space for material data

        Notes
        -----
        This must be called after each consititutive model's setup method so
        that the number of xtra variables is known.

        """
        # Model data array.  See comments above.
        self._data = np.zeros(self.ndata)
        self._glob_data = np.zeros(self.nglobdata)

    def setvars(self, **kwargs):
        for kw, arg in kwargs.items():
            self._data[self.getslice(kw)] = arg

    def setglobvars(self, **kwargs):
        for (kw, arg) in kwargs.items():
            self._glob_data[self.getslice(kw)] = arg

    def extract_paths(self, exofilepath, paths):
        pass


# --- Driver database access functions
def isdriver(drivername):
    driver = getdriver(drivername)
    if driver is None:
        return False
    return True


def getdriver(drivername):
    if DRIVER_DB is None:
        read_driver_db()
    return DRIVER_DB.get(" ".join(drivername.split()).lower())


def create_driver(drivername):
    """Create a material object from the material name

    """
    driver = getdriver(drivername)
    if driver is None:
        return None

    # Instantiate the material object
    driver_mod = load_file(driver.filepath)
    driver_cls = getattr(driver_mod, driver.mtlcls)
    return driver_cls


def read_driver_db():
    """Read the materials.db database file

    """
    global DRIVER_DB
    filepath = os.path.join(D, "drivers.db")

    DRIVER_DB = {}
    doc = xdom.parse(filepath)
    materials = doc.getElementsByTagName("Drivers")[0]

    for material in materials.getElementsByTagName("Driver"):
        ns = Namespace()
        dtype = " ".join(
            str(material.attributes.getNamedItem("type").value).lower().split())
        filepath = str(material.attributes.getNamedItem("filepath").value)
        filepath = os.path.realpath(os.path.join(D, filepath))
        if not os.path.isfile(filepath):
            raise Error1("{0}: no such file".format(filepath))
        ns.filepath = filepath
        ns.mtlcls = str(material.attributes.getNamedItem("class").value)
        DRIVER_DB[dtype] = ns

    return DRIVER_DB
