import os
import sys
import xml.dom as xmldom
import xml.dom.minidom as xdom

from utils.io import Error1
from utils.impmod import load_file
from utils.namespace import Namespace

D = os.path.dirname(os.path.realpath(__file__))

def create_driver(drivername, db=[None]):
    """Create a material object from the material name

    """
    if db[0] is None:
        db[0] = read_db()
    driver = db[0].get(" ".join(drivername.split()).lower())
    if driver is None:
        return None

    # Instantiate the material object
    driver_mod = load_file(driver.filepath)
    driver_cls = getattr(driver_mod, driver.mtlcls)
    return driver_cls()

def read_db():
    """Read the materials.db database file

    """
    filepath = os.path.join(D, "drivers.db")

    db = {}
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
        db[dtype] = ns

    return db
