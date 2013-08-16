import os
import sys
import xml.dom as xmldom
import xml.dom.minidom as xdom

from utils.errors import Error1
from utils.impmod import load_file
from utils.namespace import Namespace

D = os.path.dirname(os.path.realpath(__file__))
MTLDB_FILE = os.path.join(D, "materials.db")


def get_material_from_db(matname, mtldb=[None]):
    matname = matname.lower()
    if mtldb[0] is None:
        mtldb[0] = read_mtldb()
    if mtldb[0] is None:
        return None
    return mtldb[0].get(matname)


def create_material(matname):
    """Create a material object from the material name

    """
    # Instantiate the material object
    model = get_material_from_db(matname)
    if model is None:
        return None
    mtlmod = load_file(model.filepath)
    mtlcls = getattr(mtlmod, model.mtlcls)
    return mtlcls()


def read_mtldb():
    """Read the materials.db database file

    """
    if not os.path.isfile(MTLDB_FILE):
        return None

    mtldb = {}
    doc = xdom.parse(MTLDB_FILE)
    materials = doc.getElementsByTagName("Materials")[0]

    for mtl in materials.getElementsByTagName("Material"):
        ns = Namespace()
        name = str(mtl.attributes.getNamedItem("name").value).lower()
        filepath = str(mtl.attributes.getNamedItem("filepath").value)
        filepath = os.path.realpath(os.path.join(D, filepath))
        if not os.path.isfile(filepath):
            raise Error1("{0}: no such file".format(filepath))
        ns.filepath = filepath
        ns.mtlcls = str(mtl.attributes.getNamedItem("mclass").value)
        params = mtl.attributes.getNamedItem("parameters").value.split(",")
        ns.mtlparams = dict([(str(param).strip().lower(), i)
                             for (i, param) in enumerate(params)])
        ns.nparam = len(ns.mtlparams)
        ns.driver = str(mtl.attributes.getNamedItem("driver").value)
        mtldb[name] = ns
    return mtldb


def write_mtldb(mtldict):
    """Write the materials.db database file

    """
    mtldb = read_mtldb()
    if mtldb is None:
        mtldb = {}
    mtldb.update(mtldict)

    doc = xdom.Document()
    root = doc.createElement("Materials")
    doc.appendChild(root)

    for (name, ns) in mtldb.items():
        # create element
        child = doc.createElement("Material")
        child.setAttribute("name", name)
        for (aname, aval) in ns.items():
            child.setAttribute(aname, aval)
        root.appendChild(child)
    doc.writexml(open(MTLDB_FILE, "w"), indent="  ", addindent="  ", newl="\n")
    doc.unlink()
