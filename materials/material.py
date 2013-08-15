import os
import sys
import xml.dom as xmldom
import xml.dom.minidom as xdom

from utils.errors import Error1
from utils.impmod import load_file
from utils.namespace import Namespace

D = os.path.dirname(os.path.realpath(__file__))


def get_material_from_db(matname, mtldb=[None]):
    matname = matname.lower()
    if mtldb[0] is None:
        mtldb[0] = read_mtldb()
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
    filepath = os.path.join(D, "materials.db")

    mtldb = {}
    doc = xdom.parse(filepath)
    materials = doc.getElementsByTagName("Materials")[0]

    for mtl in materials.getElementsByTagName("Material"):
        ns = Namespace()
        name = str(mtl.attributes.getNamedItem("name").value).lower()
        filepath = str(mtl.attributes.getNamedItem("filepath").value)
        filepath = os.path.realpath(os.path.join(D, filepath))
        if not os.path.isfile(filepath):
            raise Error1("{0}: no such file".format(filepath))
        ns.filepath = filepath
        ns.mtlcls = str(mtl.attributes.getNamedItem("class").value)
        params = mtl.attributes.getNamedItem("parameters").value.split(",")
        ns.mtlparams = dict([(str(param).strip().lower(), i)
                             for (i, param) in enumerate(params)])
        ns.nparam = len(ns.mtlparams)
        ns.driver = str(mtl.attributes.getNamedItem("driver").value)
        mtldb[name] = ns

    return mtldb
