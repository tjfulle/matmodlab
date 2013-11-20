import os
import sys
import xml.dom.minidom as xdom
from xml.parsers.expat import ExpatError

from __config__ import F_MTL_MODEL_DB
from core.io import Error1
from utils.impmod import load_file

D = os.path.dirname(os.path.realpath(__file__))


def get_material_from_db(matname, mtldb=[None]):
    matname = matname.lower()
    if mtldb[0] is None:
        mtldb[0] = read_mtldb()
    if mtldb[0] is None:
        return None
    material = mtldb[0].get(matname)
    if material is None:
        return None

    # instantiate the material to get param names
    mtlmod = load_file(material[0])
    mclass = getattr(mtlmod, material[1])
    params = mclass.param_names
    pdict = {}
    for (i, n) in enumerate(mclass.param_names):
        if n.startswith("-"):
            # not to be parsed
            n = n[1:]
            i = -1
        pdict[n.lower()] = i

    del mclass

    return material[0], material[1], pdict


def instantiate_material(matname):
    """Create a material object from the material name

    """
    # Instantiate the material object
    model = get_material_from_db(matname)
    if model is None:
        return
    mtli, mtlc = model
    mtlmod = load_file(mtli)
    mclass = getattr(mtlmod, mtlc)
    return mclass()


def create_material(matname, matparams, matopts):
    """Create a material object from the material name

    """
    # Instantiate the material object
    model = get_material_from_db(matname)
    if model is None:
        return
    mtli, mtlc, mtlp = model
    mtlmod = load_file(mtli)
    mclass = getattr(mtlmod, mtlc)
    material = mclass()
    material.setup_new_material(matparams)
    material.set_constant_jacobian()
    material.set_options(**matopts)
    return material


def read_mtldb():
    """Read the F_MTL_MODEL_DB database file

    """
    if not os.path.isfile(F_MTL_MODEL_DB):
        return

    try:
        doc = xdom.parse(F_MTL_MODEL_DB)
    except ExpatError:
        os.remove(F_MTL_MODEL_DB)
        return None

    mtldb = {}
    materials = doc.getElementsByTagName("Materials")[0]
    for mtl in materials.getElementsByTagName("Material"):
        name = str(mtl.attributes.getNamedItem("name").value).lower()
        filepath = str(mtl.attributes.getNamedItem("filepath").value)
        filepath = os.path.realpath(os.path.join(D, filepath))
        if not os.path.isfile(filepath):
            raise Error1("{0}: no such file".format(filepath))
        mclass = str(mtl.attributes.getNamedItem("mclass").value)
        mtldb[name] = (filepath, mclass)

    return mtldb


def write_mtldb(built_mtls, wipe=False):
    """Write the F_MTL_MODEL_DB database file

    Parameters
    ----------
    built_mtls : list
        list of (name, filepath, mclass, parameters) tuples of built
        materials, where name is the model name, filepath is the path to its
        interface file, mclass the name of the material class, and parameters
        is a list of ordered parameter names.

    """
    if wipe and os.path.isfile(F_MTL_MODEL_DB):
        os.remove(F_MTL_MODEL_DB)
    mtldb = read_mtldb()
    if mtldb is None:
        mtldb = {}

    for name, info in built_mtls.items():
        filepath = info["interface_file"]
        mclass = info["class"]
        mtldb[name] = (filepath, mclass)

    doc = xdom.Document()
    root = doc.createElement("Materials")
    doc.appendChild(root)

    for (name, (filepath, mclass)) in mtldb.items():
        # create element
        child = doc.createElement("Material")

        child.setAttribute("name", name)
        child.setAttribute("filepath", filepath)
        child.setAttribute("mclass", mclass)

        root.appendChild(child)

    doc.writexml(open(F_MTL_MODEL_DB, "w"), addindent="  ", newl="\n")
    doc.unlink()
