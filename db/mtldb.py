import os
import xml.dom.minidom as dom

from utils.errors import Error1
from utils.namespace import Namespace


D = os.path.dirname(os.path.realpath(__file__))


def read_mtldb(filepath):
    mtldb = {}
    doc = dom.parse(filepath)
    materials = doc.getElementsByTagName("Materials")[0]
    for material in materials.getElementsByTagName("Material"):
        name = str(material.attributes.getNamedItem("name").value).lower()
        mtlid = int(material.attributes.getNamedItem("id").value)

        module = material.getElementsByTagName("Module")[0]
        filepath = str(module.attributes.getNamedItem("filepath").value)
        filepath = os.path.realpath(os.path.join(D, filepath))
        if not os.path.isfile(filepath):
            raise Error1("{0}: no such file".format(filepath))
        classname = str(module.attributes.getNamedItem("class").value)
        mtldb[name] = {"id": mtlid, "parameters": {}, "class": classname,
                       "filepath": filepath}
        for parameter in material.getElementsByTagName("Parameter"):
            pnam = str(parameter.attributes.getNamedItem("name").value).lower()
            pidx = int(parameter.attributes.getNamedItem("idx").value)
            mtldb[name]["parameters"].update({pnam: pidx})
    return mtldb


def mtlmodel(material, mtldb=[None]):
    if mtldb[0] is None:
        mtldb[0] = read_mtldb(os.path.join(D, "materials.db"))
    model = mtldb[0].get(material.lower())
    if model is None:
        return None
    ns = Namespace()
    ns.id = model["id"]
    ns.parameters = model["parameters"].keys()
    ns.name = material.lower()
    ns.paramidx = lambda x: (None if x.lower() not in model["parameters"]
                             else model["parameters"][x.lower()])
    ns.filepath = model["filepath"]
    ns.classname = model["class"]
    return ns
