import os
import sys
import xml.dom.minidom as xdom


def log_error(m):
    sys.stderr.write("*** error: {0}\n".format(m))


def read_material_params_from_db(matname, mdlname, dbfile):
    """Read the material parameters from the specified database file

    """
    if not os.path.isfile(dbfile):
        log_error("{0}: material db file not found".format(dbfile))
        return

    doc = xdom.parse(dbfile)
    materials = doc.getElementsByTagName("Materials")[0]
    materials = materials.getElementsByTagName("Material")
    for material in materials:
        if material.getAttribute("name") == matname:
            break
    else:
        log_error("{0}: material not defined in database".format(matname))
        return

    for parameters in material.getElementsByTagName("Parameters"):
        if parameters.getAttribute("model") == mdlname:
            break
    else:
        log_error("material {0} does not define parameters "
                  "for model {1}".format(matname, mdlname))
        return

    params = {}
    for node in parameters.childNodes:
        if node.nodeType != node.ELEMENT_NODE:
            continue
        name = node.nodeName
        val = float(node.firstChild.data)
        params[name] = val
        continue
    return params
