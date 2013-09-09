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

    # check if material in database
    materials = doc.getElementsByTagName("Material")
    for material in materials:
        if material.getAttribute("name") == matname:
            break
    else:
        log_error("{0}: material not defined in database".format(matname))
        return

    # check if material defines parameters for requested model
    models = material.getElementsByTagName("MaterialModels")[0]
    for model in models.getElementsByTagName("Model"):
        if model.getAttribute("name") == mdlname:
            break
    else:
        log_error("material {0} does not define parameters "
                  "for model {1}".format(matname, mdlname))
        return

    # get the material properties
    props = {}
    mtlprops = material.getElementsByTagName("MaterialProperties")[0]
    for prop in mtlprops.getElementsByTagName("Property"):
        name = prop.getAttribute("name")
        val = float(prop.firstChild.data)
        props[name] = val
        continue

    # get the mapping from the material parameter to property
    params = {}
    mdlparams = model.getElementsByTagName("Parameters")[0]
    for i in range(mdlparams.attributes.length):
        p = mdlparams.attributes.item(i)
        params[p.name] = props[p.value]
        continue

    return params
