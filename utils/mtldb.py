import os
import sys
import xml.dom.minidom as xdom

from materials.info import F_MTL_PARAM_DB


def log_error(m):
    sys.stderr.write("*** error: {0}\n".format(m))


def read_all_material_params_from_db(mdlname, dbfile=None):
    materials = {}
    if dbfile is None:
        dbfile = F_MTL_PARAM_DB
    doc = xdom.parse(dbfile)

    # check if material in database
    for material in doc.getElementsByTagName("Material"):
        mtlname = material.getAttribute("name")
        params = read_material_params_from_db(mtlname, mdlname, dbfile, error=0)
        if params is not None:
            materials[mtlname] = params
    return materials


def read_material_params_from_db(matname, mdlname, dbfile, error=1):
    """Read the material parameters from the specified database file

    """
    if not os.path.isfile(dbfile):
        log_error("{0}: material db file not found".format(dbfile))
        return

    doc = xdom.parse(dbfile)

    # check if material in database
    defined = []
    for material in doc.getElementsByTagName("Material"):
        defined.append(material.getAttribute("name"))
        if defined[-1] == matname:
            matname = defined[-1]
            break
    else:
        log_error("{0}: material not defined in database, defined "
                  "materials are:\n  {1}".format(matname, ", ".join(defined)))
        return

    # check if material defines parameters for requested model
    tag = "MaterialModels"
    el = material.getElementsByTagName(tag)
    if not el:
        missing_tag(tag)
        return
    for model in el[0].getElementsByTagName("Model"):
        if mdlname in [model.getAttribute(s) for s in ("name", "short_name")]:
            mdlname = model.getAttribute("name")
            # model now is the correct Model element
            break
    else:
        if error:
            log_error("material {0} does not define parameters "
                      "for model {1}".format(matname, mdlname))
        return

    # get the material properties
    props = {}
    tag = "MaterialProperties"
    el = material.getElementsByTagName(tag)
    if not el:
        missing_tag(tag, dbfile)
        return
    for prop in el[0].getElementsByTagName("Property"):
        name = prop.getAttribute("name")
        val = prop.getAttribute("value")
        if not val:
            # old way
            val = prop.firstChild.data
        props[name] = float(val)
        continue

    # get the mapping from the material parameter to property
    params = {}
    tag = "Parameters"
    el = model.getElementsByTagName(tag)
    if not el:
        missing_tag(tag)
        return
    for (name, value) in el[0].attributes.items():
        params[name] = props[value]
        continue

    return params


def missing_tag(tag, filepath):
    log_error("{0}: expected {0} tag".format(tag, os.path.basename(filepath)))
