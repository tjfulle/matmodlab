import os
import sys
import xml.dom.minidom as xdom

from materials.product import F_MTL_PARAM_DB
from core.logger import ConsoleLogger as logger


def read_all_material_params_from_db(mod_name, dbfile=None):
    materials = {}
    if dbfile is None:
        dbfile = F_MTL_PARAM_DB
    doc = xdom.parse(dbfile)

    # check if material in database
    for material in doc.getElementsByTagName("Material"):
        mat_name = material.getAttribute("name")
        params = read_material_params_from_db(mat_name, mod_name, dbfile, error=0)
        if params is not None:
            materials[mat_name] = params
    return materials


def read_params_from_db(mat_name, mod_name, dbfile, error=1):
    """Read the material parameters from the specified database file

    """
    if not os.path.isfile(dbfile):
        logger.raise_error("{0}: material db file not found".format(dbfile))
        return

    doc = xdom.parse(dbfile)

    # check if material in database
    defined = []
    for material in doc.getElementsByTagName("Material"):
        defined.append(material.getAttribute("name"))
        if defined[-1] == mat_name:
            mat_name = defined[-1]
            break
    else:
        args = (mat_name, ", ".join(defined))
        logger.raise_error("{0}: material not defined in database, defined "
                           "materials are:\n  {1}".format(*args))
        return

    # check if material defines parameters for requested model
    tag = "MaterialModels"
    el = material.getElementsByTagName(tag)
    if not el:
        missing_tag(tag)
        return
    for model in el[0].getElementsByTagName("Model"):
        if mod_name in [model.getAttribute(s) for s in ("name", "short_name")]:
            mod_name = model.getAttribute("name")
            # model now is the correct Model element
            break
    else:
        if error:
            logger.raise_error("material {0} does not define parameters "
                               "for model {1}".format(mat_name, mod_name))
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
    args = (tag, os.path.basename(filepath))
    logger.raise_error("{0}: expected {0} tag".format(*args))
