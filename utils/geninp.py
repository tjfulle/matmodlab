import os
import re
import sys
import xml.dom.minidom as xdom
from xml.parsers.expat import ExpatError


def create_gmd_input(runid, driver, pathtype, pathopts, path, mtlmdl, mtlparams,
                     write=0):
    xmlf = runid + ".xml"
    doc = xdom.Document()
    root = doc.createElement("GMDSpec")
    xphys = doc.createElement("Physics")
    xphys.setAttribute("driver", driver)

    # --- path
    xpath = doc.createElement("Path")
    xpath.setAttribute("type", pathtype)
    for (optname, optval) in pathopts:
        xpath.setAttribute(optname, str(optval))
    for item in path:
        p = doc.createTextNode(item)
        xpath.appendChild(p)

    # --- material
    xmaterial = doc.createElement("Material")
    xmaterial.setAttribute("model", mtlmdl)
    for (param, value) in mtlparams:
        xparam = doc.createElement(param.upper())
        xval = doc.createTextNode(str(value))
        xparam.appendChild(xval)
        xmaterial.appendChild(xparam)

    # --- append everything
    xphys.appendChild(xmaterial)
    xphys.appendChild(xpath)
    root.appendChild(xphys)
    doc.appendChild(root)

    xmlstr = doc.toprettyxml()
    if not write:
        return xmlstr

    if write == 1:
        sys.stdout.write(xmlstr)

    elif write == 2:
        sys.stderr.write(xmlstr)

    else:
        doc.writexml(open(xmlf, "w"), addindent="  ", newl="\n")

    return xmlstr


if __name__ == "__main__":

    runid = "test"
    driver = "solid"
    pathtype = "prdef"
    pathopts = (("nfac", 100), ("amplitude", 2.))
    path = ["0 1 222 0 0 0", "1 1 222 1 0 0", "2 1 222 0 0 0"]
    mtlmdl = "elastic"
    mtlparams = (("K", 13.e9), ("G", 5.4e9))

    inp = create_gmd_input(runid, driver, pathtype, pathopts, path,
                           mtlmdl, mtlparams, write=0)
