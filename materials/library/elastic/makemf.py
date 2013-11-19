"""Build the DSF python shared object library

"""
import os
import sys
import shutil

from utils.if2py import f2py


D = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, D)
import elastic as elastic

builds = ("elastic",)

def makemf(destd, fc, fio, materials=None, *args):
    name = builds[0]
    if materials and name not in [m.lower() for m in materials]:
        return [], [], [name]

    signature = os.path.join(D, "elastic.pyf")

    # source files
    source_files = ["elastic_interface.f90", "elastic.f90"]
    source_files = [os.path.join(D, f) for f in source_files]
    source_files.append(fio)
    assert all(os.path.isfile(f) for f in source_files)
    stat = f2py(name, source_files, signature, fc, destd=destd, incd=D)
    if stat != 0:
        return [], [name], []

    material = elastic.Elastic()

    name = material.name
    filepath = elastic.__file__.rstrip("c")
    mclass = material.__class__.__name__
    parameters = ", ".join(material.parameters(names=True)).lower()

    return (name, filepath, mclass, parameters), [], []

if __name__ == "__main__":
    makemf()
