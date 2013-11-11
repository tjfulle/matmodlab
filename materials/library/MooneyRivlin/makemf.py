"""Build the DSF python shared object library

"""
import os
import sys
import shutil

from utils.if2py import f2py


D = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, D)
import mnrv as mnrv

builds = ("mnrv",)

def makemf(destd, fc, fio, materials=None, *args):
    name = builds[0]
    if materials and name not in [m.lower() for m in materials]:
        return [], [], [name]

    signature = os.path.join(D, "mnrv.pyf")

    # source files
    source_files = ["mnrv.f90",]
    source_files = [os.path.join(D, f) for f in source_files]
    source_files.append(fio)
    assert all(os.path.isfile(f) for f in source_files)
    stat = f2py(name, source_files, signature, fc, D)
    if stat != 0:
        return [], [name], []

    material = mnrv.MooneyRivlin()

    name = material.name
    filepath = mnrv.__file__.rstrip("c")
    mclass = material.__class__.__name__
    parameters = ", ".join(material.parameters(names=True)).lower()

    if destd != D:
        shutil.move(name + ".so", os.path.join(destd, name + ".so"))

    return (name, filepath, mclass, parameters), [], []

if __name__ == "__main__":
    makemf()
