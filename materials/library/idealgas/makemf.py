import os
import imp

D = os.path.dirname(os.path.realpath(__file__))

builds = ("idealgas",)

def makemf(destd, fc, fio, materials=None, *args):
    """Build fortran and python material models

    Parameters
    ----------
    destd : str
        Path to directory to copy built shared object libraries (if any)
    fc : str
        Path to fortran compiler
    fio : str
        Path to the fortran IO routines
    materials : None or list
        If None, build all.  Else, build only specified
    args : tuple
        Not used

    Returns
    -------
        built.append((name, filepath, mclass, parameters))
    built : list
        list of (name, filepath, mclass, parameters) tuples of built
        materials, where name is the model name, filepath is the path to its
        interface file, mclass the name of the material class, and parameters
        is a list of ordered parameter names.
    failed : list
        list of names of failed models
    skipped : list
        list of names of skipped models

    """
    name = builds[0]
    if materials and name not in [x.lower() for x in materials]:
        return [], [], [name]
    interface = os.path.join(D, name + ".py")
    py_mod = os.path.splitext(os.path.basename(interface))[0]
    module = imp.load_source(py_mod, interface)
    material = module.IdealGas()

    name = material.name
    filepath = module.__file__.rstrip("c")
    mclass = material.__class__.__name__
    parameters = ", ".join(material.params()).lower()

    return (name, filepath, mclass, parameters), [], []

if __name__ == "__main__":
    makemf()
