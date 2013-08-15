import os
import imp
def load_file(filepath):
    if not os.path.isfile(filepath):
        raise OSError("{0}: no such file".format(filepath))

    fdir, fname = os.path.split(filepath)
    py_mod = os.path.splitext(fname)[0]
    py_path = [fdir]

    fp, pathname, description = imp.find_module(py_mod, py_path)
    try:
        return imp.load_module(py_mod, fp, pathname, description)
    finally:
        # Since we may exit via an exception, close fp explicitly.
        if fp:
            fp.close()
