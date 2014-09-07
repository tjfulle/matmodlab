import os
import imp
import sys
from select import select
import shutil
import inspect


def whoami():
    """ return name of calling function """
    return inspect.stack()[1][3]


def who_is_calling():
    """return the name of the calling function"""
    stack = inspect.stack()[2]
    return "{0}.{1}".format(
        os.path.splitext(os.path.basename(stack[1]))[0], stack[3])



def timed_raw_input(message, timeout=10, default=None):
    """A timed raw_input alternative

    from stackoverflow.com/questions/3471461/raw-input-and-timeout

    """
    sys.stdout.write(message)
    sys.stdout.flush()
    rlist, _, _ = select([sys.stdin], [], [], timeout)
    if rlist:
        return sys.stdin.readline()
    else:
        return default


def load_file(filepath, disp=0):
    """Load a python module by filepath

    Parameters
    ----------
    filepath : str
        path to python module

    Returns
    -------
    py_mod : module
        import python module

    """
    if not os.path.isfile(filepath):
        raise IOError("{0}: no such file".format(filepath))
    path, fname = os.path.split(filepath)
    module = os.path.splitext(fname)[0]
    fp, pathname, description = imp.find_module(module, [path])
    if module in sys.modules:
        del sys.modules[module]
    try:
        loaded = imp.load_module(module, fp, pathname, description)
    finally:
        # Since we may exit via an exception, close fp explicitly.
        if fp:
            fp.close()
    return loaded


def int2str(I, c=False, sep="-"):
    """Convert integer to string name

    Adapted from:
    http://stackoverflow.com/questions/8982163/
    how-do-i-tell-python-to-convert-integers-into-words

    Examples
    --------
    >>> print int2str(5)
    five

    """
    units = ["","one","two","three","four","five","six","seven","eight","nine"]
    teens = ["","eleven","twelve","thirteen","fourteen","fifteen","sixteen",
             "seventeen","eighteen","nineteen"]
    tens = ["","ten","twenty","thirty","forty","fifty","sixty","seventy",
            "eighty","ninety"]

    words = []

    if I == 0:
        words.append("zero")

    else:
        istr = "%d" % I
        istr_len = len(istr)
        groups = (istr_len + 2) / 2
        istr = istr.zfill(groups * 2)
        for i in range(0, groups * 2, 2):
            t, u = int(istr[i]), int(istr[i+1])
            g = groups - (i / 2 + 1)
            if t > 1:
                words.append(tens[t])
                if u >= 1:
                    words.append(units[u])
            elif t == 1:
                if u >= 1:
                    words.append(teens[u])
                else:
                    words.append(tens[t])
            else:
                if u >= 1:
                    words.append(units[u])

    words = sep.join(words)
    if c:
        words = words.capitalize()
    return words


def remove(path):
    """Remove file or directory -- dangerous!

    """
    if not os.path.exists(path): return
    try: os.remove(path)
    except OSError: shutil.rmtree(path)
    return
