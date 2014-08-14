import os
import imp
import sys
from select import select
import shutil

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


def load_file(filepath):
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

