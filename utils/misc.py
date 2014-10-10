import os
import imp
import sys
import shutil
import inspect
import importlib
from select import select
from contextlib import contextmanager
from os.path import splitext, split, dirname, isfile, sep, basename, exists
from core.product import ROOT_D
from core.runtime import opts

def whoami():
    """ return name of calling function """
    return inspect.stack()[1][3]


def who_is_calling():
    """return the name of the calling function"""
    stack = inspect.stack()[2]
    return "{0}.{1}".format(
        splitext(basename(stack[1]))[0], stack[3])


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


def load_file(filename, disp=0, reload=False):
    """Load a python module by filename

    Parameters
    ----------
    filename : str
        path to python module

    Returns
    -------
    py_mod : module
        import python module

    """
    if not isfile(filename):
        raise IOError("{0}: no such file".format(filename))

    path, base = split(filename)
    name = splitext(base)[0]
    if ROOT_D in path:
        d = "_".join(path.replace(ROOT_D+sep, "").split(sep))
        long_name = "{0}_{1}".format(d, name)
    else:
        long_name = name

    # if reload was requested, remove the module from sys.modules
    if reload:
        try:
            del sys.modules[long_name]
        except KeyError:
            pass

    # return already loaded module
    module = sys.modules.get(long_name)
    if module is None:
        fp, pathname, (_s, _m, ty) = imp.find_module(name, [path] + sys.path)
        if ty != imp.PY_SOURCE:
            # not Python source, can't do anything with this module
            fp.close()
            return None

        try:
            module = imp.load_module(long_name, fp, pathname, (_s, _m, ty))
        finally:
            if fp:
                fp.close()

    if disp:
        return module, long_name

    return module


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
    if not exists(path): return
    try: os.remove(path)
    except OSError: shutil.rmtree(path)
    return


def fillwithdots(a, b, width):
    dots = "." * (width - len(a) - len(b))
    return "{0}{1}{2}".format(a, dots, b)


def fileno(file_or_fd):
    fd = getattr(file_or_fd, 'fileno', lambda: file_or_fd)()
    if not isinstance(fd, int):
        raise ValueError("Expected a file (`.fileno()`) or a file descriptor")
    return fd

@contextmanager
def stdout_redirected(to=os.devnull, stdout=None):
    """From:  http://stackoverflow.com/questions/4675728/
                        redirect-stdout-to-a-file-in-python/22434262#22434262

    """
    if stdout is None:
       stdout = sys.stdout

    stdout_fd = fileno(stdout)
    # copy stdout_fd before it is overwritten
    #NOTE: `copied` is inheritable on Windows when duplicating a standard stream
    with os.fdopen(os.dup(stdout_fd), 'wb') as copied:
        stdout.flush()  # flush library buffers that dup2 knows nothing about
        try:
            os.dup2(fileno(to), stdout_fd)  # $ exec >&to
        except ValueError:  # filename
            with open(to, 'wb') as to_file:
                os.dup2(to_file.fileno(), stdout_fd)  # $ exec > to
        try:
            yield stdout # allow code to be run with the redirected stdout
        finally:
            # restore stdout to its previous value
            #NOTE: dup2 makes stdout_fd inheritable unconditionally
            stdout.flush()
            os.dup2(copied.fileno(), stdout_fd)  # $ exec >&copied


def merged_stderr_stdout():  # $ exec 2>&1
    return stdout_redirected(to=sys.stdout, stdout=sys.stderr)


def redirect_stdout_stderr():
    return stdout_redirected(stdout=sys.stderr)
