"""matmodlab configuration reader/editor.

matmodlib uses a configuration file ($MATMODLABRC or ~/.matmodlabrc) to set
user definable configurations. Currently supported configurable variables are:

    materials
    tests

The format of the file is:

[variable]
value_1
value_2
...

which is similar to - but not compatible with - standard unix configuraton
files.

"""
import os
import re
import sys
import argparse
from string import upper
from core.product import RCFILE

def write(message):
    sys.stdout.write(message + "\n")


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    p = argparse.ArgumentParser(prog="mml config",
                                description="%(prog)s: Set matmodlab options")
    p.add_argument("--add", nargs=2, metavar=("name", "value"),
        help="name and value of option to add to configuration")
    p.add_argument("--del", dest="delete", nargs="+", metavar=("name", "value"),
        help="name and value of option to remove from configuration")
    p.add_argument("--switch", action="store_true", default=False,
        help="switch from old MMLMTLS environment variable to new config file")
    p.add_argument("--cat", action="store_true", default=False,
        help="print the MATMODLABRC file to the console and exit")
    args = p.parse_args(argv)
    if args.cat:
        sys.stdout.write("{0}:\n\n{1}\n".format(RCFILE, open(RCFILE).read()))
        return 0

    if args.switch and any([args.add, args.delete]):
        p.error("--switch and [--add, --delete] are mutually exclusive")

    if args.switch:
        cfgswitch_and_warn()
        return

    if not args.add and not args.delete:
        p.error("nothing to do")

    cfgedit(RCFILE, add=args.add, delete=args.delete)

    return 0


class Options:
    def __init__(self, **kwargs):
        for (k, v) in kwargs.items():
            setattr(self, k, v)

def mybool(s):
    return s.lower() not in ("0", "no", "none", "false")

def myint(n):
    return int(float(n))

def cfgparse(option=None, default=None, _cache=[0]):
    if not _cache[0]:
        # 'a' is a dict of options, each option is a list of strings, each
        # string is a line from the config file.
        a = _cfgparse(RCFILE)

        # parse list-of-string options (default empty list)
        for i in ("materials", "tests"):
            a[i] = a.pop(i, [])

        # model switching
        switch = []
        for pair in a.pop("switch", []):
            try:
                old, new = [x.strip() for x in pair.split(":", 1)]
            except ValueError:
                raise ValueError("switch option must be specified as a:b")
            switch.append(":".join([old, new]))
        a["switch"] = switch

        errmsg = "rcfile option '{0}' requires type {1}, got {2}"
        # parse integer options
        for i in ("verbosity", "nprocs"):
            x = a.pop(i, None)
            if x is None:
                continue
            if len(x) > 1:
                raise TypeError(errmsg.format(i, "int", "list"))
            if len(x) < 1:
                raise TypeError(errmsg.format(i, "int", "nothing"))
            a[i] = myint(x[0])

        # parse boolean options
        for i in ("sqa", "debug"):
            x = a.pop(i, None)
            if x is None:
                continue
            if len(x) > 1:
                raise TypeError(errmsg.format(i, "bool", "list"))
            if len(x) < 1:
                raise TypeError(errmsg.format(i, "bool", "nothing"))
            a[i] = mybool(x[0])

        # parse string options
        for i in ("warn",):
            x = a.pop(i, None)
            if x is None:
                continue
            if len(x) > 1:
                raise TypeError(errmsg.format(i, "string", "list"))
            if len(x) < 1:
                raise TypeError(errmsg.format(i, "string", "nothing"))
            a[i] = x[0]

        _cache[0] = Options(**a)

    config = _cache[0]
    if option is not None:
        return getattr(config, option, default)

    return config


def _cfgparse(filename):
    # Set up variables and functions for removing comments
    comment_char = "#"
    rmcomments= lambda x: x[:x.find(comment_char)] if comment_char in x else x

    cfg = {}
    try:
        # read in the file and strip comments at the same time
        lines = [rmcomments(_) for _ in open(filename).readlines()]
    except IOError:
        return {}


    regex = re.compile( r'\[(.*?)\]')

    # store values from each option in a list. stack[0] will be the name of
    # the option and stack[1:] the values
    stack = []
    for line in lines:
        tag = regex.search(line)
        if tag:
            if stack:
                # write the stack to the current configuration and reset it
                cfg.setdefault(stack[0], []).extend(stack[1:])
            stack = [tag.group(1).strip()]
            continue
        if not line.split():
            continue

        # append option to stack
        stack.append(line.strip())

    # if there is a leftover stack, add it on
    if stack:
        cfg.setdefault(stack[0], []).extend(stack[1:])

    return cfg


def cfgedit(filename, add=None, delete=None):

    a = _cfgparse(RCFILE)

    if add is not None:
        write("WRITING THE FOLLOWING OPTIONS TO {0}:".format(filename))
        k, v = add
        k = k.lower()
        # dumb check to see if it is a file path
        if os.path.sep in v:
            v = os.path.realpath(os.path.expanduser(v))
        write("  {0}: {1}".format(k, v))
        a.setdefault(k, []).append(v.strip())

    if delete is not None:
        write("DELETING THE FOLLOWING OPTIONS FROM {0}:".format(filename))
        try:
            k, v = delete
        except ValueError:
            k = delete[0]
            v = None
        k = k.strip().lower()

        write("  {0}: {1}".format(k, v))
        av = a.pop(k, None)
        if not av:
            write("*** WARNING: {0}: OPTION HAD NOT BEEN SET".format(k, v))
        elif v:
            a[k] = [x for x in av if x != v]
        # if v is None, the entire option is deleted

    lines = []
    for (k, v) in a.items():
        # make sure each line is unique
        v = list(set(v))
        if not v: continue
        lines.append('[{0}]'.format(k.strip().lower()))
        lines.append("\n".join(str(s) for s in v))

    with open(filename, "w") as fh:
        fh.write("\n".join(lines))

    return


def cfgswitch_and_warn():
    """Switch configuration style and warn user

    """
    mtls_d = os.getenv("MMLMTLS")
    if not mtls_d:
        return
    write("=" * 82)
    write("**** WARNING " * 6 + "****")
    write("**** WARNING " * 6 + "****")
    write("""
   LEGACY ENVIRONMENT VARIABLE 'MMLMTLS' CONVERTED TO NEW STYLE CONFIGURATION
   IN ~/.matmodlabrc.  THE LOCATION OF THIS FILE IS ALSO CONFIGURABLE BY THE
   MATMODLABRC ENVIRONMENT VARIABLE.  REMOVE MMLMTLS FROM YOUR ENVIRONMENT
   TO AVOID THIS WARNING IN THE FUTURE
""")
    write("**** WARNING " * 6 + "****")
    write("**** WARNING " * 6 + "****")
    write("=" * 82)
    if mtls_d:
        cfgedit(RCFILE, add=("materials", mtls_d))

if __name__ == "__main__":
    main()
