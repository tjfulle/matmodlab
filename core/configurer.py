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

def Bool(it):
    return it.lower() not in ("0", "no", "none", "false")

def Int(it):
    return int(float(it))

def Float(it):
    return float(it)

def List(it):
    return [x.strip() for x in it if x.split()]

def ListOfFiles(it):
    list_of_files = []
    for f in it:
        if not os.path.exists(f):
            write("*** warning: {0}: not such file or directory".format(f))
        list_of_files.append(os.path.realpath(os.path.expanduser(f)))
    return list_of_files

options = {"materials": {"N": "+", "type": ListOfFiles, "default": []},
               "tests": {"N": "+", "type": ListOfFiles, "default": []},
              "switch": {"N": "+", "type":        List, "default": []},
                "warn": {"N": "?", "type":        Bool, "default": None},
                 "sqa": {"N": "?", "type":        Bool, "default": None},
               "debug": {"N": "?", "type":        Bool, "default": None},
           "verbosity": {"N": "?", "type":         Int, "default": None},
              "nprocs": {"N": "?", "type":         Int, "default": None}}


def write(message):
    sys.stdout.write(message + "\n")


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    p = argparse.ArgumentParser(prog="mml config",
                                description="%(prog)s: Set matmodlab options")
    p.add_argument("--add", nargs="+", metavar=("name", "value[s]"),
        help="name and value of option to add to configuration")
    p.add_argument("--del", dest="delete", nargs="+", metavar=("name", "value[s]"),
        help="name and value of option to remove from configuration")
    p.add_argument("--old2new", action="store_true", default=False,
        help="switch from old MMLMTLS environment variable to new config file")
    p.add_argument("--cat", action="store_true", default=False,
        help="print the MATMODLABRC file to the console and exit")
    args = p.parse_args(argv)
    if args.cat:
        sys.stdout.write("{0}:\n\n{1}\n".format(RCFILE, open(RCFILE).read()))
        return 0

    if args.old2new and any([args.add, args.delete]):
        p.error("--old2new and [--add, --delete] are mutually exclusive")

    if args.old2new:
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

def cfgparse(reqopt=None, default=None, _cache=[0]):
    if not _cache[0]:
        # 'a' is a dict of options, each option is a list of strings, each
        # string is a line from the config file.
        a = _cfgparse(RCFILE)

        # parse list-of-string options (default empty list)
        for (opt, info) in options.items():
            cfgopt = a.pop(opt, info["default"])
            if cfgopt is None:
                continue

            N, dtype = info["N"], info["type"]
            if N in ("?", 1):
                if len(cfgopt) > 1:
                    raise SystemExit(
                        "*** error: mml config: expected option {0} to "
                        "have only one line of options".format(opt))
                if N ==1 and not cfgopt:
                    raise SystemExit(
                        "*** error: mml config: expected option {0}".format(opt))
                cfgopt = cfgopt[0]

            # gets its type
            cfgopt = dtype(cfgopt)

            # some checks
            if opt == "switch":
                for (i, pair) in enumerate(cfgopt):
                    try:
                        old, new = pair.split()
                    except ValueError:
                        raise SystemExit("*** error: mml config: switch option "
                                         "must be specified as a b")
                    cfgopt[i] = (old.strip(), new.strip())

            a[opt] = cfgopt

        _cache[0] = Options(**a)

    config = _cache[0]
    if reqopt is not None:
        return getattr(config, reqopt, default)

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
        key = add[0].lower().strip()
        val = [x.strip() for x in add[1:] if x.split()]
        if key not in options:
            raise SystemExit("*** error: mml config: {0}: not a matmodlab "
                             "option".format(key))

        # dumb check to see if it is a file path
        if options[key]["type"] == ListOfFiles:
            for (i, v) in enumerate(val):
                if os.path.sep in v:
                    val[i] = os.path.realpath(os.path.expanduser(v))

        if key == "switch":
            try:
                old, new = val
            except ValueError:
                raise SystemExit("*** error: mml config: switch expected "
                                 "orig_mat switch_mat arguments")
            val = (old.strip(), new.strip())

        val = " ".join(val)

        write("writing the following options to {0}:".format(filename))
        write("  {0}: {1}".format(key, val))
        if options[key]["N"] == "?":
            # 0 or 1 option
            a[key] = [val]
        else:
            a.setdefault(key, []).append(val)

    if delete is not None:
        write("deleting the following options from {0}:".format(filename))
        key = delete[0].lower().strip()
        val = [x.strip() for x in delete[1:] if x.split()]
        if not val: val = None
        else: val = " ".join(val)
        write("  {0}: {1}".format(key, val))
        av = a.pop(key, None)
        if not av:
            write("*** warning: {0}: option had not been set".format(key, val))
        elif val:
            a[key] = [x for x in av if x != val]
        # if val is None, the entire option is deleted

    lines = []
    for (k, v) in a.items():
        # make sure each line is unique
        v = list(set(v))
        if not v: continue
        key = k.strip().lower()
        lines.append('[{0}]'.format(key))
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
