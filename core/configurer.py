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
from core.logger import ConsoleLogger as logger
from core.product import RCFILE


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    p = argparse.ArgumentParser(prog="mml config",
                                description="%(prog)s: Set matmodlab options")
    p.add_argument("--add", nargs=2, metavar=("name", "value"),
        help="name and value of option to add to configuration")
    p.add_argument("--del", dest="delete", nargs=2, metavar=("name", "value"),
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
    return s.lower() in ("0", "none", "false")

def myint(n):
    return int(float(n))

def cfgparse(option=None, default=None, _cache=[0]):
    if not _cache[0]:
        a = _cfgparse(RCFILE)

        # set some expected options (even if they are empty)
        a["materials"] = a.pop("materials", [])
        a["tests"] = a.pop("tests", [])

        # a is a dict of options, each option is a list of strings. modify to
        # suite some options better
        for i in ("verbosity", "nprocs"):
            x = a.pop(i, None)
            if x is not None:
                a[i] = myint(x[0])

        for i in ("sqa", "debug"):
            x = a.pop(i, None)
            if x is not None:
                a[i] = mybool(x[0])

        for i in ("switch", "mimic", "warn"):
            x = a.pop(i, None)
            if x is not None:
                a[i] = x[0]

        _cache[0] = Options(**a)

    config = _cache[0]
    if option is not None:
        return getattr(config, option, default)

    return config


def _cfgparse(filename):
    cfg = {}
    try:
        lines = open(filename).readlines()
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
        logger.write("writing the following options to {0}:".format(filename))
        k, v = add
        k = k.lower()
        # dumb check to see if it is a file path
        if os.path.sep in v:
            v = os.path.realpath(os.path.expanduser(v))
        logger.write("  {0}: {1}".format(k, v), transform=str)
        a.setdefault(k, []).append(v.strip())

    if delete is not None:
        logger.write("deleting the following options from {0}:".format(filename))
        k, v = delete
        k = k.lower()
        logger.write("  {0}: {1}".format(k, v), transform=str)
        av = a.get(k)
        if not av:
            logger.warn("{0}: option had not been set".format(k, v))
        else:
            a[k] = [x for x in av if x != v]

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
    logger.write("=" * 82)
    logger.write("**** WARNING " * 6 + "****")
    logger.write("**** WARNING " * 6 + "****")
    logger.write("""
   LEGACY ENVIRONMENT VARIABLE 'MMLMTLS' CONVERTED TO NEW STYLE CONFIGURATION
   IN ~/.matmodlabrc.  THE LOCATION OF THIS FILE IS ALSO CONFIGURABLE BY THE
   MATMODLABRC ENVIRONMENT VARIABLE.  REMOVE MMLMTLS FROM YOUR ENVIRONMENT
   TO AVOID THIS WARNING IN THE FUTURE
""", transform=str)
    logger.write("**** WARNING " * 6 + "****")
    logger.write("**** WARNING " * 6 + "****")
    logger.write("=" * 82)
    if mtls_d:
        cfgedit(RCFILE, add=("materials", mtls_d))

if __name__ == "__main__":
    main()
