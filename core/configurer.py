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
    def __init__(self, dict):
        l = dict.get("materials", [])
        self.user_mats = [os.path.expanduser(f) for f in l]
        l = dict.get("tests", [])
        self.user_tests = [os.path.expanduser(f) for f in l]


def cfgparse(option=None, filename=None, _cache=[0]):
    filename = filename or RCFILE
    if not _cache[0]:
        a = _cfgparse(filename)
        _cache[0] = Options(a)
    config = _cache[0]
    if option is not None:
        return getattr(config, option)
    return config


def _cfgparse(filename):
    cfg = {}
    try:
        lines = open(filename).readlines()
    except IOError:
        return {}
    regex = re.compile( r'\[(.*?)\]')
    stack = []
    for line in lines:
        tag = regex.search(line)
        if tag:
            if stack:
                cfg.setdefault(stack[0], []).extend(stack[1:])
            stack = [tag.group(1).strip()]
            continue
        if not line.split():
            continue
        stack.append(line.strip())
    if stack:
        cfg.setdefault(stack[0], []).extend(stack[1:])
    return cfg


def cfgedit(filename, add=None, delete=None):

    a = _cfgparse(filename)

    if add is not None:
        logger.write("writing the following options to {0}:".format(filename))
        k, v = add
        k = k.lower()
        v = os.path.expanduser(v)
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
        lines.append("\n".join(v))

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
