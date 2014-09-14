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

RCFILE = os.getenv("MATMODLABRC") or os.path.expanduser("~/.matmodlabrc")

def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    p = argparse.ArgumentParser(prog="mml-config")
    p.add_argument("option",
        help="matmodlab configuration, specified as option.value")
    args = p.parse_args(argv)
    try:
        key, val = args.option.split(".")
    except ValueError:
        p.error("expected options to be of form key.value")
    opts = {key.strip(): [val.strip()]}
    cfgwrite(RCFILE, opts)
    return 0


class Options:
    def __init__(self, dict):
        l = dict.get("materials", [])
        self.user_mats = [os.path.expanduser(f) for f in l]
        l = dict.get("tests", [])
        self.user_tests = [os.path.expanduser(f) for f in l]


def cfgparse(filename=None, disp=0):
    filename = filename or RCFILE
    a = _cfgparse(filename)
    config = Options(a)
    if disp == 2:
        return config.user_mats
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


def cfgwrite(filename, dict):
    logger.write("writing the following options to {0}:".format(filename))
    for (k, v) in dict.items():
        logger.write("  {0}: {1}".format(k, v), transform=str)

    a = _cfgparse(filename)
    for (k, v) in dict.items():
        if not isinstance(v, (list, tuple)):
            v = [v]
        a.setdefault(k, []).extend([x.strip() for x in v if x.split()])
    lines = []
    for (k, v) in a.items():
        lines.append('[{0}]'.format(k.strip()))
        # make sure each line is unique
        v = list(set(v))
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
        cfgwrite(RCFILE, {"materials": mtls_d})

if __name__ == "__main__":
    main()
