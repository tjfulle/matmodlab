import os
import sys

# setup environment for executable scripts
D = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.dirname(D))

# check prerequisites
import __config__ as cfg
errors = cfg.check_prereqs()
if errors:
    raise SystemExit("*** error: matmodlab could not run due to the "
                     "following errors:\n  {0}".format("\n  ".join(errors)))
