import os
import sys

# setup environment for executable scripts
D = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, D)

errors = []
platform = sys.platform
(major, minor, micro, relev, ser) = sys.version_info
if (major != 3 and major != 2) or (major == 2 and minor < 7):
    errors.append("python >= 2.7 required")
    errors.append("  {0} provides {1}.{2}.{3}".format(
        sys.executable, major, minor, micro))

# --- numpy
try: import numpy
except ImportError: errors.append("numpy not found")

# --- scipy
try: import scipy
except ImportError: errors.append("scipy not found")

# check prerequisites
if errors:
    raise SystemExit("*** error: matmodlab could not run due to the "
                     "following errors:\n  {0}".format("\n  ".join(errors)))

# Public methods to set up and run a simulation
from drivers.driver import Driver
from materials.material import Material
from core.mat_point_sim import MaterialPointSimulator
