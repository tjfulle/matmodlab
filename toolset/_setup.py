import os
import sys

# setup environment for executable scripts
D = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.dirname(D))

from matmodlab import *
