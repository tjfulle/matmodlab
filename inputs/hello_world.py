#!/usr/bin/env python
from matmodlab import *

# stress-controlled uniaxial strain to 1MPa in 25 steps
path = """
0  0 EEE  0.0    0.0  0.0
1 25 SEE  1.0e6  0.0  0.0
"""

# set up the driver
driver = Driver("Continuum", path)

# set up the material
parameters = {"K": 1.35e11, "G": 5.3e10}
material = Material("elastic", parameters)

# set up and run the model
mps = MaterialPointSimulator("hello_world", driver, material)
mps.run()
