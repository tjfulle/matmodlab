#!/usr/bin/env python
from matmodlab import *

# stress-controlled uniaxial strain to 1MPa in 25 steps
path = """
0  0 EEE  0.0    0.0  0.0
1 25 SEE  1.0e6  0.0  0.0
"""

mps = MaterialPointSimulator("break_point")

# set up the driver
mps.Driver("Continuum", path)

# set up the material
parameters = {"K": 1.35e11, "G": 5.3e10}
mps.Material("elastic", parameters)

# set up and run the model
mps.break_point("time>.5 and stress_xx >= .5e6")
mps.run()
