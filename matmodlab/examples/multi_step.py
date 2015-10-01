#!/usr/bin/env python
from matmodlab import *

"""Exercise the material through multiple steps using different step definitions"""

# Instantiate the simulator
mps = MaterialPointSimulator("multi-step", initial_temperature=75.)

# Define the material model for the simulator
parameters = {"K": 1.35e11, "G": 5.3e10}
mps.Material("elastic", parameters)

s11, s22, s33 = 2.056667E+10, 9.966667E+09, 9.966667E+09
N, f1 = 10, 1.105171

# Deformation steps
mps.StrainStep(increment=1., components=(.1, 0., 0.), frames=N)
mps.StrainStep(increment=1., components=(0., 0., 0.), frames=N)
mps.MixedStep(increment=1., components=(s11, s22, s33), frames=N, descriptors="SSS")
mps.MixedStep(increment=1., components=(0., 0., 0.), frames=N, descriptors="SSS")
mps.StrainRateStep(increment=1., components=(.1, 0., 0.), frames=N)
mps.StrainRateStep(increment=1., components=(.1, 0., 0.), frames=N, scale=-1)
mps.StressRateStep(increment=1., components=(s11, s22, s33), frames=N)
mps.StressRateStep(increment=1., components=(s11, s22, s33), frames=N, scale=-1)
mps.DefGradStep(increment=1., components=(f1,0.,0.,0.,1.,0.,0.,0.,1.), frames=N)
mps.DefGradStep(increment=1., components=(1.,0.,0.,0.,1.,0.,0.,0.,1.), frames=N)

# Run the simulation
mps.run()
