from matmodlab import *

"""This example demonstrates the break point capability that allows a user to
stop a simulation and probe for details

"""

mps = MaterialPointSimulator("break_point")

# stress-controlled uniaxial strain to 1MPa in 25 steps
mps.MixedStep(components=(1.e6, 0., 0.), descriptors="SEE", frames=25)

# set up the material
parameters = {"K": 1.35e11, "G": 5.3e10}
mps.Material("elastic", parameters)

# set up and run the model
mps.break_point("time>.5 and S.XX >= .5e6")
mps.run()
