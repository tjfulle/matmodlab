from matmodlab import *

# Create the material point simulator
mps = MaterialPointSimulator('mimic')

# Define the material
mps.Material('elastic', {'K': 1.35e11, 'G': 5.3e10}, switch='vonmises')

# Define the strain step
mps.StrainStep(components=(1, 0, 0), scale=.02)

# Run the simulation
mps.run()
