#!/usr/bin/env mmd
from matmodlab import *

"""Simulations to show uniaxial stress

In the first simulation, the full stress tensor is prescribed

In the second, the axial strain is prescribed and the lateral stresses held at
0

"""

models = {}

# setup the simulation
models['mps-1'] = MaterialPointSimulator("uniaxial_stress")

# define the steps
x, N = 1e6, 100
models['mps-1'].StressStep(components=(1, 0, 0), scale=x, frames=N)
models['mps-1'].StressStep(components=(2, 0, 0), scale=x, frames=N)
models['mps-1'].StressStep(components=(1, 0, 0), scale=x, frames=N)
models['mps-1'].StressStep(components=(0, 0, 0), scale=x, frames=N)

# set up the material
parameters = {"K": 9.980040E+09, "G": 3.750938E+09}
models['mps-1'].Material("elastic", parameters)

# run the simulation
models['mps-1'].run()

# Run the same steps but use the strain history from the previous simulation
# setup the simulation
models['mps-2'] = MaterialPointSimulator("uniaxial_stress-1")

# set up the steps, using strain from the previous simulaiton
exx, sxx = models['mps-1'].get('E.XX', 'S.XX', at_step=1)
for e in exx[1:]:
    models['mps-2'].MixedStep(components=(e, 0, 0), descriptors='ESS', frames=N)
models['mps-2'].Material("elastic", parameters)
models['mps-2'].run()

# check the difference
exx2, sxx2 = models['mps-2'].get('E.XX', 'S.XX', at_step=1)

m = np.amax(sxx)
assert np.allclose(sxx/m, sxx2/m)
