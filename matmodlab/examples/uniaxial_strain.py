#!/usr/bin/env mmd
from matmodlab import *

"""Simulations to show uniaxial strain

In the first simulation, the full strain tensor is prescribed

In the second, the axial stress is prescribed and the lateral strains are held
at 0

"""

models = {}

# setup the simulation
models['mps-1'] = MaterialPointSimulator("uniaxial_strain")

# define the steps
x, N = 1e-2, 25
models['mps-1'].StrainStep(components=(1, 0, 0), scale=x, frames=N)
models['mps-1'].StrainStep(components=(2, 0, 0), scale=x, frames=N)
models['mps-1'].StrainStep(components=(1, 0, 0), scale=x, frames=N)
models['mps-1'].StrainStep(components=(0, 0, 0), scale=x, frames=N)

# set up the material
parameters = {"K": 9.980040E+09, "G": 3.750938E+09}
models['mps-1'].Material("elastic", parameters)

# run the simulation
models['mps-1'].run()

# Run the same steps but use the stress history from the previous simulation
# setup the simulation
models['mps-2'] = MaterialPointSimulator("uniaxial_strain-1")

# set up the steps, using strain from the previous simulaiton
data_1 = models['mps-1'].get('E.XX', 'S.XX', at_step=1)
for row in data_1[1:]:
    models['mps-2'].MixedStep(components=(row[1], 0, 0), descriptors='SEE', frames=N)
models['mps-2'].Material("elastic", parameters)
models['mps-2'].run()

# check the difference
data_2 = models['mps-2'].get('E.XX', 'S.XX', at_step=1)

m = np.amax(data_1[:,0])
assert np.allclose(data_1[:,0]/m, data_2[:,0]/m)
