#!/usr/bin/env mmd
from matmodlab import *

"""Simulations to show uniaxial strain

In the first simulation, the full strain tensor is prescribed

In the second, the axial stress is prescribed and the lateral strains are held
at 0

"""

models = {}
parameters = {"K": 9.980040E+09, "G": 3.750938E+09}

# setup the simulation
models['mps-1'] = MaterialPointSimulator("uniaxial_strain")

# set up the material
models['mps-1'].Material("elastic", parameters)

# define the steps
x, N = 1e-2, 25
models['mps-1'].StrainStep(components=(1, 0, 0), scale=x, frames=N)
models['mps-1'].StrainStep(components=(2, 0, 0), scale=x, frames=N)
models['mps-1'].StrainStep(components=(1, 0, 0), scale=x, frames=N)
models['mps-1'].StrainStep(components=(0, 0, 0), scale=x, frames=N)
models['mps-1'].dump()

# Run the same steps but use the stress history from the previous simulation
# setup the simulation
models['mps-2'] = MaterialPointSimulator("uniaxial_strain-1")
models['mps-2'].Material("elastic", parameters)

# set up the steps, using strain from the previous simulaiton
exx, sxx = models['mps-1'].get('E.XX', 'S.XX', at_step=1)
for s in sxx[1:]:
    models['mps-2'].MixedStep(components=(s, 0, 0), descriptors='SEE', frames=N)
models['mps-2'].dump()

# check the difference
exx2, sxx2 = models['mps-2'].get('E.XX', 'S.XX', at_step=1)

m = np.amax(exx)
assert np.allclose(exx/m, exx2/m)
