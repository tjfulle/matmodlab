#!/usr/bin/env mmd
from matmodlab import *

"""Simulations to show uniaxial stress using two linear elastic models"""

# setup the simulation
models = {}
models['mps-1'] = MaterialPointSimulator('linear_elastic-1')

# define the steps
x, N = .01, 20
models['mps-1'].MixedStep(components=(1, 0, 0), descriptors='ESS',
                          scale=x, frames=N)
models['mps-1'].MixedStep(components=(2, 0, 0), descriptors='ESS',
                          scale=x, frames=N)
models['mps-1'].MixedStep(components=(1, 0, 0), descriptors='ESS',
                          scale=x, frames=N)
models['mps-1'].MixedStep(components=(0, 0, 0), descriptors='ESS',
                          scale=x, frames=N)

# set up the material, elastic model and elastic model implemented in python
parameters = {'K': 9.980040E+09, 'G': 3.750938E+09}
models['mps-1'].Material('elastic', parameters)

# run the simulation
models['mps-1'].run()

# Copy the model and run it
models['mps-2'] = models['mps-1'].copy('linear_elastic-2')
models['mps-2'].Material('pyelastic', parameters)
models['mps-2'].run()

# check the difference
exx1, sxx1 = models['mps-1'].get('E.XX', 'S.XX')
exx2, sxx2 = models['mps-2'].get('E.XX', 'S.XX')

m = np.amax(sxx1)
assert np.allclose(sxx1/m, sxx2/m)

J0_1 = models['mps-1'].material.J0
J0_2 = models['mps-2'].material.J0
assert np.allclose(J0_1, J0_2)
