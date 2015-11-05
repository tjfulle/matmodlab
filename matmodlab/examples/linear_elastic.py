#!/usr/bin/env mmd
from matmodlab import *

"""Simulations to show uniaxial stress using two linear elastic models"""

# setup the simulation
models = {}
models['mps-1'] = MaterialPointSimulator('linear_elastic-1')

# set up the material, elastic model and elastic model implemented in python
parameters = {'K': 9.980040E+09, 'G': 3.750938E+09}
models['mps-1'].Material('elastic', parameters)

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
models['mps-1'].dump()

# Copy the model and run it
models['mps-2'] = MaterialPointSimulator('linear_elastic-2')
models['mps-2'].Material('pyelastic', parameters)
models['mps-2'].copy_steps(models['mps-1'])
models['mps-2'].dump()

# check the difference
exx1, sxx1 = models['mps-1'].get('E.XX', 'S.XX')
exx2, sxx2 = models['mps-2'].get('E.XX', 'S.XX')

m = np.amax(sxx1)
assert np.allclose(sxx1/m, sxx2/m)

J0_1 = models['mps-1'].material.J0
J0_2 = models['mps-2'].material.J0
assert np.allclose(J0_1, J0_2)
