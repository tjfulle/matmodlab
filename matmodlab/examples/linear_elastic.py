#!/usr/bin/env mmd
from matmodlab import *

"""Simulations to show uniaxial stress using two linear elastic models"""

# setup the simulation
mps = MaterialPointSimulator('linear_elastic')

# define the steps
x, N = .01, 20
mps.MixedStep(components=(1, 0, 0), descriptors='ESS', scale=x, frames=N)
mps.MixedStep(components=(2, 0, 0), descriptors='ESS', scale=x, frames=N)
mps.MixedStep(components=(1, 0, 0), descriptors='ESS', scale=x, frames=N)
mps.MixedStep(components=(0, 0, 0), descriptors='ESS', scale=x, frames=N)

# set up the material, elastic model and elastic model implemented in python
parameters = {'K': 9.980040E+09, 'G': 3.750938E+09}
mps.Material('elastic', parameters, name='Material-1')
mps.Material('pyelastic', parameters, name='Material-2')

# run the simulation
mps.run(model='Material-1')
mps.run(model='Material-2')

# check the difference
data_1 = mps.get('E.XX', 'S.XX', model='Material-1')
data_2 = mps.get('E.XX', 'S.XX', model='Material-2')

m = np.amax(data_1[:,1])
assert np.allclose(data_1[:,1]/m, data_2[:,1]/m)

J0_1 = mps.models['Material-1'].material.J0
J0_2 = mps.models['Material-2'].material.J0
assert np.allclose(J0_1, J0_2)
