#!/usr/bin/env mmd
from matmodlab import *

"""Simulations to show uniaxial stress for a plastic material

"""

# setup the simulation
mps = MaterialPointSimulator('plastic')

# set up the material
parameters = {'E': 10e6, 'Nu': .33, 'Y': 40e3}
mps.Material('plastic', parameters)

# define the steps
N = 100
mps.MixedStep(components=(.02, 0, 0), descriptors='ESS', frames=N)
mps.MixedStep(components=(0, 0, 0), descriptors='ESS', frames=N)

# run the simulation
mps.run()

a = mps.get('SDV.Mises')
S = np.amax(a)
assert (mps.models['Material-1'].material.parameters['Y'] - S) < 1.e-6
