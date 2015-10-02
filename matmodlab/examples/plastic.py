#!/usr/bin/env mmd
from matmodlab import *

"""Simulations to show uniaxial stress for a plastic material

"""

# setup the simulation
mps = MaterialPointSimulator('plastic')

# set up the material
Y = 40e3
parameters = {'E': 10e6, 'Nu': .33, 'Y': Y}
mps.Material('plastic', parameters)

# define the steps
N = 100
mps.MixedStep(components=(.02, 0, 0), descriptors='ESS', frames=N)
mps.MixedStep(components=(0, 0, 0), descriptors='ESS', frames=N)

# run the simulation
mps.run()

a = mps.get('SDV_Mises')
S = np.amax(a)

assert abs(S - Y) <= 1e-8
