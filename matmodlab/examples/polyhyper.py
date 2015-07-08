#!/usr/bin/env mmd
from matmodlab import *
from os.path import join

E = 500.
Nu = .4999
C10 = E / (4. * (1. + Nu))
C10, C01 = C10 / 2., C10 / 2.
D1 = 6. * (1. - 2. * Nu) / E

mps = MaterialPointSimulator('polyhyper')
mps.MixedStep(components=(.1,0,0), descriptors='ESS', frames=100)
mps.MixedStep(components=(0,0,0), descriptors='ESS', frames=100)

# set up the material - the two materials are identical
mps.Material('hyperelastic:polynomial', {'C10': C10, 'C01': C01, 'D1': D1})

# set up and run the model
mps.run()
