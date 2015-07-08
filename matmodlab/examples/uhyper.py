#!/usr/bin/env mmd
from matmodlab import *
from os.path import join

E = 500.
Nu = .499
C10 = E / (4. * (1. + Nu))
D1 = 6. * (1. - 2. * Nu) / E

mps = MaterialPointSimulator('uhyper-neohooke')
mps.MixedStep(components=(.1,0,0), descriptors='ESS', frames=100)
mps.MixedStep(components=(0,0,0), descriptors='ESS', frames=100)

# set up the material - the two materials are identical
mps.Material(UHYPER, [C10, D1], name='Material-1',
             source_files=[join(MAT_D, 'src/uhyper_neohooke.f90')],
             libname='user_hyper_1')

mps.Material(USER, [C10, D1], name='Material-2', response=HYPERELASTIC,
             source_files=[join(MAT_D, 'src/uhyper_neohooke.f90')],
             libname='user_hyper_2', ordering=[XX, YY, ZZ, XY, XZ, YZ])

mps.Material('hyperelastic:neo hooke', [C10, D1], name='Material-3')

# set up and run the model
mps.run(model='Material-1')
mps.run(model='Material-2')
mps.run(model='Material-3')

a1 = mps.get('S.XX', model='Material-1')
a2 = mps.get('S.XX', model='Material-2')
a3 = mps.get('S.XX', model='Material-3')
assert np.allclose(a1, a2)
assert np.allclose(a1, a3)
