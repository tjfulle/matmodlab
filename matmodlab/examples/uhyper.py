#!/usr/bin/env mmd
from matmodlab import *
from os.path import join

E = 500.
Nu = .499
C10 = E / (4. * (1. + Nu))
D1 = 6. * (1. - 2. * Nu) / E

models = {}
models['mps-1'] = MaterialPointSimulator('uhyper_neohooke-1')
models['mps-1'].MixedStep(components=(.1,0,0), descriptors='ESS', frames=100)
models['mps-1'].MixedStep(components=(0,0,0), descriptors='ESS', frames=100)

# set up the material - the two materials are identical
models['mps-1'].Material(UHYPER, [C10, D1], libname='user_hyper_1',
                         source_files=[join(MAT_D, 'src/uhyper_neohooke.f90')])

models['mps-2'] = models['mps-1'].copy('uhyper_neohooke-2')
models['mps-2'].Material(USER, [C10, D1], response=HYPERELASTIC,
                         source_files=[join(MAT_D, 'src/uhyper_neohooke.f90')],
                         libname='user_hyper_2', ordering=[XX, YY, ZZ, XY, XZ, YZ])

models['mps-3'] = models['mps-1'].copy('uhyper_neohooke-3')
models['mps-3'].Material('hyperelastic:neo hooke', [C10, D1])

# set up and run the model
models['mps-1'].run()
models['mps-2'].run()
models['mps-3'].run()

assert np.allclose(models['mps-1'].S.XX, models['mps-2'].S.XX)
assert np.allclose(models['mps-1'].S.XX, models['mps-3'].S.XX)
