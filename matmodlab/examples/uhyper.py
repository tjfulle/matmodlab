#!/usr/bin/env mmd
from matmodlab import *
from os.path import join

E = 500.
Nu = .499
C10 = E / (4. * (1. + Nu))
D1 = 6. * (1. - 2. * Nu) / E

models = {}

# First model
models['mps-1'] = MaterialPointSimulator('uhyper_neohooke-1')
models['mps-1'].Material(UHYPER, [C10, D1], libname='user_hyper_1',
                         source_files=[join(MAT_D, 'src/uhyper_neohooke.f90')])
models['mps-1'].MixedStep(components=(.1,0,0), descriptors='ESS', frames=100)
models['mps-1'].MixedStep(components=(0,0,0), descriptors='ESS', frames=100)

# Second model
models['mps-2'] = MaterialPointSimulator('uhyper_neohooke-2')
models['mps-2'].Material(USER, [C10, D1], response=HYPERELASTIC,
                         source_files=[join(MAT_D, 'src/uhyper_neohooke.f90')],
                         libname='user_hyper_2', ordering=[XX, YY, ZZ, XY, XZ, YZ])
models['mps-2'].copy_steps(models['mps-1'])

# Third model
models['mps-3'] = MaterialPointSimulator('uhyper_neohooke-3')
models['mps-3'].Material('hyperelastic:neo hooke', [C10, D1])
models['mps-3'].copy_steps(models['mps-1'])

# write model output
models['mps-1'].dump()
models['mps-2'].dump()
models['mps-3'].dump()

assert np.allclose(models['mps-1'].S.XX, models['mps-2'].S.XX)
assert np.allclose(models['mps-1'].S.XX, models['mps-3'].S.XX)
