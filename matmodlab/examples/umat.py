#!/usr/bin/env python
from os.path import join
from matmodlab import *

"""Example of a user material.  The model is coded up in umat.f

"""

# Copper                  E       NU
parameters = np.array([0.110e12, .340])

# output options are
#  * REC for binary pickle of simulation data [default]
#  * TXT for whitespace-delimited text
#  * XLS or 'xlsx' for excel file
#  * PKL for python pickle

models = {}
models['mps-1'] = MaterialPointSimulator('umat')
models['mps-1'].StrainStep(components=(.2, .0, .0), frames=50)
models['mps-1'].StrainStep(components=(.0, .0, .0), frames=50)
models['mps-1'].Material(UMAT, parameters, depvar=0,
                         source_files=['umat.f'])
models['mps-1'].run()

models['mps-2'] = models['mps-1'].copy('user')
models['mps-2'].Material(USER, parameters, libname='user', depvar=0,
                         response=MECHANICAL, source_files=['umat.f'],
                         ordering=(XX,YY,ZZ,XY,XZ,YZ))
models['mps-2'].run()
