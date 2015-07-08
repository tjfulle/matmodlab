#!/usr/bin/env python
from os.path import join
from matmodlab import *

"""Example of a user material.  The model is coded up in umat.f

"""

# Copper                  E       NU
parameters = np.array([0.110e12, .340])

# output options are (default is DBX):
#  * DBX for compressed XML file
#  * EXO for exodus II file
#  * TXT for whitespace-delimited text
#  * XLS or 'xlsx' for excel file
#  * PKL for python pickle

mps = MaterialPointSimulator('umat', output=DBX)
mps.StrainStep(components=(.2, .0, .0), frames=50)
mps.StrainStep(components=(.0, .0, .0), frames=50)

mps.Material(UMAT, parameters, name='umat', depvar=0,
             source_files=[join(MAT_D, 'umat.f')])
mps.Material(USER, parameters, name='user', libname='user', depvar=0,
             response=MECHANICAL, source_files=[join(MAT_D, 'umat.f')],
             ordering=(XX,YY,ZZ,XY,XZ,YZ))
mps.run(model='umat')
mps.run(model='user')
