#!/usr/bin/env mmd
from matmodlab import *

"""Example of a user defined material. The neohooke material model is defined
in the mml_userenv.py file in this directory.

"""

E = 500.
Nu = .45
C10 = E / (4. * (1. + Nu))
D1 = 6. * (1. - 2. * Nu) / E

mps = MaterialPointSimulator('user-neohooke')

f = np.sin
t = 0.
n = 200
dt = 2. * pi / n
for i in range(n):
    t += dt
    mps.StrainStep(components=(f(t), 0, 0), increment=dt, frames=1, scale=.1)

# Instantiate the user's own neohooke model
parameters = [C10, D1]
mps.Material('neohooke_u', parameters)

# set up and run the model
mps.run()
