#!/usr/bin/env mmd
from matmodlab import *

"""Example of a user defined material. The material is defined as a std_material

"""

E = 500.
Nu = .45

mps = MaterialPointSimulator('user-elastic')

f = np.sin
t = 0.
n = 200
dt = 2. * pi / n
for i in range(n):
    t += dt
    mps.StrainStep(components=(f(t), 0, 0), increment=dt, frames=1, scale=.1)

# Instantiate the user's own elastic model
parameters = [E, Nu]
mps.Material('uelastic', parameters)

# set up and run the model
mps.run()
