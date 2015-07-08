#!/usr/bin/env mmd
from matmodlab import *

eps = 6.9314718055994529E-01
e = np.array([0, eps, 0, -eps])
t = np.linspace(0., 3., 4)
f = piecewise_linear(t, e)
N = 1000
mps = MaterialPointSimulator("mooney-rivlin")
parameters = {"C10": 72, "C01": 7.56, "NU": .49}
mps.Material("mooney_rivlin", parameters)
mps.GenSteps(MixedStep, components=(1, 0, 0), descriptors="ESS",
frames=1, steps=N, amplitude=(f,))
mps.run()
