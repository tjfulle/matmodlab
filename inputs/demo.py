#!/usr/bin/env python
from matmodlab import *

# Instantiate the simulator
mps = MaterialPointSimulator("demo-0")

# Define the path through which the simulator will be drive the material.
# This particular path is stress-controlled uniaxial strain to 1MPa in 25 steps
path = """
0  0 EEE  0.0    0.0  0.0
1 25 SEE  1.0e6  0.0  0.0
"""
mps.Driver("Continuum", path)

# Define the material model for the simulator
parameters = {"K": 1.35e11, "G": 5.3e10}
mps.Material("elastic", parameters)

# Run the simulation
mps.run()

# In the simulation below, an identical simulation will be defined, but by
# defining the legs of the deformation path directly

# Instantiate the simulator
mps = MaterialPointSimulator("demo-1")

# Stress-controlled uniaxial strain to 1MPa in 25 steps
t, dt = 0, 1
legs = [Leg(t, dt, [("S", 1.0e6), ("E", 0.0), ("E", 0.0)], num_steps=25)]
mps.Driver("Continuum", legs)

# set up the material
parameters = {"K": 1.35e11, "G": 5.3e10}
mps.Material("elastic", parameters)

# run the simulation
mps.run()
