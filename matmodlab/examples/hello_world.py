#!/usr/bin/env python
from matmodlab import *

# initialize the job, request output in plaintext ("txt")
mps = MaterialPointSimulator("hello_world", output_format='txt')

# set up the linear elastic model "pyelastic"
mps.Material("pyelastic", {"K": 1.0e10, "G": 1.0e9})

# define the steps
mps.StrainStep(components=(1., 0., 0.), frames=25)
mps.StrainStep(components=(0., 1., 0.), frames=25)
mps.StrainStep(components=(0., 0., 1.), frames=25)
mps.StrainStep(components=(0., 0., 0.), frames=25)

# run the simulation
mps.run()
