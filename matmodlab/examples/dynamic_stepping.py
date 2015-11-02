#!/usr/bin/env python
import numpy as np
import matmodlab as mml

#
# This input shows how to dynamically control a matmodlab simulation.
#
# This simulation demonstrates how to use strain-control to do most
# of the heavy lifting for a stress-controlled step. We want to perform
# uniaxial extension up to an axial stress of 'SXX_target', but we want
# to leverage the speed and stability of a strain-controlled simulation
# for as long as possible.
#

# initialize the job, request output in gzipped plaintext ("txt.gz")
mps = mml.MaterialPointSimulator("dynamic_stepping", output_format='txt.gz')

# set up the linear elastic model "pyelastic"
mps.Material("pyelastic", {"K": 1.0e10, "G": 1.0e9})

# define the axial stress target and strain increment
EXX_incr = 0.01
SXX_target = 1.0e9

# Prep the simulator
mps.arm()

# Run the first step
mps.StrainStep(components=(EXX_incr, 0., 0.), frames=50)
mps.run_uncompleted_steps()

# Track the output stress and compute the stress increment over the step
SXX = [0.0, mps.get("S.XX")[-1]]
DSXX = SXX[-1] - SXX[-2]

# Keep incrementing the strain until we get 'close' to the target stress
while SXX[-1] + 1.5 * DSXX < 1.0e9:
    
    EXX = mps.get("E.XX")[-1]
    mps.StrainStep(components=(EXX + EXX_incr, 0., 0.), frames=50)
    mps.run_uncompleted_steps()

    SXX.append(mps.get("S.XX")[-1])
    DSXX = SXX[-1] - SXX[-2]

# Finish with a stress-controlled step
mps.MixedStep(components=(SXX_target, 0., 0.), descriptors='SEE', frames=50)
mps.run_uncompleted_steps()

# Write to disk
mps.finish()
