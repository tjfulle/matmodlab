from StringIO import StringIO
from matmodlab import *

"""DataSteps allow reading in external data files and using them to create
steps. In this example, a fake data file is created and used to run a
simulation.

"""

# An example data file.
table ="""\
# TIME E.XX E.YY E.ZZ
 0 0.    0.    0.
 1  .05  -.025 -.025
 2  .1   -.05  -.05
 3  .05  -.025 -.025
 4 0.    0.    0.
 5  .05  -.025 -.025
 6  .1   -.05  -.05
 7  .05  -.025 -.025
 8 0.    0.    0.
 9  .05  -.025 -.025
10  .1   -.05  -.05
11  .05  -.025 -.025
12 0.    0.    0.
13  .05  -.025 -.025
14  .1   -.05  -.05
15  .05  -.025 -.025
16 0.    0.    0.
"""

mps = MaterialPointSimulator("data_steps")

# Read in the data and create steps, tc is the column where time is found
mps.DataSteps(StringIO(table), tc=0, frames=10, descriptors='EEE',
              columns=(1,2,3))

# set up the material
parameters = {"K": 1.35e11, "G": 5.3e10}
mps.Material("elastic", parameters)

# run the simulation
mps.run()
