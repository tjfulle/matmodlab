#!/usr/bin/env python

from matmodlab import *

K = 1.350E+11
G = 5.300E+10

path = """
0E+00 0 EEEEEE 0E+00 0E+00 0E+00 0E+00 0E+00 0E+00
1E+00 100 EEEEEE 1E-01 0E+00 0E+00 0E+00 0E+00 0E+00
2E+00 100 EEEEEE 0E+00 0E+00 0E+00 0E+00 0E+00 0E+00
3E+00 100 SSSEEE 2.056667E+10 9.966667E+09 9.966667E+09 0E+00 0E+00 0E+00
4E+00 100 SSSEEE 0E+00 0E+00 0E+00 0E+00 0E+00 0E+00
5E+00 100 DDDDDD 1E-01 0E+00 0E+00 0E+00 0E+00 0E+00
6E+00 100 DDDDDD -1E-01 0E+00 0E+00 0E+00 0E+00 0E+00
7E+00 100 RRRRRR 2.056667E+10 9.966667E+09 9.966667E+09 0E+00 0E+00 0E+00
8E+00 100 RRRRRR -2.056667E+10 -9.966667E+09 -9.966667E+09 0E+00 0E+00 0E+00
9E+00 100 FFFFFFFFF 1.105171E+00 0E+00 0E+00 0E+00 1E+00 0E+00 0E+00 0E+00 1E+00
1E+01 100 FFFFFFFFF 1E+00 0E+00 0E+00 0E+00 1E+00 0E+00 0E+00 0E+00 1E+00
"""

# set up the model
mps = MaterialPointSimulator("path-cycle")

# set up the driver
driver = Driver("Continuum", path, kappa=0.0, amplitude=1.0,
                rate_multiplier=1.0, step_multiplier=1.0, num_io_dumps=20,
                estar=1.0, tstar=1.0, sstar=1.0, fstar=1.0, efstar=1.0,
                dstar=1.0, proportional=False, termination_time=None)

# set up the material
parameters = {"K": K, "G": G}
material = Material("elastic", parameters)

# set up and run the model
mps.assign_driver(driver)
mps.assign_material(material)
mps.run()
