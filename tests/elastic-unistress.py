#!/usr/bin/env python
import os
import sys
try: sys.path.insert(0, os.environ["MMLROOT"])
except KeyError: raise SystemExit("MMLROOT environment variable not set")

from matmodlab import Driver, Material, MaterialPointSimulator

path = """
0 0 444 0 0 0
1 1 444 1 0 0
2 1 444 2 0 0
3 1 444 1 0 0
4 1 444 0 0 0
"""
# set up the driver
driver = Driver(kind="Continuum", kappa=0., tstar=1., sstar=-1e-6, amplitude=1.,
                step_multiplier=1000, rate_multiplier=1., path=path)

# set up the material
parameters = {"K": 9.980040E+09, "G": 3.750938E+09}
material = Material("elastic", parameters=parameters)

runid = "elastic-unistress"
mps = MaterialPointSimulator(runid, driver, material)

mps.run()
mps.extract_from_db(format="ascii", variables=["STRESS", "STRAIN"])
