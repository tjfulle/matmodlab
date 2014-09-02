#!/usr/bin/env mmd

LAM = 1.0e9
MU = 1.0e8
K = LAM + 2.0 / 3.0 * MU
FAC = 1.0e6
RINT = 1.0 * FAC
ZINT = sqrt(2.0) * FAC
ES = RINT / (2.0 * sqrt(2.0) * MU)
NUM = 3.0 * K ** 2 * (RINT / ZINT) ** 2
DEN = 3.0 * K * (RINT / ZINT) ** 2 + 2.0 * MU
TREPS = ZINT / (sqrt(3.0) * K - sqrt(3.0) * NUM / DEN)
EV = TREPS / 3.0

path = """
0 0 222222 0 0 0 0 0 0
1 1 111111 0 0 0 0.0035355339059327372 0 0
2 1 111111 0.0022963966338592308 0.0022963966338592308 0.0022963966338592308 0 0 0
"""

# set up the driver
driver = Driver("Continuum", path=path, kappa=0.0, amplitude=1.0,
                rate_multiplier=1.0, step_multiplier=100.0, num_io_dumps="all",
                estar=1.0, tstar=1.0, sstar=1.0, fstar=1.0, efstar=1.0,
                dstar=1.0, proportional=False, termination_time=None)

# set up the material
parameters = {"K":1.066667E+09, "G":1.000000E+08, "A1":7.071068E+05,
              "A4":2.886751E-01}
material = Material("pyplastic", parameters=parameters)

# set up and run the model
runid = "linear_drucker_prager_spherical"
mps = MaterialPointSimulator(runid, driver, material)
mps.run()
