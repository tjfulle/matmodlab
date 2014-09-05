#!/usr/bin/env mmd
from matmodlab import *

K = 9.980040E+09
G = 3.750938E+09

path = """
0 0 444444 0 0 0 0 0 0
1 1 444 -7490645504 -3739707392 -3739707392
2 1 444 -14981291008 -7479414784 -7479414784
3 1 444 -7490645504 -3739707392 -3739707392
4 1 444 0 0 0
"""

@matmodlab
def runner():
    # set up the driver
    driver = Driver("Continuum", path=path, kappa=0.0, amplitude=1.0,
                    rate_multiplier=1.0, step_multiplier=100.0, num_io_dumps=20,
                    estar=1.0, tstar=1.0, sstar=1.0, fstar=1.0, efstar=1.0,
                    dstar=1.0, proportional=False, termination_time=None)

    # set up the material
    parameters = {"K": K, "G": G}
    material = Material("elastic", parameters=parameters)

    # set up and run the model
    runid = "elastic-unistrain-stressc"
    mps = MaterialPointSimulator(runid, driver, material)
    mps.run()

    mps.dump(variables=["STRESS", "STRAIN"], format="ascii",
                        step=1, ffmt="12.6E")

runner()
