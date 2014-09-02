#!/usr/bin/env mmd

K = 9.980040E+09
G = 3.750938E+09
LAM = K - 2.0 / 3.0 * G
E   = 9.0 * K * G / (3.0 * K + G)
NU  = (3.0 * K - 2.0 * G) / (2.0 * (3.0 * K + G))

path = """
0 0 222222 0 0 0 0 0 0
1 1 222222 1 0 0 0 0 0
2 1 222222 2 0 0 0 0 0
3 1 222222 1 0 0 0 0 0
4 1 222222 0 0 0 0 0 0
"""

# set up the driver
driver = Driver("Continuum", path=path, kappa=0.0, amplitude=1.0,
                rate_multiplier=1.0, step_multiplier=50.0, num_io_dumps="all",
                estar=0.5, tstar=1.0, sstar=1.0, fstar=1.0, efstar=1.0,
                dstar=1.0, proportional=False, termination_time=None)

# set up the material
parameters = {"K":9.980040E+09, "G":3.750938E+09}
material = Material("elastic", parameters=parameters)

# set up and run the model
runid = "uniaxial_strain_extension_epscont"
mps = MaterialPointSimulator(runid, driver, material)
mps.run()

mps.extract_from_db(variables=["STRESS", "STRAIN"], format="ascii",
                    step=1, ffmt="12.6E")

