#!/usr/bin/env mmd


path = """
0.0000000000e+00 0.0000000000e+00 0.0000000000e+00 0.0000000000e+00 0.0000000000e+00 0.0000000000e+00 0.0000000000e+00
1.0000000000e+00 1.8082567543e-01 1.7414396724e-01 1.4884090386e-01 0.0000000000e+00 0.0000000000e+00 0.0000000000e+00
2.0000000000e+00 3.6165135086e-01 3.4828793447e-01 2.9768180772e-01 0.0000000000e+00 0.0000000000e+00 0.0000000000e+00
"""

# set up the driver
driver = Driver("Continuum", path=path, kappa=0.0, amplitude=1.0,
                rate_multiplier=1.0, step_multiplier=10.0, num_io_dumps="all",
                estar=1.0, tstar=1.0, sstar=1.0, fstar=1.0, efstar=1.0,
                dstar=1.0, proportional=False, termination_time=None)

# set up the material
parameters = {"K":2.151596E+02, "G":4.090153E+03, "A1":1.756496E+03,
              "A4":4.976894E+00}
material = Material("pyplastic", parameters=parameters)

# set up and run the model
runid = "linear_drucker_prager_random"
mps = MaterialPointSimulator(runid, driver, material)
mps.run()

mps.extract_from_db(variables=["STRESS", "STRAIN"], format="ascii",
                    step=1, ffmt="12.6E")

