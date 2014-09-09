#!/usr/bin/env mmd

E   =   0.1100000E+12
NU  =   0.3400000
LAM =   0.87220149253731343283582089552238805970149253731343E+11
G   =   0.41044776119402985074626865671641791044776119402985E+11
K   =   0.11458333333333333333333333333333333333333333333333E+12
USM =   0.16930970149253731343283582089552238805970149253731E+12
Y   =   70.0E+6
HFAC = 1.0 / 10.0
H = 3.0 * HFAC / (1.0 - HFAC) * G
YF = Y * (1.0 + HFAC)
EPSY = Y / (2.0 * G)
SIGAY = (2.0 * G + LAM) * EPSY
SIGLY = LAM * EPSY
EPSY2 = 2.0 * Y / (2.0 * G)
SIGAY2 = ((2 * G + 3 * LAM) * EPSY2 -YF ) / 3.0 + YF
SIGLY2 = ((2 * G + 3 * LAM) * EPSY2 -YF ) / 3.0

path = """
0 0 222222 0 0 0 0 0 0
1 1 222222 0.0008527272727272727 0 0 0 0 0
2 1 222222 0.0017054545454545454 0 0 0 0 0
"""

# set up the driver
driver = Driver("Continuum", path=path, kappa=0.0, amplitude=1.0,
                rate_multiplier=1.0, step_multiplier=200.0, num_io_dumps="all",
                estar=1.0, tstar=1.0, sstar=1.0, fstar=1.0, efstar=1.0,
                dstar=1.0, proportional=False, termination_time=None)

# set up the material
parameters = {"K":1.145833E+11, "G":4.104478E+10, "Y0":7.000000E+07,
              "H":1.368159E+10, "BETA":1.000000E+00}
material = Material("vonmises", parameters=parameters)

# set up and run the model
runid = "j2_plasticity_kin_hardening"
mps = MaterialPointSimulator(runid, driver, material)
mps.run()
