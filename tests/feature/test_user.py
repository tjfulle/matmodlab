#!/usr/bin/env mmd
from matmodlab import *

runid = "user_thermoelastic"

class TestUser(TestBase):
    def __init__(self):
        self.runid = runid
        self.keywords = ["fast", "material", "thermoelastic", "user", "builtin"]

@matmodlab
def run_user_thermoelastic(*args, **kwargs):

    path = """
    0 0 EEET 0 0 0 298
    1 1 ESST 1 0 0 300
    2 1 ESST 2 0 0 310
    3 1 ESST 1 0 0 340
    4 1 ESST 0 0 0 375
    """

    mps = MaterialPointSimulator(runid)
    mps.Driver("Continuum", path)

    E0, NU0, T0 = 29.E+06, .33, 295.E+00
    E1, NU1, T1 = 23.E+06, .33, 295.E+00
    TI, ALPHA = 298., 0
    parameters = np.array([E0, NU0, T0, E1, NU1, T1, ALPHA, TI])
    mps.Material("user", parameters,
                 initial_temp=298., depvar=12,
                 source_files=["thermoelastic.f90"],
                 source_directory=os.path.join(MAT_D, "usermats"))

    mps.run()

if __name__ == "__main__":
    run_user_thermoelastic()
