#!/usr/bin/env xpython
from matmodlab import *

E = 500.
Nu = .45
C10 = E / (4. * (1. + Nu))
D1 = 6. * (1. - 2. * Nu) / E

f2 = Function(2, "analytic_expression", lambda t: np.sin(t))
path = """
{0} 2:1.e-1 0 0
""".format(2*pi)

@matmodlab
def runner():
    runid = "uhyper-neohooke"

    # set up the driver
    driver = Driver("Continuum", path=path, path_input="function",
                    num_steps=200, cfmt="222", functions=f2)

    # set up the material
    constants = [C10, D1]
    material = Material("uhyper", parameters=constants, constants=2,
                        source_files=["uhyper.f90"],
                        source_directory="{0}/materials/abaumats".format(ROOT_D))

    # set up and run the model
    mps = MaterialPointSimulator(runid, driver, material)
    mps.run()

runner()
