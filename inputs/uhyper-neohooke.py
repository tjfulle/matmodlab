#!/usr/bin/env mmd
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
    mps = MaterialPointSimulator("uhyper-neohooke")

    # set up the driver
    mps.Driver("Continuum", path, path_input="function",
               num_steps=200, cfmt="222", functions=f2)

    # set up the material
    parameters = [C10, D1]
    mps.Material("uhyper", parameters,
                 source_files=["uhyper.f90"],
                 source_directory="{0}/materials/abaumats".format(ROOT_D))

    # set up and run the model
    mps.run()

runner()
