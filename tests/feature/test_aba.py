#!/usr/bin/env mmd
from matmodlab import *

E=500
Nu=.45
C10 = E / (4. * (1. + Nu))
D1 = 6. * (1. - 2. * Nu) / E

f2 = Function(2, "analytic_expression", lambda t: np.sin(t))
path = """
{0} 2:1.e-1 0 0
""".format(2*pi)

class TestUMat(TestBase):
    def __init__(self):
        self.runid = "umat_neohooke"
        self.keywords = ["fast", "abaqus", "umat", "neohooke", "feature", "builtin"]

class TestUHyper(TestBase):
    def __init__(self):
        self.runid = "uhyper_neohooke"
        self.keywords = ["fast", "abaqus", "uhyper", "neohooke", "feature",
                         "builtin"]

class TestUAnisoHyperInv(TestBase):
    def __init__(self):
        self.disabled = True
        self.runid = "uanisohyper_inv"
        self.keywords = ["fast", "abaqus", "uanisohyper_inv", "feature", "builtin"]

@matmodlab
def run_umat_neohooke(*args, **kwargs):
    runid = "umat_neohooke"
    logger = Logger(runid)
    driver = Driver("Continuum", path, path_input="function",
                    num_steps=200, cfmt="222", functions=f2,
                    termination_time=1.8*pi, logger=logger)
    parameters = [E, Nu]
    depvar = 2
    material = Material("umat", parameters, logger=logger,
                        source_files=["neohooke.f90"],
                        source_directory="{0}/materials/abaumats".format(ROOT_D),
                        depvar=depvar)
    mps = MaterialPointSimulator(runid, driver, material, logger=logger)
    mps.run()

@matmodlab
def run_uhyper_neohooke(*args, **kwargs):
    runid = "uhyper_neohooke"
    logger = Logger(runid)
    driver = Driver("Continuum", path, path_input="function",
                    num_steps=200, cfmt="222", functions=f2,
                    termination_time=1.8*pi, logger=logger)
    param_names = ["C10", "D1"]
    parameters = {"C10": C10, "D1": D1}
    depvar = ["MY_SDV_1", "MY_SDV_2"]
    material = Material("uhyper", parameters, logger=logger,
                        source_files=["uhyper.f90"],
                        source_directory="{0}/materials/abaumats".format(ROOT_D),
                        param_names=param_names, depvar=depvar)
    mps = MaterialPointSimulator(runid, driver, material, logger=logger)
    mps.run()

@matmodlab
def run_uanisohyper_inv(*args, **kwargs):
    runid = "uanisohyper_inv"
    logger = Logger(runid)
    driver = Driver("Continuum", path, path_input="function",
                    num_steps=200, cfmt="222", functions=f2,
                    termination_time=1.8*pi, logger=logger)

    C10 = 7.64
    D = 1.e-8
    K1 = 996.6
    K2 = 524.6
    Kappa = 0.226
    parameters = np.array([C10, D, K1, K2, Kappa])
    a = np.array([[0.643055,0.76582,0.0], [0.643055,-0.76582,0.0]])
    a = np.array([[0.643055,0.76582,0.0]])
    material = Material("uanisohyper_inv", parameters, logger=logger,
                        source_files=["uanisohyper_inv.f"],
                        source_directory="{0}/materials/abaumats".format(ROOT_D),
                        fiber_dirs=a)
    mps = MaterialPointSimulator(runid, driver, material, logger=logger)
    mps.run()

if __name__ == "__main__":
    #    run_umat_neohooke()
    #     run_uhyper_neohooke()
    run_uanisohyper_inv()
