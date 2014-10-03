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

umat = "umat_neohooke"
uhyper = "uhyper_neohooke"

class TestUMat(TestBase):
    def __init__(self):
        self.runid = umat
        self.keywords = ["fast", "abaqus", "umat", "neohooke", "feature", "builtin"]
    def run_job(self):
        run_umat(d=self.test_dir, v=0, test=1)

class TestUHyper(TestBase):
    def __init__(self):
        self.runid = uhyper
        self.keywords = ["fast", "abaqus", "uhyper", "neohooke", "feature",
                         "builtin"]
    def run_job(self):
        run_uhyper(d=self.test_dir, v=0, test=1)

class TestUAnisoHyperInv(TestBase):
    def __init__(self):
        self.disabled = True
        self.runid = uhyper
        self.keywords = ["fast", "abaqus", "uanisohyper_inv", "feature", "builtin"]
    def run_job(self):
        run_uanisohyper_inv(d=self.test_dir, v=0, test=1)

@matmodlab
def run_umat(d=None, v=1, test=0):
    d = d or os.getcwd()
    runid = umat
    logfile = os.path.join(d, runid + ".log")
    logger = Logger(logfile=logfile, verbosity=v)
    driver = Driver("Continuum", path=path, path_input="function",
                    num_steps=200, cfmt="222", functions=f2,
                    termination_time=1.8*pi, logger=logger)
    constants = [E, Nu]
    material = Material("umat", parameters=constants, constants=2,
                        source_files=["neohooke.f90"], #rebuild=test,
                        source_directory="{0}/materials/abaumats".format(ROOT_D),
                        logger=logger)
    mps = MaterialPointSimulator(runid, driver, material, logger=logger, d=d)
    mps.run()

@matmodlab
def run_uhyper(d=None, v=1, test=0):
    d = d or os.getcwd()
    runid = uhyper
    logfile = os.path.join(d, runid + ".log")
    logger = Logger(logfile=logfile, verbosity=v)
    driver = Driver("Continuum", path=path, path_input="function",
                    num_steps=200, cfmt="222", functions=f2,
                    termination_time=1.8*pi, logger=logger)
    constants = [C10, D1]
    material = Material("uhyper", parameters=constants, constants=2,
                        source_files=["uhyper.f90"], #rebuild=test,
                        source_directory="{0}/materials/abaumats".format(ROOT_D),
                        logger=logger)
    mps = MaterialPointSimulator(runid, driver, material, logger=logger, d=d)
    mps.run()

@matmodlab
def run_uanisohyper_inv(d=None, v=1, test=0):
    d = d or os.getcwd()
    runid = "uanisohyper_inv"
    logfile = os.path.join(d, runid + ".log")
    logger = Logger(logfile=logfile, verbosity=v)
    driver = Driver("Continuum", path=path, path_input="function",
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
    material = Material("uanisohyper_inv", parameters=parameters, constants=5,
                        source_files=["uanisohyper_inv.f"], #rebuild=test,
                        source_directory="{0}/materials/abaumats".format(ROOT_D),
                        logger=logger, fiber_dirs=a)
    mps = MaterialPointSimulator(runid, driver, material, logger=logger, d=d)
    mps.run()

if __name__ == "__main__":
    run_umat()
    run_uhyper()
    run_uanisohyper_inv()
