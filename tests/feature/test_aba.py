#!/usr/bin/env xpython
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
        self.keywords = ["fast", "abaqus", "umat", "neohooke", "feature"]
    def run_job(self):
        run_umat(d=self.test_dir, v=0)

class TestUHyper(TestBase):
    def __init__(self):
        self.runid = uhyper
        self.keywords = ["fast", "abaqus", "uhyper", "neohooke", "feature"]
    def run_job(self):
        run_uhyper(d=self.test_dir, v=0)

@matmodlab
def run_umat(d=None, v=1):
    d = d or os.getcwd()
    runid = umat
    logfile = os.path.join(d, runid + ".log")
    logger = Logger(logfile=logfile, verbosity=v)
    driver = Driver("Continuum", path=path, path_input="function",
                    num_steps=200, cfmt="222", functions=f2,
                    termination_time=1.8*pi, logger=logger)
    constants = [E, Nu]
    material = Material("umat", parameters=constants, constants=2,
                        source_files=["neohooke.f90"],
                        source_directory="{0}/materials/abaumats".format(ROOT_D),
                        logger=logger)
    mps = MaterialPointSimulator(runid, driver, material, logger=logger, d=d)
    mps.run()

@matmodlab
def run_uhyper(d=None, v=1):
    d = d or os.getcwd()
    runid = uhyper
    logfile = os.path.join(d, runid + ".log")
    logger = Logger(logfile=logfile, verbosity=v)
    driver = Driver("Continuum", path=path, path_input="function",
                    num_steps=200, cfmt="222", functions=f2,
                    termination_time=1.8*pi, logger=logger)
    constants = [C10, D1]
    material = Material("uhyper", parameters=constants, constants=2,
                        source_files=["uhyper.f90"],
                        source_directory="{0}/materials/abaumats".format(ROOT_D),
                        logger=logger)
    mps = MaterialPointSimulator(runid, driver, material, logger=logger, d=d)
    mps.run()

if __name__ == "__main__":
    run_umat()
    run_uhyper()
