#!/usr/bin/env xpython
from matmodlab import *

class TestElasticUnistrain(TestBase):
    def __init__(self):
        self.runid = "elastic_unistrain"
        self.keywords = ["fast", "material", "elastic", "uniaxial_strain"]
    def run_job(self):
        elastic_unistrain(d=self.test_dir, v=0)

class TestElasticUnistrainStressc(TestBase):
    def __init__(self):
        self.runid = "elastic_unistrain_stressc"
        self.keywords = ["fast", "material", "elastic", "uniaxial_strain", "stress"]
    def run_job(self):
        elastic_unistrain_stressc(d=self.test_dir, v=0)

class TestElasticUnistress(TestBase):
    def __init__(self):
        self.runid = "elastic_unistress"
        self.keywords = ["fast", "material", "elastic", "uniaxial_strain"]
    def run_job(self):
        elastic_unistress(d=self.test_dir, v=0)

K = 9.980040E+09
G = 3.750938E+09

@matmodlab
def elastic_unistrain(d=None, v=1):
    d = d or os.getcwd()
    runid = "elastic_unistrain"
    logfile = os.path.join(d, runid + ".log")
    logger = Logger(logfile=logfile, verbosity=v)
    path = """
    0 0 222222 0 0 0 0 0 0
    1 1 222222 1 0 0 0 0 0
    2 1 222222 2 0 0 0 0 0
    3 1 222222 1 0 0 0 0 0
    4 1 222222 0 0 0 0 0 0
    """

    # set up the driver
    driver = Driver("Continuum", path=path, kappa=0.0, amplitude=1.0,
                    rate_multiplier=1.0, step_multiplier=1000.0, num_io_dumps=20,
                    estar=-0.5, tstar=1.0, sstar=1.0, fstar=1.0, efstar=1.0,
                    dstar=1.0, proportional=False, termination_time=None,
                    logger=logger)

    # set up the material
    parameters = {"BMOD":K, "MU":G}
    material = Material("elastic", parameters=parameters, logger=logger)

    # set up and run the model
    mps = MaterialPointSimulator(runid, driver, material, logger=logger, d=d)
    mps.run()
    return


@matmodlab
def elastic_unistrain_stressc(d=None, v=1):
    d = d or os.getcwd()
    runid = "elastic_unistrain_stressc"
    logfile = os.path.join(d, runid + ".log")
    logger = Logger(logfile=logfile, verbosity=v)
    path = """
    0 0 222 0 0 0
    1 1 444 -7490645504 -3739707392 -3739707392
    2 1 444 -14981291008 -7479414784 -7479414784
    3 1 444 -7490645504 -3739707392 -3739707392
    4 1 444 0 0 0
    """
    driver = Driver("Continuum", path=path, kappa=0.0, amplitude=1.0,
                    rate_multiplier=1.0, step_multiplier=100.0, num_io_dumps=20,
                    estar=1.0, tstar=1.0, sstar=1.0, fstar=1.0, efstar=1.0,
                    dstar=1.0, proportional=False, termination_time=None,
                    logger=logger)
    parameters = {"K":K, "G":G}
    material = Material("elastic", parameters=parameters, logger=logger)
    mps = MaterialPointSimulator(runid, driver, material, logger=logger, d=d)
    mps.run()
    return


@matmodlab
def elastic_unistress(d=None, v=1):
    d = d or os.getcwd()
    runid = "elastic_unistress"
    logfile = os.path.join(d, runid + ".log")
    logger = Logger(logfile=logfile, verbosity=v)
    path = """
    0 0 444 0 0 0
    1 1 444 1 0 0
    2 1 444 2 0 0
    3 1 444 1 0 0
    4 1 444 0 0 0
    """
    driver = Driver(kind="Continuum", kappa=0., tstar=1., sstar=-1e-6,
                    amplitude=1., step_multiplier=1000, rate_multiplier=1.,
                    path=path, logger=logger)
    parameters = {"K": K, "G": G}
    material = Material("elastic", parameters=parameters, logger=logger)
    mps = MaterialPointSimulator(runid, driver, material, logger=logger, d=d)
    mps.run()
    return


def runner():
    elastic_unistrain()
    elastic_unistrain_stressc()
    elastic_unistress()

if __name__ == "__main__":
    runner()
