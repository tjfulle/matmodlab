#!/usr/bin/env mmd
from matmodlab import *

class TestPathCycle(TestBase):
    def __init__(self):
        self.runid = "path_default"
        self.keywords = ["fast", "feature", "path", "builtin"]

class TestPathTable(TestBase):
    def __init__(self):
        self.runid = "path_table"
        self.keywords = ["fast", "feature", "path", "table", "builtin"]

class TestPathFunction(TestBase):
    def __init__(self):
        self.runid = "path_func"
        self.keywords = ["fast", "feature", "path", "function", "builtin"]


@matmodlab
def run_path_default(d=None, v=1):
    runid = "path_default"
    d = d or os.getcwd()
    d = os.getcwd()
    v = 1
    logger = Logger(runid, verbosity=v)

    path = """
0E+00 0 EEEEEE 0E+00 0E+00 0E+00 0E+00 0E+00 0E+00
1E+00 100 EEEEEE 1E-01 0E+00 0E+00 0E+00 0E+00 0E+00
2E+00 100 EEEEEE 0E+00 0E+00 0E+00 0E+00 0E+00 0E+00
3E+00 100 SSSEEE 2.056667E+10 9.966667E+09 9.966667E+09 0E+00 0E+00 0E+00
4E+00 100 SSSEEE 0E+00 0E+00 0E+00 0E+00 0E+00 0E+00
5E+00 100 DDDDDD 1E-01 0E+00 0E+00 0E+00 0E+00 0E+00
6E+00 100 DDDDDD -1E-01 0E+00 0E+00 0E+00 0E+00 0E+00
7E+00 100 RRRRRR 2.056667E+10 9.966667E+09 9.966667E+09 0E+00 0E+00 0E+00
8E+00 100 RRRRRR -2.056667E+10 -9.966667E+09 -9.966667E+09 0E+00 0E+00 0E+00
9E+00 100 FFFFFFFFF 1.105171E+00 0E+00 0E+00 0E+00 1E+00 0E+00 0E+00 0E+00 1E+00
1E+01 100 FFFFFFFFF 1E+00 0E+00 0E+00 0E+00 1E+00 0E+00 0E+00 0E+00 1E+00
"""

    # set up the driver
    driver = Driver("Continuum", path, kappa=0.0, amplitude=1.0,
                    rate_multiplier=1.0, step_multiplier=1.0, num_io_dumps=20,
                    estar=1.0, tstar=1.0, sstar=1.0, fstar=1.0, efstar=1.0,
                    dstar=1.0, proportional=False, termination_time=None,
                    logger=logger)

    # set up the material
    parameters = {"K":1.350E+11, "G":5.300E+10}
    material = Material("elastic", parameters, logger=logger)

    # set up and run the model
    mps = MaterialPointSimulator(runid, driver, material, logger=logger, d=d)
    mps.run()
    return


@matmodlab
def run_path_table(d=None, v=1):
    runid = "path_table"
    d = d or os.getcwd()
    d = os.getcwd()
    v = 1
    logger = Logger(runid, verbosity=v)
    path = """0E+00 0E+00 0E+00 0E+00 0E+00 0E+00 0E+00
              1E+00 1E-01 0E+00 0E+00 0E+00 0E+00 0E+00
              2E+00 0E+00 0E+00 0E+00 0E+00 0E+00 0E+00"""
    driver = Driver("Continuum", path, kappa=0.0, path_input="table",
                    step_multiplier=10.0, cfmt="222222", cols=range(7),
                    tfmt="time", logger=logger)
    parameters = {"K":1.350E+11, "G":5.300E+10}
    material = Material("elastic", parameters, logger=logger)
    mps = MaterialPointSimulator(runid, driver, material, logger=logger, d=d)
    mps.run()
    return


@matmodlab
def run_path_func(d=None, v=1):
    runid = "path_func"
    d = d or os.getcwd()
    d = os.getcwd()
    v = 1
    logger = Logger(runid, verbosity=v)

    path = """
    {0} 2:1.e-1 0 0
    """.format(2*pi)

    a = np.array([[0., 2.], [1., 3.], [2., 4.]])
    f2 = Function(2, "analytic_expression", lambda t: np.sin(t))
    f3 = Function(3, "piecewise_linear", a)
    functions = [f2, f3]

    # set up the driver
    driver = Driver("Continuum", path, path_input="function",
                    num_steps=200, termination_time=1.8*pi,
                    functions=functions, cfmt="222", logger=logger)

    # set up the material
    K = 10.e9
    G = 3.75e9
    parameters = {"K":K, "G":G}
    material = Material("elastic", parameters, logger=logger)

    # set up and run the model
    mps = MaterialPointSimulator(runid, driver, material, d=d, logger=logger)

    mps.run()

if __name__ == "__main__":
    run_path_table()
    run_path_func()
    run_path_default()
