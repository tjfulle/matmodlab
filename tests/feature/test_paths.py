#!/usr/bin/env mmd
from matmodlab import *

class PathTest(TestBase):
    def __init__(self):
        self.runid = "path_cycle"
        self.keywords = ["fast", "feature", "path", "builtin"]
    def run_job(self):
        runner(d=self.test_dir, v=0)

class PathTableTest(TestBase):
    def __init__(self):
        self.runid = "path_table"
        self.keywords = ["fast", "feature", "path", "table", "builtin"]
    def run_job(self):
        run_table(d=self.test_dir, v=0)

@matmodlab
def runner(d=None, v=1):
    d = d or os.getcwd()
    runid = "path_cycle"
    logfile = os.path.join(d, runid + ".log")
    logger = Logger(logfile=logfile, verbosity=v)

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
    driver = Driver("Continuum", path=path, kappa=0.0, amplitude=1.0,
                    rate_multiplier=1.0, step_multiplier=1.0, num_io_dumps=20,
                    estar=1.0, tstar=1.0, sstar=1.0, fstar=1.0, efstar=1.0,
                    dstar=1.0, proportional=False, termination_time=None,
                    logger=logger)

    # set up the material
    parameters = {"K":1.350E+11, "G":5.300E+10}
    material = Material("elastic", parameters=parameters, logger=logger)

    # set up and run the model
    mps = MaterialPointSimulator(runid, driver, material, logger=logger, d=d)
    mps.run()
    return

@matmodlab
def run_table(d=None, v=1):
    d = d or os.getcwd()
    runid = "path_table"
    logfile = os.path.join(d, runid + ".log")
    logger = Logger(logfile=logfile, verbosity=v)
    path = """0E+00 0E+00 0E+00 0E+00 0E+00 0E+00 0E+00
              1E+00 1E-01 0E+00 0E+00 0E+00 0E+00 0E+00
              2E+00 0E+00 0E+00 0E+00 0E+00 0E+00 0E+00"""
    driver = Driver("Continuum", path=path, kappa=0.0, path_input="table",
                    step_multiplier=10.0, cfmt="222222", cols=range(7),
                    tfmt="time", logger=logger)
    parameters = {"K":1.350E+11, "G":5.300E+10}
    material = Material("elastic", parameters=parameters, logger=logger)
    mps = MaterialPointSimulator(runid, driver, material, logger=logger, d=d)
    mps.run()
    return

if __name__ == "__main__":
    run_table()
