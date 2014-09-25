#!/usr/bin/env mmd
from matmodlab import *

path = """
0 0 222222 0 0 0 0 0 0
1 1 222222 1 0 0 0 0 0
2 1 222222 2 0 0 0 0 0
3 1 222222 1 0 0 0 0 0
4 1 222222 0 0 0 0 0 0
"""

class TestZip(TestBase):
    def __init__(self):
        self.runid = "perm_zip"
        self.keywords = ["long", "correlations", "zip",
                         "permutation", "builtin", "feature", "builtin"]

    def setup(self,*args,**kwargs):
        self.make_test_dir()

    def run(self):
        self.status = self.failed_to_run
        try:
            runner("zip", d=self.test_dir, v=0)
            self.status = self.passed
        except BaseException as e:
            raise TestError(e.args[0])

class TestCombi(TestBase):
    def __init__(self):
        self.runid = "perm_combination"
        self.keywords = ["long", "correlations", "combination",
                         "permutation", "builtin", "feature", "builtin"]

    def setup(self,*args,**kwargs):
        self.make_test_dir()

    def run(self):
        self.status = self.failed_to_run
        try:
            runner("combination", d=self.test_dir, v=0, N=2)
            self.status = self.passed
        except BaseException as e:
            raise TestError(e.args[0])


def func(x, *args):

    d, runid = args[:2]
    logfile = os.path.join(d, runid + ".log")
    logger = Logger(logfile=logfile, verbosity=0)

    # set up the driver
    driver = Driver("Continuum", path=path, estar=-.5, step_multiplier=1000,
                    logger=logger)

    # set up the material
    parameters = {"K": x[0], "G": x[1]}
    material = Material("elastic", parameters=parameters, logger=logger)

    # set up and run the model
    mps = MaterialPointSimulator(runid, driver, material, logger=logger, d=d)
    mps.run()
    pres = mps.extract_from_db(["PRESSURE"])

    return np.amax(pres)

@matmodlab
def runner(method, d=None, v=1, N=3):
    d = d or os.getcwd()
    runid = "perm_{0}".format(method)
    K = PermutateVariable("K", 125e9, method="weibull", arg=14, N=N)
    G = PermutateVariable("G", 45e9, method="percentage", arg=10, N=N)
    xinit = [K, G]
    permutator = Permutator(func, xinit, runid, descriptor=["MAX_PRES"],
                            method=method, correlations=True, d=d, verbosity=v,
                            funcargs=[runid])
    permutator.run()

if __name__ == "__main__":
    runner("zip")
    runner("combination", N=2)
