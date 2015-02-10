#!/usr/bin/env mmd
from matmodlab import *
from core.test import FAILED_TO_RUN, PASSED

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

    def setup(self, *args, **kwargs):
        self.make_test_dir()

    def run(self):
        self.status = FAILED_TO_RUN
        cwd = os.getcwd()
        os.chdir(self.test_dir)
        try:
            runner("zip", v=0)
            self.status = PASSED
        except BaseException as e:
            raise TestError(e.args[0])
        os.chdir(cwd)

class TestCombi(TestBase):
    def __init__(self):
        self.runid = "perm_combination"
        self.keywords = ["long", "correlations", "combination",
                         "permutation", "builtin", "feature", "builtin"]

    def setup(self, *args, **kwargs):
        self.make_test_dir()

    def run(self):
        self.status = FAILED_TO_RUN
        cwd = os.getcwd()
        os.chdir(self.test_dir)
        try:
            runner("combination", N=2, v=0)
            self.status = PASSED
        except BaseException as e:
            raise TestError(e.args[0])
        os.chdir(cwd)


def func(x, *args):

    runid = args[0]
    d = args[1]
    logger = Logger(runid)

    # set up the driver
    driver = Driver("Continuum", path, estar=-.5, step_multiplier=50,
                    logger=logger)

    # set up the material
    parameters = {"K": x[0], "G": x[1]}
    material = Material("elastic", parameters, logger=logger)

    # set up and run the model
    mps = MaterialPointSimulator(runid, driver, material, logger=logger)
    mps.run()
    pres = mps.extract_from_db(["PRESSURE"])

    return np.amax(pres)

@matmodlab
def runner(method, N=3, v=1):

    runid = "perm_{0}".format(method)
    K = PermutateVariable("K", 125e9, b=14, N=N, method="weibull")
    G = PermutateVariable("G", 45e9, b=10, N=N, method="percentage")
    xinit = [K, G]
    permutator = Permutator(func, xinit, runid, descriptor=["MAX_PRES"],
                            method=method, correlations=True, funcargs=[runid],
                            verbosity=v)
    permutator.run()

if __name__ == "__main__":
    runner("zip")
    #    runner("combination", N=2)
