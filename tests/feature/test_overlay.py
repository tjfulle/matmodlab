#!/usr/bin/env mmd
# -*- python -*-
# basic testing of post processing functionality
from matmodlab import *

runid = gen_runid()
d = get_my_directory()

class TestOverlay(TestBase):
    def __init__(self):
        self.runid = runid
        self.disabled = True
        self.keywords = ["long", "overlay", "feature", "function", "builtin"]
        self.base_res = os.path.join(d, "path_func.base_exo")

    def setup(self, *args, **kwargs):
        self.make_test_dir()

    def run(self):
        self.status = self.failed_to_run
        try:
            runner()
        except BaseException as e:
            self.status = self.failed
            self.logger.error("{0}: failed with the following "
                              "exception: {1}".format(self.runid, e.message))
            return
        try:
            self._create_overlays()
            self._no_teardown = False
            self.status = self.passed
        except BaseException as e:
            self.status = self.failed
            self.logger.error("{0}: failed with the following "
                              "exception: {1}".format(self.runid, e.message))
        return

@matmodlab
def runner():

    logger = Logger(runid)

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
    mps = MaterialPointSimulator(runid, driver, material, logger=logger)

    mps.run()

if __name__ == "__main__":
    runner()
