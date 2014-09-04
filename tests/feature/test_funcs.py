#!/usr/bin/env xpython
# -*- python -*-
from matmodlab import *

runid = gen_runid()

class TestFunc(TestBase):
    runid = runid
    keywords = ["fast", "function", "feature", "elastic"]
    def run_job(self, d=None):
        runner(d=d, v=0)
        return

@matmodlab
def runner(d=None, v=1):

    d = d or os.getcwd()

    path = """
    {0} 2:1.e-1 0 0
    """.format(2*pi)

    a = np.array([[0., 2.], [1., 3.], [2., 4.]])
    f2 = Function(2, "analytic_expression", lambda t: np.sin(t))
    f3 = Function(3, "piecewise_linear", a)
    functions = [f2, f3]

    # set up the driver
    driver = Driver("Continuum", path=path, path_input="function",
                    num_steps=200, termination_time=1.8*pi,
                    functions=functions, cfmt="222")

    # set up the material
    K = 10.e9
    G = 3.75e9
    parameters = {"K":K, "G":G}
    material = Material("elastic", parameters=parameters)

    # set up and run the model
    mps = MaterialPointSimulator(runid, driver, material, d=d, verbosity=v)

    mps.run()

if __name__ == "__main__":
    runner()
