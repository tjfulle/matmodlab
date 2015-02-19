#!/usr/bin/env mmd
from matmodlab import *

class TestElasticUnistrain(TestBase):
    def __init__(self):
        self.runid = "elastic_unistrain"
        self.interpolate_diff = True
        self.keywords = ["fast", "material", "elastic", "uniaxial_strain",
                         "builtin"]

class TestElasticUnistrainStressc(TestBase):
    def __init__(self):
        self.runid = "elastic_unistrain_stressc"
        self.interpolate_diff = True
        self.keywords = ["fast", "material", "elastic", "uniaxial_strain", "stress",
                         "builtin"]

class TestElasticUnistress(TestBase):
    def __init__(self):
        self.runid = "elastic_unistress"
        self.interpolate_diff = True
        self.keywords = ["fast", "material", "elastic", "uniaxial_strain",
                         "builtin"]

K = 9.980040E+09
G = 3.750938E+09

@matmodlab
def run_elastic_unistrain(*args, **kwargs):

    path = """
    0 0 222222 0 0 0 0 0 0
    1 1 222222 1 0 0 0 0 0
    2 1 222222 2 0 0 0 0 0
    3 1 222222 1 0 0 0 0 0
    4 1 222222 0 0 0 0 0 0
    """

    # set up the model
    mps = MaterialPointSimulator("elastic_unistrain")

    # set up the driver
    mps.Driver("Continuum", path, step_multiplier=50.0, num_io_dumps=20,
               estar=-0.5)

    # set up the material
    parameters = {"BMOD":K, "MU":G}
    mps.Material("elastic", parameters)

    # set up and run the model
    mps.run()
    return


@matmodlab
def run_elastic_unistrain_stressc(*args, **kwargs):
    path = """
    0 0 222 0 0 0
    1 1 444 -7490645504 -3739707392 -3739707392
    2 1 444 -14981291008 -7479414784 -7479414784
    3 1 444 -7490645504 -3739707392 -3739707392
    4 1 444 0 0 0
    """
    mps = MaterialPointSimulator("elastic_unistrain_stressc")
    mps.Driver("Continuum", path, step_multiplier=250.0)
    parameters = {"K":K, "G":G}
    mps.Material("elastic", parameters)
    mps.run()
    return


@matmodlab
def run_elastic_unistress(*args, **kwargs):
    path = """
    0 0 444 0 0 0
    1 1 444 1 0 0
    2 1 444 2 0 0
    3 1 444 1 0 0
    4 1 444 0 0 0
    """
    mps = MaterialPointSimulator("elastic_unistress")
    mps.Driver("Continuum", path, sstar=-1e6, step_multiplier=50)
    parameters = {"K": K, "G": G}
    mps.Material("elastic", parameters)
    mps.run()
    return


def runner():
    run_elastic_unistrain()
    run_elastic_unistrain_stressc()
    run_elastic_unistress()

if __name__ == "__main__":
    runner()
