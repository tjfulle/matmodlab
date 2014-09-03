#!/usr/bin/env xpython
from matmodlab import *

path = """
0 0 222222 0 0 0 0 0 0
1 1 222222 1 0 0 0 0 0
2 1 222222 2 0 0 0 0 0
3 1 222222 1 0 0 0 0 0
4 1 222222 0 0 0 0 0 0
"""

def func(x, *args):

    runid = args[0]

    # set up the driver
    driver = Driver("Continuum", path=path, estar=-.5, step_multiplier=1000)

    # set up the material
    parameters = {"K": x[0], "G": x[1]}
    material = Material("elastic", parameters=parameters)

    # set up and run the model
    mps = MaterialPointSimulator(runid, driver, material, verbosity=0)
    mps.run()
    pres = mps.extract_from_db(["PRESSURE"])
    smises = mps.extract_from_db(["SMISES"])

    return np.amax(pres), np.amax(smises)

@matmodlab
def runner():

    runid = "permutation"
    K = PerturbedVariable("K", 125e9, method="weibull", arg=14, N=3)
    G = PerturbedVariable("G", 45e9, method="percentage", arg=10, N=3)
    xinit = [K, G]
    permutator = Permutator(func, xinit, runid=runid,
                            respdesc=["MAX_PRES", "MAX_SMISES"],
                            method="zip", correlations=True, funcargs=(runid,))
    permutator.run()

runner()
