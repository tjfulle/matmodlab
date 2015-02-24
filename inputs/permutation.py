#!/usr/bin/env mmd
from matmodlab import *

path = """
0 0 222222 0 0 0 0 0 0
1 1 222222 1 0 0 0 0 0
2 1 222222 2 0 0 0 0 0
3 1 222222 1 0 0 0 0 0
4 1 222222 0 0 0 0 0 0
"""

def func(x, xnames, d, runid, *args):

    mps = MaterialPointSimulator(runid)

    # set up the driver
    mps.Driver("Continuum", path, estar=-.5, step_multiplier=10)

    # set up the material
    parameters = dict(zip(xnames, x))
    mps.Material("elastic", parameters)

    # set up and run the model
    mps.run()

    pres = mps.extract_from_db(["PRESSURE"])
    return np.amax(pres)

@matmodlab
def runner():
    K = PermutateVariable("K", 125e9, method="weibull", b=14, N=3)
    G = PermutateVariable("G", 45e9, method="percentage", b=10, N=3)
    xinit = [K, G]
    permutator = Permutator("permutation", func, xinit, method="zip",
                            descriptors=["MAX_PRES"], correlations=True)
    permutator.run()

runner()
