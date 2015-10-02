#!/usr/bin/env mmd
from matmodlab import *

"""Permutate the bulk and shear modulus input parameters to the elastic
material model

"""

def func(x, xnames, d, job, *args):

    mps = MaterialPointSimulator(job)
    mps.StrainStep(components=(1, 0, 0), increment=1., scale=-.05, frames=10)
    mps.StrainStep(components=(2, 0, 0), increment=1., scale=-.05, frames=10)
    mps.StrainStep(components=(1, 0, 0), increment=1., scale=-.05, frames=10)
    mps.StrainStep(components=(0, 0, 0), increment=1., scale=-.05, frames=10)

    # set up the material
    parameters = dict(zip(xnames, x))
    mps.Material('elastic', parameters)

    # set up and run the model
    mps.run()

    s = mps.get('S.XX', 'S.YY', 'S.ZZ', disp=-1)
    p = -np.sum(s, axis=1) / 3.
    return np.amax(p)

def runner():
    N = 10
    K = PermutateVariable('K', 125e9, method=PERCENTAGE, b=20, N=N)
    G = PermutateVariable('G', 45e9, method=WEIBULL, b=14, N=N)
    xinit = [K, G]
    permutator = Permutator('permutate', func, xinit, method=ZIP,
                            descriptors=['MAX_PRES'], correlations=True)
    permutator.run()

runner()
