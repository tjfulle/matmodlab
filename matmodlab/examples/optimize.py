#!/usr/bin/env mmd
import os
import numpy as np

from matmodlab import *
import matmodlab.utils.fileio as ufio
import matmodlab.utils.numerix.nonmonotonic as unnm

"""Optimize the values of Young's modulus, yield strength, and linear
hardening parameters for a linear hardening Von Mises material model using
data from a standard uniaxial tension test. The data represents the response
of Aluminum, alloy unkown.

The strategy is to read read data from an excel file and use the axial strain
to drive the simulation. Optimization is performed by minimizing the area
between the stress-strain curve calculated and the measured stress-strain
curve.

"""

# get the experimental stress and strain
filename = os.path.join(get_my_directory(), 'optimize.xls')
strain_exp, stress_exp = zip(*ufio.loadfile(filename, sheetname='MML', disp=0,
                                            columns=['E.XX', 'S.XX']))
exp_data = ufio.loadfile(filename, sheetname='MML', disp=0,
                         columns=['E.XX', 'S.XX'])

def func(x=[], xnames=[], evald='', job='', *args):
    mps = MaterialPointSimulator(job)

    xp = dict(zip(xnames, x))
    NU = 0.32  # poisson's ratio for aluminum
    parameters = {'K': xp['E']/3.0/(1.0-2.0*NU), 'G': xp['E']/2.0/(1.0+NU),
                  'Y0': xp['Y0'], 'H': xp['H'], 'BETA': 0.0}
    mps.Material('vonmises', parameters)

    # create steps from data. note, len(columns) below is < len(descriptors).
    # The missing columns are filled with zeros -> giving uniaxial stress in
    # this case. Declaring the steps this way does require loading the excel
    # file anew for each run
    mps.DataSteps(filename, steps=30, sheetname='MML',
                  columns=('E.XX',), descriptors='ESS')

    mps.run()
    if not mps.ran:
        return 1.0e9

    exx, sxx = mps.get('E.XX', 'S.XX')
    error = unnm.calculate_bounded_area(exp_data[:,0], exp_data[:,1], exx, sxx)
    return error

def runjob(method, v=1):
    '''Set up and run the optimization job

    '''

    # set the variables to be optimized
    E = OptimizeVariable('E',  2.0e6, bounds=(1.0e5, 1.0e7))
    Y0= OptimizeVariable('Y0', 0.3e5, bounds=(1.0e4, 1.0e6))
    H = OptimizeVariable('H',  1.0e6, bounds=(1.0e4, 1.0e7))
    xinit = [E, Y0, H]

    # the optimzer object
    optimizer = Optimizer('optimize', func, xinit, method=method,
                          maxiter=200, tolerance=1.e-3)

    # run the job
    optimizer.run()
    xopt = optimizer.xopt

    return xopt


if __name__ == '__main__':
    runjob([POWELL, SIMPLEX, COBYLA][1])
