#!/usr/bin/env mmd
from matmodlab import *

def fun1():
    fun = np.sin
    mps = MaterialPointSimulator("fun1")

    # set up the material
    parameters = {"K": 10.e9, "G": 3.75e9}
    mps.Material("elastic", parameters)

    # Generate steps using a function (of time) to define the first component
    # of strain. The components actually sent to the StrainStep are
    # components[i]=components[i]*amplitude[i](t), where t is the time at the
    # end of the step. If an amplitude isn't give for a component, it is
    # assumed to be a constant 1. The increment given to GenSteps is the
    # increment given for all generated steps, so the increment for individual
    # steps is increment/steps.
    mps.GenSteps(StrainStep, components=(1, 0, 0), increment=2*pi,
                 steps=200, frames=1, scale=.1, amplitude=(fun,))

    # run the model
    mps.run()

def fun2():

    mps = MaterialPointSimulator("fun2", initial_temperature=75)

    # set up the material
    parameters = {"K": 10.e9, "G": 3.75e9}
    mps.Material("elastic", parameters)

    # set up piecewise linear amplitude.
    e, tf = .1, 3.
    t = np.array([0, tf/3., 2*tf/3., tf])
    E = np.array([0, e, 0, -e])
    T = np.linspace(75, 150, len(t))
    f1 = piecewise_linear(t, E)
    temperature = piecewise_linear(t, T)
    steps = 200
    dt = tf / steps
    mps.GenSteps(MixedStep, components=(1, 0, 0), increment=tf/2.,
                 steps=steps/2, amplitude=(f1,), temperature=temperature,
                 descriptors="ESS")

    # run the model
    mps.run()

if __name__ == '__main__':
    fun2()
