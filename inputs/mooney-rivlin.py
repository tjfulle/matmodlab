#!/usr/bin/env mmd
from matmodlab import *

eps = 6.9314718055994529E-01
path = """
0  1 222    0 0 0
1 10 244  {E} 0 0
2 10 244    0 0 0
3 10 244 -{E} 0 0
""".format(E=eps)

@matmodlab
def runner():
    mps = MaterialPointSimulator("mooney-rivlin")
    mps.Driver("Continuum", path, step_multiplier=100)
    parameters = {"C10": 72, "C01": 7.56, "NU": .49}
    mps.Material("mooney_rivlin", parameters)
    mps.run()

runner()
