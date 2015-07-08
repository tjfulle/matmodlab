from matmodlab import *

consts = elas(E=500., Nu=.45)
K = consts['K']
G = consts['G']

mps = MaterialPointSimulator('expansion', initial_temperature=75)
parameters = [K, G]
mat = mps.Material('elastic', parameters)
mat.Expansion(ISOTROPIC, [1.E-5])
mps.MixedStep(components=(.1,0,0), descriptors='ESS',
              temperature=75., frames=10)
mps.StrainRateStep(components=(0,0,0), temperature=300., frames=10)
mps.run()
