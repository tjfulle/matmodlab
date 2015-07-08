from matmodlab import *
from os.path import join
from matmodlab.mmd.simulator import StrainStep

raise SystemExit('model has errors (still under development), simulation stopped')
mps = MaterialPointSimulator('uanisohyper_inv')
C10, D, K1, K2, Kappa = 7.64, 1.e-8, 996.6, 524.6, 0.226
parameters = np.array([C10, D, K1, K2, Kappa])
a = np.array([[0.643055,0.76582,0.0]])
mps.Material(UANISOHYPER_INV, parameters, fiber_dirs=a,
             source_files=[join(MAT_D, 'abaumats/uanisohyper_inv.f')],
             libname='uanisohyper_inv_t', rebuild=1)
mps.GenSteps(StrainStep, components=(1,0,0), increment=2*pi,
             steps=200, frames=1, scale=.1, amplitude=(np.sin,))
mps.run(termination_time=1.8*pi)
