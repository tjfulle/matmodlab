from matmodlab import *

temp = (75, 95)
time_f = 50
E, Nu = 500, .45

mps = MaterialPointSimulator('viscoelastic', initial_temperature=75)
parameters = [E, Nu]
prony_series =  np.array([[.35, 600.],
                          [.15, 20.],
                          [.25, 30.],
                          [.05, 40.],
                          [.05, 50.],
                          [.15, 60.]])
mat = mps.Material(UMAT, parameters, libname='umat_neohooke',
                   source_files=[join(MAT_D, 'src/umat_neohooke.f90')])

mat.Expansion(ISOTROPIC, [1.E-5])
mat.TRS(WLF, [75, 35, 50])
mat.Viscoelastic(PRONY, prony_series)
mps.MixedStep(components=(.1, 0., 0.), descriptors='ESS', increment=1.,
              temperature=75., frames=100)
mps.StrainRateStep(components=(0., 0., 0.), increment=50.,
                   temperature=95., frames=50)

mps.run()
