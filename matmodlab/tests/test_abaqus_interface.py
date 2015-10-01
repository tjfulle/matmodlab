from testconf import *
from matmodlab import *
from matmodlab.mmd.simulator import StrainStep

KW = {'disp': -1}

@pytest.mark.abaqus
class TestAbaqusModels(StandardMatmodlabTest):
    '''Test the abaqus model wrappers'''
    E = 500
    Nu = .45
    C10 = E / (4. * (1. + Nu))
    D1 = 6. * (1. - 2. * Nu) / E
    def setup(self):
        '''Set up test to allow abaqus models to run'''
        # by removing the abaqus libs, they will be rebuilt for each test
        for lib in ('umat', 'uhyper', 'uanisohyper_inv'):
            remove(join(LIB_D, lib + '.so'))

    @pytest.mark.uhyper
    def test_uhyper(self):
        V0 = ('E.XX', 'E.YY', 'E.ZZ',
              'S.XX', 'S.YY', 'S.ZZ',
              'F.XX', 'F.YY', 'F.ZZ')
        X = .1
        mps = MaterialPointSimulator('uhyper', verbosity=0, d=this_directory)
        param_names = ('C10', 'D1')
        parameters = dict(zip(param_names, (self.C10, self.D1)))
        mps.Material(UHYPER, parameters, libname='uhyper_t',
                     source_files=[join(MAT_D, 'src/uhyper_neohooke.f90')],
                     param_names=param_names)
        mps.MixedStep(components=(1,0,0), descriptors='ESS', frames=10, scale=X)
        mps.MixedStep(components=(0,0,0), descriptors='ESS', frames=10)
        mps.run()
        a = mps.get(*V0, **KW)

        # make sure the strain table was interpoloated correctly
        i = argmax(a[:,0])
        assert allclose(a[i,0], X)

        # analytic solution for uniaxial stress
        C1 = self.C10
        D1 = 1 / self.D1

        J = prod(a[i, [6,7,8]])
        L = exp(a[i,0])
        S = 2. * C1 / (J ** (5. / 3.)) * (L ** 2 - J / L)
        assert allclose(a[i,3], S)

        # analytic solution for J
        f = lambda j: D1*j**(8./3.) - D1*j**(5./3.) + C1/(3.*L)*J - C1*L**2./3.
        df = lambda j: 8./3.*D1*j**(5./3.) - 5./3.*D1*j**(2./3.) + C1/(3.*L)
        j = newton(1., f, df)
        assert allclose(J, j)
        self.completed_jobs.append(mps.job)

    @pytest.mark.umat
    def test_umat(self):
        V0 = ('E.XX', 'E.YY', 'E.ZZ',
              'S.XX', 'S.YY', 'S.ZZ',
              'F.XX', 'F.YY', 'F.ZZ')
        X = .1
        mps = MaterialPointSimulator('umat', verbosity=0, d=this_directory)
        param_names = ('E', 'Nu')
        parameters = dict(zip(param_names, (self.E, self.Nu)))
        mps.Material(UMAT, parameters, libname='umat_t',
                     source_files=[join(MAT_D, 'src/umat_neohooke.f90')],
                     param_names=param_names)
        mps.MixedStep(components=(1,0,0), descriptors='ESS', frames=10, scale=X)
        mps.MixedStep(components=(0,0,0), descriptors='ESS', frames=10)
        mps.run()
        a = mps.get(*V0, **KW)

        # make sure the strain table was interpoloated correctly
        i = argmax(a[:,0])
        assert allclose(a[i,0], X)

        # analytic solution for uniaxial stress
        C1 = self.C10
        D1 = 1 / self.D1

        J = prod(a[i, [6,7,8]])
        L = exp(a[i,0])
        S = 2. * C1 / (J ** (5. / 3.)) * (L ** 2 - J / L)
        assert allclose(a[i,3], S)

        # analytic solution for J
        f = lambda j: D1*j**(8./3.) - D1*j**(5./3.) + C1/(3.*L)*J - C1*L**2./3.
        df = lambda j: 8./3.*D1*j**(5./3.) - 5./3.*D1*j**(2./3.) + C1/(3.*L)
        j = newton(1., f, df)
        assert allclose(J, j)
        self.completed_jobs.append(mps.job)

    #@pytest.mark.uanisohyper_inv
    #@pytest.mark.skipif(True, reason='baseline not established')
    def xtest_uanisohyper_inv(self):
        mps = MaterialPointSimulator('uanisohyper_inv', verbosity=0,
                                     d=this_directory)
        C10, D, K1, K2, Kappa = 7.64, 1.e-8, 996.6, 524.6, 0.226
        parameters = np.array([C10, D, K1, K2, Kappa])
        a = np.array([[0.643055,0.76582,0.0]])
        mps.Material(UANISOHYPER_INV, parameters, fiber_dirs=a,
                     libname='uanisohyper_inv_t',
                     source_files=[join(MAT_D, 'src/uanisohyper_inv.f')])
        mps.GenSteps(StrainStep, components=(1,0,0), increment=2*pi,
                     steps=200, frames=1, scale=.1, amplitude=(np.sin,))
        mps.run(termination_time=1.8*pi)

    @pytest.mark.umat
    @pytest.mark.thermoelastic
    #@pytest.mark.skipif(True, reason='Test is incompatible with other umats')
    def test_umat_thermoelastic(self):
        # This test clashes with other umat's so it is disabled.  When I can
        # figure out how to reliably reload umats, it'll be added back.
        # Issue: once a module is loaded, I can't figure out how to delete it
        # from sys.modules and reload it reliably. This is important for umat
        # materials since many umats share the common library name lib.umat.
        # Once lib.umat is loaded for one umat, it would have to be wiped,
        # rebuilt, and reloaded for another. The problem is trivially avoided
        # by only running one umat per interpreter session. But, when testing,
        # we want to test several umats - so this isn't an option.
        E0, NU0, T0 = 29.E+06, .33, 298.E+00
        E1, NU1, T1 = 29.E+06, .33, 295.E+00
        TI, ALPHA = 298., 1.E-5
        mps = MaterialPointSimulator('umat_thermoelastic', verbosity=0,
                                     d=this_directory, initial_temperature=TI)
        parameters = np.array([E0, NU0, T0, E1, NU1, T1, ALPHA, TI])
        mps.Material(UMAT, parameters, depvar=12,
                     libname='umat_th',
                     source_files=[join(MAT_D, 'src/umat_thermoelastic.f90')])
        mps.MixedStep(components=(.2, 0, 0), descriptors='ESS',
                      temperature=500, frames=100)
        mps.run()
        T, E, S = mps.get('T', 'E.XX', 'S.XX')
	DT = T - T[0]
	EE = E - ALPHA * DT
	assert np.allclose(S, E0 * EE)
        self.completed_jobs.append(mps.job)
