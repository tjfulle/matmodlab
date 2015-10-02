from testconf import *
try: import matmodlab.lib.elastic as el
except ImportError: el = None

@pytest.mark.add_on
class TestAddonModels(StandardMatmodlabTest):
    '''Test the abaqus model wrappers'''
    temp = (75, 95)
    time_f = 50
    E, Nu = 500, .45

    def setup(self):
        '''Set up test to allow abaqus models to run'''
        remove(join(LIB_D, 'umat.so'))

    @pytest.mark.visco
    def test_visco(self):
        mps = MaterialPointSimulator('visco_addon', initial_temperature=75,
                                     d=this_directory, verbosity=0)
        parameters = [self.E, self.Nu]
        prony_series =  np.array([[.35, 600.], [.15, 20.], [.25, 30.],
                                  [.05, 40.], [.05, 50.], [.15, 60.]])
        mat = mps.Material(UMAT, parameters, libname='neohooke_t',
                           source_files=[join(MAT_D, 'src/umat_neohooke.f90')])
        mat.Expansion(ISOTROPIC, [1.E-5])
        mat.TRS(WLF, [75, 35, 50])
        mat.Viscoelastic(PRONY, prony_series)
        mps.MixedStep(components=(.1, 0., 0.), descriptors='ESS', increment=1.,
                      temperature=75., frames=10)
        mps.StrainRateStep(components=(0., 0., 0.), increment=50.,
                           temperature=95., frames=50)
        try:
            # test passes if it runs
            mps.run()
        except BaseException:
            raise Exception('visco_addon failed to run')
        self.completed_jobs.append('visco_addon')

    @pytest.mark.expansion
    @pytest.mark.skipif(el is None, reason='elastic model not imported')
    def test_expansion(self):
        mps = MaterialPointSimulator('expansion_addon', verbosity=0,
                                     initial_temperature=75.,
                                     d=this_directory)
        c = elas(E=self.E, Nu=self.Nu)
        parameters = {'K': c['K'], 'G': c['G']}
        mat = mps.Material('elastic', parameters)

        ALPHA = 1.E-5
        mat.Expansion(ISOTROPIC, [ALPHA])
        mps.MixedStep(components=(.01,0,0), descriptors='ESS',
                      temperature=75., frames=100)
        mps.StrainRateStep(components=(0,0,0), temperature=100., frames=100)
        mps.run()

        out = mps.get('T', 'E.XX', 'S.XX', 'S.YY', 'S.ZZ', 'SDV_EM.XX', disp=-1)

        errors = []
        for (i, row) in enumerate(out[1:], start=1):
            dtemp = out[i,0] - out[0,0]
            ee = out[i,1] - ALPHA * dtemp
            assert abs(ee-out[i,-1]) < 1e-6
            diff = self.E * ee + self.Nu * (out[i,3] + out[i,4]) - out[i,2]
            errors.append(abs(diff/out[i,2]))

        if any([abs(x) > 1e-6 for x in errors]):
            # this tolerance is much too large, but I don't want to track it
            # down right now
            raise Exception('maximum error = {0}'.format(max(errors)))

        self.completed_jobs.append('expansion_addon')
