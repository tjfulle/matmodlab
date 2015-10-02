from testconf import *
from matmodlab.mmd.simulator import StrainStep
from matmodlab.utils.fileio import loadfile
try: import matmodlab.lib.elastic as el
except ImportError: el = None

@pytest.mark.fast
@pytest.mark.step_factories
@pytest.mark.skipif(el is None, reason='elastic model not imported')
class TestStepFactories(StandardMatmodlabTest):

    def test_multi_step(self):
        '''Test each *Step factory method'''
        mps = MaterialPointSimulator('multi_step', verbosity=0, d=this_directory)
        parameters = {'K':1.350E+11, 'G':5.300E+10}
        mps.Material('elastic', parameters)
        N = 20
        mps.StrainStep(increment=1., frames=N, components=(.1, 0, 0))
        mps.StrainStep(increment=1., frames=N, components=(0, 0, 0))
        mps.StressStep(increment=1., frames=N,
            components=(2.056667E+10,9.966667E+09,9.966667E+09))
        mps.StressStep(increment=1., frames=N, components=(0, 0, 0))
        mps.StrainRateStep(increment=1., frames=N, components=(.1, 0, 0))
        mps.StrainRateStep(increment=1., frames=N, components=(-.1, 0, 0))
        mps.StressRateStep(increment=1., frames=N,
            components=(2.056667E+10,9.966667E+09,9.966667E+09))
        mps.StressRateStep(increment=1., frames=N,
            components=(-2.056667E+10,-9.966667E+09,-9.966667E+09))
        mps.DefGradStep(increment=1., frames=N,
            components=(1.105171,0,0,0,1,0,0,0,1))
        mps.DefGradStep(increment=1., frames=N, components=(1,0,0,0,1,0,0,0,1))
        mps.run()
        status = self.compare_with_baseline(mps)
        assert status == 0
        self.completed_jobs.append(mps.job)

    def test_data_steps(self):
        '''Test the DataSteps factory method'''
        mps = MaterialPointSimulator('data_steps', verbosity=0, d=this_directory)
        parameters = {'K':1.350E+11, 'G':5.300E+10}
        mps.Material('elastic', parameters)
        table = '''0E+00 0E+00 0E+00 0E+00 0E+00 0E+00 0E+00
                   1E+00 1E-01 0E+00 0E+00 0E+00 0E+00 0E+00
                   2E+00 0E+00 0E+00 0E+00 0E+00 0E+00 0E+00'''
        mps.DataSteps(StringIO(table), tc=0, columns=[1,2,3,4,5,6],
                      descriptors='EEEEEE', frames=10)
        mps.run()
        status = self.compare_with_baseline(mps)
        assert status == 0
        self.completed_jobs.append(mps.job)

    def test_gen_steps(self):
        '''Test the GenSteps factory method'''
        mps = MaterialPointSimulator('gen_steps', verbosity=0, d=this_directory)
        parameters = {'K': 10.0E+09, 'G': 3.75E+09}
        mps.Material('elastic', parameters)
        mps.GenSteps(StrainStep, components=(1,0,0), increment=2*pi,
                     steps=200, frames=1, scale=.1, amplitude=(np.sin,))
        mps.run(termination_time=1.8*pi)
        status = self.compare_with_baseline(mps, adjust_n=1)
        assert status == 0
        self.completed_jobs.append(mps.job)

@pytest.mark.slow
@pytest.mark.permutate
@pytest.mark.skipif(el is None, reason='elastic model not imported')
class TestPermutation(StandardMatmodlabTest):

    @staticmethod
    def func(x, xnames, d, job, *args):
        path = '''0 0 0 0 0 0 0
                  1 1 0 0 0 0 0
                  2 2 0 0 0 0 0
                  3 1 0 0 0 0 0
                  4 0 0 0 0 0 0'''
        mps = MaterialPointSimulator(job, verbosity=0, d=d)
        parameters = dict(zip(xnames, x))
        mps.Material('elastic', parameters)
        mps.DataSteps(StringIO(path), scale=-.5, frames=5, descriptors='E'*6)
        mps.run()
        pres = -np.sum(mps.get('S.XX', 'S.YY', 'S.ZZ', disp=-1), axis=1) / 3
        return np.amax(pres)

    def test_permutate_zip(self):
        '''Test the Permutator'''
        K = PermutateVariable('K', 125e9, b=14, N=2, method=WEIBULL)
        G = PermutateVariable('G', 45e9, b=10, N=2, method=PERCENTAGE)
        permutator = Permutator('permutate_zip', self.func, [K, G], method=ZIP,
                                descriptors=['MAX_PRES'], correlations=True,
                                funcargs=['permutate_zip'], verbosity=0,
                                d=this_directory)
        try:
            # test passes if it runs
            permutator.run()
        except BaseException:
            raise Exception('permutate_zip failed to run')
        self.completed_jobs.append('permutate_zip')

    def test_permutate_combination(self):
        '''Test the Permutator'''
        K = PermutateVariable('K', 125e9, b=14, N=2, method=WEIBULL)
        G = PermutateVariable('G', 45e9, b=10, N=2, method=PERCENTAGE)
        permutator = Permutator('permutate_combination', self.func, [K, G],
                                method=COMBINATION, descriptors=['MAX_PRES'],
                                correlations=True, d=this_directory, verbosity=0,
                                funcargs=['permutate_combination'])
        try:
            # test passes if it runs
            permutator.run()
        except BaseException:
            raise Exception('permutate_combination failed to run')
        self.completed_jobs.append('permutate_combination')

@pytest.mark.slow
@pytest.mark.optimize
@pytest.mark.skipif(el is None, reason='elastic model not imported')
class TestOptimization(StandardMatmodlabTest):
    path_file = join(this_directory, "opt.base_dat")
    xact = np.array([135e9, 53e9])
    @staticmethod
    def func(x, xnames, evald, job, *args):
        mps = MaterialPointSimulator(job, verbosity=0, d=evald)
        parameters = dict(zip(xnames, x))
        mps.Material("elastic", parameters)
        mps.DataSteps(TestOptimization.path_file, tc=0,
                      columns=[2,3,4], descriptors='EEE')
        mps.run()
        error = opt_sig_v_time(mps.filename)
        #error = opt_pres_v_evol(mps.filename)
        return error

    @staticmethod
    def run_method(method):
        K = OptimizeVariable("K", 148e9, bounds=(125e9, 150e9))
        G = OptimizeVariable("G", 56e9, bounds=(45e9, 57e9))
        xinit = [K, G]
        optimizer = Optimizer(method, TestOptimization.func, xinit, method=method,
                              d=this_directory, descriptors=["SIG_V_TIME"],
                              maxiter=25, tolerance=1.e-4, verbosity=0)
        optimizer.run()
        return optimizer.xopt

    @pytest.mark.cobyla
    def test_cobyla(self):
        xopt = self.run_method(COBYLA)
        # check error
        err = (xopt - self.xact) / self.xact * 100
        err = np.sqrt(np.sum(err ** 2))
        assert err < 2.0
        self.completed_jobs.append('cobyla')

    @pytest.mark.powell
    def test_powell(self):
        xopt = self.run_method(POWELL)
        # check error
        err = (xopt - self.xact) / self.xact * 100
        err = np.sqrt(np.sum(err ** 2))
        assert err < .0002
        self.completed_jobs.append('powell')

    @pytest.mark.simplex
    def test_simplex(self):
        xopt = self.run_method(SIMPLEX)
        # check error
        err = (xopt - self.xact) / self.xact * 100
        err = np.sqrt(np.sum(err ** 2))
        assert err < .02
        self.completed_jobs.append('simplex')

def opt_pres_v_evol(outf):

    vars_to_get = ('Time', 'E.XX', 'E.YY', 'E.ZZ', 'S.XX', 'S.YY', 'S.ZZ')

    # read in baseline data
    aux = join(this_directory, 'opt.base_dat')
    auxhead, auxdat = loadfile(aux, variables=vars_to_get, disp=1)
    baseevol = auxdat[:,1] + auxdat[:,2] + auxdat[:,3]
    basepress = -(auxdat[:,4] + auxdat[:,5] + auxdat[:,6]) / 3.

    # read in output data
    head, simdat = loadfile(outf, variables=vars_to_get, disp=1)
    simevol = simdat[:,1] + simdat[:,2] + simdat[:,3]
    simpress = -(simdat[:,4] + simdat[:,5] + simdat[:,6]) / 3.

    # do the comparison
    n = auxdata[:,0].shape[0]
    t0 = max(np.amin(auxdata[:,0]), np.amin(simdat[:,0]))
    tf = min(np.amax(auxdata[:,0]), np.amax(simdat[:,0]))
    evb = lambda x: np.interp(x, auxdata[:,0], baseevol)
    evs = lambda x: np.interp(x, simdat[:,0], simevol)

    base = lambda x: np.interp(evb(x), baseevol, basepress)
    comp = lambda x: np.interp(evs(x), simevol, simpress)

    dnom = np.amax(np.abs(simpress))
    if dnom < 1.e-12: dnom = 1.

    error = np.sqrt(np.mean([((base(t) - comp(t)) / dnom) ** 2
                             for t in np.linspace(t0, tf, n)]))
    return error

def opt_sig_v_time(outf):
    vars_to_get = ('Time', 'S.XX', 'S.YY', 'S.ZZ')

    # read in baseline data
    auxf = join(this_directory, 'opt.base_dat')
    auxhead, auxdat = loadfile(auxf, variables=vars_to_get, disp=1)

    # read in output data
    simhead, simdat = loadfile(outf, variables=vars_to_get, disp=1)

    # do the comparison
    error = -1
    t0 = max(np.amin(auxdat[:,0]), np.amin(simdat[:,0]))
    tf = min(np.amax(auxdat[:,0]), np.amax(simdat[:,0]))
    n = auxdat[:,0].shape[0]
    for idx in range(1, 4):
        base = lambda x: np.interp(x, auxdat[:,0], auxdat[:,idx])
        comp = lambda y: np.interp(y, simdat[:,0], simdat[:,idx])
        dnom = np.amax(np.abs(simdat[:,idx]))
        if dnom < 1.e-12:
            dnom = 1.
        rms = np.sqrt(np.mean([((base(t) - comp(t)) / dnom) ** 2
                               for t in np.linspace(t0, tf, n)]))
        error = max(rms, error)
        continue

    return error
