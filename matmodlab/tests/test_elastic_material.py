import random
from testconf import *
from matmodlab.utils.numerix import rms_error
from matmodlab.utils.fileio import loadfile
try: import matmodlab.lib.elastic as el
except ImportError: el = None

KW = {'disp': -1}

@pytest.mark.material
@pytest.mark.elastic
class TestElasticMaterial(StandardMatmodlabTest):
    K = 9.980040E+09
    G = 3.750938E+09
    parameters = {'K': K, 'G': G}

    @pytest.mark.skipif(el is None, reason='elastic model not imported')
    def test_uniaxial_strain(self):
        pathtable = [[1.0, 0.0, 0.0],
                     [2.0, 0.0, 0.0],
                     [1.0, 0.0, 0.0],
                     [0.0, 0.0, 0.0]]
        mps = MaterialPointSimulator('elastic_unistrain',
                                     verbosity=2, d=this_directory)
        mps.Material("elastic", self.parameters)
        for c in pathtable:
            mps.StrainStep(components=c, scale=-0.5)
        mps.run()
        H = self.K + 4. / 3. * self.G
        Q = self.K - 2. / 3. * self.G
        exx, sxx, syy, szz = mps.get('E.XX', 'S.XX', 'S.YY', 'S.ZZ')
        assert np.allclose(syy, szz)
        assert np.allclose(sxx, H * exx)
        assert np.allclose(syy, Q * exx)
        self.completed_jobs.append(mps.job)

    @pytest.mark.skipif(el is None, reason='elastic model not imported')
    def test_uniaxial_stress(self):
        pathtable = [[1.0, 0.0, 0.0],
                     [2.0, 0.0, 0.0],
                     [1.0, 0.0, 0.0],
                     [0.0, 0.0, 0.0]]
        mps = MaterialPointSimulator('elastic_unistress',
                                     verbosity=0, d=this_directory)
        mps.Material('elastic', self.parameters)
        for c in pathtable:
            mps.MixedStep(components=c, frames=50, scale=-1.e6, descriptors='SSS')
        mps.run()
        exx, sxx, syy, szz = mps.get('E.XX', 'S.XX', 'S.YY', 'S.ZZ')
        E = 9. * self.K * self.G / (3. * self.K + self.G)
        assert np.allclose(syy, 0)
        assert np.allclose(szz, 0)
        diff = (sxx - E * exx) / E
        assert max(abs(diff)) < 1e-10
        self.completed_jobs.append(mps.job)

    @pytest.mark.skipif(el is None, reason='elastic model not imported')
    def test_uniaxial_strain_with_stress_control(self):
        pathtable = [[ -7490645504., -3739707392., -3739707392.],
                     [-14981291008., -7479414784., -7479414784.],
                     [ -7490645504., -3739707392., -3739707392.],
                     [           0.,           0.,           0.]]
        mps = MaterialPointSimulator('elastic_unistrain_stressc',
                                     verbosity=0, d=this_directory)
        mps.Material('elastic', self.parameters)
        for c in pathtable:
            mps.MixedStep(components=c, frames=250, descriptors='SSS')
        mps.run()
        exx, eyy, ezz, sxx = mps.get('E.XX', 'E.YY', 'E.ZZ', 'S.XX')
        assert np.allclose(eyy, 0)
        assert np.allclose(ezz, 0)
        H = self.K + 4. / 3. * self.G
        diff = (sxx - H * exx) / H
        assert max(abs(diff)) < 1e-7
        self.completed_jobs.append(mps.job)

    @pytest.mark.parametrize('realization', range(1,4))
    def test_random_linear_elastic(self, realization):
        difftol = 5.e-08
        failtol = 1.e-07
        myvars = ('Time',
                  'E.XX', 'E.YY', 'E.ZZ', 'E.XY', 'E.YZ', 'E.XZ',
                  'S.XX', 'S.YY', 'S.ZZ', 'S.XY', 'S.YZ', 'S.XZ')

        job = 'rand_linear_elastic_{0}'.format(realization)
        mps = MaterialPointSimulator(job, verbosity=0, d=this_directory)
        NU, E, K, G, LAM = gen_rand_elast_params()
        analytic = gen_analytical_response(LAM, G)
        for (i, row) in enumerate(analytic[1:], start=1):
            incr = analytic[i, 0] - analytic[i-1, 0]
            mps.StrainStep(components=row[1:7], increment=incr, frames=10)
        with open(join(this_directory, mps.job + '.dat'), 'w') as fh:
            fh.write(''.join(['{0:>20s}'.format(_) for _ in myvars]) + '\n')
            for row in analytic:
                fh.write(''.join(['{0:20.10e}'.format(_) for _ in row]) + '\n')
        parameters = {'K': K, 'G': G}
        mps.Material('pyelastic', parameters)
        mps.run()
        simulation = mps.get(*myvars, **KW)
        fh = open(join(this_directory, mps.job + '.difflog'), 'w')
        fh.write('Comaring outputs\n')
        fh.write('  DIFFTOL = {0:.5e}\n'.format(difftol))
        fh.write('  FAILTOL = {0:.5e}\n'.format(failtol))
        T = analytic[:, 0]
        t = simulation[:, 0]
        stats = []
        for col in range(1, len(myvars)):
            X = analytic[:, col]
            x = simulation[:, col]
            nrms = rms_error(T, X, t, x, disp=0)
            fh.write('  {0:s} NRMS = {1:.5e}\n'.format(myvars[col], nrms))
            if nrms < difftol:
                fh.write('    PASS\n')
                stat = 0
            elif nrms < failtol:
                fh.write('    DIFF\n')
                stat = 1
            else:
                fh.write('    FAIL\n')
                stat = (2, myvars[col])
            stats.append(stat)
        fh.close()
        print stats
        assert all([stat == 0 for stat in stats])
        self.completed_jobs.append(mps.job)

    @pytest.mark.fast
    @pytest.mark.analytic
    @pytest.mark.material
    def test_supreme(self):
        ''' This test is 'supreme' because it compares the following values
        against the analytical solution:

        * Stress
        * Strain
        * Deformation gradient
        * Symmetric part of the velocity gradient

        This is meant to be a static test for linear elasticity. It's primary
        purpose is to be THE benchmark for linear elasticity as it checks each
        component of stress/strain as well as exercises key parts of the
        driver (like how it computes inputs).

        For uniaxial strain:

            | a  0  0 |                | exp(a)  0  0 |
        e = | 0  0  0 |            U = |   0     1  0 |
            | 0  0  0 |                |   0     0  1 |

         -1  | 1/exp(a)  0  0 |    dU   da | exp(a)  0  0 |
        U  = |    0      1  0 |    -- = -- |   0     0  0 |
             |    0      0  1 |    dt   dt |   0     0  0 |

                da | 1  0  0 |
        D = L = -- | 0  0  0 |
                dt | 0  0  0 |


        For pure shear

            | 0  a  0 |         1         | exp(2a)+1  exp(2a)-1  0 |   | 0  0  0 |
        e = | a  0  0 |     U = - exp(-a) | exp(2a)-1  exp(2a)+1  0 | + | 0  0  0 |
            | 0  0  0 |         2         |     0          0      0 |   | 0  0  1 |


         -1  1 | exp(-a) + exp(a)  exp(-a) - exp(a)  0 |
        U  = - | exp(-a) - exp(a)  exp(-a) + exp(a)  0 |
             2 |         0                 0         2 |


        dU   da / | exp(a)  exp(a)  0 |     \
        -- = -- | | exp(a)  exp(a)  0 | - U |
        dt   dt \ |   0       0     1 |     /

               da | 0  1  0 |
        D = L = -- | 1  0  0 |
                dt | 0  0  0 |

        '''
        difftol = 5.e-08
        failtol = 1.e-07
        job = 'supreme_linear_elastic'
        mps = MaterialPointSimulator(job, verbosity=0, d=this_directory)

        N = 25
        solfile = join(this_directory, mps.job + '.dat')
        path, LAM, G, tablepath = generate_solution(solfile, N)
        for row in tablepath:
            mps.StrainStep(components=row, increment=1.0, frames=N)

        # set up the material
        K = LAM + 2.0 * G / 3.0
        params = {'K': K, 'G': G}
        mps.Material('pyelastic', params)

        # set up and run the model
        mps.run()

        # check output with analytic (all shared variables)
        ahead, analytic = loadfile(solfile)
        shead, simulation = mps.get(disp=1)

        tdxa = ahead.index('Time') # time index for analytic solution
        tdxs = shead.index('Time') # time index for simulate solution
        T = analytic[:, tdxa]
        t = simulation[:, tdxs]

        fh = open(join(this_directory, mps.job + '.difflog'), 'w')
        fh.write('Comparing outputs\n')
        fh.write('  DIFFTOL = {0:.5e}\n'.format(difftol))
        fh.write('  FAILTOL = {0:.5e}\n'.format(failtol))

        stats = []
        for varname in shead:
            if varname == 'Time' or varname not in ahead:
                continue
            idxa = ahead.index(varname) # variable index for analytic solution
            idxs = shead.index(varname) # variable index for simulate solution
            X = analytic[:, idxa]
            x = simulation[:, idxs]
            nrms = rms_error(T, X, t, x, disp=0)
            fh.write('  {0:s} NRMS = {1:.5e}\n'.format(varname, nrms))
            if nrms < difftol:
                fh.write('    PASS\n')
                stat = 0
            elif nrms < failtol:
                fh.write('    DIFF\n')
                stat = 1
            else:
                fh.write('    FAIL\n')
                stat = 2
            stats.append(stat)
        fh.close()
        assert all([stat == 0 for stat in stats])
        self.completed_jobs.append(mps.job)

def get_D_E_F_SIG(dadt, a, LAM, G, loc):
    # This is just an implementation of the above derivations.
    #
    # 'dadt' is the current time derivative of the strain
    # 'a' is the strain at the end of the step
    # 'LAM' and 'G' are the lame and shear modulii
    # 'loc' is the index for what's wanted (0,1) for xy

    if loc[0] == loc[1]:
        # axial
        E = np.zeros((3,3))
        E[loc] = a

        F = np.eye(3)
        F[loc] = np.exp(a)

        D = np.zeros((3,3))
        D[loc] = dadt

        SIG = LAM * a * np.eye(3)
        SIG[loc] = (LAM + 2.0 * G) * a
    else:
        # shear
        l0, l1 = loc

        E = np.zeros((3,3))
        E[l0, l1] = a
        E[l1, l0] = a

        fac = np.exp(-a) / 2.0
        F = np.eye(3)
        F[l0,l0] = fac * (np.exp(2.0 * a) + 1.0)
        F[l1,l1] = fac * (np.exp(2.0 * a) + 1.0)
        F[l0,l1] = fac * (np.exp(2.0 * a) - 1.0)
        F[l1,l0] = fac * (np.exp(2.0 * a) - 1.0)

        D = np.zeros((3,3))
        D[l0,l1] = dadt
        D[l1,l0] = dadt

        SIG = np.zeros((3,3))
        SIG[l0,l1] = 2.0 * G * a
        SIG[l1,l0] = 2.0 * G * a

    return D, E, F, SIG

def generate_solution(solfile, N):
    # solfile = filename to write analytical solution to
    # N = number of steps per leg
    a = 0.1                 # total strain increment for each leg
    LAM = 1.0e9             # Lame modulus
    G = 1.0e9               # Shear modulus
    T = [0.0]               # time
    E = [np.zeros((3,3))]   # strain
    SIG = [np.zeros((3,3))] # stress
    F = [np.eye(3)]         # deformation gradient
    D = [np.zeros((3,3))]   # symmetric part of velocity gradient

    #
    # Generate the analytical solution
    #
    # strains:    xx     yy     zz     xy     xz     yz
    for loc in [(0,0), (1,1), (2,2), (0,1), (0,2), (1,2)]:
        t0 = T[-1]
        tf = t0 + 1.0
        for idx in range(1, N+1):
            fac = float(idx) / float(N)
            ret = get_D_E_F_SIG(a, fac * a, LAM, G, loc)
            T.append(t0 + fac)
            D.append(ret[0])
            E.append(ret[1])
            F.append(ret[2])
            SIG.append(ret[3])

        for idx in range(1, N+1):
            fac = float(idx) / float(N)
            ret = get_D_E_F_SIG(-a, (1.0 - fac) * a, LAM, G, loc)
            T.append(t0 + 1.0 + fac)
            D.append(ret[0])
            E.append(ret[1])
            F.append(ret[2])
            SIG.append(ret[3])

    #
    # Write the output
    #
    headers = ['Time',
               'E.XX', 'E.YY', 'E.ZZ', 'E.XY', 'E.YZ', 'E.XZ',
               'S.XX', 'S.YY', 'S.ZZ', 'S.XY', 'S.YZ', 'S.XZ',
               'F.XX', 'F.XY', 'F.XZ',
               'F.YX', 'F.YY', 'F.YZ',
               'F.ZX', 'F.ZY', 'F.ZZ',
               'D.XX', 'D.YY', 'D.ZZ', 'D.XY', 'D.YZ', 'D.XZ']
    symlist = lambda x: [x[0,0], x[1,1], x[2,2], x[0,1], x[1,2], x[0,2]]
    matlist = lambda x: list(np.reshape(x, 9))
    fmtstr = lambda x: '{0:>25s}'.format(x)
    fmtflt = lambda x: '{0:25.15e}'.format(x)

    with open(solfile, 'w') as FOUT:
        FOUT.write(''.join(map(fmtstr, headers)) + '\n')
        for idx in range(0, len(T)):
            vals = ([T[idx]] +
                     symlist(E[idx]) +
                     symlist(SIG[idx]) +
                     matlist(F[idx]) +
                     symlist(D[idx]))
            FOUT.write(''.join(map(fmtflt, vals)) + '\n')

    #
    # Pass the relevant data so the sim can run
    #

    # inputs    xx   yy   zz   xy   yz   xz
    path = '''
    0 0 222222 0.0  0.0  0.0  0.0  0.0  0.0
    1 1 222222 {0}  0.0  0.0  0.0  0.0  0.0
    2 1 222222 0.0  0.0  0.0  0.0  0.0  0.0
    3 1 222222 0.0  {0}  0.0  0.0  0.0  0.0
    4 1 222222 0.0  0.0  0.0  0.0  0.0  0.0
    5 1 222222 0.0  0.0  {0}  0.0  0.0  0.0
    6 1 222222 0.0  0.0  0.0  0.0  0.0  0.0
    7 1 222222 0.0  0.0  0.0  {0}  0.0  0.0
    8 1 222222 0.0  0.0  0.0  0.0  0.0  0.0
    9 1 222222 0.0  0.0  0.0  0.0  0.0  {0}
   10 1 222222 0.0  0.0  0.0  0.0  0.0  0.0
   11 1 222222 0.0  0.0  0.0  0.0  {0}  0.0
   12 1 222222 0.0  0.0  0.0  0.0  0.0  0.0
    '''.format('{0:.1f}'.format(a))

    tablepath = ((  a, 0.0, 0.0, 0.0, 0.0, 0.0),
                 (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                 (0.0,   a, 0.0, 0.0, 0.0, 0.0),
                 (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                 (0.0, 0.0,   a, 0.0, 0.0, 0.0),
                 (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                 (0.0, 0.0, 0.0,   a, 0.0, 0.0),
                 (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                 (0.0, 0.0, 0.0, 0.0, 0.0,   a),
                 (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                 (0.0, 0.0, 0.0, 0.0,   a, 0.0),
                 (0.0, 0.0, 0.0, 0.0, 0.0, 0.0))

    return path, LAM, G, tablepath

def get_stress(e11, e22, e33, e12, e23, e13, LAM, G):
    #standard hooke's law
    sig11 = (2.0 * G + LAM) * e11 + LAM * (e22 + e33)
    sig22 = (2.0 * G + LAM) * e22 + LAM * (e11 + e33)
    sig33 = (2.0 * G + LAM) * e33 + LAM * (e11 + e22)
    sig12 = 2.0 * G * e12
    sig23 = 2.0 * G * e23
    sig13 = 2.0 * G * e13
    return sig11, sig22, sig33, sig12, sig23, sig13

def gen_rand_elast_params():
    # poisson_ratio and young's modulus
    nu = random.uniform(-1.0 + 1.0e-5, 0.5 - 1.0e-5)
    E = max(1.0, 10 ** random.uniform(0.0, 12.0))

    # K and G are used for parameterization
    K = E / (3.0 * (1.0 - 2.0 * nu))
    G = E / (2.0 * (1.0 + nu))

    # LAM is used for computation
    LAM = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))

    return nu, E, K, G, LAM

def const_elast_params():
    K = 9.980040E+09
    G = 3.750938E+09
    LAM = K - 2.0 / 3.0 * G
    E   = 9.0 * K * G / (3.0 * K + G)
    NU  = (3.0 * K - 2.0 * G) / (2.0 * (3.0 * K + G))
    return NU, E, K, G, LAM

def gen_analytical_response(LAM, G, nlegs=4, test_type="PRINCIPAL"):
    stiff = (LAM * np.outer(np.array([1,1,1,0,0,0]), np.array([1,1,1,0,0,0])) +
             2.0 * G * np.identity(6))

    rnd = lambda: random.uniform(-0.01, 0.01)
    table = [np.zeros(1 + 6 + 6)]
    for idx in range(1, nlegs):
        if test_type == "FULL":
            strains = np.array([rnd(), rnd(), rnd(), rnd(), rnd(), rnd()])
        elif test_type == "PRINCIPAL":
            strains = np.array([rnd(), rnd(), rnd(), 0.0, 0.0, 0.0])
        elif test_type == "UNIAXIAL":
            strains = np.array([rnd(), 0.0, 0.0, 0.0, 0.0, 0.0])
        elif test_type == "BIAXIAL":
            tmp = rnd()
            strains = np.array([tmp, tmp, 0.0, 0.0, 0.0, 0.0])
        table.append(np.hstack(([idx], strains, np.dot(stiff, strains))))

    # returns a tablewith each row comprised of
    # time=table[0], strains=table[1:7], stresses=table[7:]
    return np.array(table)
