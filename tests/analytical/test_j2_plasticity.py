#!/usr/bin/env mmd

from matmodlab import *
import random
from utils.misc import remove
from utils.exojac.exodiff import rms_error
from core.test import PASSED, DIFFED, FAILED

FAILTOL = 1.E-02
DIFFTOL = 5.E-03

my_dir = get_my_directory()

RUNID = "j2_plasticity"

class TestJ2Plasticity1(TestBase):
    def __init__(self):
        self.runid = RUNID + "1"
        self.keywords = ["fast", "material", "vonmises", "analytic", "material"]
        self.base_res = os.path.join(my_dir, "j2_plast.base_dat")
        self.interpolate_diff = True
    def run_job(self):
        runner1(d=self.test_dir, v=0, runid=self.runid)

class TestJ2PlasticityIsotropicHardening(TestBase):
    def __init__(self):
        self.runid = RUNID + "_iso_hard"
        self.keywords = ["fast", "material", "vonmises", "analytic", "material",
                         "hardening"]
        self.base_res = os.path.join(my_dir, "j2_plast_iso_hard.base_dat")
        self.interpolate_diff = True
        self.gen_overlay_if_fail = True
    def run_job(self):
        iso_hard_runner(d=self.test_dir, v=0, runid=self.runid)

class TestJ2PlasticityKinematicHardening(TestBase):
    def __init__(self):
        self.runid = RUNID + "_kin_hard"
        self.keywords = ["fast", "material", "vonmises", "analytic", "material",
                         "hardening"]
        self.base_res = os.path.join(my_dir, "j2_plast_kin_hard.base_dat")
        self.interpolate_diff = True
        self.gen_overlay_if_fail = True
    def run_job(self):
        kin_hard_runner(d=self.test_dir, v=0, runid=self.runid)

class TestJ2PlasticityMixedHardening(TestBase):
    def __init__(self):
        self.runid = RUNID + "_mix_hard"
        self.keywords = ["fast", "material", "vonmises", "analytic", "material",
                         "hardening"]
        self.base_res = os.path.join(my_dir, "j2_plast_mix_hard.base_dat")
        self.interpolate_diff = True
        self.gen_overlay_if_fail = True
    def run_job(self):
        mix_hard_runner(d=self.test_dir, v=0, runid=self.runid)

class TestRandomJ2Plasticity1(TestBase):
    def __init__(self):
        self.runid = "rand_" + RUNID
        self.keywords = ["fast", "random", "material", "vonmises", "analytic"]

    def setup(self, *args, **kwargs):
        pass

    def run(self):
        for I in range(10):
            runid = self.runid + "_{0}".format(I+1)
            self.status = rand_runner1(d=self.test_dir, v=0, runid=runid, test=1)
            if self.status == FAILED:
                return self.status
        return self.status


class TestRandomJ2Plasticity2(TestBase):
    def __init__(self):
        self.runid = "rand_" + RUNID + "2"
        self.keywords = ["long", "random", "material", "vonmises", "analytic"]

    def setup(self, *args, **kwargs):
        pass

    def run(self):
        for I in range(10):
            runid = self.runid + "_{0}".format(I+1)
            self.status = rand_runner2(d=self.test_dir, v=0, runid=runid, test=1)
            if self.status == FAILED:
                return self.status
        return self.status


@matmodlab
def rand_runner1(d=None, runid=None, v=1, test=0):

    d = d or os.getcwd()
    runid = runid or "rand_" + RUNID
    logfile = os.path.join(d, runid + ".log")
    logger = Logger(logfile=logfile, verbosity=v)

    # Set up the path and random material constants
    nu, E, K, G, LAM = gen_random_params()
    analytic_response, Y0 = analytic_response1(nu, E, K, G, LAM)

    # generate the path (must be a string")
    path = []
    for row in analytic_response:
        path.append("{0} 1 222 {1} {2} {3}".format(*row[[0,1,2,3]]))
    path = "\n".join(path)

    # set up the driver
    driver = Driver("Continuum", path=path, logger=logger)

    # set up the material
    parameters = {"K": K, "G": G, "Y0": Y0, "H": 0., "BETA": 0.}
    material = Material("vonmises", parameters=parameters, logger=logger)

    # set up and run the model
    mps = MaterialPointSimulator(runid, driver, material, logger=logger, d=d)
    mps.run()

    if not test: return

    # check output with analytic
    variables = ["STRAIN_XX", "STRAIN_YY", "STRAIN_ZZ",
                 "STRESS_XX", "STRESS_YY", "STRESS_ZZ"]
    simulate_response = mps.extract_from_db(variables, t=1)

    T = analytic_response[:, 0]
    t = simulate_response[:, 0]
    nrms = -1
    for col in range(1,7):
        X = analytic_response[:, col]
        x = simulate_response[:, col]
        nrms = max(nrms, rms_error(T, X, t, x, disp=0))
        if nrms < DIFFTOL:
            continue
        elif nrms < FAILTOL:
            return DIFFED
        else:
            return FAILED
    return PASSED

@matmodlab
def rand_runner2(d=None, runid=None, v=1, test=0):

    d = d or os.getcwd()
    runid = runid or "rand_" + RUNID + "2"
    logfile = os.path.join(d, runid + ".log")
    logger = Logger(logfile=logfile, verbosity=v)

    # Set up the path and random material constants
    K = 150.0e9
    G = 100.0e9
    Y0 = 40.0e6

    LAM = K - 2.0 * G / 3.0
    E = 9.0 * K * G / (3.0 * K + G)
    NU = (3.0 * K - 2.0 * G) / (6.0 * K + 2.0 * G)

    e0 = np.zeros(3)

    # Here we generate a random unit vector that will not be too close to
    # the hydrostat. This is accomplished by generating a random unit vector
    # in 3D space but require it to be at least theta_cutoff degrees from
    # the z-axis. We then rotate the vectors such that the old z-axis is now
    # the hydrostatic axis.
    #
    # We then compute the required magnitude of this strain so that the yield
    # surface is hit exactly halfway through the first leg.

    # random generation
    theta_cutoff = np.radians(30.0)
    theta_range = np.cos(theta_cutoff)
    theta = np.arccos(theta_range * (1.0 - 2.0 * random.random()))
    phi = 2.0 * np.pi * random.random()
    e1 = np.array([np.sin(theta) * np.cos(phi),
                   np.sin(theta) * np.sin(phi),
                   np.cos(theta)])

    # rotation
    e1 = np.dot(rotation_matrix(np.array([-1.0, 1.0, 0.0]),
                                np.arccos(1.0 / np.sqrt(3.0))), e1)

    # scale for desired intersection with yield surface
    e1dev = e1 - e1.sum() / 3.0 * np.ones(3)
    fac = 2.0 * (Y0 / np.sqrt(12.0 * G ** 2 * np.dot(e1dev, e1dev) / 2.0))
    e1 = fac * e1
    e1vol = e1.sum() / 3.0 * np.ones(3)
    e1dev = e1 - e1vol


    # e2 is just a rotation (of random magnitude not greater than 90degrees) of
    # the deviatoric part of e1 around the hydrostat while maintaining e1's
    # hydrostatic strain rate.
    e2_rot = np.radians(random.uniform(-60.0, 60.0))
    e2 = np.dot(rotation_matrix(np.ones(3), e2_rot), e1dev) #+ e1vol

    # generate the path (must be a string")
    p = []
    p.append("0 0 222 {0} {1} {2}".format(e0[0], e0[1], e0[2]))
    p.append("1 1 222 {0} {1} {2}".format(e1[0], e1[1], e1[2]))
    p.append("2 100 222 {0} {1} {2}".format(e1[0]+e2[0], e1[1]+e2[1], e1[2]+e2[2]))
    path = "\n".join(p)

    # analytic solution
    analytic_response = []
    for t in np.linspace(0, 2, 1000+1):
        if 0.0 <= t <= 1.0:
            fac = t
            cur_strain = (1.0 - fac) * e0 + fac * e1
        else:
            fac = t - 1.0
            cur_strain = (1.0 - fac) * e1 + fac * (e1 + e2)
        tmp = get_stress2(K, G, Y0, e1, e2, t)
        analytic_response.append([t, cur_strain[0], cur_strain[1], cur_strain[2],
                                  tmp[0], tmp[1], tmp[2]])
    analytic_response = np.array(analytic_response)

    # set up the driver
    driver = Driver("Continuum", path=path, logger=logger,
                    num_io_dumps=100, step_multiplier=100)

    # set up the material
    parameters = {"K": K, "G": G, "Y0": Y0, "H": 0., "BETA": 0.}
    material = Material("vonmises", parameters=parameters, logger=logger)

    # set up and run the model
    mps = MaterialPointSimulator(runid, driver, material, logger=logger, d=d)
    mps.run()

    test  =1
    if not test: return

    # check output with analytic
    variables = ["STRAIN_XX", "STRAIN_YY", "STRAIN_ZZ",
                 "STRESS_XX", "STRESS_YY", "STRESS_ZZ"]
    simulate_response = mps.extract_from_db(variables, t=1)

    T = analytic_response[:, 0]
    t = simulate_response[:, 0]
    nrms = -1
    for col in range(1,7):
        X = analytic_response[:, col]
        x = simulate_response[:, col]
        nrms = max(nrms, rms_error(T, X, t, x, disp=0))
        if nrms < DIFFTOL:
            continue
        elif nrms < FAILTOL:
            return DIFFED
        else:
            return FAILED
    return PASSED

@matmodlab
def runner1(d=None, v=1, runid=None):

    d = d or os.getcwd()
    runid = runid or RUNID + "1"
    logfile = os.path.join(d, runid + ".log")
    logger = Logger(logfile=logfile, verbosity=v)

    NU, E, K, G, LAM, Y = copper_params()

    # no hardening
    YF = Y
    H = 0
    BETA = 0

    EPSY = Y / (2.0 * G) #  axial strain at yield
    SIGAY = (2.0 * G + LAM) * EPSY #  axial stress at yield
    SIGLY = LAM * EPSY #  lateral stress at yield

    EPSY2 = 2.0 * Y / (2.0 * G) #  final axial strain
    SIGAY2 = ((2 * G + 3 * LAM) * EPSY2 - YF ) / 3.0 + YF # final axial stress
    SIGLY2 = ((2 * G + 3 * LAM) * EPSY2 - YF ) / 3.0 # final lateral stress

    path = """
    0 0 222222 0       0 0 0 0 0
    1 1 222222 {EPSY}  0 0 0 0 0
    2 1 222222 {EPSY2} 0 0 0 0 0
    """.format(EPSY=EPSY, EPSY2=EPSY2)

    # set up the driver
    driver = Driver("Continuum", path=path, step_multiplier=200, logger=logger)

    # set up the material
    parameters = {"K": K, "G": G, "Y0": YF, "H": H, "BETA": 0}
    material = Material("vonmises", parameters=parameters, logger=logger)

    # set up and run the model
    mps = MaterialPointSimulator(runid, driver, material, logger=logger, d=d)
    mps.run()

@matmodlab
def iso_hard_runner(d=None, v=1, runid=None):

    d = d or os.getcwd()
    runid = runid or RUNID + "_iso_hard"
    logfile = os.path.join(d, runid + ".log")
    logger = Logger(logfile=logfile, verbosity=v)

    NU, E, K, G, LAM, Y = copper_params()

    # Isotropic hardening
    HFAC = 1.0 / 10.0
    H = 3.0 * HFAC / (1.0 - HFAC) * G
    YF = Y * (1.0 + HFAC)
    EPSY = Y / (2.0 * G)
    SIGAY = (2.0 * G + LAM) * EPSY
    SIGLY = LAM * EPSY
    EPSY2 = 2.0 * Y / (2.0 * G)
    SIGAY2 = ((2 * G + 3 * LAM) * EPSY2 - YF ) / 3.0 + YF
    SIGLY2 = ((2 * G + 3 * LAM) * EPSY2 - YF ) / 3.0
    BETA = 0

    path = """
    0 0 222222 0       0 0 0 0 0
    1 1 222222 {EPSY}  0 0 0 0 0
    2 1 222222 {EPSY2} 0 0 0 0 0
    """.format(EPSY=EPSY, EPSY2=EPSY2)

    # set up the driver
    driver = Driver("Continuum", path=path, step_multiplier=200, logger=logger)

    # set up the material
    parameters = {"K": K, "G": G, "Y0": Y, "H": H, "BETA": BETA}
    material = Material("vonmises", parameters=parameters, logger=logger)

    # set up and run the model
    mps = MaterialPointSimulator(runid, driver, material, logger=logger, d=d)
    mps.run()


@matmodlab
def kin_hard_runner(d=None, v=1, runid=None):

    d = d or os.getcwd()
    runid = runid or RUNID + "_kin_hard"
    logfile = os.path.join(d, runid + ".log")
    logger = Logger(logfile=logfile, verbosity=v)

    NU, E, K, G, LAM, Y = copper_params()

    # Isotropic hardening
    HFAC = 1.0 / 10.0
    H = 3.0 * HFAC / (1.0 - HFAC) * G
    YF = Y * (1.0 + HFAC)
    EPSY = Y / (2.0 * G)
    SIGAY = (2.0 * G + LAM) * EPSY
    SIGLY = LAM * EPSY
    EPSY2 = 2.0 * Y / (2.0 * G)
    SIGAY2 = ((2 * G + 3 * LAM) * EPSY2 - YF ) / 3.0 + YF
    SIGLY2 = ((2 * G + 3 * LAM) * EPSY2 - YF ) / 3.0
    BETA = 1.0

    path = """
    0 0 222222 0       0 0 0 0 0
    1 1 222222 {EPSY}  0 0 0 0 0
    2 1 222222 {EPSY2} 0 0 0 0 0
    """.format(EPSY=EPSY, EPSY2=EPSY2)

    # set up the driver
    driver = Driver("Continuum", path=path, step_multiplier=200, logger=logger)

    # set up the material
    parameters = {"K": K, "G": G, "Y0": Y, "H": H, "BETA": BETA}
    material = Material("vonmises", parameters=parameters, logger=logger)

    # set up and run the model
    mps = MaterialPointSimulator(runid, driver, material, logger=logger, d=d)
    mps.run()


@matmodlab
def mix_hard_runner(d=None, v=1, runid=None):

    d = d or os.getcwd()
    runid = runid or RUNID + "_mix_hard"
    logfile = os.path.join(d, runid + ".log")
    logger = Logger(logfile=logfile, verbosity=v)

    NU, E, K, G, LAM, Y = copper_params()

    # Mixed Hardening
    HFAC = 1.0 / 10.0
    H = 3.0 * HFAC / (1.0 - HFAC) * G
    YF = Y * (1.0 + HFAC)
    EPSY = Y / (2.0 * G)
    SIGAY = (2.0 * G + LAM) * EPSY
    SIGLY = LAM * EPSY
    EPSY2 = 2.0 * Y / (2.0 * G)
    SIGAY2 = ((2 * G + 3 * LAM) * EPSY2 - YF ) / 3.0 + YF
    SIGLY2 = ((2 * G + 3 * LAM) * EPSY2 - YF ) / 3.0
    BETA = .5

    path = """
    0 0 222222 0       0 0 0 0 0
    1 1 222222 {EPSY}  0 0 0 0 0
    2 1 222222 {EPSY2} 0 0 0 0 0
    """.format(EPSY=EPSY, EPSY2=EPSY2)

    # set up the driver
    driver = Driver("Continuum", path=path, step_multiplier=200, logger=logger)

    # set up the material
    parameters = {"K": K, "G": G, "Y0": Y, "H": H, "BETA": BETA}
    material = Material("vonmises", parameters=parameters, logger=logger)

    # set up and run the model
    mps = MaterialPointSimulator(runid, driver, material, logger=logger, d=d)
    mps.run()


def gen_random_params():
    # poisson_ratio and young's modulus
    nu = random.uniform(-1.0 + 1.0e-5, 0.5 - 1.0e-5)
    E = max(1.0, 10 ** random.uniform(0.0, 12.0))

    # K and G are used for parameterization
    K = E / (3.0 * (1.0 - 2.0 * nu))
    G = E / (2.0 * (1.0 + nu))

    # LAM is used for computation
    LAM = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))

    return nu, E, K, G, LAM


def get_rootj2(sig11, sig22, sig33, sig12, sig23, sig13):
    rootj2 = ((sig11 - sig22) ** 2 +
              (sig22 - sig33) ** 2 +
              (sig33 - sig11) ** 2 +
              6.0 * (sig12 ** 2 + sig23 ** 2 + sig13 ** 2))
    return np.sqrt(rootj2 / 6.0)


def get_stress1(e11, e22, e33, e12, e23, e13, LAM, G, rootj2lim):
    YIELDING = False
    #standard hooke's law
    sig11 = (2.0 * G + LAM) * e11 + LAM * (e22 + e33)
    sig22 = (2.0 * G + LAM) * e22 + LAM * (e11 + e33)
    sig33 = (2.0 * G + LAM) * e33 + LAM * (e11 + e22)
    sig12 = 2.0 * G * e12
    sig23 = 2.0 * G * e23
    sig13 = 2.0 * G * e13

    rootj2 = get_rootj2(sig11, sig22, sig33, sig12, sig23, sig13)
    if rootj2 > rootj2lim:
        YIELDING = True
        sigmean = (sig11 + sig22 + sig33) / 3.0
        s11 = sig11 - sigmean
        s22 = sig22 - sigmean
        s33 = sig33 - sigmean
        dev_mag = np.sqrt(sum([_ ** 2 for _ in
                               [s11, s22, s33, sig12, sig23, sig13]]))
        fac = rootj2lim / dev_mag * np.sqrt(2.0)
        sig11 = sigmean + s11 * fac
        sig22 = sigmean + s22 * fac
        sig33 = sigmean + s33 * fac
        sig12 = sig12 * fac
        sig23 = sig23 * fac
        sig13 = sig13 * fac

    rootj2 = get_rootj2(sig11, sig22, sig33, sig12, sig23, sig13)

    return (sig11, sig22, sig33, sig12, sig23, sig13), YIELDING


def analytic_response1(nu, E, K, G, LAM):
    N = 500
    epsrand = lambda: random.uniform(-0.1, 0.1)
    eps1 = epsrand()
    eps2 = epsrand()
    eps3 = epsrand()
    expanded = [list(np.linspace(0.0, 1.0, N)),         # time
                list(np.linspace(0.0, eps1, N)),   # e11
                list(np.linspace(0.0, eps2, N)),   # e22
                list(np.linspace(0.0, eps3, N))]   # e33

    counter = 0
    rootj2lim = 0.0
    args_1 = [eps1, eps2, eps3, 0.0, 0.0, 0.0, LAM, G, rootj2lim]
    args_2 = [eps1/4.0, eps2/4.0, eps3/4.0, 0.0, 0.0, 0.0, LAM, G, rootj2lim]
    while (not get_stress1(*args_1)[1] or get_stress1(*args_2)[1]):
        rootj2lim = max(1.0, 10 ** random.uniform(0.0, 12.0))
        counter += 1
        if counter > 100:
            break
        args_1[-1] = rootj2lim
        args_2[-1] = rootj2lim
        continue

    analytic_solution = []
    for idx in range(0, len(expanded[0])):
        t = expanded[0][idx]
        e1 = expanded[1][idx]
        e2 = expanded[2][idx]
        e3 = expanded[3][idx]
        sig, YIELDING = get_stress1(e1, e2, e3, 0.0, 0.0, 0.0, LAM, G, rootj2lim)
        sig11, sig22, sig33, sig12, sig23, sig13 = sig
        analytic_solution.append([t, e1, e2, e3, sig11, sig22, sig33])

    Y0 = np.sqrt(3) * rootj2lim

    return np.array(analytic_solution), Y0


def get_stress2(K, G, Y, e0, e1, t):
    e0vol = e0.sum() / 3.0 * np.ones(3)
    e0dev = e0 - e0vol

    e1vol = e1.sum() / 3.0 * np.ones(3)
    e1dev = e1 - e1vol

    # Y = sqrt(3 * J2)
    j2_of_e0 = np.dot(e0dev, e0dev) / 2.0
    final_q = 2.0 * G * np.sqrt(3.0 * j2_of_e0)

    fac = min(min(t, 1.0), Y / final_q)
    p0 = 3.0 * K * e0vol * min(t, 1.0)
    s0 = 2.0 * G * e0dev * fac
    if 0.0 <= t <= 1.0:
        return p0 + s0

    # R = sqrt(2 * J2) =  mag(s) at yield
    R = np.sqrt(2.0 / 3.0) * Y
    fac = t - 1.0

    ds = 2.0 * G * e1dev * fac
    cospsi = np.dot(s0, ds) / (np.linalg.norm(s0) * np.linalg.norm(ds))
    c = np.exp(-np.linalg.norm(ds) / R)
    beta = (1.0 - c ** 2 + (1.0 - c) ** 2 * cospsi) * R
    beta /= (2.0 * c * np.linalg.norm(ds))
    alpha = 2.0 * c / (1.0 + c ** 2 + (1.0 - c ** 2) * cospsi)

    p1 = p0 + 3.0 * K * e1vol * fac
    s1 =  alpha * (s0 + beta * ds)
    return p1 + s1


def copper_params():
    E   =   0.1100000E+12
    NU  =   0.3400000
    LAM =   0.87220149253731343283582089552238805970149253731343E+11
    G   =   0.41044776119402985074626865671641791044776119402985E+11
    K   =   0.11458333333333333333333333333333333333333333333333E+12
    USM =   0.16930970149253731343283582089552238805970149253731E+12
    Y   =   70.0E+6
    return NU, E, K, G, LAM, Y


def rotation_matrix(a, theta):
    ahat = a / np.linalg.norm(a)
    part1 = np.cos(theta) * np.eye(3)
    part2 = (1.0 - np.cos(theta)) * np.outer(ahat, ahat)
    part3 = np.sin(theta) * np.array([[0.0, -ahat[2], ahat[1]],
                                      [ahat[2], 0.0, -ahat[0]],
                                      [-ahat[1], ahat[0], 0.0]])
    return part1 + part2 + part3

if __name__ == "__main__":
    a = mix_hard_runner()
