#!/usr/bin/env mmd

from matmodlab import *
import random
from utils.misc import remove
from utils.exojac.exodiff import rms_error
from core.test import PASSED, DIFFED, FAILED, DIFFTOL, FAILTOL

RUNID = "rand_linear_elastic"

class TestRandomLinearElastic(TestBase):
    def __init__(self):
        self.runid = RUNID
        self.keywords = ["fast", "random", "material", "elastic", "analytic"]

    def setup(self, *args, **kwargs):
        pass

    def run(self):
        for I in range(10):
            runid = RUNID + "_{0}".format(I+1)
            self.status = runner(d=self.test_dir, v=0, runid=runid, test=1)
            if self.status == FAILED:
                return self.status
        return self.status

    def tear_down(self):
        if self.status != self.passed:
            return
        for f in os.listdir(self.test_dir):
            for I in range(10):
                runid = RUNID + "_{0}".format(I+1)
                if self.module in f or runid in f:
                    if f.endswith((".log", ".exo", ".pyc", ".con", ".eval")):
                        remove(os.path.join(self.test_dir, f))
        self.torn_down = 1

@matmodlab
def runner(d=None, runid=None, v=1, test=0):

    d = d or os.getcwd()
    runid = RUNID or runid
    logfile = os.path.join(d, runid + ".log")
    logger = Logger(logfile=logfile, verbosity=v)

    # Set up the path and random material constants
    nu, E, K, G, LAM = gen_elastic_params()
    analytic_response = gen_analytical_response(LAM, G)

    # generate the path (must be a string")
    path = []
    for row in analytic_response:
        path.append("{0} 1 222 {1} {2} {3}".format(*row[[0,1,2,3]]))
    path = "\n".join(path)

    # set up the driver
    driver = Driver("Continuum", path=path, logger=logger)

    # set up the material
    parameters = {"K": K, "G": G}
    material = Material("pyelastic", parameters=parameters, logger=logger)

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

def get_stress(e11, e22, e33, e12, e23, e13, LAM, G):
    #standard hooke's law
    sig11 = (2.0 * G + LAM) * e11 + LAM * (e22 + e33)
    sig22 = (2.0 * G + LAM) * e22 + LAM * (e11 + e33)
    sig33 = (2.0 * G + LAM) * e33 + LAM * (e11 + e22)
    sig12 = 2.0 * G * e12
    sig23 = 2.0 * G * e23
    sig13 = 2.0 * G * e13
    return sig11, sig22, sig33, sig12, sig23, sig13

def gen_elastic_params():
    # poisson_ratio and young's modulus
    nu = random.uniform(-1.0 + 1.0e-5, 0.5 - 1.0e-5)
    E = max(1.0, 10 ** random.uniform(0.0, 12.0))

    # K and G are used for parameterization
    K = E / (3.0 * (1.0 - 2.0 * nu))
    G = E / (2.0 * (1.0 + nu))

    # LAM is used for computation
    LAM = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))

    return nu, E, K, G, LAM

def gen_analytical_response(LAM, G):
    N = 100
    strain_fac = 0.01
    #               time   e11   e22   e33
    strain_table = [[0.0, 0.0, 0.0, 0.0]]
    for idx in range(1, 6):
        strain_table.append([float(idx), random.uniform(-strain_fac, strain_fac),
                                         random.uniform(-strain_fac, strain_fac),
                                         random.uniform(-strain_fac, strain_fac)])

    expanded = [[_] for _ in strain_table[0]]
    for idx in range(0, len(strain_table) - 1):
        for jdx in range(0, len(strain_table[0])):
            start = strain_table[idx][jdx]
            end = strain_table[idx + 1][jdx]
            expanded[jdx] = expanded[jdx] + list(np.linspace(start, end, N))[1:]

    table = []
    for idx in range(0, len(expanded[0])):
        t = expanded[0][idx]
        e1 = expanded[1][idx]
        e2 = expanded[2][idx]
        e3 = expanded[3][idx]
        sig = get_stress(e1, e2, e3, 0.0, 0.0, 0.0, LAM, G)
        sig11, sig22, sig33, sig12, sig23, sig13 = sig
        table.append([t, e1, e2, e3, sig11, sig22, sig33])
    return np.array(table)


if __name__ == "__main__":
    runner()
