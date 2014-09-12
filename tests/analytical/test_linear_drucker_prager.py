#!/usr/bin/env mmd
from matmodlab import *
from utils.exojac.exodiff import rms_error
from core.test import PASSED, DIFFED, FAILED, DIFFTOL, FAILTOL

import lin_druck_prag_routines as ldpr

RUNID = "linear_drucker_prager"
my_dir = get_my_directory()


class TestSphericalLinearDruckerPrager(TestBase):
    def __init__(self):
        self.runid = RUNID + "_spherical"
        self.keywords = ["fast", "druckerprager", "material",
                         "spherical", "analytic", "builtin"]
        self.interpolate_diff = True
        self.base_res = os.path.join(my_dir, "lin_druck_prag_spher.base_dat")
        self.gen_overlay_if_fail = True
    def run_job(self):
        spherical_runner(d=self.test_dir, v=0, runid=self.runid)


class TestRandomLinearDruckerPrager(TestBase):
    def __init__(self):
        self.runid = RUNID + "_rand"
        self.keywords = ["long", "druckerprager", "material",
                         "random", "analytic", "builtin"]

    def setup(self, *args, **kwargs):
        pass

    def run(self):
        self.make_test_dir()
        for n in range(10):
            runid = RUNID + "_{0}".format(n+1)
            self.status = rand_runner(d=self.test_dir, v=0, runid=runid, test=1)
            if self.status == FAILED:
                return self.status
        return self.status


@matmodlab
def rand_runner(d=None, runid=None, v=1, test=0):

    d = d or os.getcwd()
    runid = RUNID or runid
    logfile = os.path.join(d, runid + ".log")
    logger = Logger(logfile=logfile, verbosity=v)

    # Set up the path and random material constants
    nu, E, K, G, LAM = ldpr.gen_rand_elastic_params()
    A1, A4 = ldpr.gen_rand_surface_params(K, G)

    # set up the material
    parameters = {"K": K, "G": G, "A1": A1, "A4": A4}
    material = Material("pyplastic", parameters=parameters, logger=logger)

    # set up the driver
    path, strain = ldpr.gen_path(K, G, A1, A4)
    driver = Driver("Continuum", path=path, logger=logger)

    # set up and run the model
    mps = MaterialPointSimulator(runid, driver, material, logger=logger, d=d)
    mps.run()

    if not test: return

    # check output with analytic
    variables = ["STRAIN_XX", "STRAIN_YY", "STRAIN_ZZ",
                 "STRESS_XX", "STRESS_YY", "STRESS_ZZ"]
    simulate_response = mps.extract_from_db(variables, t=1)
    analytic_response = ldpr.gen_analytic_solution(K, G, A1, A4, strain)

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
def spherical_runner(d=None, v=1, runid=None):

    d = d or os.getcwd()
    runid = runid or RUNID + "_spherical"
    logfile = os.path.join(d, runid + ".log")
    logger = Logger(logfile=logfile, verbosity=v)

    # Elastic modulii
    LAM = 1.0e9
    MU = 1.0e8
    K = LAM + 2.0 / 3.0 * MU

    # Intersects
    FAC = 1.0e6
    RINT = 1.0 * FAC
    ZINT = sqrt(2.0) * FAC

    # set up the driver
    path = ldpr.gen_spherical_path(K, MU, RINT, ZINT)
    driver = Driver("Continuum", path=path, step_multiplier=100, logger=logger)

    # set up the material
    parameters = {"K": K, "G": MU, "A1": RINT/sqrt(2.0), "A4": RINT/sqrt(6.0)/ZINT}
    material = Material("pyplastic", parameters=parameters, logger=logger)

    # set up and run the model
    mps = MaterialPointSimulator(runid, driver, material, logger=logger, d=d)
    mps.run()


if __name__ == "__main__":
    a = spherical_runner()
