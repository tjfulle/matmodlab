#!/usr/bin/env mmd
from matmodlab import *
from utils.exojac.exodiff import rms_error
from core.test import PASSED, DIFFED, FAILED, DIFFTOL, FAILTOL
from core.tester import RES_MAP

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
        self.nruns = 10
        self.keywords = ["long", "druckerprager", "material",
                         "random", "analytic", "builtin"]

    def setup(self, *args, **kwargs):
        self.make_test_dir()

    def run(self):
        logger = Logger(logfile=os.path.join(self.test_dir, self.runid + ".stat"), verbosity=0)
        logger.write("Running {0:d} realizations".format(self.nruns))

        stats = []
        for idx in range(0, self.nruns):
            runid = self.runid + "_{0}".format(idx + 1)
            logger.write("* Spawning {0}".format(runid))
            stats.append(rand_runner(d=self.test_dir, v=0, runid=runid, test=1))
            logger.write("    Status: {0:s}".format(RES_MAP[stats[-1]]))

        # Set the overall status (lowest common denominator)
        self.status = PASSED
        if FAILED in stats:
            self.status = FAILED
        elif DIFFED in stats:
            self.status = DIFFED

        logger.write("Overall test status: {0:s}".format(RES_MAP[self.status]))
        return self.status


@matmodlab
def rand_runner(d=None, runid=None, v=1, test=0):

    d = d or os.getcwd()
    runid = runid or RUNID
    logfile = os.path.join(d, runid + ".log")
    solfile = os.path.join(d, runid + ".base_dat")
    logger = Logger(logfile=logfile, verbosity=v)
    logger.write("logfile is: {0}".format(logfile))

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
    logger.write("Comaring outputs")
    logger.write("  DIFFTOL = {0:.5e}".format(DIFFTOL))
    logger.write("  FAILTOL = {0:.5e}".format(FAILTOL))

    # check output with analytic
    VARIABLES = ["STRAIN_XX", "STRAIN_YY", "STRAIN_ZZ",
                 "STRESS_XX", "STRESS_YY", "STRESS_ZZ"]
    simulate_response = mps.extract_from_db(VARIABLES, t=1)
    analytic_response = ldpr.gen_analytic_solution(K, G, A1, A4, strain)

    logger.write("Writing analytical solution to {0}".format(solfile))
    with open(solfile, 'w') as f:
        headers = ["TIME"] + VARIABLES
        f.write("".join(["{0:>20s}".format(_) for _ in headers]) + "\n")
        for row in analytic_response:
            f.write("".join(["{0:20.10e}".format(_) for _ in row]) + "\n")
    logger.write("  Writing is complete")

    T = analytic_response[:, 0]
    t = simulate_response[:, 0]

    stat = PASSED
    for col in range(1, 1 + len(VARIABLES)):
        X = analytic_response[:, col]
        x = simulate_response[:, col]
        nrms = rms_error(T, X, t, x, disp=0)
        logger.write("  {0:s} NRMS = {1:.5e}".format(VARIABLES[col-1], nrms))
        if nrms < DIFFTOL:
            logger.write("    PASS")
            continue
        elif nrms < FAILTOL and stat is PASSED:
            logger.write("    DIFF")
            stat = DIFFED
        else:
            logger.write("    FAIL")
            stat = FAILED

    return stat


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
