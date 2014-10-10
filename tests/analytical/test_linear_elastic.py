#!/usr/bin/env mmd

from matmodlab import *
from utils.misc import remove
from utils.exojac.exodiff import rms_error
from core.test import PASSED, DIFFED, FAILED, DIFFTOL, FAILTOL
from core.tester import RES_MAP
import lin_elast_routines as ler

RUNID = "linear_elastic"

class TestRandomLinearElastic(TestBase):
    def __init__(self):
        self.runid = "rand_" + RUNID
        self.nruns = 10
        self.keywords = ["fast", "random", "material", "elastic", "analytic",
                         "builtin"]

    def setup(self,*args,**kwargs):
        self.make_test_dir()

    def run(self):
        logger = Logger(logfile=os.path.join(self.test_dir, self.runid + ".stat"), verbosity=0)
        logger.write("Running {0:d} realizations".format(self.nruns))

        stats = []
        for idx in range(0, self.nruns):
            runid = self.runid + "_{0}".format(idx + 1)
            logger.write("* Spawning {0}".format(runid))
            stats.append(runner(d=self.test_dir, v=0, runid=runid))
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
def runner(d=None, runid=None, v=1, test=0):

    d = d or os.getcwd()
    runid = runid or "rand_" + RUNID
    logfile = os.path.join(d, runid + ".log")
    solfile = os.path.join(d, runid + ".base_dat")
    pathfile = os.path.join(d, runid + ".path")
    logger = Logger(logfile=logfile, verbosity=v)
    VARIABLES = ["STRAIN_XX", "STRAIN_YY", "STRAIN_ZZ",
                 "STRAIN_XY", "STRAIN_YZ", "STRAIN_XZ",
                 "STRESS_XX", "STRESS_YY", "STRESS_ZZ",
                 "STRESS_XY", "STRESS_YZ", "STRESS_XZ"]

    # Set up the path and random material constants
    NU, E, K, G, LAM = ler.gen_rand_elast_params()
    analytic_response = ler.gen_analytical_response(LAM, G)

    # set up the driver
    path = []
    for row in analytic_response:
        path.append("{0} 1 222222 {1} {2} {3} {4} {5} {6}".format(*row[[0,1,2,3,4,5,6]]))
    path = "\n".join(path)

    # Write the path to a file (just in case)
    with open(pathfile, 'w') as f:
        f.write(path)
    logger.write("Path written to {0}".format(pathfile))

    # Write the analytic solution for visualization comparison
    logger.write("Writing analytical solution to {0}".format(solfile))
    with open(solfile, 'w') as f:
        headers = ["TIME"] + VARIABLES
        f.write("".join(["{0:>20s}".format(_) for _ in headers]) + "\n")
        for row in analytic_response:
            f.write("".join(["{0:20.10e}".format(_) for _ in row]) + "\n")
    logger.write("  Writing is complete")

    # set up the material and driver
    parameters = {"K": K, "G": G}
    material = Material("pyelastic", parameters, logger=logger)
    driver = Driver("Continuum", path, logger=logger)

    # Run the simulation
    mps = MaterialPointSimulator(runid, driver, material, logger=logger, d=d)
    mps.run()

    # check output with analytic
    logger.write("Comaring outputs")
    logger.write("  DIFFTOL = {0:.5e}".format(DIFFTOL))
    logger.write("  FAILTOL = {0:.5e}".format(FAILTOL))

    simulate_response = mps.extract_from_db(VARIABLES, t=1)

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


def run_biax_strain_ext_stressc(d=None, runid=None, v=1):
    d = d or os.getcwd()
    runid = runid or RUNID + "_biax_strain_ext_stressc"
    logfile = os.path.join(d, runid + ".log")
    logger = Logger(logfile=logfile, verbosity=v)

    # set up the material
    NU, E, K, G, LAM = ler.const_elast_params()
    parameters = {"K": K, "G": G}
    material = Material("pyelastic", parameters, logger=logger)

    # set up the driver
    path = """
      0 0 444444           0.0                0.0                0.0      0 0 0
      1 1 444444 11230352666.666668 11230352666.666668  7479414666.666667 0 0 0
      2 1 444444 22460705333.333336 22460705333.333336 14958829333.333334 0 0 0
      3 1 444444 11230352666.666668 11230352666.666668  7479414666.666667 0 0 0
      4 1 444444           0.0                0.0                0.0      0 0 0
      """
    driver = Driver("Continuum", path, logger=logger, step_multiplier=50)

    # set up and run the model
    mps = MaterialPointSimulator(runid, driver, material, logger=logger, d=d)
    mps.run()


def run_biax_strain_comp_stressc(d=None, runid=None, v=1):
    d = d or os.getcwd()
    runid = runid or RUNID + "_biax_strain_comp_stressc"
    logfile = os.path.join(d, runid + ".log")
    logger = Logger(logfile=logfile, verbosity=v)

    # set up the material
    NU, E, K, G, LAM = ler.const_elast_params()
    parameters = {"K": K, "G": G}
    material = Material("pyelastic", parameters, logger=logger)

    # set up the driver
    path = """
      0 0 444444            0.0                 0.0                 0.0      0 0 0
      1 1 444444 -11230352666.666668 -11230352666.666668  -7479414666.666667 0 0 0
      2 1 444444 -22460705333.333336 -22460705333.333336 -14958829333.333334 0 0 0
      3 1 444444 -11230352666.666668 -11230352666.666668  -7479414666.666667 0 0 0
      4 1 444444            0.0                 0.0                 0.0      0 0 0
      """
    driver = Driver("Continuum", path, logger=logger, step_multiplier=50)

    # set up and run the model
    mps = MaterialPointSimulator(runid, driver, material, logger=logger, d=d)
    mps.run()

def run_biax_strain_ext_strainc(d=None, runid=None, v=1):
    d = d or os.getcwd()
    runid = runid or RUNID + "_biax_strain_ext_strainc"
    logfile = os.path.join(d, runid + ".log")
    logger = Logger(logfile=logfile, verbosity=v)

    # set up the material
    NU, E, K, G, LAM = ler.const_elast_params()
    parameters = {"K": K, "G": G}
    material = Material("pyelastic", parameters, logger=logger)

    # set up the driver
    path = """
      0 0 222222 0 0 0 0 0 0
      1 1 222222 1 1 0 0 0 0
      2 1 222222 2 2 0 0 0 0
      3 1 222222 1 1 0 0 0 0
      4 1 222222 0 0 0 0 0 0
      """
    driver = Driver("Continuum", path, logger=logger, estar=.5,
                    step_multiplier=50)

    # set up and run the model
    mps = MaterialPointSimulator(runid, driver, material, logger=logger, d=d)
    mps.run()

def run_biax_strain_comp_strainc(d=None, runid=None, v=1):
    d = d or os.getcwd()
    runid = runid or RUNID + "_biax_strain_comp_strainc"
    logfile = os.path.join(d, runid + ".log")
    logger = Logger(logfile=logfile, verbosity=v)

    # set up the material
    NU, E, K, G, LAM = ler.const_elast_params()
    parameters = {"K": K, "G": G}
    material = Material("pyelastic", parameters, logger=logger)

    # set up the driver
    path = """
      0 0 222222 0 0 0 0 0 0
      1 1 222222 1 1 0 0 0 0
      2 1 222222 2 2 0 0 0 0
      3 1 222222 1 1 0 0 0 0
      4 1 222222 0 0 0 0 0 0
      """
    driver = Driver("Continuum", path, logger=logger, estar=-.5,
                    step_multiplier=50)

    # set up and run the model
    mps = MaterialPointSimulator(runid, driver, material, logger=logger, d=d)
    mps.run()


def run_uniax_strain_comp_strainc(d=None, runid=None, v=1):
    d = d or os.getcwd()
    runid = runid or RUNID + "_uniax_strain_comp_strainc"
    logfile = os.path.join(d, runid + ".log")
    logger = Logger(logfile=logfile, verbosity=v)

    # set up the material
    NU, E, K, G, LAM = ler.const_elast_params()
    parameters = {"K": K, "G": G}
    material = Material("pyelastic", parameters, logger=logger)

    # set up the driver
    path = """
      0 0 222222 0 0 0 0 0 0
      1 1 222222 1 0 0 0 0 0
      2 1 222222 2 0 0 0 0 0
      3 1 222222 1 0 0 0 0 0
      4 1 222222 0 0 0 0 0 0
      """
    driver = Driver("Continuum", path, logger=logger, estar=-.5,
                    step_multiplier=50)

    # set up and run the model
    mps = MaterialPointSimulator(runid, driver, material, logger=logger, d=d)
    mps.run()

def run_uniax_strain_ext_strainc(d=None, runid=None, v=1):
    d = d or os.getcwd()
    runid = runid or RUNID + "_uniax_strain_ext_strainc"
    logfile = os.path.join(d, runid + ".log")
    logger = Logger(logfile=logfile, verbosity=v)

    # set up the material
    NU, E, K, G, LAM = ler.const_elast_params()
    parameters = {"K": K, "G": G}
    material = Material("pyelastic", parameters, logger=logger)

    # set up the driver
    path = """
      0 0 222222 0 0 0 0 0 0
      1 1 222222 1 0 0 0 0 0
      2 1 222222 2 0 0 0 0 0
      3 1 222222 1 0 0 0 0 0
      4 1 222222 0 0 0 0 0 0
      """
    driver = Driver("Continuum", path, logger=logger, estar=.5,
                    step_multiplier=50)

    # set up and run the model
    mps = MaterialPointSimulator(runid, driver, material, logger=logger, d=d)
    mps.run()

def run_uniax_strain_comp_stressc(d=None, runid=None, v=1):
    d = d or os.getcwd()
    runid = runid or RUNID + "_uniax_strain_comp_stressc"
    logfile = os.path.join(d, runid + ".log")
    logger = Logger(logfile=logfile, verbosity=v)

    # set up the material
    NU, E, K, G, LAM = ler.const_elast_params()
    parameters = {"K": K, "G": G}
    material = Material("pyelastic", parameters, logger=logger)

    # set up the driver
    path = """
      0 0 444444            0.0                 0.0                 0.0      0 0 0
      1 1 444444  -7490645333.333334 -3739707333.3333335 -3739707333.3333335 0 0 0
      2 1 444444 -14981290666.666668 -7479414666.6666667 -7479414666.6666667 0 0 0
      3 1 444444  -7490645333.333334 -3739707333.3333335 -3739707333.3333335 0 0 0
      4 1 444444            0.0                 0.0                 0.0      0 0 0
      """
    driver = Driver("Continuum", path, logger=logger, step_multiplier=50)

    # set up and run the model
    mps = MaterialPointSimulator(runid, driver, material, logger=logger, d=d)
    mps.run()


def run_uniax_strain_ext_stressc(d=None, runid=None, v=1):
    d = d or os.getcwd()
    runid = runid or RUNID + "_uniax_strain_ext_stressc"
    logfile = os.path.join(d, runid + ".log")
    logger = Logger(logfile=logfile, verbosity=v)

    # set up the material
    NU, E, K, G, LAM = ler.const_elast_params()
    parameters = {"K": K, "G": G}
    material = Material("pyelastic", parameters, logger=logger)

    # set up the driver
    path = """
      0 0 444444           0.0               0.0                0.0       0 0 0
      1 1 444444  7490645333.333334 3739707333.3333335 3739707333.3333335 0 0 0
      2 1 444444 14981290666.666668 7479414666.6666667 7479414666.6666667 0 0 0
      3 1 444444  7490645333.333334 3739707333.3333335 3739707333.3333335 0 0 0
      4 1 444444           0.0               0.0                0.0       0 0 0
      """
    driver = Driver("Continuum", path, logger=logger, step_multiplier=50)

    # set up and run the model
    mps = MaterialPointSimulator(runid, driver, material, logger=logger, d=d)
    mps.run()

if __name__ == "__main__":
    run_biax_strain_ext_strainc()
    run_biax_strain_comp_strainc()
    run_biax_strain_ext_stressc()
    run_biax_strain_comp_stressc()
    run_uniax_strain_ext_stressc()
    run_uniax_strain_comp_stressc()
