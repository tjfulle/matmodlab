#!/usr/bin/env mmd

from matmodlab import *
from utils.misc import remove
from utils.exojac.exodiff import rms_error
from core.test import PASSED, DIFFED, FAILED

import j2_plast_routines as j2pr

FAILTOL = 1.E-02
DIFFTOL = 5.E-03

my_dir = get_my_directory()

RUNID = "j2_plasticity"

VARIABLES = ["STRAIN_XX", "STRAIN_YY", "STRAIN_ZZ",
             "STRESS_XX", "STRESS_YY", "STRESS_ZZ"]

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
        self.keywords = ["long", "random", "material", "vonmises", "analytic"]

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
    NU, E, K, G, LAM = j2pr.gen_rand_params()

    # set up the driver
    path, analytic_response, Y0 = j2pr.gen_rand_analytic_resp_1(NU, E, K, G, LAM)
    driver = Driver("Continuum", path=path, logger=logger)

    # set up the material
    parameters = {"K": K, "G": G, "Y0": Y0, "H": 0., "BETA": 0.}
    material = Material("vonmises", parameters=parameters, logger=logger)

    # set up and run the model
    mps = MaterialPointSimulator(runid, driver, material, logger=logger, d=d)
    mps.run()

    if not test: return

    # check output with analytic
    simulate_response = mps.extract_from_db(VARIABLES, t=1)

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

    # set up the driver
    path, analytic_response = j2pr.gen_rand_analytic_resp_2(NU, E, K, G, LAM, Y0)
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

    NU, E, K, G, LAM, Y = j2pr.copper_params()

    # no hardening
    YF = Y
    H = 0
    BETA = 0

    # set up the driver
    path = j2pr.gen_uniax_strain_path(Y, YF, G, LAM)
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

    NU, E, K, G, LAM, Y = j2pr.copper_params()

    # Isotropic hardening
    HFAC = 1.0 / 10.0
    H = 3.0 * HFAC / (1.0 - HFAC) * G
    YF = Y * (1.0 + HFAC)
    BETA = 0

    # set up the driver
    path = j2pr.gen_uniax_strain_path(Y, YF, G, LAM)
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

    NU, E, K, G, LAM, Y = j2pr.copper_params()

    # Isotropic hardening
    HFAC = 1.0 / 10.0
    H = 3.0 * HFAC / (1.0 - HFAC) * G
    YF = Y * (1.0 + HFAC)
    BETA = 1.0

    # set up the driver
    path = j2pr.gen_uniax_strain_path(Y, YF, G, LAM)
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

    NU, E, K, G, LAM, Y = j2pr.copper_params()

    # Mixed Hardening
    HFAC = 1.0 / 10.0
    H = 3.0 * HFAC / (1.0 - HFAC) * G
    YF = Y * (1.0 + HFAC)
    BETA = .5

    # set up the driver
    path = j2pr.gen_uniax_strain_path(Y, YF, G, LAM)
    driver = Driver("Continuum", path=path, step_multiplier=200, logger=logger)

    # set up the material
    parameters = {"K": K, "G": G, "Y0": Y, "H": H, "BETA": BETA}
    material = Material("vonmises", parameters=parameters, logger=logger)

    # set up and run the model
    mps = MaterialPointSimulator(runid, driver, material, logger=logger, d=d)
    mps.run()


if __name__ == "__main__":
    a = mix_hard_runner()
