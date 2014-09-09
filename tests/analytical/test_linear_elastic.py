#!/usr/bin/env mmd

from matmodlab import *
from utils.misc import remove
from utils.exojac.exodiff import rms_error
from core.test import PASSED, DIFFED, FAILED, DIFFTOL, FAILTOL
import lin_elast_routines as ler

RUNID = "linear_elastic"

class TestRandomLinearElastic(TestBase):
    def __init__(self):
        self.runid = "rand_" + RUNID
        self.keywords = ["fast", "random", "material", "elastic", "analytic"]

    def setup(self, *args, **kwargs):
        pass

    def run(self):
        for I in range(10):
            runid = self.runid + "_{0}".format(I+1)
            self.status = runner(d=self.test_dir, v=0, runid=runid, test=1)
            if self.status == FAILED:
                return self.status
        return self.status


@matmodlab
def runner(d=None, runid=None, v=1, test=0):

    d = d or os.getcwd()
    runid = "rand_" + RUNID or runid
    logfile = os.path.join(d, runid + ".log")
    logger = Logger(logfile=logfile, verbosity=v)

    # Set up the path and random material constants
    nu, E, K, G, LAM = ler.gen_rand_elast_params()
    analytic_response = ler.gen_analytical_response(LAM, G)

    # set up the driver
    path = []
    for row in analytic_response:
        path.append("{0} 1 222 {1} {2} {3}".format(*row[[0,1,2,3]]))
    path = "\n".join(path)
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

if __name__ == "__main__":
    runner()
