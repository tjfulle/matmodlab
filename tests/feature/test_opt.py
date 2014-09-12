#!/usr/bin/env xpython
from matmodlab import *
import opt_routines as my_opt

my_dir = get_my_directory()
path_file = os.path.join(my_dir, "opt.base_dat")

class TestCobyla(TestBase):
    def __init__(self):
        self.runid = "opt_cobyla"
        self.keywords = ["long", "cobyla", "optimization", "opt", "feature",
                         "builtin"]
    def setup(self,*args,**kwargs): pass
    def run(self):
        self.make_test_dir()
        self.status = self.failed_to_run
        try:
            runner("cobyla", d=self.test_dir, v=0)
            self.status = self.passed
        except BaseException as e:
            self.logger.error("{0}: failed with the following "
                              "exception: {1}".format(self.runid, e.message))

class TestSimplex(TestBase):
    def __init__(self):
        self.runid = "opt_simplex"
        self.keywords = ["long", "simplex", "optimization", "opt", "feature",
                         "builtin"]
    def setup(self,*args,**kwargs): pass
    def run(self):
        self.make_test_dir()
        self.status = self.failed_to_run
        try:
            runner("simplex", d=self.test_dir, v=0)
            self.status = self.passed
        except BaseException as e:
            self.logger.error("{0}: failed with the following "
                              "exception: {1}".format(self.runid, e.message))

class TestPowell(TestBase):
    def __init__(self):
        self.disabled = True
        self.runid = "opt_powell"
        self.keywords = ["long", "powell", "optimization", "opt", "feature",
                         "builtin"]
    def setup(self,*args,**kwargs): pass
    def run(self):
        self.make_test_dir()
        self.status = self.failed_to_run
        try:
            runner("powell", d=self.test_dir, v=0)
            self.status = self.passed
        except BaseException as e:
            self.logger.error("{0}: failed with the following "
                              "exception: {1}".format(self.runid, e.message))

def func(x, *args):

    evald, runid = args[:2]

    logfile = os.path.join(evald, runid + ".log")
    logger = Logger(logfile=logfile, verbosity=0)

    # set up driver
    driver = Driver("Continuum", path_file=path_file, cols=[0,2,3,4],
                    cfmt="222", tfmt="time", path_input="table", logger=logger)

    # set up material
    parameters = {"K": x[0], "G": x[1]}
    material = Material("elastic", parameters=parameters, logger=logger)

    # set up and run the model
    mps = MaterialPointSimulator(runid, driver, material, d=evald, logger=logger)
    mps.run()

    error = my_opt.opt_sig_v_time(mps.exodus_file)
    return error

@matmodlab
def runner(method, d=None, v=1):

    d = d or os.getcwd()
    runid = "opt_{0}".format(method)

    # run the optimization job.
    # the optimizer expects:
    #    1) A list of OptimizeVariable to optimize
    #    2) An objective function -> a MaterialPointSimulator simulation
    #       that returns some error measure
    #    3) A method
    # It's that simple!

    K = OptimizeVariable("K", 129e9, bounds=(125e9, 150e9))
    G = OptimizeVariable("G", 54e9, bounds=(45e9, 57e9))
    xinit = [K, G]

    optimizer = Optimizer(func, xinit, runid, d=d,
                          descriptor=["PRES_V_EVOL"], method=method,
                          maxiter=25, tolerance=1.e-4, verbosity=v,
                          funcargs=[runid])
    optimizer.run()
    return

if __name__ == "__main__":
    runner("simplex")
