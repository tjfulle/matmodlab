import os
import sys
import time
import numpy as np

from core.runtime import opts
from core.logger import Logger
from utils.exomgr import ExodusII
from core.driver import PathDriver
from core.product import MAT_LIB_DIRS
from core.material import MaterialModel
from utils.data_containers import DataContainer
from utils.variable import Variable, VAR_SCALAR

class MaterialPointSimulator(object):
    def __init__(self, runid, driver, material, termination_time=None,
                 verbosity=1, d=None, logger=None):
        """Initialize the MaterialPointSimulator object

        """
        self._vars = []
        self.runid = runid
        self.termination_time = termination_time

        # check input
        if not isinstance(driver, PathDriver):
            raise UserInputError("driver must be instance of Driver")
        self.driver = driver
        if not isinstance(material, MaterialModel):
            raise UserInputError("material must be instance of Material")
        self.material = material

        # setup IO
        run_dir = d or os.getcwd()
	self.title = "matmodlab single element simulation"
        if logger is None:
            logfile = os.path.join(run_dir, self.runid + ".log")
            logger = Logger(logfile=logfile, verbosity=verbosity)
        self.logger = logger

        material.logger = self.logger
        driver.logger = self.logger

        self.exo_db = ExodusII(self.runid, d=run_dir)
        self.exo_file = self.exo_db.filepath

	# register global variables
        self.register_variable("TIME_STEP", VAR_SCALAR)
        self.register_variable("STEP_NUM", VAR_SCALAR)
        self.register_variable("LEG_NUM", VAR_SCALAR)

        # two data arrays: global, element
        glob_data = []
        self.glob_vars = []
        for d in self.variables:
            assert len(d.keys) == len(d.initial_value)
            glob_data.append((d.name, d.keys, d.initial_value))
            self.glob_vars.extend(d.keys)
        self.glob_data = DataContainer(glob_data)

        elem_data = []
        self.elem_vars = []
        for item in (self.driver.variables, self.material.variables):
            for d in item:
                assert len(d.keys) == len(d.initial_value)
                self.elem_vars.extend(d.keys)
                elem_data.append((d.name, d.keys, d.initial_value))
        self.elem_data = DataContainer(elem_data)

        # set up timing
        self.timing = {}
        self.timing["start"] = time.time()

        self.setup_io()
        self.write_summary()
        pass

    def register_variable(self, var_name, var_type):
        self._vars.append(Variable(var_name, var_type))

    @property
    def logger(self):
        return self._logger

    @logger.setter
    def logger(self, value):
        self._logger = value

    @property
    def variables(self):
	return self._vars

    def write_summary(self):
        s = "\n   ".join("{0}".format(x) for x in MAT_LIB_DIRS)
        summary = """
SIMULATION SUMMARY
---------- -------
MATERIAL SEARCH DIRECTORIES:
   {6}
MATERIAL INTERFACE FILE:
   {7}
RUNID: {0}
DRIVER: {1}
  NUMBER OF LEGS: {2}
  TOTAL NUMBER OF STEPS: {8}
MATERIAL: {3}
  NUMBER OF PROPS: {4}
    NUMBER OF SDV: {5}
""".format(self.runid, self.driver.kind, self.driver.num_leg,
           self.material.name, self.material.num_prop, self.material.num_xtra,
           s, self.material.file, self.driver.num_steps)
        self.logger.write(summary, transform=str)

    def setup_io(self):

        self.exo_db.put_init(self.glob_data.data, self.glob_vars,
                             self.elem_data.data, self.elem_vars, title=self.title)

        # Write info to log file
        L = max(max(len(n) for n in self.elem_vars), 10)
        param_names = self.material.parameter_names
        iparam_vals = self.material.initial_parameters
        param_vals = self.material.parameters

        self.logger.debug("Material Parameters")
        self.logger.debug("  {1:{0}s}  {2:12}  {3:12}".format(
            L, "Name", "iValue", "Value"))
        for p in zip(param_names, iparam_vals, param_vals):
            self.logger.debug("  {1:{0}s} {2: 12.6E} {3: 12.6E}".format(L, *p))

        # write out plotable data
        self.logger.debug("Output Variables:")
        self.logger.debug("Global")
        for item in self.glob_vars:
            self.logger.debug("  " + item)
        self.logger.debug("Element")
        for item in self.elem_vars:
            self.logger.debug("  " + item)

    def run(self):
        """Run the problem

        """
        self.logger.write("starting calculations...")
	retcode = self.driver.run(self.glob_data, self.elem_data,
                                  self.material, self.exo_db,
                                  termination_time=self.termination_time)
        self.finish()
        return retcode

    def finish(self):
        # udpate and close the file
        self.timing["end"] = time.time()
        if self.driver.ran:
            dt_run = self.timing["end"] - self.timing["start"]
            self.logger.write("...calculations completed ({0:.4f}s)".format(dt_run))
        else:
            self.logger.error("calculations did not complete", r=0)
        self.exo_db.finish()
        self.logger.finish()

        if opts.viz_on_completion:
            self.visualize_results()

        return

    def dump(self, variables=None, paths=None, format="ascii", step=1, ffmt=".18f"):
        from utils.exojac.exodump import exodump
        if variables:
            exodump(self.exo_file, step=step, ffmt=ffmt,
                    variables=variables, ofmt=format)
        if paths:
            self.driver.extract_paths(self.exo_file, paths)

    def extract_from_db(self, variables, step=1, t=0):
        from utils.exojac.exodump import read_vars_from_exofile
        data = read_vars_from_exofile(self.exo_file, variables=variables,
                                      step=step, h=0, t=t)
        return data

    def visualize_results(self, overlay=None):
        from viz.plot2d import create_model_plot
        create_model_plot(self.exo_file)

    @property
    def exodus_file(self):
        return self.exo_file
