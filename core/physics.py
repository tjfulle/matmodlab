import os
import sys
import time
import numpy as np

from __config__ import cfg
import core.io as io
from utils.exodump import exodump

class PhysicsHandler(object):

    def __init__(self, runid, verbosity, driver, mtlmdl, mtlprops,
                 extract, restart, driver_opts):
        """Initialize the PhysicsHandler object

        Parameters
        ----------
        mtlmdl : str
            Name of material model

        mtlprops : ndarray
            The (unchecked) material properties

        """
        mode = "a" if restart else "w"
        io.setup_logger(runid, verbosity, mode=mode)

        self.runid = runid

        self.driver = driver()
        self.material = (mtlmdl, mtlprops)
        self.mtlprops = np.array(mtlprops)
        self.extract = extract
        self.driver_opts = [driver_opts, restart]
        self.setup_restart = bool(restart)

        # set up timing
        self.timing = {}
        self.timing["start"] = time.time()

    def setup(self):
        """Setup the run

        """
        io.log_message("{0}: setting up".format(self.runid))

        # set up the driver
        self.driver.setup(self.runid, self.material, *self.driver_opts)

        if self.setup_restart:
            return self._setup_restart_io()

        else:
            return self._setup_new_io()

    def _setup_restart_io(self):
        # open exodus file to append
        # set vars
        self.exo = io.ExoManager.from_existing(self.runid, self.runid + ".exo",
                                               self.driver.step_num)

    def _setup_new_io(self):
        # get global and element info to write to exodus file
        glob_var_names = self.driver.glob_vars()
        ele_var_names = self.driver.elem_vars()
        title = "gmd {0} simulation".format(self.driver.name)
        info = [self.driver.mtlmdl.name, self.driver.mtlmdl._param_vals,
                self.driver.name, self.driver.kappa,
                self.driver.paths, self.driver.surfaces, self.extract]
        self.exo = io.ExoManager.new_from_runid(
            self.runid, title, glob_var_names, ele_var_names, info)

        glob_var_vals = self.driver.glob_var_vals()
        elem_var_vals = self.driver.elem_var_vals()
        num_dim, num_nodes = 3, 8
        u = np.zeros(num_dim * num_nodes)
        time_end = 0.
        self.exo.write_data(time_end, glob_var_vals, elem_var_vals, u)

        # write to the log file the material props
        L = max(max(len(n) for n in ele_var_names), 10)
        param_ivals = self.mtlprops
        param_names = self.driver.mtlmdl.params()
        param_vals = self.driver.mtlmdl.param_vals()
        io.log_debug("Material Parameters")
        io.log_debug("  {1:{0}s}  {2:12}  {3:12}".format(
            L, "Name", "iValue", "Value"))
        for p in zip(param_names, param_ivals, param_vals):
            io.log_debug("  {1:{0}s} {2: 12.6E} {3: 12.6E}".format(L, *p))

        # write out plotable data
        io.log_debug("Output Variables:")
        io.log_debug("Global")
        for item in glob_var_names:
            io.log_debug("  " + item)
        io.log_debug("Element")
        for item in ele_var_names:
            io.log_debug("  " + item)

        return

    def run(self):
        """Run the problem

        """
        io.log_message("{0}: starting calculations".format(self.runid))
        retcode = self.driver.process_paths_and_surfaces(self.dump_state)
        return retcode

    def finish(self):
        # udpate and close the file
        self.timing["end"] = time.time()
        if self.driver._paths_and_surfaces_processed:
            io.log_message("{0}: calculations completed ({1:.4f}s)".format(
                self.runid, self.timing["end"] - self.timing["start"]))
        else:
            io.log_error("{0}: calculations did not complete".format(self.runid),
                         r=0)

        self.exo.finish()

        if self.extract and self.driver._paths_and_surfaces_processed:
            ofmt, step, ffmt, variables, paths = self.extract[:5]
            if variables:
                exodump(self.exo.filepath, step=step, ffmt=ffmt,
                        variables=variables, ofmt=ofmt)
            if paths:
                self.driver.extract_paths(self.exo.filepath, paths)

            self.timing["extract"] = time.time()
            io.log_message("{0}: extraction completed ({1:.4f}s)".format(
                self.runid, self.timing["extract"] - self.timing["end"]))

        # close the log
        io.close_and_reset_logger()

        return

    def dump_state(self, time_end):
        """Dump current state to exodus file

        """
        # global data
        glob_var_vals = self.driver.glob_var_vals()

        # element data
        elem_var_vals = self.driver.elem_var_vals()

        # determine displacement
        F = np.reshape(self.driver.elem_var_vals("DEFGRAD"), (3, 3))
        u = np.zeros(self.exo.num_nodes * self.exo.num_dim)
        for i, X in enumerate(self.exo.coords):
            k = i * self.exo.num_dim
            u[k:k+self.exo.num_dim] = np.dot(F, X) - X

        self.exo.write_data(time_end, glob_var_vals, elem_var_vals, u)
        return

    def variables(self):
        return self.driver.elem_vars()

    def output(self):
        return self.runid + ".exo"
