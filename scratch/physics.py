import os
import sys
import time
import numpy as np

from project import RESTART
from core.runtime import set_runtime_opt
import utils.mmlio as io
from drivers.driver import create_driver

class PhysicsHandler(object):

    def __init__(self, runid, verbosity, driver_name, driver_path, driver_opts,
                 mat_model, mat_params, mat_opts, mat_istate, extract):
        """Initialize the PhysicsHandler object

        """
        self.runid = runid
        set_runtime_opt("runid", runid)

        restart = driver_opts["restart"] == RESTART
        mode = "a" if restart else "w"
        io.setup_logger(runid, verbosity, mode=mode)
        io.log_message("{0}: setting up".format(self.runid))

        # Create the material
        mat_opts["initial_temperature"] = driver_opts["initial_temperature"]
        mat_opts["user_field"] = driver_opts["initial_user_field"]
        mat_opts["kappa"] = driver_opts["kappa"]
        mat = mat_model.instantiate_material(mat_params, mat_istate, **mat_opts)

        self.driver = create_driver(driver_name, driver_path, driver_opts, mat)
        self.extract = extract

        # set up timing
        self.timing = {}
        self.timing["start"] = time.time()

        if restart:
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
        title = "mmd {0} simulation".format(self.driver.name)
        mat = (self.driver.material.name, self.driver.material.param_names,
               self.driver.material.params)
        drv = (self.driver.name, self.driver.path, self.driver.options)
        info = [mat, drv, self.extract]
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
        param_names = self.driver.material.parameters(names=True)
        iparam_vals = self.driver.material.parameters(ival=True)
        param_vals = self.driver.material.parameters()
        io.log_debug("Material Parameters")
        io.log_debug("  {1:{0}s}  {2:12}  {3:12}".format(
            L, "Name", "iValue", "Value"))
        for p in zip(param_names, iparam_vals, param_vals):
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
        from utils.exo.exodump import exodump
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
