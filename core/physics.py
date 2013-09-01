import os
import sys
import time
import numpy as np

from __config__ import cfg
import core.io as io
from utils.exodump import exodump

class PhysicsHandler(object):

    def __init__(self, runid, verbosity, driver, mtlmdl, mtlprops, tterm,
                 extract, driver_opts):
        """Initialize the PhysicsHandler object

        Parameters
        ----------
        mtlmdl : str
            Name of material model

        mtlprops : ndarray
            The (unchecked) material properties

        """

        io.setup_logger(runid, verbosity)

        self.runid = runid

        self.driver = driver()
        self.material = (mtlmdl, mtlprops)
        self.mtlprops = np.array(mtlprops)
        self.tterm = tterm
        self.extract = extract
        self.driver_opts = driver_opts

        # set up timing
        self.timing = {}
        self.timing["start"] = time.time()

    def setup(self):

        # set up the driver

        io.log_message("{0}: setting up".format(self.runid))

        self.driver.setup(self.runid, self.material, *self.driver_opts)

        # Set up the "mesh"
        self.num_dim = 3
        self.num_nodes = 8
        self.coords = np.array([[-1, -1, -1], [ 1, -1, -1],
                                [ 1,  1, -1], [-1,  1, -1],
                                [-1, -1,  1], [ 1, -1,  1],
                                [ 1,  1,  1], [-1,  1,  1]],
                               dtype=np.float64) * .5
        connect = np.array([range(8)], dtype=np.int)

        # get global and element info to write to exodus file
        glob_var_names = self.driver.glob_vars()
        glob_var_vals = self.driver.glob_var_vals()
        glob_var_data = [glob_var_names, glob_var_vals]

        elem_blk_id = 1
        elem_blk_els = [0]
        num_elem_this_blk = 1
        elem_type = "HEX"
        num_nodes_per_elem = 8
        ele_var_names = self.driver.elem_vars()
        elem_blk_data = self.driver.elem_var_vals()
        elem_blks = [[elem_blk_id, elem_blk_els, elem_type,
                      num_nodes_per_elem, ele_var_names]]
        all_element_data = [[elem_blk_id, num_elem_this_blk, elem_blk_data]]
        title = "gmd {0} simulation".format(self.driver.name)

        self.exo = io.ExoManager(self.runid, self.num_dim, self.coords, connect,
                                 glob_var_data, elem_blks,
                                 all_element_data, title)

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

    def run(self):
        """Run the problem

        """
        io.log_message("{0}: starting calculations".format(self.runid))
        run_opts = (self.tterm, )
        retcode = self.driver.process_paths(self.dump_state, *run_opts)
        return retcode

    def finish(self):
        # udpate and close the file
        self.timing["end"] = time.time()
        io.log_message("{0}: calculations completed ({1:.4f}s)".format(
            self.runid, self.timing["end"] - self.timing["start"]))
        self.exo.finish()

        if self.extract:
            ofmt, step, ffmt, variables = self.extract
            exodump(self.runid + ".exo", step=step, ffmt=ffmt,
                    variables=variables, ofmt=ofmt)
            self.timing["extract"] = time.time()
            io.log_message("{0}: extraction completed ({1:.4f}s)".format(
                self.runid, self.timing["extract"] - self.timing["end"]))

        # close the log
        io.close_and_reset_logger()

        return

    def dump_state(self, time_end):
        # global data
        glob_data = self.driver.glob_var_vals()

        # element data
        elem_blk_id = 1
        num_elem_this_blk = 1
        elem_blk_data = self.driver.elem_var_vals()
        all_element_data = [[elem_blk_id, num_elem_this_blk, elem_blk_data]]

        # determine displacement
        F = np.reshape(self.driver.elem_var_vals("DEFGRAD"), (3, 3))
        u = np.zeros(self.num_nodes * self.num_dim)
        for i, X in enumerate(self.coords):
            k = i * self.num_dim
            u[k:k+self.num_dim] = np.dot(F, X) - X

        self.exo.write_data(time_end, glob_data, all_element_data, u)

    def variables(self):
        return self.driver.elem_vars()

    def output(self):
        return self.runid + ".exo"
