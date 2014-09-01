import os
import sys
import time
import numpy as np

from core.runtime import set_runtime_opt
from core.mmlio import exo, logger
from core.varinc import *
from core.variable import Variable, VariableContainer

class MaterialPointSimulator(object):
    _vars = []
    def __init__(self, runid, driver, material, verbosity=0, title=None):
        """Initialize the MaterialPointSimulator object

        """
        self.runid = runid
        set_runtime_opt("runid", runid)
	self.title = "matmodlab single element simulation"

        logger.add_file_handler(os.path.join(os.getcwd(), self.runid + ".log"))
	logger.write_intro()
        logger.write("{0}: setting up".format(self.runid))

        self.driver = driver
        self.material = material

	# register global variables
        self.register_variable("TIME_STEP", VAR_SCALAR)
        self.register_variable("STEP_NUM", VAR_SCALAR)
        self.register_variable("LEG_NUM", VAR_SCALAR)

	self.data = VariableContainer(self.variables, self.driver.variables,
				      self.material.variables)


        # set up timing
        self.timing = {}
        self.timing["start"] = time.time()

        self.setup_exo_db()

    def register_variable(self, var_name, var_type):
        self._vars.append(Variable(var_name, var_type))

    @property
    def variables(self):
	return self._vars

    @property
    def plot_keys(self):
        return [x for l in [v.keys for v in self.variables] for x in l]

    def setup_exo_db(self):
        time_step = 0
	num_dim = 3
	num_nodes = 8
	num_elem = 1
	coords = np.array([[-1, -1, -1], [ 1, -1, -1],
			   [ 1,  1, -1], [-1,  1, -1],
			   [-1, -1,  1], [ 1, -1,  1],
			   [ 1,  1,  1], [-1,  1,  1]],
			   dtype=np.float64) * .5
        connect = np.array([range(8)], dtype=np.int)
        num_elem_blk = 1
        num_node_sets = 0
	node_sets = []
        num_side_sets = 0
	side_sets = []

        elem_var_names = self.driver.plot_keys
	ebid = 1
	neeb = 1
	vals = self.data[3:]
	elems_this_blk = [0]
        elem_type = "HEX"
        num_node_per_elem = 8
	elem_data = [[ebid, neeb, vals.reshape((neeb, len(elem_var_names)))]]
	elem_blks = [[ebid, elems_this_blk, elem_type, num_node_per_elem,
		      elem_var_names]]
        elem_num_map = [0]

        glob_var_names = self.plot_keys
	glob_var_vals = self.data[:3]
	glob_data = [glob_var_names, glob_var_vals]

        exo.put_init(self.runid, num_dim, coords, connect, elem_blks,
                     node_sets, side_sets, glob_data, elem_data, elem_num_map,
                     title=self.title)

        L = max(max(len(n) for n in elem_var_names), 10)
        param_names = self.material.parameter_names
        iparam_vals = self.material.initial_parameters
        param_vals = self.material.parameters

        logger.debug("Material Parameters")
        logger.debug("  {1:{0}s}  {2:12}  {3:12}".format(
            L, "Name", "iValue", "Value"))
        for p in zip(param_names, iparam_vals, param_vals):
            logger.debug("  {1:{0}s} {2: 12.6E} {3: 12.6E}".format(L, *p))

        # write out plotable data
        logger.debug("Output Variables:")
        logger.debug("Global")
        for item in glob_var_names:
            logger.debug("  " + item)
        logger.debug("Element")
        for item in elem_var_names:
            logger.debug("  " + item)

    def run(self):
        """Run the problem

        """
        logger.write("{0}: starting calculations".format(self.runid))
	retcode = self.driver.run(self.data[:3], self.data[3:], self.material)
        self.finish()
        return retcode

    def finish(self):
        # udpate and close the file
        from utils.exo.exodump import exodump
        self.timing["end"] = time.time()
        if self.driver._paths_and_surfaces_processed:
            logger.write("{0}: calculations completed ({1:.4f}s)".format(
                self.runid, self.timing["end"] - self.timing["start"]))
        else:
            logger.error("{0}: calculations did not complete".format(self.runid),
                         r=0)

        self.exo.finish()

        return

    def extract_from_db(self, format="ascii", step=1, ffmt=".18f",
                        variables=None, paths=None):

        if variables:
            exodump(self.exo.filepath, step=step, ffmt=ffmt,
                    variables=variables, ofmt=format)
        if paths:
            self.driver.extract_paths(self.exo.filepath, paths)

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
