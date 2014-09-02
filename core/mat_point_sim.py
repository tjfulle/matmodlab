import os
import sys
import time
import numpy as np

from core.runtime import set_runtime_opt
from core.mmlio import exo, logger
from core.variable import Variable, VAR_SCALAR
from utils.array import Array

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

        glob_data = []
        # two data arrays: global, element
        for d in self.variables:
            assert len(d.keys) == len(d.initial_value)
            glob_data.append((d.name, d.keys, d.initial_value))
        self.glob_data = Array(glob_data)

        elem_data = []
        for item in (self.driver.variables, self.material.variables):
            for d in item:
                assert len(d.keys) == len(d.initial_value)
                elem_data.append((d.name, d.keys, d.initial_value))
        self.elem_data = Array(elem_data)

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
	elems_this_blk = [0]
        elem_type = "HEX"
        num_node_per_elem = 8
	elem_data = [[ebid, neeb,
                      self.elem_data.reshape((neeb, len(elem_var_names)))]]
	elem_blks = [[ebid, elems_this_blk, elem_type, num_node_per_elem,
		      elem_var_names]]
        elem_num_map = [0]

	glob_data = [self.plot_keys, self.glob_data]

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
        for item in self.plot_keys:
            logger.debug("  " + item)
        logger.debug("Element")
        for item in elem_var_names:
            logger.debug("  " + item)

    def run(self):
        """Run the problem

        """
        logger.write("{0}: starting calculations".format(self.runid))
	retcode = self.driver.run(self.glob_data, self.elem_data, self.material)
        self.finish()
        return retcode

    def finish(self):
        # udpate and close the file
        self.timing["end"] = time.time()
        if self.driver.ran:
            logger.write("{0}: calculations completed ({1:.4f}s)".format(
                self.runid, self.timing["end"] - self.timing["start"]))
        else:
            logger.error("{0}: calculations did not complete".format(self.runid),
                         r=0)

        exo.finish()

        return

    def extract_from_db(self, variables=None, paths=None, format="ascii",
                        step=1, ffmt=".18f"):
        from utils.exo.exodump import exodump
        if variables:
            exodump(exo.filepath, step=step, ffmt=ffmt,
                    variables=variables, ofmt=format)
        if paths:
            self.driver.extract_paths(exo.filepath, paths)
