import os
import sys
import time
import numpy as np

from __config__ import cfg
import utils.io as io
from utils.exodump import exodump
from utils.errors import Error1
from drivers.drivers import create_driver

class ModelDriver(object):

    def __init__(self, runid, driver, mtlmdl, mtlprops, legs, extract, *opts):
        """Initialize the ModelDriver object

        Parameters
        ----------
        mtlmdl : str
            Name of material model

        mtlprops : ndarray
            The (unchecked) material properties

        legs : list
            The deformation legs

        """
        self.runid = runid

        self.driver = create_driver(driver)
        if self.driver is None:
            raise Error1("{0}: unknown driver type".format(driver))

        self.mtlmdl = mtlmdl
        self.mtlprops = mtlprops
        self.legs = legs
        self.extract = extract
        self.opts = opts

        # set up timing
        self.timing = {}
        self.timing["initial"] = time.time()

    def setup(self):

        # set up the driver
        self.driver.setup(self.runid, self.mtlmdl, self.mtlprops, *self.opts)

        # Set up the "mesh"
        self.num_dim = 3
        self.num_nodes = 8
        self.coords = np.array([[-1, -1, -1], [ 1, -1, -1],
                                [ 1,  1, -1], [-1,  1, -1],
                                [-1, -1,  1], [ 1, -1,  1],
                                [ 1,  1,  1], [-1,  1,  1]],
                               dtype=np.float64) * .5
        connect = np.array([range(8)], dtype=np.int)

        elem_blk_id = 1
        elem_blk_els = [0]
        num_elem_this_blk = 1
        elem_type = "HEX"
        num_nodes_per_elem = 8
        ele_var_names = self.variables()
        elem_blk_data = self.driver.data()
        elem_blks = [[elem_blk_id, elem_blk_els, elem_type,
                      num_nodes_per_elem, ele_var_names]]
        all_element_data = [[elem_blk_id, num_elem_this_blk, elem_blk_data]]
        title = "gmd {0} simulation".format(self.driver.name)

        self.io = io.IOManager(self.runid, self.num_dim, self.coords, connect,
                               elem_blks, all_element_data, title)

    def dump_state(self, dt, time_end):
        elem_blk_id = 1
        num_elem_this_blk = 1
        elem_blk_data = self.driver.data()
        all_element_data = [[elem_blk_id, num_elem_this_blk, elem_blk_data]]

        # determine displacement
        F = np.reshape(self.driver.data("DEFGRAD"), (3, 3))
        u = np.zeros(self.num_nodes * self.num_dim)
        for i, X in enumerate(self.coords):
            k = i * self.num_dim
            u[k:k+self.num_dim] = np.dot(F, X) - X

        self.io.write_data(time_end, dt, all_element_data, u)

    def variables(self):
        return self.driver.variables()

    def finish(self):
        # udpate and close the file
        self.timing["final"] = time.time()
        print "Finished calculations for simulation {0} in {1:.4f}s".format(
            self.runid, self.timing["final"] - self.timing["initial"])
        self.io.finish()

        if self.extract:
            otype, variables = self.extract
            if otype.lower() == "ascii":
                exodump(self.runid + ".exo", variables=variables)
            else:
                raise Error1("{0}: unkown extraction format".format(otype))
        return

    def run(self):
        """Run the problem

        """
        retcode = self.driver.process_legs(self.legs, self.dump_state)
        return retcode
