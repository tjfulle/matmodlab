import sys
import numpy as np
import datetime

from utils.exojac import ExodusIIFile

NUM_DIM = 3
COORDS = np.array([[-1, -1, -1], [ 1, -1, -1],
                   [ 1,  1, -1], [-1,  1, -1],
                   [-1, -1,  1], [ 1, -1,  1],
                   [ 1,  1,  1], [-1,  1,  1]],
                   dtype=np.float64) * .5
EBID = 1
NEEB = 1
NUM_NODES = 8
NUM_ELEM = 1
ELEM_TYPE = "HEX"
NUM_ELEM_BLK = 1
ELS_IN_BLK = [0]
NUM_NODE_IN_EL = 8
ELEM_NUM_MAP = [0]
NUM_NODE_SETS = 0
NODE_SETS = []
NUM_SIDE_SETS = 0
SIDE_SETS = []
CONNECT = np.array([range(8)], dtype=np.int)


class ExodusII(object):
    """The main ExodusII manager

    """

    def __init__(self, runid, d=None):
        # create new file
        self.runid = runid
        self.exofile = ExodusIIFile(runid, mode="w", d=d)
        self.filepath = self.exofile.filepath

    def put_init(self, glob_data, glob_vars, elem_data, elem_vars, title=None):
        """Put initial data in to the ExodusII file

        """
        # constants for single element sim
	elem_data = [[EBID, NEEB, elem_data.reshape((NEEB, len(elem_vars)))]]
	elem_blks = [[EBID, ELS_IN_BLK, ELEM_TYPE, NUM_NODE_IN_EL, elem_vars]]

	glob_data = [glob_vars, glob_data]

        # initialize file with parameters
        if not title:
            title = "matmodlab single element simulation"

        self.exofile.put_init(title, NUM_DIM, NUM_NODES, NUM_ELEM,
                              NUM_ELEM_BLK, NUM_NODE_SETS, NUM_SIDE_SETS)

        # write nodal coordinates values and names to database
        coord_names = np.array(["COORDX", "COORDY", "COORDZ"])[:NUM_DIM]
        self.exofile.put_coord_names(coord_names)
        z = np.zeros(NUM_NODES) if NUM_DIM <= 2 else COORDS[:, 2]
        self.exofile.put_coord(COORDS[:, 0], COORDS[:, 1], z)

        # write element order map
        self.exofile.put_elem_num_map(ELEM_NUM_MAP)

        # write element block parameters
        for item in elem_blks:
            (elem_blk_id, elem_blk_els, elem_type,
             num_nodes_per_elem, ele_var_names) = item
            # for now, we do not use attributes
            num_attr = 0
            num_elem_this_blk = len(elem_blk_els)
            self.exofile.put_elem_block(elem_blk_id, elem_type, num_elem_this_blk,
                                        num_nodes_per_elem, num_attr)

            # write element connectivity for each element block
            blk_conn = CONNECT[elem_blk_els][:, :num_nodes_per_elem]
            self.exofile.put_elem_conn(elem_blk_id, blk_conn)
            continue

        # write node sets
        for (node_set_id, node_set_nodes) in NODE_SETS:
            num_nodes_in_set = len(node_set_nodes)
            # no distribution factors
            num_dist_fact = 0
            self.exofile.put_node_set_param(node_set_id,
                                            num_nodes_in_set, num_dist_fact)
            self.exofile.put_node_set(node_set_id, node_set_nodes)
            continue

        # write side sets
        for (side_set_id, side_set_elems, side_set_sides) in SIDE_SETS:
            num_dist_fact = 0
            num_sides_in_set = len(side_set_elems)
            self.exofile.put_side_set_param(side_set_id, num_sides_in_set,
                                            num_dist_fact)
            self.exofile.put_side_set(side_set_id, side_set_elems, side_set_sides)
            continue

        # write QA records
        now = datetime.datetime.now()
        day = now.strftime("%m/%d/%y")
        hour = now.strftime("%H:%M:%S")
        num_qa_rec = 1
        qa_title = "FEM finite element simulation"
        qa_record = np.array([[qa_title, self.runid, day, hour]])
        self.exofile.put_qa(num_qa_rec, qa_record)

        # write results variables parameters and names
        glob_var_names = glob_data[0]
        self.num_glob_vars = len(glob_var_names)
        self.exofile.put_var_param("g", self.num_glob_vars)
        self.exofile.put_var_names("g", self.num_glob_vars, glob_var_names)

        nod_var_names = ["DISPLX", "DISPLY", "DISPLZ"][:NUM_DIM]
        self.num_nod_vars = len(nod_var_names)
        self.exofile.put_var_param("n", self.num_nod_vars)
        self.exofile.put_var_names("n", self.num_nod_vars, nod_var_names)

        self.num_elem_vars = len(ele_var_names)
        self.exofile.put_var_param("e", self.num_elem_vars)
        self.exofile.put_var_names("e", self.num_elem_vars, ele_var_names)

        # write element variable truth table
        truth_tab = np.empty((NUM_ELEM_BLK, self.num_elem_vars), dtype=np.int)
        for i in range(NUM_ELEM_BLK):
            for j in range(self.num_elem_vars):
                truth_tab[i, j] = 1
        self.exofile.put_elem_var_tab(NUM_ELEM_BLK, self.num_elem_vars, truth_tab)

        # write first step
        time_val = 0.
        self.count = 0
        self.exofile.put_time(self.count, time_val)

        # global values
        glob_var_vals = glob_data[1]
        self.exofile.put_glob_vars(self.count, self.num_glob_vars,
                                   glob_var_vals)

        # nodal values
        nodal_var_vals = np.zeros(NUM_NODES, dtype=np.float64)
        for k in range(self.num_nod_vars):
            self.exofile.put_nodal_var(self.count, k, NUM_NODES, nodal_var_vals)
            continue

        # element values
        for (elem_blk_id, num_elem_this_blk, elem_blk_data) in elem_data:
            for k in range(self.num_elem_vars):
                self.exofile.put_elem_var(
                    self.count, k, elem_blk_id,
                    num_elem_this_blk, elem_blk_data.T[k])
            continue

        self.exofile.update()
        self.count += 1
        pass

    def finish(self):
        # udpate and close the file
        self.exofile.update()
        self.exofile.close()
        del self.exofile.db
        self.exofile = None
        return

    def snapshot(self, time, glob_data, elem_data):
        """Dump information from current time step to results file

        Parameters
        ----------
        time : float
            Current time

        dt : float
            Time step

        u : array_like
            Nodal displacements

        """
        glob_var_vals = glob_data.data
        elem_var_vals = elem_data.data

        # determine displacement
        F = np.reshape(elem_data["DEFGRAD"], (3, 3))
        u = np.zeros(NUM_NODES * NUM_DIM)
        for i, X in enumerate(COORDS):
            k = i * NUM_DIM
            u[k:k+NUM_DIM] = np.dot(F, X) - X

        # write time value
        self.exofile.put_time(self.count, time)

        # write global variables
        self.exofile.put_glob_vars(self.count, self.num_glob_vars, glob_var_vals)

        # write nodal variables
        for k in range(self.num_nod_vars):
            self.exofile.put_nodal_var(self.count, k, NUM_NODES, u[k::NUM_DIM])

        # write element variables
        elem_blk_id = 1
        num_elem_this_blk = 1
        num_elem_vars = len(elem_var_vals)
        for k in range(num_elem_vars):
            self.exofile.put_elem_var(self.count, k, elem_blk_id,
                                      num_elem_this_blk, elem_var_vals.T[k])
            continue

        # udpate and close the file
        self.count += 1
        self.exofile.update()

        return
