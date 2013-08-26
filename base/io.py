import os
import sys
import numpy as np
import datetime
import logging

from __config__ import cfg
import exowriter as exo


WARNINGS_LOGGED = 0
LOGGER = None
LEVELS = {2: logging.DEBUG,
          1: logging.INFO,
          0: logging.WARNING,}


class Error1(Exception):
    def __init__(self, message):
        logging.exception(message)
        if cfg.debug:
            raise Exception("*** gmd: error: {0}".format(message))
        raise SystemExit(2)


def setup_logger(runid, verbosity, d=None):
    global LOGGER
    if LOGGER is not None:
        raise Error1("Logger already setup")

    if d is None:
        d = os.getcwd()
    if not os.path.isdir(d):
        raise OSError("{0}: no such directory".format(d))
    logfile = os.path.join(d, runid + ".log")

    logging.basicConfig(level=logging.DEBUG,
        format="gmd: %(asctime)s %(levelname)s: %(message)s",
        datefmt="%b %d %Y, %H:%M:%S", filename=logfile, filemode='w')

    # console logging
    ch = logging.StreamHandler()
    ch.setLevel(LEVELS.get(verbosity, logging.INFO))
    cf = logging.Formatter("gmd: %(levelname)s: %(message)s")
    ch.setFormatter(cf)
    logging.getLogger("").addHandler(ch)

    LOGGER = logging.getLogger("")

    return


def log_debug(message):
    LOGGER.debug(message)


def log_message(message):
    LOGGER.info(message)


def log_warning(message):
    increment_warning()
    LOGGER.warning(message)


def log_error(message):
    raise Error1(message)


def increment_warning():
    global WARNINGS_LOGGED
    WARNINGS_LOGGED += 1


class ExoManager(object):
    """The main IO manager

    """
    def __init__(self, runid, num_dim, coords, conn, glob_var_data,
                 elem_blks, all_element_data, title):
        """Instantiate a IOManager object

        runid : str
            The simulation ID

        num_dim : int
            Number of spatial dimensions of problem

        coords : array_like, (num_node, 3)
            Coordinates of the num_nodeth node

        conn : array_like
            Connectivity array

        elem_blks : array_like, (num_elem_blk, 5)
            Element block information
            elem_blks[i] -> [ID, [ELS IN BLK], ETYPE, NNODES, VARS]

        all_element_data : list
            Initial data for each element
            all_element_data[i]:
              0 -> element block id
              1 -> number of elements in block
              2 -> element block data
                   element block data is of the form:
                   data[k] -> average element data for kth element


        """

        # create new file
        self.runid = runid
        self.num_dim = num_dim
        self.exofile = exo.ExodusIIWriter.new_from_runid(runid)

        # initialize file with parameters
        self.elem_blks = elem_blks
        self.num_nodes = len(coords)
        self.num_elem = len(conn)
        self.num_elem_blk = len(self.elem_blks)
        num_node_sets = 0
        num_side_sets = 0
        title = title.format(str(self.num_dim) + "D")

        self.exofile.put_init(title, self.num_dim, self.num_nodes, self.num_elem,
                              self.num_elem_blk, num_node_sets, num_side_sets)

        # write nodal coordinates values and names to database
        coord_names = np.array(["COORDX", "COORDY", "COORDZ"])
        self.exofile.put_coord_names(coord_names)
        self.exofile.put_coord(coords[:, 0], coords[:, 1], coords[:, 2])

        # write element block parameters
        for item in self.elem_blks:
            (elem_blk_id, elem_blk_els, elem_type,
             num_nodes_per_elem, ele_var_names) = item
            # for now, we do not use attributes
            num_attr = 0
            num_elem_this_blk = len(elem_blk_els)
            self.exofile.put_elem_block(elem_blk_id, elem_type, num_elem_this_blk,
                                        num_nodes_per_elem, num_attr)

            # write element connectivity for each element block
            blk_conn = conn[elem_blk_els][:, :num_nodes_per_elem]
            self.exofile.put_elem_conn(elem_blk_id, blk_conn)
            continue

        # write QA records
        now = datetime.datetime.now()
        day = now.strftime("%m/%d/%y")
        hour = now.strftime("%H:%M:%S")
        num_qa_rec = 1
        qa_title = "Wasatch finite element simulation"
        qa_record = np.array([[qa_title, runid, day, hour]])
        self.exofile.put_qa(num_qa_rec, qa_record)

        # write results variables parameters and names
        glob_var_names = glob_var_data[0]
        self.num_glob_vars = len(glob_var_names)
        self.exofile.put_var_param("g", self.num_glob_vars)
        self.exofile.put_var_names("g", self.num_glob_vars, glob_var_names)

        nod_var_names = ["DISPLX", "DISPLY", "DISPLZ"]
        if self.num_dim == 3:
            nod_var_names.append("DISPLZ")
        self.num_nod_vars = len(nod_var_names)
        self.exofile.put_var_param("n", self.num_nod_vars)
        self.exofile.put_var_names("n", self.num_nod_vars, nod_var_names)

        self.num_elem_vars = len(ele_var_names)
        self.exofile.put_var_param("e", self.num_elem_vars)
        self.exofile.put_var_names("e", self.num_elem_vars, ele_var_names)

        # write element variable truth table
        truth_tab = np.empty((self.num_elem_blk, self.num_elem_vars), dtype=np.int)
        for i in range(self.num_elem_blk):
            for j in range(self.num_elem_vars):
                truth_tab[i, j] = 1
        self.exofile.put_elem_var_tab(self.num_elem_blk, self.num_elem_vars,
                                      truth_tab)

        # write first step
        time_val = 0.
        self.time_step = 0
        self.exofile.put_time(self.time_step, time_val)

        # global values
        glob_var_vals = glob_var_data[1]
        self.exofile.put_glob_vars(self.time_step, self.num_glob_vars,
                                   glob_var_vals)

        # nodal values
        nodal_var_vals = np.zeros(self.num_nodes)
        for k in range(self.num_nod_vars):
            self.exofile.put_nodal_var(self.time_step, k, self.num_nodes,
                                       nodal_var_vals)
            continue

        # element values
        for (elem_blk_id, num_elem_this_blk, elem_blk_data) in all_element_data:
            for k in range(self.num_elem_vars):
                self.exofile.put_elem_var(
                    self.time_step, k, elem_blk_id,
                    num_elem_this_blk, elem_blk_data.T[k])
            continue

        self.exofile.update()
        self.time_step += 1
        pass

    def write_data(self, time, glob_var_vals, all_element_data, u):
        """Dump information from current time step to results file

        Parameters
        ----------
        time : float
            Current time

        dt : float
            Time step

        all_element_data : list
            Data for each element
            all_element_data[i]:
              0 -> element block id
              1 -> number of elements in block
              2 -> element block data
                   element block data is of the form:
                   data[k] -> average element data for kth element

        u : array_like
            Nodal displacements

        """

        # write time value
        self.exofile.put_time(self.time_step, time)

        # write global variables
        assert len(glob_var_vals) == self.num_glob_vars
        self.exofile.put_glob_vars(self.time_step, self.num_glob_vars,
                                   glob_var_vals)

        # write nodal variables
        for k in range(self.num_nod_vars):
            self.exofile.put_nodal_var(self.time_step, k, self.num_nodes,
                                       u[k::self.num_dim])

        # write element variables
        for (elem_blk_id, num_elem_this_blk, elem_blk_data) in all_element_data:
            for k in range(self.num_elem_vars):
                self.exofile.put_elem_var(self.time_step, k, elem_blk_id,
                                          num_elem_this_blk, elem_blk_data.T[k])
            continue

        # udpate and close the file
        self.time_step += 1
        self.exofile.update()

        return

    def finish(self):
        # udpate and close the file
        self.exofile.update()
        self.exofile.close()
        return
