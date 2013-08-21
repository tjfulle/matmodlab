import sys
import numpy as np
import datetime

from __config__ import cfg
import exowriter as exo


class Error1(Exception):
    def __init__(self, message):
        if cfg.debug:
            raise Exception("*** gmd: error: {0}".format(message))
        sys.stderr.write("*** gmd: error: {0}\n".format(message))
        raise SystemExit(2)


LOGFILE = None
WARNINGS_LOGGED = 0
VERBOSITY = 1
class Logger(object):
    def __init__(self, runid, verbosity):
        global LOGFILE, VERBOSITY
        if runid is None and LOGFILE is None:
            raise Error("inconsistent logger instantiation")
        if verbosity is not None:
            VERBOSITY = verbosity
        if LOGFILE is None:
            LOGFILE = open(runid + ".log", "w")

    def logmes(self, message, logger=[None]):
        message = "gmd: {0}\n".format(message)
        if VERBOSITY:
            sys.stdout.write(message)
        LOGFILE.write(message)

    def logwrn(self, message):
        global WARNINGS_LOGGED
        if message is None:
            return WARNINGS_LOGGED
        message = "*** gmd: warning: {0}\n".format(message)
        sys.stderr.write(message)
        LOGFILE.write(message)
        WARNINGS_LOGGED += 1

    @classmethod
    def getlogger(cls):
        if LOGFILE is None:
            raise Error1("Logger not yet initialized")
        return cls(None, None)


def logmes(message, logger=[None]):
    if logger[0] is None:
        logger[0] = Logger.getlogger()
    logger[0].logmes(message)


def logwrn(message=None, logger=[None]):
    if logger[0] is None:
        logger[0] = Logger.getlogger()
    return logger[0].logwrn(message)


class ExoManager(object):
    """The main IO manager

    """
    def __init__(self, runid, num_dim, coords, conn, elem_blks,
                 all_element_data, title):
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
        glob_var_names = ["TIME_STEP"]
        self.num_glob_vars = len(glob_var_names)
        self.exofile.put_var_param("g", self.num_glob_vars)
        self.exofile.put_var_names("g", self.num_glob_vars, glob_var_names)

        nod_var_names = ["DISPLX", "DISPLY"]
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
        glob_var_vals = np.zeros(self.num_glob_vars)
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

    def write_data(self, time, dt, all_element_data, u):
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
        glob_var_vals = np.zeros(self.num_glob_vars)
        for j in range(self.num_glob_vars):
            glob_var_vals[j] = dt
            continue
        self.exofile.put_glob_vars(self.time_step, self.num_glob_vars,
                                   glob_var_vals)

        # write nodal variables
        for k in range(self.num_nod_vars):
            self.exofile.put_nodal_var(self.time_step, k, self.num_nodes,
                                       u[k::self.num_dim])

        # write element variables
        for (elem_blk_id, num_elem_this_blk, elem_blk_data) in all_element_data:
            for k in range(self.num_elem_vars):
                self.exofile.put_elem_var(
                    self.time_step, k, elem_blk_id,
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
