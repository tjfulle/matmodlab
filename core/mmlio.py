import os
import sys
import numpy as np
import datetime
import logging

from mml import __version__
from core.runtime import opts


INP_ERRORS = 0
WARNINGS_LOGGED = 0
LOGGER = None
LEVELS = {2: logging.DEBUG,
          1: logging.INFO,
          0: logging.WARNING,}


class Error1(Exception):
    def __init__(self, message):
        self.message = message.rstrip()
        try: LOGGER.exception(message)
        except: pass
        super(Error1, self).__init__(message)


def fatal_inp_error(message):
    global INP_ERRORS
    INP_ERRORS += 1
    sys.stderr.write("*** error: {0}\n".format(message))
    if INP_ERRORS > 5:
        raise SystemExit("*** error: maximum number of input errors exceeded")


def input_errors():
    return INP_ERRORS


def setup_logger(runid, verbosity, d=None, mode="w"):
    global LOGGER
    if LOGGER is not None:
        raise Error1("Logger already setup")

    if d is None:
        d = os.getcwd()
    if not os.path.isdir(d):
        raise OSError("{0}: no such directory".format(d))
    logfile = os.path.join(d, runid + ".log")

    logging.basicConfig(level=logging.DEBUG,
        format="mml: %(asctime)s %(levelname)s: %(message)s",
        datefmt="%b %d %Y, %H:%M:%S", filename=logfile, filemode=mode)

    # console logging
    ch = logging.StreamHandler()
    ch.setLevel(LEVELS.get(verbosity, logging.INFO))
    cf = logging.Formatter("mml: %(levelname)s: %(message)s")
    ch.setFormatter(cf)
    logging.getLogger("").addHandler(ch)

    # set the logger
    LOGGER = logging.getLogger("")

    return


def close_and_reset_logger():
    global LOGGER, INP_ERRORS
    for handler in [x for x in LOGGER.handlers]:
        LOGGER.removeHandler(handler)
    del LOGGER
    LOGGER = None
    INP_ERRORS = 0


def log_debug(message):
    LOGGER.debug(message)


def log_message(message):
    LOGGER.info(message)


def log_redirect(_handler=[]):
    if not _handler:
        for handler in [x for x in LOGGER.handlers]:
            if type(handler) == type(logging.StreamHandler()):
                _handler.append(handler)
                LOGGER.removeHandler(handler)
    else:
        LOGGER.addHandler(_handler[0])
        _handler.pop()


def log_warning(message, limit=False):
    increment_warning()
    max_warn = 10
    if limit and WARNINGS_LOGGED >= max_warn:
        if WARNINGS_LOGGED == max_warn:
            LOGGER.warning("maximum number of warnings reached, "
                           "remainder suppressed")
        return
    LOGGER.warning(message.rstrip())


def log_error(message, r=1):
    if r:
        raise Error1(message)
    else:
        LOGGER.error(message)


def increment_warning():
    global WARNINGS_LOGGED
    WARNINGS_LOGGED += 1


def cout(message, end="\n"):
    """Write message to stdout """
    if opts.verbosity:
        sys.__stdout__.write(message + end)
        sys.__stdout__.flush()


def cerr(message):
    """Write message to stderr """
    sys.__stderr__.write(message + "\n")
    sys.__stderr__.flush()


class ExoManager(object):
    """The main IO manager

    """
    time_step = 0
    num_dim = 3
    num_nodes = 8
    num_elem = 1
    coords = np.array([[-1, -1, -1], [ 1, -1, -1],
                       [ 1,  1, -1], [-1,  1, -1],
                       [-1, -1,  1], [ 1, -1,  1],
                       [ 1,  1,  1], [-1,  1,  1]],
                      dtype=np.float64) * .5

    def __init__(self, runid, filepath=None):
        """Instantiate a IOManager object

        runid : str
            The simulation ID

        """
        from utils.exo import ExodusIIFile
        self.runid = runid
        if filepath is not None:
            self.exofile = ExodusIIFile(filepath, mode="a")
            self.filepath = filepath
        else:
            self.exofile = ExodusIIFile(runid, mode="w")
            self.filepath = self.exofile.filename

    def setup_new(self, title, glob_var_names, elem_var_names, info):
        """Set up the exodus file

        """
        from core.restart import write_restart_info
        # "mesh" information
        conn = np.array([range(8)], dtype=np.int)
        num_elem_blk = 1
        num_node_sets = 0
        num_side_sets = 0

        # initialize file with parameters
        title = title.format(str(self.num_dim) + "D")
        self.exofile.put_init(title, self.num_dim, self.num_nodes,
                              self.num_elem, num_elem_blk, num_node_sets,
                              num_side_sets)

        # write nodal coordinates values and names to database
        coord_names = np.array(["COORDX", "COORDY", "COORDZ"])
        self.exofile.put_coord_names(coord_names)
        self.exofile.put_coord(self.coords[:, 0], self.coords[:, 1],
                               self.coords[:, 2])

        # write element block parameters
        num_attr = 0 # for now, we do not use attributes
        elem_blk_id = 1
        elem_blk_els = [0]
        num_elem_this_blk = 1
        elem_type = "HEX"
        num_nodes_per_elem = self.num_nodes
        self.exofile.put_elem_block(elem_blk_id, elem_type, num_elem_this_blk,
                                    num_nodes_per_elem, num_attr)

        # write element connectivity for each element block
        blk_conn = conn[elem_blk_els][:, :num_nodes_per_elem]
        self.exofile.put_elem_conn(elem_blk_id, blk_conn)

        # write QA records
        now = datetime.datetime.now()
        day = now.strftime("%m/%d/%y")
        hour = now.strftime("%H:%M:%S")
        num_qa_rec = 1
        vers = ".".join(str(x) for x in __version__)
        qa_title = "MML {0} simulation".format(vers)
        qa_record = np.array([[qa_title, self.runid, day, hour]])
        self.exofile.put_qa(num_qa_rec, qa_record)

        # write results variables parameters and names
        num_glob_vars = len(glob_var_names)
        self.exofile.put_var_param("g", num_glob_vars)
        self.exofile.put_var_names("g", num_glob_vars, glob_var_names)

        nod_var_names = ["DISPLX", "DISPLY", "DISPLZ"]
        num_nod_vars = len(nod_var_names)
        self.exofile.put_var_param("n", num_nod_vars)
        self.exofile.put_var_names("n", num_nod_vars, nod_var_names)

        num_elem_vars = len(elem_var_names)
        self.exofile.put_var_param("e", num_elem_vars)
        self.exofile.put_var_names("e", num_elem_vars, elem_var_names)

        # write element variable truth table
        truth_tab = np.empty((num_elem_blk, num_elem_vars), dtype=np.int)
        for i in range(num_elem_blk):
            for j in range(num_elem_vars):
                truth_tab[i, j] = 1
        self.exofile.put_elem_var_tab(num_elem_blk, num_elem_vars, truth_tab)

        # write the restart information
        material, driver, extract = info
        write_restart_info(self.exofile, material, driver, extract)

        self.exofile.update()
        pass

    def setup_existing(self, time, time_step, glob_var_vals, elem_var_vals):
        """Initialize some setup

        """
        self.time_step = time_step

    def finish(self):
        # udpate and close the file
        self.exofile.update()
        self.exofile.close()
        return

    def write_data(self, time, glob_var_vals, elem_var_vals, u):
        """Dump information from current time step to results file

        Parameters
        ----------
        time : float
            Current time

        dt : float
            Time step

        glob_var_vals : ndarray
           global variable values

        elem_var_vals : ndarray
           global variable values

        u : array_like
            Nodal displacements

        """
        # write time value
        self.exofile.put_time(self.time_step, time)

        # write global variables
        num_glob_vars = len(glob_var_vals)
        self.exofile.put_glob_vars(self.time_step, num_glob_vars, glob_var_vals)

        # write nodal variables
        num_nod_vars = 3
        for k in range(num_nod_vars):
            self.exofile.put_nodal_var(self.time_step, k, self.num_nodes,
                                       u[k::self.num_dim])

        # write element variables
        elem_blk_id = 1
        num_elem_this_blk = 1
        num_elem_vars = len(elem_var_vals)
        for k in range(num_elem_vars):
            self.exofile.put_elem_var(self.time_step, k, elem_blk_id,
                                      num_elem_this_blk, elem_var_vals.T[k])
            continue

        # udpate and close the file
        self.time_step += 1
        self.exofile.update()

        return

    @classmethod
    def from_existing(cls, runid, filepath, time_step):
        exof = cls(runid, filepath=filepath)
        exof.time_step = time_step
        return exof

    @classmethod
    def new_from_runid(cls, runid, title, glob_var_names, elem_var_names, info):
        exof = cls(runid)
        exof.setup_new(title, glob_var_names, elem_var_names, info)
        return exof
