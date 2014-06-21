import os
import sys
import numpy as np
import datetime
import logging
from mmlinc import *

import __config__ as cfg


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
    LOGGER.warning(message)


def log_error(message, r=1):
    if r:
        raise Error1(message)
    else:
        LOGGER.error(message)


def increment_warning():
    global WARNINGS_LOGGED
    WARNINGS_LOGGED += 1


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
        vers = ".".join(str(x) for x in cfg.__version__)
        qa_title = "MML {0} simulation".format(vers)
        qa_record = np.array([[qa_title, self.runid, day, hour]])
        self.exofile.put_qa(num_qa_rec, qa_record)

        # information records
        material, driver, extract = info
        self.write_mat_params(*material)
        self.write_driver_info(*driver)
        if extract:
            self.write_extract_info(*extract)

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

    def write_mat_params(self, mat_name, param_names, param_vals):
        """Write an array to the database

        Parameters
        ----------
        mat_name : str
            The material name

        param_names : list of str
            List of parameter names

        param_vals : array_like
            Array of material parameters

        """
        import utils.exo.exoinc as ei
        db = self.exofile.db

        name = NAME_MAT_NAME(1)
        db.createVariable(name, ei.DTYPE_TXT, (ei.DIM_STR, ))
        db.variables[name][:] = ei.format_string(mat_name)

        # put the names
        num_params = len(param_vals)
        param_names = ei.format_list_of_strings(param_names)
        dim = DIM_MAT_PARAM(mat_name)
        name = NAME_MAT_PARAM(mat_name)
        vals = VALS_MAT_PARAM(mat_name)
        db.createDimension(dim, num_params)
        db.createVariable(name, ei.DTYPE_TXT, (dim, ei.DIM_STR))
        db.createVariable(vals, ei.DTYPE_FLT, (dim,))
        for (i, param_name) in enumerate(param_names):
            db.variables[name][i, :] = param_name

        # put the values
        db.variables[vals][:num_params] = param_vals
        return

    def write_driver_info(self, name, path, options):
        """Write the driver information to the exodus file

        Parameters
        ----------
        name : str
            The driver name

        path : array_like
            Deformation path

        options : array_like
            List of driver options

        """
        import utils.exo.exoinc as ei
        db = self.exofile.db

        db.createVariable(NAME_DRIVER, ei.DTYPE_TXT, (ei.DIM_STR, ))
        db.variables[NAME_DRIVER][:] = ei.format_string(name)

        # write the legs
        dim = DIM_NUM_LEG_COMP(name)
        num_legs, num_comps = path.shape
        db.createDimension(dim, num_comps)
        db.createDimension(DIM_NUM_LEG(name), num_legs)
        for (i, leg) in enumerate(path):
            var = VALS_LEG(name, i+1)
            db.createVariable(var, ei.DTYPE_FLT, (dim,))
            db.variables[var][:num_comps] = leg[:]

        # write the options
        dim = DIM_NUM_DRIVER_OPTS(name)
        vals = VALS_DRIVER_OPTS(name)
        num_opts = len(options)
        db.createDimension(dim, num_opts)
        db.createVariable(vals, ei.DTYPE_FLT, (dim, ))
        db.variables[vals][:] = options

        return

    def write_extract_info(self, ofmt, step, ffmt, variables, paths):
        """Write the extraction information to the exodus file

        Parameters
        ----------
        ofmt : str
            The output format

        step : int

        ffmt : str
            Float format

        variables : list of str
            List of variables to extract

        paths : list
            List of paths to extract

        """
        import utils.exo.exoinc as ei

        # write the format
        db = self.exofile.db
        db.createVariable(VALS_EXTRACT_FMT, ei.DTYPE_TXT, (ei.DIM_STR, ))
        db.variables[VALS_EXTRACT_FMT][:] = ei.format_string(ofmt)

        db.createDimension(DIM_NUM_EXTRACT_STEP, 1)
        db.createVariable(VALS_EXTRACT_STEP, ei.DTYPE_INT, (DIM_NUM_EXTRACT_STEP,))
        db.variables[VALS_EXTRACT_STEP][:] = step

        db.createVariable(VALS_EXTRACT_FFMT, ei.DTYPE_TXT, (ei.DIM_STR, ))
        db.variables[VALS_EXTRACT_FFMT][:] = ei.format_string(ffmt)

        num_vars = len(variables)
        if num_vars:
            variables = ei.format_list_of_strings(variables)
            dim = DIM_NUM_EXTRACT_VARS
            db.createDimension(dim, num_vars)
            db.createVariable(NAME_EXTRACT_VARS, ei.DTYPE_TXT, (dim, ei.DIM_STR))
            # put the names
            for (i, variable) in enumerate(variables):
                db.variables[NAME_EXTRACT_VARS][i, :] = variable

        num_paths = len(paths)
        if num_paths:
            dim_a = DIM_NUM_EXTRACT_PATHS
            dim_b = DIM_NUM_EXTRACT_PATHS_COMPS
            vals = VALS_EXTRACT_PATHS
            db.createDimension(dim_a, num_paths)
            db.createDimension(dim_b, 4)
            for path in paths:
                db.variables[vals][0, :] = path[0]     # number of increments
                db.variables[vals][1, :] = path[1][0]  # list of density bounds
                db.variables[vals][2, :] = path[1][1]  # list of density bounds
                db.variables[vals][3, :] = path[2]     # initial temperature

        return

def read_restart(filepath, time=-1):
    """Read information records written to exodus database for setting up a
    simulation

    Parameters
    ----------
    filepath : str
        Path to exodus file

    time : float
        Time from which to restart

    Returns
    -------
    all_ex_info : list of strings
        Formatted list of strings to put in exodus file

    """
    from drivers.driver import create_driver
    from utils.exo import ExodusIIFile
    raise Error1("reading restart not done")

    runid = os.path.splitext(os.path.basename(filepath))[0]
    if not os.path.isfile(filepath):
        raise Error1("{0}: no such file".format(filepath))
    exof = ExodusIIFile(filepath, "r")

    all_ex_info = exof.get_info()
    all_ex_info = all_ex_info.tolist()

    try:
        start = all_ex_info.index(S_MML_DECL)
    except ValueError:
        return
    end = all_ex_info.index(S_MML_FINI)

    ex_info = all_ex_info[start+1:end]

    rvers = int(ex_info[ex_info.index(S_REST_VERS)+1])
    if rvers != RESTART_VERSION:
        raise Error1("restart file version mismatch "
                     "({0}!={1})".format(rvers, RESTART_VERSION))

    # read material information
    mtlname = ex_info[ex_info.index(S_MTL)+1]
    i = ex_info.index(S_MTL_PARAMS)
    nparams = int(ex_info[i+1])
    mtlparams = np.array([float(x) for x in ex_info[i+2:i+nparams+2]])

    # driver
    dname = ex_info[ex_info.index(S_DRIVER)+1]

    i = ex_info.index(S_DOPTS)
    ndopts = int(ex_info[i+1])
    dopts = [float(x) for x in ex_info[i+2:i+2+ndopts]]

    # paths
    # i is the index to the S_PATH information keyword
    # i + j is the index to the beginning of each path specification
    i = ex_info.index(S_PATH)
    rw = int(ex_info[i+1])
    cl = int(ex_info[i+2])
    dpath = np.reshape([float(x) for x in ex_info[i+3:i+3+rw*cl]], (rw, cl))

    # extraction requests
    i = ex_info.index(S_EXREQ)
    extract = None
    if int(ex_info[i+1]):

        ofmt = ex_info[i+2]
        step = int(ex_info[i+3])
        ffmt = ex_info[i+4]

        nvars = int(ex_info[i+5])
        start = i + 6
        end = start + nvars
        variables = ex_info[start:end]

        npaths = int(ex_info[end])
        start = end + 1
        paths = []
        for i in range(npaths):
            ninc = int(ex_info[start])
            r0, rf = float(ex_info[start+1]), float(ex_info[start+2])
            t0 = float(ex_info[start+3])
            start = start + 4
            paths.append([ninc, np.array([r0, rf]), t0])

        extract = [ofmt, step, ffmt, variables, paths]

    # driver is now set up, find step number corresponding to time and then
    # get all data from that step
    times = exof.get_all_times()
    path_times = dpath[:, 0]
    if time < 0:
        time = times[-1]

    try:
        leg_num = np.where(time < path_times)[0][0]
    except IndexError:
        raise SystemExit("All legs completed, nothing to do")
    path_time = path_times[leg_num]
    step = np.where(path_time > times)[0][-1]

    time = float(times[step])
    glob_data = exof.get_glob_vars(step)
    elem_data = []
    for elem_var_name in exof.elem_var_names():
        elem_data.extend(exof.get_elem_var(step, elem_var_name)[0])
    elem_data = np.array(elem_data)

    exof.close()

    return (runid, mtlname, mtlparams, dname, dpath, dopts,
            leg_num, time, glob_data, elem_data, extract)
