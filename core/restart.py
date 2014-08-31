import os
import numpy as np

if __name__ == "__main__":
    import sys
    from os.path import dirname, realpath
    sys.path.insert(0, dirname(dirname(realpath(__file__))))

from mml import __version__
from utils.exo.exoinc import *


RESTART_VERSION = 2
ATT_RESTART_VERSION = "restart_version"

DIM_NUM_MAT = "num_mat"
NAME_MAT_NAME = lambda i: "mat{0}".format(i)
DIM_MAT_PARAM = lambda s: "num_{0}_param".format(s)
NAME_MAT_PARAM = lambda s: "name_{0}_param".format(s)
VALS_MAT_PARAM = lambda s: "vals_{0}_param".format(s)

DIM_NUM_LEG = lambda s: "num_{0}_legs".format(s)
DIM_NUM_LEG_COMP = lambda s: "num_{0}_legs_comps".format(s)
VALS_LEGS = lambda s: "vals_{0}_legs".format(s)

DIM_NUM_DRIVER = "num_driver"
NAME_DRIVER_NAME = lambda i: "driver{0}".format(i)
DIM_NUM_DRIVER_OPTS = lambda s: "num_{0}_opts".format(s)
VALS_DRIVER_OPTS = lambda s: "vals_{0}_opts".format(s)

VALS_EXTRACT_FMT = "vals_extract_format"
VALS_EXTRACT_STEP = "vals_extract_step"
DIM_NUM_EXTRACT_STEP = "num_extract_step"
VALS_EXTRACT_FFMT = "vals_extract_float_format"
NAME_EXTRACT_VARS = "name_extract_vars"
DIM_NUM_EXTRACT = "num_extract"
DIM_NUM_EXTRACT_VARS = "num_extract_vars"
DIM_NUM_EXTRACT_PATHS = "num_extract_paths"
DIM_NUM_EXTRACT_PATHS_COMPS = "num_extract_paths_comps"
VALS_EXTRACT_PATHS = "vals_extract_paths"

class RestartError(Exception):
    def __init__(self, message):
        self.message = message
        super(RestartError, self).__init__(message)


def assert_restart_version(db):
    restart_version = getattr(db, ATT_RESTART_VERSION)
    if restart_version != RESTART_VERSION:
        raise RestartError("restart version mismatch.  expected "
                           "{0} got {1}".format(RESTART_VERSION, restart_version))


def write_restart_info(exofile, material_info, driver_info, extract_info):
    db = exofile.db

    setattr(db, ATT_RESTART_VERSION, RESTART_VERSION)

    mat_name, param_names, param_vals = material_info
    write_mat_info(db, mat_name, param_names, param_vals)

    driver_name, driver_path, driver_opts = driver_info
    write_driver_info(db, driver_name, driver_path, driver_opts)

    if extract_info:
        ofmt, step, ffmt, vars, paths = extract_info
        write_extract_info(db, ofmt, step, ffmt, vars, paths)


def read_restart_info(filepath, time):
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
    from utils.exo import ExodusIIFile
    if not os.path.isfile(filepath):
        raise RestartError("{0}: no such file".format(filepath))

    runid = os.path.splitext(os.path.basename(filepath))[0]
    exof = ExodusIIFile(filepath, "r")

    mat_name, param_names, param_vals = read_mat_info(exof.db)
    driver_name, driver_path, driver_opts = read_driver_info(exof.db)
    extract = read_extract_info(exof.db)
    if extract:
        ofmt, step, ffmt, variables, paths = extract

    # get the leg number for the time
    leg_num = driver_path.shape[0]
    if time >= 0.:
        for (i, leg) in enumerate(driver_path):
            if leg[0] > time:
                leg_num = i
                break
    leg_num -= 1
    leg_time = exof.get_time(leg_num)

    glob_data = exof.get_glob_vars(leg_num)
    elem_data = []
    for elem_var in exof.elem_var_names:
        elem_data.extend(exof.get_elem_var(leg_num, elem_var)[0])
    elem_data = np.array(elem_data)
    exof.close()

    return (runid, mat_name, param_vals,
            driver_name, driver_path, driver_opts,
            leg_num, leg_time, glob_data, elem_data, extract)


def write_mat_info(db, mat_name, param_names, param_vals):
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
    db.createDimension(DIM_NUM_MAT, 1)
    uniq_name = NAME_MAT_NAME(1)
    db.createVariable(uniq_name, DTYPE_TXT, (DIM_STR, ))
    db.variables[uniq_name][:] = format_string(mat_name)

    # put the names
    num_params = len(param_vals)
    param_names = format_list_of_strings(param_names)
    dim = DIM_MAT_PARAM(uniq_name)
    name = NAME_MAT_PARAM(uniq_name)
    vals = VALS_MAT_PARAM(uniq_name)
    db.createDimension(dim, num_params)
    db.createVariable(name, DTYPE_TXT, (dim, DIM_STR))
    db.createVariable(vals, DTYPE_FLT, (dim,))
    for (i, param_name) in enumerate(param_names):
        db.variables[name][i, :] = param_name

    # put the values
    db.variables[vals][:num_params] = param_vals
    return


def read_mat_info(db):
    """Read material information from the exodus database

    Parameters
    ----------
    db : object
        The exodus_file db attribute

    Returns
    -------
    mat_name : str
        The material name

    param_names : list of str
        List of parameter names

    param_vals : array_like
        Array of material parameters

    Notes
    -----
    Companion to write_mat_info.  See it for comments

    """
    assert_restart_version(db)

    num_mat = db.dimensions[DIM_NUM_MAT]

    # get the parameters
    uniq_name = NAME_MAT_NAME(1)
    mat_name = chara_to_text(db.variables[uniq_name][:])
    vals = np.array(db.variables[VALS_MAT_PARAM(uniq_name)][:])
    names = chara_to_text(db.variables[NAME_MAT_PARAM(uniq_name)][:], aslist=1)

    return mat_name, names, vals


def write_driver_info(db, name, path, options):
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
    db.createDimension(DIM_NUM_DRIVER, 1)
    uniq_name = NAME_DRIVER_NAME(1)
    db.createVariable(uniq_name, DTYPE_TXT, (DIM_STR, ))
    db.variables[uniq_name][:] = format_string(name)

    # write the legs
    num_legs, num_comps = path.shape

    dim_a = DIM_NUM_LEG(uniq_name)
    db.createDimension(dim_a, num_legs)

    dim_b = DIM_NUM_LEG_COMP(uniq_name)
    db.createDimension(dim_b, num_comps)

    var = VALS_LEGS(uniq_name)
    db.createVariable(var, DTYPE_FLT, (dim_a, dim_b))
    for (i, leg) in enumerate(path):
        db.variables[var][i, :num_comps] = leg[:]

    # write the options
    dim = DIM_NUM_DRIVER_OPTS(uniq_name)
    vals = VALS_DRIVER_OPTS(uniq_name)
    num_opts = len(options)
    num_opts = 3
    opts = [options["restart"], options["kappa"], options["proportional"]]
    db.createDimension(dim, num_opts)
    db.createVariable(vals, DTYPE_FLT, (dim, ))
    # tjf: not right, since options is a dictionary, values won't be in order
    db.variables[vals][:] = opts

    return


def read_driver_info(db):
    """Read the driver information from the exodus file

    Parameters
    ----------
    db : object
        The exodus_file db attribute

    Returns
    ----------
    name : str
        The driver name

    path : array_like
        Deformation path

    options : array_like
        List of driver options

    Notes
    -----
    Companion to write_driver_info.  See it for comments

    """
    assert_restart_version(db)

    num_driver = db.dimensions[DIM_NUM_DRIVER]
    uniq_name = NAME_DRIVER_NAME(1)
    driver_name = chara_to_text(db.variables[uniq_name][:])

    # read the legs
    num_legs = db.dimensions[DIM_NUM_LEG(uniq_name)]
    num_comps = db.dimensions[DIM_NUM_LEG_COMP(uniq_name)]
    path = np.empty((num_legs, num_comps))
    path[:] = db.variables[VALS_LEGS(uniq_name)][:]

    # read the options
    num_opts = db.dimensions[DIM_NUM_DRIVER_OPTS(uniq_name)]
    options = np.empty(num_opts)
    options[:] = db.variables[VALS_DRIVER_OPTS(uniq_name)][:]
    options = options.tolist()

    return driver_name, path, options


def write_extract_info(db, ofmt, step, ffmt, variables, paths):
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
    db.createDimension(DIM_NUM_EXTRACT, 1)

    # write the format
    db.createVariable(VALS_EXTRACT_FMT, DTYPE_TXT, (DIM_STR, ))
    db.variables[VALS_EXTRACT_FMT][:] = format_string(ofmt)

    db.createDimension(DIM_NUM_EXTRACT_STEP, 1)
    db.createVariable(VALS_EXTRACT_STEP, DTYPE_INT, (DIM_NUM_EXTRACT_STEP,))
    db.variables[VALS_EXTRACT_STEP][:] = step

    db.createVariable(VALS_EXTRACT_FFMT, DTYPE_TXT, (DIM_STR, ))
    db.variables[VALS_EXTRACT_FFMT][:] = format_string(ffmt)

    num_vars = len(variables)
    if num_vars:
        variables = format_list_of_strings(variables)
        dim = DIM_NUM_EXTRACT_VARS
        db.createDimension(dim, num_vars)
        db.createVariable(NAME_EXTRACT_VARS, DTYPE_TXT, (dim, DIM_STR))
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
        # tjfulle: this is wrong. we just keep overwriting what we put in the
        # database
        for path in paths:
            db.variables[vals][0, :] = path[0]     # number of increments
            db.variables[vals][1, :] = path[1][0]  # list of density bounds
            db.variables[vals][2, :] = path[1][1]  # list of density bounds
            db.variables[vals][3, :] = path[2]     # initial temperature

    return


def read_extract_info(db):
    """Read the extraction information from the exodus file

    Parameters
    ----------
    db : object
        The exodus_file db attribute

    Returns
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

    Notes
    -----
    Companion to write_extract_info.  See it for comments

    """
    num_extract = db.dimensions.get(DIM_NUM_EXTRACT)
    if not num_extract:
        return

    ofmt = chara_to_text(db.variables[VALS_EXTRACT_FMT][:])
    step = db.variables[VALS_EXTRACT_STEP][:][0]
    ffmt = chara_to_text(db.variables[VALS_EXTRACT_FFMT][:])

    variables = []
    num_vars = db.dimensions.get(DIM_NUM_EXTRACT_VARS)
    if num_vars:
        variables = format_list_of_strings(variables)
        variables = chara_to_text(db.variables[NAME_EXTRACT_VARS][:])

    num_paths = db.dimensions.get(DIM_NUM_EXTRACT_PATHS, 0)
    paths = []
    for i in range(num_paths):
        # tjfulle: this is wrong, so is the write
        p = db.variables[VALS_EXTRACT_PATHS][:]
        paths.append([p[0], (p[1], p[2]), p[3]])

    return ofmt, step, ffmt, variables, paths


if __name__ == "__main__":
    restart_info = read_restart_info(sys.argv[1], 4.53)
