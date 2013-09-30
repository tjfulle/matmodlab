import numpy as np

from _exoconst import *
from exoreader import ExodusIIReader

from __config__ import __version__

RESTART_VERSION = 1

S_MML_DECL = "MML: START INFORMATION RECORD"
S_MML_FINI = "MML: END INFORMATION RECORD"
S_REST_VERS = "RESTART VERSION"
S_MTL = "MATERIAL"
S_MTL_PARAMS = "MATERIAL PARAMETERS"
S_DRIVER = "DRIVER"
S_PATH = "PATH"
S_DOPTS = "DRIVER_OPTS"
S_EXREQ = "EXTRACTION REQUESTS"

class RestartError(Exception):
    def __init__(self, message):
        self.message = message
        super(RestartError, self).__init__(message)


def format_exrestart_info(material, driver, extract):
    """Format information records to be written to exodus database for reading in
    later

    Parameters
    ----------
    material : tuple of str
        material[0] - mtlname, Material model name
        material[1] - mtlparams, Array of material parameters
    driver : list
        Name of driver, opts, path
    extract : list
        List of extraction info

    Returns
    -------
    ex_info : list of strings
        Formatted list of strings to put in exodus file

    """
    ex_info = []
    ex_info.append(S_MML_DECL)
    ex_info.append(S_REST_VERS)
    ex_info.append(RESTART_VERSION)

    mtlname, mtlparams = material
    ex_info.append(S_MTL)
    ex_info.append(mtlname)

    ex_info.append(S_MTL_PARAMS)
    ex_info.append(len(mtlparams))
    ex_info.extend(mtlparams)

    dname, dpath, dopts = driver
    ex_info.append(S_DRIVER)
    ex_info.append(dname)

    ex_info.append(S_DOPTS)
    ex_info.append(len(dopts))
    ex_info.extend(dopts)

    ex_info.append(S_PATH)
    ex_info.append(dpath.shape[0])
    ex_info.append(dpath.shape[1])
    [ex_info.extend(line) for line in dpath]

    ex_info.append(S_EXREQ)
    if not extract:
        ex_info.append(0)
    else:
        ex_info.append(1)
        ofmt, step, ffmt, variables, paths = extract[:5]
        ex_info.append(ofmt)
        ex_info.append(step)
        ex_info.append(ffmt)
        ex_info.append(len(variables))
        ex_info.extend(variables)
        ex_info.append(len(paths))
        for path in paths:
            ex_info.append(path[0]) # nincrement
            ex_info.extend(path[1]) # list of density bounds (len = 2)
            ex_info.append(path[2]) # initial temperature

    ex_info.append(S_MML_FINI)

    return ["{0:{1}s}".format(str(x).strip(), MAX_LINE_LENGTH)[:MAX_LINE_LENGTH]
            for x in ex_info]


def read_exrestart_info(filepath, time=-1):
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

    if not os.path.isfile(filepath):
        raise RestartError("{0}: no such file".format(auxfile))
    exof = ExodusIIReader.new_from_exofile(filepath)
    all_ex_info = exof.get_info()

    try:
        start = all_ex_info.index(S_MML_DECL)
    except ValueError:
        return
    end = all_ex_info.index(S_MML_FINI)

    ex_info = all_ex_info[start+1:end]

    rvers = int(ex_info[ex_info.index(S_REST_VERS)+1])
    if rvers != RESTART_VERSION:
        raise RestartError("restart file version mismatch "
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

    return (mtlname, mtlparams, dname, dpath, dopts,
            leg_num, time, glob_data, elem_data, extract)
