import numpy as np

from _exoconst import *
from exoreader import ExodusIIReader

from __config__ import __version__

RESTART_VERSION = 1

S_GMD_DECL = "GMD: START INFORMATION RECORD"
S_GMD_FINI = "GMD: END INFORMATION RECORD"
S_REST_VERS = "RESTART VERSION"
S_MTL = "MATERIAL"
S_MTL_PARAMS = "MATERIAL PARAMETERS"
S_DRIVER = "DRIVER"
S_PATHS = "PATHS"
S_SURFS = "SURFACES"
S_EXREQ = "EXTRACTION REQUESTS"

class RestartError(Exception):
    def __init__(self, message):
        self.message = message
        super(RestartError, self).__init__(message)


def format_exrestart_info(mtlname, mtlparams, dname, kappa, dpath, extract):
    """Format information records to be written to exodus database for reading in
    later

    Parameters
    ----------
    mtlname : str
        Material model name
    mtlparams : ndarray
        Array of material parameters
    driver : string
        Name of driver
    paths : list
        list holding driver paths
    extract : list
        List of extraction info

    Returns
    -------
    ex_info : list of strings
        Formatted list of strings to put in exodus file

    """
    ex_info = []
    ex_info.append(S_GMD_DECL)
    ex_info.append(S_REST_VERS)
    ex_info.append(RESTART_VERSION)

    ex_info.append(S_MTL)
    ex_info.append(mtlname)

    ex_info.append(S_MTL_PARAMS)
    ex_info.append(len(mtlparams))
    ex_info.extend(mtlparams)

    ex_info.append(S_DRIVER)
    ex_info.append(dname)
    ex_info.append(kappa)

    ex_info.append(S_PATHS)
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
        ex_info.extend([p.toxml() for p in paths])

    ex_info.append(S_GMD_FINI)

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
        start = all_ex_info.index(S_GMD_DECL)
    except ValueError:
        return
    end = all_ex_info.index(S_GMD_FINI)

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
    kappa = float(ex_info[ex_info.index(S_DRIVER)+2])

    # paths
    # i is the index to the S_PATHS information keyword
    # i + j is the index to the beginning of each path specification
    dpaths = {}
    i = ex_info.index(S_PATHS)
    npaths = int(ex_info[i+1])
    j = 2
    for n in range(npaths):
        path = []
        key = int(ex_info[i+j])
        nrow = int(ex_info[i+j+1])
        ncol = int(ex_info[i+j+2])
        start = i + j + 3
        end = start + nrow * ncol
        step = ncol
        for l in range(start, end, step):
            path.append([float(x) for x in ex_info[l:l+step]])
        dpaths[key] = np.array(path)
        j = l

    # surfaces
    # i is the index to the S_SURFS information keyword
    # i + j is the index to the beginning of each path specification
    dsurfaces = {}
    i = ex_info.index(S_SURFS)
    nsurfs = int(ex_info[i+1])
    j = 2
    for n in range(nsurfs):
        surf = []
        key = int(ex_info[i+j])
        nrow = int(ex_info[i+j+1])
        ncol = int(ex_info[i+j+2])
        start = i + j + 3
        end = start + nrow * ncol
        step = ncol
        for l in range(start, end, step):
            surf.append([float(x) for x in ex_info[l:l+step]])
        dsurfaces[key] = np.array(surf)
        j = l

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
        end = start + npaths
        paths = ex_info[start:end]

        extract = [ofmt, step, ffmt, variables, paths]

    # set up and register paths with driver
    driver = create_driver(dname)
    driver.register_paths_and_surfaces(paths=dpaths, surfaces=dsurfaces)

    # driver is now set up, find step number corresponding to time and then
    # get all data from that step
    times = exof.get_all_times()
    path_times = dpaths[0][:, 0]
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

    return (mtlname, mtlparams, driver, kappa, extract, leg_num,
            time, glob_data, elem_data)
