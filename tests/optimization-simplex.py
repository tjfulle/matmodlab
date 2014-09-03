#!/usr/bin/env xpython
from matmodlab import *

def func(x, *args):

    runid = args[0]

    # set up driver
    path_file = "exmpls.tbl"
    driver = Driver("Continuum", path_file=path_file, cols=[0,2,3,4],
                    cfmt="222", tfmt="time", path_input="table")

    # set up material
    parameters = {"K": x[0], "G": x[1]}
    material = Material("elastic", parameters=parameters)

    # set up and run the model
    mps = MaterialPointSimulator(runid, driver, material, verbosity=0)
    mps.run()

    vars_to_get = ("STRAIN_XX", "STRAIN_YY", "STRAIN_ZZ",
                   "STRESS_XX", "STRESS_YY", "STRESS_ZZ")

    # read in baseline data
    aux = "opt-baseline.dat"
    auxhead = open(aux).readline().split()
    auxdat = np.loadtxt(aux, skiprows=1)
    I = [auxhead.index(var) for var in vars_to_get]
    baseevol = auxdat[:, I[0]] + auxdat[:, I[1]] + auxdat[:, I[2]]
    basepress = (auxdat[:, I[3]] + auxdat[:, I[4]] + auxdat[:, I[5]]) / 3.
    basetime = auxdat[:, auxhead.index("TIME")]

    # read in output data
    simdat = mps.extract_from_db(vars_to_get, t=1)
    simtime = simdat[:, 0]

    simevol = simdat[:, 1] + simdat[:, 2] + simdat[:, 3]
    simpress = -(simdat[:, 4] + simdat[:, 5] + simdat[:, 6]) / 3.

    # do the comparison
    n = basetime.shape[0]
    t0 = max(np.amin(basetime), np.amin(simtime))
    tf = min(np.amax(basetime), np.amax(simtime))
    evb = lambda x: np.interp(x, basetime, baseevol)
    evs = lambda x: np.interp(x, simtime, simevol)

    base = lambda x: np.interp(evb(x), baseevol, basepress)
    comp = lambda x: np.interp(evs(x), simevol, simpress)

    rms = np.sqrt(np.mean([(base(t) - comp(t)) ** 2
                           for t in np.linspace(t0, tf, n)]))
    dnom = np.amax(np.abs(simpress))
    if dnom < 1.e-12: dnom = 1.
    error = rms / dnom

    return error

@matmodlab
def runner():
    runid = "optimization-simplex"
    K = OptimizedVariable("K", 129e9, bounds=(125e9, 150e9))
    G = OptimizedVariable("G", 54e9, bounds=(45e9, 57e9))
    print K
    print G
    xinit = [K, G]
    exit("finish me")
    optimizer = Optimizer(func, xinit, runid=runid,
                          respdesc=["PRES_V_EVOL"], method="simplex",
                          maxiter=25, tolerance=1.e-4, funcargs=(runid,))
    optimizer.run()

runner()
