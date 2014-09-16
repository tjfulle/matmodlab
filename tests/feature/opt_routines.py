#!/usr/bin/env python
"""Provides a recipe for computing the maximum error between the expected
pressure vs. volume strain slope and the simulated.

"""
import os
import sys
import numpy as np
D = os.path.dirname(os.path.realpath(__file__))
CCHAR = "#"

from utils.exojac import ExodusIIFile

def opt_pres_v_evol(exof):

    vars_to_get = ("STRAIN_XX", "STRAIN_YY", "STRAIN_ZZ",
                   "STRESS_XX", "STRESS_YY", "STRESS_ZZ")

    # read in baseline data
    aux = os.path.join(D, "opt.base_dat")
    auxhead, auxdat = loadtxt(aux)
    I = [auxhead[var] for var in vars_to_get]
    baseevol = auxdat[:, I[0]] + auxdat[:, I[1]] + auxdat[:, I[2]]
    basepress = (auxdat[:, I[3]] + auxdat[:, I[4]] + auxdat[:, I[5]]) / 3.
    basetime = auxdat[:, auxhead["TIME"]]

    # read in output data
    exof = ExodusIIFile(exof)
    simdat = np.transpose([exof.get_elem_var_time(var, 0) for var in vars_to_get])
    simtime = exof.get_all_times()
    simevol = simdat[:, 0] + simdat[:, 1] + simdat[:, 2]
    simpress = -(simdat[:, 3] + simdat[:, 4] + simdat[:, 5]) / 3.

    # do the comparison
    n = basetime.shape[0]
    t0 = max(np.amin(basetime), np.amin(simtime))
    tf = min(np.amax(basetime), np.amax(simtime))
    evb = lambda x: np.interp(x, basetime, baseevol)
    evs = lambda x: np.interp(x, simtime, simevol)

    base = lambda x: np.interp(evb(x), baseevol, basepress)
    comp = lambda x: np.interp(evs(x), simevol, simpress)

    dnom = np.amax(np.abs(simpress))
    if dnom < 1.e-12: dnom = 1.

    error = np.sqrt(np.mean([((base(t) - comp(t)) / dnom) ** 2
                             for t in np.linspace(t0, tf, n)]))
    return error


def opt_sig_v_time(exof):
    vars_to_get = ("STRESS_XX", "STRESS_YY", "STRESS_ZZ")

    # read in baseline data
    aux = os.path.join(D, "opt.base_dat")
    auxhead, auxdat = loadtxt(aux)
    I = np.array([auxhead[var] for var in vars_to_get], dtype=np.int)
    basesig = auxdat[:, I]
    basetime = auxdat[:, auxhead["TIME"]]

    # read in output data
    exof = ExodusIIFile(exof)
    simtime = exof.get_all_times()
    simsig = np.transpose([exof.get_elem_var_time(var, 0) for var in vars_to_get])

    # do the comparison
    error = -1
    t0 = max(np.amin(basetime), np.amin(simtime))
    tf = min(np.amax(basetime), np.amax(simtime))
    n = basetime.shape[0]
    for idx in range(3):
        base = lambda x: np.interp(x, basetime, basesig[:, idx])
        comp = lambda y: np.interp(y, simtime, simsig[:, idx])
        dnom = np.amax(np.abs(simsig[:, idx]))
        if dnom < 1.e-12: dnom = 1.
        rms = np.sqrt(np.mean([((base(t) - comp(t)) / dnom) ** 2
                               for t in np.linspace(t0, tf, n)]))
        error = max(rms, error)
        continue

    return error


def loadtxt(filename):
    head = open(filename).readline().strip(CCHAR).split()
    head = dict([(a, i) for (i, a) in enumerate(head)])
    data = np.loadtxt(filename, skiprows=1)
    return head, data
