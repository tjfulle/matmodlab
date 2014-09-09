import random
import numpy as np

def get_stress(e11, e22, e33, e12, e23, e13, LAM, G):
    #standard hooke's law
    sig11 = (2.0 * G + LAM) * e11 + LAM * (e22 + e33)
    sig22 = (2.0 * G + LAM) * e22 + LAM * (e11 + e33)
    sig33 = (2.0 * G + LAM) * e33 + LAM * (e11 + e22)
    sig12 = 2.0 * G * e12
    sig23 = 2.0 * G * e23
    sig13 = 2.0 * G * e13
    return sig11, sig22, sig33, sig12, sig23, sig13

def gen_rand_elast_params():
    # poisson_ratio and young's modulus
    nu = random.uniform(-1.0 + 1.0e-5, 0.5 - 1.0e-5)
    E = max(1.0, 10 ** random.uniform(0.0, 12.0))

    # K and G are used for parameterization
    K = E / (3.0 * (1.0 - 2.0 * nu))
    G = E / (2.0 * (1.0 + nu))

    # LAM is used for computation
    LAM = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))

    return nu, E, K, G, LAM

def gen_analytical_response(LAM, G):
    N = 100
    strain_fac = 0.01
    #               time   e11   e22   e33
    strain_table = [[0.0, 0.0, 0.0, 0.0]]
    for idx in range(1, 6):
        strain_table.append([float(idx), random.uniform(-strain_fac, strain_fac),
                                         random.uniform(-strain_fac, strain_fac),
                                         random.uniform(-strain_fac, strain_fac)])

    expanded = [[_] for _ in strain_table[0]]
    for idx in range(0, len(strain_table) - 1):
        for jdx in range(0, len(strain_table[0])):
            start = strain_table[idx][jdx]
            end = strain_table[idx + 1][jdx]
            expanded[jdx] = expanded[jdx] + list(np.linspace(start, end, N))[1:]

    table = []
    for idx in range(0, len(expanded[0])):
        t = expanded[0][idx]
        e1 = expanded[1][idx]
        e2 = expanded[2][idx]
        e3 = expanded[3][idx]
        sig = get_stress(e1, e2, e3, 0.0, 0.0, 0.0, LAM, G)
        sig11, sig22, sig33, sig12, sig23, sig13 = sig
        table.append([t, e1, e2, e3, sig11, sig22, sig33])
    return np.array(table)


