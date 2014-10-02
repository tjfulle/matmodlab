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

def const_elast_params():
    K = 9.980040E+09
    G = 3.750938E+09
    LAM = K - 2.0 / 3.0 * G
    E   = 9.0 * K * G / (3.0 * K + G)
    NU  = (3.0 * K - 2.0 * G) / (2.0 * (3.0 * K + G))
    return NU, E, K, G, LAM

def gen_analytical_response(LAM, G, nlegs=4, test_type="PRINCIPAL"):
    stiff = (LAM * np.outer(np.array([1,1,1,0,0,0]), np.array([1,1,1,0,0,0])) +
             2.0 * G * np.identity(6))

    rnd = lambda: random.uniform(-0.01, 0.01)
    table = [np.zeros(1 + 6 + 6)]
    for idx in range(1, nlegs):
        if test_type == "FULL":
            strains = np.array([rnd(), rnd(), rnd(), rnd(), rnd(), rnd()])
        elif test_type == "PRINCIPAL":
            strains = np.array([rnd(), rnd(), rnd(), 0.0, 0.0, 0.0])
        elif test_type == "UNIAXIAL":
            strains = np.array([rnd(), 0.0, 0.0, 0.0, 0.0, 0.0])
        elif test_type == "BIAXIAL":
            tmp = rnd()
            strains = np.array([tmp, tmp, 0.0, 0.0, 0.0, 0.0])
        table.append(np.hstack(([idx], strains, np.dot(stiff, strains))))

    # returns a tablewith each row comprised of
    # time=table[0], strains=table[1:7], stresses=table[7:]
    return np.array(table)
