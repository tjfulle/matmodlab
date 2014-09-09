import random
from math import *
import numpy as np
I6 = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])

def gen_rand_elastic_params():
    # poisson_ratio and young's modulus
    nu = random.uniform(-1.0 + 1.0e-5, 0.5 - 1.0e-5)
    E = max(1.0, 10 ** random.uniform(0.0, 12.0))

    # K and G are used for parameterization
    K = E / (3.0 * (1.0 - 2.0 * nu))
    G = E / (2.0 * (1.0 + nu))

    # LAM is used for computation
    LAM = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    return nu, E, K, G, LAM


def gen_rand_surface_params(K, G):
    A1 = 2.0 * G * random.uniform(0.0001, 0.5)
    A4 = np.tan(random.uniform(np.pi / 100.0, np.pi / 2.1))
    return A1, A4


def gen_strain_table(K, G, A1, A4):
    strain = gen_unit_strain(A1, A4)
    gamma = mag(dev(strain))

    # Figure out how much strain we'll need to achieve yield
    fac = A1 / (3. * K * A4 * np.sum(strain[:3]) + sqrt(2.) * G * gamma)
    strain = fac * strain

    strain_table = np.zeros((3, 6))
    strain_table[1] = strain
    strain_table[2] = 2.0 * strain
    return strain_table


def gen_path(K, G, A1, A4):
    strain_table = gen_strain_table(K, G, A1, A4)

    # generate the path (must be a string)
    path = []
    for (t, row) in enumerate(strain_table):
        path.append("{0} 1 222 {1} {2} {3}".format(t, *row[[0,1,2]]))
    path = "\n".join(path)
    return path, strain_table


def gen_unit_strain(A1, A4):
    # Generate a random strain deviator
    straingen = lambda: random.uniform(-1, 1)
    #devstrain = np.array([straingen(), straingen(), straingen(),
    #                      straingen(), straingen(), straingen()])
    devstrain = np.array([straingen(), straingen(), straingen(),
                          0., 0., 0.])
    while mag(dev(devstrain)) == 0.0:
        devstrain = np.array([straingen(), straingen(), straingen(),
                              0., 0., 0.])

    snorm = unit(dev(devstrain))
    return (sqrt(2.0) * A4 * I6 + snorm) / sqrt(6.0 * A4 ** 2 + 1.0)


def gen_analytic_solution(K, G, A1, A4, strain):

    flatten = lambda arg: [x for y in arg for x in y]

    stress = np.zeros((3, 6))
    stress[1] = 3.0 * K * iso(strain[1]) + 2.0 * G * dev(strain[1])
    # Stress stays constant while strain increases
    stress[2] = stress[1]

    state = []
    for i in range(3):
        state.append(flatten([[i], strain[i], stress[i]]))
    state = np.array(state)
    return state[:, [0, 1, 2, 3, 7, 8, 9]]


def iso(A):
    return np.sum(A[:3]) / 3.0 * I6


def dev(A):
    return A - iso(A)


def mag(A):
    return sqrt(np.dot(A[:3], A[:3]) + 2.0 * np.dot(A[3:], A[3:]))


def unit(A):
    return A / mag(A)


def gen_spherical_path(K, MU, RINT, ZINT):
    # Shear strain
    ES = RINT / (2.0 * sqrt(2.0) * MU)

    # Spherical (volumetric) strain
    RNUM = 3.0 * K ** 2 * (RINT / ZINT) ** 2
    DNOM = 3.0 * K * (RINT / ZINT) ** 2 + 2.0 * MU
    TREPS = ZINT / (sqrt(3.0) * K - sqrt(3.0) * RNUM / DNOM)
    EV = TREPS / 3.0

    ##### Stress State
    MAX_SHEAR_STRESS = 2.0 * MU * ES
    MAX_HYDRO_STRESS = ZINT / sqrt(3.0)

    path = """
    0 0 222222     0    0    0    0    0    0
    1 1 111111     0    0    0 {ES}    0    0
    2 1 111111  {EV} {EV} {EV}    0    0    0""".format(EV=EV, ES=ES)

    return path

