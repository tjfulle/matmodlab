#!/usr/bin/env mmd
from matmodlab import *
from utils.exojac.exodiff import rms_error
from utils.exojac.exodump import load_data
from core.test import PASSED, DIFFED, FAILED, DIFFTOL, FAILTOL
from core.tester import RES_MAP

import numpy as np

RUNID = "supreme_linear_elastic"
my_dir = get_my_directory()

#
# This is meant to be a static test for linear elasticity.
# It's primary purpose is to be THE benchmark for linear
# elasticity as it checks each component of stress/strain
# as well as exercises key parts of the driver (like how
# it computes inputs).
#
# For uniaxial strain:
#
#      | a  0  0 |                | exp(a)  0  0 |
#  e = | 0  0  0 |            U = |   0     1  0 |
#      | 0  0  0 |                |   0     0  1 |
#
#  -1  | 1/exp(a)  0  0 |    dU   da | exp(a)  0  0 |
# U  = |    0      1  0 |    -- = -- |   0     0  0 |
#      |    0      0  1 |    dt   dt |   0     0  0 |
#
#         da | 1  0  0 |
# D = L = -- | 0  0  0 |
#         dt | 0  0  0 |
#
#
# For pure shear
#
#      | 0  a  0 |         1         | exp(2a)+1  exp(2a)-1  0 |   | 0  0  0 |
#  e = | a  0  0 |     U = - exp(-a) | exp(2a)-1  exp(2a)+1  0 | + | 0  0  0 |
#      | 0  0  0 |         2         |     0          0      0 |   | 0  0  1 |
#
#
#   -1  1 | exp(-a) + exp(a)  exp(-a) - exp(a)  0 |
#  U  = - | exp(-a) - exp(a)  exp(-a) + exp(a)  0 |
#       2 |         0                 0         2 |
#
#
#  dU   da / | exp(a)  exp(a)  0 |     \
#  -- = -- | | exp(a)  exp(a)  0 | - U |
#  dt   dt \ |   0       0     1 |     /
#
#         da | 0  1  0 |
# D = L = -- | 1  0  0 |
#         dt | 0  0  0 |
#

def get_D_E_F_SIG(dadt, a, LAM, G, loc):
    # This is just an implementation of the above derivations.
    #
    # 'dadt' is the current time derivative of the strain
    # 'a' is the strain at the end of the step
    # 'LAM' and 'G' are the lame and shear modulii
    # 'loc' is the index for what's wanted (0,1) for xy

    if loc[0] == loc[1]:
        # axial
        E = np.zeros((3,3))
        E[loc] = a

        F = np.eye(3)
        F[loc] = np.exp(a)

        D = np.zeros((3,3))
        D[loc] = dadt

        SIG = LAM * a * np.eye(3)
        SIG[loc] = (LAM + 2.0 * G) * a
    else:
        # shear
        l0, l1 = loc

        E = np.zeros((3,3))
        E[l0, l1] = a
        E[l1, l0] = a

        fac = np.exp(-a) / 2.0
        F = np.eye(3)
        F[l0,l0] = fac * (np.exp(2.0 * a) + 1.0)
        F[l1,l1] = fac * (np.exp(2.0 * a) + 1.0)
        F[l0,l1] = fac * (np.exp(2.0 * a) - 1.0)
        F[l1,l0] = fac * (np.exp(2.0 * a) - 1.0)

        D = np.zeros((3,3))
        D[l0,l1] = dadt
        D[l1,l0] = dadt

        SIG = np.zeros((3,3))
        SIG[l0,l1] = 2.0 * G * a
        SIG[l1,l0] = 2.0 * G * a

    return D, E, F, SIG


def generate_solution(solfile):
    a = 0.1                 # total strain increment for each leg
    N = 50                  # number of steps per leg
    LAM = 1.0e9             # Lame modulus
    G = 1.0e9               # Shear modulus
    T = [0.0]               # time
    E = [np.zeros((3,3))]   # strain
    SIG = [np.zeros((3,3))] # stress
    F = [np.eye(3)]         # deformation gradient
    D = [np.zeros((3,3))]   # symmetric part of velocity gradient

    #
    # Generate the analytical solution
    #
    # strains:    xx     yy     zz     xy     xz     yz
    for loc in [(0,0), (1,1), (2,2), (0,1), (0,2), (1,2)]:
        t0 = T[-1]
        tf = t0 + 1.0
        for idx in range(1, N+1):
            fac = float(idx) / float(N)
            ret = get_D_E_F_SIG(a, fac * a, LAM, G, loc)
            T.append(t0 + fac)
            D.append(ret[0])
            E.append(ret[1])
            F.append(ret[2])
            SIG.append(ret[3])

        for idx in range(1, N+1):
            fac = float(idx) / float(N)
            ret = get_D_E_F_SIG(-a, (1.0 - fac) * a, LAM, G, loc)
            T.append(t0 + 1.0 + fac)
            D.append(ret[0])
            E.append(ret[1])
            F.append(ret[2])
            SIG.append(ret[3])

    #
    # Write the output
    #
    headers = ["TIME",
               "STRAIN_XX", "STRAIN_YY", "STRAIN_ZZ",
               "STRAIN_XY", "STRAIN_YZ", "STRAIN_XZ",
               "STRESS_XX", "STRESS_YY", "STRESS_ZZ",
               "STRESS_XY", "STRESS_YZ", "STRESS_XZ",
               "DEFGRAD_XX", "DEFGRAD_XY", "DEFGRAD_XZ",
               "DEFGRAD_YX", "DEFGRAD_YY", "DEFGRAD_YZ",
               "DEFGRAD_ZX", "DEFGRAD_ZY", "DEFGRAD_ZZ",
               "SYMM_L_XX", "SYMM_L_YY", "SYMM_L_ZZ",
               "SYMM_L_XY", "SYMM_L_YZ", "SYMM_L_XZ",
              ]
    symlist = lambda x: [x[0,0], x[1,1], x[2,2], x[0,1], x[1,2], x[0,2]]
    matlist = lambda x: list(np.reshape(x, 9))
    fmtstr = lambda x: "{0:>25s}".format(x)
    fmtflt = lambda x: "{0:25.15e}".format(x)

    with open(solfile, 'w') as FOUT:
        FOUT.write("".join(map(fmtstr, headers)) + "\n")
        for idx in range(0, len(T)):
            vals = ([T[idx]] +
                     symlist(E[idx]) +
                     symlist(SIG[idx]) +
                     matlist(F[idx]) +
                     symlist(D[idx]))
            FOUT.write("".join(map(fmtflt, vals)) + "\n")

    #
    # Pass the relevant data so the sim can run
    #

    # inputs    xx   yy   zz   xy   yz   xz
    path = """
    0 0 222222 0.0  0.0  0.0  0.0  0.0  0.0
    1 1 222222 {0}  0.0  0.0  0.0  0.0  0.0
    2 1 222222 0.0  0.0  0.0  0.0  0.0  0.0
    3 1 222222 0.0  {0}  0.0  0.0  0.0  0.0
    4 1 222222 0.0  0.0  0.0  0.0  0.0  0.0
    5 1 222222 0.0  0.0  {0}  0.0  0.0  0.0
    6 1 222222 0.0  0.0  0.0  0.0  0.0  0.0
    7 1 222222 0.0  0.0  0.0  {0}  0.0  0.0
    8 1 222222 0.0  0.0  0.0  0.0  0.0  0.0
    9 1 222222 0.0  0.0  0.0  0.0  0.0  {0}
   10 1 222222 0.0  0.0  0.0  0.0  0.0  0.0
   11 1 222222 0.0  0.0  0.0  0.0  {0}  0.0
   12 1 222222 0.0  0.0  0.0  0.0  0.0  0.0
    """.format("{0:.1f}".format(a))

    return path, LAM, G



class TestElasticComponents(TestBase):
    def __init__(self):
        self.runid = RUNID
        self.keywords = ["fast", "material", "elastic", "analytic", "defgrad"]

    def setup(self, *args, **kwargs):
        self.make_test_dir()

    def run(self):
        cwd = os.getcwd()
        os.chdir(self.test_dir)
        self.status = runner(d=self.test_dir, runid=self.runid, test=1)
        os.chdir(cwd)
        return self.status


@matmodlab
def runner(d=None, runid=None, test=0):

    d = d or os.getcwd()
    runid = runid or RUNID
    solfile = os.path.join(d, runid + ".base_dat")

    mps = MaterialPointSimulator(runid)

    path, LAM, G = generate_solution(solfile)
    mps.Driver("Continuum", path, step_multiplier=20)

    # set up the material
    K = LAM + 2.0 * G / 3.0
    params = {"K": K, "G": G}
    mps.Material("pyelastic", params)

    # set up and run the model
    mps.run()

    #
    # Compare with solution
    #

    myFAILTOL = FAILTOL
    myDIFFTOL = DIFFTOL
    # check output with analytic
    mps.logger.write("Comaring outputs")
    mps.logger.write("  DIFFTOL = {0:.5e}".format(myDIFFTOL))
    mps.logger.write("  FAILTOL = {0:.5e}".format(myFAILTOL))

    # check output with analytic
    VARIABLES = ["STRAIN_XX", "STRAIN_YY", "STRAIN_ZZ",
                 "STRAIN_XY", "STRAIN_XZ", "STRAIN_YZ",
                 "STRESS_XX", "STRESS_YY", "STRESS_ZZ",
                 "STRESS_XY", "STRESS_XZ", "STRESS_YZ",
                 "DEFGRAD_XX", "DEFGRAD_XY", "DEFGRAD_XZ",
                 "DEFGRAD_YX", "DEFGRAD_YY", "DEFGRAD_YZ",
                 "DEFGRAD_ZX", "DEFGRAD_ZY", "DEFGRAD_ZZ",
                 "SYMM_L_XX", "SYMM_L_YY", "SYMM_L_ZZ",
                 "SYMM_L_XY", "SYMM_L_XZ", "SYMM_L_YZ",
                ]

    analytic_headers, analytic_response = load_data(solfile)
    simulate_response = mps.extract_from_db(VARIABLES, t=1)

    T = analytic_response[:, 0]
    t = simulate_response[:, 0]

    stat = PASSED
    for col in range(1, 1 + len(VARIABLES)):
        X = analytic_response[:, col]
        x = simulate_response[:, col]
        nrms = rms_error(T, X, t, x, disp=0)
        mps.logger.write("  {0:s} NRMS = {1:.5e}".format(VARIABLES[col-1], nrms))
        if nrms < myDIFFTOL:
            mps.logger.write("    PASS")
            continue
        elif nrms < myFAILTOL:
            mps.logger.write("    DIFF")
            if stat is PASSED:
                stat = DIFFED
        else:
            mps.logger.write("    FAIL")
            stat = FAILED

    return stat


if __name__ == "__main__":
    a = runner()
