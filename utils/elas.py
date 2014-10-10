#!/usr/bin/env python

'''
NAME
    pyelas

BACKGROUND
    This is a python implementation of a fortran program by Rebecca Brannon
    (brannon@mech.utah.edu)

PURPOSE
    Given any two INDEPENDENT elastic constants (from the list below) this
    program will compute all of the others. This program also contains formulas
    that will give you an estimate of how the elastic properties will change when
    the material contains pores or cracks. If you don't care about this
    information, simply enter zero (0) when asked to provide a damage measure (or
    just hit return because undamaged properties are the default). If you ARE
    interested in getting the changes in elastic properties that result from
    PORES, then first enter any two independent elastic properties of the
    nonporous matrix material, and then enter the pore volume fraction when asked
    for the damage measure. If you seek the changes in elastic properties
    resulting from CRACKS, then specify the NEGATIVE of the number of cracks per
    unit volume times the mean of the cube of the crack radius (using a negative
    is simply a low-tech way of telling this program that the damage measure
    represents crack data instead of porosity data). Again note: the first two
    parameters must be the moduli of the undamaged matrix material. Solving the
    inverse problem of getting the undamaged moduli when the damaged moduli are
    known requires an iterative solver -- contact me (brannon@mech.utah.edu) if
    you need routines that will do that.

USAGE
    Interactive

     % pyelas

    pyelas will ask to enter values of elastic constants, as follows

       FIRST elastic constant? E = 200e9
       SECOND elastic constant? G = 79e9
       Density? [None] (enter zero if wave speeds not desired)

   Defaults [in brackets] may be accepted by hitting return. After entering data
   one time, the program will loop back for a new set of inputs, and all your old
   inputs will be the new defaults (very useful for correcting typos).

   Direct

     % pyelas --emod1=val --emod2=val

   Execute pyelas -h for possible --emod[1,2] strings

   A NOTE ABOUT WAVE SPEEDS...
   When shock physicists say `cs', they mean what we call `cl'. Longitudinal
   waves (called "P-waves" by geologists) travel at speed `cl'. Shear waves
   ("S-waves") travel at speed ct. When you compute elastic moduli using wave
   speeds, be aware that you are actually finding the ISENTROPIC moduli.
   Conversely, if you seek the wave speeds, then you must enter ISENTROPIC
   moduli. For isotropic materials, converting between isentropic moduli and
   isothermal moduli is straightforward, but this program won't do it for you.
   Wave speeds correspond to ISENTROPIC moduli because sound waves travel such
   that the stress-strain response is the same as that when the entropy is
   constant. Elastic moduli that are measured quasistatically using a load cell
   are typically determined from stress-strain response at constant temperature
   (e.g., room temperature) Here is how the conversions are computed... Let
   subscripts "s" and "T" indicate isentropic and isothermal, respectively. Then
   relevant formulas are: G_s = G_T and K_s/K_T = c_p/c_v, where c_p and c_v are
   specific heats at const. pressure and volume. If you don't have values for
   both types of specific heats, consider looking at
   http://mech.utah.edu/~brannon/DerivativeRecursionTables.pdf for alternative
   formulas.
'''

constformulas="""
   ######################################################################
   lam = First Lame parameter         = G(E-2G)/(3G-E) = K-2G/3
     G = Shear modulus (= 2nd Lame parameter, mu)   = E/2/(1+nu)
     E = Young's modulus              = 3K(1-2nu) = 2G(1+nu) = 9KG/(3K+G)
    nu = Poisson's ratio              = (3K-E)/6K = lam/2/(lam+G)
     K = bulk modulus                 = E/3/(1-2nu) = lam + 2G/3
     H = constrained modulus          = 2G+lam = 3K-2lam = K + 4G/3
    ko = SIGy/SIGx in uniaxial strain = nu/(1-nu)
    cl = longitudinal wave speed      = sqrt(H/rho)
    ct = shear (TRANSVERSE) wave speed  = sqrt(G/rho)
    co = bulk/plastic wave speed      = sqrt(K/rho)=SQRT(cl^2-4(ct^2)/3)
    cr = thin rod elastic wave speed  = sqrt(E/rho)
   ######################################################################
"""
import sys
import os
import re
import optparse
from math import sqrt

exe = os.path.basename(__file__)

manpage = "{0} \n {1}".format(__doc__,constformulas)

interactive_help = ("Enter elastic constants in the form\n"
                    "\tconst = val\n\n{0} recognizes \n {1}\n"
                    "Enter h to display this message again\n"
                    "Enter q to quit\n"
                    .format(exe,constformulas))

NAME_MAP = {"lam": 0, "G": 1, "E": 2, "nu": 3, "K": 4, "H": 5,
            "ko": 6, "cl": 7, "ct": 8, "co": 9, "cr": 10, "rho": 11,}

def compute_elastic_constants(args, disp=0):
    """
    PURPOSE
        Given any two elastic elastic, compute all remaining constants

    INPUT
        args: array containing elastic constants in the following order
        (only 2 may be nonzero on input):
          args[0] = "lam"
          args[1] = "G"
          args[2] = "E"
          args[3] = "nu"
          args[4] = "K"
          args[5] = "H"
          args[6] = "ko"
          args[7] = "cl"
          args[8] = "ct"
          args[9] = "co"
          args[10] = "cr"
          args[11] = "rho"]
    OUTPUT
        args: array containing all elastic constants
    """

    lconstnams = len(NAME_MAP)
    if len(args) != lconstnams:
        sys.stderr.write(
            "ERROR: Wrong number of input sent to compute_elastic_constants")
        return {"retcode": 8}

    lam, G, E, nu, K, H, ko, cl, ct, co, cr, rho = args

    consts = [x for x in args if x is not None]
    nconsts = len(consts)
    if nconsts > 2 and not rho:
        sys.stderr.write("too many nonzero elastic constants sent to {0}"
                         .format(exe))
        return {"retcode": 7}

    elif nconsts == 0:
        sys.stderr.write("no elastic constants sent to {0}".format(exe))
        return {"retcode": 6}

    elif nconsts == 1 and nu is not None:
        sys.stderr.write("only one elastic constants sent to {0}".format(exe))
        return {"retcode": 5}

    idx0 = args.index(consts[0])
    if nconsts == 1:
        idx1 = 3
    else:
        idx1 = args.index(consts[1],idx0+1)
    needrho = idx0 > 6 and idx1 > 6

    if needrho and (rho is None or rho < 0.):
        sys.stderr.write("density must be positive when a wave speed is given")
        return {"retcode": 4}

    if idx0 == 6: # ko
        idx0 = 3
        nu = ko/(1. + ko)
    elif idx0 == 7: # cl
        idx0 = 5
        H = rho*cl*cl
    elif idx0 == 8: # ct
        idx0 = 1
        G = rho*ct*ct
    elif idx0 == 9: # co
        idx0 = 4
        K = rho*co*co
    elif idx0 == 10: # cr
        idx0 = 2
        E = rho*cr*cr

    if idx1 == 6: # ko
        idx1 = 3
        nu = ko/(1. + ko)
    elif idx1 == 7: # cl
        idx1 = 5
        H = rho*cl*cl
    elif idx1 == 8: # ct
        idx1 = 1
        G = rho*ct*ct
    elif idx1 == 9: # co
        idx1 = 4
        K = rho*co*co
    elif idx1 == 10: # cr
        idx1 = 2
        E = rho*cr*cr

    #  At this point, idx0 and idx1 each range from 1 to 6, and are distinct.
    #  There are 15 possible ways to choose 2 numbers from 6:
    case = 0
    if idx0 < idx1:
        case = 10*(idx0 + 1) + idx1 + 1
    else:
        case = 10*(idx1 + 1)+idx0 + 1

    # Get G and nu
    if case == 12: # lam, G
        if lam + G == 0.:
            return {"retcode": -1}
        nu = lam/2./(lam + G)
    elif case == 13: # lam, E
        A = E*E + 2.*lam*E + 9.*lam*lam
        if A < 0.:
            return {"retcode": -1}
        a = sqrt(a)
        G = (A - 3.*lam + E)/4.
        nu = (A - E - lam)/4./lam
    elif case == 14: # lam, nu
        if nu == 0.:
            return {"retcode": -2}
        G  = lam*(1. - 2.*nu)/2./nu
    elif case == 15: # lam,K
        if 3.*K - lam == 0.:
            return {"retcode": -1}
        G  = 3.*(K - lam)/2.
        nu = lam/(3.*K - lam)
    elif case == 16: # lam, H
        if H + lam == 0.:
            return {"retcode": -1}
        G  = (H - lam)/2.
        nu = lam/(H + lam)
    elif case == 23: # G, E
        nu = (E - 2.*G)/2./G
    elif case == 24: # G, nu
        pass
    elif case == 25: # G,K
        if 3. * K + G == 0.:
            return {"retcode": -1}
        nu = (3.*K - 2.*G)/2./(3.*K + G)
    elif case == 26: # G, H
        if H - G == 0.:
            return {"retcode": -1}
        nu = (H - 2.*G)/2./(H - G)
    elif case == 34: # E, nu
        if 1. + nu == 0.:
            return {"retcode": -1}
        G  = E/2./(1. + nu)
    elif case == 35: # E,K
        if 9.*K - E == 0.  or  K == 0.:
            return {"retcode": -1}
        G  = 3.*E*K/(9.*K - E)
        nu = (3.*K - E)/R6/K
    elif case == 36: # E, H
        B = E*E + 9.*H*H - 10.*E*H
        if B <0.  or  H == 0.:
            return {"retcode": -1}
        B = SQRT(B)
        G  = (3.*H - B + E)/R8
        nu = (B - H + E)/4./H
    elif case == 45: # nu,K
        if 1. + nu == 0.:
            return {"retcode": -1}
        G  = 3.*K*(1. - 2.*nu)/2./(1. + nu)
    elif case == 46: # nu, H
        if 1. - nu == 0.:
            return {"retcode": -1}
        G  = H*(1. - 2.*nu)/2./(1. - nu)
    elif case == 56: # K, H
        if 3.*K + H == 0.:
            return {"retcode": -1}
        G  = 3.*(H - K)/4.
        nu = (3.*K - H)/(3.*K + H)
    else:
        sys.stderr.write("unexpected case")
        return {"retcode": 3}

    lam = 2.*G*nu/(1. - 2.*nu)
    E = 2.*G*(1. + nu)
    K = 2.*G*(1. + nu)/3./(1. - 2.*nu)
    H = 2.*G*(1. - nu)/(1 - 2*nu)
    ko = nu/(1. - nu)

    if G <= 0. or K <= 0.:
        return {"retcode": -1}

    if rho:
        cl = sqrt(H/rho)
        ct = sqrt(G/rho)
        co = sqrt(K/rho)
        cr = sqrt(E/rho)

    consts = [("lam", lam),
              ("G", G),
              ("E", E),
              ("nu", nu),
              ("K", K),
              ("H", H),
              ("ko", ko),
              ("cl", cl),
              ("ct", ct),
              ("co", co),
              ("cr", cr),
              ("rho", rho)]
    if disp == 1:
        consts_dict = dict([(NAME_MAP[k], v) for (k, v) in consts])
    else:
        consts_dict = dict(consts)
    return consts_dict

def non_positive_definite():
    sys.stderr.write("ERROR: nonpositive definite elastic constants, try again\n\n")

def ambiguous_elastic_params():
    sys.stderr.write("ERROR: ambiguous elastic params, try again\n\n")

def elastic_param_conversion(argv):

    """
    NAME
        elastic_param_conversion

    PURPOSE
        Fetches user input and sends in right format to compute_elastic_constants

    INPUT
        argv input arguments

    AUTHORS
        Tim Fuller Sandia National Laboratories tjfulle@sandia.gov
    """

    # -- command line option parsing
    usage = "usage: %s [options]"%exe
    parser= optparse.OptionParser(usage=usage, version="%prog 1.0")

    parser.add_option("--lam",dest="lam",action="store",default=None,
                      help=("First Lame parameter = G(E-2G)/(3G-E) = K-2G/3 "
                            "[default: %default]"))
    parser.add_option("--shear",dest="shear",action="store",default=None,
                      help=("Shear modulus (= 2nd Lame parameter, mu) = E/2/(1+nu) "
                            "[default: %default]"))
    parser.add_option("--youngs",dest="youngs",action="store",default=None,
                      help=("Young's modulus = 3K(1-2nu) = 2G(1+nu) = 9KG/(3K+G) "
                            "[default: %default]"))
    parser.add_option("--poissons",dest="poissons",action="store",default=None,
                      help=("Poisson's ratio = (3K-E)/6K = lam/2/(lam+G) "
                            "[default: %default]"))
    parser.add_option("--bulk",dest="bulk",action="store",default=None,
                      help=("bulk modulus = E/3/(1-2nu) = lam + 2G/3 "
                            "[default: %default]"))
    parser.add_option("--constrained",dest="constrained",action="store",
                      default=None,
                      help=("constrained modulus = 2G+lam = 3K-2lam = K + 4G/3 "
                            "[default: %default]"))
    parser.add_option("--ko",dest="sigratio",action="store",default=None,
                      help=("SIGy/SIGx in uniaxial strain = nu/(1-nu) "
                            "[default: %default]"))
    parser.add_option("--cl",dest="longwvspd",action="store",default=None,
                      help=("longitudinal wave speed = sqrt(H/rho) "
                            "[default: %default]"))
    parser.add_option("--ct",dest="tranwvspd",action="store",default=None,
                      help=("shear (TRANSVERSE) wave speed = sqrt(G/rho) "
                            "[default: %default]"))
    parser.add_option("--co",dest="bulkwvspd",action="store",default=None,
                      help=("bulk/plastic wave speed = "
                            "sqrt(K/rho)=SQRT(cl^2-4(ct^2)/3) "
                            "[default: %default]"))
    parser.add_option("--cr",dest="thinwvspd",action="store",default=None,
                      help=("thin rod elastic wave speed  = sqrt(E/rho) "
                            "[default: %default]"))
    parser.add_option("--rho",dest="rho",action="store",default=None,
                      help=("density [default: %default]"))
    parser.add_option("-H","--man",dest="MANPAGE",action="store_true",
                      default=False,help="display manpage")

    (opts,args) = parser.parse_args(argv)

    if opts.MANPAGE:
        parser.print_help()
        sys.exit(manpage)

    # check if user gave input
    ui = [None] * len(NAME_MAP)
    if opts.lam is not None:
        ui[NAME_MAP["lam"]] = float(opts.lam)
    if opts.shear is not None:
        ui[NAME_MAP["G"]] = float(opts.shear)
    if opts.youngs is not None:
        ui[NAME_MAP["E"]] = float(opts.youngs)
    if opts.poissons is not None:
        ui[NAME_MAP["nu"]] = float(opts.poissons)
    if opts.bulk is not None:
        ui[NAME_MAP["K"]] = float(opts.bulk)
    if opts.constrained is not None:
        ui[NAME_MAP["H"]] = float(opts.constrained)
    if opts.sigratio is not None:
        ui[NAME_MAP["ko"]] = float(opts.sigratio)
    if opts.longwvspd is not None:
        ui[NAME_MAP["cl"]] = float(opts.longwvspd)
    if opts.tranwvspd is not None:
        ui[NAME_MAP["ct"]] = float(opts.tranwvspd)
    if opts.bulkwvspd is not None:
        ui[NAME_MAP["co"]] = float(opts.bulkwvspd)
    if opts.thinwvspd is not None:
        ui[NAME_MAP["cr"]] = float(opts.thinwvspd)
    if opts.rho is not None:
        rho = float(opts.rho)
        if rho < 0.: sys.exit("density must be > 0., got %e"%rho)
        ui[NAME_MAP["rho"]] = float(opts.rho)

    if any(ui):
        if (lui < 2 or lui > 3
            or (lui > 2 and not opts.rho)
            or (lui == 2 and opts.rho)):
            parser.print_help()
            print("\nERROR: must provide 2 elastic "
                  "constants and optionally density")
            return 1

        else:
            ret = compute_elastic_constants(*ui)
            retcode = ret["retcode"]
            if retcode > 0:
                return retcode
            elif retcode == -1:
                non_positive_definite()
                return retcode
            elif retcode == -2:
                ambiguous_elastic_params()
                return retcode
            else:
                print_elastic_constants(ret)

        return 0

    # no elastic constants given, ask for them
    nam1, val1, nam2, val2, dens = None, None, None, None, None
    print(interactive_help)
    while True:
        nam1, val1, idx1 = ask_input("FIRST elastic constant", nam1, val1)

        while True:
            nam2, val2, idx2 = ask_input("SECOND elastic constant", nam2, val2)

            if nam1 == nam2:
                print("SECOND elastic constant must differ from first\ntry again")
                continue

            break

        nam3,dens,idx3 = ask_input("DENSITY","rho",dens)

        ui = [None]*12
        ui[idx1],ui[idx2],ui[idx3] = val1, val2, dens
        ret = compute_elastic_constants(*ui)
        if ret == -1:
            non_positive_definite()
        elif ret == -2:
            ambiguous_elastic_params()
        else:
            print_elastic_constants(ret)
        continue
    return 0

def ask_input(query, defnam, defval):

    rho = query.lower() == "density"

    while True:

        # default value
        try:
            defval = "{0:12.6E}".format(defval)
        except ValueError:
            pass

        # ask for name, value pairs
        if rho:
            inp = get_input(query + "? [default: {0}] ".format(defval))
        else:
            inp = get_input(query + "? [default: {0} = {1}] ".format(
                    defnam, defval))

        if inp == "q":
            sys.exit("done")

        if inp == "h":
            print(interactive_help)
            continue

        if not inp:
            nam = defnam
            try:
                val = eval(defval)
            except TypeError:
                val = defval

        else:
            inp = re.split(r"[=,]", inp)
            if len(inp) == 1:
                inp.insert(0, defnam)

            try:
                nam, val = inp[0].strip(), float(inp[1].strip())

            except ValueError:
                if rho:
                    bad_density()
                else:
                    bad_syntax()
                continue

            except:
                bad_syntax()
                continue

            if rho and val < 0.:
                bad_density()
                continue

            if nam == "nu" and (val <= -1. or val > 0.5):
                bad_poissons()


        if nam is None:
            print("\nunknown constant %s\nvalid entries are %s\ntry again"
                  %("None",', '.join(NAME_MAP.keys())))
            continue

        elif nam not in NAME_MAP:
            print("\nunknown constant %s\nvalid entries are %s\ntry again"
                  %(nam,', '.join(NAME_MAP.keys())))
            continue

        break

    idx = NAME_MAP[nam]
    return nam, val, idx

def print_elastic_constants(econsts):
    emods = [x for x in econsts if x in
             sorted(NAME_MAP, key=NAME_MAP.__getitem__)[:7]]
    espds = [x for x in econsts if x in
             sorted(NAME_MAP, key=NAME_MAP.__getitem__)[7:]]
    emods.sort()
    espds.sort()
    espds.remove("rho")
    espds.insert(0,"rho")
    print("\nElastic moduli:")
    for key in emods:
        try:
            print("{0} = {1:12.6E}".format(key, econsts[key]))
        except:
            print("{0} = {1}".format(key, econsts[key]))
        continue
    if econsts["rho"]:
        print("\nWavespeeds:")
        for key in espds:
            try:
                print("{0} = {1:12.6E}".format(key, econsts[key]))
            except:
                print("{0} = {1}".format(key, econsts[key]))
            continue
    return

def bad_syntax():
    print("\nInvalid syntax, syntax should be\n\tvariableName = value\ntry again\n")
    return

def bad_density():
    print("\nInvalid density, density must be > 0.\ntry again\n")
    return

def bad_poissons():
    print("\nInvalid Poisson ratio, must be < .5 and > -1.\ntry again\n")
    return

def get_input(string):
    try:
        return raw_input(string)
    except KeyboardInterrupt:
        raise SystemExit("\n")

if __name__ == "__main__":
    elastic_param_conversion(sys.argv[1:])
