import logging
import numpy as np
from math import sin, cos
from ..constants import TOOR3, ROOT3
from ..utils.elas import ElasticConstants

EC = ElasticConstants()

# Elastic constants
# Names of recognized elastic constants:
# LAME: First lame constant
# G: Shear modulus
# E: Young's modulus
# NU: Poisson's ratio
# K: Bulk modulus
# H: Constrained modulus
# KO: SIGy/SIGx in uniaxial strain = nu/(1-nu)

# Names of recognized wave speeds
# CL: longitudinal wave speed = sqrt(H/rho)
# CT: shear (TRANSVERSE) wave speed = sqrt(G/rho)
# CO: bulk/plastic wave speed = sqrt(K/rho)=SQRT(cl^2-4(ct^2)/3)
# CR: thin rod elastic wave speed  = sqrt(E/rho)

# Name of density
# RHO

# Common hyperelastic parameters
# C10, C01, D1

# Strengths
# YS: Yield in shear
# YT: Yield in tension

# Hardening
# HM: Hardening modulus
# HP: Hardening parameter

# Drucker-Prager parameters: Sqrt[J2] = A + B I1
# DPA
# DPB

# Cohesion and friction angle
# COHESION
# FRICTION_ANGLE

# TEMP0


class Completion:
    """Container class for material completions.

    Note that __getitem__ is hijacked to not raise an exception if you are
    trying to get something that is not in __dict___. While this is normally
    dangerous, it is nice here so that we don't have to explicitly set every
    possible property in the completion (some for which there isn't enough
    info to compute in the first place).

    """
    def __init__(self, d):
        self._dict = dict(d)
    def __getitem__(self, key):
        return self._dict.get(key)
    def __str__(self):
        x = ", ".join("{0}={1}".format(k, v) for (k, v) in self._dict.items())
        return "PropertyCompletion({0})".format(x)


def complete_properties(parameters, propmap):
    """Complete the material properties

    Parameters
    ----------
    parameters : ndarray
        Array of material parameters
    propmap : dict
        Mapping of property to parameter

    """
    if not propmap:
        return

    # check if any hyperelastic parameters are given and, if so, compute
    # equivalent linear elastic parameters
    if "C10" in propmap and "G" not in propmap:
        c10 = parameters[propmap["C10"]]
        if "C01" in propmap:
            c01 = parameters[propmap["C01"]]
        else:
            c01 = 0.
        g = 2. * (c10 + c01)
        propmap["G"] = len(parameters)
        parameters = np.append(parameters, g)

    if "D1" in propmap and "K" not in propmap:
        k = 2. / parameters[propmap["D1"]]
        propmap["K"] = len(parameters)
        parameters = np.append(parameters, k)

    # prepopulate the completion mapping with properties already specified by
    # the material model. save the elastic constants in their own list
    completion = {}
    elas = {}
    for (key, idx) in propmap.items():
        if key is None:
            continue
        key = key.upper()
        completion[key] = parameters[idx]
        if key in EC:
            elas[key] = parameters[idx]

    # complete the elastic constants
    completion.update(EC.compute_elastic_constants(**elas))

    # complete inelastic properties

    # yield strength in shear and tension
    ys = completion.get("YS")
    yt = completion.get("YT")

    # Linear Drucker-Prager and Mohr-Coulomb properties
    A = completion.get("DPA")
    B = completion.get("DPB")
    C = completion.get("COHESION")
    phi = completion.get("FRICTION_ANGLE")

    if ys is None and yt is None and A is not None and B == 0.0:
        ys = A
    if ys is not None:
        if yt is None:
            yt = ys / TOOR3
    elif yt is not None:
        ys = yt * TOOR3
    completion["YS"] = ys
    completion["YT"] = yt

    if phi is None and B is not None:
        # B = 2 Sin[PHI] / ROOT3 / (3 + Sin[PHI])
        x = 0.
        for i in range(25):
            f = B - 2. * sin(x) * TOOR3 / (3. + sin(x))
            df = (2. * ROOT3 * cos(x) / (3. * (sin(x) + 3.)) -
                  2. * ROOT3 * sin(x) * cos(x)/(3. * (sin(x) + 3.) ** 2))
            dx = -f / df
            x += dx
            if dx < 1.E-08:
                phi = x
                break
        else:
            logging.warn("unable to determine friction angle")

    if C is None and A is not None and phi is not None:
        C = A * ROOT3 * (3. - sin(phi)) / 6. / cos(phi)

    # by this point, A, B, C, phi are known - if there is enough info to
    # compute them. If there is not, compute A based on yield in shear
    if A is None:
        A = ys

    completion["DPA"] = A
    completion["DPB"] = B
    completion["COHESION"] = C
    completion["FRICTION_ANGLE"] = phi

    return Completion(completion)
