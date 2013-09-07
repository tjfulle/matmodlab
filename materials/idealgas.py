import sys
import numpy as np

from core.io import log_error
from materials._material import Material


class IdealGas(Material):
    name = "idealgas"
    param_names = ["M", "CV"]
    def __init__(self):
        super(IdealGas, self).__init__()
        self.register_parameters(*self.param_names)

    # Public methods
    def setup(self, params):

        self.set_param_vals(params)

        # Variables already registered:
        #   density, temperature, energy, pressure
        self.register_mtl_variable("SNDSPD", "SCALAR", units="VELOCITY_UNITS")
        self.register_mtl_variable("DPDR", "SCALAR",
            units="PRESSURE_UNITS_OVER_DENSITY_UNITS")
        self.register_mtl_variable("DPDT", "SCALAR",
            units="PRESSURE_UNITS_OVER_TEMPERATURE_UNITS")
        self.register_mtl_variable("DEDT", "SCALAR",
            units="SPECIFIC_ENERGY_UNITS_OVER_TEMPERATURE_UNITS")
        self.register_mtl_variable("DEDR", "SCALAR",
            units="SPECIFIC_ENERGY_UNITS_OVER_DENSITY_UNITS")

    def update_state(self, *args, **kwargs):
        """Evaluate the eos - rho and tmpr are in CGSEV

        By the end of this routine, the following variables should be
        updated
                  density, temperature, energy, pressure
        """
        mode = kwargs.get("mode", 0)
        disp = kwargs.get("disp", 0)

        # unit_system = kwargs["UNITS"]
        M = self._param_vals[0]
        CV = self._param_vals[1]

        R = 8.3144621
        #R = UnitManager.transform(
        #    8.3144621,
        #    "ENERGY_UNITS_OVER_TEMPERATURE_UNITS_OVER_DISCRETE_AMOUNT",
        #    "SI", unit_system)


        if mode == 0:
            # get (pres, enrgy) as functions of args=(rho, tmpr)
            rho, tmpr = args
            pres, enrgy = eosigr(M, CV, R, rho, tmpr)

        elif mode == 1:
            # get (pres, tmpr) as functions of args=(rho, enrgy)
            rho, enrgy = args
            pres, tmpr = eosigv(M, CV, R, rho, enrgy)
        else:
            raise Error1("idealgas: {0}: unrecognized mode".format(mode))

        cs = (R * tmpr / M) ** 2
        dpdr = R * tmpr / M
        dpdt = R * rho / M
        dedt = CV * R
        dedr = CV * pres * M / rho ** 2
        scratch = np.array([cs, dedt, dedr, dpdt, dpdr])

        if mode == 0:
            retvals = [pres, enrgy]
        else:
            retvals = [pres, tmpr]
        if disp:
            retvals.append(scratch)

        return retvals


def eosigr(M, CV, R, rho, tmpr) :
    """Pressure and energy as functions of density and temperature

    """
    enrgy = CV * R * tmpr
    pres = R * tmpr * rho / M
    return pres, enrgy

def eosigv(M, CV, R, rho, enrgy) :
    """Pressure and temperature as functions of density and energy

    """
    tmpr = enrgy / CV / R
    pres = R * tmpr * rho / M
    return pres, tmpr
