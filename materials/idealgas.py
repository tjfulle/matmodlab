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

    def update_state(self, dt, d, stress, xtra, *args, **kwargs):
        """
          Evaluate the eos - rho and tmpr are in CGSEV

          By the end of this routine, the following variables should be
          updated and stored in matdat:
                  density, temperature, energy, pressure
        """
        unit_system = kwargs["UNITS"]
        rho = kwargs.get("RHO")
        tmpr = kwargs.get("TEMPERATURE")
        enrgy = kwargs.get("ENERGY")

        M = self._param_vals[0]
        CV = self._param_vals[1]

        R = UnitManager.transform(
            8.3144621,
            "ENERGY_UNITS_OVER_TEMPERATURE_UNITS_OVER_DISCRETE_AMOUNT",
            "SI", unit_system)


        if all((rho, tmpr)):
            # get (pres, enrgy) as functions of (rho, tmpr)
            pres, energy = eosigr(M, CV, R, rho, tmpr)

        elif all((rho, enrgy)):
            # get (pres, tmpr) as functions of (rho, enrgy)
            pres, tmpr = eosigv(M, CV, R, rho, enrgy)

        else:
            log_error("IdealGas: update_state not called correctly.")

        cs = (R * tmpr / M) ** 2
        dpdr = R * tmpr / M
        dpdt = R * rho / M
        dedt = CV * R
        dedr = CV * pres * M / rho ** 2
        scratch = np.array([cs, dpdr, dpdt, dedt, dedr])

        return pres, tmpr, enrgy, scratch


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
