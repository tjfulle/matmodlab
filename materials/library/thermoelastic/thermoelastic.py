import numpy as np
from materials.abaumat import AbaUmat
from core.mmlio import Error1, log_message, log_error
try:
    from lib.mmlabpack import mmlabpack
except ImportError:
    import utils.mmlabpack as mmlabpack
try:
    import lib.thermoelastic as thermoelastic
except ImportError:
    thermoelastic = None


class ThermoElastic(AbaUmat):
    """Constitutive model class for the thermoelastic model

    """
    name = "thermoelastic"
    param_names = ["E0", "NU0", "T0", "E1", "NU1", "T1", "ALPHA", "TI"]

    def setup(self):
        """Set up the domain Mooney-Rivlin materia

        """
        tiny = 1.E-12
        if thermoelastic is None:
            raise Error1("thermoelastic model not imported")

        if self.params["E0"] < 0.:
            log_error("E0 < 0.")
        if abs(self.params["E0"]) < tiny:
            log_error("required parameter E0 missing")
        if self.params["E1"] < 0.: log_error("E1 < 0.")
        if abs(self.params["E1"]) < tiny:
            self.params["E1"] = self.params["E0"]

        if -1. > self.params["NU0"] > .5: log_error("bad NU0")
        if -1. > self.params["NU1"] > .5: log_error("bad NU1")
        if abs(self.params["NU1"]) < tiny:
            self.params["NU1"] = self.params["NU0"]

        if abs(self.params["T0"]) < tiny:
            log_error("T0 undefined")

        if abs(self.params["T1"]) < tiny:
            log_error("T1 undefined")

        if self.params["T0"] >= self.params["T1"]:
            log_error("T0 must be < T1")

        if abs(self.params["TI"]) < tiny:
            log_error("TI undefined")

        nu = self.params["NU0"]
        smod = self.params["E0"] / 2. / (1 + nu)
        bmod = 2. * smod * (1.+ nu) / 3. / (1 - 2. * nu)

        self.bulk_modulus = bmod
        self.shear_modulus = smod

        # register extra variables
        nxtra = 12
        tc = ["XX", "YY", "ZZ", "XY", "YZ", "XZ"]
        keys = ["EELAS{0}".format(c) for c in tc]
        keys.extend(["ETHERM{0}".format(c) for c in tc])
        self.register_xtra_variables(keys)
        self.set_initial_state(np.zeros(nxtra))
        return

    def update_state_umat(self, stress, statev, ddsdde,
            sse, spd, scd, rpl, ddsddt, drplde, drpldt, stran, dstran,
            time, dtime, temp, dtemp, predef, dpred, cmname, ndi, nshr,
            nxtra, params, coords, drot, pnewdt, celent, dfgrd0,
            dfgrd1, noel, npt, layer, kspt, kstep, kinc):
        """update the material state"""
        thermoelastic.umat(stress, statev, ddsdde,
            sse, spd, scd, rpl, ddsddt, drplde, drpldt, stran, dstran,
            time, dtime, temp, dtemp, predef, dpred, cmname, ndi, nshr,
            nxtra, params, coords, drot, pnewdt, celent, dfgrd0,
            dfgrd1, noel, npt, layer, kspt, kstep, kinc, log_error, log_message)
        return stress, statev, ddsdde
