import sys
import numpy as np
from materials.abaumat import AbaUmat
from core.mmlio import Error1, log_message, log_error
try:
    from lib.mmlabpack import mmlabpack
except ImportError:
    import utils.mmlabpack as mmlabpack
try:
    import lib.umat as umat
except ImportError:
    umat = None


class Umat(AbaUmat):
    """Constitutive model class for the umat model"""
    name = "umat"
    param_names = []

    def setup(self):
        self.register_xtra_variables(self._xkeys)
        self.set_initial_state(self._istate)
        del self._xkeys
        del self._istate

        ddsdde = self.get_initial_jacobian()
        mu = ddsdde[3, 3]
        lam = ddsdde[0, 0] - 2. * mu

        self.bulk_modulus = lam + 2. / 3. * mu
        self.shear_modulus = mu

    def setup_umat(self, params, statev):
        """Set up the umat"""
        tiny = 1.E-12
        if umat is None:
            raise Error1("umat model not imported")

        self.param_names = ["PARAM{0}".format(i+1)
                            for i in range(len(params))]

        self._xkeys = ["SDV{0}".format(i+1) for i in range(len(statev))]
        self._istate = np.array(statev)
        return

    def update_state_umat(self, stress, statev, ddsdde,
            sse, spd, scd, rpl, ddsddt, drplde, drpldt, stran, dstran,
            time, dtime, temp, dtemp, predef, dpred, cmname, ndi, nshr,
            nxtra, params, coords, drot, pnewdt, celent, dfgrd0,
            dfgrd1, noel, npt, layer, kspt, kstep, kinc):
        """update the material state"""
        umat.umat(stress, statev, ddsdde,
            sse, spd, scd, rpl, ddsddt, drplde, drpldt, stran, dstran,
            time, dtime, temp, dtemp, predef, dpred, cmname, ndi, nshr,
            nxtra, params, coords, drot, pnewdt, celent, dfgrd0,
            dfgrd1, noel, npt, layer, kspt, kstep, kinc, log_error, log_message)
        return stress, statev, ddsdde
