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
    """Abaqus umat interface

    """
    name = "umat"
    param_names = None

    def setup(self):
        """Set up the domain Mooney-Rivlin materia

        """
        if umat is None:
            raise Error1("umat model not imported")

        # call model to get initial jacobian ddsdde
        ddsdde = self.get_initial_jacobian()
        mu = ddsdde[5, 5]
        lam = ddsdde[0, 0] - 2 * mu
        self.bulk_modulus = lam + 2. / 3. * mu
        self.shear_modulus = mu

        # register extra variables
        keys = ["SDV{0}".format(i) for i in range(self.nxtra)]
        self.register_xtra_variables(keys)
        self.set_initial_state(np.zeros(self.nxtra))
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
