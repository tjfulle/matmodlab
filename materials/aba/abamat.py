import numpy as np

import utils.mmlabpack as mmlabpack
from core.material import MaterialModel
from utils.misc import who_is_calling
try:
    import lib.umat as umat
except ImportError:
    umat = None


class AbaqusMaterial(MaterialModel):
    param_names = []

    @classmethod
    def _parameter_names(cls, n):
        cls.param_names = ["PROP{0:02d}".format(i+1) for i in range(n+1)]
        return cls.param_names

    def setup(self, props, depvar):
        self.check_import()
        if not depvar:
            depvar = 1
        statev = np.zeros(depvar)
        xkeys = ["SDV{0}".format(i+1) for i in range(depvar)]
        self.register_xtra_variables(xkeys, statev)

        ddsdde = self.get_initial_jacobian()
        mu = ddsdde[3, 3]
        lam = ddsdde[0, 0] - 2. * mu

        self.bulk_modulus = lam + 2. / 3. * mu
        self.shear_modulus = mu

    def update_state(self, time, dtime, temp, dtemp, energy, rho, F0, F,
        stran, d, elec_field, user_field, stress, statev, logger, **kwargs):
        """Update the state of an anisotropic hyperelastic model.  Implementation
        based on Abaqus conventions

        Parameters
        ----------
        dtime : float
            Time increment
        dstran : array_like
            Strain rate
        stress : array_like
            Stress
        statev : array_like
            Array of state variables
        args : tuple
            time
            deformation gradient at beginning of step
            deformation gradient at end of step
            electric field
            strain
            temperature
            temperature increment

        """
        cmname = "{0:8s}".format("umat")

        dfgrd0 = np.reshape(F0, (3, 3), order="F")
        dfgrd1 = np.reshape(F, (3, 3), order="F")
        dstran = d * dtime

        ddsdde = np.zeros((6, 6), order="F")
        ddsddt = np.zeros(6, order="F")
        drplde = np.zeros(6, order="F")
        predef = np.zeros(1, order="F")
        dpred = np.zeros(1, order="F")
        coords = np.zeros(3, order="F")
        drot = np.eye(3)
        ndi = nshr = 3
        sse = 0.
        spd = 0.
        scd = 0.
        rpl = 0.
        drpldt = 0.
        celent = 0.
        pnewdt = 0.
        noel = 1
        npt = 1
        layer = 1
        kspt = 1
        kstep = 1
        kinc = 1
        time = np.array([time,time])
        stress, statev, ddsdde = self.update_state_umat(
            stress, statev, ddsdde, sse, spd, scd, rpl, ddsddt, drplde, drpldt,
            stran, dstran, time, dtime, temp, dtemp, predef, dpred, cmname,
            ndi, nshr, self.nxtra, self.params, coords, drot, pnewdt, celent,
            dfgrd0, dfgrd1, noel, npt, layer, kspt, kstep, kinc, logger)
        if np.any(np.isnan(stress)):
            logger.error("umat stress contains nan's")
        ddsdde[3:, 3:] *= 2.
        return stress, statev, ddsdde
