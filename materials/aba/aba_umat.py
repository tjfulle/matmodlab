import numpy as np

import utils.mmlabpack as mmlabpack
from materials.material import Material
from utils.misc import who_is_calling
from core.mmlio import log_error


class AbaUMat(Material):

    def update_state(self, time, dtime, temp, dtemp, energy, rho, F0, F,
        stran, d, elec_field, user_field, stress, statev, **kwargs):
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
        cmname = "{0:8s}".format(self._umat_name)

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
            dfgrd0, dfgrd1, noel, npt, layer, kspt, kstep, kinc)
        if np.any(np.isnan(stress)):
            log_error("umat stress contains nan's")
        ddsdde[3:, 3:] *= 2.
        return stress, statev, ddsdde
