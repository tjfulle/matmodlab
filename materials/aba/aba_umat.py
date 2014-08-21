import numpy as np

try:
    from lib.mmlabpack import mmlabpack
except ImportError:
    import utils.mmlabpack as mmlabpack
from materials.material import Material
from utils.misc import who_is_calling
from core.mmlio import log_error


class AbaUMat(Material):

    def get_initial_jacobian(self):
        dtime = 0.
        time = 0.

        dstran = np.zeros(6, order="F")
        stran = np.zeros(6, order="F")

        stress = np.zeros(6, order="F")
        statev = np.array(self.xinit)

        F0 = np.eye(3)
        F = np.eye(3)

        dtemp = 0.
        temp = self._initial_temperature

        v = np.arange(6)

        elec_field = np.zeros(3)
        user_field = np.zeros(1)

        return self.jacobian(time, dtime, temp, dtemp, F0, F, stran, dstran,
                             stress, statev, elec_field, user_field, v)

    def jacobian(self, time, dtime, temp, dtemp, F0, F, stran, d,
                 stress, statev, elec_field, user_field, v):
        if self.visco_params is not None:
            ddsdde = self.numerical_jacobian(time, dtime, temp, dtemp, F0, F,
                                             stran, d, stress, statev,
                                             elec_field, user_field, v)
        else:
            args = (time, F0, F, stran, elec_field, temp, dtemp, user_field)
            ddsdde = np.zeros((6, 6))
            sig = np.array(stress)
            sv = np.array(statev)
            kwargs = {"jacobian": ddsdde}
            self.compute_updated_state(time, dtime, temp, dtemp, F0, F, stran, d,
                                       sig, sv, elec_field, user_field, **kwargs)
            ddsdde = kwargs["jacobian"]
            ddsdde[3:, 3:] *= 2.
            ddsdde = ddsdde[[[x] for x in v], v]
        return ddsdde

    def update_state(self, dtime, dstran, stress, statev, *args, **kwargs):
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

        time = args[0]
        dfgrd0 = np.reshape(args[1], (3, 3), order="F")
        dfgrd1 = np.reshape(args[2], (3, 3), order="F")
        stran = args[3]
        temp = args[5]
        dtemp = args[6]

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
        stress, statev, ddsdde = self.update_state_umat(
            stress, statev, ddsdde, sse, spd, scd, rpl, ddsddt, drplde, drpldt,
            stran, dstran, time, dtime, temp, dtemp, predef, dpred, cmname,
            ndi, nshr, self.nxtra, self.params, coords, drot, pnewdt, celent,
            dfgrd0, dfgrd1, noel, npt, layer, kspt, kstep, kinc)
        if kwargs.get("jacobian") is not None:
            kwargs["jacobian"][:,:] = ddsdde
        if np.any(np.isnan(stress)):
            log_error("umat stress contains nan's")
        return stress, statev
