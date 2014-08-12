import numpy as np

try:
    from lib.mmlabpack import mmlabpack
except ImportError:
    import utils.mmlabpack as mmlabpack
from materials.material import Material


class AbaUmat(Material):

    def get_initial_jacobian(self):
        dtime = 0.
        dstran = np.zeros(6, order="F")
        stress = np.zeros(6, order="F")
        statev = np.array(self.xinit)
        v = np.arange(6)

        time = 0.
        dfgrd0 = np.eye(3)
        dfgrd1 = np.eye(3)
        stran = np.zeros(6, order="F")
        temp = self._initial_temperature
        dtemp = 0.
        args = (time, dfgrd0, dfgrd1, stran, np.zeros(3), temp, dtemp)

        return self.jacobian(dtime, dstran, stress, statev, v, *args)

    def jacobian(self, dtime, dstran, stress, statev, v, *args):
        kwargs = {"return jacobian": True}
        sig = np.array(stress)
        sv = np.array(statev)
        ddsdde = self.update_state(dtime, dstran, sig, sv, *args, **kwargs)
        ddsdde[3:, 3:] *= 2.
        return ddsdde[[[x] for x in v], v]

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
        if kwargs.get("return jacobian"):
            return ddsdde
        return stress, statev
