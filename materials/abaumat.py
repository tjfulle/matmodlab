import sys
import numpy as np

try:
    from lib.mmlabpack import mmlabpack
except ImportError:
    import utils.mmlabpack as mmlabpack
from materials.material import Material


class AbaUmat(Material):

    def jacobian(self, dtime, dstran, stress, statev, v, *args):
        kwargs = {"return jacobian": True}
        ddsdde = self.update_state(dtime, dstran, stress, statev,
                                   *args, **kwargs)
        return ddsdde[[[x] for x in v], v]

    def update_state(self, dtime, dstran, stress, statev, *args, **kwargs):
        """Update the state of an anisotropic hyperelastic model.  Implementation
        based on Abaqus conventions

        """
        cmname = "umat    "
        time = args[2]
        temp = 298.
        dtemp = 0.
        dfgrd0 = np.reshape(args[0], (3, 3), order="F")
        dfgrd1 = np.reshape(args[0], (3, 3), order="F")
        ddsdde = np.zeros((6, 6), order="F")
        stran = args[5]
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
