import os
import logging
import numpy as np
from numpy import zeros, ones, eye, array, reshape
from matmodlab.product import MAT_D
from matmodlab.materials.product import ABA_UTL
from matmodlab.mmd.material import MaterialModel
from matmodlab.utils.errors import MatModLabError, StopFortran
import matmodlab.utils.mmlabpack as mmlabpack

class Plastic(MaterialModel):
    name = 'plastic'

    @classmethod
    def param_names(cls, n):
        return ['E', 'Nu', 'Y', 'H']

    @staticmethod
    def completions_map():
        return {'E': 0, 'Nu': 1, 'YS': 2, 'HM': 3}

    @classmethod
    def source_files(cls):
        d = os.path.join(MAT_D, 'src')
        return [os.path.join(MAT_D, 'src/plastic.f90'),
                os.path.join(MAT_D, 'src/plastic.pyf'), ABA_UTL]

    def import_lib(self, libname=None):
        try:
            import matmodlab.lib.plastic as mat
        except ImportError:
            raise MatModLabError('model plastic not imported')
        self.lib = mat

    def setup(self, **kwargs):
        """Set up the Plastic material

        """
        if self.parameters['E'] <= 0.:
            raise MatModLabError("Negative Young's modulus")
        if -1 > self.parameters['Nu'] >= .5:
            raise MatModLabError("Invalid Poisson's ratio")
        self.ordering = [0, 1, 2, 3, 5, 4]
        components = ['XX', 'YY', 'ZZ', 'XY', 'XZ', 'YZ']
        sdv_keys = ['Pres', 'Mises']
        sdv_keys.extend(['EE.{0}'.format(c) for c in components])
        sdv_keys.extend(['EP.{0}'.format(c) for c in components])
        sdv_keys.extend(['AL.{0}'.format(c) for c in components])
        return sdv_keys, np.zeros(len(sdv_keys))

    def update_state(self, time, dtime, temp, dtemp, energy, rho, F0, F,
        stran, d, elec_field, stress, statev, **kwargs):
        """Compute updated stress given strain increment"""
        log = logging.getLogger('matmodlab.mmd.simulator')

        # defaults
        cmname = '{0:8s}'.format('umat')
        dfgrd0 = reshape(F0, (3, 3), order='F')
        dfgrd1 = reshape(F, (3, 3), order='F')
        dstran = d * dtime
        ddsdde = zeros((6, 6), order='F')
        ddsddt = zeros(6, order='F')
        drplde = zeros(6, order='F')
        predef = zeros(1, order='F')
        dpred = zeros(1, order='F')
        coords = zeros(3, order='F')
        drot = eye(3)
        ndi = nshr = 3
        spd = scd = rpl = drpldt = pnewdt = 0.
        noel = npt = layer = kspt = kinc = 1
        sse = mmlabpack.ddot(stress, stran) / rho
        celent = 1.
        kstep = 1
        time = array([time, time])

        self.lib.umat(stress, statev, ddsdde,
            sse, spd, scd, rpl, ddsddt, drplde, drpldt, stran, dstran,
            time, dtime, temp, dtemp, predef, dpred, cmname, ndi, nshr,
            self.num_sdv, self.params, coords, drot, pnewdt, celent, dfgrd0,
            dfgrd1, noel, npt, layer, kspt, kstep, kinc, log.info, log.warn,
            StopFortran)

        return stress, statev, ddsdde
