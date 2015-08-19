import os
import sys
import logging
from numpy import zeros, ones, eye, array, reshape
from matmodlab.product import MAT_D
from matmodlab.materials.product import ABA_UTL
from matmodlab.mmd.material import MaterialModel
from matmodlab.utils.errors import MatModLabError, StopFortran
import matmodlab.utils.mmlabpack as mmlabpack

class Elastic(MaterialModel):
    name = 'elastic'

    @classmethod
    def source_files(cls):
        return [os.path.join(MAT_D, 'src/elastic.f90'),
                os.path.join(MAT_D, 'src/elastic.pyf'), ABA_UTL]

    @classmethod
    def param_names(cls, n):
        return ['K', 'G']

    @staticmethod
    def completions_map():
        return {'K': 0, 'G': 1}

    def import_lib(self, libname):
        """Set up the Elastic material

        """
        try:
            import matmodlab.lib.elastic as mat
        except ImportError:
            raise MatModLabError('elastic model not imported')
        self.lib = mat

    @classmethod
    def from_other(cls, other_mat):
        # reset the initial parameter array to represent this material
        iparray = zeros(2)
        iparray[0] = other_mat.completions['K'] or 0.
        iparray[1] = other_mat.completions['G'] or 0.
        if other_mat.completions['YT']:
            logging.getLogger('matmodlab.mmd.simulator').warn(
                '{0} cannot mimic a strength limit, only '
                'an elastic response will occur'.format(cls.name))
        return cls(iparray)

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

        nx = 1
        xtra = zeros(nx)
        s = stress.copy()
        self.lib.umat(stress, xtra, ddsdde,
            sse, spd, scd, rpl, ddsddt, drplde, drpldt, stran, dstran,
            time, dtime, temp, dtemp, predef, dpred, cmname, ndi, nshr,
            nx, self.params, coords, drot, pnewdt, celent, dfgrd0,
            dfgrd1, noel, npt, layer, kspt, kstep, kinc, log.info, log.warn,
            StopFortran)

        return stress, statev, ddsdde
