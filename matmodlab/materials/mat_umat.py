import os
import sys
import logging
import numpy as np
from matmodlab.product import PKG_D
import matmodlab.utils.mmlabpack as mmlabpack
from matmodlab.mmd.material import MaterialModel
from matmodlab.materials.product import ABA_UMAT_PYF, ABA_UTL, UMAT
from matmodlab.utils.errors import StopFortran

class UMat(MaterialModel):
    '''Constitutive model class for the umat model'''
    name = UMAT
    libname = 'umat'
    lapack = 'lite'

    @classmethod
    def aux_files(cls):
        return [ABA_UMAT_PYF, ABA_UTL]

    @classmethod
    def param_names(cls, n):
        return ['Prop{0:02}'.format(i+1) for i in range(n)]

    def import_lib(self, libname=None):
        libname = libname or 'umat'
        string = 'import matmodlab.lib.{0} as mat'.format(libname)
        code = compile(string, '<string>', 'exec')
        exec code in globals()
        self.lib = mat

    def setup(self, **kwargs):
        '''initialize the material state'''
        log = logging.getLogger('matmodlab.mmd.simulator')

        self.ordering = kwargs.get('ordering', [0, 1, 2, 3, 5, 4])

        # depvar must be at least 1 (cannot pass reference to empty list)
        depvar = kwargs.get('depvar', 1)

        # depvar allowed to be an integer (number of SDVs) or a list (names of
        # SDVs)
        xkeys = lambda n: ['SDV{0}'.format(i+1) for i in range(depvar)]
        try:
            depvar, sdv_keys = len(depvar), depvar
        except TypeError:
            depvar = max(depvar, 1)
            sdv_keys = xkeys(depvar)
        statev = np.zeros(depvar)

        # initialize the model
        coords = np.zeros(3, order='F')
        noel = npt = layer = kspt = 1
        statev = np.zeros(depvar)
        self.lib.sdvini(statev, coords, noel, npt, layer, kspt,
                        log.info, log.warn, StopFortran)
        return sdv_keys, statev

    def update_state(self, time, dtime, temp, dtemp, energy, rho, F0, F,
        stran, d, elec_field, stress, statev, **kwargs):

        log = logging.getLogger('matmodlab.mmd.simulator')

        # abaqus defaults
        cmname = '{0:8s}'.format('umat')
        dfgrd0 = np.reshape(F0, (3, 3), order='F')
        dfgrd1 = np.reshape(F, (3, 3), order='F')
        dstran = d * dtime
        ddsdde = np.zeros((6, 6), order='F')
        ddsddt = np.zeros(6, order='F')
        drplde = np.zeros(6, order='F')
        predef = np.zeros(1, order='F')
        dpred = np.zeros(1, order='F')
        coords = np.zeros(3, order='F')
        drot = np.eye(3)
        ndi = nshr = 3
        spd = scd = rpl = drpldt = pnewdt = 0.
        noel = npt = layer = kspt = kinc = 1
        sse = mmlabpack.ddot(stress, stran) / rho
        celent = 1.
        kstep = 1
        time = np.array([time, time])
        # abaqus ordering
        stress = stress[self.ordering]
        # abaqus passes engineering strain
        dstran = dstran[self.ordering]
        stran = stran[self.ordering]

        self.lib.umat(stress, statev, ddsdde,
            sse, spd, scd, rpl, ddsddt, drplde, drpldt, stran, dstran,
            time, dtime, temp, dtemp, predef, dpred, cmname, ndi, nshr,
            self.num_sdv, self.params, coords, drot, pnewdt, celent, dfgrd0,
            dfgrd1, noel, npt, layer, kspt, kstep, kinc, log.info, log.warn,
            StopFortran)
        stress = stress[self.ordering]
        ddsdde = ddsdde[self.ordering, [[i] for i in self.ordering]]
        return stress, statev, ddsdde
