import logging
import numpy as np
import matmodlab.utils.mmlabpack as mmlabpack
from matmodlab.mmd.material import MaterialModel
from matmodlab.utils.errors import StopFortran
from matmodlab.materials.product import (DGPADM_F, TENSALG_F90,
    UANISOHYPER_INV, ABA_UANISOHYPER_PYF, ABA_UANISOHYPER_JAC_F90, ABA_UTL)

class UAnisoHyperInv(MaterialModel):
    """Constitutive model class for the uanisohyper model"""
    name = UANISOHYPER_INV
    lapack = 'lite'
    libname = 'uanisohyper_inv'

    @classmethod
    def aux_files(cls):
        return [DGPADM_F, TENSALG_F90, ABA_UANISOHYPER_PYF,
                ABA_UANISOHYPER_JAC_F90, ABA_UTL]

    @classmethod
    def param_names(cls, n):
        return ['Prop{0:02}'.format(i+1) for i in range(n)]

    def import_lib(self, libname=None):
        libname = libname or 'uanisohyper_inv'
        string = 'import matmodlab.lib.{0} as mat'.format(libname)
        code = compile(string, '<string>', 'exec')
        exec code in globals()
        self.lib = mat

    def setup(self, **kwargs):
        '''initialize the material state'''
        log = logging.getLogger('matmodlab.mmd.simulator')

        fiber_dirs = kwargs.get("fiber_dirs", [1, 0, 0])
        self.fiber_dirs = np.array(fiber_dirs, dtype=np.float64)
        self.nfibers = self.fiber_dirs.shape[0]
        assert self.fiber_dirs.shape[1] == 3
        assert self.nfibers == 1, "uanisohyper_inv currently limited to 1 fiber"

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
        """update the material state"""
        log = logging.getLogger('matmodlab.mmd.simulator')
        # abaqus defaults
        w = np.array([1, 1, 1, 2, 2, 2], dtype=np.float64)
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
        dstran = dstran[self.ordering] #* w
        stran = stran[self.ordering] #* w
        self.lib.umat(stress, statev, ddsdde,
            sse, spd, scd, rpl, ddsddt, drplde, drpldt, stran, dstran,
            time, dtime, temp, dtemp, predef, dpred, cmname, ndi, nshr,
            self.num_sdv, self.params, self.fiber_dirs, drot, pnewdt,
            celent, dfgrd0, dfgrd1, noel, npt, layer, kspt, kstep, kinc,
            log.info, log.warn, StopFortran)
        stress = stress[self.ordering]
        ddsdde = ddsdde[self.ordering, [[i] for i in self.ordering]]
        return stress, statev, ddsdde
