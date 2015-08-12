import os
import logging
import numpy as np
from matmodlab.product import MAT_D
import matmodlab.utils.mmlabpack as mmlabpack
from matmodlab.mmd.material import MaterialModel
from matmodlab.utils.errors import StopFortran

mat = None

d = os.path.join(MAT_D, "src")
f1 = os.path.join(d, "mooney_rivlin.f90")
f2 = os.path.join(d, "mooney_rivlin.pyf")

class MooneyRivlin(MaterialModel):
    """Constitutive model class for the Mooney-Rivlin model

    """
    name = "mooney_rivlin"
    lapack = "lite"

    @classmethod
    def source_files(cls):
        return [f1, f2]

    @classmethod
    def param_names(cls, n):
        return ['C10', 'C01', 'NU', 'T0', 'MC10', 'MC01']

    @staticmethod
    def completions_map():
        return {'C10': 0, 'C01': 1, 'NU': 2, 'TEMP0': 3}

    def import_lib(self, libname=None):
        import matmodlab.lib.mooney_rivlin as mat
        self.lib = mat

    def update_state(self, time, dtime, temp, dtemp, energy, rho, F0, F,
        stran, d, elec_field, stress, xtra, **kwargs):
        """ update the material state based on current state and stretch """

        Fij = np.reshape(F, (3, 3))
        Vij = mmlabpack.sqrtm(np.dot(Fij, Fij.T))
        Rij = np.reshape(np.dot(np.linalg.inv(Vij), Fij), (9,))
        Vij = mmlabpack.asarray(Vij, 6)
        log = logging.getLogger('matmodlab.mmd.simulator')
        sig, ddsdde = self.lib.mnrv_mat(self.params, Rij, Vij,
                                        log.info, log.warn, StopFortran)

        return np.reshape(sig, (6,)), np.reshape(xtra, (self.num_sdv,)), ddsdde
