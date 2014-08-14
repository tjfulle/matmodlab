import sys
import numpy as np
from materials.aba.aba_uanisohyper_inv import AbaUAnisoHyper
from core.mmlio import Error1, log_message, log_error, log_warning
try:
    from lib.mmlabpack import mmlabpack
except ImportError:
    import utils.mmlabpack as mmlabpack
try:
    import lib.uanisohyper as umat
except ImportError:
    umat = None


class UAnisoHyper(AbaUAnisoHyper):
    """Constitutive model class for the uanisohyper model"""
    name = "uanisohyper_inv"
    param_names = []

    def setup(self):
        self.model_type = "anisotropic_hyper"
        self.register_xtra_variables(self._xkeys)
        self.set_initial_state(self._istate)
        del self._xkeys
        del self._istate

        ddsdde = self.get_initial_jacobian()
        mu = ddsdde[3, 3]
        lam = ddsdde[0, 0] - 2. * mu

        self.bulk_modulus = lam + 2. / 3. * mu
        self.shear_modulus = mu

    def setup_umat(self, params, statev, **kwargs):
        """Set up the umat"""
        if umat is None:
            raise Error1("umat model not imported")

        self.param_names = ["PARAM{0}".format(i+1)
                            for i in range(len(params))]

        self._xkeys = ["SDV{0}".format(i+1) for i in range(len(statev))]
        self._istate = np.array(statev)
        self.fiber_direction = kwargs.get("fiber_direction")
        return

    def update_state_anisohyper(self, Ainv, zeta, nfibers, temp, noel,
                                cmname, incmpflag, ihybflag, statev,
                                fieldv, fieldvinc):
        return umat.uanisohyper_inv(Ainv, zeta, nfibers, temp, noel, cmname,
                                    incmpflag, ihybflag, statev, fieldv,
                                    fieldvinc, self.params[:-4],
                                    log_error, log_message, log_warning)

