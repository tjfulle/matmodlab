import os
import sys
import numpy as np
from numpy.linalg import solve, lstsq

from __config__ import cfg
import utils.io as io
import utils.tensor as tensor
from utils.kinematics import deps2d, sig2d, update_deformation
from utils.tensor import NSYMM, NTENS, NVEC, I9
from utils.io import Error1
from materials.material import create_material
from drivers.driver import Driver

np.set_printoptions(precision=4)

class SolidDriver(Driver):
    name = "solid"
    def __init__(self):
        pass

    def setup(self, runid, material, mtlprops, *opts):
        """Setup the driver object

        """
        self.runid = runid
        self.mtlmdl = create_material(material)
        self.kappa, self.density, self.proportional, self.ndumps = opts[:4]

        # Save the unchecked parameters
        self.mtlmdl.unchecked_params = mtlprops

        # register variables
        self.register_glob_variable("TIME_STEP")
        self.register_glob_variable("STEP_NUM")
        self.register_glob_variable("LEG_NUM")
        self.register_variable("STRESS", vtype="SYMTENS")
        self.register_variable("STRAIN", vtype="SYMTENS")
        self.register_variable("DEFGRAD", vtype="TENS")
        self.register_variable("SYMM_L", vtype="SYMTENS")
        self.register_variable("EFIELD", vtype="VECTOR")
        self.register_variable("EQSTRAIN", vtype="SCALAR")
        self.register_variable("VSTRAIN", vtype="SCALAR")
        self.register_variable("DENSITY", vtype="SCALAR")
        self.register_variable("PRESSURE", vtype="SCALAR")
        self.register_variable("DSTRESS", vtype="SYMTENS")

        # Setup
        self.mtlmdl.setup(mtlprops)

        # register material variables
        self.xtra_start = self.ndata
        for (var, vtype) in self.mtlmdl.material_variables():
            self.register_variable(var, vtype=vtype)

        nxtra = self.ndata - self.xtra_start
        self.xtra_end = self.xtra_start + nxtra
        setattr(self, "xtra_slice", slice(self.xtra_start, self.xtra_end))

        # allocate storage
        self.allocd()

        # initialize nonzero data
        self._data[self.defgrad_slice] = I9
        self._data[self.density_slice] = self.density

        # initialize material
        sig = np.zeros(6)
        xtra = self.mtlmdl.initial_state()
        args = (I9, np.zeros(3))

        sig, xtra = self.mtlmdl.call_material_zero_state(sig, xtra, *args)

        # -------------------------- quantities derived from final state
        pres = -np.sum(sig[:3]) / 3.

        xtra = self.mtlmdl.adjust_initial_state(xtra)

        self.setvars(stress=sig, pressure=pres, xtra=xtra)

        return

    def process_legs(self, legs, iomgr, *args):
        """Process the legs

        Parameters
        ----------

        Returns
        -------

        """
        termination_time = args[0]
        if termination_time is None:
            termination_time = legs[-1][0] + 1.e-06

        kappa = self.kappa

        # initial leg
        glob_step_num = 0
        rho = self.data("DENSITY")[0]
        xtra = self.data("XTRA")
        sig = self.data("STRESS")
        tleg = np.zeros(2)
        d = np.zeros(NSYMM)
        dt = 0.
        eps = np.zeros(NSYMM)
        f = np.reshape(np.eye(3), (9, 1))
        depsdt = np.zeros(NSYMM)
        sigdum = np.zeros((2, NSYMM))

        # compute the initial jacobian
        J0 = self.mtlmdl.constant_jacobian()

        # v array is an array of integers that contains the rows and columns of
        # the slice needed in the jacobian subroutine.
        nv = 0
        vdum = np.zeros(6, dtype=np.int)

        # Process each leg
        nlegs = len(legs)
        lsl = len(str(nlegs))
        for leg_num, leg in enumerate(legs):

            tleg[0] = tleg[1]
            sigdum[0] = sig[:]
            if nv:
                sigdum[0, v] = sigspec[1]

            tleg[1], nsteps, ltype, c, ef = leg
            delt = tleg[1] - tleg[0]
            if delt == 0.:
                continue

            # ndumps_per_leg is the number of times to write to the output
            # file in this leg
            dump_interval = max(1, int(float(nsteps / self.ndumps)))
            lsn = len(str(nsteps))
            consfmt = ("leg {{0:{0}d}}, step {{1:{1}d}}, time {{2:.4E}}, "
                       "dt {{3:.4E}}".format(lsl, lsn))

            nv = 0
            for i, cij in enumerate(c):
                if ltype[i] == 1:                            # -- strain rate
                    depsdt[i] = cij

                elif ltype[i] == 2:                          # -- strain
                    depsdt[i] = (cij - eps[i]) / delt

                elif ltype[i] == 3:                          # -- stress rate
                    sigdum[1, i] = sigdum[0, i] + cij * delt
                    vdum[nv] = i
                    nv += 1

                elif ltype[i] == 4:                          # -- stress
                    sigdum[1, i] = cij
                    vdum[nv] = i
                    nv += 1

                continue

            sigspec = np.empty((3, nv))
            v = vdum[:nv]
            sigspec[:2] = sigdum[:2, v]
            Jsub = J0[[[x] for x in v], v]

            t = tleg[0]
            dt = delt / nsteps

            if not nv:
                # strain or strain rate prescribed and d is constant over
                # entire leg
                d = deps2d(dt, kappa, eps, depsdt)

                if cfg.sqa and kappa == 0.:
                    if not np.allclose(d, depsdt):
                        log_message("sqa: d != depsdt (k=0, leg"
                                    "={0})".format(leg_num))

            else:
                # Initial guess for d[v]
                try:
                    depsdt[v] = solve(Jsub, (sigspec[1] - sigspec[0]) / delt)
                except:
                    depsdt[v] -= lstsq(Jsub, (sigspec[1] - sigspec[0]) / delt)[0]

            # process this leg
            for n in range(nsteps):

                # increment time
                t += dt

                # interpolate values to the target values for this step
                a1 = float(nsteps - (n + 1)) / nsteps
                a2 = float(n + 1) / nsteps
                sigspec[2] = a1 * sigspec[0] + a2 * sigspec[1]

                # --- find current value of d: sym(velocity gradient)
                if nv:
                    # One or more stresses prescribed
                    # get just the prescribed stress components
                    d = sig2d(self.mtlmdl, dt, depsdt,
                              sig, xtra, v, sigspec[2], self.proportional)

                # compute the current deformation gradient and strain from
                # previous values and the deformation rate
                f, eps = update_deformation(dt, kappa, f, d)

                # update material state
                sigsave = np.array(sig)
                xtrasave = np.array(xtra)
                sig, xtra = self.mtlmdl.update_state(dt, d, sig, xtra, f, ef)

                # -------------------------- quantities derived from final state
                eqeps = np.sqrt(2. / 3. * (np.sum(eps[:3] ** 2)
                                           + 2. * np.sum(eps[3:] ** 2)))
                epsv = np.sum(eps[:3])
                rho = rho * np.exp(-np.sum(d[:3]) * dt)

                pres = -np.sum(sig[:3]) / 3.
                dstress = (sig - sigsave) / dt

                # advance all data after updating state
                glob_step_num += 1
                self.setglobvars(leg_num=leg_num,
                                 step_num=glob_step_num, time_step=dt)

                self.setvars(stress=sig, strain=eps, defgrad=f,
                             symm_l=d, efield=ef, eqstrain=eqeps,
                             vstrain=epsv, density=rho, pressure=pres,
                             dstress=dstress, xtra=xtra)


                # --- write state to file
                endstep = abs(t - tleg[1]) / tleg[1] < 1.E-12
                if (nsteps - n) % dump_interval == 0 or endstep:
                    iomgr(t)

                if n == 0 or round(nsteps / 2.) == n or endstep:
                    io.log_message(consfmt.format(leg_num, n + 1, t, dt))

                if t > termination_time:
                    return 0

                continue  # continue to next step

            continue # continue to next leg


        return 0
