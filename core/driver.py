import os
import sys
import numpy as np
from numpy.linalg import solve, lstsq

from core.logger import Logger
from core.legs import SingleLeg, LegRepository
from utils.variable import Variable
from utils.variable import VAR_SYMTENSOR, VAR_TENSOR, VAR_SCALAR, VAR_VECTOR
from utils.errors import FileNotFoundError, MatModLabError
from core.runtime import opts
from utils.constants import NSYMM, NTENS, I9, VOIGHT
import utils.mmlabpack as mmlabpack
from core.solvers import sig2d
from core.dparse import continuum_legs, cflags


class PathDriver(object):
    kind = None
    ran = False
    _logger = None
    itemp = None

    @property
    def variables(self):
        return self._vars

    @property
    def logger(self):
        return self._logger

    @logger.setter
    def logger(self, new_logger):
        try:
            new_logger.write
            new_logger.warn
            new_logger.error
        except AttributeError, TypeError:
            raise TypeError("attempting to assign a non logger "
                            "to the {0} Driver logger".format(self.kind))
        self._logger = new_logger

    @property
    def initial_temp(self):
        return self.itemp

    @initial_temp.setter
    def initial_temp(self, value):
        self.itemp = value

    @property
    def num_leg(self):
        return len(self.legs)

    @property
    def num_steps(self):
        return int(sum([x.num_steps for x in self.legs.values()]))

    def register_variable(self, var_name, var_type, initial_value=None):
        self._vars.append(Variable(var_name, var_type, initial_value=initial_value))


class ContinuumDriver(PathDriver):
    kind = "Continuum"
    def __init__(self, input, path_input="default",
                 kappa=0., amplitude=1., rate_multiplier=1., step_multiplier=1.,
                 num_io_dumps="all", estar=1., tstar=1., sstar=1., fstar=1.,
                 efstar=1., dstar=1., proportional=False, termination_time=None,
                 functions=None, cfmt=None, tfmt="time", num_steps=None,
                 cols=None, skiprows=0, logger=None):

        if logger is None:
            logger = Logger("driver", filename=None)
        self.logger = logger
        self.logger.write("setting up the {0} driver".format(self.kind))

        self._vars = []
        self.kappa = kappa
        self.proportional = proportional

        if isinstance(input[0], SingleLeg):
            self.legs = LegRepository(input)
        else:
            try:
                input = [line.split() for line in input.split("\n") if line.split()]
            except AttributeError:
                pass
            legs = continuum_legs(path_input, input, num_steps, amplitude,
                                  rate_multiplier, step_multiplier,
                                  num_io_dumps, termination_time, tfmt,
                                  cols, cfmt, skiprows, functions, kappa,
                                  estar, tstar, sstar, fstar, efstar, dstar)
            self.legs = LegRepository(legs)
        self.itemp = self.legs.values()[0].temp

        # Register variables specifically needed by driver
        self.register_variable("STRESS", VAR_SYMTENSOR)
        self.register_variable("STRAIN", VAR_SYMTENSOR)
        self.register_variable("DEFGRAD", VAR_TENSOR, initial_value=I9)
        self.register_variable("SYMM_L", VAR_SYMTENSOR)
        self.register_variable("EFIELD", VAR_VECTOR)
        self.register_variable("VSTRAIN", VAR_SCALAR)
        self.register_variable("EQSTRAIN", VAR_SCALAR)
        self.register_variable("PRESSURE", VAR_SCALAR)
        self.register_variable("SMISES", VAR_SCALAR)
        self.register_variable("DSTRESS", VAR_SYMTENSOR)
        self.register_variable("TEMP", VAR_SCALAR, initial_value=self.itemp)

    def tostr(self, obj="mps"):
        """Write the python code neccesary to create this path"""
        # representation of legs
        legs = ", ".join([self.lrepr(l) for (i, l) in self.legs.items()])
        string = "legs = [{0}]\n"
        string += "{1}.Driver('{2}', legs, kappa={3}, proportional={4})\n"
        return string.format(legs, obj, self.kind,
                             self.kappa, self.proportional)

    def lrepr(self, leg):
        c = [(cflags(a, r=1), b) for (a, b) in zip(leg.control, leg.components)]
        ef = [float(a) for a in leg.elec_field]
        uf = [float(a) for a in leg.user_field]
        string = "Leg({0}, {1}, {2}, num_steps={3}, num_io_dumps={4}, "
        string += "elec_field={5}, temp={6}, user_field={7})"
        return string.format(leg.start_time, leg.dtime, c, leg.num_steps,
                             leg.num_dumps, ef, leg.temp, uf)

    def run(self, glob_data, elem_data, material, out_db, bp,
            termination_time=None):
        """Process the deformation path

        Parameters
        ----------
        material : Material instance
            The material model

        Returns
        -------
        stat : int
            == 0 -> success
            != 0 -> fail

        """
        kappa = self.kappa

        # initial leg
        glob_step_num = 0
        sig = elem_data["STRESS"]
        xtra = elem_data["XTRA"]
        temp = np.zeros(3)
        temp[1] = elem_data["TEMP"]
        tleg = np.zeros(2)
        d = np.zeros(NSYMM)
        dt = 0.
        eps = np.zeros(NSYMM)
        f0 = elem_data["DEFGRAD"]
        f = np.reshape(np.eye(3), (NTENS,))
        depsdt = np.zeros(NSYMM)
        sigdum = np.zeros((2, NSYMM))

        # compute the initial jacobian
        J0 = material.constant_jacobian
        if J0 is None:
            self.logger.raise_error("J0 has not been initialized")

        # v array is an array of integers that contains the rows and columns of
        # the slice needed in the jacobian subroutine.
        nv = 0
        vdum = np.zeros(NSYMM, dtype=np.int)

        # Process each leg
        lsl = len(str(self.num_leg))
        start_leg = 0

        for (ileg, leg) in self.legs.items():
            leg_num = start_leg + ileg
            tleg[0] = tleg[1]
            temp[0] = temp[1]
            sigdum[0] = sig[:]
            if nv:
                sigdum[0, v] = sigspec[1]

            tleg[1] = leg.termination_time
            temp[1] = leg.temp

            delt = tleg[1] - tleg[0]
            if delt == 0.:
                continue

            # ndumps_per_leg is the number of times to write to the output
            # file in this leg
            dump_interval = max(1, int(leg.num_steps / leg.num_dumps))
            lsn = len(str(leg.num_steps))
            consfmt = ("leg {{0:{0}d}}, step {{1:{1}d}}, time {{2:.4E}}, "
                       "dt {{3:.4E}}".format(lsl, lsn))

            nv = 0
            for i, cij in enumerate(leg.components):
                if leg.control[i] == 1:                        # -- strain rate
                    depsdt[i] = cij * VOIGHT[i]

                elif leg.control[i] == 2:                      # -- strain
                    depsdt[i] = (cij * VOIGHT[i] - eps[i]) / delt

                elif leg.control[i] == 3:                      # -- stress rate
                    sigdum[1, i] = sigdum[0, i] + cij * delt
                    vdum[nv] = i
                    nv += 1

                elif leg.control[i] == 4:                      # -- stress
                    sigdum[1, i] = cij
                    vdum[nv] = i
                    nv += 1

                continue

            sigspec = np.empty((3, nv))
            v = vdum[:nv]
            sigspec[:2] = sigdum[:2, v]
            Jsub = J0[[[x] for x in v], v]

            time = tleg[0]
            dt = delt / leg.num_steps
            dtemp = (temp[1] - temp[0]) / leg.num_steps

            if not nv:
                # strain or strain rate prescribed and d is constant over
                # entire leg
                d = mmlabpack.deps2d(dt, kappa, eps, depsdt)

                if opts.sqa and kappa == 0.:
                    if not np.allclose(d, depsdt):
                        self.logger.write("sqa: d != depsdt (k=0, leg"
                                          "={0})".format(leg_num))

            else:
                # Initial guess for d[v]
                try:
                    depsdt[v] = solve(Jsub, (sigspec[1] - sigspec[0]) / delt)
                except:
                    depsdt[v] -= lstsq(Jsub, (sigspec[1] - sigspec[0]) / delt)[0]

            warned = False
            # process this leg
            for n in range(int(leg.num_steps)):

                # increment time
                time += dt
                self.time = time

                # interpolate values to the target values for this step
                a1 = float(leg.num_steps - (n + 1)) / leg.num_steps
                a2 = float(n + 1) / leg.num_steps
                sigspec[2] = a1 * sigspec[0] + a2 * sigspec[1]
                temp[2] = a1 * temp[0] + a2 * temp[1]
                tempn = temp[2] - dtemp

                # --- find current value of d: sym(velocity gradient)
                if nv:
                    # One or more stresses prescribed
                    # get just the prescribed stress components
                    d = sig2d(material, time, dt, tempn, dtemp,
                              kappa, f0, f, eps, depsdt, sig, xtra, leg.elec_field,
                              leg.user_field, v, sigspec[2], self.proportional,
                              self.logger)

                # compute the current deformation gradient and strain from
                # previous values and the deformation rate
                f, eps = mmlabpack.update_deformation(dt, kappa, f0, d)

                # update material state
                sigsave = np.array(sig)
                xtrasave = np.array(xtra)
                sig, xtra = material.compute_updated_state(time, dt, tempn,
                    dtemp, kappa, f0, f, eps, d, leg.elec_field, leg.user_field,
                    sig, xtra, last=True, disp=1)

                # -------------------------- quantities derived from final state
                eqeps = np.sqrt(2. / 3. * (np.sum(eps[:3] ** 2)
                                           + .5 * np.sum(eps[3:] ** 2)))
                epsv = np.sum(eps[:3])

                pres = -np.sum(sig[:3]) / 3.
                dstress = (sig - sigsave) / dt
                smises = np.sqrt(3./2.) * mmlabpack.mag(mmlabpack.dev(sig))
                f0 = f

                # advance all data after updating state
                glob_step_num += 1
                glob_data.update(leg_num=leg_num, step_num=glob_step_num,
                                 time_step=dt)

                elem_data.update(stress=sig, strain=eps/VOIGHT, defgrad=f,
                                 symm_l=d/VOIGHT, eqstrain=eqeps,
                                 vstrain=epsv, dstress=dstress,
                                 smises=smises, xtra=xtra, temp=temp[2],
                                 pressure=pres)

                # --- write state to file
                endstep = abs(time - tleg[1]) / tleg[1] < 1.E-12
                if (leg.num_steps - n) % dump_interval == 0 or endstep:
                    out_db.snapshot(time, glob_data, elem_data)

                if n == 0 or round((leg.num_steps - 1) / 2.) == n or endstep:
                    self.logger.write(consfmt.format(leg_num, n + 1, time, dt))

                if n > 1 and nv and not warned:
                    absmax = lambda a: np.max(np.abs(a))
                    sigerr = np.sqrt(np.sum((sig[v] - sigspec[2]) ** 2))
                    warned = True
                    _tol = np.amax(np.abs(sig[v])) / material.completions["K"]
                    _tol = max(_tol, 1e-4)
                    if sigerr > _tol:
                        self.logger.warn("leg: {0}, prescribed stress error: "
                                         "{1: .5f}. consider increasing number of "
                                         "steps".format(ileg, sigerr))

                if bp:
                    bp.eval(time, glob_data, elem_data)

                if termination_time is not None and time >= termination_time:
                    self.ran = True
                    return 0

                continue  # continue to next step

            continue # continue to next leg

        self.ran = True
        return 0


# --------------------------------------------- The Driver factory method --- #
def Driver(kind, path, **kwargs):
    for cls in PathDriver.__subclasses__():
        if cls.kind.lower() == kind.lower():
            return cls(path, **kwargs)
    raise MatModLabError("{0}: unrecognized driver kind".format(kind))
