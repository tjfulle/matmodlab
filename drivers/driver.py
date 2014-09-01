import os
import sys
import numpy as np

from core.varinc import *
from core.variable import Variable
from core.mmlio import exo, logger
from drivers.parsing import parse_default_path, format_continuum_path

D = os.path.dirname(os.path.realpath(__file__))

class FileNotFoundError(Exception):
    def __init__(self, filename):
        message = "{0}: no such file".format(filename)
        super(FileNotFoundError, self).__init__(message)


class _Driver(object):
    kind = None
    _vars = []
    _opts = []
    ran = False

    @property
    def variables(self):
        return self._vars

    @property
    def plot_keys(self):
        return [x for l in [v.keys for v in self.variables] for x in l]

    def register_variable(self, var_name, var_type):
        self._vars.append(Variable(var_name, var_type))


class ContinuumDriver(_Driver):
    kind = "Continuum"
    def __init__(self, path=None, path_file=None, path_input="default",
                 kappa=0., amplitude=1., rate_multiplier=1., step_multiplier=1.,
                 num_io_dumps="all", estar=1., tstar=1., sstar=1., fstar=1.,
                 efstar=1., dstar=1., termination_time=None):

        # Register variables specifically needed by driver
        self.register_variable("STRESS", VAR_SYMTENSOR)
        self.register_variable("STRAIN", VAR_SYMTENSOR)
        self.register_variable("DEFGRAD", VAR_TENSOR)
        self.register_variable("SYMM_L", VAR_SYMTENSOR)
        self.register_variable("EFIELD", VAR_VECTOR)
        self.register_variable("VSTRAIN", VAR_SCALAR)
        self.register_variable("EQSTRAIN", VAR_SCALAR)
        self.register_variable("PRESSURE", VAR_SCALAR)
        self.register_variable("SMISES", VAR_SCALAR)
        self.register_variable("DSTRESS", VAR_SYMTENSOR)
        self.register_variable("TEMP", VAR_SCALAR)

        if path is None and path_file is None:
            raise ValueError("Expected one of path or path_file")
        if path is not None and path_file is not None:
            raise ValueError("Expected only one of path or path_file")

        if path_file is not None:
            if not os.path.isfile(path_file):
                raise FileNotFoundError(path_file)
            path = open(path_file).read()

        path = [line.split() for line in path.split("\n") if line.split()]
        if path_input.lower() == "default":
            path = parse_default_path(path)
        else:
            raise ValueError("{0}: path_input not recognized".format(path_input))

        self.path = format_continuum_path(path, kappa, amplitude,
            rate_multiplier, step_multiplier, num_io_dumps, estar,
            tstar, sstar, fstar, efstar, dstar, termination_time)
        self.kappa = kappa

    def run(self, glob_data, elem_data, material):
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
        legs = self.path[0:]
        kappa = self.kappa

        # initial leg
        glob_step_num = 0
        sig = elem_data["STRESS"]
        xtra = elem_data["XTRA"]
        temp = np.zeros(3)
        print elem_data.keys
        temp[1] = elem_data["TEMP"]
        print temp
        sys.exit("need to make sure all data is initialized, particularly temp")
        tleg = np.zeros(2)
        d = np.zeros(NSYMM)
        dt = 0.
        eps = np.zeros(NSYMM)
        f0 = np.reshape(np.eye(3), (NTENS,))
        f = np.reshape(np.eye(3), (NTENS,))
        depsdt = np.zeros(NSYMM)
        sigdum = np.zeros((2, NSYMM))

        # compute the initial jacobian
        J0 = self.material.constant_jacobian

        # v array is an array of integers that contains the rows and columns of
        # the slice needed in the jacobian subroutine.
        nv = 0
        vdum = np.zeros(NSYMM, dtype=np.int)

        # Process each leg
        nlegs = len(legs)
        lsl = len(str(nlegs))
        for (ileg, leg) in enumerate(legs):
            leg_num = self.start_leg + ileg
            tleg[0] = tleg[1]
            temp[0] = temp[1]
            sigdum[0] = sig[:]
            if nv:
                sigdum[0, v] = sigspec[1]

            tleg[1] = leg[0]
            nsteps = leg[1]
            control = leg[2:8]
            c = leg[8:14]
            ndumps = leg[14]
            ef = leg[15:18]
            temp[1] = leg[18]
            ufield = leg[19:]

            delt = tleg[1] - tleg[0]
            if delt == 0.:
                continue

            # ndumps_per_leg is the number of times to write to the output
            # file in this leg
            dump_interval = max(1, int(nsteps / ndumps))
            lsn = len(str(nsteps))
            consfmt = ("leg {{0:{0}d}}, step {{1:{1}d}}, time {{2:.4E}}, "
                       "dt {{3:.4E}}".format(lsl, lsn))

            nv = 0
            for i, cij in enumerate(c):
                if control[i] == 1:                            # -- strain rate
                    depsdt[i] = cij

                elif control[i] == 2:                          # -- strain
                    depsdt[i] = (cij - eps[i]) / delt

                elif control[i] == 3:                          # -- stress rate
                    sigdum[1, i] = sigdum[0, i] + cij * delt
                    vdum[nv] = i
                    nv += 1

                elif control[i] == 4:                          # -- stress
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
            dtemp = (temp[1] - temp[0]) / nsteps

            if not nv:
                # strain or strain rate prescribed and d is constant over
                # entire leg
                d = mmlabpack.deps2d(dt, kappa, eps, depsdt)

                if opts.sqa and kappa == 0.:
                    if not np.allclose(d, depsdt):
                        log_message("sqa: d != depsdt (k=0, leg"
                                    "={0})".format(leg_num))

            else:
                # Initial guess for d[v]
                try:
                    depsdt[v] = solve(Jsub, (sigspec[1] - sigspec[0]) / delt)
                except:
                    depsdt[v] -= lstsq(Jsub, (sigspec[1] - sigspec[0]) / delt)[0]

            warned = False
            # process this leg
            for n in range(int(nsteps)):

                # increment time
                t += dt
                self.time = t

                # interpolate values to the target values for this step
                a1 = float(nsteps - (n + 1)) / nsteps
                a2 = float(n + 1) / nsteps
                sigspec[2] = a1 * sigspec[0] + a2 * sigspec[1]
                temp[2] = a1 * temp[0] + a2 * temp[1]
                tempn = temp[2] - dtemp

                # --- find current value of d: sym(velocity gradient)
                if nv:
                    # One or more stresses prescribed
                    # get just the prescribed stress components
                    d = sig2d(self.material, t, dt, tempn, dtemp,
                              f0, f, eps, depsdt, sig, xtra, ef, ufield,
                              v, sigspec[2], self.proportional)

                # compute the current deformation gradient and strain from
                # previous values and the deformation rate
                f, eps = mmlabpack.update_deformation(dt, kappa, f0, d)

                # update material state
                sigsave = np.array(sig)
                xtrasave = np.array(xtra)
                sig, xtra = self.material.compute_updated_state(t, dt, tempn,
                    dtemp, f0, f, eps, d, ef, ufield, sig, xtra, last=True, disp=1)

                # -------------------------- quantities derived from final state
                eqeps = np.sqrt(2. / 3. * (np.sum(eps[:3] ** 2)
                                           + 2. * np.sum(eps[3:] ** 2)))
                epsv = np.sum(eps[:3])

                pres = -np.sum(sig[:3]) / 3.
                dstress = (sig - sigsave) / dt
                smises = np.sqrt(3./2.) * mag(dev(sig))
                f0 = f

                # advance all data after updating state
                glob_step_num += 1
                glob_data["LEG_NUM"] = leg_num
                glob_data["STEP_NUM"] = glob_step_num
                glob_data["TIME_STEP"] = dt

                elem_data["STRESS"] = sig
                elem_data["STRAIN"] = eps
                elem_data["DEFGRAD"] = f
                elem_data["SYMM_L"] = d
                elem_data["EQSTRAIN"] = eqeps
                elem_data["vSTRAIN"] = eqeps
                elem_data["DSTRESS"] = epsv
                elem_data["SMISES"] = smises
                elem_data["XTRA"] = xtra
                elem_data["TEMP"] = temp[2]

                # --- write state to file
                endstep = abs(t - tleg[1]) / tleg[1] < 1.E-12
                if (nsteps - n) % dump_interval == 0 or endstep:
                    exo.snapshot(time, glob_data, elem_data)

                if n == 0 or round((nsteps - 1) / 2.) == n or endstep:
                    log_message(consfmt.format(leg_num, n + 1, t, dt))

                if n > 1 and nv and not warned:
                    absmax = lambda a: np.max(np.abs(a))
                    sigerr = np.sqrt(np.sum((sig[v] - sigspec[2]) ** 2))
                    warned = True
                    _tol = np.amax(np.abs(sig[v])) / self.material.bulk_modulus
                    _tol = max(_tol, 1e-4)
                    if sigerr > _tol:
                        log_warning("leg: {0}, prescribed stress error: "
                                    "{1: .5f}. consider increasing number of "
                                    "steps".format(ileg, sigerr))

                continue  # continue to next step

            continue # continue to next leg

        self.ran = True
        return 0


# --------------------------------------------- The Driver factory method --- #
def Driver(kind="Continuum", **kwargs):
    for cls in _Driver.__subclasses__():
        if cls.kind.lower() == kind.lower():
            return cls(**kwargs)
    raise ValueError, "{0}: unrecognized driver kind".format(kind)
