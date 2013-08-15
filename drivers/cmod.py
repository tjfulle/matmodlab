import os
import sys
import numpy as np

import core.kinematics as kin
import utils.tensor as tensor
from utils.tensor import NSYMM, NTENS, NVEC, I9
from utils.errors import Error1
from materials.material import create_material

np.set_printoptions(precision=4)


class ConstitutiveModelDriver(object):
    name = "constitutive model"
    def __init__(self):
        self._variables = []
        self.ndata = 0
        self.data = np.zeros(self.ndata)
        self.data_map = {}

    def setup(self, material, mtlprops):
        """Setup the driver object

        """

        self.mtlmdl = create_material(material)

        # Save the unchecked parameters
        self.mtlmdl.unchecked_params = mtlprops

        # Setup and initialize material model
        self.mtlmdl.setup(mtlprops)
        self.mtlmdl.initialize()

        # Material data is stored in an array with the following shape:
        #     MTLDAT = (2, NDAT)
        # The material data array contains the following data:
        #     MTLDAT[i] -> data at beginning (i=0) of step and current (i=1)
        #     MTLDAT[:, k]     -> kth data point for element, according to:
        #       k =  0:5   -> Stress
        #       k =  6:11  -> Strain
        #       k = 12:20  -> Deformation gradient
        #       k = 21:26* -> Symmetric part of velocity gradient
        #       k = 27:29* -> Skew symmetric part of velocity gradient (vorticity)
        #       k = 30:    -> Material extra variables
        # * Variables are stored only for plotting.
        self.register_variable("STRESS", vtype="SYMTENS")
        self.register_variable("STRAIN", vtype="SYMTENS")
        self.register_variable("DEFGRAD", vtype="TENS")
        self.register_variable("SYMM-L", vtype="SYMTENS")
        self.register_variable("SKEW-L", vtype="SKEWTENS")
        self.register_variable("EFIELD", vtype="VECTOR")

        # register material variables
        self.xtra_start = self.ndata
        self.xtra_end = self.xtra_start + self.mtlmdl.nxtra
        for var in self.mtlmdl.variables():
            self.register_variable(var, vtype="SCALAR")

        # allocate storage
        self.allocd()

        return

    def var_index(self, var):
        """Get the index of var in the data array

        """
        if var.upper() == "XTRA":
            return self.xtra_start, self.xtra_end

        try:
            start, end = self.data_map.get(var.upper())
        except TypeError:
            raise Error1("{0}: not in element data".format(var))
        if end == -1:
            end = self.ndata
        return start, end

    def get_data(self, var=None):
        """Return the current material data

        Returns
        -------
        data : array_like
            Material data

        """
        start, end = 0, self.ndata
        if var is not None:
            start, end = self.var_index(var)
        return self.data[0, start:end]

    def set_data(self, data, var=None, spot=1):
        start, end = 0, self.ndata
        if var is not None:
            start, end = self.var_index(var)
        self.data[spot, start:end] = data

    def advance_state(self):
        """Advance the material state

        """
        self.data[0] = self.data[1]

    def allocd(self):
        """Allocate space for material data

        Notes
        -----
        This must be called after each consititutive model's setup method so
        that the number of xtra variables is known.

        """
        # Model data array.  See comments above.
        self.data = np.zeros((2, self.ndata))

        # initialize nonzero data
        start, end = self.data_map["DEFGRAD"]
        self.data[:, start:end] = I9
        self.data[:, self.xtra_start:self.xtra_end] = self.mtlmdl.initial_state()

    def register_variable(self, var, vtype="SCALAR"):
        """Register material variable

        """
        vtype = vtype.upper()
        name = var.upper()
        if vtype == "SCALAR":
            var = [name]

        elif vtype == "TENS":
            var = ["{0}-{1}".format(name, x) for x in ("XX", "XY", "XZ",
                                                       "YX", "YY", "YZ",
                                                       "ZX", "ZY", "ZZ")]
        elif vtype == "SYMTENS":
            var = ["{0}-{1}".format(name, x)
                   for x in ("XX", "YY", "ZZ", "XY", "YZ", "XZ")]

        elif vtype == "SKEWTENS":
            var = ["{0}-{1}".format(name, x) for x in ("XY", "YZ", "XZ")]

        elif vtype == "VECTOR":
            var = ["{0}-{1}".format(name, x) for x in ("X", "Y", "X")]

        else:
            raise Error1("{0}: unrecognized vtype".format(vtype))

        start = self.ndata
        self.ndata += len(var)
        end = self.ndata
        self.data_map[name] = (start, end)
        self._variables.extend(var)

    def variables(self):
        return self._variables


    def process_leg(self, t_beg, leg_num, leg):
        """Process the current leg

        Parameters
        ----------

        Returns
        -------

        """

        cons_msg = "leg {0:{1}d}, step {2:{3}d}, time {4:.4E}, dt {5:.4E}"

        # -------------------------------------------------------- initialize model

        # ----------------------------------------------------------------------- #
        # V is an array of integers that contains the columns of prescribed stress
        V = []

        Pc = self.get_data("STRESS")
        Ec = self.get_data("STRAIN")

        # --------------------------------------------------- begin{processing leg}

        # pass values from the end of the last leg to beginning of this leg
        E0 = self.get_data("STRAIN")
        P0 = self.get_data("STRESS")
        F0 = self.get_data("DEFGRAD")
        EF0 = self.get_data("EFIELD")

        # read inputs and initialize for this leg
        t_end, nsteps, ltype, Cij, Rij, EFf = leg
        lns = len(str(nsteps))
        delt = t_end - t_beg
        if delt == 0.:
            return
        nv, dflg = 0, list(set(ltype))

        nprints = 10
        print_interval = max(1, int(nsteps / nprints))

        print cons_msg.format(leg_num, len(str(leg_num)), 1, lns, t_beg, delt)

        # --- loop through components of Cij and compute the values at the end
        #     of this leg:
        #       for ltype = 1: Ef at t_end -> E0 + Cij*delt
        #       for ltype = 2: Ef at t_end -> Cij
        #       for ltype = 3: Pf at t_end -> P0 + Cij*delt
        #       for ltype = 4: Pf at t_end -> Cij
        #       for ltype = 6: EFf at t_end-> Cij
        Ef = tensor.Z6
        Ph = tensor.Z6

        # if stress is prescribed, we don't compute Pf just yet, but Ph
        # which holds just those values of stress that are actually prescribed.
        for i, c in enumerate(Cij):

            # -- strain rate
            if ltype[i] == 1:
                Ef[i] = E0[i] + c * delt

            # -- strain
            elif ltype[i] == 2:
                Ef[i] = c

            # stress rate
            elif ltype[i] == 3:
                Ph[i] = P0[i] + c * delt
                Vhld[nv] = i
                nv += 1

            # stress
            elif ltype[i] == 4:
                Ph[i] = c
                Vhld[nv] = i
                nv += 1

            continue

        V = Vhld[0:nv]
        if len(V):
            PS0, PSf = P0[V], Ph[V]

        t = t_beg
        dt = delt / nsteps

        # ---------------------------------------------- begin{processing step}
        for n in range(nsteps):

            # advance data from end of last step to this step
            matdat.advance()
            simdat.advance()

            # increment time
            t += dt

            # interpolate values of E, F, EF, and P for the target values for
            # this step
            a1 = float(nsteps - (n + 1)) / nsteps
            a2 = float(n + 1) / nsteps
            Et = a1 * E0 + a2 * Ef
            Ft = a1 * F0 + a2 * Ff
            EFt = a1 * EF0 + a2 * EFf

            if len(V):
                # prescribed stress components given
                Pt = a1 * PS0 + a2 * PSf  # target stress

            # advance known values to end of step
            simdat.store("time", t)
            simdat.store("time step", dt)
            matdat.store("electric field", EFt)

            # --- find current value of d: sym(velocity gradient)
            if not len(V):

                if dflg[0] == 5:
                    # --- deformation gradient prescribed
                    Dc, Wc = kin.velgrad_from_defgrad(dt, Fc, Ft)

                else:
                    # --- strain or strain rate prescribed
                    Dc, Wc = kin.velgrad_from_strain(dt, K, Ec, Rij, dR, Et)

            else:
                # --- One or more stresses prescribed
                Dc, Wc = kin.velgrad_from_stress(
                    material, simdat, matdat, dt, Ec, Et, Pc, Pt, V)

            # compute the current deformation gradient and strain from
            # previous values and the deformation rate
            Fc, Ec = kin.update_deformation(dt, K, Fc, Dc, Wc)

            # --- update the deformation to the end of the step at this point,
            #     the rate of deformation and vorticity to the end of the step
            #     are known, advance them.
            matdat.store("rate of deformation", Dc)
            matdat.store("vorticity", Wc)
            matdat.store("deformation gradient", Fc)
            matdat.store("strain", Ec)
            matdat.store("vstrain", np.sum(Ec[:3]))
            # compute the equivalent strain
            matdat.store(
                "equivalent strain",
                np.sqrt(2. / 3. * (np.sum(Ec[:3] ** 2) + 2. * np.sum(Ec[3:] ** 2))))

            # udpate density
            dev = tensor.trace(matdat.get("rate of deformation")) * dt
            rho_old = simdat.get("payette density")
            simdat.store("payette density", rho_old * math.exp(-dev))

            # update material state
            material.update_state(simdat, matdat)

            # advance all data after updating state
            Pc = matdat.get("stress")
            matdat.store("stress rate", (Pc - matdat.get("stress", "-")) / dt)
            matdat.store("pressure", -np.sum(Pc[:3]) / 3.)

            # --- write state to file
            endstep =  abs(t - t_end) / t_end < 1.E-12
            if (nsteps - n) % print_interval == 0 or endstep:
                model.write_state()

            if simdat.SCREENOUT or (2 * n - nsteps) == 0:
                print cons_msg.format(leg_num, len(str(leg_num)), n, lns, t, dt)

            # -------------------------------------------- end{end of step SQA}

            continue  # continue to next step

        return
