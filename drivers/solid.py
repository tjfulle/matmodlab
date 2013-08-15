import os
import sys
import numpy as np

import core.kinematics as kin
import utils.tensor as tensor
from utils.tensor import NSYMM, NTENS, NVEC, I9
from utils.errors import Error1
from materials.material import create_material

np.set_printoptions(precision=2)


class SolidDriver(object):
    name = "solid"
    def __init__(self):
        self._variables = []
        self.ndata = 0
        self.data = np.zeros(self.ndata)
        self.data_map = {}

    def setup(self, material, mtlprops, *opts):
        """Setup the driver object

        """

        self.mtlmdl = create_material(material)

        # Save the unchecked parameters
        self.mtlmdl.unchecked_params = mtlprops

        # Setup and initialize material model
        self.mtlmdl.setup(mtlprops)
        self.mtlmdl.initialize()

        self.kappa, self.density = opts[:2]

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
        self.register_variable("EQSTRAIN", vtype="SCALAR")
        self.register_variable("VSTRAIN", vtype="SCALAR")
        self.register_variable("DENSITY", vtype="SCALAR")
        self.register_variable("PRESSURE", vtype="SCALAR")
        self.register_variable("DSTRESS", vtype="SYMTENS")

        # register material variables
        self.xtra_start = self.ndata
        self.xtra_end = self.xtra_start + self.mtlmdl.nxtra
        for var in self.mtlmdl.variables():
            self.register_variable(var, vtype="SCALAR")
        setattr(self, "xtra_slice", slice(self.xtra_start, self.xtra_end))

        # allocate storage
        self.allocd()

        # initialize nonzero data
        self.data[:, self.defgrad_slice] = I9
        self.data[:, self.xtra_slice] = self.mtlmdl.initial_state()
        self.data[0, self.density_slice] = self.density

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

    def get_data(self, var=None, rows=0):
        """Return the current material data

        Returns
        -------
        data : array_like
            Material data

        """
        start, end = 0, self.ndata
        if var is not None:
            start, end = self.var_index(var)
        return self.data[rows, start:end]

    def setopt(self, opt, val):
        setattr(self, opt, val)

    def set_data(self, data, var=None):
        start, end = 0, self.ndata
        if var is not None:
            start, end = self.var_index(var)
        self.data[0, start:end] = data

    def advance_data(self):
        """Advance the material state

        """
        self.data[0] = self.data[2]

    def allocd(self):
        """Allocate space for material data

        Notes
        -----
        This must be called after each consititutive model's setup method so
        that the number of xtra variables is known.

        """
        # Model data array.  See comments above.
        self.data = np.zeros((3, self.ndata))

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
        setattr(self, "{0}_slice".format(name.lower()), slice(start, end))

    def variables(self):
        return self._variables


    def process_leg(self, gmd, t_beg, leg_num, leg):
        """Process the current leg

        Parameters
        ----------

        Returns
        -------

        """
        t_end, nsteps, ltype, Cij, Rij, EFf = leg
        dR = np.zeros(9)

        # --- console message to write to screen
        consfmt = ("leg {{0:{0}d}}, step {{1:{1}d}}, time {{2:.4E}}, dt "
                   "{{3:.4E}}".format(len(str(leg_num)), len(str(nsteps))))

        # v is an array of integers that contains the columns of prescribed stress
        nv = 0
        v = np.zeros(NSYMM, dtype=np.int)

        # --------------------------------------------------- begin{processing leg}
        # --- data from end of the last leg -> beginning of this leg
        # row 0 -> beginning of leg
        #     1 ->       end of leg
        #     2 ->          current
        strain = self.data[:, self.strain_slice]
        stress = self.data[:, self.stress_slice]
        dstress = self.data[:, self.dstress_slice]
        efield = self.data[:, self.efield_slice]
        defgrad = self.data[:, self.defgrad_slice]
        xtra = self.data[:, self.xtra_slice]
        for item in (strain, stress, efield, defgrad, xtra, dstress):
            item[2] = item[0]
        efield[1] = EFf

        density = self.data[:, self.density_slice]
        eqstrain = self.data[:, self.eqstrain_slice]
        vstrain = self.data[:, self.vstrain_slice]
        pres = self.data[:, self.pressure_slice]

        # holder for computing the stress rate
        old_stress = np.zeros(NSYMM)

        # read inputs and initialize this leg
        delt = t_end - t_beg
        if delt == 0.:
            return

        nprints = 10
        print_interval = max(1, int(nsteps / nprints))

        # --- loop through components of Cij and compute the values at the end
        #     of this leg:
        #       for ltype = 1: strain at t_end -> strain[0] + Cij*delt
        #       for ltype = 2: strain at t_end -> Cij
        #       for ltype = 3: stress at t_end -> stress[0] + Cij*delt
        #       for ltype = 4: stress at t_end -> Cij
        for i, Ci in enumerate(Cij):

            # -- strain rate
            if ltype[i] == 1:
                strain[1, i] = strain[0, i] + Ci * delt

            # -- strain
            elif ltype[i] == 2:
                strain[1, i] = Ci

            # stress rate
            elif ltype[i] == 3:
                stress[1, i] = stress[0, i] + Ci * delt
                v[nv] = i
                nv += 1

            # stress
            elif ltype[i] == 4:
                stress[1, i] = Ci
                v[nv] = i
                nv += 1

            continue
        v = v[0:nv]

        t = t_beg
        dt = delt / nsteps

        # ---------------------------------------------- begin{processing step}
        for n in range(nsteps):

            # increment time
            t += dt

            # interpolate values of strain, stress, and electric field to the
            # target values for this step
            a1 = float(nsteps - (n + 1)) / nsteps
            a2 = float(n + 1) / nsteps
            trg_strain = a1 * strain[0] + a2 * strain[1]
            trg_efield = a1 * efield[0] + a2 * efield[1]

            # --- find current value of d: sym(velocity gradient)
            if nv:
                # One or more stresses prescribed
                # get just the prescribed stress components
                trg_stress = a1 * stress[0, v] + a2 * stress[1, v]  # target stress
                d, w = kin.velgrad_from_stress(material, simdat, matdat, dt,
                                               eps, strain[2], stress[2],
                                               trg_stress, v)
            else:
                # strain or strain rate prescribed
                d, w = kin.velgrad_from_strain(dt, self.kappa, strain[2],
                                               Rij, dR, trg_strain)

            # compute the current deformation gradient and strain from
            # previous values and the deformation rate
            defgrad[2], strain[2] = kin.update_deformation(dt, self.kappa,
                                                           defgrad[2], d, w)
            # quantities derived from strain
            eqstrain[2, 0] = np.sqrt(2. / 3. * (np.sum(strain[2, :3] ** 2)
                                                + 2. * np.sum(strain[2, 3:] ** 2)))
            vstrain[2, 0] = tensor.trace(strain[2, :3])
            density[2, 0] = density[2, 0] * np.exp(-tensor.trace(d) * dt)

            # update material state
            old_stress[:] = stress[2]
            stress[2], xtra[2] = self.mtlmdl.update_state(dt, d,
                                                          stress[2], xtra[2])

            # advance all data after updating state
            pres[2, 0] = -tensor.trace(stress[2, :3]) / 3.
            dstress[2] = (stress[2] - old_stress) / dt

            self.advance_data()

            # --- write state to file
            endstep = abs(t - t_end) / t_end < 1.E-12
            if (nsteps - n) % print_interval == 0 and not endstep:
                gmd.dump_state(dt, t_end)

            if n == 0 or round(nsteps / 2.) == n or endstep:
                print consfmt.format(leg_num, n + 1, t, dt)

            # -------------------------------------------- end{end of step SQA}

            continue  # continue to next step

        return
