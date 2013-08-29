import os
import sys
import math
import numpy as np

from core.io import Error1, fatal_inp_error, input_errors
from utils.opthold import OptionHolder, OptionHolderError as OptionHolderError

np.set_printoptions(precision=4)

EPSILON = np.finfo(np.float).eps
TOL = 1.E-09


def EOSDriver(Driver):
    name = "eos"
    def __init__(self):
        pass

    def setup(self, runid, material, *opts):
        """Setup the driver object

        """
        self.runid = runid
        self.mtlmdl = create_material(material[0])

        self.register_glob_variable("TIME_STEP")
        self.register_glob_variable("STEP_NUM")
        self.register_glob_variable("LEG_NUM")
        self.register_variable("DENSITY", vtype="SCALAR",
                               key="RHO", units="DENSITY_UNITS")
        self.register_variable("TEMPERATURE", vtype="SCALAR",
                               key="TEMP", units="TEMPERATURE_UNITS")
        self.register_variable("ENERGY", vtype="SCALAR",
                               key="ENRG", units="SPECIFIC_ENERGY_UNITS")
        self.register_variable("PRESSURE", vtype="SCALAR",
                               key="PRES", units="PRESSURE_UNITS")

        # Setup
        self.mtlmdl.setup(material[1])

        # register material variables
        self.xtra_start = self.ndata
        for (var, vtype) in self.mtlmdl.material_variables():
            self.register_variable(var, vtype=vtype)

        nxtra = self.ndata - self.xtra_start
        self.xtra_end = self.xtra_start + nxtra
        setattr(self, "xtra_slice", slice(self.xtra_start, self.xtra_end))

        # allocate storage
        self.allocd()

        return


    def process_legs(self, legs, iomgr, *args):

        # legs are of form:
        # legs[0]: surface
        #          [R0, Rf, T0, Tf Ns]
        # legs[1]: rho/density pairs
        #          [[R0, T0], [R1, T1], ..., [Rn, Tn]]
        # legs[2]: hugoniot path
        #          [T0, R0, Rf, Ns]
        # legs[3]: isotherm path
        #          [T, R0, Rf, Ns]
        cons_msg = "leg {0:{1}d}, step {2:{3}d}, time {4:.4E}, dt {5:.4E}"

        K2eV = 8.617343e-5
        erg2joule = 1.0e-4

        # jrh: Added this dict to get extra file names
        extra_files = kwargs.get("extra_files")
        if extra_files is None:
            extra_files = {}
        extra_files['path files'] = {}

        simdat = the_model.simulation_data()
        material = the_model.material
        matdat = material.material_data()

        eos_model = material.constitutive_model

        # This will make sure that everything has units set.
        eos_model.ensure_all_parameters_have_valid_units()
        simdat.ensure_valid_units()
        matdat.ensure_valid_units()

        nprints = simdat.NPRINTS

        # get boundary data
        Rrange = the_model.boundary.density_range()
        Trange = the_model.boundary.temperature_range()
        Sinc = the_model.boundary.surface_increments()
        Pinc = the_model.boundary.path_increments()
        isotherm = the_model.boundary.path_isotherm()
        hugoniot = the_model.boundary.path_hugoniot()
        RT_pairs = the_model.boundary.rho_temp_pairs()
        simdir = the_model.simdir
        simnam = the_model.name

        input_unit_system = the_model.boundary.input_units()
        output_unit_system = the_model.boundary.output_units()

        if not UnitManager.is_valid_unit_system(input_unit_system):
            pu.report_and_raise_error(
                "Input unit system '{0}' is not a valid unit system"
                .format(input_unit_system))
        if not UnitManager.is_valid_unit_system(output_unit_system):
            pu.report_and_raise_error(
                "Output unit system '{0}' is not a valid unit system"
                .format(output_unit_system))

        pu.log_message("Input unit system: {0}".format(input_unit_system))
        pu.log_message("Output unit system: {0}".format(output_unit_system))

        # -------------------------------------------- DENSITY-TEMPERATURE LEGS ---
        if RT_pairs:
            out_fnam = os.path.join(simdir, simnam + ".out")
            the_model._setup_out_file(out_fnam)

            for pair in RT_pairs:
                eos_model.evaluate_eos(simdat, matdat, input_unit_system,
                                       rho=pair[0], temp=pair[1])
                matdat.advance()
                simdat.advance()
                the_model.write_state(iu=input_unit_system, ou=output_unit_system)
                ro.ISTEP += 1
            pu.log_message("Legs file: {0}".format(out_fnam))

        # ------------------------------------------------------------- SURFACE ---
        if all(x is not None for x in (Sinc, Rrange, Trange,)):

            T0, Tf = Trange
            R0, Rf = Rrange
            Ns = Sinc

            out_fnam = os.path.join(simdir, simnam + ".surface")
            the_model._setup_out_file(out_fnam)

            pu.log_message("=" * (80 - 6))
            pu.log_message("Begin surface")
            DEJAVU = False
            idx = 0
            for rho in np.linspace(R0, Rf, Ns):
                for temp in np.linspace(T0, Tf, Ns):
                    idx += 1
                    if idx % int(Ns ** 2 / float(nprints)) == 0:
                        pu.log_message(
                            "Surface step {0}/{1}".format(idx, Ns ** 2))

                    eos_model.evaluate_eos(simdat, matdat, input_unit_system,
                                           rho=rho, temp=temp)
                    matdat.advance()
                    simdat.advance()
                    the_model.write_state(
                        iu=input_unit_system, ou=output_unit_system)
                    ro.ISTEP += 1

            pu.log_message("End surface")
            pu.log_message("Surface file: {0}".format(out_fnam))
            extra_files['surface file'] = out_fnam

        # ------------------------------------------------------------ ISOTHERM ---
        if all(x is not None for x in (Pinc, isotherm, Rrange,)):

            # isotherm = [density, temperature]
            R0I, TfI = isotherm
            R0, Rf = Rrange
            Np = Pinc

            if not R0 <= R0I <= Rf:
                pu.report_and_raise_error(
                    "initial isotherm density not within range")

            out_fnam = os.path.join(simdir, simnam + ".isotherm")
            the_model._setup_out_file(out_fnam)

            pu.log_message("=" * (80 - 6))
            pu.log_message("Begin isotherm")
            DEJAVU = False
            idx = 0
            for rho in np.linspace(R0I, Rf, Np):
                idx += 1
                if idx % int(Np / float(nprints)) == 0:
                    pu.log_message("Isotherm step {0}/{1}".format(idx, Np))

                eos_model.evaluate_eos(simdat, matdat, input_unit_system,
                                       rho=rho, temp=TfI)

                matdat.advance()
                simdat.advance()
                the_model.write_state(iu=input_unit_system, ou=output_unit_system)
                ro.ISTEP += 1

            pu.log_message("End isotherm")
            pu.log_message("Isotherm file: {0}".format(out_fnam))
            extra_files['path files']['isotherm'] = out_fnam

        # ------------------------------------------------------------ HUGONIOT ---
        if all(x is not None for x in (Pinc, hugoniot, Trange, Rrange,)):

            # hugoniot = [density, temperature]
            RH, TH = hugoniot
            R0, Rf = Rrange
            T0, Tf = Trange
            Np = Pinc

            if not R0 <= RH <= Rf:
                pu.report_and_raise_error(
                    "initial hugoniot density not within range")
            if not T0 <= TH <= Tf:
                pu.report_and_raise_error(
                    "initial hugoniot temperature not within range")

            out_fnam = os.path.join(simdir, simnam + ".hugoniot")
            the_model._setup_out_file(out_fnam)

            pu.log_message("=" * (80 - 6))
            pu.log_message("Begin Hugoniot")

            init_density_MKSK = RH
            init_temperature_MKSK = TH

            # Convert to CGSEV
            init_density = init_density_MKSK
            init_temperature = init_temperature_MKSK

            eos_model.evaluate_eos(simdat, matdat, input_unit_system,
                                   rho=init_density, temp=init_temperature)

            init_energy = matdat.get("energy")
            init_pressure = matdat.get("pressure")

            idx = 0
            DEJAVU = False
            e = init_energy
            for rho in np.linspace(RH, Rf, Np):
                idx += 1
                if idx % int(Np / float(nprints)) == 0:
                    pu.log_message("Hugoniot step {0}/{1}".format(idx, Np))

                # Here we solve the Rankine-Hugoniot equation as
                # a function of energy with constant density:
                #
                # E-E0 == 0.5*[P(E,V)+P0]*(V0-V)
                #
                # Where V0 = 1/rho0 and V = 1/rho. We rewrite it as:
                #
                # 0.5*[P(E,V)+P0]*(V0-V)-E+E0 == 0.0 = f(E)
                #
                # The derivative is given by:
                #
                # df(E)/dE = 0.5*(dP/dE)*(1/rho0 - 1/rho) - 1
                #
                # The solution to the first equation is found by a simple
                    # application of newton's method:
                #
                # x_n+1 = x_n - f(E)/(df(E)/dE)

                r = rho
                a = (1. / init_density - 1. / r) / 2.

                converged_idx = 0
                CONVERGED = False
                while not CONVERGED:
                    converged_idx += 1
                    eos_model.evaluate_eos(simdat, matdat, input_unit_system,
                                           rho=r, enrg=e)

                    f = (matdat.get(
                        "pressure") + init_pressure) * a - e + init_energy
                    df = matdat.get("dpdt") / matdat.get("dedt") * a - 1.0
                    e = e - f / df
                    errval = abs(f / init_energy)
                    if errval < TOL:
                        CONVERGED = True
                        if converged_idx > 100:
                            pu.log_message(
                                "Max iterations reached (tol={0:14.10e}).\n".format(TOL) +
                                "rel error   = {0:14.10e}\n".format(float(errval)) +
                                "abs error   = {0:14.10e}\n".format(float(f)) +
                                "func val    = {0:14.10e}\n".format(float(f)) +
                                "init_energy = {0:14.10e}\n".format(float(init_energy)))
                            break

                matdat.advance()
                simdat.advance()
                the_model.write_state(iu=input_unit_system, ou=output_unit_system)
                ro.ISTEP += 1

            pu.log_message("End Hugoniot")
            pu.log_message("Hugoniot file: {0}".format(out_fnam))
            extra_files['path files']['hugoniot'] = out_fnam

        return 0

    # --------------------------------------------------------- Parsing methods
    @classmethod
    def parse_and_register_paths(cls, pathlmns, *args):
        """Parse the Path elements of the input file and register the formatted
        paths to the class

        """
        path_fcns = {"hugoniot": cls.pHugoniot, "isotherm": cls.pIsotherm,
                     "prstate": cls.pPrstate}

        paths = {}
        for pathlmn in pathlmns:
            ptype = pathlmns.getAttribute("type")
            if not ptype:
                fatal_inp_error("Path requires type attribute")
                continue
            pathlmn.removeAttribute("type")
            parse_fcn = path_fcns.get(ptype.strip().lower())
            if parse_fcn is None:
                fatal_inp_error("{0}: unkown Path type".format(ptype))
                continue
            paths[ptype] = parse_fcn(pathlmn, *args)

        if input_errors():
            raise Error1("EOS Driver: Stopping due to input errors")

        sys.exit("not finished with EOS driver")

        return 0

    @staticmethod
    def pHugoniot(pathlmn, *args):
        """Parse the Path block and set defaults

        """
        iam = "Path: Hugoniot"
        # Set up options for Hugoniot path
        options = OptionHolder()
        options.addopt("increments", 10, dtype=int, test=lambda x: x > 0)

        # Get control terms
        for i in range(pathlmn.attributes.length):
            name, value = xmltools.item_name_and_value(pathlmn.attributes.item(i))
            try:
                options.setopt(name, value)
            except OptionHolderError, e:
                fatal_inp_error(e.message)
                continue

        # parse the individual elements
        rho, tmpr = None, None
        for node in pathlmn.childNodes:
            if node.nodeType != node.ELEMENT_NODE:
                continue

            name, val = xmltools.node_name_and_value(node)
            if name == "DensityRange":
                rho = density_range(val)
                if rho is None:
                    fatal_inp_error("{0}: error parsing density range".format(iam))
                    return

            elif name == "InitialTemperature":
                try:
                    tmpr = float(val)
                except ValueError:
                    fatal_inp_error("{0}: {1}: invalid initial "
                                    "temperature".format(val))
                    return

            else:
                fatal_inp_error("{0}: {1}: unrecognized "
                                "element".format(name))

            if rho is None:
                fatal_inp_error("{0}: missing required element: "
                                "DensityRange".format(iam))

            if tmpr is None:
                fatal_inp_error("{0}: missing required element: "
                                "InitialTemperature".format(iam))
            elif tmpr < 0:
                fatal_inp_error("{0}: InitialTemperature must be > 0".format(iam))

        return rho, tmpr

    @classmethod
    def parse_path_default(cls, lines):
        """Parse the individual path

        """
        path = []
        final_time = 0.
        leg_num = 1
        for line in lines:
            if not line:
                continue
            termination_time, num_steps, control_hold = line[:3]
            Cij_hold = line[3:]

            # check entries
            # --- termination time
            try:
                termination_time = float(termination_time)
            except ValueError:
                raise Error1("Path: termination time of leg {0} must be a float, "
                             "got {1}".format(leg_num, termination_time))
            if termination_time < 0.:
                raise Error1("Path: termination time {0} of leg {1} must be "
                             "positive".format(termination_time, leg_num))
            elif termination_time < final_time:
                raise Error("Path: time must increase monitonically at leg "
                            "{0}".format(leg_num))
            final_time = termination_time

            # --- number of steps
            try:
                num_steps = int(num_steps)
            except ValueError:
                raise Error1("Path: number of steps of leg {0} must be an integer, "
                             "got {1}".format(leg_num, num_steps))
            if num_steps < 0:
                raise Error1("Path: number of steps {0} of leg {1} must be "
                             "positive".format(num_steps, leg_num))

            # --- control
            control = cls.format_path_control(control_hold, leg_num=leg_num)

            # --- Cij
            Cij = []
            for (i, comp) in enumerate(Cij_hold):
                try:
                    comp = float(comp)
                except ValueError:
                    raise Error1("Path: Component {0} of leg {1} must be a "
                                 "float, got {2}".format(i+1, leg_num, comp))
                Cij.append(comp)

            Cij = np.array(Cij)

            # --- Check lengths of Cij and control are consistent
            if len(Cij) != len(control):
                raise Error1("Path: len(Cij) != len(control) in leg {0}"
                             .format(leg_num))

            path.append([termination_time, num_steps, control, Cij])
            leg_num += 1
            continue

        return path

    @classmethod
    def parse_path_table(cls, lines, tbltfmt, tblcols, tblcfmt):
        """Parse the path table

        """
        path = []
        final_time = 0.
        termination_time = 0.
        leg_num = 1

        # Convert tblcols to a list
        columns = cls.format_tbl_cols(tblcols)

        # check the control
        control = cls.format_path_control(tblcfmt)

        for line in lines:
            if not line:
                continue
            try:
                line = np.array([float(x) for x in line])
            except ValueError:
                raise Error1("Expected floats in leg {0}, got {1}".format(
                    leg_num, line))
            try:
                line = line[columns]
            except IndexError:
                raise Error1("Requested column not found in leg "
                             "{0}".format(leg_num))

            if tbltfmt == "dt":
                termination_time += line[0]
            else:
                termination_time = line[0]

            Cij = line[1:]

            # check entries
            # --- termination time
            if termination_time < 0.:
                raise Error1("Path: termination time {0} of leg {1} must be "
                             "positive".format(termination_time, leg_num))
            elif termination_time < final_time:
                raise Error("Path: time must increase monitonically at leg "
                            "{0}".format(leg_num))
            final_time = termination_time

            # --- number of steps
            num_steps = 1

            # --- Check lengths of Cij and control are consistent
            if len(Cij) != len(control):
                raise Error1("Path: len(Cij) != len(control) in leg {0}"
                             .format(leg_num))

            path.append([termination_time, num_steps, control, Cij])
            leg_num += 1
            continue

        return path

    @classmethod
    def parse_path_cijfcn(cls, lines, functions):
        """Parse the path given by functions

        """
        start_time = 0.
        leg_num = 1

        if not lines:
            raise Error1("No table functions defined")
        elif len(lines) > 1:
            raise Error1("Only one line of table functions allowed, "
                         "got {0}".format(len(lines)))

        termination_time, num_steps, control_hold = lines[0][:3]
        cijfcns = lines[0][3:]

        # check entries
        # --- termination time
        try:
            termination_time = float(termination_time)
        except ValueError:
            raise Error1("Path: termination time must be a float, "
                         "got {0}".format(termination_time))
        if termination_time < 0.:
            raise Error1("Path: termination time {0} must be "
                         "positive".format(termination_time))
        final_time = termination_time

        # --- number of steps
        try:
            num_steps = int(num_steps)
        except ValueError:
            raise Error1("Path: number of steps must be an integer, "
                         "got {0}".format(num_steps))
        if num_steps < 0:
            raise Error1("Path: number of steps {0} must be "
                         "positive".format(num_steps))

        # --- control
        control = cls.format_path_control(control_hold, leg_num=leg_num)

        # --- get the actual functions
        Cij = []
        for icij, cijfcn in enumerate(cijfcns):
            cijfcn = cijfcn.split(":")
            try:
                fid, scale = cijfcn
            except ValueError:
                fid, scale = cijfcn[0], 1
            try:
                fid = int(float(fid))
            except ValueError:
                raise Error1("Function ID must be an integer, got {0}".format(fid))
            try:
                scale = float(scale)
            except ValueError:
                raise Error1("Function scale must be a float, got "
                             "{0}".format(scale))

            fcn = functions.get(fid)
            if fcn is None:
                raise Error1("{0}: function not defined".format(fid))
            Cij.append((scale, fcn))

        # --- Check lengths of Cij and control are consistent
        if len(Cij) != len(control):
            raise Error1("Path: len(Cij) != len(control) in leg {0}"
                         .format(leg_num))

        path = []
        for time in np.linspace(start_time, final_time, num_steps):
            leg = [time, 1, control]
            leg.append(np.array([s * f(time) for (s, f) in Cij]))
            path.append(leg)

        return path

    @staticmethod
    def format_path_control(cfmt, leg_num=None):
        leg = "" if leg_num is None else "(leg {0})".format(leg_num)
        valid_control_flags = [1, 2, 3, 4, 5, 6, 8, 9]
        control = []
        for (i, flag) in enumerate(cfmt):
            try:
                flag = int(flag)
            except ValueError:
                raise Error1("Path: control flag {0} must be an "
                             "integer, got {1} {2}".format(i+1, flag, leg))

            if flag not in valid_control_flags:
                valid = ", ".join(xmltools.stringify(x)
                                  for x in valid_control_flags)
                raise Error1("Path: {0}: invalid control flag choose from "
                             "{1} {2}".format(flag, valid, leg))

            control.append(flag)

        if 5 in control:
            if any(flag != 5 and flag not in (6, 9) for flag in control):
                raise Error1("Path: mixed mode deformation not allowed with "
                             "deformation gradient control {0}".format(leg))

            # must specify all components
            elif len(control) != 9:
                raise Error1("all 9 components of deformation gradient must "
                             "be specified {0}".format(leg))

        if 8 in control:
            # like deformation gradient control, if displacement is specified
            # for one, it must be for all
            if any(flag != 8 and flag not in (6, 9) for flag in control):
                raise Error1("Path: mixed mode deformation not allowed with "
                             "displacement control {0}".format(leg))

            # must specify all components
            elif len(control) != 3:
                raise Error1("all 3 components of displacement must "
                             "be specified {0}".format(leg))

        return np.array(control, dtype=np.int)


    @staticmethod
    def format_tbl_cols(tblcols):
        columns = []
        for item in [x.split(":")
                     for x in xmltools.str2list(
                             re.sub(r"\s*:\s*", ":", tblcols), dtype=str)]:
            try:
                item = [int(x) for x in item]
            except ValueError:
                raise Error1("Path: tblcols items must be int, got "
                             "{0}".format(tblcols))
            item[0] -= 1

            if len(item) == 1:
                columns.append(item[0])
            elif len(item) not in (2, 3):
                raise Error1("Path: tblcfmt range must be specified as "
                             "start:end:[step], got {0}".format(
                                 ":".join(str(x) for x in item)))
            if len(item) == 2:
                columns.extend(range(item[0], item[1]))
            elif len(item) == 3:
                columns.extend(range(item[0], item[1], item[2]))
        return columns

    @staticmethod
    def format_path(path, options):
        """Format the path by applying multipliers

        """
        # stress control if any of the control types are 3 or 4
        stress_control = any(c in (3, 4) for leg in path for c in leg[2])
        kappa = options.getopt("kappa")
        if stress_control and kappa != 0.:
            raise Error1("kappa must be 0 with stress control option")

        # From these formulas, note that AMPL may be used to increase or
        # decrease the peak strain without changing the strain rate. ratfac is
        # the multiplier on strain rate and stress rate.
        amplitude = options.getopt("amplitude")
        ratfac = options.getopt("ratfac")
        nfac = options.getopt("nfac")
        ndumps = options.getopt("ndumps")
        if ndumps == "all":
            ndumps = 100000000
        ndumps= int(ndumps)

        # factors to be applied to deformation types
        efac = amplitude * options.getopt("estar")
        tfac = abs(amplitude) * options.getopt("tstar") / ratfac
        sfac = amplitude * options.getopt("sstar")
        ffac = amplitude * options.getopt("fstar")
        effac = amplitude * options.getopt("efstar")
        dfac = amplitude * options.getopt("dstar")

        # for now unit tensor for rotation
        Rij = np.reshape(np.eye(3), (9,))

        # format each leg
        for ileg, (termination_time, num_steps, control, Cij) in enumerate(path):

            leg_num = ileg + 1

            num_steps = int(nfac * num_steps)
            termination_time = tfac * termination_time

            # pull out electric field from other deformation specifications
            efcomp = np.zeros(3)
            trtbl = np.array([True] * len(control))
            j = 0
            for i, c in enumerate(control):
                if c == 6:
                    efcomp[j] = effac * Cij[i]
                    trtbl[i] = False
                    j += 1
            Cij = Cij[trtbl]
            control = control[trtbl]

            if 5 in control:
                # check for valid deformation
                defgrad = np.reshape(ffac * Cij, (3, 3))
                jac = np.linalg.det(defgrad)
                if jac <= 0:
                    raise Error1("Inadmissible deformation gradient in "
                                 "leg {0} gave a Jacobian of "
                                 "{1:f}".format(leg_num, jac))

                # convert defgrad to strain E with associated rotation given by
                # axis of rotation x and angle of rotation theta
                Rij, Vij = np.linalg.qr(defgrad)
                if np.max(np.abs(Rij - np.eye(3))) > np.finfo(np.float).eps:
                    raise Error1("Rotation encountered in leg {0}. "
                                 "Rotations are not yet supported".format(leg_num))
                Uij = tensor.asarray(np.dot(Rij.T, np.dot(Vij, Rij)))
                Rij = np.reshape(Rij, (9,))
                Cij = tensor.u2e(Uij, kappa)

                # deformation gradient now converted to strains
                control = np.array([2] * 6, dtype=np.int)

            elif 8 in control:
                # displacement control check
                # convert displacments to strains
                Uij = np.zeros(6)
                Uij[:3] = dfac * Cij[:3] + 1.
                Cij = tensor.u2e(Uij, kappa)

                # displacements now converted to strains
                control = np.array([2] * 6, dtype=np.int)

            elif 2 in control and len(control) == 1:
                # only one strain value given -> volumetric strain
                evol = Cij[0]
                if kappa * evol + 1. < 0.:
                    raise Error1("1 + kappa * ev must be positive in leg "
                                 "{0}".format(leg_num))

                if kappa == 0.:
                    eij = evol / 3.

                else:
                    eij = ((kappa * evol + 1.) ** (1. / 3.) - 1.)
                    eij = eij / kappa

                control = np.array([2] * 6, dtype=np.int)
                Cij = np.array([eij, eij, eij, 0., 0., 0.])

            elif 4 in control and len(control) == 1:
                # only one stress value given -> pressure
                Sij = -Cij[0]
                control = np.array([4, 4, 4, 2, 2, 2], dtype=np.int)
                Cij = np.array([Sij, Sij, Sij, 0., 0., 0.])

            if len(control) != len(Cij):
                raise Error1("len(cij) != len(control) in leg {0}".format(leg_num))

            control = np.append(control, [2] * (6 - len(control)))
            Cij = np.append(Cij, [0.] * (6 - len(Cij)))

            # adjust components based on user input
            for idx, ctype in enumerate(control):
                if ctype in (1, 3):
                    # adjust rates
                    Cij[idx] *= ratfac

                elif ctype == 2:
                    # adjust strain
                    Cij[idx] *= efac

                    if kappa * Cij[idx] + 1. < 0.:
                        raise Error("1 + kappa*E[{0}] must be positive in "
                                    "leg {1}".format(idx, leg_num))

                elif ctype == 4:
                    # adjust stress
                    Cij[idx] *= sfac

                continue

            # initial stress check
            if termination_time == 0.:
                if 3 in control:
                    raise Error1("initial stress rate ambiguous")
                elif 4 in control and any(x != 4 for x in control):
                    raise Error1("Mixed initial state not allowed")

            # Replace leg with modfied values
            path[ileg][0] = termination_time
            path[ileg][1] = num_steps
            path[ileg][2] = control
            path[ileg][3] = Cij
            path[ileg].append(ndumps)

            # legs[ileg].append(Rij)
            path[ileg].append(efcomp)

            continue

        return path

def mybool(a):
    if str(a).lower().strip() in ("false", "no", "0"):
        return 0
    else:
        return 1


def density_range(a):
    try:
        a = xmltools.str2list(a, dtype=float)
    except ValueError:
        fatal_inp_error("{0}: invalid density range".format(a))
        return None

    if len(a) != 2:
        fatal_inp_error("DensityRange must have len == 2")
        a = None

    elif any(r <= 0. for r in a):
        fatal_inp_error("densities in DensityRange must be > 0")
        a = None

    return a
