import os
import sys
import math
import numpy as np

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
