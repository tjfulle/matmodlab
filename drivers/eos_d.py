import os
import re
import sys
import math
import numpy as np
from itertools import product

from __config__ import cfg
import utils.tensor as tensor
import utils.xmltools as xmltools
from drivers.driver import Driver
from core.kinematics import deps2d, sig2d, update_deformation
from utils.tensor import NSYMM, NTENS, NVEC, I9
from utils.opthold import OptionHolder
from core.io import fatal_inp_error, input_errors, log_message, log_error
from materials.material import create_material

np.set_printoptions(precision=4)

K_RTSPC = 0

class EOSDriver(Driver):
    name = "eos"
    def __init__(self):
        super(EOSDriver, self).__init__()
        pass

    def setup(self, runid, material, *opts):
        """Setup the driver object

        """
        self.runid = runid
        self.mtlmdl = create_material(material[0])

        self.register_glob_variable("TIME_STEP")
        self.register_glob_variable("STEP_NUM")
        self.register_variable("DEFGRAD", vtype="TENS")
        self.register_variable("RHO", vtype="SCALAR",
                               units="DENSITY_UNITS")
        self.register_variable("TMPR", vtype="SCALAR",
                               units="TEMPERATURE_UNITS")
        self.register_variable("ENRGY", vtype="SCALAR",
                               units="SPECIFIC_ENERGY_UNITS")
        self.register_variable("PRES", vtype="SCALAR",
                               units="PRESSURE_UNITS")
        self.register_variable("CS", vtype="SCALAR",
                               units="VELOCITY_UNITS")
        self.register_variable("DPDR", vtype="SCALAR")
        self.register_variable("DPDT", vtype="SCALAR")
        self.register_variable("DEDT", vtype="SCALAR")
        self.register_variable("DEDR", vtype="SCALAR")

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

        # get initial pressure, energy, and set initial state
        self._data[self.defgrad_slice] = I9
        rho = self.surfaces[K_RTSPC][0, 0]
        tmpr = self.surfaces[K_RTSPC][1, 0]
        pres, e, cs, s = self.mtlmdl.update_state(rho, tmpr)
        self.setvars(rho=rho, tmpr=tmpr, enrgy=e, pres=pres, cs=cs,
                     dpdr=s[0], dpdt=s[1], dedt=s[2], dedr=s[3])

        return


    def process_paths_and_surfaces(self, iomgr, *args):
        if self.surfaces is None:
            io.log_message("eos: no surfaces to process")

        K2eV = 8.617343e-5
        erg2joule = 1.0e-4

        # This will make sure that everything has units set.
        #eos_model.ensure_all_parameters_have_valid_units()
        #simdat.ensure_valid_units()
        #matdat.ensure_valid_units()

        #input_unit_system = the_model.boundary.input_units()
        #output_unit_system = the_model.boundary.output_units()

        #if not UnitManager.is_valid_unit_system(input_unit_system):
        #    pu.report_and_raise_error(
        #        "Input unit system '{0}' is not a valid unit system"
        #        .format(input_unit_system))
        #if not UnitManager.is_valid_unit_system(output_unit_system):
        #    pu.report_and_raise_error(
        #        "Output unit system '{0}' is not a valid unit system"
        #        .format(output_unit_system))
        t = 0.
        nprints = 10
        step_num = 0
        surface = self.surfaces[K_RTSPC]
        num_steps = len(surface[0]) * len(surface[1])
        dt = 1. / surface.shape[0] / surface.shape[1]
        for rho in surface[0]:
            for tmpr in surface[1]:
                step_num += 1
                if step_num == 1:
                    # initial state used to initialize output file
                    continue
                t += dt
                p, e, cs, s = self.mtlmdl.update_state(rho, tmpr)
                F = rho / surface[0, 0] * I9
                self.setvars(rho=rho, tmpr=tmpr, enrgy=e, pres=p, cs=cs,
                             dpdr=s[0], dpdt=s[1], dedt=s[2], dedr=s[3], defgrad=F)
                self.setglobvars(step_num=step_num, time_step=dt)
                iomgr(t)
                if step_num % int(num_steps / float(nprints)) == 0:
                    log_message("surface step {0}/{1}".format(step_num, num_steps))
            self._paths_and_surfaces_processed = True
        return

    # --------------------------------------------------------- Parsing methods
    @classmethod
    def parse_and_register_paths_and_surfaces(cls, pathlmns, surflmns, functions,
                                              tterm):
        """Parse the Path elements of the input file and register the formatted
        paths to the class

        """
        if pathlmns:
            fatal_inp_error("EOSDriver: <Path> elements not directly supported. "
                            "State paths are extracted from generated surfaces "
                            "in a <Extract><Path type=.../> element")

        if len(surflmns) > 1:
            fatal_inp_error("EOSDriver: expected at most one surface")
            return
        if not surflmns:
            return
        cls.surfaces = {K_RTSPC: cls.pSurface(surflmns[0], functions)}
        return

    @classmethod
    def pSurface(cls, surflmn, functions):
        """Parse the Path block and set defaults

        """
        # Set up options for Path
        options = OptionHolder()
        options.addopt("amplitude", 1.)
        options.addopt("rstar", 1., test=lambda x: x > 0.)
        options.addopt("tstar", 1., test=lambda x: x > 0.)
        options.addopt("nfac", 1., test=lambda x: x > 0.)
        options.addopt("href", None, dtype=str)
        options.addopt("format", "default", dtype=str,
                       choices=("default", "table"))

        # the following options are for table formatted Path
        options.addopt("cols", "1:2", dtype=str)
        options.addopt("cfmt", "12", dtype=str)

        # Get control terms
        for i in range(surflmn.attributes.length):
            options.setopt(*xmltools.get_name_value(surflmn.attributes.item(i)))

        # Read in the actual Path - splitting them in to lists
        href = options.getopt("href")
        if href:
            if not os.path.isfile(href):
                if not os.path.isfile(os.path.join(cfg.I, href)):
                    fatal_inp_error("{0}: no such file".format(href))
                    return
                href = os.path.join(cfg.I, href)
            lines = open(href, "r").readlines()
        else:
            lines = []
            for node in surflmn.childNodes:
                if node.nodeType == node.COMMENT_NODE:
                    continue
                lines.extend([" ".join(xmltools.uni2str(item).split())
                              for item in node.nodeValue.splitlines()
                              if item.split()])
        lines = [xmltools.str2list(line, dtype=str) for line in lines]

        # parse the Surface depending on type
        pformat = options.getopt("format")
        if pformat == "default":
            surface = cls.parse_surf_default(lines)

        elif pformat == "table":
            surface = cls.parse_surf_table(lines,
                                           options.getopt("cols"),
                                           options.getopt("cfmt"))

        else:
            fatal_inp_error("Path: {0}: invalid format".format(pformat))
            return

        if input_errors():
            return

        return cls.format_surface(surface, options)

    @classmethod
    def parse_surf_table(cls, lines, tblcols, tblcfmt):
        """Parse the path table

        """
        surf = []
        leg_num = 1

        # Convert tblcols to a list
        columns = cls.format_tbl_cols(tblcols)

        # check the control
        control = cls.format_surf_control(tblcfmt)

        for line in lines:
            if not line:
                continue
            try:
                line = np.array([float(x) for x in line])
            except ValueError:
                raise Error1("Expected floats in leg {0}, got {1}".format(
                    leg_num, line))
            try:
                Cij = line[columns]
            except IndexError:
                fatal_inp_error("Requested column not found in leg "
                                "{0}".format(leg_num))
                continue

            # --- number of steps
            num_steps = 1

            # --- Check lengths of Cij and control are consistent
            if len(Cij) != len(control):
                fatal_inp_error("Path: len(Cij) != len(control) in leg {0}"
                                .format(leg_num))
                continue

            surf.append([num_steps, control, Cij])
            leg_num += 1
            continue

        return surf

    @staticmethod
    def format_surf_control(cfmt, leg_num=None):
        leg = "" if leg_num is None else "(leg {0})".format(leg_num)
        valid_control_flags = [1, 2]
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
        return np.array(control, dtype=np.int)

    @staticmethod
    def format_surface(surf, options):
        """Format the surface by applying multipliers

        """
        amplitude = options.getopt("amplitude")
        nfac = options.getopt("nfac")

        # factors to be applied to deformation types
        rfac = amplitude * options.getopt("rstar")
        tfac = amplitude * options.getopt("tstar")

        # format each leg
        ndindex = lambda a, i: np.where(a == i)[0][0]
        for i, (num_steps, control, Cij) in enumerate(surf):
            num_steps = int(nfac * num_steps)
            rho = Cij[ndindex(control, 1)]
            tmpr = Cij[ndindex(control, 2)]
            surf[i] = [num_steps, rho, tmpr]
            continue
        rho, tmpr = [], []
        for isurf, (n, rhof, tmprf) in enumerate(surf[1:]):
            (_, rhoi, tmpri) = surf[isurf]
            rho.extend(np.linspace(rhoi, rhof, n).tolist())
            tmpr.extend(np.linspace(tmpri, tmprf, n).tolist())
        return np.array([rho, tmpr])

    @classmethod
    def parse_surf_default(cls, lines):
        """Parse the individual path

        """
        surf = []
        leg_num = 1
        for line in lines:
            if not line:
                continue
            num_steps, control_hold = line[:2]
            Cij_hold = line[2:]

            # --- number of steps
            num_steps = format_num_steps(leg_num, num_steps)
            if num_steps is None:
                num_steps = 10000

            # --- control
            control = cls.format_surf_control(control_hold, leg_num=leg_num)

            # --- Cij
            Cij = []
            for (i, comp) in enumerate(Cij_hold):
                try:
                    comp = float(comp)
                except ValueError:
                    fatal_inp_error("Path: Component {0} of leg {1} must be a "
                                    "float, got {2}".format(i+1, leg_num, comp))
                Cij.append(comp)

            Cij = np.array(Cij)

            # --- Check lengths of Cij and control are consistent
            if len(Cij) != len(control):
                fatal_inp_error("Path: len(Cij) != len(control) in leg {0}"
                                .format(leg_num))
                continue


            surf.append([num_steps, control, Cij])
            leg_num += 1
            continue

        return surf

    @staticmethod
    def format_tbl_cols(cols):
        columns = []
        for item in [x.split(":")
                     for x in xmltools.str2list(
                             re.sub(r"\s*:\s*", ":", cols), dtype=str)]:
            try:
                item = [int(x) for x in item]
            except ValueError:
                fatal_inp_error("Path: expected integer cols, got "
                                "{0}".format(cols))
                continue
            item[0] -= 1

            if len(item) == 1:
                columns.append(item[0])

            elif len(item) not in (2, 3):
                fatal_inp_error("Path: expected cfmt range to be specified as "
                                "start:end:[step], got {0}".format(
                                    ":".join(str(x) for x in item)))
                continue

            if len(item) == 2:
                columns.extend(range(item[0], item[1]))

            elif len(item) == 3:
                columns.extend(range(item[0], item[1], item[2]))

        return columns

    def extract_paths(self, exofilepath, pathlmns):
        """From the data in the exodus file path, extract requested information

        """
        if not self._paths_and_surfaces_processed:
            return

        from utils.exodump import read_vars_from_exofile
        from utils.srfdump import extract_isotherm, extract_hugoniot
        # Set up options for Path
        options = OptionHolder()
        options.addopt("type", None, dtype=str, choices=("hugoniot", "isotherm"))
        options.addopt("increments", 100, dtype=int, test=lambda x: x > 0.)
        options.addopt("density_range", None, dtype=str)
        options.addopt("initial_temperature", None,
                       dtype=float, test=lambda x: x > 0.)

        variables=["RHO", "TMPR", "ENRGY", "PRES", "DPDR", "DPDT", "DEDT", "DEDR"]
        surf = read_vars_from_exofile(exofilepath, variables=variables, h=0)[:, 1:]

        # Get control terms
        for pathlmn in pathlmns:
            for i in range(pathlmn.attributes.length):
                options.setopt(*xmltools.get_name_value(pathlmn.attributes.item(i)))

            n = options.getopt("increments")
            r = density_range(options.getopt("density_range"), n)
            t = options.getopt("initial_temperature")

            if options.getopt("type") == "hugoniot":
                log_message("extracting Hugoniot from EOS surface")
                ep = extract_hugoniot(r, t, surf)
                if ep is None:
                    log_error("unable to extract Hugoniot form EOS surface", r=0)
                    continue
                log_message("Hugoniot extracted from EOS surface")
                ehug, phug, thug = ep

            elif options.getopt("type") == "isotherm":
                log_message("extracting Isotherm from EOS surface")
                ep = extract_isotherm(r, t, surf)
                if ep is None:
                    log_error("unable to extract Isotherm form EOS surface", r=0)
                    continue
                log_message("Isotherm extracted from EOS surface")
                eiso, piso = ep

        pass


def density_range(a, n):
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

    return np.linspace(a[0], a[1], n)


def temperature_range(a, n):
    try:
        a = xmltools.str2list(a, dtype=float)
    except ValueError:
        fatal_inp_error("{0}: invalid temperature range".format(a))
        return None

    if len(a) != 2:
        fatal_inp_error("TemperatureRange must have len == 2")
        a = None

    elif any(r <= 0. for r in a):
        fatal_inp_error("temperatures in TemperatureRange must be > 0")
        a = None

    return np.linspace(a[0], a[1], n)


def format_num_steps(leg_num, num_steps):
    try:
        num_steps = int(num_steps)
    except ValueError:
        fatal_inp_error("Path: expected integer number of steps in "
                        "leg {0} got {1}".format(leg_num, num_steps))
        return
    if num_steps < 0:
        fatal_inp_error("Path: expected positive integer number of "
                        "steps in leg {0} got {1}".format(
                            leg_num, num_steps))
        return

    return num_steps


def inrange(a, b):
    """Test if the array a is in the range of b

    """
    if isinstance(a, (int, float)):
        mina = maxa = a
    else:
        mina = np.amin(a)
        maxa = np.amax(a)
    return mina >= np.amin(b) and maxa <= np.amax(b)
