import os
import sys
import time
import numpy as np

from core.runtime import opts
from core.logger import Logger
from utils.exomgr import ExodusII
from core.driver import PathDriver, Driver
from core.product import MAT_LIB_DIRS
from core.material import MaterialModel, Material
from utils.errors import MatModLabError
from utils.data_containers import DataContainer
from utils.variable import Variable, VAR_SCALAR
from utils.constants import DEFAULT_TEMP

class MaterialPointSimulator(object):
    def __init__(self, runid, termination_time=None, verbosity=1, d=None):
        """Initialize the MaterialPointSimulator object

        """
        self.runid = runid
        self._vars = []
        self.termination_time = termination_time
        self.bp = None

        self._material = None
        self._driver = None

        # setup IO
        opts.simulation_dir = d or os.getcwd()
	self.title = "matmodlab single element simulation"
        logfile = os.path.join(opts.simulation_dir, self.runid + ".log")
        logger = Logger(runid, filename=logfile, verbosity=verbosity)
        self.logger = logger

    def register_variable(self, var_name, var_type):
        self._vars.append(Variable(var_name, var_type))

    @property
    def logger(self):
        return self._logger

    @logger.setter
    def logger(self, value):
        self._logger = value

    @property
    def driver(self):
        return self._driver

    @driver.setter
    def driver(self, value):
        if not isinstance(value, PathDriver):
            raise MatModLabError("material must be an instance of PathDriver")
        self._driver = value

    def Driver(self, kind="Continuum", path=None, **kwargs):
        """Method that delays the instantiation of the material model

        """
        def fun(**kwds):
            kwargs["logger"] = self.logger
            return Driver(kind, path, **kwargs)
        self._driver = fun
        self.set_dm()

    def set_dm(self):
        if self.material is not None and self.driver is not None:
            try:
                self.driver = self.driver()
                self.material = self.material(initial_temp=self.driver.initial_temp)
            except TypeError:
                pass

    @property
    def material(self):
        return self._material

    @material.setter
    def material(self, value):
        if not isinstance(value, MaterialModel):
            raise MatModLabError("material must be an instance of MaterialModel")
        self._material = value

    def Material(self, model, parameters, **kwargs):
        """Method that delays the instantiation of the material model

        """
        def fun(**kwds):
            kwargs["logger"] = self.logger
            kwargs.update(**kwds)
            return Material(model, parameters, **kwargs)
        self._material = fun
        self.set_dm()

    @property
    def variables(self):
	return self._vars

    def write_summary(self):
        s = "\n   ".join("{0}".format(x) for x in MAT_LIB_DIRS)
        summary = """
Simulation Summary
---------- -------
Runid: {0}
Material search directories:
   {6}
Material interface file:
   {7}
Driver: {1}
  Number of legs: {2}
  Total number of steps: {8}
Material: {3}
  Number of props: {4}
    Number of sdv: {5}
""".format(self.runid, self.driver.kind, self.driver.num_leg,
           self.material.name, self.material.num_prop, self.material.num_xtra,
           s, self.material.file, self.driver.num_steps)
        self.logger.write(summary)

    def _setup(self):
        """Last items to set up before running

        """
        # set up the driver and material
        if not self.driver: raise MatModLabError("no driver assigned")
        if not self.material: raise MatModLabError("no material assigned")
        self.set_dm()

        if abs(self.driver.initial_temp - self.material.initial_temp) > 1.e-12:
            raise MatModLabError("driver initial temperature != "
                                 "material initial temperature")

        # Exodus database setup
        self.exo_db = ExodusII(self.runid, d=opts.simulation_dir)
        self.exo_file = self.exo_db.filepath

	# register global variables
        self.register_variable("TIME_STEP", VAR_SCALAR)
        self.register_variable("STEP_NUM", VAR_SCALAR)
        self.register_variable("LEG_NUM", VAR_SCALAR)

        # two data arrays: global, element
        glob_data = []
        self.glob_vars = []
        for d in self.variables:
            assert len(d.keys) == len(d.initial_value)
            glob_data.append((d.name, d.keys, d.initial_value))
            self.glob_vars.extend(d.keys)
        self.glob_data = DataContainer(glob_data)

        elem_data = []
        self.elem_vars = []
        for item in (self.driver.variables, self.material.variables):
            for d in item:
                assert len(d.keys) == len(d.initial_value)
                self.elem_vars.extend(d.keys)
                elem_data.append((d.name, d.keys, d.initial_value))
        self.elem_data = DataContainer(elem_data)

        # set up timing
        self.timing = {}
        self.timing["start"] = time.time()

        self.setup_io()
        self.write_summary()
        pass

    def setup_io(self):

        self.exo_db.put_init(self.glob_data.data, self.glob_vars,
                             self.elem_data.data, self.elem_vars, title=self.title)

        # Write info to log file
        L = max(max(len(n) for n in self.elem_vars), 10)
        param_names = self.material.parameter_names
        self.param_vals = np.array(self.material.parameters)
        iparam_vals = self.material.initial_parameters
        param_vals = self.material.parameters

        self.logger.debug("Material Parameters")
        self.logger.debug("  {1:{0}s}  {2:12}  {3:12}".format(
            L, "Name", "iValue", "Value"))
        for p in zip(param_names, iparam_vals, param_vals):
            self.logger.debug("  {1:{0}s} {2: 12.6E} {3: 12.6E}".format(L, *p))

        # write out plotable data
        self.logger.debug("Output Variables:")
        self.logger.debug("Global")
        for item in self.glob_vars:
            self.logger.debug("  " + item)
        self.logger.debug("Element")
        for item in self.elem_vars:
            self.logger.debug("  " + item)

    def run(self):
        """Run the problem

        """
        self._setup()
        self.check_break_points()
        self.logger.write("Starting calculations...")
        retcode = self.driver.run(self.glob_data, self.elem_data,
                                  self.material, self.exo_db, self.bp,
                                  termination_time=self.termination_time)
        self.finish()
        return retcode

    def finish(self):
        # udpate and close the file
        self.timing["end"] = time.time()
        if self.driver.ran == "broke":
            self.logger.write("Calculations terminated at break point")
        elif self.driver.ran:
            dt_run = self.timing["end"] - self.timing["start"]
            self.logger.write("...calculations completed ({0:.4f}s)".format(dt_run))
        else:
            self.logger.error("Calculations did not complete")
        self.exo_db.finish()

        if opts.viz_on_completion:
            self.visualize_results()

        return

    def dump(self, variables, format="ascii", step=1, time=True, ffmt=".18f"):
        from utils.exojac.exodump import exodump
        exodump(self.exo_file, step=step, ffmt=ffmt,
                variables=variables, ofmt=format, time=time)

    def extract_from_db(self, variables, step=1, t=0):
        from utils.exojac.exodump import read_vars_from_exofile
        data = read_vars_from_exofile(self.exo_file, variables=variables,
                                      step=step, h=0, t=t)
        return data

    def visualize_results(self, overlay=None):
        from core.plotter import create_model_plot
        create_model_plot(self.exo_file)

    @property
    def exodus_file(self):
        return self.exo_file

    def break_point(self, condition):
        """Define a break point for the simulation

        """
        self.bp = BreakPoint(condition, self)

    def check_break_points(self):
        if not self.bp:
            return
        for name in self.bp.names:
            if name.upper() not in ["TIME"] + self.glob_vars + self.elem_vars:
                self.logger.error("break point variable {0} not a "
                                  "simulation variable".format(name))
        if self.logger.errors:
            raise MatModLabError("stopping due to previous errors")

def cprint(string):
    sys.stderr.write(string + "\n")

class BreakPointError(Exception):
    def __init__(self, c):
        super(BreakPointError, self).__init__("{0}: not a valid conditon".format(c))

class BreakPoint:
    def __init__(self, condition, mps):
        self._condition = condition
        self.mps = mps
        self.condition, self.names = self.parse_condition(condition)
        self.first = 1

    @staticmethod
    def parse_condition(condition):
        """Parse the original condition given in the input file

        """
        from StringIO import StringIO
        from tokenize import generate_tokens
        from token import NUMBER, OP, NAME, ENDMARKER
        wrap = lambda x: "{{{0}}}".format(x).upper()
        s = StringIO(condition)
        g = [x for x in generate_tokens(s.readline)]
        stack = []
        names = []
        expr = []
        for toknum, tokval, _, _, _ in g:
            if len(expr) == 3:
                stack.append(" ".join(expr))
                expr = []
            if toknum == ENDMARKER:
                break
            if not expr:
                if toknum != NAME:
                    raise BreakPointError(condition)
                if tokval in ("and", "or"):
                    stack.append(tokval)
                    continue
                names.append(tokval)
                expr.append(wrap(tokval))
            elif len(expr) == 1:
                if toknum != OP:
                    raise BreakPointError(condition)
                if tokval not in ("==", ">", ">=", "<", "<="):
                    raise BreakPointError(condition)
                expr.append(tokval)
            elif len(expr) == 2:
                if toknum not in (NUMBER, NAME):
                    raise BreakPointError(condition)
                if toknum == NAME:
                    names.append(tokval)
                    tokval = wrap(tokval)
                expr.append(tokval)
        condition = " ".join(stack)
        return condition, names

    def eval(self, time, glob_data, elem_data):
        """Evaluate the break condition

        """
        if not self.condition:
            return
        kwds = {"TIME": time}
        kwds.update(glob_data.todict())
        kwds.update(elem_data.todict())
        condition = self.condition.format(**kwds)
        if not eval(condition):
            return

        # Break condition met.  Enter the UI
        self.ui(condition, time, glob_data, elem_data)

        return

    def generate_summary(self, time, glob_data, elem_data):
        params = zip(self.mps.material.parameter_names, self.mps.material.params)
        params = "\n".join("  {0} = {1}".format(a, b) for a, b in params)
        summary = """
SUMMARY OF PARAMETERS
{0}

SUMMARY OF GLOBAL DATA
  TIME : {1:.4f}
{2}

SUMMARY OF ELEMENT DATA
{3}

""".format(params, time, glob_data.summary("  "), elem_data.summary("  "))
        return summary

    def ui(self, condition, time, glob_data, elem_data):
        self.summary = self.generate_summary(time, glob_data, elem_data)

        if self.first:
            cprint(self.manpage(condition, time))
            self.first = 0
        else:
            cprint("BREAK CONDITION {0} ({1}) "
                   "MET AT TIME={2}".format(self._condition, condition, time))

        while 1:
            resp = raw_input("mml > ").lower().split()

            if not resp:
                continue

            if resp[0] == "c":
                self.condition = None
                return

            if resp[0] == "h":
                cprint(self.manpage(condition, time))
                continue

            if resp[0] == "s":
                return

            if resp[0] == "set":
                try:
                    name, value = resp[1:]
                except ValueError:
                    cprint("***error: must specify 'set name value'")
                    continue
                key = name.upper()
                value = eval(value)
                if key in glob_data:
                    glob_data[key] = value
                elif key in elem_data:
                    elem_data[key] = value
                elif key in self.mps.material.parameter_names:
                    idx = self.mps.material.parameter_names.index(key)
                    self.mps.material.params[idx] = value
                else:
                    cprint("  {0}: not valid variable/parameter".format(item))
                    continue
                continue

            if resp[0] == "p":
                toprint = resp[1:]
                if not toprint:
                    cprint(self.summary)
                    continue

                for item in toprint:
                    if item.upper() == "TIME":
                        name = "TIME"
                        value = time
                    elif item[:5] == "param":
                        name = "PARAMETERS"
                        value = self.mps.material.parameters
                    elif item[:4] in ("xtra", "stat"):
                        name = "XTRA"
                        value = elem_data["XTRA"]
                    elif item in glob_data:
                        name = item.upper()
                        value = glob_data[name]
                    elif item in elem_data:
                        name = item.upper()
                        value = elem_data[name]
                    elif item.upper() in self.mps.material.parameter_names:
                        name = item.upper()
                        idx = self.mps.material.parameter_names.index(name)
                        value = self.mps.material.params[idx]
                    else:
                        cprint("  {0}: not valid variable".format(item))
                        continue
                    cprint("  {0} = {1}".format(name, value))
                continue

            if resp[0] == "q":
                self.mps.driver.ran = "broke"
                self.mps.finish()
                sys.exit(0)

            else:
                cprint("{0}: unrecognized command".format(" ".join(resp)))

    def manpage(self, condition, time):
        page = """

BREAK CONDITION {0} ({1}) MET AT TIME={2}

SYNOPSIS OF COMMANDS
    c
      Continue the analysis until completion [break condition removed].

    h
      Print the help message

    p [name_1[ name_2[ ...[name_n]]]]
      Print simulation information to the screen.  If the optional name
      is given, the current value of that variable and/or parameter is
      printed.

    s
      Step through the analysis, reevaluating the break point at each step.

    set <name> <value> [type]
      Set the variable or parameter to the new value.
      In the case that both a variable and parameter have the same name
      specify type=v for variable or type=p for parameter [default].
      Note, this could have unintended consequences on the rest of the
      simulation

    q
      Quit the analysis gracefully.

EXAMPLES
    o Setting the value of the bulk modulus K for the remainder of the analysis

      set K 100
      c

    """.format(self._condition, condition, time)
        return page
