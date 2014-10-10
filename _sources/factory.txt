.. _Factory Methods:

Factory Methods
===============

The Material Factory Method
---------------------------

A material instance is created through the ``Material`` factory method.  Minimally, the factory method requires the material model name ``model`` and the model parameters ``parameters``

.. code:: python

   material = Material(mat_name, parameters)

The object returned from ``Material`` is an instance of the class defining
``model``.

The formal parameters to ``Material`` are

.. function:: Material(model, parameters, initial_temp=None, logger=None, expansion=None, trs=None, viscoelastic=None, rebuild=False, source_files=None, source_directory=None, depvar=None, param_names=None, switch=None)

   Factory method for subclasses of MaterialModel

   :param model: material model name
   :type model: str
   :param parameters: model parameters.  For Abaqus umat models and matmodlab user models, parameters is a ndarray of model constants (specified in the order expected by the model).  For other model types, parameters is a dictionary of name:value pairs.
   :type parameters: dict or ndarray
   :param initial_temp: Initial temperature.  The initial temperature, if given, must be consistent with that of the simulation driver.  Defaults to 298K if not specified.
   :type initial_temp: float or None
   :param logger: An instance of a Logger
   :type logger: instance or None
   :param expansion: An instance of an Expansion model.
   :type expansion: instance or None
   :param trs: An instance of a time-temperature shift (TRS) model
   :type trs: instance or None
   :param viscoelastic: An instance of a Viscoelastic model.
   :type viscoelastic: instance or None
   :param rebuild: Rebuild the material, or not.
   :type rebuild: bool
   :param source_files: List of model source files*.  Each file name given in source_files must exist.  If the optional source_directory is given, source files are looked for in it.
   :type source_files: list or None
   :param source_directory: Directory containing source files*.  source_directory is optional, but allows giving source_files as a list of file names only - not fully qualified paths.
   :type source_directory: str or None
   :param depvar: Number of state dependent variables*.
   :type depvar: int or None
   :param param_names: List of model parameter names*.  If specified, parameters are given as dict and not ndarray.
   :type param_names: list or None
   :param switch: Model switch.  Two tuple given as (original material, new material).
   :type switch: tuple or None
   :rtype: MaterialModel instance

The Driver Factory Method
-------------------------

A driver instance is created through the ``Driver`` factory method. Minimally,
the factory method requires the driver kind ``driver_kind`` and the path
specification ``path``

.. code:: python

   driver = Driver(driver_kind, path)

The object returned from ``Driver`` is an instance of the class defining
``driver_kind``.  At present, only ``driver_kind="Continuum"`` is defined.

The formal parameters to ``Driver`` are

.. function:: Driver(driver_kind, path, path_input="default", kappa=0., amplitude=1., rate_multiplier=1., step_multiplier=1., num_io_dumps="all", estar=1., tstar=1., sstar=1., fstar=1., efstar=1., dstar=1., proportional=False, termination_time=None, functions=None, cfmt=None, tfmt="time", num_steps=None, cols=None, skiprows=0, logger=None)

   Factory method for subclasses of PathDriver

   :param driver_kind: The driver kind
   :type driver_kind: str
   :param path: The deformation path through which to driver the material
   :type path: str
   :param path_input: Type of path input.  Choices are default, table, function [default: default]
   :type path_input: str or None
   :param kappa: The Seth-Hill parameter [default: 0.]
   :type kappa: float
   :param amplitude: Factor multiplied to all components of deformation [default: 1.]
   :type amplitude: float
   :param rate_multiplier: Divisor to the termination time of each leg, thereby effectively increasing the rate of deformation [default: 1.]
   :type rate_multiplier: float
   :param step_multiplier: Multiplier on number of steps in each leg [default: 1.]
   :type step_multiplier: float
   :param num_io_dumps: Total number of dumps to the output database [default: all]
   :param [de(ef)fst]star: Multipliers on the components of displacement, strain, electric field, deformation gradient, stress, and temperature, respectively. The [de(ef)fst]star} are first multiplied by amplitude [default: 1.].
   :type [de(ef)fst]star: float
   :param proportional: For stress controlled loading, attempt to maintain proportional loading when seeking strain increments [default: False]
   :type proportional: bool
   :param termination_time: Termination time.  If not specified, termination time is taken as last time in path.
   :type termination_time: float or None
   :param functions: List of Function objects.  Functions used to generate path.
   :type functions: List of Function
   :param cfmt: Column format if path_input is table or function.
   :type cfmt: str or None
   :param tfmt: Time format if path_input is table or function [default: time]
   :type tfmt: str or None
   :param num_steps: Total number of steps if path_input is function [default: 1]
   :type num_steps: int
   :param cols: Columns from which to extract data if path_input is table.
   :type cols: List of int
   :param skiprows: Rows to skip when reading path_file or table data [default=0]
   :type skiprows: int or None
   :param logger: An instance of a Logger
   :type logger: instance or None
   :rtype: PathDriver instance

Defining the Path
~~~~~~~~~~~~~~~~~

The path through which a material is driven is defined by deformation "legs"
specifying the type of deformation to be prescribed over the time period of
each leg. The method in which the path is defined is dependent on the value of
the ``path_input`` parameter.

``path_input="default"``
........................

For ``path_input="default"``, each leg of deformation is given as::

    tf n cfmt Cij

where ``tf``, ``n``, ``cfmt``, and ``Cij`` are the termination time, number of
steps, control format, and control format of the particular leg. The control
format ``cfmt`` is concatenated integer list specifying in its
:math:`i^\text{ith}` component the :math:`i^\text{th}` component of
deformation, i.e., ``cfmt[i]`` instructs the driver as to the type of
deformation represented by ``Cij[i]``. Consider the :ref:`First Example`,
where the path was prescribed as::

   path = """0  0 222  0 0 0
             1 10 222 .1 0 0"""

or, shown below with parts explicitly labeled

.. figure:: ./images/path_desc.png
   :align: center
   :width: 3in

In this example, for the second leg, ``tf=1``, the number of steps is
``n=10``, ``cfmt=222``, and ``Cij=.1 0 0``

Consider now how ``cfmt`` corresponds to ``Cij``

.. figure:: ./images/cfmt_desc.png
   :align: center
   :width: 2in

Types of deformation represented by ``cfmt`` are shown in `Table 1`_

.. _Table 1:

+----------+----------------------+
| ``cfmt`` | Deformation type     |
+==========+======================+
|     1    | Strain rate          |
+----------+----------------------+
|     2    | Strain               |
+----------+----------------------+
|     3    | Stress rate          |
+----------+----------------------+
|     4    | Stress               |
+----------+----------------------+
|     5    | Deformation gradient |
+----------+----------------------+
|     6    | Electric field       |
+----------+----------------------+
|     7    | Temperature          |
+----------+----------------------+
|     8    | Displacement         |
+----------+----------------------+
|     9    | User defined field   |
+----------+----------------------+

The component ordering of vectors and tensors follows what is described in
:ref:`Conventions`. If ``len(Cij)`` does not equal 6, (or 9 for deformation
gradient), the missing components are assumed to be zero strain.

If temperature is not prescribed, it is presumed to have a constant value of 298K.

If a user defined field is not prescribed, it is presumed to be ``None``.

For example, the following ``cfmt`` instructs the driver that the components
of ``Cij`` represent [stress, strain, stress rate, strain rate, strain,
strain], respectively::

  cfmt="423122"

Mixed modes are allowed only for components of strain rate, strain, stress
rate, and stress.

Electric field components can be included with any deformation type.

Temperature can be included with any deformation type.

User defined field can be included with any deformation type.

If only one component of stress rate, stress, strain rate, or strain is
specified, the component ``Cij`` is taken to be either the pressure or
volumetric strain.

.. _tblform:

``path_input="table"``
......................

The ``table`` ``path_input`` format allows reading in deformation paths from a
columnar table of data. Control format is uniform for all legs and is
specified by the ``cfmt`` keyword argument to ``Driver``. Specify which
columns to read data with the ``cols`` keyword argument. Column indexing is
zero based and the first column is assumed to be the time specifier. The
``tfmt`` keyword argument specifies if the time column represents the actual
time (``tfmt="time"``) or time step (``tmft="dt"``). The number of steps for
each leg can be set by ``num_steps`` keyword argument.

The following input stubs sets up the driver with the same path as in the
:ref:`First Example`, but specified by a table::

   path = """0  0 0 0
             1 .1 0 0"""
   driver = Driver("Continuum", path, path_input="table",
                   cols=[0,1,2,3], cfmt="222", tfmt="time, num_steps=10)

The table input format is convenent for using experimental data, contained in
columnar ascii data files, to drive a material model.

``path_input="function"``
.........................

The ``function`` ``path_input`` format allows defining a deformation path by a
function. A deformation path defined by ``function`` must have only 1 leg
defining the termination time and the function specifier defining the values
of the components of deformation. The function specifier is of the form::

   function_id[:scale]

where ``function_id`` is the ID of a ``Function`` object. The optional scale
is a scalar multiplier applied to the return value of the function identified
with ``function_id``.  See :ref:`Functions` for more information on defining ``Function`` objects.

The following input stub demonstrates uniaxial strain deformation, using a
user defined function to specify the 11 component of strain through time

.. code:: python

   # set up the driver with a function
   func = Function(2, "analytic_expression", lambda t: np.sin(t))
   functions = [func,]
   path = "{0} 2:1.e-1 0 0".format(2*pi)
   driver = Driver("Continuum", path, path_input="function",
                   num_steps=200, termination_time=1.8*pi,
                   functions=functions, cfmt="222")

.. _mps:

The Material Point Simulator
----------------------------

A ``MaterialPointSimulator`` instance is created directly through the
``MaterialPointSimulator`` constructor. Minimally, the
``MaterialPointSimulator`` constructor requires the specification of a
``runid``, ``driver``, ``material``

.. code:: python

   mps = MaterialPointSimulator(runid, driver, material)

The formal parameters to the ``MaterialPointSimulator`` constructor are

.. class:: MaterialPointSimulator(self, runid, driver, material, termination_time=None, verbosity=1, d=None, logger=None)

   MaterialPointSimulator constructor.  Creates a MaterialPointSimulator object and sets up the simulation

   :param runid: The simulation runid.  All simulation output will be named runid.ext, where ext is log, exo, etc.
   :type runid: str
   :param driver: The driver object with which to drive the simulation
   :type driver: PathDriver
   :param material: The material model object
   :type material: MaterialModel
   :param termination_time: Simulation termination time.  If not given, the last time in the driver path will be used [default: None].
   :type termination_time: float or None
   :param verbosity: Level of verbosity.  0->quiet, 3->noisy [default: 1]
   :type verbosity: int
   :param d: Directory to run simulation [default: PWD]
   :type d: str or None
   :param logger: Logger object to log simulation process.  If not specified, a new logger will be created [default: None].
   :type logger: Logger or None

Public Methods of ``MaterialPointSimulator``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. method:: MaterialPointSimulator.run()

   Run the simulation

.. method:: MaterialPointSimulator.dump(self, variables, format="ascii", step=1, time=True, ffmt=".18f")

   Dump variables from ExodusII database to other ascii formats

   :param variables: Variables to dump
   :type variables: list of str
   :param format: Output format.  Must be one of ascii, mathematica, ndarray [default: ascii].
   :type format: str
   :param step: Step interval to dump data [default: 1].
   :type step: int
   :param time: Dump time [default: True].
   :type time: bool
   :param ffmt: Floating point number format.  Used as "{0:{1}}".format(number, ffmt)
   :type ffmt: str

.. method:: MaterialPointSimulator.extract_from_db(variables, step=1, t=0)

   Extract variables from ExodusII database.

   :param variables: Variables to extract
   :type variables: list of str
   :param step: Step interval to dump data [default: 1].
   :type step: int
   :param time: Extract time [default: 0].
   :type time: int

.. method:: MaterialPointSimulator.visualize_results(overlay=None)

   Display simulation results in visualizer.

   :param overlay: Filename for which data is to be overlayed on top of simulation data.
   :type overlay: str or None


.. _Functions:

The Function Factory Method
---------------------------

A function instance is created through the ``Function`` factory method. Minimally,
the factory method requires a unique function ID ``func_id``, function type ``func_type``, and function expression ``expr``.

.. code:: python

   func = Function(func_id, func_type, func_expr)

The object returned from ``Function`` is an instance of the class defining
``func_type``. At present, ``analytical expression`` and ``piecewise linear``
function types are supported.  When evaluated, functions are called as::

   Function(t)

where ``t`` is the time of interest in the simulation.

The formal parameters to ``Function`` are

.. function:: Function(func_id, func_type, func_defn)

   Build a function object with which a Path path can be created.

   :parameter func_id: Unique integer ID to identify the function.  IDs 0 and 1 are reserved for the constant 0 and 1 functions.
   :type function_id: int
   :parameter func_type: The type of function.  One of "analytic expression" or "piecewise linear".  Analytic expression should be a function with 1 callable argument.  If piecewise linear, the current value are interpolated through time.
   :type function_id: int
   :parameter func_defn: The function definition.  If func_type is analytic expression, a function of a single argument.  If func_type is piecewise linear, a 2 column table; the first column represents time and the second the values to interpolate through time.
   :type function_id: callable or str


The Logger
----------

Logging in *matmodlab* is through the ``Logger`` class.

It is useful to setup and pass the same logger to ``Material``, ``Driver``, and ``MaterialPointSimulator``.  A logger instance is created through the ``Logger`` constructor.  The ``Logger`` constructor requires no arguments to setup

.. code:: python

   logger = Logger()

The formal parameters to ``Logger`` are

.. class:: Logger(logfile=None, verbosity=1)

   The matmodlab logger.  Logs messages and warnings to the console and/or file

   :param logfile: File name of log file.  If not given, messages are only logged to the console [default: None].
   :type logfile: str or None
   :param verbosity: Verbosity.  If verbosity < 1, then messages are only logged to file [default: 1].
   :type verbosity: int

Logger Methods
~~~~~~~~~~~~~~

.. method:: Logger.write(message)

   Log a message

   :parameter message: The message to log
   :type message: str

.. method:: Logger.warn(message)

   Log a warning message

   :parameter message: The warning message to log
   :type message: str

.. method:: Logger.error(message)

   Log an error message

   :parameter message: The error message to log.  The simulation will not stop.
   :type message: str

.. method:: Logger.raise_error(message)

   Log an error message and raise an exception

   :parameter message: The error message to log. An exception will be raised.
   :type message: str
