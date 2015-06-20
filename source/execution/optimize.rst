:tocdepth: 2

Optimizer
#########

Optimize specified parameters against user specified objective function. Ideal
for finding optimal model parameters. A optimizer instance is created through
the ``Optimizer`` constructor. Minimally, the constructor requires a function
to evaluate ``func``, initial parameters ``xinit``, and a ``runid``.

.. code:: python

   optimizer = Optimizer(func, xinit, runid)

``func`` is called as ``func(x, *args)`` where ``x`` are the current values of
the permutated variables and ``args`` contains in its last component::

   dirname = args[-1]

where ``dirname`` is the directory where simulations should is to be run.

Optimizer Constructor
=====================

The formal parameters to ``Optimizer`` are

.. class:: Optimizer(func, xinit, runid, method="simplex", d=None, maxiter=50, tolerance=1.e-6, descriptor=None, funcargs=None)

   Create a Optimizer object

   :parameter func: Function that evaluates a matmodlab simulation.  Must have signature ``func(x, *args)``, where x are the current values of the permutated variable and funcargs are described below.
   :type callable:
   :parameter xinit: Initial values of simulation parameters.
   :type xinit: List of PermutateVariable objects
   :parameter method: The optimization method. One of simplex, powell, cobyla.
   :type method: str
   :parameter d: Parent directory to run jobs.  If the directory does not exist, it will be created.  If not given, the current directory will be used.
   :type d: str or None
   :parameter maxiter: Maximum number of iterations
   :type maxiter: int
   :parameter tolerance: The tolerance.
   :type tolerance: float
   :parameter descriptor: Descriptors of return values from func
   :type descriptor: list of str or None
   :parameter funcargs: Additional arguments to be sent to func.  The directory of the current job is appended to the end of funcargs.  If None,
   :type funcargs: list or None

Each ``Optimzer`` job creates a directory ``runid.eval`` with the following
contents::

   ls runid.eval
   eval_000/    eval_002/    mml-evaldb.xml
   eval_001/    ...          runid.log

The ``eval_...`` directory holds the output of the ith job and a ``params.in``
file with the values of each parameter to optimize for that job.
``mml-evaldb.xml`` contains a summary of each job run. ``mml view`` recognizes
``mml-evaldb.xml`` files.

OptimizeVariable Factory Method
===============================

The formal parameters to ``OptimizeVariable`` are

.. function:: OptimizeVariable(name, initial_value, bounds=None)

   Create a OptimizeVariable object

   :parameter name: Name of variable
   :type name: str
   :parameter initial_value: Initial value or values, dependending on method.
   :type init: float or list
   :parameter bounds: Bounds on the variable.  If given, (lower_bound, upper_bound)
   :type b: tuple of None

Examples
--------

The following input stub demonstrates how to permutate the ``K`` parameter

.. code:: python

   K = OptimizeVariable("K", 75)

.. code:: python

   K = OptimizeVariable("K", 125, bounds=(100, 150))

Example
=======

The following input demonstrates how to optimize the ``K`` and ``G``
parameters and can be found in ``matmodlab/examples/optimize.py``.  The objective function calls ``calculate_bounded_area`` to find the area between the calculated stress strain curve and the experimental.

.. code:: python

  import os
  import numpy as np

  from matmodlab import *
  import matmodlab.utils.fileio as ufio
  import matmodlab.utils.numerix.nonmonotonic as unnm

  filename = os.path.join(get_my_directory(), "optimize.xls")
  strain_exp, stress_exp = zip(*ufio.loadfile(filename, sheet="MML", disp=0,
                                              columns=["STRAIN_XX", "STRESS_XX"]))

  def func(x=[], xnames=[], evald="", runid="", *args):
      mps = MaterialPointSimulator(runid)

      xp = dict(zip(xnames, x))
      NU = 0.32  # poisson's ratio for aluminum
      parameters = {"K": xp["E"]/3.0/(1.0-2.0*NU), "G": xp["E"]/2.0/(1.0+NU),
                    "Y0": xp["Y0"], "H": xp["H"], "BETA": 0.0}
      mps.Material("vonmises", parameters)

      # create steps from data. note, len(columns) below is < len(descriptors).
      # The missing columns are filled with zeros -> giving uniaxial stress in
      # this case. Declaring the steps this way does require loading the excel
      # file anew for each run
      mps.DataSteps(filename, steps=30, sheet='MML',
                    columns=('STRAIN_XX',), descriptors='ESS')

      mps.run()
      if not mps.ran:
          return 1.0e9

      strain_sim, stress_sim = zip(*mps.get("STRAIN_XX", "STRESS_XX"))
      error = unnm.calculate_bounded_area(strain_exp, stress_exp,
                                        strain_sim, stress_sim)
      return error

  def runjob(method, v=1):
      E = OptimizeVariable("E",  2.0e6, bounds=(1.0e5, 1.0e7))
      Y0= OptimizeVariable("Y0", 0.3e5, bounds=(1.0e4, 1.0e6))
      H = OptimizeVariable("H",  1.0e6, bounds=(1.0e4, 1.0e7))
      xinit = [E, Y0, H]

      optimizer = Optimizer("optimize", func, xinit, method=method,
                          maxiter=200, tolerance=1.e-3)
      optimizer.run()
      xopt = optimizer.xopt
      return xopt

  runjob('powell')
