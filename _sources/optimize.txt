
Optimization
############

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

.. class:: Optimizer(func, xinit, runid, method="simplex", d=None,
                     maxiter=50, tolerance=1.e-6, descriptor=None, funcargs=None)

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

The following input stub demonstrates how to optimize the ``K`` and ``G``
parameters. The ``opt_sig_v_time`` function reads in the simulation output
file and a baseline file and computes the error between the simulation results
and the expected results.

.. code:: python

   import os
   import sys
   import numpy as np
   CCHAR = "#"

   from utils.exojac import ExodusIIFile


   def func(x, *args):

       runid = args[0]
       evald = args[-1]

       name = "{0}.{1}".format(os.path.basename(evald), runid)
       logger = Logger(name)

       # set up driver
       driver = Driver("Continuum", open(path_file, "r").read(), cols=[0,2,3,4],
                       cfmt="222", tfmt="time", path_input="table", logger=logger)

       # set up material
       parameters = {"K": x[0], "G": x[1]}
       material = Material("elastic", parameters, logger=logger)

       # set up and run the model
       mps = MaterialPointSimulator(runid, driver, material, logger=logger)
       mps.run()

       error = opt_sig_v_time(mps.exodus_file)

       return error

   @matmodlab
   def runner(method, v=1):

       runid = "opt_{0}".format(method)

       # run the optimization job.
       # the optimizer expects:
       #    1) A list of OptimizeVariable to optimize
       #    2) An objective function -> a MaterialPointSimulator simulation
       #       that returns some error measure
       #    3) A method
       # It's that simple!

       K = OptimizeVariable("K", 129e9, bounds=(125e9, 150e9))
       G = OptimizeVariable("G", 54e9, bounds=(45e9, 57e9))
       xinit = [K, G]

       optimizer = Optimizer(func, xinit, runid,
                             descriptor=["PRES_V_EVOL"], method=method,
                             maxiter=25, tolerance=1.e-4, verbosity=v,
                             funcargs=[runid])
       optimizer.run()

       return

   def opt_sig_v_time(exof):
       """Find the error in stress vs. time for the current simulation"""
       vars_to_get = ("STRESS_XX", "STRESS_YY", "STRESS_ZZ")

       # read in baseline data
       aux = "opt.base_dat"
       auxhead, auxdat = loadtxt(aux)
       I = np.array([auxhead[var] for var in vars_to_get], dtype=np.int)
       basesig = auxdat[:, I]
       basetime = auxdat[:, auxhead["TIME"]]

       # read in output data
       exof = ExodusIIFile(exof)
       simtime = exof.get_all_times()
       simsig = np.transpose([exof.get_elem_var_time(var, 0)
                              for var in vars_to_get])

       # do the comparison
       error = -1
       t0 = max(np.amin(basetime), np.amin(simtime))
       tf = min(np.amax(basetime), np.amax(simtime))
       n = basetime.shape[0]
       for idx in range(3):
           base = lambda x: np.interp(x, basetime, basesig[:, idx])
           comp = lambda y: np.interp(y, simtime, simsig[:, idx])
           dnom = np.amax(np.abs(simsig[:, idx]))
           if dnom < 1.e-12: dnom = 1.
           rms = np.sqrt(np.mean([((base(t) - comp(t)) / dnom) ** 2
                                  for t in np.linspace(t0, tf, n)]))
           error = max(rms, error)
           continue

       return error


   def loadtxt(filename):
       head = open(filename).readline().strip(CCHAR).split()
       head = dict([(a, i) for (i, a) in enumerate(head)])
       data = np.loadtxt(filename, skiprows=1)
       return head, data

   if __name__ == "__main__":
       runner("cobyla")
