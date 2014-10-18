.. _inpperm:

Permutation
###########

.. todo::

   clean up the description to be more clear what func is (a function that
   runs a simulation)

It is useful to investiage sensitivities of models to model inputs.
Permutating values of model inputs involves running jobs with different
realization of parameters. Ideal for investigating model sensitivities. A
permutator instance is created through the ``Permutator`` constructor.
Minimally, the constructor requires a function to evaluate ``func``, initial
parameters ``xinit``, and a ``runid``.

.. code:: python

   permutator = Permutator(func, xinit, runid)

``func`` is called as ``func(x, *args)`` where ``x`` are the current values of
the permutated variables and ``args`` contains in its last component::

   dirname = args[-1]

where ``dirname`` is the directory where simulations should is to be run.

Permutator Constructor
======================

The formal parameters to ``Permutator`` are

.. class:: Permutator(func, xinit, runid, method="zip", correlations=False, verbosity=1, descriptor=None, nprocs=1, funcargs=None, d=None)

   Create a Permutator object

   :parameter func: Function that evaluates a matmodlab simulation.  Must have signature ``func(x, *args)``, where x are the current values of the permutated variable and funcargs are described below.
   :type callable:
   :parameter xinit: Initial values of simulation parameters.
   :type xinit: List of PermutateVariable objects
   :parameter runid: runid for the parent Permutation process.
   :type runid: str
   :parameter method: The method for determining how to combine parameter values. One of zip or combine. The zip method runs one job for each set of parameters (and, thus, the number of realizations for each parameter must be identical), the combine method runs every combination of parameters.
   :type method: str
   :parameter correlations: Create correlation table and plots of relating permutated parameters and return value of func [default: False].
   :type correlations: bool
   :parameter descriptor: Descriptors of return values from func
   :type descriptor: list of str or None
   :parameter nprocs: Number of simultaneous jobs [default: None]
   :type descriptor: int of None
   :parameter funcargs: Additional arguments to be sent to func.  The directory of the current job is appended to the end of funcargs.  If None,
   :type funcargs: list or None
   :parameter d: Parent directory to run jobs.  If the directory does not exist, it will be created.  If not given, the current directory will be used.
   :type d: str or None

Each ``Permutator`` job creates a directory ``runid.eval`` with the following
contents::

   ls runid.eval
   eval_000/    eval_002/    mml-evaldb.xml
   eval_001/    ...          runid.log

The ``eval_...`` directory holds the output of the ith job and a ``params.in``
file with the values of each permutated parameter for that job.
``mml-evaldb.xml`` contains a summary of each job run. ``mml view``
recognizes ``mml-evaldb.xml`` files.

PermutateVariable Factory Method
================================

The formal parameters to ``PermutateVariable`` are

.. function:: PermutateVariable(name, init, method="list", b=None, N=10)

   Create a PermutateVariable object

   :parameter name: Name of variable
   :type name: str
   :parameter init: Initial value or values, dependending on method.
   :type init: float or list
   :parameter method: Method used to generate all values.  If list, than all values shall be given in init.  Otherwise, values will be generated. Valid methods are list, weibull, uniform, normal, percentage.
   :type method: str
   :parameter b: For methods other than list, values are generated from a function called as fun(init, b, N).  The meaning of b is dependent on which method fun represents.
   :type b: float
   :parameter N: For methods other than list, the number of values to generate
   :type N: int

Examples
--------

The following input stub demonstrates how to permutate the ``K`` parameter

.. code:: python

   K = PermutateVariable("K", [75, 125, 155])

.. code:: python

   K = PermutateVariable("K", 125, method="weibull", b=14)

.. code:: python

   K = PermutateVariable("K", 125, method="percentage", b=10, N=10)

Example
=======

The following input stub demonstrates how to permutate the ``K`` and ``G``
parameters

.. code:: python

   from matmodlab import *

   def func(x, *args):

       path = """
       0 0 222222 0 0 0 0 0 0
       1 1 222222 1 0 0 0 0 0
       2 1 222222 2 0 0 0 0 0
       3 1 222222 1 0 0 0 0 0
       4 1 222222 0 0 0 0 0 0
       """
       d, runid = args[:2]
       logfile = os.path.join(d, runid + ".log")
       logger = Logger(logfile=logfile, verbosity=0)

       # set up the driver
       driver = Driver("Continuum", path=path, step_multiplier=1000,
                       logger=logger, estar=-.5)

       # set up the material
       parameters = {"K": x[0], "G": x[1]}
       material = Material("elastic", parameters=parameters, logger=logger)

       # set up and run the model
       mps = MaterialPointSimulator(runid, driver, material, logger=logger, d=d)
       mps.run()
       pres = mps.extract_from_db(["PRESSURE"])
       return np.amax(pres)

   @matmodlab
   def runner():
       method = "zip"
       d = os.getcwd()
       runid = "perm_{0}".format(method)
       K = PermutateVariable("K", 125e9, method="weibull", b=14, N=3)
       G = PermutateVariable("G", 45e9, method="percentage", b=10, N=3)
       xinit = [K, G]
       permutator = Permutator(func, xinit, runid, descriptor=["MAX_PRES"],
                               method=method, correlations=True, d=d, verbosity=v,
                               funcargs=[runid])
       permutator.run()

   if __name__ == "__main__":
       runner()
