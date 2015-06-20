.. _inpperm:

Permutator
##########

.. todo::

   clean up the description to be more clear what func is (a function that
   runs a simulation)

It is useful to investiage sensitivities of models to model inputs.
Permutating values of model inputs involves running jobs with different
realization of parameters. Ideal for investigating model sensitivities. A
permutator instance is created through the ``Permutator`` constructor.
Minimally, the constructor requires a function to evaluate ``func``, initial
parameters ``xinit``, and a ``job``.

.. code:: python

   permutator = Permutator(func, xinit, job)

``func`` is called as ``func(x, *args)`` where ``x`` are the current values of
the permutated variables and ``args`` contains in its last component::

   dirname = args[-1]

where ``dirname`` is the directory where simulations should is to be run.

Permutator Constructor
======================

The formal parameters to ``Permutator`` are

.. class:: Permutator(func, xinit, job, method=ZIP, correlations=False, verbosity=1, descriptor=None, nprocs=1, funcargs=None, d=None)

   Create a Permutator object

   :parameter func: Function that evaluates a matmodlab simulation.  Must have signature ``func(x, *args)``, where x are the current values of the permutated variable and funcargs are described below.
   :type callable:
   :parameter xinit: Initial values of simulation parameters.
   :type xinit: List of PermutateVariable objects
   :parameter job: job for the parent Permutation process.
   :type job: str
   :parameter method: The method for determining how to combine parameter values. One of ``ZIP`` or ``COMBINATION``. The ``ZIP`` method runs one job for each set of parameters (and, thus, the number of realizations for each parameter must be identical), the ``COMBINATION`` method runs every combination of parameters.
   :type method: symbolic constant
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

Each ``Permutator`` job creates a directory ``job.eval`` with the following
contents::

   ls job.eval
   eval_000/    eval_002/    mml-evaldb.xml
   eval_001/    ...          job.log

The ``eval_...`` directory holds the output of the ith job and a ``params.in``
file with the values of each permutated parameter for that job.
``mml-evaldb.xml`` contains a summary of each job run. ``mml view``
recognizes ``mml-evaldb.xml`` files.

Run a permutation job by invoking the ``Permutator.run()`` method.

PermutateVariable Factory Method
================================

The formal parameters to ``PermutateVariable`` are

.. function:: PermutateVariable(name, init, method=LIST, b=None, N=10)

   Create a PermutateVariable object

   :parameter name: Name of variable
   :type name: str
   :parameter init: Initial value or values, dependending on method.
   :type init: float or list
   :parameter method: Method used to generate all values.  If ``LIST``, than all values shall be given in init.  Otherwise, values will be generated. Valid methods are ``LIST, WEIBULL, UNIFORM, NORMAL, PERCENTAGE``.
   :type method: symbolic constant
   :parameter b: For methods other than ``LIST``, values are generated from a function called as fun(init, b, N).  The meaning of b is dependent on which method fun represents.
   :type b: float
   :parameter N: For methods other than ``LIST``, the number of values to generate
   :type N: int

Examples
--------

The following input stub demonstrates how to permutate the ``K`` parameter

.. code:: python

   K = PermutateVariable("K", [75, 125, 155])

.. code:: python

   K = PermutateVariable("K", 125, method=WEIBULL, b=14)

.. code:: python

   K = PermutateVariable("K", 125, method=PERCENTAGE, b=10, N=10)

Example
=======

The following input demonstrates how to permutate the ``K`` and ``G``
parameters to the ``elastic`` model.  The input can be found in ``matmodlab/examples/permutate.py``.

.. code:: python

  from matmodlab import *

  def func(x, xnames, d, job, *args):

      mps = MaterialPointSimulator(job)
      mps.StrainStep(components=(1, 0, 0), increment=1., scale=-.5, frames=10)
      mps.StrainStep(components=(2, 0, 0), increment=1., scale=-.5, frames=10)
      mps.StrainStep(components=(1, 0, 0), increment=1., scale=-.5, frames=10)
      mps.StrainStep(components=(0, 0, 0), increment=1., scale=-.5, frames=10)

      # set up the material
      parameters = dict(zip(xnames, x))
      mps.Material('elastic', parameters)

      # set up and run the model
      mps.run()

      s = mps.get('STRESS_XX')
      return np.amax(s)

  def runjob():
      N = 15
      K = PermutateVariable('K', 125e9, method=WEIBULL, b=14, N=N)
      G = PermutateVariable('G', 45e9, method=PERCENTAGE, b=10, N=N)
      xinit = [K, G]
      permutator = Permutator('permutation', func, xinit, method=ZIP,
                              descriptors=['MAX_PRES'], correlations=True)
      permutator.run()

  runjob()
