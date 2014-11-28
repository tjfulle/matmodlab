.. _Running Simulations:

.. _First Example:

Running a Simulation
====================

Simulations are run by processing *matmodlab* input scripts with the ``mml``
command line utility. A *matmodlab* input script consists of defining
(minimally) a ``driver``, ``material``, and ``simulator`` through the
``Driver``, ``Material``, and ``MaterialPointSimulator`` factory methods. By
including

.. code::

   from matmodlab import *

each of these factory methods is exposed to the simulation script's namespace.
Consider, for example, the following input, contained in ``runid.py``, in
which a material whose constitutive response is defined by the ``elastic``
material model, parameterized by the bulk modulus :math:`K` and shear modulus
:math:`G`, that is driven through a path of uniaxial strain

.. code::

   from matmodlab import *

   runid = "elastic_uniaxial_strain"

   # setup the material
   parameters = {"K": 135E+09, "G": 54E+09}
   material = Material("elastic", parameters)

   # set up the deformation path and driver
   path = """0  0 222  0 0 0
             1 10 222 .1 0 0"""
   driver = Driver("Continuum", path)

   # set up the simulator and run
   mps = MaterialPointSimulator(runid, driver, material)
   mps.run()

To automatically launch the visualization utility add the following line::

  mps.visualize_results()

Executing::

  mml run runid.py

will exercise the material for the deformation path defined.  The following files will be created by ``mml``::

  ls runid*

  runid.exo         runid.log         runid.py

In this guide, the ``Material``, ``Driver``, and ``MaterialPointSimulator``, and other, factory methods will be described in more detail.
