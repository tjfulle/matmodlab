.. _Running Simulations:

.. _First Example:

Running a Simulation
====================

Simulations are run by processing *matmodlab* input scripts with the ``mml``
command line utility. A *matmodlab* input script consists of defining
an instance of the ``MaterialPointSimulator`` class and defininig for it
a ``driver`` and ``material``.  By including

.. code::

   from matmodlab import *

the ``MaterialPointSimulator`` and other necessary methods are exposed to the
simulation script's namespace. Consider, for example, the following input,
contained in ``runid.py``, in which a material's constitutive response is
defined by the ``elastic`` material model, parameterized by the bulk modulus
:math:`K` and shear modulus :math:`G`, and is driven through a path of
uniaxial strain

.. code::

   from matmodlab import *

   runid = "elastic_uniaxial_strain"

   # set up the simulator
   mps = MaterialPointSimulator(runid)

   # setup the material
   parameters = {"K": 135E+09, "G": 54E+09}
   mps.Material("elastic", parameters)

   # set up the deformation path and driver
   path = """0  0 222  0 0 0
             1 10 222 .1 0 0"""
   mps.Driver("Continuum", path)

   # run the simulation
   mps.run()

Executing::

  mml run runid.py

will exercise the material for the deformation path defined.  The following files will be created by ``mml``::

  ls runid*

  runid.exo         runid.log         runid.py

In this guide, the ``MaterialPointSimulator`` and its associated ``Material``
and ``Driver`` methods will be described in more detail.
