Input Syntax Guidelines
#######################

.. topic:: References

   * :ref:`model_create_and_execute`
   * :ref:`mml_out_dbs`

Overview
========

This section describes the conventions used in the Matmodlab API.

matmodlab Namespace
===================

Input scripts to Matmodlab should include::

   from matmodlab import *

to populate the script's namespace with Matmodlab specific parameters,
classes, and symbolic constants.

Parameters
----------

The parameters exposed by importing ``matmodlab`` are

* ``ROOT_D``, the root ``matmodlab`` directory
* ``PKG_D``, the ``matmodlab/lib`` directory, the location shared objects are copied
* ``MAT_D``, the directory where builtin materials are contained

Classes
-------

The classes exposed by importing ``matmodlab`` are

* ``MaterialPointSimulator``, the Matmodlab material point simulator
* ``Permutator``, the Matmodlab permutator
* ``Optimizer``, the Matmodlab optimizer

Each of these classes is described in more detail in the following sections.

Symbolic Constants
------------------

The following are symbolic constants that are exposed by importing ``matmodlab``.

Symbolic Constants Relating to MaterialPointSimulator
.....................................................

* ``DBX, EXO, TXT, PKL, XLS, XLSX``, constants representing the output file formats.  See :ref:`mml_out_dbs`.

Symbolic Constants Relating to User Defined Materials
.....................................................

* ``XX, YY, ZZ, XY, YZ, XZ``, constants representing the *xx*, *yy*, *zz*, *xy*, *yz*, and *xz* components of second-order symmetric tensors.
* ``MECHANICAL``, ``HYPERELASTIC``, ``ANISOHYPER`` are constants representing user defined materials for mechanical, hyperelastic, and anisotropic-hyperelastic material behaviors (see :ref:`user_mats`).
* ``WLF`` specifies a WLF time-temperature shift (see :ref:`trs`)
* ``PRONY`` specifies a prony series input to the viscoelastic model (see :ref:`viscoelastic`)
* ``ISOTROPIC`` specifies isotropic thermal expansion (see :ref:`expansion`)
* ``USER`` specifies that a user developed mode is of type "user"
* ``UMAT`` specifies that a user developed mode is of type "umat"
* ``UHYPER`` specifies that a user developed mode is of type "uhyper"
* ``UANISOHYPER_INV`` specifies that a user developed mode is of type "uanisohyper_inv"

Naming Conventions
==================

Throughout Matmodlab, the following naming conventions are adopted (see the `PEP8 <https://www.python.org/dev/peps/pep-0008>`_ guidelines for more guidance):

* Class names use the CapWords convention.
* Method names use lowercase with words separated by underscores as necessary to improve readability.
* Variable names adopt the rul as method names.
* Symbolic constants are written in all capital letters with underscores separating words.
* Factory methods adopt the same rule as class names.
