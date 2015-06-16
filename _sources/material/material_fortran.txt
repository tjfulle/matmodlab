
Fortran User Material Interface
###############################

Matmodlab can build and exercise Abaqus ``UMAT``, ``UHYPER``, and
``UANISOHYPER_INV`` user material models. Matmodlab builds the Abaqus models
and calls them with the same calling arguments as Abaqus. Abaqus user material
models use the same ``Material`` factory method as other materials, but requires the following:

* ``model`` argument must be one of ``umat``, ``uhyper``, or
  ``uanisohyper_inv``.
* ``parameters`` must be a ndarray of model constants (specified in the order
  expected by the model).
* ``depvar`` [optional], is the number of state dependent variables required
  for the model. Can also be specified as a list of state dependent variable
  names, specified in the order expected by the model. If given as a list, the
  number of state variables allocated is inferred from its length. Matmodlab
  allocates storage for the ``depvar`` state dependent variables and
  initializes their values to 0.
* ``source_files`` [optional] is a list of model source files. If not
  specified, Matmodlab will look for ``umat.[Ff](90)?`` in the current
  working directory.
* ``source_directory`` [optional] is a directory containing source files.
* ``param_names`` [optional] is a list of parameter names. If given,
  ``parameters`` must be given as dict of ``param_name:value`` pairs (just as
  in the native models).
* ``cmname`` [optional] is the constitutive model name. Passed to the user
  procedure as in Abaqus

.. note::
   Only one ``UMAT`` material can be run and exercised at a time.

Abaqus Utility Procedures
=========================

Matmodlab implements the following Abaqus utility procedures:

* ``XIT``.  Stops calculations immediately.
* ``STDB_ABQERR``.  Message passing interface from material model to host code.

Consult the Abaqus documentation for more information.

Examples
========

Example UMAT User Subroutines
-----------------------------

See ``matmodlab/materials/abaumats`` for some sample ``umat`` materials.

Example Material Inputs
-----------------------

Two parameter Neo-Hookean nonlinear elastic model implemented as a ``UMAT``.

.. code::

   E = 200
   nu = .333
   material = Material("umat", [E, nu],
                        source_files=["neohooke.f90"],
                        source_directory="{0}/abaumats".format(MAT_D))

Two parameter Neo-Hookean nonlinear elastic model implemented as a ``UHYPER``.

.. code::

   C10 = 200
   D1 = 1E-05
   material = Material("uhyper", [C10, D1],
                       source_files=["uhyper.f90"],
                       source_directory="{0}/abaumats".format(MAT_D))

Two parameter Neo-Hookean nonlinear elastic model implemented as a ``UHYPER``, specifying the ``param_names``

.. code::

   C10 = 200
   D1 = 1E-05
   param_names = ["C10", "D1"]
   paramters = dict(zip(param_names, [C10, D1]))
   material = Material("uhyper", parameters,
                       source_files=["uhyper.f90"],
                       source_directory="{0}/abaumats".format(MAT_D))
