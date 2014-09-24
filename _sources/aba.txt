
Abaqus Materials
################

*matmodlab* can build and exercise Abaqus ``UMAT`` and ``UHYPER`` material
models. *matmodlab* builds the Abaqus models and calls the ``UMAT`` and
``UHYPER`` procedures with the same calling arguments as Abaqus. ``UMAT`` and
``UHYPER`` materials use the same ``Material`` factory method as other
materials, but adds the following additional requirements:

* ``model="umat"`` or ``model="uhyper"``
* ``parameters`` must be a ndarray of model constants (specified in the order
  expected by the model).
* ``constants`` must be specified and the length of ``parameters`` and
  ``constants`` must be the same.
* ``depvar``, if specified, is the number of state dependent variables
  required for the model.
* ``source_files`` [optional] List of model source files.  If not specified, *matmodlab* will look for ``umat.[Ff](90)?`` in the current working directory.
* ``source_directory`` [optional] Directory containing source files.

.. note::
   Only one ``UMAT`` material can be run and exercised at a time.

.. note::
   *matmodlab* modifies the ``parameters`` array to have length ``constants`` + 1 and appends an extra paramter to its end. This extra parameter can be used as a debug flag.

*matmodlab* implements the following Abaqus utility functions:

* ``XIT``.  Stops calculations immediately.
* ``STDB_ABQERR``.  Message passing interface from material model to host code.

Consult the Abaqus documentation for more information.

Examples
========

Two parameter Neo-Hookean nonlinear elastic model implemented as a ``UMAT``.

.. code::

   E = 200
   nu = .333
   material = Material("umat", parameters=[E, nu], constants=2,
                        source_files=["neohooke.f90"], rebuild=test,
                        source_directory="{0}/abaumats".format(MAT_D))

Two parameter Neo-Hookean nonlinear elastic model implemented as a ``UHYPER``.

.. code::

   C10 = 200
   D1 = 1E-05
   material = Material("uhyper", parameters=[C10, D1], constants=2,
                       source_files=["uhyper.f90"],
                       source_directory="{0}/abaumats".format(MAT_D))
