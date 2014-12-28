
Quick Start Guide
#################


Setup and Build, Method 1
=========================

Build *matmodlab*.  See :ref:`Building` for more details.

* Download *matmodlab*
* Put ``MML_ROOT/bin`` on ``PATH``.
* [Optional] Build fortran components::

   cd path/to/matmodlab
   mml build


Setup and Build, Method 2
=========================

If only the python functionality of *matmodlab* is desired, no building is
required.

* Download *matmodlab*

* When using *matmodlab* simply ensure that your working directory is
  ``MML_ROOT`` and run *matmodlab* by invoking the regular Python interpreter::

    python mml_input.py

* Notice: without adding ``MML_ROOT/bin`` to the ``PATH`` the regular ``mml``
  command will not be available.

Prepare Input
=============

Inputs are Python scripts. See :ref:`Running Simulations` and :ref:`Factory
Methods` for more details.

* Define the driver
* Define the material
* Define simulator

Run
===

Run the input script with ``mml``.  See :ref:`Running Simulations` for more
details::

  mml run [options] runid.py

``mml`` will create the following files::

  ls runid.*
  runid.exo            runid.log            runid.py

``runid.exo`` is the ExodusII output database and ``runid.log`` the simulation
log file. See ``mml help run`` for a complete list of options

Postprocess
===========

View results in *matmodlab* viewer.  See :ref:`Postprocessing` for more details::

  mml view filename.exo [filename_2.exo [... filename_n.exo]]
