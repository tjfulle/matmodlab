
Quick Start Guide
#################

Setup and Build
===============

Build *matmodlab*.  See :ref:`Building` for more details.

* Download *matmodlab*
* Put ``MML_ROOT/bin`` on ``PATH``.
* [Optional] Build fortran components::

   cd path/to/matmodlab
   mml build

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
