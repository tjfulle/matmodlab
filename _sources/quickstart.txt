
Quick Start Guide
#################

The steps for running a simulation in Matmodlab are

* :ref:`setup_and_build`
* :ref:`prepare_input`
* :ref:`run_the_input`
* :ref:`post_proc_results`

.. _setup_and_build:

Setup and Build
===============

*The following commands are to be executed from a command prompt*

Users
-----

Simply execute

::

  pip install matmodlab

or

::

  easy_install matmodlab


Matmodlab Developers
----------------------

See :ref:`Installing` for more details.

* Clone Matmodlab from `<https://github.com/tjfulle/matmodlab>`_::

   git clone https://github.com/tjfulle/matmodlab.git

* Navigate to the ``matmodlab`` directory and execute::

   python setup.py develop

All components of Matmodlab will be built and installed.

.. _prepare_input:

Prepare Input
=============

Input files are Python scripts. See :ref:`examples` and :ref:`Methods for Creating Simulations` for more details.

* Instantiate a ``MaterialPointSimulator`` object.
* Define the material model
* Define the deformation path

.. _run_the_input:

Run the Simulation
==================

Run the input script with ``mml``.  See :ref:`examples` for more
details::

  mml run [options] filename.py

``mml`` will create the following files::

  ls filename.*
  filename.dbx            filename.log            filename.py

``filename.dbx`` is the dbx output database and ``filename.log`` the simulation
log file. See ``mml help run`` for a complete list of options

.. _post_proc_results:

Postprocess
===========

View results in Matmodlab viewer.  See :ref:`Postprocessing` for more details::

  mml view filename.dbx
