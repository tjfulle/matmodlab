
Quick Start Guide
#################

The steps for running a simulation in *matmodlab* are

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


*matmodlab* Developers
----------------------

See :ref:`Building` for more details.

* Clone *matmodlab* from `<https://github.com/tjfulle/matmodlab>`_::

   git clone https://github.com/tjfulle/matmodlab.git

* Navigate to the ``matmodlab`` directory and execute::

   python setup.py develop

All components of *matmodlab* will be built and installed.

.. _prepare_input:

Prepare Input
=============

Input files are Python scripts. See :ref:`Annotated Examples` and :ref:`Methods for Creating Simulations` for more details.

* Instantiate a ``MaterialPointSimulator`` object.
* Define the material model
* Define the deformation path

.. _run_the_input:

Run the Simulation
==================

Run the input script with ``mml``.  See :ref:`Annotated Examples` for more
details::

  mml run [options] filename.py

``mml`` will create the following files::

  ls filename.*
  filename.exo            filename.log            filename.py

``filename.exo`` is the ExodusII output database and ``filename.log`` the simulation
log file. See ``mml help run`` for a complete list of options

.. _post_proc_results:

Postprocess
===========

View results in *matmodlab* viewer.  See :ref:`Postprocessing` for more details::

  mml view filename.exo
