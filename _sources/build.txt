.. _Building:

Building
########

*matmodlab*'s code base is largely written in Python and requires no
additional compiling. However, several [optional] linear algebra packages and
material models are written in Fortran and require a seperate compile step.


System Requirements
===================

*matmodlab* has been built and tested extensively on several versions of Linux
and the Apple Mac OSX 10.9 operating systems. It is unknown whether or not
*matmodlab* will run on Windows.


Required Software
=================

*matmodlab* requires the following software installed for your platform:

#) `Python 2.7 <http://www.python.org/>`_ or newer

#) `NumPy 1.6 <http://www.numpy.org/>`_ or newer

#) `SciPy 0.1 <http://www.scipy.org/>`_ or newer

The required software may be obtained in several ways, though all development
has been made using the Annoconda `<http://www.continuum.io>`_ and Enthought
Canopy `<http://www.enthought.com>`_ Python Distributions.

.. _installation:

Installation
============

Ensure that all *matmodlab* prerequisites are installed and working properly
before proceeding.

Set Environment and Path
------------------------

Add ``matmodlab/bin`` to your ``PATH``.

Build
-----

::

   cd path/to/matmodlab
   mml build


This will build the *matmodlab* Fortran utilities and material libraries. The
resultant shared object libraries are copied to ``matmodlab/lib``.

.. _Config:

Optional Configuration
======================

*matmodlab* must be configured to be made aware of materials and tests that
reside outside of *matmodlab*. Use ``mml config`` to add directories to find
these materials and tests::

  mml config --add materials path/to/material
  mml config --add tests path/to/tests

On completion of the preceding commands, *matmodlab* will treat materials and
tests in ``path/to/material`` and ``path/to/tests``, respectively, as built in
materials and tests.

See ``mml help config`` for more configuration options.


Testing the Installation
========================

To test *matmodlab* after installation, execute::

	mml test -k fast

which will run the "fast" "regression" tests. To run the full test suite execute::

	mml test -j4

Please note that running all of the tests takes several minutes.

Troubleshooting
===============

If you experience problems when building/installing/testing *matmodlab*, you can
ask help from `Tim Fuller <timothy.fuller@utah.edu>`_.
