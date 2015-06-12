.. _Building:

Building*
#########

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

Building is performed by the ``mml build`` command::

  usage: mml build [-h] [-v V] [-w] [-W] [-m M [M ...]] [-u]

  mml build: build fortran utilities and materials.

  optional arguments:
    -h, --help    show this help message and exit
    -v V          Verbosity [default: 1]
    -w            Wipe before building [default: False]
    -W            Wipe and exit [default: False]
    -m M [M ...]  Materials to build [default: all]
    -u            Build auxiliary support files only [default: False]

Example
.......

::

  mml build

This will build the *matmodlab* Fortran utilities and material libraries. The
resultant shared object libraries are copied to ``matmodlab/lib``.

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
