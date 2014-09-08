Building
########

The material model laboratory (matmodlab) is an object oriented model driver.
The majority of the source code is written in Python and requires no
additional building. Many of the material models, however, are written in
Fortran and require a seperate compile step. The compiling step is handled
dynamically at the time of material instantiation.


System Requirements
===================

*matmodlab* has been built and tested extensively on several versions of linux
and the Apple Mac OSX 10.8 operating systems.


Required Software
=================

*matmodlab* requires the following software installed for your platform:

#) `Python 2.7 <http://www.python.org/>`_ or newer

#) `NumPy 1.8 <http://www.numpy.org/>`_ or newer

#) `SciPy 0.13 <http://www.scipy.org/>`_ or newer

*matmodlab* is developed and tested using the `Enthought <http://www.enthought.com/>`_ *Python distribution* and `Anaconda <http://continuum.io/>`_.

Optional Software
=================

#) Fortran compiler


.. _installation:

Installation
============

#) Make sure that all *matmodlab* prerequisites are installed and working properly.

#) Add ``matmodlab/bin`` to your ``PATH`` environment variable

#) [Optional] Build the fortran utilities.  While ``matmodlab`` is written in python, many core linear algebra utilities and several built in material models have been implemented in python.  To build them run::

	$ mml build

Executables
===========

The user interface is provided by the ``bin/mml`` executable script

Testing the Installation
========================

To test *matmodlab* after installation, execute::

	$ mml test -k fast

which will run the "fast" tests found in matmodlab/tests

To run the full test suite execute::

	$ mml test [-jN]

Please note that running all of the tests takes several minutes.


Troubleshooting
===============

If you experience problems when building/installing/testing *matmodlab*, you
can ask help from `Tim Fuller <timothy.fuller@utah.edu>`_. Please include the
following information in your message:
