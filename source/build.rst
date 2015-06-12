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

The basic functionality of *matmodlab* requires the following software installed
for your platform:

#) `Python 2.7 <http://www.python.org/>`_ or newer (A, E)

#) `NumPy 1.6 <http://www.numpy.org/>`_ or newer (A, E)

#) `SciPy 0.1 <http://www.scipy.org/>`_ or newer (A, E)


*matmodlab* has further functionality that can be utilized if the appropriate
packages are installed.

#) `traits <http://pypi.python.org/pypi/traits>`_, `traitsui <http://pypi.python.org/pypi/traitsui>`_, and `chaco <http://pypi.python.org/pypi/chaco>`_ for data visualization (E)

#) `pytest <http://pypi.python.org/pypi/pytest>`_ for running built-in benchmarks (A, E)

#) `openpyxl <http://pypi.python.org/pypi/openpyxl>`_ for simulation output and visualization of .xlsx data (A, E)

#) `xlwt <http://pypi.python.org/pypi/xlwt>`_ for simulation output as .xls (A)

#) `xlrd <http://pypi.python.org/pypi/xlrd>`_ for visualization of .xls data (A)


The required software may be obtained in several ways, though most development
has been made using the Anaconda `<http://www.continuum.io>`_ and Enthought
Canopy `<http://www.enthought.com>`_ Python Distributions (E=available in
base Enthought Canopy distribution, A=available in base Anaconda distribution).
It is also possible to get all of the required packages through a linux
distribution's package manager or, for all installations of python, by running

::

  easy_install xlrd

or

::

  pip install chaco

or, for Anaconda,

::

  conda install traitsui

and so on for each required package.

If using a linux distribution's version of python, you might need to install the
python development headers in order to build fortran models or use the faster
linear algebra package.

.. _installation:

Installation
============

Ensure that all *matmodlab* prerequisites are installed and working properly
before proceeding.


The Easy Way
------------

Because *matmodlab* is in `PyPI <http://pypi.python.org/pypi/matmodlab>`_, you
can simply run

::

  easy_install matmodlab

or

::

  pip install matmodlab

and you're done! Note: you may have to have administrative privileges to
install. If you don't know the difference between ``pip`` and ``easy_install``,
try to use ``pip`` first.


The Manual Way
--------------

After downloading and unpacking *matmodlab* from
`PyPI <http://pypi.python.org/pypi/matmodlab>`_ or from
`github <http://github.com/tjfulle/matmodlab>`_, there will be a folder that
contains, among other files, the directories ``femlib``, ``matmodlab``, and
``tabfileio``.

Using your preferred python interpreter, run

::

  python setup.py install

or

::

  python setup.py develop

Both commands make *matmodlab* usable in the same way as using ``pip`` or
``easy_install``. The only difference is that when you setup using the
``develop`` argument the downloaded files are linked to, not moved, so changes
are applied immediately and do not require you to re-install *matmodlab*.



The Hard Way
------------

Get *matmodlab* as detailed in ``The Manual Way`` but do the following:

#) Add ``path/to/files/matmodlab/bin`` to your ``PATH``

#) Add ``path/to/files`` to your ``PYTHONPATH``


Build (Optional)
----------------

Fortran models are built as-needed when *matmodlab* attempts to run a
simulation and it cannot find the compiled model. However, if you want
to build a model without running a simulation or if you want to build an
extension pack you will need to use the ``mml build`` command.

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

Testing is done through the ``mml test`` command. However, this is just a
wrapper around the ``py.test`` command, which can also be used. To test
*matmodlab* after installation, execute::

	mml test -k fast

which will run the "fast" tests. To run the full test suite execute::

	mml test

Please note that running all of the tests takes several minutes.

Troubleshooting
===============

If you experience problems when building/installing/testing *matmodlab*, you can
ask help from `Tim Fuller <timothy.fuller@utah.edu>`_ or
`Scot Swan <scot.swan@gmail.com>`_.
