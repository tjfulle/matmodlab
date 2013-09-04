Building *gmd*
##############

*gmd* is an object oriented model driver. The majority of the source code
is written in Python and requires no additional building. Many of the material
models, however, are written in Fortran and require a seperate compile step.


System Requirements
===================

*gmd* has been built and tested extensively on several versions of linux and
the Apple Mac OSX 10.8 operating systems.


Required Software
=================

*gmd* requires the following software installed for your platform:

#) `Python 2.7 <http://www.python.org/>`_ or newer

#) `NumPy 1.6 <http://www.numpy.org/>`_ or newer

#) `SciPy 0.10.1 <http://www.scipy.org/>`_ or newer

#) A fortran compiler

*gmd* is developed and tested using the `Enthought <http://http://www.enthought.com/>`_ *Python distribution*.

.. _installation:

Installation
============

#) Make sure that all *gmd* prerequisites are installed and working properly.

#) Add ``GMD_ROOT/toolset`` to your ``PATH`` environment variable

#) [Optional] Set the ``GMDMTLS`` environment variable to point to directories
   containing additional material models (user developed, not part of *gmd*.

#) [Optional] Set the ``GMDTESTS`` environment variable to point to
   directories containing additional tests.

#) Change to ``GMD_ROOT/toolset`` and run::

        % PYTHON setup.py

   where ``PYTHON`` is the python interpreter that has *NumPy* and *SciPy*
   installed. :file:`setup.py` will build the third party libraries and write
   the following executable scripts::

       GMD_ROOT
         toolset/
           buildmtls
           gmd
	   gmddiff
	   gmddump
	   gmdviz
           runtests

#) execute::

	% buildmtls

   which will build the *gmd* material libraries and create a database all
   built and installed materials in
   :file:`GMD_ROOT/materials/materials.db`


Testing the Installation
========================

To test *gmd* after installation, execute::

	% runtests -k fast

which will run the "fast" tests. To run the full test suite execute::

	% runtests [-jN]

Please note that running all of the tests takes several minutes.


Troubleshooting
===============

If you experience problems when building/installing/testing *gmd*, you can ask
help from `Tim Fuller <tjfulle@sandia.gov>`_.  Please include the following
information in your message:

#) Platform information OS, its distribution name and version information etc.::

        % PYTHON -c 'import os,sys;print os.name,sys.platform'
	% uname -a


#) Information about C,C++,Fortran compilers/linkers as reported by
   the compilers when requesting their version information, e.g.,
   the output of::

        % gcc -v
        % gfortran --version

#) Python version::

        % PYTHON -c 'import sys;print sys.version'

#) *NumPy* version::

        % PYTHON -c 'import numpy;print numpy.__version__'

#) *SciPy* version::

        % PYTHON -c 'import scipy;print scipy.__version__'

#) Feel free to add any other relevant information.
