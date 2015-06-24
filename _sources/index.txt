The Material Model Laboratory
#############################

The Material Model Laboratory (Matmodlab) is a single element material model
driver aimed at developers of constitutive routines targeted for deployment in
finite element codes.

This guide is separated into four main parts:

* Part 1: :ref:`intro_and_overview`
* Part 2: :ref:`model_create_and_execute`
* Part 3: :ref:`mat_index`
* Part 4: :ref:`test_index`

About This Guide
================

This guide serves as both a User's Guide and Application Programming Interface
(API) guide to Matmodlab. The guide assumes a working knowledge of the
computing languages Matmodlab is developed in, namely `Python
<https://www.python.org>`_ and `Fortran
<http://www.fortran.com/the-fortran-company-homepage/fortran-tutorials>`_. No
attempt to describe them is made. Online tutorials for each language are
readily available. Likewise, the target audience is assumed to have a basic
knowledge of continuum mechanics and familiarity with other finite element
codes. These concepts are also not described in detail.

Conventions Used in the Guide
-----------------------------

* Python objects are typeset in ``fixed point font``.

License
=======

Matmodlab is an open source project licensed under the `MIT <http://opensource.org/licenses/MIT>`_ license.

Obtaining Matmodlab
===================

Latest Stable Version
---------------------

The latest stable version of Matmodlab can be installed via pip::

  pip install matmodlab

Source Code Repository
----------------------

Matmodlab is maintained with git. The source code can be obtained from `<https://github.com/tjfulle/matmodlab>`_

See :ref:`intro_install` for more installation details.

Obtaining Additional Help
=========================

In addition to this guide, many examples can be found in
``matmodlab/examples`` and ``matmodlab/tests``

Indices and tables
==================

* :ref:`genindex`

* :ref:`search`

.. toctree::
   :maxdepth: 4
   :hidden:
   :numbered: 2

   intro/index
   execution/index
   material/index
   test/index
