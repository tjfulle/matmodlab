The Material Model Laboratory
#############################

This guide to the Material Model Laboratory is a work in progress.  The documentation is currently in a state of flux between describing version 2 and the new default version 3.  Section headings containing an asterisk (*) are not up to date with version 3.  Efforts are currently under way to update the documentation.

Obtaining Matmodlab
===================

Matmodlab is an open source project licensed under the MIT license. The source can be obtained from `<https://github.com/tjfulle/matmodlab>`_

Matmodlab can be installed via pip::

  pip install matmodlab

See :ref:`Installing` for more installation details.

About This Guide
================

Matmodlab is developed as a tool for developers and analysts who care to
understand the responses of material models to specific deformation paths. The
target audience is assumed to have a basic knowledge of continuum mechanics
and familiarity with other finite element codes. Accordingly, concepts of
continuum mechanics and finite element methods are not described in detail and
programing techniques are also not described.

Obtaining Additional Help
=========================

In addition to this guide, many example input files can be found in ``matmodlab/inputs`` and ``matmodlab/tests``

Indices and tables
==================

* :ref:`genindex`

* :ref:`search`

.. toctree::
   :maxdepth: 3
   :hidden:
   :numbered: 2

   intro
   basic
   material/material
   test
