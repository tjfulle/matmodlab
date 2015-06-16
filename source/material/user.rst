
User Defined Materials
######################

Overview
========

Matmodlab calls user defined materials at each deformation increment through all analysis steps.  User defined materials are

* written in fortran or python
* called from python

References
==========

User Material Interface
=======================

Matmodlab provides two avenues for interacting with user materials

.. toctree::
   :maxdepth: 1
   :titlesonly:

   python
   fortran

Of the two options, the fortran interface is the most similar to typical FE code interfaces.
