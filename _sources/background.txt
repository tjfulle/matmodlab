
Introduction
############

The Material Model Laboratory (*matmodlab*) is a suite of tools whose
purpose is to aid in the rapid development and testing of material models.
*matmodlab* is made up of several components, the most notable being the
Material Model Driver. The material model driver can be thought to drive a
single material point of a finite element simulation through very specific
user designed paths. This permits exercising material models in ways not
possible in finite element calculations, desgining verification and validation
tests of the material response, among others. *matmodlab* is a small suite
of tools at the developers disposal to aid in the design and implementation of
material models in larger finite element host codes. It is also a useful tool
to analysists for understanding and parameterizing a material's response to
deformation.

The core of *matmodlab* code base is written in Python and leverages
Python's object oriented programming (OOP) design. OOP techniques are used
throughout *matmodlab* to setup and manage simulation data. Computationally
heavy portions of the code, and many material models themselves are written in
Fortran for its speed and ubiquity in scientific computing. Calling Fortran
procedures from Python is made possible by the ``f2py`` module, standard in
Numpy, that compiles and creates Python shared object libraries from Fortran
sources.

Output files from *matmodlab* simulations are either tabular text files or,
by default, in the ExodusII_ database format, developed at Sandia National Labs
for storing finite element simulation data. Since *matmodlab* is designed to
be used by material model developers, it is expected that the typical user will
want access to *all* available output from a material model, thus all
simulation data is written to the output database. ExodusII database files can
be visualized with the *matmodlab* visualization utility, in addition to other
visualization packages such as ParaView_.

*matmodlab* is free software released under the MIT License.


Why a Single Element Driver?
============================

Due to their complexity, it is often over-kill to use a finite element code
for constitutive model development. In addition, features such as artificial
viscosity can mask the actual material response from constitutive model
development. Single element drivers allow the constituive model developer to
concentrate on model development and not the finite element response. Other
advantages of *matmodlab* (or, more generally, of any stand-alone
constitutive model driver) are

  * *matmodlab* is a very small, special purpose, code. Thus, maintaining
    and adding new features to *matmodlab* is very easy.

  * Simulations are not affected by irrelevant artifacts such as artificial
    viscosity or uncertainty in the handling of boundary conditions.

  * It is straightforward to produce supplemental output for deep analysis of
    the results that would otherwise constitute an unnecessary overhead in a
    finite element code.

  * Specific material benchmarks may be developed and automatically run
    quickly any time the model is changed.

  * Specific features of a material model may be exercised easily by the model
    developer by prescribing strains, strain rates, stresses, stress rates,
    electric fields, temperatures, and deformation gradients as functions of
    time.

Why Python?
===========

Python is an interpreted, high level object oriented language. It allows for
writing programs rapidly and, because it is an interpreted language, does not
require a compiling step. While this might make programs written in python slower
than those written in a compiled language, modern packages and computers make the
speed-up difference between python and a compiled language for single element
problems almost insignificant.

For numeric computations, the `NumPy <http://www.numpy.org>`_ and `SciPy
<http://www.scipy.org>`_ modules allow programs written in Python to leverage
a large set of numerical routines provided by ``LAPACK``, ``BLASPACK``,
``EIGPACK``, etc. Python's APIs also allow for calling subroutines written in
C or Fortran (in addition to a number of other languages), a prerequisite for
model development as most legacy material models are written in Fortran. In
fact, most modern material models are still written in Fortran to this day.

Python's object oriented nature allows for rapid installation of new material
models.

Historical Background
=====================

When I was a graduate student at the University of Utah I had the good fortune
to have as my advisor Dr. Rebecca Brannon `Rebecca Brannon's
<http://www.mech.utah.edu/~brannon/>`_. Prof. Brannon instilled in me the
necessity to develop material models in small special purpose drivers, free
from the complexities of larger finite element codes. To this end, I began
developing material models in Prof. Brannon's *MED* driver (available upon
request from Prof. Brannon). The *MED* driver was a special purpose driver for
driving material models through predefined strain paths. After completing
graduate school I began employment as a member of the Technical Staff at
Sandia National Labs. Among the many projects I worked on was the development
of material models for geologic applications. There, I found need to drive the
material models through prescribed stress paths to match experimental records.
This capability was not present in the *MED* and I sought a different
solution. The solution came from the *MMD* driver, created years earlier at
Sandia, by Tom Pucick. The *MMD* driver had the capability to drive material
models through prescribed stress and strain paths, but also lacked many of the
IO features of the *MED*. And so, for some time I used both the *MED* and
*MMD* drivers in applications that suited their respective strengths. After
some time using both drivers, I decided to combine the best features of each
into my own driver. Both the *MED* and *MMD* drivers were written in Fortran
and I decided to write the new driver in Python so that I could leverage the
large number of builtin libraries. The Numpy and Scipy Python libraries would
be used for handling most number crunching. The new driver came to be known as
*matmodlab*. *matmodlab* added many unique capabilities and became a capable piece
of software used by other staff members at Sandia. But, *matmodlab* suffered
from the fact that it was my first foray in to programming with Python. After
some time, the bloat and bad programming practices with *matmodlab* caused me to
spend a few weekends re-writing it in to what is now known as *matmodlab*.

Obtaining *matmodlab*
=====================

*matmodlab* is an open source project licensed under the MIT license. A copy
of may be obtained from `<https://github.com/tjfulle/matmodlab>`_

About This Guide
================

*matmodlab* is developed as a tool for developers and analysts who care to
understand the responses of material models to specific deformation paths. The
target audience is assumed to have a basic knowledge of continuum mechanics
and familiarity with other finite element codes. Accordingly, concepts of
continuum mechanics and finite element methods are not described in detail and
programing techniques are also not described.

.. _solmeth:


References
==========

.. [ExodusII] Schoof, L. A.and Victor R. Yarberry, "EXODUS II A Finite Element Data Model", SAND92-2137, Sandia National Laboratories, (1995).

.. [ParaView] A. Henderson, ParaView Guide, A Parallel Visualization Application. Kitware Inc., 2007.
