Matmodlab Background
####################

Why a Single Element Driver?
============================

Single element drivers allow the constituive model developer to concentrate on
model development and not the finite element response. Advantages of
Matmodlab (or, more generally, of any stand-alone constitutive model driver)
are

  * Matmodlab is a very small, special purpose, code. Thus, maintaining
    and adding new features to Matmodlab is very easy.

  * Simulations are not affected by irrelevant artifacts such as artificial
    viscosity or uncertainty in the handling of boundary conditions.

  * It is straightforward to produce supplemental output for deep analysis of
    the results that would otherwise constitute an unnecessary overhead in a
    finite element code.

  * Specific material benchmarks may be developed and automatically run
    quickly any time the model is changed.

  * Specific features of a material model may be exercised easily by the model
    developer by prescribing strains, strain rates, stresses, stress rates, and
    deformation gradients as functions of time.

Why Python?
===========

Python is an interpreted, high level object oriented language. It allows for
writing programs rapidly and, because it is an interpreted language, does not
require a compiling step. While this might make programs written in python slower
than those written in a compiled language, modern packages and computers make the
speed up difference between python and a compiled language for single element
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
in to my own driver. Both the *MED* and *MMD* drivers were written in Fortran
and I decided to write the new driver in Python so that I could leverage the
large number of builtin libraries. The Numpy and Scipy Python libraries would
be used for handling most number crunching. The new driver came to be known as
Matmodlab. Matmodlab added many unique capabilities and became a capable piece
of software used by other staff members at Sandia. But, Matmodlab suffered
from the fact that it was my first foray in to programming with Python. After
some time, the bloat and bad programming practices with Matmodlab caused me to
spend a few weekends re-writing it in to what is now known as Matmodlab.
