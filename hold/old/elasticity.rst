
##########################
Elastic Constitutive Model
##########################

The elastic constitutive model is an implementation of isotropic linear
elasticity.

User Input
==========

Invocation
----------

The elastic constitutive model is invoked in the material leg of the input file by

::

  constitutive model elastic

=========== ========================================================
Name        Aliases
=========== ========================================================
``elastic`` ``linear elastic``, ``hooke``
=========== ========================================================

Registered Parameters
---------------------

====================== ========== ==============================================
Symbol                 Input Name Description
====================== ========== ==============================================
:math:`\lambda`        ``LAM``    First Lame constant
:math:`\SMod`          ``G``      Elastic shear modulus
:math:`E`              ``E``      Young's modulus
:math:`\nu`            ``NU``     Poisson's ratio
:math:`\BMod`          ``K``      Elastic bulk modulus
:math:`H`              ``H``      Constrained modulus
:math:`K_o`             ``KO``    SIGy/SIGx in uniaxial strain
:math:`c_l`            ``CL``     Longitudinal wave speed
:math:`c_t`            ``CT``     shear (transverse) wave speed
:math:`c_0`            ``C0``     bulk/plastic wave speed
:math:`c_r`            ``CR``     Thin rod wave speed
:math:`\rho`           ``RHO``    Density
====================== ========== ==============================================

.. note::

   The ``elastic`` model uses only the bulk and shear moduli, but any *two* of
   the above elastic moduli, or *two* wave speeds plus the density, can be
   specified in the input deck, from which the bulk and shear moduli will be
   computed.

Options
-------

============ ==================================================================
Option       Description
============ ==================================================================
``fortran``  Use the fortran implementation of the ``plastic`` model
============ ==================================================================


Background
==========

Elastic Response
----------------

The rate of stress is given by

.. math::
   \dStress = \Stiffness\DDotProd\dStrain

where :math:`\dStrain` is the rate of strain.


Numerical Implementation
------------------------

Presuming an isotropic stiffness with constant bulk and shear moduli, first
order integration of the governing equations gives

.. math::
   \Stress = \Stress_{n} + \dStress\dt

where

.. math::
   \dStress = \BMod\text{tr}\dStrain\SOIdentity + 2\SMod\dDevStrain

and :math:`\BMod` and :math:`\SMod` are the bulk and shear moduli,
respectively.
