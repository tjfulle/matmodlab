
##########################
Plastic Constitutive Model
##########################

The plastic constitutive model is an implementation of plasticity model with a
von Mises type yield function that supports both isotropic and kinematic
hardening.

User Input
==========

Invocation
----------

The plastic constitutive model is invoked in the material leg of the input file by

::

  constitutive model plastic

=========== ========================================================
Name        Aliases
=========== ========================================================
``plastic`` ``elastic plastic``, ``von mises``
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
:math:`K_o`            ``KO``     SIGy/SIGx in uniaxial strain
:math:`c_l`            ``CL``     Longitudinal wave speed
:math:`c_t`            ``CT``     shear (transverse) wave speed
:math:`c_0`            ``C0``     bulk/plastic wave speed
:math:`c_r`            ``CR``     Thin rod wave speed
:math:`\rho`           ``RHO``    Density
:math:`\InitYldStress` ``Y``      Initial yield stress
:math:`a`              ``A``      Kinematic hardening parameter
:math:`c`              ``C``      Isotropic hardening parameter
:math:`m`              ``M``      Isotropic hardening parameter
====================== ========== ==============================================

.. note::

   The ``plastic`` model uses only the bulk and shear moduli, but any *two* of
   the above elastic moduli, or *two* wave speeds plus the density, can be
   specified in the input deck, from which the bulk and shear moduli will be
   computed.

Plotable Output
---------------

======================== ============= =========================================
Symbol                   Plot Key      Description
======================== ============= =========================================
:math:`\DPS`             ``GAM``       Distortional plastic strain
:math:`\BackStress_{11}` ``BSTRESS11`` ``11`` component of back stress
:math:`\BackStress_{22}` ``BSTRESS22`` ``22`` component of back stress
:math:`\BackStress_{33}` ``BSTRESS33`` ``33`` component of back stress
:math:`\BackStress_{12}` ``BSTRESS12`` ``12`` component of back stress
:math:`\BackStress_{23}` ``BSTRESS23`` ``23`` component of back stress
:math:`\BackStress_{13}` ``BSTRESS13`` ``13`` component of back stress
======================== ============= =========================================

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
   \dStress = \Stiffness\DDotProd\dElastStrain

where :math:`\dElastStrain` is the elastic part of the strain rate. The the
total strain rate is assumed to be the sum of elastic and plastic parts

.. math::
   \dStrain = \dElastStrain + \dPlastStrain

The plastic strain rate is expressed as the product of its magnitude
:math:`\Mpsr` and direction :math:`\FlowDir`

.. math::
   \dPlastStrain = \Mpsr\FlowDir

giving for the stress increment

.. math::
   \dStress = \Stiffness\DDotProd\left(\dStrain - \Mpsr\FlowDir\right)
            = \Stiffness\DDotProd\dStrain - \Mpsr\ReturnDir

where :math:`\ReturnDir = \Stiffness\DDotProd\FlowDir` is the *return direction*.


Plastic Flow
------------

The boundary between elastically obtainable stress states and states
unobtainable through inviscid processes is termed the *yield surface*.
Mathematically, the yield surface is expressed in terms of a stress and
internal variable dependent *yield function*
:math:`\YldFcn[\Stress,\TensIsv{i},\Isv{i}]`, where :math:`\TensIsv{i}` and
:math:`\Isv{i}` represent the contribution of all individual tensor and scalar
internal state variables (ISVs) that change only with plastic loading,
respectively. The *yield criterion* is the statement that
:math:`\YldFcn[\Stress,\TensIsv{i}, \Isv{i}]\leq0\quad\forall\quad\Stress,
\:\TensIsv{i},\:\Isv{i}`. The material is said to be yielded if for a *trial
elastic stress* :math:`\TrialStress`, computed assuming the entire step is
elastic, the yield criterion is violated, or :math:`\YldFcn[\TrialStress,
\TensIsv{i}, \Isv{i}] > 0`. In this case, the material "flows" plastically
such that the yield criterion is satisfied. It can be shown that the yield
criterion will be satisfied if the *consistency condition* is met. The
consistency condition is the statement that

.. math::
   \dYldFcn = \PDer{\YldFcn}{\Stress}\DDotProd\dStress
              + \PDer{\YldFcn}{\TensIsv{i}}\DDotProd\dTensIsv{i}
              + \PDer{\YldFcn}{\Isv{i}}\dIsv{i} = 0

Assuming that the internal variables change only during plastic loading,
allows evolutionary type of equations of the form

.. math::
   \dIsv{i} = \Mpsr\IsvModulus{i}, \quad \dTensIsv{i} = \Mpsr\TensIsvModulus{i}

Substituting the evolution equation for the internal state variables and the
governing equation for the elastic stress increment, the consistency
condition becomes

.. math::
   \PDer{\YldFcn}{\Stress}\DDotProd\left(\Stiffness\DDotProd\dStrain
       - \Mpsr\ReturnDir\right)
   + \Mpsr\PDer{\YldFcn}{\TensIsv{i}}\DDotProd\TensIsvModulus{i}
   + \Mpsr\IsvModulus{i}\PDer{\YldFcn}{\Isv{i}} = 0

from which

.. math::
   \Mpsr = \frac{\YldNormal\DDotProd\Stiffness\DDotProd\dStrain}
                {\YldNormal\DDotProd\ReturnDir - H}
         = \frac{\YldNormal\DDotProd\dTrialStress}
                {\YldNormal\DDotProd\ReturnDir - H}

where :math:`\dTrialStress=\Stiffness\DDotProd\dStrain` is the trial elastic
stress increment, :math:`\YldNormal` is the unit normal to the yield surface
and :math:`H` is the ensemble hardening modulus, given by

.. math::
   \YldNormal = \frac{\PDer{\YldFcn}{\Stress}}
                     {\Norm{\PDer{\YldFcn}{\Stress}}}
   , \quad
   H = \frac{\IsvModulus{i}\PDer{\YldFcn}{\Isv{i}} +
             \PDer{\YldFcn}{\TensIsv{i}}\DDotProd\TensIsvModulus{i}}
            {\Norm{\PDer{\YldFcn}{\Stress}}}


Yield Function and Evolution Equations
--------------------------------------

The ``plastic`` material's yield function is a combined kinematic/isotropic
hardening von Mises (:math:`\sqrt{\JTwo}`) type yield function, given by

.. math::
   \YldFcn[\Stress, \BackStress, \YldStress] =
       \sqrt{\JTwo[\ShiftedStressSym]} - \frac{1}{\sqrt{3}}\YldStress

where :math:`\ShiftedStress` is the stress tensor relative to the back stress
:math:`\BackStress`,

.. math:: \ShiftedStress = \Stress - \BackStress

The yield stress :math:`\YldStress` is given by the following empirical
hardening power law

.. math::
   \YldStress = \InitYldStress + c{\DPS}^{m}

where :math:`\InitYldStress`, :math:`c` and :math:`m` are constants and
:math:`\DPS` is the distortional plastic strain, defined as

.. math::
   \DPS = \int\Mpsr\sqrt{\DevFlowDir\DDotProd\DevFlowDir}\dt

Taking the rate of the yield stress, and substituting :math:`\dDPS`, allows
the yield stress rate to be written as

.. math::
   \dYldStress = \IsvModulus{\YldStress}\Mpsr, \quad
   \IsvModulus{\YldStress} =
      mc\left(\frac{\YldStress - \InitYldStress}{c}\right)^{(m-1)/m}


The back stress :math:`\BackStress` evolves according to

.. math::
   \dBackStress = \frac{2}{3}a\dDevPlastStrain = \frac{2}{3}a\Mpsr\DevFlowDir

which allows the ISV modulus associated with the back stress to be expressed as

.. math::
   \TensIsvModulus{\BackStressSym} = \frac{2}{3}a\DevFlowDir

The derivatives of the yield function with respect to stress, back stress, and
yield stress are given by

.. math::
   \PDer{\YldFcn}{\Stress} =
     \PDer{\YldFcn}{\JTwo[\ShiftedStressSym]}
     \PDer{\JTwo[\ShiftedStressSym]}{{\ShiftedStress}}
     \PDer{{\ShiftedStress}}{\Stress} =
     \frac{\DevShiftedStress}{\sqrt{2}r}

.. math::
   \Norm{\PDer{\YldFcn}{\Stress}} =
     \frac{\sqrt{\DevShiftedStress\DDotProd\DevShiftedStress}}{\sqrt{2}r} =
     \frac{1}{\sqrt{2}}

.. math::
   \PDer{\YldFcn}{\BackStress} =
     \PDer{\YldFcn}{\JTwo[\ShiftedStressSym]}
     \PDer{\JTwo[\ShiftedStressSym]}{{\ShiftedStress}}
     \PDer{{\ShiftedStress}}{\BackStress} =
     -\frac{\DevShiftedStress}{\sqrt{2}r}

.. math::
   \PDer{\YldFcn}{\YldStress} = -\frac{1}{\sqrt{3}}

where the radius :math:`r=\sqrt{2/3}\YldStress`.

Combining the above results, and assuming an associative flow rule
:math:`\FlowDir=\YldNormal`, the magnitude of the plastic strain rate is

.. math::
   \Mpsr =
     \frac{\YldNormal\DDotProd\dTrialStress}
          {2\SMod + \frac{2}{3}a + \sqrt{\frac{2}{3}}\IsvModulus{\YldStress}}


Numerical Implementation
------------------------

Presuming an isotropic stiffness with constant bulk and shear moduli, first
order integration of the governing equations gives

.. math::
   \DPS = \DPS_{n}

.. math::
   \TrialStress = \Stress_{n} + \dTrialStress\dt

where

.. math::
   \dTrialStress = \BMod\text{tr}\dStrain\SOIdentity + 2\SMod\dDevStrain

and :math:`\BMod` and :math:`\SMod` are the bulk and shear moduli,
respectively. Using the above results we get

.. math::
   \YldStress = \InitYldStress + c\left(\DPS\right)^{m}

.. math::
   \ShiftedStress = \TrialStress - \BackStress_{n}

.. math::
   \sqrt{\JTwo[\ShiftedStress]} =
     \sqrt{\frac{1}{2}\DevShiftedStress\DDotProd\DevShiftedStress}

.. math::
   \YldNormal = \frac{\DevShiftedStress}{r}, \quad
   \ReturnDir = 2\SMod\YldNormal

and finally,

.. math::
    \DPS = {\DPS}_{n} + \Mpsr\dt

.. math::
   \Stress = \TrialStress - \Mpsr\ReturnDir\dt

.. math::
   \BackStress = \BackStress_{n} + \frac{2}{3}\Mpsr\YldNormal\dt

where

.. math::
   \Mpsr =
     \begin{cases}
       0, & \quad \YldFcn[\TrialStress, \BackStress, \YldStress] < 0 \\
       \frac{\YldNormal\DDotProd\dTrialStress}
          {2\SMod + \frac{2}{3}a + \sqrt{\frac{2}{3}}\IsvModulus{\YldStress}}
           & \quad \YldFcn[\TrialStress, \BackStress, \YldStress] \geq 0
      \end{cases}


