.. _Solution Method:

The Matmodlab Solution Method
#############################

.. contents:: Contents
   :depth: 1
   :local:

Overview
========

Matmodlab exercises a material model directly by "driving" it through user
specified paths. Matmodlab computes an increment in deformation for a given
step and requires that the material model update the stress in the material to
the end of that step.

.. _Role of Material Model:

The Role of the Material Model in Continuum Mechanics
=====================================================

.. _Conservation Laws:

Conservation Laws
-----------------

Conservation of mass, momentum, and energy are the central tenets of the
analysis of the response of a continuous media to deformation and/or load.
Each conservation law can be summarized by the statement that *the time rate
of change of a quantity in a continuous body is equal to the rate of
production in the interior plus flux through the boundary*

Mathematically, the conservation laws for a point in the continuum are

* Conservation of mass

  .. math::

     \dDensity + \Density\Del\DotProd\dDisplacement = 0

* Conservtion of momentum per unit volume

  .. math::

     \Density\Der{}{t}\dDisplacement =
     \underset{\text{internal forces}}{\boxed{\Del\DotProd\Stress}} +
     \underset{\text{body forces}}{\boxed{\BodyForce}}

* Conservation of energy per unit volume

  .. math::

     \Density\Der{}{t}\Energy =
     \underset{\text{heat source}}{\boxed{\Density s}} +
     \underset{\text{strain energy}}{\boxed{\Stress\DDotProd\dStrain}} +
     \underset{\text{heat flux}}{\boxed{\Del\DotProd\HeatFlux}}

where :math:`\Displacement` is the displacement, :math:`\Density` the mass
density, :math:`\Stress` the stress, :math:`\dStrain` the rate of strain,
:math:`\BodyForce` the body force per unit volume, :math:`\HeatFlux` the heat
flux, :math:`s` the heat source, and :math:`\Energy` is the internal energy
per unit mass.

In solid mechanics, mass is conserved trivially, and many problems are
adiabatic or isotrhermal, so that only the momentum balance is explicitly
solved

.. math::
   :label: mbal

   \Density\Der{}{t}\dDisplacement =
   \underset{\text{internal forces}}{\boxed{\Del\DotProd\Stress}} +
   \underset{\text{body forces}}{\boxed{\BodyForce}}

The balance of linear momentum is the continuum mechanics generalization of
Newton's second law :math:`F=ma`.

The first term on the RHS of :eq:`mbal` represents the internal forces, which
arise in the medium to resist imposed deformation. This resistance is a
fundamental response of matter and is given by the divergence of the stress
field.

The balance of linear momentum represents an initial boundary value problem
for applications of interest in solid dynamics:

.. math::
   :label: ibvp

   \begin{aligned}
     \Density\Der{}{t}\dDisplacement = \Del\DotProd\Stress + \BodyForce&
     &&\quad\text{in }\Omega \\
     \Displacement = \Displacement_0& &&\quad\text{on }\Gamma_0 \\
     \Stress\DotProd\normal = \Traction& &&\quad\text{on }\Gamma_t \\
     \dDisplacement\left(\position, 0\right) =
     \dDisplacement_0\left(\position\right)&
     &&\quad\text{on }\position\in\Omega
   \end{aligned}

.. _femeth:

The Finite Element Method
-------------------------

The form of the momentum equation in :eq:`ibvp` is termed the **strong** form.
The strong form of the initial BVP problem can also be expressed in the weak
form by introducing a test function :math:`\Tensor{w}{}{}` and integrating
over space

.. math::
   :label: ibvp-1

     \begin{aligned}
       \int_{\Omega}\Tensor{w}{}{}\DotProd\left(
	 \Del\DotProd\Stress + \BodyForce - \Density\Der{}{t}\dDisplacement
       \right)\,d\Omega& &&\quad \forall \Tensor{w}{}{} \\
       \Displacement = \Displacement_0& &&\quad\text{on }\Gamma_0 \\
       \Stress\DotProd\normal = \Traction& &&\quad\text{on }\Gamma_t \\
       \dDisplacement\left(\position, 0\right) =
       \dDisplacement_0\left(\position\right)&
       &&\quad\text{on }\position\in\Omega
     \end{aligned}

Integrating :eq:`ibvp-1` by parts allows the traction boundary conditions to
be incorporated in to the governing equations

.. math::
   :label: weak

    \begin{aligned}
       \int_{\Omega}\Density\Tensor{w}{}{}\DotProd\Acceleration +
       \Stress\DDotProd\Del\Tensor{w}{}{}\,d\Omega
       = \int_{\Omega}\Tensor{w}{}{}\DotProd\BodyForce\,d\Omega +
       \int_{\Gamma}\Tensor{w}{}{}\DotProd\Traction\,d\Gamma_{t}&
       &&\forall \Tensor{w}{}{} \\
       %
       \Displacement = \Displacement_0& &&\quad\text{on }\Gamma_0 \\
       \dDisplacement\left(\position, 0\right) =
       \dDisplacement_0\left(\position\right)&
       &&\quad\text{on }\position\in\Omega
    \end{aligned}

This form of the IBVP is called the **weak** form. The weak form poses the
IBVP as a integro-differential equation and eliminates singularities that may
arise in the strong form. Traction boundary conditions are incorporated in the
governing equations. The weak form forms the basis for finite element methods.

In the finite element method, forms of :math:`\Tensor{w}{}{}` are assumed in
subdomains (elements) in :math:`\Omega` and displacements are sought such that
the force imbalance :math:`R` is minimized:

.. math::
   :label: resid

   R = \int_{\Omega}\Tensor{w}{}{}\DotProd\BodyForce\,d\Omega +
   \int_{\Gamma}\Tensor{w}{}{}\DotProd\Traction\,d\Gamma_{t} -
    \int_{\Omega}\Density\Tensor{w}{}{}\DotProd\Acceleration +
           \Stress\DDotProd\Del\Tensor{w}{}{}\,d\Omega

The equations of motion as described in :eq:`resid` are not closed, but
require relationships relating :math:`\Stress` to :math:`\Displacement`

.. centered::
   Constitutive model :math:`\longrightarrow` relationship between
   :math:`\Stress` and :math:`\Displacement`

In the typical finite element procedure, the host finite element code passes
to the constitutive routine the stress and material state at the beginning of
a finite step (in time) and kinematic quantities at the end of the step. The
constitutive routine is responsible for updating the stress to the end of the
step. At the completion of the step, the host code then uses the updated
stress to compute kinematic quantities at the end of the next step. This
process is continued until the simulation is completed. The host finite
element handles the allocation and management of all memory, including memory
required for material variables.

.. _mmlsol:

Solution Procedure
==================

In addition to providing a platform for material model developers to formulate
and test constitutive routines, Matmodlab aims to provide users of material
models an independent platform to exercise, parameterize, and compare material
responses against single element finite element simulations. To this end, the
solution procedure in Matmodlab is similar to that of the finite element
method, in that the host code (Matmodlab) provides to the constitutive
routine a measure of deformation at the end of a finite step and expects the
updated stress in return. However, rather than solve the momentum equation at
the beginning of each step and advancing kinematic quantities to the step's
end, Matmodlab retrieves updated kinematic quantities from user defined
tables and/or functions.

The path through which a material is exercised is defined by piecewise
continuous "steps" in which tensor components of stress and/or deformation are
specified at discrete points in time. The components are used to obtain a
sequence of piecewise constant strain rates that are used to advance the
kinematic state. Supported components are strain, strain rate, stress,
stress rate, deformation gradient, displacement, and velocity. "Mixed-modes"
of strain and stress (and their rates) are supported. Components of
displacement and velocity control are applied only to the "+" faces of a unit
cube centered at the coordinate origin.

.. _strain_tensor:

The Strain Tensor
-----------------

The components of strain are defined by

.. math::
   \Strain = \frac{1}{\kappa}\left(\RightStretch^\kappa - \SOIdentity\right)

where :math:`\RightStretch` is the right Cauchy stretch tensor, defined by the
polar decomposition of the deformation gradient :math:`\DefGrad =
\Rotation\DotProd\RightStretch`, and :math:`\kappa` is a user specified
"Seth-Hill" parameter that controls the strain definition. Choosing
:math:`\kappa=2` gives the Lagrange strain, which might be useful when testing
models cast in a reference coordinate system. The choice :math:`\kappa=1`,
which gives the engineering strain, is convenient when driving a problem over
the same strain path as was used in an experiment. The choice :math:`\kappa=0`
corresponds to the logarithmic (Hencky) strain. Common values of
:math:`\kappa` and the associated names for each (there is some ambiguity in
the names) are listed in `Table 1`_

.. _Table 1:

+----------------+--------------------------+
| :math:`\kappa` | Name(s)                  |
+================+==========================+
|  -2            | Green                    |
+----------------+--------------------------+
|  -1            | True, Cauchy             |
+----------------+--------------------------+
|   0            | Logarithmic, Hencky, True|
+----------------+--------------------------+
|   1            | Engineering, Swainger    |
+----------------+--------------------------+
|   2            | Lagrange, Almansi        |
+----------------+--------------------------+

The volumetric strain :math:`\Strain[v]` is defined

.. math::
   :label: volstrain

   \Strain[v] =
   \begin{cases}
       \OneOver{\kappa}\left(\Jacobian^{\kappa} - 1\right)
       & \text{if }\kappa \ne 0 \\
       \ln{\Jacobian} & \text{if }\kappa = 0
   \end{cases}

where the Jacobian :math:`\Jacobian` is the determinant of the deformation gradient.

Each step component, from time :math:`t=0` to :math:`t=t_f` is
subdivided into a user-specified number of "frames" and the material model
evaluated at each frame. When volumetric strain, deformation gradient,
displacement, or velocity are specified for a step, Matmodlab internally
determines the corresponding strain components. If a component of stress is
specified, Matmodlab determines the strain increment that minimizes the
distance between the prescribed stress component and model response.

.. _Stress Control:


Stress Control
--------------

Stress control is accomplished through an iterative scheme that seeks to
determine the unkown strain rates, :math:`\dStrain\,[\text{v}]`, that satisfy

.. math::

   \Stress\left(\dStrain\,[\text{v}]\right) = \PrescStress

where, :math:`\text{v}` is a vector subscript array containing the components
for which stresses are prescribed, and :math:`\PrescStress` are the components
of prescribed stress.

The approach is an iterative scheme employing a multidimensional Newton's
method. Each iteration begins by determining the submatrix of the material
stiffness

.. math::

   \Stiffness_{\text{v}} = \Stiffness\,[\text{v}, \text{v}]

where :math:`\Stiffness` is the full stiffness matrix
:math:`\Stiffness=d\Stress/d\Strain`. The value of
:math:`\dStrain\,[\text{v}]` is then updated according to

.. math::

   \dStrain_{n+1}\,[\text{v}] =
       \dStrain_{n}\,[\text{v}] -
       \Stiffness_{\text{v}}^{-1}\DDotProd\Stress^{*}(\dStrain_{n}\,[\text{v}])/dt

where

.. math::

   \Stress^{*}(\dStrain\,[\text{v}]) = \Stress(\dStrain\,[\text{v}])
                                     - \PrescStress

The Newton procedure will converge for valid stress states. However, it is
possible to prescribe invalid stress state, e.g. a stress state beyond the
material's elastic limit. In these cases, the Newton procedure may not
converge to within the acceptable tolerance and a Nelder-Mead simplex method
is used as a back up procedure. A warning is logged in these cases.

.. _The Material Stiffness:

The Material Stiffness
----------------------

As seen in `Stress Control`_, the material tangent stiffness matrix, commonly referred
to as the material's "Jacobian", plays an integral roll in the solution of the
inverse stress problem (determining strains as a function of prescribed
stress). Similarly, the Jacobian plays a role in implicit finite element
methods. In general, the Jacobian is a fourth order tensor in :math:`\R{3}`
with 81 independent components. Casting the stress and strain second order
tensors in :math:`\R{3}` as first order tensors in :math:`\R{9}` and the
Jacobian as a second order tensor in :math:`\R{9}`, the stress/strain relation
in `Stress Control`_ can be written in matrix form as

.. math::

   \begin{Bmatrix}
     \dStress[11] \\
     \dStress[22] \\
     \dStress[33] \\
     \dStress[12] \\
     \dStress[23] \\
     \dStress[13] \\
     \dStress[21] \\
     \dStress[32] \\
     \dStress[31]
   \end{Bmatrix} =
   \begin{bmatrix}
     C_{1111} & C_{1122} & C_{1133} & C_{1112} & C_{1123} & C_{1113} & C_{1121} & C_{1132} & C_{1131} \\
     C_{2211} & C_{2222} & C_{2233} & C_{2212} & C_{2223} & C_{2213} & C_{2221} & C_{2232} & C_{2231} \\
     C_{3311} & C_{3322} & C_{3333} & C_{3312} & C_{3323} & C_{3313} & C_{3321} & C_{3332} & C_{3331} \\
     C_{1211} & C_{1222} & C_{1233} & C_{1212} & C_{1223} & C_{1213} & C_{1221} & C_{1232} & C_{1231} \\
     C_{2311} & C_{2322} & C_{2333} & C_{2312} & C_{2323} & C_{2313} & C_{2321} & C_{2332} & C_{2331} \\
     C_{1311} & C_{1322} & C_{1333} & C_{1312} & C_{1323} & C_{1313} & C_{1321} & C_{1332} & C_{1331} \\
     C_{2111} & C_{2122} & C_{2133} & C_{2212} & C_{2123} & C_{2213} & C_{2121} & C_{2132} & C_{2131} \\
     C_{3211} & C_{3222} & C_{3233} & C_{3212} & C_{3223} & C_{3213} & C_{3221} & C_{3232} & C_{3231} \\
     C_{3111} & C_{3122} & C_{3133} & C_{3312} & C_{3123} & C_{3113} & C_{3121} & C_{3132} & C_{3131}
   \end{bmatrix}
   \begin{Bmatrix}
     \dStrain[11] \\
     \dStrain[22] \\
     \dStrain[33] \\
     \dStrain[12] \\
     \dStrain[23] \\
     \dStrain[13] \\
     \dStrain[21] \\
     \Strain[32] \\
     \dStrain[31]
   \end{Bmatrix}

Due to the symmetries of the stiffness and strain tensors (:math:`\Stiffness[ijkl]=\Stiffness[ijlk]`, :math:`\dStrain[ij]=\dStrain[ji]`), the expression above can be simplified by removing the last three columns of :math:`\Stiffness`:

.. math::

   \begin{Bmatrix}
     \dStress[11] \\
     \dStress[22] \\
     \dStress[33] \\
     \dStress[12] \\
     \dStress[23] \\
     \dStress[13] \\
     \dStress[21] \\
     \dStress[32] \\
     \dStress[31]
   \end{Bmatrix} =
   \begin{bmatrix}
     C_{1111} & C_{1122} & C_{1133} & C_{1112} & C_{1123} & C_{1113} \\
     C_{2211} & C_{2222} & C_{2233} & C_{2212} & C_{2223} & C_{2213} \\
     C_{3311} & C_{3322} & C_{3333} & C_{3312} & C_{3323} & C_{3313} \\
     C_{1211} & C_{1222} & C_{1233} & C_{1212} & C_{1223} & C_{1213} \\
     C_{2311} & C_{2322} & C_{2333} & C_{2312} & C_{2323} & C_{2313} \\
     C_{1311} & C_{1322} & C_{1333} & C_{1312} & C_{1323} & C_{1313} \\
     C_{2111} & C_{2122} & C_{2133} & C_{2212} & C_{2123} & C_{2213} \\
     C_{3211} & C_{3222} & C_{3233} & C_{3212} & C_{3223} & C_{3213} \\
     C_{3111} & C_{3122} & C_{3133} & C_{3112} & C_{3123} & C_{3113}
   \end{bmatrix}
   \begin{Bmatrix}
     \dStrain[11] \\
     \dStrain[22] \\
     \dStrain[33] \\
     2\dStrain[12] \\
     2\dStrain[23] \\
     2\dStrain[13]
   \end{Bmatrix}

Considering the symmetry of the stress tensor (:math:`\dStress[ij]=\dStress[ji]`) and the major symmetry of :math:`\Stiffness` (:math:`\Stiffness[ijkl]=\Stiffness[klij]`), the final three rows of :math:`\Stiffness` may also be ommitted, resulting in the symmetric form

.. math::

   \begin{Bmatrix}
     \dStress[11] \\
     \dStress[22] \\
     \dStress[33] \\
     \dStress[12] \\
     \dStress[23] \\
     \dStress[13]
   \end{Bmatrix} =
   \begin{bmatrix}
     C_{1111} & C_{1122} & C_{1133} & C_{1112} & C_{1123} & C_{1113} \\
              & C_{2222} & C_{2233} & C_{2212} & C_{2223} & C_{2213} \\
              &          & C_{3333} & C_{3312} & C_{3323} & C_{3313} \\
              &          &          & C_{1212} & C_{1223} & C_{1213} \\
              &          &          &          & C_{2323} & C_{2313} \\
    symm      &          &          &          &          & C_{1313} \\
   \end{bmatrix}
   \begin{Bmatrix}
     \dStrain[11] \\
     \dStrain[22] \\
     \dStrain[33] \\
     2\dStrain[12] \\
     2\dStrain[23] \\
     2\dStrain[13]
   \end{Bmatrix}

Letting :math:`\{\dStress[1],\dStress[2],\dStress[3], \dStress[4], \dStress[5], \dStress[6]\}= \{\dStress[11],\dStress[22],\dStress[33], \dStress[12],\dStress[23],\dStress[13]\}` and :math:`\{\dStrain[1],\dStrain[2],\dStrain[3], \dot{\gamma_4}, \dot{\gamma_5}, \dot{\gamma_6}\}= \{\dStrain[11],\dStrain[22],\dStrain[33],2\dStrain[12],2\dStrain[23],2\dStrain[13]\}`, the above stress-strain relationship is re-written as

.. math::

   \begin{Bmatrix}
     \dStress[1] \\
     \dStress[2] \\
     \dStress[3] \\
     \dStress[4] \\
     \dStress[5] \\
     \dStress[6]
   \end{Bmatrix} =
   \begin{bmatrix}
     C_{11} & C_{12} & C_{13} & C_{14} & C_{15} & C_{16} \\
            & C_{22} & C_{23} & C_{24} & C_{25} & C_{26} \\
            &        & C_{33} & C_{34} & C_{35} & C_{36} \\
            &        &        & C_{44} & C_{45} & C_{46} \\
            &        &        &        & C_{55} & C_{56} \\
    symm    &        &        &        &        & C_{66} \\
   \end{bmatrix}
   \begin{Bmatrix}
     \dStrain[1] \\
     \dStrain[2] \\
     \dStrain[3] \\
     \dot{\gamma_4} \\
     \dot{\gamma_5} \\
     \dot{\gamma_6}
   \end{Bmatrix}

As expressed, the components of :math:`\dStrain` and :math:`\dStress` are first order tensors and :math:`\Stiffness` is a second order tensor in :math:`\R{6}`, respectively.

Alternative Representations of Tensors in :math:`\R{6}`
.......................................................

The representation of symmetric tensors at the end of `The Material
Stiffness`_ is known as the "Voight" representation. The shear strain
components :math:`\dStrain[I]=2\dStrain[ij], \ I=4,5,6, \ ij=12,23,13` are
known as the engineering shear strains (in contrast to :math:`\dStrain[ij], \
ij=12,23,13` which are known as the tensor components). An advantage of the
Voight representation is that the scalar product
:math:`\Stress[ij]\dStrain[ij]=\Stress[I]\dStrain[I]` is preserved and the
components of the stiffness tensor are unchanged in :math:`\R{6}`. However,
one must take care to account for the factor of 2 in the engineering shear
strain components.

Alternatively, one can express symmetric second order tensors with their
"Mandel" components
:math:`\{\AA[1],\AA[2],\AA[3],\AA[4],\AA[5],\AA[6]\}=\{\AA[11],\AA[22],\AA[33],
\sqrt{2}\AA[12],\sqrt{2}\AA[23],\sqrt{2}\AA[13]\}`.  Representing both the stress and strain with their Mandel representation also preserves the scalar product, without having to treat the components of stress and strain differently (at the expense of carrying around the factor of :math:`\sqrt{2}` in the off-diagonal components of both).  The Mandel representation has the advantage that its basis in :math:`\R{6}` is orthonormal, whereas the basis for the Voight representation is only orthogonal.  If Mandel components are used, the components of the stiffness must be modified as

.. math::

   \Stiffness =
   \begin{bmatrix}
     C_{11} & C_{12} & C_{13} & \sqrt{2}C_{14}   & \sqrt{2}C_{15} & \sqrt{2}C_{16} \\
            & C_{22} & C_{23} & \sqrt{2}C_{24}   & \sqrt{2}C_{25} & \sqrt{2}C_{26} \\
            &        & C_{33} & \sqrt{2}C_{34}   & \sqrt{2}C_{35} & \sqrt{2}C_{36} \\
            &        &        & 2C_{44}          & 2C_{45}        & 2C_{46} \\
            &        &        &                  & 2C_{55}        & 2C_{56} \\
    symm    &        &        &                  &                & 2C_{66} \\
   \end{bmatrix}
