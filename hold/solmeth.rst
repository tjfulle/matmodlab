.. _solmeth:

*matmodlab* Solution Method
###########################

*matmodlab* exercises a material model directly by "driving" it through user
specified paths. *matmodlab* computes an increment in deformation for a given
step and requires that the material model update the stress in the material to
the end of that step, given the current state and an increment in deformation.
Because of the similarity of the material model interface in *matmodlab* with
many commercial finite element codes, transitioning material models developed
and tested in *matmodlab* to full finite element codes should be an easy
process. In this chapter, the role and importance of the material model in a
finite element procedure is reviewed. The solution method adopted by each
driver in *matmodlab* is then described and compared with that of finite
elements.

.. _roleofmatmod:

The Role of the Material Model in Continuum Mechanics
=====================================================

.. _cons-laws:

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
and test constitutive routines, *matmodlab* aims to provide users of material
models an independent platform to exercise, parameterize, and compare material
responses against single element finite element simulations. To this end, the
solution procedure in *matmodlab* is similar to that of the finite element
method, in that the host code (*matmodlab*) provides to the constitutive
routine a measure of deformation at the end of a finite step and expects the
updated stress in return. However, rather than solve the momentum equation at
the beginning of each step and advancing kinematic quantities to the step's
end, *matmodlab* retrieves updated kinematic quantities from user defined
tables and/or functions.

The path through which a material is exercised is defined by piecewise
continuous "legs" in which components of the "control type" :math:`c_{i}` are
specified at discrete points in time, shown in `Figure 1`_. The :math:`c_{i}`
are used to obtain a sequence of piecewise constant strain rates that are used
to advance the kinematic state. Supported control types are strain, strain
rate, stress, stress rate, deformation gradient, displacement, and velocity.
"Mixed-modes" of strain and stress (and their rates) are supported. Components
of displacement and velocity control are applied only to the "+" faces of a
unit cube centered at the coordinate origin.

.. _Figure 1:

.. figure:: ./images/path.png
   :width: 5in
   :align: center
   :figclass: align-center
    User defined path for the :math:`i^{\text{th}}` component of ":math:`c`".
    :math:`c` may represent strain, strain rate, stress, stress rate,
    deformation gradient, displacement, or velocity.

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

Each leg in the control table, from time :math:`t=0` to :math:`t=t_f` is
subdivided into a user-specified number of steps and the material model
evaluated at each step. When volumetric strain, deformation gradient,
displacement, or velocity are specified for a leg, *matmodlab* internally
determines the corresponding strain components. If a component of stress is
specified, *matmodlab* determines the strain increment that minimizes the
distance between the prescribed stress component and model response.

.. _sig2d:

Strain Rate from Prescribed Stress
----------------------------------

The approach to determining unknown components of the strain rate from the
prescribed stress is an iterative scheme employing a multidimensional Newton's
method that satisfies

.. math::

   \Stress\left(\dStrain\,[\text{v}]\right) = \PrescStress

where, :math:`\text{v}` is a vector subscript array containing the components
for which stresses (or stress rates) are prescribed, and :math:`\PrescStress`
are the components of prescribed stress.

Each iteration begins by determining the submatrix of the material stiffness

.. math::

   \Stiffness_{\text{v}} = \Stiffness\,[\text{v}, \text{v}]

where :math:`\Stiffness` is the full stiffness matrix
:math:`\Stiffness=d\Stress/d\Strain`. The value of
:math:`\dStrain\,[\text{v}]` is then updated according to

.. math::

   \dStrain\,[\text{v}] =
       \dStrain\,[\text{v}] -
       \Stiffness_{\text{v}}\DDotProd\Stress^{*}(\dStrain\,[\text{v}])/dt

where

.. math::

   \Stress^{*}(\dStrain\,[\text{v}]) = \Stress(\dStrain\,[\text{v}])
                                     - \PrescStress

The Newton procedure will converge for valid stress states. However, it is
possible to prescribe invalid stress state, e.g. a stress state beyond the
material's elastic limit. In these cases, the Newton procedure may not
converge to within the acceptable tolerance and a Nelder-Mead simplex method
is used as a back up procedure. A warning is logged in these cases.

.. _continuum_d:

Continuum Driver
----------------

As the name implies, the *Continuum* driver is designed to exercise the type
of material models encountered in continuum mechanics, with an emphasis on
solid materials. The solution method is similar to that of many finite element
codes, so that material models developed and tested in *matmodlab* can be
easily transitioned to them.

Electrical
----------

Electric field can be prescribed for testing piezoelectric models.
