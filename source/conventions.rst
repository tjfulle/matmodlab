.. _Conventions:

Conventions
###########

Dimension
=========

Material models are always called with full 3D tensors.

Vector Storage
==============

Vector components are stored as

.. math::

   \Tensor{v}{}{}{} = \{v_x, v_y, v_z\}

Tensor Storage
==============

In general, second-order symmetric tensors are stored as 6x1 arrays with the
following ordering

.. math::
   :label: order-symtens

   \AA = \{A_{xx}, A_{yy}, A_{zz}, A_{xy}, A_{yz}, A_{xz}\}

Tensor components are used for all second-order symmetric tensors.

Nonsymmetric, Second-order tensors are stored as 9x1 arrays in row major
ordering, i.e.,

.. math::

   \BB = \{A_{xx}, A_{xy}, A_{xz},
           A_{yx}, A_{yy}, A_{yz},
           A_{zx}, A_{zy}, A_{zz}\}


Abaqus Materials
----------------

For Abaqus materials, the order of the last two components of second-order
tensors are modified when the material is called to be consistent with
Abaqus/Standard.   i.e., the second-order tensor in :eq:`order-symtens` is
passed to an Abaqus material as

.. math::

   \AA = \{A_{xx}, A_{yy}, A_{zz}, A_{xy}, A_{xz}, A_{yz}\}

Also consistent with Abaqus conventions, the shear components of strain-like
tensors are sent to the material model as engineering strains, i.e.

.. math::

   \Strain = \{\epsilon_{xx}, \epsilon_{yy}, \epsilon_{zz}, 2\epsilon_{xy}, 2\epsilon_{xz}, 2\epsilon_{yz}\}
           = \{\epsilon_{xx}, \epsilon_{yy}, \epsilon_{zz}, \gamma_{xy}, \gamma_{xz}, \gamma_{yz}\}

Nonsymmetric, Second-order tensors are sent as 3x3 matrices.

matmodlab Namespace
===================

Input scripts to *matmodlab* should include::

   from matmodlab import *

to populate the script's namespace with *matmodlab* specific parameters and methods.

Parameters
----------

Some useful parameters exposed by importing ``matmodlab`` are

* ``ROOT_D``, The root ``matmodlab`` directory
* ``PKG_D``, The ``matmodlab/lib`` directory, the location shared objects are copied
* ``EXO_D``, The directory where the ExodusII tools are contained
* ``MAT_D``, The directory where builtin materials are contained

Methods
-------

Some useful methods exposed by importing ``matmodlab`` are

* ``Driver``, The driver factory method
* ``Material``, The material model factory method
* ``MaterialPointSimulator``, The material point simulator constructor
* ``Permutator``, The permutator constructor
* ``Optimizer``, The optimizer constructor
* ``Logger``, The logger factory method.
* ``Expansion``, The expansion model constructor
* ``TRS``, The time-temperature shift model constructor
* ``Viscoelastic``, The viscoelastic model constructor

Each of these methods is described in more detail in the following chapters
and sections.
