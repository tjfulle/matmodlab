.. _Conventions:

Conventions
===========

Dimension
---------

Material models are always called with full 3D tensors.

Vector Storage
--------------

Vector components are stored as

.. math::

   \Tensor{v}{}{}{} = \{v_x, v_y, v_z\}

Tensor Storage
--------------

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
~~~~~~~~~~~~~~~~

For Abaqus materials, the order of the last two components of second-order
tensors are modified when the material is called to be consistent with
Abaqus/Standard.   i.e., the second-order tensor in :eq:`order-symtens` is
passed to an Abaqus material as

.. math::

   \AA = \{A_{xx}, A_{yy}, A_{zz}, A_{xy}, A_{xz}, A_{yz}\}

Also consistent with Abaqus conventions, the shear components of strain-like tensors are sent to the material model as engineering strains.

Nonsymmetric, Second-order tensors are sent as 3x3 matrices.
