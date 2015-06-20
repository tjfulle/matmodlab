.. _python_models:

Python User Material Interface
##############################

Overview
========

Material models written in Python are implemented as subclasses
``matmodlab.mmd.material.MaterialModel`` and are treated as builtin materials.

References
==========

* :ref:`Role of Material Model`
* :ref:`defining_a_material`
* :ref:`intro_conventions`
* :ref:`comm_w_matmodlab`

Invoking User Materials
=======================

User materials that subclass ``MaterialModel`` are invoked using *model="*name*", where *name* is the material model's name.

Required Attributes
===================

Material models that subclass ``MaterialModel`` must provide the following class attributes:

* *name*, as string defining the material's name.  Must be unique.
* *param_names*, a list of parameter names, in the order expected by the model.

Required Methods
================

.. method:: MaterialModel.setup(**kwargs)

   Sets up the material model and return a list of state dependent variable
   names and initial values. By the time that *setup* is called, the model
   parameters have been

   *kwargs* are optional keywords sent in to the model.

   *setup* must return *sdv_keys*, *sdv_vals*, *sdv_keys* being the list of
    state dependent variable names and *sdv_vals* being their initial values.
    Both should be consistent with the ordering expected by the material
    model.

.. method:: MaterialModel.update_state(time, dtime, temp, dtemp, energy, density, F0, F1, strain, dstrain, elec_field, user_field, stress, xtra, last=False)

   Update the the material state

   The following parameters are sent in for information and should not be
   updated:

   * *time* The time at the beginning of the time step
   * *dtime* Step time step size
   * *temp* The temperature at the beginning of the time step
   * *dtemp* Step temperature increment
   * *energy* The energy at the beginning of the time step
   * *density* The material density
   * *F0* The deformation gradient at the beginning of the time step
   * *F1* The deformation gradient at the beginning of the time step
   * *strain* The strain at the beginning of the time step
   * *dstrain* The strain increment over the step
   * *elec_field* The electric field at the end of the step
   * *user_field* The user defined field at the end of the step

   The following parameter are sent in for information and should be
   updated to the end of the step:

   * *stress* The stress at the beginning of the step
   * *xtra* The state dependent variables at the beginning of the step

   The following variables should be calculated:

   *stiff*, the 6x6 material stiffness

   *update_state* must return *stress*, *statev*, *stiff*
