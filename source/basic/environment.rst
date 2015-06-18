
.. _basic_config:

Environment Settings
####################

Overview
========

The Matmodlab execution can be customized with Matmodlab user environment file ``mml_userenv.py``.  Matmodlab looks for this file in your home directory, the location specified by the environment variable ``MML_USERENV``, and the current working directory, in that order.  Matmodlab will read each file if found, meaning settings in the current working will overwrite similar settings previously read.

References
==========

* :ref:`invoke_user_f`

Recognized Environment Settings and Defaults
============================================

Below are the recognized environment settings and their defaults.  Any of these settings can be changed by specifying a different value in a user environment file.

.. note::

   When specifying environment settings in a user environment file, the
   setting must have the same type as the default. If the default is a list,
   the user setting is inserted in the list. If the default is a dictionary,
   it is updated with the user setting.

IO Settings
-----------

::

   verbosity = 1
   warn = "std"
   Wall = False
   Werror = False
   Wlimit = 10

Debugging and SQA
-----------------

::

   raise_e = False
   sqa = False
   debug = False
   sqa_stiff = False

Performance
-----------

::

   nprocs = 1

Material Switching
------------------

::

   switch = []

User Material Models
--------------------

::

    materials = {}
    std_materials = [MAT_D]

Simulation Directory
--------------------

::

   simulation_dir = os.getcwd()

A Note on Defining User Material Models
=======================================

The ``materials`` and ``std_materials`` user settings are used to inform Matmodlab concerning user defined materials.  The ``std_materials`` is a list of python interface files for standard models.  The ``materials`` dictionary is a dictionary of ``model_name: attribute_dict`` key:value pairs with the dictionary of model attributes containing the following information:

* *source_files*: [list, required] A list of model source files
* *model*: [string, optional] The model type.  One of user, umat, uhyper, uanisohyper.  The default is user.
* *behavior*: [string, optional] The model behavior, one of mechanical, hperelastic, anisohyper.  The default is mechanical.
* *source_directory*: [string, optional] Directory to find source files. Useful for defining files in *source_files* relative to *source_directory*.
* *ordering*: [list of int, optional] Symmetric tensor ordering.  The default is XX, YY, ZZ, XY, YZ, XZ
* *user_ics*: [boolean, optional] Does the model provide its own SDVINI

Example
-------

The following user environment file is found in ``matmodlab/examples`` and is used by ``examples/users.py`` to define the material model's attributes::

  materials = {'neohooke': {'model': 'user', 'behavior': 'hyperelastic',
                            'source_directory': ROOT_D + '/materials/abaumats',
                            'source_files': ['uhyper.f90'],
                            'ordering': [XX, YY, ZZ, XY, XZ, YZ]}}
