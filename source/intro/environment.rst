
.. _environment:

Environment Settings
####################

.. topic:: References

   * :ref:`intro_conventions`
   * :ref:`basic_cli`
   * :ref:`python_models`
   * :ref:`fortran_models`

Overview
========

Matmodlab sets up and performs execution of input scripts in a customized
environment (not to be confused with Python virtural environment). The
environment can be modifed by user environment files that are read at the
beginning of each job.

Environment File Locations
==========================

Matmodlab searches for the optional user environment file, ``mml_userenv.py``, in three locations, in the following order:

1) Your home directory.
2) The location specified by the environment variable ``MML_USERENV``
3) The current working directory.

The value of a parameter is the the last definition encountered, meaning that the order of precedence for user settings is the current working directory, ``MML_USERENV``, and the home directory.

Environment files use Python syntax, meaning that entries will have the following syntax::

  parameter = value

All usual Python conventions apply.

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

verbosity

  Matmodlab will print more, or less, information during job execution. Possible values are 0, 1, and 2. Set the value to 0 to suppress printing of information. Set the value to 2, or higher, to print increase the amount of information printed. The default value is 1.

warn

  Define how Matmodlab is to interpret warnings.  Possible values are 'std', 'ignore', 'error'.  Set the value to 'ignore' to suppress warnings and to 'error' to treat warnings as errors.  The default 'std' prints warning messages.

Wlimit

  Define the number of warnings can be printed.  The default is ``10``.

Debugging and SQA
-----------------

raise_e

  By default, Matmodlab prints errors encountered and quits.  If raise_e is set to True, errors and not merely printed, but raised. The default is ``False``.

sqa

  Run additional SQA checks.  The default is ``False``.

debug

  Run additional debug code.  The default is ``False``.

sqa_stiff

  Perform checks of stiffness sent back from constitutive models.  The default is ``False``.

Performance
-----------

nprocs

  The number of simultaneous jobs to run.  The option is only used by the Matmodlab.Permutator.  The default is ``1``.


Material Switching
------------------

swith

  A list of (old, new) tuples specifying the model switching behavior.  The default is ``[]``.


User Material Models
--------------------

materials

  A dictionary describing user material models.  The dictionary consists of ``{model:information}`` key, value pairs. ``information`` is itself a dictionary containing material model meta data needed by Matmodlab.  The default is ``{}``.

std_materials

  A list containing directories and files to search for standard material models.  The default is ``[MAT_D]``.

A Note on Defining User Material Models
.......................................

``std_materials`` and ``materials``  user settings are used to inform Matmodlab concerning user defined materials.  ``std_materials`` is a list of python interface files for standard models.  The ``materials`` dictionary is a dictionary of ``model_name: attribute_dict`` key:value pairs with the dictionary of model attributes containing the following information:

* *source_files*: [list, required] A list of model source files
* *model*: [symbolic constant, optional] The model type.  One of USER, UMAT, UHYPER, UANISOHYPER.  The default is USER.
* *behavior*: [symbolic constant, optional] The model behavior, one of MECHANICAL, HPERELASTIC, ANISOHYPER.  The default is MECHANICAL.
* *source_directory*: [string, optional] Directory to find source files. Useful for defining files in *source_files* relative to *source_directory*.
* *ordering*: [list of symbolic constants, optional] Symmetric tensor ordering.  The default is XX, YY, ZZ, XY, YZ, XZ
* *user_ics*: [boolean, optional] Does the model provide its own SDVINI

Example
'''''''

The following is a portion of the user environment file found in ``matmodlab/examples`` and is used by ``examples/users.py`` to define the material model's attributes::

  materials = {'neohooke': {'model': 'user', 'behavior': 'hyperelastic',
                            'source_directory': ROOT_D + '/materials/abaumats',
                            'source_files': ['uhyper.f90'],
                            'ordering': [XX, YY, ZZ, XY, XZ, YZ]}}

Simulation Directory
--------------------

simulation_dir

  The directory to run the simulation.  The default is the current working directory.
