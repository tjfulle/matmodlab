
.. _user_config:

User Configuration
==================

User's can get/set/delete *matmodlab* configuration options through the ``mml
config`` script.

Usage
-----

::

  usage: mml config [-h] [--add name [value[s] ...]] [--del name [value[s] ...]]
                    [--old2new] [--cat]

  mml config: Set matmodlab options

  optional arguments:
    -h, --help            show this help message and exit
    --add name [value[s] ...]
                          name and value of option to add to configuration
    --del name [value[s] ...]
                          name and value of option to remove from configuration
    --old2new             switch from old MMLMTLS environment variable to new
                          config file
    --cat                 print the MATMODLABRC file to the console and exit


Adding Options
--------------

::

  mml config --add option value

Deleting Options
----------------

::

  mml config --del option value

Ignoring Options
----------------

The ``-E`` flag to ``mml run`` suppresses use of configuration file.

Usefule Configurations
----------------------

*matmodlab* must be configured to be made aware of materials and tests that
reside outside of *matmodlab*. Use ``mml config`` to add directories to find
these materials and tests::

  mml config --add materials path/to/material
  mml config --add tests path/to/tests

On completion of the preceding commands, *matmodlab* will treat materials and
tests in ``path/to/material`` and ``path/to/tests``, respectively, as built in
materials and tests.

Example
-------

Set the sqa flag.  Useful if you want to run in sqa mode (sqa mode runs several software quality assurance checks during a simulation)::

   mml config --add sqa true

If you would like to delete the flag::

   mml config --del sqa
