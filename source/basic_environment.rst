
.. _basic_config:

Environment Settings
####################

Overview
========

The Matmodlab execution can be customized with the Matmodlab rc file.  Matmodlab looks for this file in ``~/.matmodlabrc``.  See :ref:`basic_config` for more information on customizing the execution environment.

User's can get/set/delete Matmodlab configuration options through the ``mml
config`` script.

The mml config Procedure
========================

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

Common Environment Setting
==========================

materials
---------

Location of user defined materials.

::

  mml config --add materials path/to/material

sqa
---

Run extra SQA checks during procedure execution

::

  mml config --add sqa true
