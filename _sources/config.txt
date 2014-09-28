
User Configuration
==================

User's can get/set/delete *matmodlab* configuration options through the ``mml
config`` script.

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

The ``-E`` flag suppresses use of configuration file.

Example
=======

Set the sqa flag.  Useful if you want to run in sqa mode (sqa mode runs several software quality assurance checks during a simulation)::

   mml config --add sqa true

If you would like to delete the flag::

   mml config --del sqa
