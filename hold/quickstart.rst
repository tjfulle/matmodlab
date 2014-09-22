.. _quickstart:

*matmodlab* Quick Start Guide
#############################

This guide provides an outline for building and running *matmodlab*.

Build *matmodlab*.  See Chapter building_.

  * Download *matmodlab* and setup environment
  * ``cd path/to/matmodlab``
  * ``mml build``

Prepare Input
-------------

Inputs are Python scripts. See Chapters

  * Define the driver
  * Define the material.
  * Define simulator.

Run
---

  * ``mml run [options] runid.py``

See ``mml help run`` for a complete list of options

Postprocess
-----------

  * ``mml view filename.exo [filename_2.exo [... filename_n.exo]]``
  * ParaView also reads exodus files.
