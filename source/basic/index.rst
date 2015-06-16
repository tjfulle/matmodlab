
Matmodlab Execution
###################

A Matmodlab model is defined in a Python input script and is composed of
several different components that together describe the conditions under which
a material model is exercised. At a minimum, a model consists of the
following: a ``MaterialPointSimulator`` object, a material model and
associated data, and a description of the analysis steps. Optionally, material
data can be permutated and the model run under different conditions by a
``Permutator`` object, parameters can be optimized by an ``Optimizer`` object,
and the material can be exercised in conjunction with several add-on models.

The model is executed by using the ``mml`` command line utility.  The ``mml`` command is described in :ref:`basic_cli`.

The process flow of Matmodlab job execution is

.. figure:: ../images/flow.png


.. toctree::
   :hidden:

   cli
   execution
   output
   post
   permutate
   optimize
   environment
   examples
