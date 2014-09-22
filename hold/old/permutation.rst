###########
Permutation
###########

*Payette* has the ability to assist in understanding the sensitivity of a
material model to user inputs by running a batch of simulations using
permutated user inputs. Below is a summary of the permutation block.

Permutation Block
=================

The ``permutation`` block is used to run a batch of simulations that are very
similar to the base input given. The basic syntax is::

  begin permutation
    method zip
    permutate PARAM1 VAL1 VAL2 VAL3 ...
    permutate PARAM2 VAL4 VAL5 VAL6 ...
    ...
  end permutation

This would take the general input and spawn simulations where the value associated
with PARAM1 is replace by VAL1 and the value associated with PARAM2 is replaced by
VAL4, then another simulation where PARAM1->VAL2 and PARAM2->VAL5, etc.

Directories containing each individual simulation and its output are put into
a general directory called ``SIMNAME.perm`` with the individual simulations
being contained in subdirectories named ``job.X`` where the ``SIMNAME`` is the
simulation name defined in the first line of the input file (``begin
simulation SIMNAME``) and ``X`` is replaced by the index of the job being run.


Methods
-------

**zip** (default)
  Zip together ranges and/or sequences of permutated parameters. Length of
  each parameter's permutation list must be the same.

**combination**
  Combine ranges and/or sequences of permutated parameters in to all possible
  combinations.

Examples
========

Example 1
---------

Zip a range of bulk and shear moduli.

::

  begin permutation

    method zip

    permutate K, range = (125.e9, 150.e9, 10) # parameter to permutate
    permutate G, range = (45.e9, 57.e9, 10)  # parameter to permutate

  end permutation

Example 2
---------

Combine an unequal length range of bulk and shear moduli.

::

  begin permutation

    method combination

    permutate K, range = (125.e9, 150.e9, 5) # parameter to permutate
    permutate G, range = (45.e9, 57.e9, 10)  # parameter to permutate

  end permutation

Example 3
---------

Zip a sequence of bulk and shear moduli.

::

  begin permutation

    method zip

    permutate K, sequence = (125.e9, 145.e9, 150.e9)
    permutate G, sequence = (45.e9, 50.e9, 57.e9)

  end permutation

Example 4
---------

Combine sequence of bulk and shear moduli.

::

  begin permutation

    method combination

    permutate K, sequence = (125.e9, 150.e9)
    permutate G, sequence = (45.e9, 50.e9, 57.e9)

  end permutation


