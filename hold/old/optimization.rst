############
Optimization
############

*Payette* has the ability to assist in the parameterization of material models
through an optimization process whereby optimal parameters are found by
minimizing errors on selected simulation output. Minimization is performed
using *SciPy's* minimization module. Below is a summary of the optimization
block.

Optimization Block
==================

Input for the optimization problem is done in the ``optimization`` block. We
specify every available option showing its default value in brackets, along
with other available options in braces, if applicable.

Below, we specify the optimization method and options. Thus far, all of the
optimization methods are from ``scipy.optimize`` module.

Parameters to be optimized, with optional upper and lower bounds, must also be
specified. Many optimization methods do not accept bounds, but we have
implemented a penalty method to force the bounds to be respected independent
of the optimizer. For elastic moduli, bounds really should be used so the the
optimizer does not send in invalid moduli that would crash the simulation.

In addition to specifying the parameters to be optimized, simulation output to
be minimized must be specified along with a "gold file" that contains results
against which we compare the output from the simulation. The gold file must be
columnar data with column names in the first row. The optimization algorithm
optimizes the parameters specified above by minimizing the difference between
the minimizing variables in the gold file and the simulation output file.

Example Optimization Block
--------------------------

::

  begin optimization

    method simplex  # optizing method [simplex] {simplex, powell, cobyla}
    maxiter 25  # maximum number of iterations [25]
    tolerance 1.e-4  # tolerance between out and gold file [1.e-4]
    disp 0 # set to not zero to get detailed output from optimizer [0]

    optimize K, bounds = (125.e9, 150.e9) # parameter to optimize
    optimize G, bounds = (45.e9, 57.e9)  # parameter to optimize

    gold file exmpls.gold
    minimize sig11, sig22, sig33 versus time

  end optimization

The full example can be found in :download:`exmpl_7.inp
<./exmpl_7.inp>`.

Options
-------

**optimize** <param>, [bounds] (*required*)
  Parameter[s] to be minimized. Optional bounds are the lower and bounds on
  ``param``.

**minimize** <param_1[, param_2[,...param_n]]> [versus param] (*required*)
  Variables[s] to be minimized.  The optional *versus* flag does something.

**gold file** <file name> (*required*)
  File against which to compare simulation output

**method** (*optional*, default: simplex)
  Optimization method.  Available methods: simplex, powell, cobyla

**maxiter** (*optional*, default: 25)
  Maximum number of iterations.

**tolerance** (*optional*, default: 1.e-4)
  Tolerance between out and gold file.

**disp** (*optional*, default: 0)
  Set to not zero to get detailed output from optimizer.


Optimization Methods
--------------------

simplex
  Nelder-Meid simplex method [default], ``scipy.optimize.fmin``

powell
  Powell method, ``scipy.optimize.fmin_powell``

cobyla
  Constrain optimization by linear approximation, ``scipy.optimize.fmin_cobyla``


Limitations
===========

Known limitations

* Comparison between data and gold file limited

* Documentation
