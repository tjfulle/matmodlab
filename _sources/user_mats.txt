
.. _User_Mats:

Writing User Material Models
############################

*matmodlab* can find, build, and exercise the following types of user
materials:

* ``native`` material models.  See :ref:`native`.
* Abaqus ``umat`` material models.  See :ref:`abamods`.
* *matmodlab* ``user`` material models.

.. note:: The source code from user materials is never copied in to *matmodlab*.

.. _native:

Native Material Models
======================

``native`` material models behave as if they were builtin models of
*matmodlab*. *matmodlab* interacts with ``native`` through a material model
interface file that defines a material class (called ``NativeMaterial`` in the
following. The ``NativeMaterial`` must be a subclass of
``matmodlab/core/material/MaterialModel``. ``NativeMaterial`` defines

.. class:: NativeMaterial(MaterialModel)

   Class defining a native type user material

   .. data:: name

      (required) the name by which it is invoked in the ``Material`` factory method.

   .. data:: source_files

      (optional) - list of model (fortran) source files.

   .. data:: lapack

      (optional) - flag to use lapack when building the source files.

   .. method:: NativeMaterial.__init__()

      Initialize the material.

   .. method:: setup()

      Checks user input and requests allocation of storage for state dependent variables. By the time ``setup`` is called, the user input parameters have been parsed by the base ``MaterialModel`` class and are stored in the ``params`` attribute.  ``setup`` should check the goodness of the material parameters.

   .. method:: update_state(time, dtime, temp, dtemp, energy, density, F0, F1, strain, dstrain, elec_field, user_field, stress, xtra, last=False)

      Update the the material state

      :parameter time: The time at the beginning of the time step
      :type time: float
      :parameter dtime: Step time step size
      :type dtime: float
      :parameter temp: The temperature at the beginning of the time step
      :type temp: float
      :parameter dtemp: Step temperature increment
      :type dtemp: float
      :parameter energy: The energy at the beginning of the time step
      :type energy: float
      :parameter density: The material density
      :type density: float
      :parameter F0: The deformation gradient at the beginning of the time step
      :type time: ndarray
      :parameter F1: The deformation gradient at the beginning of the time step
      :type time: ndarray
      :parameter strain: The strain at the beginning of the time step
      :type time: ndarray
      :parameter dstrain: The strain increment over the step
      :type dstrain: ndarray
      :parameter elec_field: The electric field at the end of the step
      :type elec_field: ndarray
      :parameter user_field: The user defined field at the end of the step
      :type user_field: ndarray or None
      :parameter stress: The stress at the beginning of the step
      :type stress: ndarray
      :parameter xtra: The state dependent variables at the beginning of the step
      :type xtra: ndarray
      :returns: ``(stress, xtra, stiff)``. The stress, state dependent variables at the end of the step, and the 6x6 material stiffness

..
   Each class must define its name (\verb|Material.name|) and an ordered list of
   material parameter names (\verb|Material.param_names|) as they should appear
   in the input file. Parameter aliases are supported by specifying a parameter
   name as a ``:'' separated list of allowed names. The class should not define
   an \texttt{\uus{}init\uus{}} method and if it does, should call the
   \texttt{\uus{}init\uus{}} of the base class.

   \begin{interface}
     \textbf{Material: Interface}
   \end{interface}
   \usage{mtl = Material()}

   The following is an example of a \texttt{Material} declaration for the
   \texttt{Elastic} material model.  Aliases for \texttt{K} are noted.
   %
   \begin{example}
   from materials._material import Material
   class Elastic(Material)
       name = "elastic"
       param_names = ["K:BMOD:B0", "G"]
   \end{example}

   % ----------------------------------------------------------------------------- %
   \subsection{Setup the Material}
   \label{sec:setup-mtl}
   Each material must provide the method \texttt{setup} that sets up the material
   model by checking and adjusting the material parameter array, requesting
   allocation of storage of material variables, and computing and storing the
   \verb|bulk_modulus| and \verb|shear_modulus| of the material. \verb|setup| is
   called by the base class method \verb|setup_new_material| that parses and
   stores the user given parameters in the \verb|Material.params| array.

   \begin{interface}
     \textbf{Material.setup: Interface}
   \end{interface}
   \usage{mtl.setup()}\\[5pt]

   The following is an example of a \texttt{setup} method

   \begin{example}
   def setup(self):
       if elastic is None:
	   raise Error1("elastic model not imported")
       elastic.elastic_check(self.params, log_error, log_message)
       K, G = self.params
       self.bulk_modulus = K
       self.shear_modulus = G
   \end{example}

   % ----------------------------------------------------------------------------- %
   \subsection{Adjust the Initial State}
   \label{sec:adjust-istate}
   The method \texttt{adjust\us{}initial\us{}state} adjusts the initial state
   after the material is setup. Method provided by base class should be adequate
   for most materials. A material should only overide the base method if
   absolutely necessary.

   \begin{interface}
     \textbf{Material.adjust\us{}initial\us{}state: Interface}
   \end{interface}
   \usage{mtl.adjust\us{}initial\us{}state(xtra)}\\[5pt]
   \param{ndarray xtra}{Material variables}

   % ----------------------------------------------------------------------------- %
   \subsection{Update the Material State}
   \label{sec:update-state}
   The material state is updated to the end of the step via the
   \verb|update_state| method. Each material model must provide its own
   \verb|update_state| method.

   \begin{interface}
     \textbf{Material.update\us{}state: Interface}
   \end{interface}
   \usage{stress, xtra = mtl.update\us{}state(dt, d, sig, xtra,\\
     \indent\hspace{3.2in}f, ef, t, rho, tmpr, *args)}\\[5pt]
   \param{real dt}{timestep size}
   \param{ndarray d}{rate of deformation}
   \param{ndarray sig}{stress at beginning of step}
   \param{ndarray xtra}{extra state variables at beginning of step}
   \param{ndarray f}{deformation gradient at end of step}
   \param{ndarray ef}{electric field}
   \param{real t}{time}
   \param{real rho}{density at end of step}
   \param{real tmpr}{temperature at end of step}
   \param{tuple args}{extra args (not used)}
   \param{dict kwargs}{extra keyword args (not used)}
   \param{ndarray stress}{stress at end of step}
   \param{ndarray xtra}{extra state variables at end of step}

   The following code segment is used by the driver to update the material state
   \begin{example}
   args = []
   sig, xtra = mtl.update_state(dt, d, sig, xtra,
				f, ef, t, rho, tmpr, *args)
   \end{example}

   % ----------------------------------------------------------------------------- %
   \subsection{Example}
   \label{sec:update-state-ex}
   The following example demonstrates the implementation of a simple elastic
   model.
   \begin{example}
   import numpy as np
   from materials._material import Material
   from core.io import Error1, log_error, log_message
   try:
       import lib.elastic as elastic
   except ImportError:
       elastic = None

   class Elastic(Material):
       name = "elastic"
       param_names = ["K", "G"]
       def __init__(self):
	   super(Elastic, self).__init__()

       def setup(self):
	   if elastic is None:
	       raise Error1("elastic model not imported")
	   elastic.elastic_check(self.params, log_error, log_message)
	   K, G, = self.params
	   self.bulk_modulus = K
	   self.shear_modulus = G

       def update_state(self, dt, d, stress, xtra, *args):
	   elastic.elastic_update_state(dt, self.params, d, stress,
					log_error, log_message)
	   return stress, xtra

       def jacobian(self, dt, d, stress, xtra, v):
	   return self.constant_jacobian(v)
   \end{example}

   % ----------------------------------------------------------------------------- %
   \section{Building and Linking Materials}
   \label{sec:usrbld}
   *matmodlab* comes with and builds several builtin material models that are
   specified in \\
   \verb|$MMLROOT/materials/library/mmats.py|. User materials are
   found by looking in directories in the \verb|$MMLMTLS| environment variable
   for a single file \texttt{umat.py}. \texttt{umat.conf} communicates to *matmodlab*
   information needed to build the material's extension module.

   % ----------------------------------------------------------------------------- %
   \subsection{Building User Materials}
   \label{sec:bld-usr}
   User materials are built %
   \footnote{Only pure python and fortran models have been implemented.
     Implementing models in other languages is possible, but would have to be
     sorted out.}
   by *matmodlab* using \texttt{numpy}'s distutils.  A
   material communicates to *matmodlab* information required by distutils back to
   *matmodlab* through the \texttt{umat.conf} function.

   % --- makemf API
   \begin{interface}
   \textbf{umat.conf: Interface}
   \end{interface}
   \usage{name, info = conf(*args)}

   \param{tuple args}{not currently used}

   \param{str name}{The name of the material model}
   \param{dict info}{Information dictionary}

   % ----------------------------------------------------------------------------- %
   \subsection{The \texttt{info} Dictionary}
   \label{sec:infodict}
   The \texttt{info} dict contains the following keys

   \param{list source\us{}files}{The list of source files to be built.  If the
     material is a pure python module, specify as \texttt{None}}

   \param{str includ\us{}dir}{Directory to look for includes during compile
     [default: dirname(interface\us{}file)}

   \param{str interface\us{}file}{Path to the material's interface file}

   \param{str class}{The name of the material model class}


   Below is an example of \texttt{umat.conf}
   \begin{example}
   D = os.path.dirname(os.path.realpath(__file__))

   def conf(*args):
       name = "dsf"
       source_files = [os.path.join(D, f) for f in ("material.F", "material.pyf")]
       assert all(os.path.isfile(f) for f in source_files)
       info = {"source_files": source_files, "includ_dir": D,
	       "interface_file": os.path.join(D, "material.py"),
	       "class": "MaterialModel"}
       return name, info
   \end{example}

.. _abamods:

Abaqus Materials
================

*matmodlab* can build and exercise Abaqus ``UMAT`` and ``UHYPER`` material
models. *matmodlab* builds the Abaqus models and calls the ``UMAT`` and
``UHYPER`` procedures with the same calling arguments as Abaqus. ``UMAT`` and
``UHYPER`` materials use the same ``Material`` factory method as other
materials, but adds the following additional requirements:

* ``model="umat"`` or ``model="uhyper"``
* ``parameters`` must be a ndarray of model constants (specified in the order
  expected by the model).
* ``constants`` must be specified and the length of ``parameters`` and
  ``constants`` must be the same.
* ``depvar``, if specified, is the number of state dependent variables
  required for the model.
* ``source_files`` [optional] List of model source files.  If not specified, *matmodlab* will look for ``umat.[Ff](90)?`` in the current working directory.
* ``source_directory`` [optional] Directory containing source files.

.. note::
   Only one ``UMAT`` material can be run and exercised at a time.

.. note::
   *matmodlab* modifies the ``parameters`` array to have length ``constants`` + 1 and appends an extra paramter to its end. This extra parameter can be used as a debug flag.

*matmodlab* implements the following Abaqus utility functions:

* ``XIT``.  Stops calculations immediately.
* ``STDB_ABQERR``.  Message passing interface from material model to host code.

Consult the Abaqus documentation for more information.

Examples
========

Two parameter Neo-Hookean nonlinear elastic model implemented as a ``UMAT``.

.. code::

   E = 200
   nu = .333
   material = Material("umat", parameters=[E, nu], constants=2,
                        source_files=["neohooke.f90"], rebuild=test,
                        source_directory="{0}/abaumats".format(MAT_D))

Two parameter Neo-Hookean nonlinear elastic model implemented as a ``UHYPER``.

.. code::

   C10 = 200
   D1 = 1E-05
   material = Material("uhyper", parameters=[C10, D1], constants=2,
                       source_files=["uhyper.f90"],
                       source_directory="{0}/abaumats".format(MAT_D))
