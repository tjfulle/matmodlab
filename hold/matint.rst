.. _usrmtl:

User Material Interface
#######################

*matmodlab* can be made to find, build, and execute user materials outside of
``MML_ROOT``. User materials are written in Python and/or Fortran and
*matmodlab* interacts with them through the application programming interface
(API). In general, the following pattern is followed for exercising a material
model with *matmodlab*

* create a material model interface (MMI)
* build and link the material model to *matmodlab*
* exercise the model

.. _usrint:

Material Model Interface
========================

*matmodlab* interacts with materials through a material interface file. The
material interface file defines the material class which must be a subclass of
``MML_ROOT/core/material.MaterialModel``. In this section, methods of the
``MaterialModel`` base class are described.


.. _basecls:

Material Class Instantiation
----------------------------

The base class ``MaterialModel`` in ``MML_ROOT/core/material.py`` creates new
*matmodlab* materials and provides the interface with which *matmodlab*
interacts. The constructor for each material must define its name
(``Material.name``), an ordered list of material parameter names
(``Material.param_names``) as they should appear in the input file, and (for
Fortran models) a list of the material's source files
(``Material.source_files``).

.. code::

   import os
   from core.material import MaterialModel
   d = os.path.dirname(os.path.realpath(__file__))
   class Material(MaterialModel):
       """Material model for a material"""
       def __init__(self):
           self.name = "the_material"
           self.param_names = ["A", "B", "C"]
           self.source_files = [os.path.join(d, "source.f90"),
                                os.path.join(d, "source.pyf")]


Parameter aliases are supported by specifying a parameter name as a ":"
separated list of allowed names.

.. code::

   self.param_names = ["A:alias1:alias2", "B", "C"]

..
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
