.. _installing:

########################
Installing New Materials
########################

*Payette* was born from the need for an environment in which material models
could be rapidly developed, tested, deployed, and maintained, independent of
host finite element code implementation. Important prerequisites in the design
of *Payette* were ease of model installation and support for constitutive
routines written in Fortran. This requirement is met by providing a simple API
with which a model developer can install a material in *Payette* as a new
Python class and use `f2py <http://www.scipy.org/F2py>`_ to compile Python
extension modules from the material's Fortran source (if applicable). In this
section, the required elements of the constitutive model interface and
instructions on compiling the material's Fortran source (if applicable) are
provided.


Build Process
=============

The ``buildPayette`` script, described in :ref:`installation`, scans the
``PAYETTE_ROOT/Source/Materials`` directory for a materials xml control file.
The xml control file contains directives informing *Payette* the locations of
the following files: the build module and the interface module.

.. note::

   *Payette* is an open source project, without restrictions on how the source
   code is used and/or distributed. However, many material models have
   restrictive access controls and cannot, therefore, be included in
   *Payette*. This is the case for many materials currently being developed
   with *Payette*. For these materials, *Payette* provides two methods for
   extending the list of directories it scans for material interface modules:

   #) ``PAYETTE_MTLDIR`` environment variable. A colon separated list of
      directories containing *Payette* interface modules.

   #) ``--mtldir`` :file:`configure.py` option. At the configuration step,
      directories containing *Payette* interface modules can be passed to
      *Payette* through the ``--mtldir=/path/to/material/directory`` option.


Naming Convention
=================

For default materials included with *Payette*, the following naming
convention has been adopted for a material "material model":

XML control file
  :file:`PAYETTE_ROOT/Materials/Models/MaterialModel/MaterialModel_control.xml`

Interface file
  :file:`PAYETTE_ROOT/Materials/Models/MaterialModel/Payette_material_model.py`

Build script
  :file:`PAYETTE_ROOT/Materials/Models/MaterialModel/Build_material_model.py`

Fortran source
  :file:`PAYETTE_ROOT/Materials/Models/MaterialModel/material_model.f`


Control File Format
===================
The XML control file must provide the following nodes

::

  <MaterialModel>
    <Name>ModelName</Name>

    <Type> Material type [mechanical, eos, electromechanical] </Type>

    <Description>
      Material model description
    </Description>

    <Owner>Owner Name (owner@domain.com)</Owner>

    <Files>
      <Core>
        material_model.f90
      </Core>
      <Interface type="payette">
        Build_material_model.py
        Payette_material_model.py
        Payette_material_model.pyf
      </Interface>
    </Files>

    <Distribution>
      distribution level [uur, unlimited]
    </Distribution>

    <ModelParameters>
      <Key>material_model</Key>
      <Aliases>material_model_aliases</Aliases>

      <Units>parameter_units</Units>

      <Parameter name="PARAM_0" order="0"  type="double" default="0" units="UNITS">
        Description of PARAM_0
      </Parameter>
          .
          .
          .
      <Parameter name="PARAM_N" order="N" type="double" default="0" units="UNITS">
        Description of PARAM_N
      </Parameter>

      <Material name="MATERIAL_NAME" dist="dist_level" PARAM_0="value" ... PARAM_N="value" aliases="any_aliases"/>

    </ModelParameters>

  </MaterialModel>

Material Interface Module
=========================

Each material model must provide an interface module used by *Payette* to
interact with that material. The interface module must provide a material
class derived from the ``ConstitutiveModelPrototype`` base class.

Material Class
--------------

*Payette* provides a simple interface for interacting with material models
through the Python class structure. Material models are installed as separate
Python classes, derived from the ``ConstitutiveModelPrototype`` base class.


Inheritance From Base Class
"""""""""""""""""""""""""""

A new material model "material model" is only recognized as a material model
by Payette if it inherits from the ``ConstitutiveModelPrototype`` base class::

  class MaterialModel(ConstitutiveModelPrototype):


The ``ConstitutiveModelPrototype`` base class provides several methods in its
API for material models to communicate with *Payette*. Minimally, the material
model must provide the following data: ``aliases``, ``bulk_modulus``,
``imported``, ``name``, ``nprop``, and ``shear_modulus``, and methods:
``__init__``, ``set_up``, and ``update_state``.

Required Data
"""""""""""""

.. data:: MaterialModel.aliases

   List. The aliases by which the constitutive model can be called (case
   insensitive).

.. data:: MaterialModel.bulk_modulus

   Float. The bulk modulus. Used for determining the material's Jacobian matrix

.. data:: MaterialModel.imported

   Boolean. Indicator of whether the material's extension library (if
   applicable) was imported.

.. data:: MaterialModel.name

   String. The name by which users can invoke the constituve model from the
   input file (case insensitive).

.. data:: MaterialModel.nprop

   Int. The number of required parameters for the model.

.. data:: MaterialModel.shear_modulus

   Float. The shear modulus. Used for determining the material's Jacobian
   matrix


Required Functions
------------------

``__init__(self)``

   Instantiate the material model. Register parameters with *Payette*.
   Parameters are registered by the ``register_parameter`` method

   ::

     register_parameter(self, name, ui_loc, aliases=[])
         """Register the parameter name with Payette.

         ui_loc is the integer location (starting at 0) of the parameter in
         the material's user input array. aliases are aliases by which the
         parameter can be specified in the input file."""

   Alternatively, the ``register_parameters_from_control_file()`` method can
   be called and parameters from the control file will be registered
   automatically.

``set_up(self, simdat, matdat, user_params, f_params)``

   Check user inputs and register extra variables with *Payette*. *simdat* and
   *matdat* are the simulation and material data containers, respectively,
   *user_params* are the parameters read in from the input file, and *f_params*
   are parameters from a parameters file.

``update_state(self, simdat, matdat)``

   Update the material state to the end of the current time step. *simdat* and
   *matdat* are the simulation and material data containers, respectively.


Example: Elastic Material Model Interface File
==============================================

The required elements of the material's interface file described above are now
demonstrated by an annotated version of the elastic material's interface.

**View the source code:** :download:`Payette_elastic.py
<./Payette_elastic.py>`

::

  import sys
  from numpy import array

  from Source.Payette_utils import log_warning, log_message, report_and_raise_error
  from Source.Payette_tensor import iso, dev
  from Source.Payette_constitutive_model import ConstitutiveModelPrototype
  from Payette_config import PC_F2PY_CALLBACK
  from Toolset.elastic_conversion import compute_elastic_constants

  try:
      import Source.Materials.Library.elastic as mtllib
      imported = True
  except:
      imported = False
      pass


  class Elastic(ConstitutiveModelPrototype):
      """ Elasticity model. """

      def __init__(self, control_file, *args, **kwargs):
          super(Elastic, self).__init__(control_file, *args, **kwargs)

          self.imported = True if self.code == "python" else imported

          # register parameters
          self.register_parameters_from_control_file()

          pass

      # public methods
      def set_up(self, matdat):

          # parse parameters
          self.parse_parameters()

          # the elastic model only needs the bulk and shear modulus, but the
          # user could have specified any one of the many elastic moduli.
          # Convert them and get just the bulk and shear modulus
          eui = compute_elastic_constants(*self.ui0[0:12])
          for key, val in eui.items():
              if key.upper() not in self.parameter_table:
                  continue
              idx = self.parameter_table[key.upper()]["ui pos"]
              self.ui0[idx] = val

          # Payette wants ui to be the same length as ui0, but we don't want to
          # work with the entire ui, so we only pick out what we want
          mu, k = self.ui0[1], self.ui0[4]
          self.ui = self.ui0
          mui = array([k, mu])

          self.bulk_modulus, self.shear_modulus = k, mu

          if self.code == "python":
              self.mui = self._py_set_up(mui)
          else:
              self.mui = self._fort_set_up(mui)

          return

      def jacobian(self, simdat, matdat):
          v = matdat.get_data("prescribed stress components")
          return self.J0[[[x] for x in v],v]

      def update_state(self, simdat, matdat):
          """
             update the material state based on current state and strain increment
          """
          # get passed arguments
          dt = simdat.get_data("time step")
          d = matdat.get_data("rate of deformation")
          sigold = matdat.get_data("stress")

          if self.code == "python":
              sig = _py_update_state(self.mui, dt, d, sigold)

          else:
              a = [1, dt, self.mui, sigold, d]
              if PC_F2PY_CALLBACK:
                  a.extend([report_and_raise_error, log_message])
              sig = mtllib.elast_calc(*a)

          # store updated data
          matdat.store_data("stress", sig)

      def _py_set_up(self, mui):

          k, mu = mui

          if k <= 0.:
              report_and_raise_error("Bulk modulus K must be positive")

          if mu <= 0.:
              report_and_raise_error("Shear modulus MU must be positive")

          # poisson's ratio
          nu = (3. * k - 2 * mu) / (6 * k + 2 * mu)
          if nu < 0.:
              log_warning("negative Poisson's ratio")

          ui = array([k, mu])

          return ui

      def _fort_set_up(self, mui):
          props = array(mui)
          a = [props]
          if PC_F2PY_CALLBACK:
              a .extend([report_and_raise_error, log_message])
          ui = mtllib.elast_chk(*a)
          return ui


  def _py_update_state(ui, dt, d, sigold):

      # strain increment
      de = d * dt

      # user properties
      k, mu = ui
      twomu = 2. * mu
      threek = 3. * k

      # elastic stress update
      return sigold + threek * iso(de) + twomu * dev(de)


Building Material Fortran Extension Modules in *Payette*
==========================================================

.. note::

   This is not an exhaustive tutorial for how to link Python programs with
   compiled source code. Instead, it demonstrates through an annotated example
   the strategy that *Payette* uses to build and link with material models
   written in Fortran.

The strategy used in *Payette* to build and link to material models written in
Fortran is to use *f2py* to compile the Fortran source in to a shared object
library recognized by Python. The same task can be accomplished through Python's
built in `ctypes <http://docs.python.org/library/ctypes.html>`_, `weave
<http://www.scipy.org/Weave>`_\, or other methods. We have found that *f2py*
offers the most robust and easy to use solution. For more detailed examples of
how to use compiled libraries with Python see `Using Python as glue
<http://docs.scipy.org/doc/numpy/user/c-info.python-as-glue.html>`_ at the SciPy
website or `Using Compiled Code Interactively
<http://www.sagemath.org/doc/numerical_sage/using_compiled_code_iteractively.html>`_
on Sage's website.

Rather than provide an exhaustive tutorial on linking Python programs to compiled
libraries, we demonstrate how the ``elastic`` material model accomplishes this
task through annotated examples.


Creating the Elastic Material Signature File
--------------------------------------------

First, a Python signature file for the ``elatic`` material's Fortran source must
be created. A signature file is a Fortran 90 file that contains all of the
information that is needed to construct Python bindings to Fortran (or C)
functions.

For the elastic model, change to
:file:`PAYETTE_ROOT/Source/Materials/Fortran` and execute

::

  % f2py -m elastic -h Payette_elastic.pyf elastic.F

which will create the :file:`Payette_elastic.pyf` signature file.

f2py will create a signature for every function in :file:`elastic.F`. However,
only three public functions need to be bound to our Python program. So, after
creating the signature file, all of the signatures for the private functions can
safely be removed.

The signature file can be modified even further. See the above links on how to
specialize your signature file for maximum speed and efficiency.

**View the Payette_elastic.pyf file:** :download:`Payette_elastic.pyf
<./Payette_elastic.pyf>`


Elastic Material Build Script
-----------------------------

Materials are built by *f2py* through the ``MaterialBuilder`` class from which
each material derives its ``Build`` class. The ``Build`` class must provide a
``build_extension_module`` function, as shown below in the elastic material's
build script.

**View the elastic material build script:** :download:`Build_elastic.py
<./Build_elastic.py>`

::

  import os,sys

  from Payette_config import *
  from Source.Payette_utils import BuildError
  from Source.Materials.Payette_build_material import MaterialBuilder

  class Build(MaterialBuilder):

      def __init__(self, name, libname, compiler_info):

          fdir,fnam = os.path.split(os.path.realpath(__file__))
          self.fdir, self.fnam = fdir, fnam

          # initialize base class
          srcd = os.path.join(fdir, "Fortran")
          sigf = os.path.join(fdir, "Payette_elastic.pyf")
          MaterialBuilder.__init__(
              self, name, libname, srcd, compiler_info, sigf=sigf)


      def build_extension_module(self):

          # fortran files
          srcs = ["elastic.F"]
          self.source_files = [os.path.join(self.source_directory, x)
          for x in srcs]
          self.build_extension_module_with_f2py()

          return 0

.. note::

   For the elastic material, the ``build_extension_module`` function defines the
   Fortran source files and the calls the base class's
   ``build_extension_module_with_f2py`` function.
