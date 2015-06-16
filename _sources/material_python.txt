
Python User Material Interface
##############################

Matmodlab can find, build, and exercise the following types of user
materials:

* ``native`` material models.  See :ref:`native`.
* Abaqus ``umat`` material models. See :ref:`aba_mats`

.. note:: The source code from user materials is never copied in to Matmodlab.

.. _native:

Native Material Models
======================

``native`` material models behave as if they were builtin models of
Matmodlab. Matmodlab interacts with ``native`` through a material model
interface file that defines a material class (called ``NativeMaterial`` in the
following. The ``NativeMaterial`` must be a subclass of
``matmodlab/core/material/MaterialModel``. ``NativeMaterial`` defines

.. class:: NativeMaterial(MaterialModel)

   Class defining a native type user material

.. attribute:: NativeMaterial.name

   (required) the name by which it is invoked in the ``Material`` factory method.

.. attribute:: NativeMaterial.source_files

   (optional) - list of model (fortran) source files.

.. attribute:: NativeMaterial.lapack

   (optional) - flag to use lapack when building the source files.

.. method:: NativeMaterial.__init__()

   Initialize the material.

.. method:: NativeMaterial.setup()

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

Material Interface File
-----------------------

A material interface file is a python file containing material class
information. For fortran models, this file acts as a wrapper to the fortran
model procedures. Matmodlab finds materials by looking for material
interface files in ``matmodlab/materials`` and directories in the
``materials`` section of the configuration file. Material interface files are
python files whose names match ``(?:^|[\\b_\\.-])[Mm]at``. For Abaqus ``umat``
models, a material interface is not necessary.



.. _aba_umat_models:

Abaqus umat Material Models
===========================

See :ref:`aba_mats`

Building and Linking Materials
==============================

Matmodlab comes with and builds several builtin material models that are
contained ``matmodlab/materials``. User materials are found by looking for
material interface files whose names match ``(?:^|[\\b_\\.-])[Mm]at`` and that
contain a material class subclassing either ``MaterialModel`` or
``AbaqusMaterial``. Material models implemented in pure python require no
additional steps to linked to Matmodlab. Models implemented in Fortran will
need to be built by Matmodlab and are built using numpy's distutils and f2py.

.. note::

   Only pure python and Fortran models have been implemented.

Building Material Models Implemented in Fortran
-----------------------------------------------

Matmodlab must be able to find, compile, and link the Fortran source files.
For Abaqus umats, this is done by passing a list of source file names to the
``Material`` factory method, see :ref:`aba_mats`. Native materials communicate
information regarding source file locations through the ``source_files`` class
attribute.

f2py Signature File
...................

Signature files are hybrid fortran/python files generated by f2py that
communicate information about procedures contained in Fortran source files.
See the `Signature file documentation
<http://docs.scipy.org/doc/numpy-dev/f2py/signature-file.html>`_ for more
information.

Lapack
......

If a material requires lapack, set the ``lapack`` class attribute to
``'lite'`` for a stripped down version of lapack built by Matmodlab or
``True`` to link to the system's lapack. If set to ``True`` and distutils is
unable to find lapack on your system the material may still build, but will
fail at run time.

Communicating Information from Fortran Materials to the Logger
==============================================================

All materials are linked against a Matmodlab utility library containing the
following utility procedures. Utility procedures that communicate information
back to the Matmodlab logger must have implement callback functions in the
material's f2py signature file for the information to get back to Matmodlab.
See :ref:`sig_file` for an example of how to setup the callbacks.

logmes
------

``logmes`` communicates information to the simulation logger.

.. code:: fortran

   subroutine logmes(message)
     character*120, intent(in) :: message

bombed
------

``bombed`` communicates information to the simulation logger and ends the
simulation.

.. code:: fortran

   subroutine bombed(message)
     character*120, intent(in) :: message

faterr
------

``faterr`` communicates information to the simulation logger and ends the
simulation.

.. code:: fortran

   subroutine faterr(caller, message)
     character*8, intent(in) :: caller
     character*120, intent(in) :: message

Example
=======

The following is an elastic material model implemented as a Native material.

Interface File
--------------

``mat_hooke.py``::

  from core.material import MaterialModel
  mat = None

  class Elastic(MaterialModel):
      source_files = ["hooke.f90", "hooke.pyf"]

      def __init__(self):
          name = "hooke"
          self.param_names = ["E", "NU"]

      def setup(self):
          global mat
          try:
	      import lib.hooke as mat
          except ImportError:
              raise ModelNotImportedError("elastic")
          comm = (self.logger.write, self.logger.warn, self.logger.raise_error)
          mat.hooke_check(self.params, *comm)

      def hooke_update_state(self, time, dtime, temp, dtemp, energy,
              rho, F0, F, stran, d, elec_field, user_field, stress,
              xtra, **kwargs):
          comm = (self.logger.write, self.logger.warn, self.logger.raise_error)
	  ddsdde = np.zeros((6,6), order="F")
          mat.update_state(dtime, self.params, d, stress, ddsdde, *comm)
          return stress, xtra, ddsdde

Fortran Source File
-------------------

``hooke.f90``::

  subroutine hooke_check(nui, ui)
    implicit none
    integer, intent(in) :: nui
    real(8), intent(in) :: ui(nui)
    real(8) :: k, g, nu
    k = ui(1)
    g = ui(2)
    if (k < 0.0) call bombed("bulk modulus must be positive")
    if (g < 0.0) call bombed("shear modulus must be positive")
    nu = (3. * k - 2. * g) / (6. * k + 2. * g)
    if (nu > .5) call faterr(iam, "Poisson's ratio > .5")
    if (nu < -1.) call faterr(iam, "Poisson's ratio < -1.")
    if(nu < 0.) call logmes("#---- WARNING: negative Poisson's ratio")
  end subroutine hooke_check

  subroutine hooke_update_state(dtime, ui, d, stress, ddsdde)
    implicit none
    integer, intent(in) :: nui
    real(kind=rk), intent(in) :: dtime, ui(nui), d(6)
    real(kind=rk), intent(inout) :: stress(6), ddsdde(6,6)
    real(8) :: de(6), de_iso(6), de_dev(6), k, g, nu, c1, c2, c3
    k = ui(1)
    g = ui(2)
    de = d * dtime
    dstress = 0.; de_iso = 0.
    de_iso(1:3) = sum(de(1:3)) / 3.
    de_dev = de - de_iso
    dstress = 3. * k * de_iso + 2. * g * de_dev
    stress = stress + dstress

    ! Material stiffness
    ddsdde = 0.
    nu = (3. * k - 2. * g) / (2. * 3. * k + 2. * g)
    c1 = (1. - nu) / (1. + nu)
    c2 = nu / (1. + nu)

    ! set diagonal
    do i = 1, 3
      ddsdde(i, i) = 3. * k * c1
    end do
    do i = 3, 6
      ddsdde(i, i) = 2. * g
    end do

    ! off diagonal
    c3 = 3. * k * c2
                       ddsdde(1, 2) = c3; ddsdde(1, 3) = c3
    ddsdde(2, 1) = c3;                    ddsdde(1, 3) = c3
    ddsdde(3, 1) = c3; ddsdde(3, 2) = c3

  end subroutine hooke_update_state

.. _sig_file:

Signature File
--------------

Signature files are generated by f2py and modified to include the
``mml__user__routines`` module to pass information regarding the utility
routines from Matmodlab to Fortran procedures.

``hooke.pyf``::

  python module mml__user__routines
      interface mml_user_interface
          subroutine log_message(message)
              intent(callback) log_message
              character*(*) :: message
          end subroutine log_message
          subroutine log_warning(message)
              intent(callback) log_warning
              character*(*) :: message
          end subroutine log_warning
          subroutine log_error(message)
              intent(callback) log_error
              character*(*) :: message
          end subroutine log_error
      end interface mml_user_interface
  end python module mml__user__routines

  python module hooke ! in
  interface  ! in :hooke
     subroutine hooke_check(nui, ui)
       use mml__user__routines
       intent(callback) log_message
       external log_message
       intent(callback) log_warning
       external log_warning
       intent(callback) log_error
       external log_error
       integer, intent(in) :: nui
       real(kind=8) dimension(nui),intent(inout) :: ui
     end subroutine hooke_check
     subroutine hooke_update_state(dt,nui,ui,d,stress,ddsdde)
       use mml__user__routines
       intent(callback) log_message
       external log_message
       intent(callback) log_warning
       external log_warning
       intent(callback) log_error
       external log_error
       real(kind=8) intent(in) :: dt
       integer, intent(in) :: nui
       real(kind=8) dimension(nui),intent(in) :: ui
       real(kind=8) dimension(6),intent(in) :: d
       real(kind=8) dimension(6),intent(inout) :: stress
       real(kind=8) dimension(6,6),intent(inout) :: ddsdde
     end subroutine hooke_update_state
  end interface
  end python module hooke
