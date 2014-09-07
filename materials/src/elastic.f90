module elastic

  integer, parameter :: rk=selected_real_kind(14)

  ! parameter pointers
  integer, parameter :: nui=2
  integer, parameter :: ipk=1
  integer, parameter :: ipmu=2

  ! numbers
  real(kind=rk), parameter :: half=.5_rk
  real(kind=rk), parameter :: zero=0._rk, one=1._rk, two=2._rk
  real(kind=rk), parameter :: three=3._rk, six=6._rk, ten=10._rk
  real(kind=rk), parameter :: refeps=.01_rk
  real(kind=rk), parameter :: tol1=1.e-8_rk, tol2=1.e-6_rk
  real(kind=rk), parameter :: toor3=5.7735026918962584E-01
  real(kind=rk), parameter :: root2=1.4142135623730951E+00
  real(kind=rk), parameter :: root3=1.7320508075688772E+00

  ! tensors
  real(kind=rk), parameter :: delta(6) = (/one, one, one, zero, zero, zero/)
  real(kind=rk), parameter :: ez(6) = (/toor3, toor3, toor3, zero, zero, zero/)

contains

  subroutine echk(ui)
    ! ----------------------------------------------------------------------- !
    ! Check the validity of user inputs and set defaults.
    !
    ! In/Out
    ! ------
    ! ui : array
    !   User input
    !
    ! ----------------------------------------------------------------------- !
    implicit none
    !..................................................................parameters
    !......................................................................passed
    real(kind=rk), intent(inout) :: ui(*)
    !.......................................................................local
    character*4 iam
    parameter(iam='echk' )
    real(kind=rk) :: nu
    !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ plastic_chk
    ! pass parameters to local variables
    ! check elastic moduli, calculate defaults if necessary
    if(ui(ipk) <= zero) then
       call faterr(iam, "Bulk modulus K must be positive")
    end if
    if(ui(ipmu) <= zero) then
       call faterr(iam, "Shear modulus MU must be positive")
    end if
    nu = (three * ui(ipk) - two * ui(ipmu)) / (six * ui(ipk) + two * ui(ipmu))
    if (nu > half) call faterr(iam, "Poisson's ratio > .5")
    if (nu < -one) call faterr(iam, "Poisson's ratio < -1.")
    if(nu < zero) call logmes("#---- WARNING: negative Poisson's ratio")
    return
  end subroutine echk

  ! ************************************************************************* !

  subroutine ecalc(dt, ui, d, stress)
    ! ----------------------------------------------------------------------- !
    ! Hooke's law elasticity
    ! ----------------------------------------------------------------------- !
    implicit none
    ! ...................................................................passed
    real(kind=rk), intent(in) :: dt
    real(kind=rk), intent(in) :: ui(nui), d(6)
    real(kind=rk), intent(inout) :: stress(6)
    ! ....................................................................local
    real(kind=rk) :: de(6)
    ! ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ elastic_calc
    de = d * dt
    stress = stress + three * ui(ipk) * iso(de) + two * ui(ipmu) * dev(de)
    return
  end subroutine ecalc

  !***************************************************************************!

  function iso(a)
    ! ----------------------------------------------------------------------- !
    ! Isotropic part of second order tensor
    ! ----------------------------------------------------------------------- !
    implicit none
    real(kind=rk) :: iso(6)
    real(kind=rk), intent(in) :: a(6)
    iso = trace(a) * delta / three
    return
  end function iso

  !***************************************************************************!

  function trace(a)
    ! ----------------------------------------------------------------------- !
    ! Trace of second order tensor
    ! ----------------------------------------------------------------------- !
    implicit none
    real(kind=rk) :: trace
    real(kind=rk), intent(in) :: a(6)
    trace = sum(a(1:3))
    return
  end function trace

  !***************************************************************************!

  function dev(a)
    ! ----------------------------------------------------------------------- !
    ! Deviatoric part of second order tensor
    ! ----------------------------------------------------------------------- !
    implicit none
    real(kind=rk) :: dev(6)
    real(kind=rk), intent(in) :: a(6)
    dev = a - iso(a)
    return
  end function dev

end module elastic
! to create the elastic f2py interface:
! f2py -h elastic.pyf --overwrite-signature -m elastic elastic_interface.f90
subroutine elastic_check(ui)
  use elastic, only: echk
  implicit none
  integer, parameter :: rk=selected_real_kind(14)
  integer, parameter :: nui=2
  real(kind=rk), intent(inout) :: ui(nui)
  call echk(ui)
  return
end subroutine elastic_check
subroutine elastic_update_state(dt, ui, d, stress)
  use elastic, only: ecalc
  integer, parameter :: rk=selected_real_kind(14)
  integer, parameter :: nui=2
  real(kind=rk), intent(in) :: dt, ui(nui), d(6)
  real(kind=rk), intent(inout) :: stress(6)
  call ecalc(dt, ui, d, stress)
  return
end subroutine elastic_update_state
