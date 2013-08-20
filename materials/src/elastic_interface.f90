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
subroutine elastic_jacobian(dt, ui, nv, v, sig, d, Jsub)
  ! ------------------------------------------------------------------------- !
  ! Numerically compute material Jacobian by a centered difference scheme.
  ! ------------------------------------------------------------------------- !
  !......................................................................passed
  implicit none
  integer, parameter :: nui=2
  real(8), intent(in) :: dt
  real(8), intent(in) :: ui(nui)
  integer, intent(in) :: nv
  integer, intent(in) :: v(nv)
  real(8), intent(in) :: sig(6)
  real(8), intent(in) :: d(6)
  real(8), intent(out) :: Jsub(nv, nv)
  !.......................................................................local
  integer :: n
  real(8) :: sigp(6), sigm(6), dp(6), dm(6), deps
  ! ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ elastic_jacobian
  deps = sqrt(epsilon(d))
  do n = 1, nv
     dp = d
     dp(v(n)) = d(v(n)) + (deps / dt) / 2
     sigp = sig
     call ecalc(dt, ui, dp, sigp)
     dm = d
     dm(v(n)) = d(v(n)) - (deps / dt) / 2
     sigm = sig
     call ecalc(dt, ui, dm, sigm)
     Jsub(:, n) = (sigp(v) - sigm(v)) / deps
  end do
end subroutine elastic_jacobian
