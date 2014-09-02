! to create the plastic f2py interface:
! f2py -h plastic.pyf --overwrite-signature -m plastic plastic_interface.f90
subroutine plastic_check(ui)
  use plastic, only: pchk
  implicit none
  integer, parameter :: rk=selected_real_kind(14)
  integer, parameter :: nui=4
  real(kind=rk), intent(inout) :: ui(nui)
  call pchk(ui)
  return
end subroutine plastic_check
subroutine plastic_update_state(dt, ui, d, stress)
  use plastic, only: pcalc
  integer, parameter :: rk=selected_real_kind(14)
  integer, parameter :: nui=4
  real(kind=rk), intent(in) :: dt, ui(nui), d(6)
  real(kind=rk), intent(inout) :: stress(6)
  call pcalc(dt, ui, d, stress)
  return
end subroutine plastic_update_state
