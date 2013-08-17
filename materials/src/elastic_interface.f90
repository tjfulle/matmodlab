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
subroutine elastic_stiff(dt, ui, d, stress, J)
  use elastic, only: estiff
  integer, parameter :: rk=selected_real_kind(14)
  integer, parameter :: nui=2
  real(kind=rk), intent(in) :: dt, ui(nui), d(6)
  real(kind=rk), intent(in) :: stress(6)
  real(kind=rk), intent(out) :: J(6, 6)
  call estiff(dt, ui, d, stress, J)
  return
end subroutine elastic_stiff
