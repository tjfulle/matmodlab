subroutine test_inv_3x3(a, ainv)
  use tensalg, only: inv
  real(8), intent(in) :: a(3,3)
  real(8), intent(out) :: ainv(3,3)
  ainv = inv(a)
  return
end subroutine test_inv_3x3

subroutine test_inv_6x1(a, ainv)
  use tensalg, only: inv
  real(8), intent(in) :: a(6)
  real(8), intent(out) :: ainv(6)
  ainv = inv(a)
  return
end subroutine test_inv_6x1
