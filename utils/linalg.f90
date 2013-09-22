module linalg
  implicit none
  real(kind=8), parameter :: half=5.e-01_8, zero=0.e+00_8
  real(kind=8), parameter :: one=1.e+00_8, two=2.e+00_8
contains
  function det9(a)
    ! ----------------------------------------------------------------------- !
    ! determinant of 3x3 array stored as 9x1, row major ordering assumed
    ! ----------------------------------------------------------------------- !
    real(kind=8) :: det9
    real(kind=8), intent(in) :: a(9)
    det9 = a(1) * (a(5) * a(9) - a(6) * a(8)) &
         + a(2) * (a(6) * a(7) - a(4) * a(9)) &
         + a(3) * (a(4) * a(8) - a(5) * a(7))
  end function det9
  function det6(a)
    ! ----------------------------------------------------------------------- !
    ! determinant of second order symmetric tensor stored as 6x1 array
    ! ----------------------------------------------------------------------- !
    real(kind=8) :: det6
    real(kind=8), intent(in) :: a(6)
    det6 = a(1) * a(2) * a(3) + two * a(4) * a(5) * a(6) &
         - (a(1) * a(5) * a(5) + a(2) * a(6) * a(6) + a(3) * a(4) * a(4))
  end function det6
  function dot(a, b)
    ! ----------------------------------------------------------------------- !
    ! determinant of second order symmetric tensor stored as 6x1 array
    ! ----------------------------------------------------------------------- !
    real(kind=8) :: dot(3, 3)
    real(kind=8), intent(in) :: a(3,3), b(3,3)
    dot = matmul(a, b)
  end function dot
  subroutine expm(a, e)
    ! exp(A) = I + A + 1/2 A^2 + 1/3! A^3 + ...
    real(kind=8), intent(in) :: a(3, 3)
    real(kind=8), intent(out) :: e(3, 3)
    real(kind=8) :: f(3, 3)
    integer :: n, k
    e = zero
    f = zero
    f(1, 1) = one
    f(2, 2) = one
    f(3, 3) = one
    k = 1
    do n = 1, 3
       if(insignificant(e, f)) then
          exit
       end if
       e = e + f
       f = matmul(a, f) / real(k, kind=8)
       k = k + 1
    end do
    return
  end subroutine expm
  function insignificant(r, s)
    logical insignificant
    real(kind=8), intent(in) :: r(3,3), s(3,3)
    logical value
    integer :: i, j
    real(kind=8) :: t, tol
    value = .true.
    do j = 1, 3
       do i = 1, 3
          t = r(i,j) + s(i,j)
          tol = epsilon( r(i,j)) * abs( r(i,j))
          if( tol < abs( r(i,j) - t)) then
             value = .false.
             exit
          end if
       end do
    end do
    insignificant = value
    return
  end function insignificant
 !  subroutine logm(a)
!   A = asarray(A)
!     if len(A.shape)!=2:
!         raise ValueError("Non-matrix input to matrix function.")
!     T, Z = schur(A)
!     T, Z = rsf2csf(T,Z)
!     n,n = T.shape

!     R = np.zeros((n,n),T.dtype.char)
!     for j in range(n):
!         R[j,j] = sqrt(T[j,j])
!         for i in range(j-1,-1,-1):
!             s = 0
!             for k in range(i+1,j):
!                 s = s + R[i,k]*R[k,j]
!             R[i,j] = (T[i,j] - s)/(R[i,i] + R[j,j])

!     R, Z = all_mat(R,Z)
!     X = (Z * R * Z.H)

!     if disp:
!         nzeig = np.any(diag(T)==0)
!         if nzeig:
!             print("Matrix is singular and may not have a square root.")
!         return X.A
!     else:
!         arg2 = norm(X*X - A,'fro')**2 / norm(A,'fro')
!         return X.A, arg2
!   end subroutine logm(a)
end module linalg
