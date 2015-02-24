module mmlabpack
  implicit none
  integer, parameter :: nsymm=6, ntens=9
  real(kind=8), parameter :: zero=0.e+00_8, half=5.e-01_8
  real(kind=8), parameter :: one=1.e+00_8, two=2.e+00_8, three=3.e+00_8
  real(kind=8), parameter :: VOIGHT(6)=(/one,one,one,two,two,two/)

  interface expm
     module procedure expm_3x3, expm_6x1
  end interface expm

contains

  ! ------------------------------------------------------------------------- !
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

  ! ------------------------------------------------------------------------- !
  function det6(a)
    ! ----------------------------------------------------------------------- !
    ! determinant of second order symmetric tensor stored as 6x1 array
    ! ----------------------------------------------------------------------- !
    real(kind=8) :: det6
    real(kind=8), intent(in) :: a(6)
    det6 = a(1) * a(2) * a(3) + two * a(4) * a(5) * a(6) &
         - (a(1) * a(5) * a(5) + a(2) * a(6) * a(6) + a(3) * a(4) * a(4))
  end function det6

  ! ------------------------------------------------------------------------- !
  function dot(a, b)
    ! ----------------------------------------------------------------------- !
    ! determinant of second order symmetric tensor stored as 6x1 array
    ! ----------------------------------------------------------------------- !
    real(kind=8) :: dot(3, 3)
    real(kind=8), intent(in) :: a(3,3), b(3,3)
    dot = matmul(a, b)
  end function dot

  ! ------------------------------------------------------------------------- !
  function u2e(u, kappa)
    real(kind=8) :: u2e(6)
    real(kind=8), intent(in) :: u(3,3), kappa
    real(kind=8) :: eps(3,3)
    u2e = zero
    if (kappa /= zero) then
       eps = one / kappa * (powm(u, kappa) - one)
    else
       eps = logm(u)
    end if
    u2e = symarray(eps) * VOIGHT
  end function u2e

  ! ------------------------------------------------------------------------- !
  function symarray(a)
    real(kind=8) :: symarray(6)
    real(kind=8), intent(in) :: a(3,3)
    real(kind=8) :: x(3,3)
    x = half * (a + transpose(a))
    symarray(1) = x(1,1); symarray(4) = x(1,2); symarray(6) = x(1,3)
                          symarray(2) = x(2,2); symarray(5) = x(2,3)
                                                symarray(3) = x(3,3)
    return
  end function symarray

  ! ------------------------------------------------------------------------- !
  subroutine asarray(a, n, arr)
    integer, intent(in) :: n
    real(kind=8), intent(in) :: a(3,3)
    real(kind=8), intent(out) :: arr(n)
    arr = zero
    if (n == 6) then
       arr = symarray(a)
    else if (n == 9) then
       arr = reshape(a, shape(arr))
    end if
    return
  end subroutine asarray

  ! ------------------------------------------------------------------------- !
  ! Computes the matrix exponential
  ! ------------------------------------------------------------------------- !
  function expm_6x1(a)
    ! ----------------------------------------------------------------------- !
    real(kind=8) :: expm_6x1(6)
    real(kind=8), intent(in) :: a(6)
    real(kind=8) :: b(3,3),expb(3,3)
    ! ----------------------------------------------------------------------- !
    b = asmat(a)
    expb = expm_3x3(b)
    call asarray(expb, 6, expm_6x1)
  end function expm_6x1
  function expm_3x3(a)
    ! ----------------------------------------------------------------------- !
    real(kind=8) :: expm_3x3(3,3)
    real(kind=8), intent(in) :: a(3,3)
    integer, parameter :: m=3, ldh=3, ideg=6, lwsp=4*m*m+ideg+1
    integer, parameter :: n=3, lwork=3*n-1
    real(kind=8) :: t, wsp(lwsp), v(3,3)
    real(kind=8) :: w(n), work(lwork), l(3,3)
    integer :: ipiv(m), iexph, ns, iflag, info
    character*120 :: msg
    ! ----------------------------------------------------------------------- !
    expm_3x3 = zero
    if (all(abs(a) <= epsilon(a))) then
       expm_3x3 = eye(3)
       return
    else if (isdiag(a)) then
       expm_3x3(1,1) = exp(a(1,1))
       expm_3x3(2,2) = exp(a(2,2))
       expm_3x3(3,3) = exp(a(3,3))
       return
    end if

    ! try dgpadm (usually good)
    t = one
    iflag = 0
    call DGPADM(ideg, m, t, a, ldh, wsp, lwsp, ipiv, iexph, ns, iflag)
    if (iflag >= 0) then
       expm_3x3 = reshape(wsp(iexph:iexph+m*m-1), shape(expm_3x3))
       return
    end if

    ! problem with dgpadm, use other method
    if (iflag == -8) then
       msg = 'bad sizes (in input of DGPADM)'
    else if (iflag == -9) then
       msg = 'Error - null H in input of DGPADM.'
    else if (iflag == -7) then
       msg = 'Problem in DGESV (within DGPADM)'
    end if
    call bombed(msg)
    v = a
    call DSYEV("V", "L", 3, v, 3, w, work, lwork, info)
    l = zero
    l(1,1) = exp(w(1))
    l(2,2) = exp(w(2))
    l(3,3) = exp(w(3))
    expm_3x3 = matmul(matmul(v, l ), transpose(v))
    return
  end function expm_3x3

  ! ------------------------------------------------------------------------- !
  function powm(a, m)
    ! ----------------------------------------------------------------------- !
    ! Computes the matrix power
    ! ----------------------------------------------------------------------- !
    real(kind=8) :: powm(3,3)
    real(kind=8), intent(in) :: a(3,3), m
    integer, parameter :: n=3, lwork=3*n-1
    real(kind=8) :: w(n), work(lwork), v(3,3), l(3,3)
    integer :: info
    ! eigenvalues/vectors of a
    v = a
    powm = zero
    if (isdiag(a)) then
       powm(1,1) = a(1,1) ** m
       powm(2,2) = a(2,2) ** m
       powm(3,3) = a(3,3) ** m
       return
    end if
    call DSYEV("V", "L", 3, v, 3, w, work, lwork, info)
    l = zero
    l(1,1) = w(1) ** m
    l(2,2) = w(2) ** m
    l(3,3) = w(3) ** m
    powm = matmul(matmul(v, l ), transpose(v))
    return
  end function powm

  ! ------------------------------------------------------------------------- !
  function sqrtm(a)
    ! ----------------------------------------------------------------------- !
    ! Computes the matrix sqrt
    ! ----------------------------------------------------------------------- !
    real(kind=8) :: sqrtm(3,3)
    real(kind=8), intent(in) :: a(3,3)
    integer, parameter :: n=3, lwork=3*n-1
    real(kind=8) :: w(n), work(lwork), v(3,3), l(3,3)
    integer :: info
    sqrtm = zero
    if (isdiag(a)) then
       sqrtm(1,1) = sqrt(a(1,1))
       sqrtm(2,2) = sqrt(a(2,2))
       sqrtm(3,3) = sqrt(a(3,3))
       return
    end if
    ! eigenvalues/vectors of a
    v = a
    call DSYEV("V", "L", 3, v, 3, w, work, lwork, info)
    l = zero
    l(1,1) = sqrt(w(1))
    l(2,2) = sqrt(w(2))
    l(3,3) = sqrt(w(3))
    sqrtm = matmul(matmul(v, l ), transpose(v))
    return
  end function sqrtm

  ! ------------------------------------------------------------------------- !
  function logm(a)
    ! ----------------------------------------------------------------------- !
    ! Computes the matrix logarithm
    ! ----------------------------------------------------------------------- !
    real(kind=8) :: logm(3,3)
    real(kind=8), intent(in) :: a(3,3)
    integer, parameter :: n=3, lwork=3*n-1
    real(kind=8) :: w(n), work(lwork), v(3,3), l(3,3)
    integer :: info
    if (isdiag(a)) then
       logm = zero
       logm(1,1) = log(a(1,1))
       logm(2,2) = log(a(2,2))
       logm(3,3) = log(a(3,3))
       return
    end if
    ! eigenvalues/vectors of a
    v = a
    call DSYEV("V", "L", 3, v, 3, w, work, lwork, info)
    l = zero
    l(1,1) = log(w(1))
    l(2,2) = log(w(2))
    l(3,3) = log(w(3))
    logm = matmul(matmul(v, l), transpose(v))
    return
  end function logm

  ! ------------------------------------------------------------------------- !
  function isdiag(a)
    logical :: isdiag
    real(kind=8), intent(in) :: a(3,3)
    isdiag = all(abs(a - diag(a)) <= epsilon(a))
    return
  end function isdiag

  ! ------------------------------------------------------------------------- !
  function diag(a)
    real(kind=8) :: diag(3,3)
    real(kind=8), intent(in) :: a(3,3)
    diag = zero
    diag(1,1) = a(1,1)
    diag(2,2) = a(2,2)
    diag(3,3) = a(3,3)
    return
  end function diag

  ! ------------------------------------------------------------------------- !
  function deps2d(dt, k, e, de)
    ! ----------------------------------------------------------------------- !
    ! Compute symmetric part of velocity gradient given depsdt
    ! ----------------------------------------------------------------------- !
    ! Velocity gradient L is given by
    !             L = dFdt * Finv
    !               = dRdt*I*Rinv + R*dUdt*Uinv*Rinv
    ! where F, I, R, U are the deformation gradient, identity, rotation, and
    ! right stretch tensor, respectively. d*dt and *inv are the rate and
    ! inverse or *, respectively,

    ! The stretch U is given by
    !              if k != 0:
    !                  U = (k*E + I)**(1/k)
    !              else:
    !                  U = exp(E)
    ! and its rate
    !                  dUdt = 1/k*(k*E + I)**(1/k - 1)*k*dEdt
    !                       = (k*E + I)**(1/k)*(k*E + I)**(-1)*dEdt
    !                       = U*X*dEdt
    !                  where X = (kE + I)**(-1)
    !    Then
    !              d = sym(L)
    !              w = skew(L)
    real(kind=8) :: deps2d(6)
    real(kind=8), intent(in) :: dt, k, e(6), de(6)
    real(kind=8) :: eps(3,3), depsdt(3,3), epsf(3,3), u(3,3), i(3,3), x(3,3)
    real(kind=8) :: d(3,3), L(3,3), du(3,3)

    ! convert 1x6 arrays to 3x3 matrices for easier processing
    ! strain
    deps2d = zero
    i = eye(3)
    eps = as3x3(e / VOIGHT)
    depsdt = as3x3(de / VOIGHT)
    epsf = eps + depsdt * dt

    ! stretch and its rate
    if (k == zero) then
       u = expm(epsf)
    else
       u = powm(k * epsf + i, one / k)
    end if

    ! center X on half step
    x = half * (inv(k * epsf + i) + inv(k * eps + I))
    du = matmul(matmul(u, x), depsdt)

    ! velocity gradient
    L = matmul(du, inv(u))
    d = half * (L + transpose(L))

    deps2d = symarray(d) * VOIGHT
    return

  end function deps2d

  function eye(n)
    integer, intent(in) :: n
    real(kind=8) :: eye(n,n)
    integer :: i
    eye = zero
    do i=1, n
       eye(i,i) = one
    end do
    return
  end function eye

  function matinv(A)
    ! This procedure computes the inverse of a real, general matrix using Gauss-
    ! Jordan elimination with partial pivoting. The input matrix, A, is returned
    ! unchanged, and the inverted matrix is returned in Ainv. The procedure also
    ! returns an integer flag, icond, indicating whether A is well- or ill-
    ! conditioned. If the latter, the contents of Ainv will be garbage.
    !
    ! The logical dimensions of the matrices A(1:n,1:n) and Ainv(1:n,1:n) are
    ! assumed to be the same as the physical dimensions of the storage arrays
    ! A(1:np,1:np) and Ainv(1:np,1:np), i.e., n = np. If A is not needed, A and
    ! Ainv can share the same storage locations.
    !
    ! input arguments:
    !
    !     A         real  matrix to be inverted
    !
    ! output arguments:
    !
    !     Ainv      real  inverse of A
    !     icond     int   conditioning flag:
    !                       = 0  A is well-conditioned
    !                       = 1  A is  ill-conditioned

    implicit none
    integer, parameter :: n=3
    real(kind=8)  :: matinv(n,n)
    real(kind=8), intent(in)  :: A(n,n)
    real(kind=8)  :: W(n,n), Wmax, dum(n), fac, Wcond=1.d-13
    integer  :: row, col, v(1)
    integer  :: icond

    ! -----------------------------------------------------------------------------

    ! Initialize
    icond = 0
    matinv  = 0
    do row = 1, n
       matinv(row,row) = 1
    end do
    W = A
    do row = 1, n
       v = maxloc( abs( W(row,:) ) )
       Wmax = W(row,v(1))
       if (Wmax == 0) then
          icond = 1
          return
       end if
       W   (row,:) = W   (row,:) / Wmax
       matinv(row,:) = matinv(row,:) / Wmax
    end do

    ! Gauss-Jordan elimination with partial pivoting

    do col = 1, n
       v = maxloc( abs( W(col:,col) ) )
       row = v(1) + col - 1
       dum(col:)   = W(col,col:)
       W(col,col:) = W(row,col:)
       W(row,col:) = dum(col:)
       dum(:)      = matinv(col,:)
       matinv(col,:) = matinv(row,:)
       matinv(row,:) = dum(:)
       Wmax = W(col,col)
       if ( abs(Wmax) < Wcond ) then
          icond = 1
          return
       end if
       row = col
       W(row,col:) = W(row,col:) / Wmax
       matinv(row,:) = matinv(row,:) / Wmax
       do row = 1, n
          if (row == col) cycle
          fac = W(row,col)
          W(row,col:) = W(row,col:) - fac * W(col,col:)
          matinv(row,:) = matinv(row,:) - fac * matinv(col,:)
       end do
    end do
  end function matinv

  function as3x3(a)
    real(kind=8) :: as3x3(3,3)
    real(kind=8), intent(in) :: a(6)
    as3x3(1, 1) = a(1); as3x3(1, 2) = a(4); as3x3(1, 3) = a(6)
    as3x3(2, 1) = a(4); as3x3(2, 2) = a(2); as3x3(2, 3) = a(5)
    as3x3(3, 1) = a(6); as3x3(3, 2) = a(5); as3x3(3, 3) = a(3)
    return
  end function as3x3

  ! ------------------------------------------------------------------------- !
  subroutine update_deformation(dt, k, farg, darg, f, e)
    ! ----------------------------------------------------------------------- !
    ! From the value of the Seth-Hill parameter kappa, current strain E,
    ! deformation gradient F, and symmetric part of the velocit gradient d,
    ! update the strain and deformation gradient.
    ! ----------------------------------------------------------------------- !
    ! Deformation gradient is given by
    !
    !              dFdt = L*F                                             (1)
    !
    ! where F and L are the deformation and velocity gradients, respectively.
    ! The solution to (1) is
    !
    !              F = F0*exp(Lt)
    !
    ! Solving incrementally,
    !
    !              Fc = Fp*exp(Lp*dt)
    !
    ! where the c and p refer to the current and previous values, respectively.
    !
    ! With the updated F, Fc, known, the updated stretch is found by
    !
    !              U = (trans(Fc)*Fc)**(1/2)
    !
    ! Then, the updated strain is found by
    !
    !              E = 1/k * (U**k - I)
    !
    ! where k is the Seth-Hill strain parameter.
    real(kind=8), intent(in) :: dt, k, farg(9), darg(6)
    real(kind=8), intent(out) :: f(9), e(6)
    real(kind=8) :: f0(3,3), d(3,3), i(3,3), eps(3,3), u(3,3), ff(3,3)

    ! convert arrays to matrices for upcoming operations
    f0 = transpose(reshape(farg, shape(f0)))
    d = as3x3(darg / VOIGHT)
    i = eye(3)

    ff = matmul(expm(d * dt), f0)
    u = sqrtm(matmul(transpose(ff), ff))
    if (k == 0) then
       eps = logm(u)
    else
       eps = one / k * (powm(u, k) - i)
    end if
    if (det(ff) <= zero) then
       stop "negative Jacobian encountered"
    end if
    f = reshape(transpose(ff), shape(f))
    e = symarray(eps) * VOIGHT
    return
  end subroutine update_deformation

  ! ------------------------------------------------------------------------- !
  subroutine update_strain(k, farg, e)
    ! ----------------------------------------------------------------------- !
    ! Update strain by
    !
    !              E = 1/k * (U**k - I)
    !
    ! where k is the Seth-Hill strain parameter.
    real(kind=8), intent(in) :: k, farg(9)
    real(kind=8), intent(out) :: e(6)
    real(kind=8) :: f(3,3), eps(3,3), u(3,3), i(3,3)
    f = transpose(reshape(farg, (/3,3/)))
    u = sqrtm(matmul(transpose(f), f))
    i = eye(3)
    if (k == 0) then
       eps = logm(u)
    else
       eps = one / k * (powm(u, k) - i)
    end if
    if (det(f) <= zero) then
       stop "negative Jacobian encountered"
    end if
    e = symarray(eps) * VOIGHT
    return
  end subroutine update_strain

  ! ------------------------------------------------------------------------- !
  function det(a)
    ! ----------------------------------------------------------------------- !
    ! determinant of 3x3
    ! ----------------------------------------------------------------------- !
    implicit none
    real(kind=8) :: det
    real(kind=8), intent(in) :: a(3,3)
    det = a(1,1)*a(2,2)*a(3,3)  &
        - a(1,1)*a(2,3)*a(3,2)  &
        - a(1,2)*a(2,1)*a(3,3)  &
        + a(1,2)*a(2,3)*a(3,1)  &
        + a(1,3)*a(2,1)*a(3,2)  &
        - a(1,3)*a(2,2)*a(3,1)
      return
    end function det

  ! ------------------------------------------------------------------------- !
  function inv(a)
    ! ----------------------------------------------------------------------- !
    ! inverse of 3x3
    ! ----------------------------------------------------------------------- !
    implicit none
    real(kind=8) :: inv(3,3)
    real(kind=8), intent(in)  :: a(3,3)
    real(kind=8) :: deta
    real(kind=8) :: cof(3,3)
    deta = det(a)
    if (abs(deta) .le. epsilon(deta)) then
       inv = zero
       stop "non-invertible matrix sent to inv"
    end if
    cof(1,1) = +(a(2,2)*a(3,3)-a(2,3)*a(3,2))
    cof(1,2) = -(a(2,1)*a(3,3)-a(2,3)*a(3,1))
    cof(1,3) = +(a(2,1)*a(3,2)-a(2,2)*a(3,1))
    cof(2,1) = -(a(1,2)*a(3,3)-a(1,3)*a(3,2))
    cof(2,2) = +(a(1,1)*a(3,3)-a(1,3)*a(3,1))
    cof(2,3) = -(a(1,1)*a(3,2)-a(1,2)*a(3,1))
    cof(3,1) = +(a(1,2)*a(2,3)-a(1,3)*a(2,2))
    cof(3,2) = -(a(1,1)*a(2,3)-a(1,3)*a(2,1))
    cof(3,3) = +(a(1,1)*a(2,2)-a(1,2)*a(2,1))
    inv = transpose(cof) / deta
    return
  end function inv

  ! ------------------------------------------------------------------------- !
  function mag(a)
    ! ----------------------------------------------------------------------- !
    ! trace of symmetric tensor stored as 6x1 array
    ! ----------------------------------------------------------------------- !
    implicit none
    real(kind=8) :: mag
    real(kind=8), intent(in) :: a(6)
    mag = sqrt(ddot(a, a))
    return
  end function mag

  ! ------------------------------------------------------------------------- !
  function dyad(a, b)
    real(kind=8) :: dyad(6)
    real(kind=8), intent(in) :: a(3), b(3)
    dyad = (/a(1) * b(1), a(2) * b(2), a(3) * b(3), &
             a(1) * b(2), a(2) * b(3), a(1) * b(3)/)
    return
  end function dyad

  ! ------------------------------------------------------------------------- !
  function ddot(a, b)
    ! ----------------------------------------------------------------------- !
    ! double of symmetric tensors stored as 6x1 arrays
    ! ----------------------------------------------------------------------- !
    implicit none
    real(kind=8) :: ddot
    real(kind=8), parameter :: w(6)=(/one,one,one,two,two,two/)
    real(kind=8), intent(in) :: a(6), b(6)
    ddot = sum(a * b * w)
    return
  end function ddot

  function asmat(a)
    implicit none
    real(kind=8) :: asmat(3,3)
    real(kind=8), intent(in) :: a(6)
    asmat = as3x3(a)
    return
  end function asmat

  function trace(a)
    implicit none
    real(kind=8) :: trace
    real(kind=8), intent(in) :: a(6)
    trace = sum(a(1:3))
    return
  end function trace

  function dev(a)
    implicit none
    real(kind=8) :: dev(6)
    real(kind=8), intent(in) :: a(6)
    dev = a - iso(a)
    return
  end function dev

  function iso(a)
    implicit none
    real(kind=8) :: iso(6)
    real(kind=8), intent(in) :: a(6)
    iso = trace(a) / three * (/one, one, one, zero, zero, zero/)
    return
  end function iso

  subroutine get_invariants(a, b, n)
    implicit none
    real(kind=8), intent(in) :: a(3,3), n(3)
    real(kind=8), intent(out) :: b(5)
    real(kind=8) :: asq(3,3), tra, trasq, deta
    asq = matmul(a, a)
    tra = a(1, 1) + a(2, 2) + a(3, 3)
    trasq = asq(1, 1) + asq(2, 2) + asq(3, 3)
    deta = det(a)

    b(1) = tra
    b(2) = .5 * (tra ** 2 - trasq)
    b(3) = deta

    b(4) = dot_product(matmul(n, a), n)
    b(5) = dot_product(matmul(n, asq), n)
    return
  end subroutine get_invariants

end module mmlabpack
