! --------------------------------------------------------------------------- !
!    UMAT FOR ISOTROPIC ELASTICITY
!
!    PROPS(1) - KAPPA
!    PROPS(2) - E
!    PROPS(3) - NU
! --------------------------------------------------------------------------- !
subroutine umat(stress, statev, ddsdde, sse, spd, scd, rpl, &
     ddsddt, drplde, drpldt, stran, dstran, time, dtime, temp, dtemp, &
     predef, dpred, cmname, ndi, nshr, ntens, nstatv, props, nprops, &
     coords, drot, pnewdt, celent, dfgrd0, dfgrd1, noel, npt, layer, &
     kspt, kstep, kinc)
  implicit none
  character*8, intent(in) :: cmname
  integer, intent(in) :: ndi, nshr, ntens, nstatv, nprops
  integer, intent(in) :: noel, npt, layer, kspt, kstep, kinc
  real(8), intent(in) :: sse, spd, scd, rpl, drpldt, time, dtime, temp, dtemp
  real(8), intent(in) :: pnewdt, celent
  real(8), intent(inout) :: stress(ntens), statev(nstatv), ddsdde(ntens, ntens)
  real(8), intent(inout) :: ddsddt(ntens), drplde(ntens)
  real(8), intent(in) :: stran(ntens), dstran(ntens)
  real(8), intent(in) :: predef(1), dpred(1), props(nprops), coords(3)
  real(8), intent(in) :: drot(3, 3), dfgrd0(3, 3), dfgrd1(3, 3)!

  integer :: i, j
  real(8) :: K, K3, G, G2, Lam, Kappa, E, Nu
  real(8) :: R(3,3), U(6), E_iso(3,3), E_dev(3,3)
  real(8) :: Ident(3,3), V(3), Q(3,3), Work(3,3), L(3,3), sig(3,3)
  character*120 :: msg
  character*8 :: charv(1)
  integer :: intv(1)
  real(8) :: realv(1)
  ! ------------------------------------------------------------------------- !

  if (ndi /= 3) then
     msg = 'this umat may only be used for elements &
          &with three direct stress components'
     call stdb_abqerr(-3, msg, intv, realv, charv)
  end if

  ! elastic properties
  Kappa = props(1)
  E = props(2)
  Nu = props(3)

  ! Get the bulk, shear, and Lame constants
  K = E / 3. / (1. - 2. * Nu)
  G = E / 2. / (1. + Nu)
  
  K3 = 3. * K
  G2 = 2. * G
  Lam = (K3 - G2) / 3.

  ! elastic stiffness
  ddsdde = 0.
  do i=1,ndi
     do j = 1,ndi
        ddsdde(j,i) = Lam
     end do
     ddsdde(i,i) = G2 + Lam
  end do
  do i=ndi+1,ntens
     ddsdde(i,i) = G
  end do

  ! compute the strain
  Ident = 0.
  Ident(1,1) = 1.; Ident(2,2) = 1.; Ident(3,3) = 1.
  call polar_decomp(dfgrd1, R, Work)
  call asarray(Work, U)

  ! eigenvalues and vectors of U
  call sprind(U, V, Q, 1, 3, 3)

  L = 0.
  if (abs(Kappa) <= 1e-12_8) then
     L(1,1) = log(V(1))
     L(2,2) = log(V(2))
     L(3,3) = log(V(3))
     Work = matmul(matmul(Q, L), transpose(Q))
  else
     L(1,1) = V(1) ** (2. * Kappa)
     L(2,2) = V(2) ** (2. * Kappa)
     L(3,3) = V(3) ** (2. * Kappa)
     Work = (matmul(matmul(Q, L), transpose(Q)) - Ident) / 2. / Kappa
  end if

  ! stress update
  E_iso = (Work(1,1) + Work(2,2) + Work(3,3)) / 3. * Ident
  E_dev = Work - E_iso
  sig = K3 * E_iso + G2 * E_dev

  ! convert the stress to an array
  call asarray(sig, stress)

  return
end subroutine umat

subroutine polar_decomp(F, R, U)
  implicit none
  real(kind=8), intent(in) :: F(3,3)
  real(kind=8), intent(out) :: R(3,3), U(3,3)
  real(kind=8) :: I(3,3)
  integer :: j
  character*120 :: msg
  character*8 :: charv(1)
  integer :: intv(1)
  real(8) :: realv(1)

  I = 0.
  I(1,1) = 1.; I(2,2) = 1.; I(3,3) = 1.
  R = F
  do j = 1, 20
    R = .5 * matmul(R, 3. * I - matmul(transpose(R), R))
    if (maxval(abs(matmul(transpose(R), R) - I)) < 1.e-6_8) then
      U = matmul(transpose(R), F)
      return
    end if
  end do

  msg = 'polar decomposition failed to converge'
  call stdb_abqerr(-3, msg, intv, realv, charv)

end subroutine polar_decomp

subroutine asarray(A, X)
  implicit none
  real(8), intent(in) :: A(3,3)
  real(8), intent(out) :: X(6)
  X(1) = A(1,1)
  X(2) = A(2,2)
  X(3) = A(3,3)
  X(4) = A(1,2)
  X(5) = A(1,3)
  X(6) = A(2,3)
  return
end subroutine asarray