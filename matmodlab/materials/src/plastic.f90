! --------------------------------------------------------------------------- !
! UMAT FOR ISOTROPIC ELASTICITY AND MISES PLASTICITY
! WITH KINEMATIC HARDENING - CANNOT BE USED FOR PLANE STRESS
!
!    PROPS(1) - E
!    PROPS(2) - NU
!    PROPS(3) - SYIELD
!    PROPS(4) - HARD
!
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
  real(8), intent(in) :: sse, scd, rpl, drpldt, time, dtime, temp, dtemp
  real(8), intent(in) :: pnewdt, celent
  real(8), intent(inout) :: stress(ntens), statev(nstatv), ddsdde(ntens, ntens)
  real(8), intent(inout) :: spd, ddsddt(ntens), drplde(ntens)
  real(8), intent(in) :: stran(ntens), dstran(ntens)
  real(8), intent(in) :: predef(1), dpred(1), props(nprops), coords(3)
  real(8), intent(in) :: drot(3, 3), dfgrd0(3, 3), dfgrd1(3, 3)

  ! ------------------------------------------------------------------------- !
  ! local variables
  ! ------------------------------------------------------------------------- !
  !    EE  - Elastic strains
  !    EP  - Plastic strains
  !    AL  - Shift tensor
  !    M   - Plastic flow directions
  !    olds   - Stress at start of increment
  !    oldpl  - Plastic strains at start of increment
  real(8) :: EE(6), EP(6), AL(6), M(6), olds(6), oldpl(6)
  real(8), parameter :: maxNu=.4999_8, toler=1.0e-6_8
  real(8) :: E, Nu, G, G2, G3, effG, effG2, effG3, K3, Lam, effLam
  real(8) :: Y, H, effH, S, P, deqpl
  integer :: i, j
  ! ------------------------------------------------------------------------- !

  ! Material properties
  E = props(1)
  Nu = min(props(2), maxNu)
  Y = props(3)
  H = props(4)

  ! Derived properties
  K3 = E / (1. - 2. * Nu)
  G2 = E /(1. + Nu)
  G = G2 / 2.
  G3 = 3. * G
  Lam = (K3 - G2) / 3.

  ! Elastic stiffness
  do i=1,ndi
     do j=1,ndi
        ddsdde(j,i) = Lam
     end do
     ddsdde(i,i) = G2 + Lam
  end do
  do i=ndi+1,ntens
     ddsdde(i,i) = G
  end do

  ! Recover elastic strain, plastic strain and shift tensor and rotate note:
  ! use code 1 for (tensor) stress, code 2 for (engineering) strain
  call rotsig(statev(        3), drot, EE, 2, ndi, nshr)
  call rotsig(statev(  ntens+3), drot, EP, 2, ndi, nshr)
  call rotsig(statev(2*ntens+3), drot, AL, 1, ndi, nshr)

  ! Save stress and plastic strains and calculate predictor stress and elastic
  ! strain
  olds = stress
  oldpl = EP
  EE(i) = EE(i) + dstran(i)
  stress = stress + matmul(ddsdde, dstran)

  ! Calculate the trial equivalent von mises stress
  S = (stress(1) - AL(1) - stress(2) + AL(2)) ** 2 &
    + (stress(2) - AL(2) - stress(3) + AL(3)) ** 2 &
    + (stress(3) - AL(3) - stress(1) + AL(1)) ** 2
  do i=ndi+1,ntens
     S = S + 6. * (stress(i) - AL(i)) ** 2
  end do
  S = sqrt(S / 2.)

  ! Determine if actively yielding
  if(S > (1. + toler) * Y) then

     ! Actively yielding
     ! Separate the hydrostatic from the deviatoric stress calculate the flow
     ! direction

     P = -sum(stress(1:3)) / 3.
     M(1:ndi) = (stress(1:ndi) - AL(1:ndi) + P) / S
     M(ndi+1:ntens) = (stress(ndi+1:ntens) - AL(ndi+1:ntens)) / S

     ! Solve for equivalent plastic strain increment
     deqpl = (S - Y) / (G3 + H)

     ! Update shift tensor, elastic and plastic strains
     AL(1:ndi) = AL(1:ndi) + H * M(1:ndi) * deqpl
     AL(ndi+1:ntens) = AL(ndi+1:ntens) + H * M(ndi+1:ntens) * deqpl

     EP(1:ndi) = EP(1:ndi) + 3. / 2. * M(1:ndi) * deqpl
     EP(ndi+1:ntens) = EP(ndi+1:ntens) + 3. * M(ndi+1:ntens) * deqpl

     EE(1:ndi) = EE(1:ndi) - 3. / 2. * M(1:ndi) * deqpl
     EE(ndi+1:ntens) = EE(ndi+1:ntens) - 3. * M(ndi+1:ntens) * deqpl

     stress(1:ndi) = AL(1:ndi) + M(1:ndi) * Y - P
     stress(ndi+1:ntens) = AL(ndi+1:ntens) + M(ndi+1:ntens) * Y

     ! Plastic dissipation
     spd = sum((stress + olds) * (EP - oldpl)) / 2.

     ! Formulate the jacobian (material tangent)
     effG = G *(Y + H * deqpl) / S
     effG2 = 2. * effG
     effG3 = 3. * effG
     effLam = (K3 - effG2) / 3.
     effH = G3 * H/(G3 + H) - effG3
     do i=1,ndi
        do j=1,ndi
           ddsdde(j,i) = effLam
        end do
        ddsdde(i,i) = effG2 + effLam
     end do
     do i=ndi+1,ntens
        ddsdde(i,i) = effG
     end do
     do i=1, ntens
        do j=1, ntens
           ddsdde(j,i) = ddsdde(j, i) + effH * M(j) * M(i)
        end do
     end do
  endif

  ! Store elastic strains, plastic strains and shift tensor
  ! in state variable array

  S = (stress(1) - AL(1) - stress(2) + AL(2)) ** 2 &
    + (stress(2) - AL(2) - stress(3) + AL(3)) ** 2 &
    + (stress(3) - AL(3) - stress(1) + AL(1)) ** 2
  do i=ndi+1,ntens
     S = S + 6. * (stress(i) - AL(i)) ** 2
  end do
  S = sqrt(S / 2.)
  P = -sum(stress(1:3)) / 3.
  statev(1) = P
  statev(2) = S
  do i=1,ntens
     statev(i+2) = EE(i)
     statev(i+2+ntens) = EP(i)
     statev(i+2+2*ntens) = AL(i)
  end do

  return

end subroutine umat
