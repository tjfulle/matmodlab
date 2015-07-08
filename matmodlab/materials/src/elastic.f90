! --------------------------------------------------------------------------- !
!    UMAT FOR ISOTROPIC ELASTICITY
!    CANNOT BE USED FOR PLANE STRESS
!
!    PROPS(1) - E
!    PROPS(2) - NU
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
  real(8) :: K, K3, G, G2, Lam
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
  K = props(1)
  K3 = 3. * K
  G = props(2)
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

  ! stress update
  stress = stress + matmul(ddsdde, dstran)

  return
end subroutine umat
