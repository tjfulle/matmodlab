subroutine user(stress, statev, ddsdde, stran, dstran, time, dtime, temp, &
     dtemp, nstatv, props, nprops, dfgrd0, dfgrd1)

  implicit none
  real(8), parameter :: zero=0.e+00_8, one=1.e+00_8, two=2.e+00_8
  real(8), parameter :: three=3.e+00_8, six=6.e+00_8
  integer, intent(in) :: nstatv, nprops
  real(8), intent(in) :: time, dtime, temp, dtemp
  real(8), intent(inout) :: stress(6), statev(nstatv), ddsdde(6, 6)
  real(8), intent(in) :: stran(6), dstran(6)
  real(8), intent(in) :: props(nprops)
  real(8), intent(in) :: dfgrd0(3, 3), dfgrd1(3, 3)

  ! LOCAL ARRAYS
  ! eelas - elastic strains
  ! etherm - thermal strains
  ! dtherm - incremental thermal strains
  ! deldse - change in stiffness due to temperature change
  integer :: k1, k2
  real(8) :: eelas(6), etherm(6), dtherm(6), deldse(6,6), enu
  real(8) :: fac0, fac1, ebulk3, eg, eg0, eg2, eg20, elam, elam0, emod

  ! UMAT FOR ISOTROPIC THERMO-ELASTICITY WITH LINEARLY VARYING
  ! MODULI - CANNOT BE USED FOR PLANE STRESS
  ! PROPS(1) - E(T0)
  ! PROPS(2) - NU(T0)
  ! PROPS(3) - T0
  ! PROPS(4) - E(T1)
  ! PROPS(5) - NU(T1)
  ! PROPS(6) - T1
  ! PROPS(7) - ALPHA
  ! PROPS(8) - T_INITIAL

  ! ELASTIC PROPERTIES AT START OF INCREMENT
  fac1 = (temp - props(3)) / (props(6) - props(3))
  if (fac1 .lt. zero) fac1 = zero
  if (fac1 .gt. one) fac1 = one
  fac0 = one - fac1
  emod = fac0 * props(1) + fac1 * props(4)
  enu = fac0 * props(2) + fac1 * props(5)
  ebulk3 = emod / (one - two * enu)
  eg20 = emod / (one + enu)
  eg0 = eg20 / two
  elam0 = (ebulk3 - eg20) / three

  ! ELASTIC PROPERTIES AT END OF INCREMENT
  fac1 = (temp + dtemp - props(3)) / (props(6) - props(3))
  if (fac1 .lt. zero) fac1 = zero
  if (fac1 .gt. one) fac1 = one
  fac0 = one - fac1
  emod = fac0 * props(1) + fac1 * props(4)
  enu = fac0 * props(2) + fac1 * props(5)
  ebulk3 = emod / (one - two * enu)
  eg2 = emod / (one + enu)
  eg = eg2 / two
  elam = (ebulk3 - eg2) / three

  ! ELASTIC STIFFNESS AT END OF INCREMENT AND STIFFNESS CHANGE
  do k1 = 1,3
     do k2 = 1,3
        ddsdde(k2,k1) = elam
        deldse(k2,k1) = elam - elam0
     end do
     ddsdde(k1,k1) = eg2 + elam
     deldse(k1,k1) = eg2 + elam - eg20 - elam0
  end do
  do k1 = 3 + 1,6
     ddsdde(k1,k1) = eg
     deldse(k1,k1) = eg - eg0
  end do

  ! CALCULATE THERMAL EXPANSION
  do k1 = 1,3
     etherm(k1) = props(7) * (temp - props(8))
     dtherm(k1) = props(7) * dtemp
  end do
  do k1 = 3 + 1,6
     etherm(k1) = zero
     dtherm(k1) = zero
  end do

  ! CALCULATE STRESS, ELASTIC STRAIN AND THERMAL STRAIN
  do k1 = 1, 6
     do k2 = 1, 6
        stress(k2) = stress(k2) + ddsdde(k2,k1) * (dstran(k1) - dtherm(k1)) &
                   + deldse(k2,k1) * (stran(k1) - etherm(k1))
     end do
     etherm(k1) = etherm(k1) + dtherm(k1)
     eelas(k1) = stran(k1) + dstran(k1) - etherm(k1)
  end do

  ! STORE ELASTIC AND THERMAL STRAINS IN STATE VARIABLE ARRAY
  do k1 = 1, 6
     statev(k1) = eelas(k1)
     statev(k1 + 6) = etherm(k1)
  end do
  return
end subroutine user
