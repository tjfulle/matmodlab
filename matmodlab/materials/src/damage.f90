MODULE DAMAGE

  IMPLICIT NONE
  INTEGER, PARAMETER :: DP=SELECTED_REAL_KIND(14)
  INTEGER, PARAMETER :: NUI=3, NSDV=3
  INTEGER, PARAMETER :: IPR=1
  INTEGER, PARAMETER :: IPM=2
  INTEGER, PARAMETER :: IPB=3
  INTEGER, PARAMETER :: KUDEVM=1
  INTEGER, PARAMETER :: KDMGDISS=2
  INTEGER, PARAMETER :: KUREC=3
  REAL(KIND=DP), PARAMETER :: ONE=1._DP, TWO=2._DP
  REAL(KIND=DP), PARAMETER :: RTPI=1.772453850905515881919427556567825E+00_DP

CONTAINS

  ! ************************************************************************* !

  SUBROUTINE MULLINS(NPROP, PROPS, UVOL, UDEV, TEMP, DTEMP, &
       NSTATV, STATEV, ETA, DETADW)
    ! ----------------------------------------------------------------------- !
    ! IMPLEMENTATION OF ABAQUS MULLINS DAMAGE MODEL
    ! ----------------------------------------------------------------------- !
    INTEGER, INTENT(IN) :: NPROP, NSTATV
    REAL(KIND=DP), INTENT(IN) :: PROPS(NPROP), UVOL, UDEV, TEMP, DTEMP
    REAL(KIND=DP), INTENT(INOUT) :: STATEV(NSTATV)
    REAL(KIND=DP), INTENT(OUT) :: ETA, DETADW
    REAL(KIND=DP) :: R, M, B, UDIFF, DNOM, DETADERF, DERFDW, ETAM, DMGDISS
    REAL(KIND=DP) :: UREC, UDEVM, DMGA
    ! ----------------------------------------------------------- MULLINS --- !

    R = PROPS(IPR)
    M = PROPS(IPM)
    B = PROPS(IPB)
    UDEVM = MAX(STATEV(KUDEVM), UDEV)

    ! DAMAGE PARAMETER AND DERIVATIVE
    UDIFF = UDEVM - UDEV
    DNOM = M + B * UDEVM
    ETA = ONE - ONE / R * ERF(UDIFF / DNOM)
    DETADERF = -ONE / R * DERF(UDIFF / DNOM)
    DERFDW = -ONE / DNOM
    DETADW = DETADERF * DERFDW

    ! DAMAGE DISSIPATION
    ETAM =  ONE - ONE / R * ERF(UDEVM / DNOM)
    DMGDISS = DMGFUNM(M, B, R, UDEV, UDEVM, ETAM)
    STATEV(KDMGDISS) = DMGDISS

    ! RECOVERABLE ENERGY
    DMGA = DMGFUNM(M, B, R, UDEV, UDEVM, ETA)
    STATEV(KUREC) = ETA * UDEV + UVOL + DMGA - DMGDISS

    ! STORE MAX DEVIATORIC ENERGY
    STATEV(KUDEVM) = UDEVM

    RETURN

  END SUBROUTINE MULLINS

  ! ************************************************************************* !

  REAL(KIND=DP) FUNCTION DMGFUNM(M, B, R, UDEV, UDEVM, ETA)
    ! ----------------------------------------------------------------------- !
    ! COMPUTE THE ABAQUS MULLINS DAMAGE FUNCTION
    ! ----------------------------------------------------------------------- !
    REAL(KIND=DP), INTENT(IN) :: M, B, R, UDEV, UDEVM, ETA
    REAL(KIND=DP) :: FAC, TERM1, TERM2
    ! ----------------------------------------------------------- DMGFUNM --- !
    FAC = M + B * UDEVM
    TERM1 = FAC / R / RTPI * (EXP(-(UDEVM - UDEV) / FAC)) ** 2 - ONE
    TERM2 = (ONE - ETA) * UDEVM
    DMGFUNM = TERM1 + TERM2
    RETURN
  END FUNCTION DMGFUNM

  ! ************************************************************************* !

  REAL(KIND=DP) FUNCTION DERF(Z)
    ! ----------------------------------------------------------------------- !
    ! DERIVATIVE OF THE ERROR FUNCTION
    ! ----------------------------------------------------------------------- !
    REAL(KIND=DP), INTENT(IN) :: Z
    ! -------------------------------------------------------------- DERF --- !
    DERF = TWO / RTPI * EXP(-Z ** 2)
    RETURN
  END FUNCTION DERF

END MODULE DAMAGE
