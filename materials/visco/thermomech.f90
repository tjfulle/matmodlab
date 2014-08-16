MODULE THERMOMECH
  ! ----------------------------------------------------------------------- !
  ! PROCEDURES TO DETERMINE STRESS FREE STRAIN DUE TO THERMAL EXPANSION
  !
  ! NOTE
  ! ----
  ! SYMMETRIC SECOND ORDER TENSOR ORDERING CORRESPONDS TO ABAQUS EXPLICIT,
  ! THAT IS, A SYMMETRIC SECOND ORDERING TENSOR A IS STORED AS
  !
  !                          | A(1)  A(4)  A(6) |
  !                          |       A(2)  A(5) |
  !                          |             A(3) |
  !
  ! THIS ORDERING DIFFERS FROM ABAQUS STANDARD THAT USES
  !
  !                          | A(1)  A(4)  A(5) |
  !                          |       A(2)  A(6) |
  !                          |             A(3) |
  !
  ! WHY CHOOSE THE FIRST? HABIT, AND IT IS THE ORDERING USED AT SANDIA, ANSYS,
  ! LSDYNA, ...
  !
  ! FOR USE IN MATMODLAB, THIS FILE MUST BE LINKED WITH mmlfio.f90
  ! ----------------------------------------------------------------------- !
  IMPLICIT NONE
  INTEGER, PARAMETER :: DP=SELECTED_REAL_KIND(14)
  REAL(KIND=DP), PARAMETER :: ZERO=0._DP, ONE=1._DP, THREE=3._DP
  REAL(KIND=DP), PARAMETER :: P3RD=ONE/THREE
  REAL(KIND=DP), PARAMETER :: MACHINE_EPSILON=EPSILON(ONE)

CONTAINS

  SUBROUTINE MECHDEF(NPROP, PROPS, TEMPN, DTEMP, F, FM)
    ! ----------------------------------------------------------------------- !
    ! COMPUTE THE MECHANICAL DEFORMATION GRADIENT
    !
    ! NOTES
    ! -----
    ! DEFORMATION GRADIENT
    !
    !                    [F] = [Fm] [Fth]
    !
    ! WHERE, WITH ISOTROPIC THERMAL EXPANSION
    !
    !                   [Fth] = Jth^(1/3) [I]
    !
    ! SO
    !
    !                   [Fm] = Jth^(-1/3) [F]
    ! NOTE
    ! ----
    ! THIS PROCEDURE CALLS THERMAL_EXPANSION TO GET THE LINEAR THERMAL STRAIN
    ! FROM THE CTE AND COMPUTES THE THERMAL DEFORMATION. THE THERMAL EXPANSION
    ! PROCEDURE COULD BE GENERALIZED TO INTERPOLATE FROM A TABLE RATHER THAN
    ! USE A CONSTANT CTE. THEN, AN INITIAL THERMAL STRAIN COULD BE USED.
    ! ----------------------------------------------------------------------- !
    INTEGER :: NPROP
    REAL(KIND=DP), INTENT(IN) :: PROPS(NPROP), TEMPN, DTEMP, F(3,3)
    REAL(KIND=DP), INTENT(OUT) :: FM(3,3)
    REAL(KIND=DP) :: CTE, FAC1, FAC2, FAC, TEMP, THS0, THS
    ! ----------------------------------------------------------------------- !

    FM = F
    CTE = PROPS(1)

    IF (ABS(CTE) < MACHINE_EPSILON) RETURN

    ! EVALUATE THE THERMAL STRAIN. IF THERMAL_EXPANSION IS GENERALIZED, COULD
    ! BE USED TO GET THS0
    THS0 = ZERO
    TEMP = TEMPN + DTEMP
    CALL THERMAL_EXPANSION(CTE, TEMP, DTEMP, THS)
    FAC1 = EXP(THS0) ** 3
    FAC2 = EXP(THS) ** 3

    ! COMPUTE THE MECHANICAL DEFORMATION GRADIENT
    FAC = (FAC1 / FAC2) ** P3RD
    FM = FAC * F
    RETURN

  END SUBROUTINE MECHDEF

  ! ************************************************************************* !

  SUBROUTINE THERMAL_EXPANSION(CTE, TEMP, DTEMP, THS)
    ! COMPUTE (LINEAR) THERMAL EXPANSION
    REAL(KIND=DP), INTENT(IN) :: CTE, TEMP, DTEMP
    REAL(KIND=DP), INTENT(OUT) :: THS
    THS = CTE * DTEMP
  END SUBROUTINE THERMAL_EXPANSION

END MODULE THERMOMECH
