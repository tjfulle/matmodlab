SUBROUTINE UHYPER(I1B, I2B, JAC, U, DU, D2U, D3U, TEMP, NOEL, CMNAME, &
     INCMPFLAG, NSTATEV, STATEV, NFIELDV, FIELDV, FIELDVINC, &
     NPROPS, PROPS)
  ! ----------------------------------------------------------------------- !
  ! HYPERELASTIC POLYNOMIAL MODEL
  ! ----------------------------------------------------------------------- !
  IMPLICIT NONE
  INTEGER, PARAMETER :: DP=SELECTED_REAL_KIND(14)
  REAL(KIND=DP), PARAMETER :: ZERO=0._DP, ONE=1._DP, TWO=2._DP, THREE=3._DP
  REAL(KIND=DP), PARAMETER :: FOUR=4._DP
  CHARACTER*8, INTENT(IN) :: CMNAME
  INTEGER, INTENT(IN) :: NPROPS, NOEL, NSTATEV, INCMPFLAG, NFIELDV
  REAL(KIND=DP), INTENT(IN) :: I1B, I2B, JAC, PROPS(NPROPS), TEMP
  REAL(KIND=DP), INTENT(INOUT) :: U(2), DU(3), D2U(6), D3U(6), STATEV(NSTATEV)
  REAL(KIND=DP), INTENT(INOUT) :: FIELDV(NFIELDV), FIELDVINC(NFIELDV)
  ! ----------------------------------------------------------------------- !
  NC = PROPS(1)
  ND = PROPS(2)
  CALL    KU(NC, ND, PROPS(3), PROPS(4+NC), I1B, I2B, JAC, U)
  CALL   KDU(NC, ND, PROPS(3), PROPS(4+NC), I1B, I2B, JAC, DU)
  CALL  KD2U(NC, ND, PROPS(3), PROPS(4+NC), I1B, I2B, JAC, D2U)
  CALL  KD3U(NC, ND, PROPS(3), PROPS(4+NC), I1B, I2B, JAC, D3U)

CONTAINS

  SUBROUTINE KU(NC, ND, C, D, PHI, I1B, I2B, JAC, U)
    ! ----------------------------------------------------------------------- !
    ! HYPER ELASTIC ENERGY FUNCTION WITH POROSITY
    ! ----------------------------------------------------------------------- !
    INTEGER, INTENT(IN) :: NC
    REAL(KIND=DP), INTENT(IN) :: C(NC), D(ND), I1B, I2B, JAC
    REAL(KIND=DP), INTENT(OUT) :: U(2)
    REAL(KIND=DP) :: UVOL, UDEV
    INTEGER :: I, J, L
    ! ----------------------------------------------------------------------- !
    ! NOTES
    ! -----
    !              N
    !     U = A = SUM(Cij(I1-3)^i(I2-3)^j + K / 2 (J - 1)^2 - 2/D (J - 1)
    !            i+j=1
    !
    ! ----------------------------------------------------------------------- !

    ! --- DISTORTION
    UDEV = ZERO
    DO L = 1, NC
       I = NCI(L); J = NCJ(L)
       UDEV = UDEV + C(L) * PPOW(I1B - THREE, I) * PPOW(I2B - THREE, J)
    END DO

    ! --- DILATATION
    UVOL = ZERO
    DO I = 1,ND
       IF (D(I) .GT. 1E+20_DP) THEN
          UVOL = UVOL + TWO * REAL(I, DP) / D(I) * (JAC - ONE) ** (2 * I - 1)
       END IF
    END DO

    ! TOTAL
    U(1) = UVOL + UDEV
    U(2) = UDEV
    RETURN

  END SUBROUTINE KU

  ! ************************************************************************* !

  SUBROUTINE KDU(NC, ND, C, D, I1B, I2B, JAC, DU)
    ! ----------------------------------------------------------------------- !
    ! FIRST DERIVATIVE OF ENERGY: DU/DIi
    ! ----------------------------------------------------------------------- !
    INTEGER, INTENT(IN) :: NC, ND
    REAL(8), INTENT(IN) :: C(NC), D(ND), I1B, I2B, JAC
    REAL(8), INTENT(OUT) :: DU(3)
    INTEGER :: I, J, L
    REAL(8) :: C1, C2, RI, RJ
    ! ----------------------------------------------------------------------- !

    DU = ZERO

    ! --- DISTORTION
    DO L = 1, NC
       I = NCI(L); J = NCJ(L)
       IF (I + J < 1) CYCLE
       RI = REAL(I, DP); RJ = REAL(J, DP)

       ! DU(1)
       C1 = PPOW(I1B - THREE, I - 1)
       C2 = PPOW(I2B - THREE, J)
       DU(1) = DU(1) + RI * C(L) * C1 * C2

       ! DU(2)
       C1 = PPOW(I1B - THREE, I)
       C2 = PPOW(I2B - THREE, J - 1)
       DU(2) = DU(2) + RJ * C(L) * C1 * C2
    END DO

    ! --- DILATATION
    DO I = 1,ND
       IF (D(I) .GT. 2E-16_DP) THEN
          RI = REAL(I, DP)
          DU(3) = DU(3) + TWO * RI / D(I) * (JAC - ONE) ** (2 * I - 1)
       END IF
    END DO

    RETURN
  END SUBROUTINE KDU

  ! ************************************************************************* !

  SUBROUTINE KD2U(NC, ND, C, D, I1B, I2B, JAC, D2U)
    ! ----------------------------------------------------------------------- !
    ! SECOND DERIVATIVE OF ENERGY: D2U/DIiBDIjB
    ! ----------------------------------------------------------------------- !
    INTEGER, INTENT(IN) :: NC, ND
    REAL(KIND=DP), INTENT(IN) :: C(NC), D(ND), I1B, I2B, JAC
    REAL(KIND=DP), INTENT(OUT) :: D2U(6)
    INTEGER :: I, J, L
    REAL(KIND=DP) :: C1, C2, RI, RJ
    ! ----------------------------------------------------------------------- !

    D2U = ZERO

    ! --- DISTORTION
    DO L = 1, NC
       I = NCI(L); J = NCJ(L)
       IF (I + J < 1) CYCLE
       RI = REAL(I, DP); RJ = REAL(J, DP)

       ! DU_11 (D2U(1))
       C1 = PPOW(I1B - THREE, I - 2)
       C2 = PPOW(I2B - THREE, J)
       D2U(1) = D2U(1) + RI * (RI - ONE) * C(L) * C1 * C2

       ! DU_22 (D2U(2))
       C1 = PPOW(I1B - THREE, I)
       C2 = PPOW(I2B - THREE, J - 2)
       D2U(2) = D2U(2) + RJ * (RJ - ONE) * C(L) * C1 * C2

       ! DU_12 (D2U(4))
       C1 = PPOW(I1B - THREE, I - 1)
       C2 = PPOW(I2B - THREE, J - 1)
       D2U(4) = D2U(4) + RI * RJ * C(L) * C1 * C2
    END DO

    ! --- DILATATION
    DO I = 1,ND
       RI = REAL(I, DP)
       IF (D(L) .GT. 1E+20_DP) THEN
          C1 = PPOW(JAC - ONE, 2 * (I - 1))
          DU(3) = DU(3) + TWO * RI * (TWO * RI - ONE) / D(L) * C1
       END IF
    END DO

    RETURN
  END SUBROUTINE KD2U

  ! ************************************************************************* !

  SUBROUTINE KD3U(NC, ND, C, D, I1B, I2B, JAC, D3U)
    ! ----------------------------------------------------------------------- !
    ! CALCULATE THE THIRD PARTIAL DERIVATIVES WITH RESPECT TO INVARIANTS
    ! ----------------------------------------------------------------------- !
    INTEGER, INTENT(IN) :: NC, ND
    REAL(KIND=DP), INTENT(IN) :: C(NC), D(ND), I1B, I2B, JAC
    REAL(KIND=DP), INTENT(OUT) :: D3U(6)
    INTEGER :: I
    REAL(KIND=DP) :: RI, C1
    ! ----------------------------------------------------------------------- !
    D3U = ZERO
    DO I = 1,ND
       IF (D(I) > 1.E-16_DP) THEN
          C1 = PPOW(JAC - ONE, 2 * I - 3)
          D3U(6) = D3U(6) + FOUR * RI * (TWO * RI - ONE) * (RI - ONE) / D(I) * C1
       END IF
    END DO
    RETURN
  END SUBROUTINE KD3U

  REAL(KIND=DP) FUNCTION PPOW(X, M)
    INTEGER, INTENT(IN) :: M
    REAL(KIND=DP), INTENT(IN) :: X
    ! ----------------------------------------------------------------------- !
    ! WHEN M IS LESS THAN OR EQUAL TO ZERO, SET EQUAL TO 0 REGARDLESS
    ! OF ROOT VALUE
    ! ----------------------------------------------------------------------- !
    IF (M == 0) THEN
       PPOW = ONE
    ELSE IF (M < 0) THEN
       PPOW = ZERO ! ONE
    ELSE
       PPOW = X ** M
    END IF
    RETURN
  END FUNCTION PPOW

END SUBROUTINE UHYPER
