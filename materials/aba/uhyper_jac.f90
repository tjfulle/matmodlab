MODULE HYPERELASTIC

  IMPLICIT NONE
  INTEGER, PARAMETER :: DP=SELECTED_REAL_KIND(14)
  REAL(KIND=DP), PARAMETER :: HALF=.5_DP, MONE=-1._DP
  REAL(KIND=DP), PARAMETER :: ZERO=0._DP, ONE=1._DP, TWO=2._DP, THREE=3._DP
  REAL(KIND=DP), PARAMETER :: FOUR=4._DP
  REAL(KIND=DP), PARAMETER :: THIRD=ONE/THREE, P2THIRD=TWO/THREE, P3HALF=THREE/TWO
  REAL(KIND=DP), PARAMETER :: FOURTH=ONE/FOUR
  REAL(KIND=DP), PARAMETER :: IDENTITY(6)=[ONE, ONE, ONE, ZERO, ZERO, ZERO]
  REAL(KIND=DP), PARAMETER :: IIMI4(6,6) = RESHAPE(&
                         [ZERO,  ONE,  ONE, ZERO, ZERO, ZERO, &
                           ONE, ZERO,  ONE, ZERO, ZERO, ZERO, &
                           ONE,  ONE, ZERO, ZERO, ZERO, ZERO, &
                          ZERO, ZERO, ZERO, MONE, ZERO, ZERO, &
                          ZERO, ZERO, ZERO, ZERO, MONE, ZERO, &
                          ZERO, ZERO, ZERO, ZERO, ZERO, MONE], SHAPE(IIMI4))

CONTAINS

  SUBROUTINE HYPEREL(NPROPS, PROPS, TEMP, NOEL, CMNAME, INCMPFLAG, &
       NSTATEV, STATEV, NFIELDV, FIELDV, FIELDVINC, JAC, C, PK2, DDTDDC)
    ! ----------------------------------------------------------------------- !
    ! HYPERELASTIC MATERIAL MODEL
    ! CALLS A UYHPER MODEL AND FORMULATES THE STRESS AND STIFFNESS ARRAYS
    ! ----------------------------------------------------------------------- !
    ! --- PASSED ARGUMENTS
    CHARACTER*8, INTENT(IN) :: CMNAME
    INTEGER, INTENT(IN) :: NPROPS, NOEL, NSTATEV, INCMPFLAG, NFIELDV
    REAL(KIND=DP), INTENT(IN) :: PROPS(NPROPS), TEMP, STATEV(NSTATEV)
    REAL(KIND=DP), INTENT(INOUT) :: FIELDV(NFIELDV), FIELDVINC(NFIELDV)
    REAL(KIND=DP), INTENT(IN) :: JAC, C(6)
    REAL(KIND=DP), INTENT(OUT) :: PK2(6)
    REAL(KIND=DP), INTENT(OUT), OPTIONAL :: DDTDDC(6,6)
    ! --- LOCAL VARIABLES
    INTEGER :: IJ, I
    REAL(KIND=DP) :: U(2), DU(3), D2U(6), D3U(6)
    REAL(KIND=DP) :: I1, I2, I3, I1B, I2B
    REAL(KIND=DP) :: SCALE, CINV(6), A(3), DA(3,6), B(3,6), DB(3,6,6)
    REAL(KIND=DP), DIMENSION(6,6) :: TERM1, TERM2, CII, ICI, CICI, LCICI, CCI, CIC
    ! ----------------------------------------------------------- HYPEREL --- !

    ! INVARIANTS OF C
    CALL INVARS(C, I1, I2, I3)

    ! INVARIANTS OF CBAR
    SCALE = JAC ** (-ONE / THREE)
    I1B = I1 * (SCALE ** 2)
    I2B = I2 * (SCALE ** 4)

    CALL UHYPER(I1B, I2B, JAC, U, DU, D2U, D3U, TEMP, NOEL, CMNAME, &
         INCMPFLAG, NSTATEV, STATEV, NFIELDV, FIELDV, FIELDVINC, &
         NPROPS, PROPS)

    ! UPDATE THE STRESS AND MATERIAL STIFFNESS
    ! THE NOTATION BELOWS FOLLOWS THE APPENDIX OF ***TITLE*** WHERE
    ! THE SECOND PIOLA-KIRCHOFF STRESS IS GIVEN BY
    !
    !                           DU   DU  DIB
    !                   PK2 = 2 -- = --- --- = A.B
    !                           DC   DIB DC
    !
    ! WHERE A IS AN ARRAY CONTAINING THE DERIVATIVES OF U WRT ISOCHORIC
    ! INVARIANTS AND B IS AN ARRAY OF TENSORS CONTAINING THE DERIVATIVES OF
    ! THE ISOCHORIC INVARIANTS WRT C
    !
    ! USING THE NOTATION FROM ABOVE, THE MATERIAL STIFFNESS L IS
    !
    !                      L  = 4(DA.B + A.DB)
    !
    !            DA   DA  DIB   D2U
    ! WHERE DA = -- = --- --- = ----.B
    !            DC   DIB DC    DIB2
    !
    !          DB
    ! AND DB = --
    !          DC
    !
    ! NOTE THAT DA AND B CONTAIN SECOND ORDER TENSORS SO THAT THE ARRAY DA.B
    ! IS AN ARRAY OF DYADIC PRODUCTS OF SECOND ORDER TENSORS (FOURTH ORDER
    ! TENSORS). DB IS AN ARRAY OF FOURTH ORDER TENSORS.

    ! HELPER QUANTITIES
    CINV = SYMINV(C)

    A = DU
    B(1,1:6) = SCALE ** 2 * (IDENTITY - THIRD * I1 * CINV)
    B(2,1:6) = SCALE ** 4 * (I1 * IDENTITY - C - P2THIRD * I2 * CINV)
    B(3,1:6) = HALF * JAC * CINV

    ! SECOND PIOLA-KIRCHHOFF STRESS: DERIVATIVE OF ENERGY WRT C
    FORALL(IJ=1:6) PK2(IJ) = TWO * SUM(A(1:3) * B(1:3, IJ))

    IF (.NOT. PRESENT(DDTDDC)) RETURN

    CICI = DYAD(CINV, CINV)
    CIC = DYAD(CINV, C)
    CCI = DYAD(C, CINV)
    LCICI = SYMLEAF(CINV, CINV)
    ICI = DYAD(IDENTITY, CINV)
    CII = DYAD(CINV, IDENTITY)
    DA=ZERO
    DA(1,1:6) = D2U(1) * B(1,1:6) + D2U(4) * B(2,1:6) + D2U(5) * B(3,1:6)
    DA(2,1:6) = D2U(4) * B(1,1:6) + D2U(2) * B(2,1:6) + D2U(6) * B(3,1:6)
    DA(3,1:6) = D2U(5) * B(1,1:6) + D2U(6) * B(2,1:6) + D2U(3) * B(3,1:6)

    DB = ZERO
    DB(1,1:6,1:6) = THIRD * SCALE ** 2 * (-CII - ICI + I1 * (THIRD * CICI + LCICI))
    TERM1 = -I1 * (CII + ICI) + I2 * (P2THIRD * CICI + LCICI)
    TERM2 = CIC + CCI + P3HALF * IIMI4
    DB(2,1:6,1:6) = P2THIRD * SCALE ** 4 * (TERM1 + TERM2)
    DB(3,1:6,1:6) = FOURTH * JAC * (CICI - TWO * LCICI)

    DDTDDC = ZERO
    DO I=1,3
       DDTDDC = DDTDDC + FOUR * (DYAD(DA(I,1:6), B(I,1:6)) + A(I) * DB(I,1:6,1:6))
    END DO
    DDTDDC = HALF * (DDTDDC + TRANSPOSE(DDTDDC))

  END SUBROUTINE HYPEREL

  ! ************************************************************************* !

  SUBROUTINE JACOBIAN(NPROPS, PROPS, TEMP, NOEL, CMNAME, INCMPFLAG, &
       NSTATEV, STATEV, NFIELDV, FIELDV, FIELDVINC, C, PK2, DDTDDC)
    ! THIS PROCEDURE NUMERICALLY COMPUTES AND RETURNS THE JACOBIAN MATRIX
    !
    !                      J = (JIJ) = (DSIGI/DEPSJ).
    !
    ! THE COMPONENTS OF JSUB ARE COMPUTED NUMERICALLY USING A CENTERED
    ! DIFFERENCING SCHEME WHICH REQUIRES TWO CALLS TO THE MATERIAL MODEL
    ! SUBROUTINE FOR EACH ELEMENT OF PK2. THE CENTERING IS ABOUT THE POINT
    !                       EPS = EPSOLD + D * DT,
    ! WHERE D IS THE RATE-OF-STRAIN ARRAY.
    CHARACTER*8, INTENT(IN) :: CMNAME
    INTEGER, INTENT(IN) :: NPROPS, NOEL, NSTATEV, INCMPFLAG, NFIELDV
    REAL(KIND=DP), INTENT(IN) :: PROPS(NPROPS), C(6), TEMP, STATEV(NSTATEV)
    REAL(KIND=DP), INTENT(INOUT) :: FIELDV(NFIELDV), FIELDVINC(NFIELDV)
    REAL(KIND=DP), INTENT(OUT) :: PK2(6)
    REAL(KIND=DP), INTENT(OUT) :: DDTDDC(6,6)
    INTEGER :: N
    REAL(KIND=DP) :: E(6), DE
    REAL(KIND=DP) :: EP(6), CP(6), JP, SVP(NSTATEV), TP(6)
    REAL(KIND=DP) :: EM(6), CM(6), JM, SVM(NSTATEV), TM(6)
    INTEGER, PARAMETER :: V(6)=(/1,2,3,4,5,6/)
    ! ---------------------------------------------------------- JACOBIAN --- !
    E = HALF * (C - IDENTITY)
    DE = SQRT(EPSILON(ONE))
    DDTDDC = ZERO
    DO N = 1, 6
       EP = E
       EP(V(N)) = EP(V(N)) + DE / TWO
       CP = TWO * EP + IDENTITY
       JP = SQRT(SYMDET(CP))
       SVP = STATEV
       CALL HYPEREL(NPROPS, PROPS, TEMP, NOEL, CMNAME, INCMPFLAG, &
            NSTATEV, SVP, NFIELDV, FIELDV, FIELDVINC, JP, CP, TP)
       EM = E
       EM(V(N)) = EM(V(N)) + DE / TWO
       CM = TWO * EM + IDENTITY
       JM = SQRT(SYMDET(CM))
       SVM = STATEV
       CALL HYPEREL(NPROPS, PROPS, TEMP, NOEL, CMNAME, INCMPFLAG, &
            NSTATEV, SVM, NFIELDV, FIELDV, FIELDVINC, JM, CM, TM)
       DDTDDC(:, N) = (TP(V) - TM(V) ) / DE
    END DO

    WHERE (ABS(DDTDDC) / MAXVAL(DDTDDC) < 1.E-06_DP)
       DDTDDC = ZERO
    END WHERE
    ! MATMODLAB USES TENSOR VALUES - NOT ENGINEERING.  UNCOMMENT FOR ABAQUS
!    DDTDDC(1:3,4:6) = TWO*DDTDDC(1:3,4:6)
!    DDTDDC(4:6,1:3) = TWO*DDTDDC(4:6,1:3)
!    DDTDDC(4:6,4:6) = HALF*DDTDDC(4:6,4:6)
  END SUBROUTINE JACOBIAN

  ! ************************************************************************* !

  ! ************************************************************************* !
  ! ************************************************** UTILITY PROCEDURES *** !
  ! ************************************************************************* !
  FUNCTION TRACE(A)
    REAL(KIND=DP) :: TRACE
    REAL(KIND=DP), INTENT(IN) :: A(:)
    TRACE = SUM(A(1:3))
    RETURN
  END FUNCTION TRACE
  ! ************************************************************************* !
  FUNCTION MAG(A)
    REAL(KIND=DP) :: MAG
    REAL(KIND=DP), INTENT(IN) :: A(:)
    MAG = SQRT(DBD(A, A))
    RETURN
  END FUNCTION MAG
  ! ************************************************************************* !
  FUNCTION ISO(A, METRIC)
    REAL(KIND=DP), INTENT(IN) :: A(:)
    REAL(KIND=DP), INTENT(IN), OPTIONAL :: METRIC(:)
    REAL(KIND=DP) :: ISO(SIZE(A))
    REAL(KIND=DP), ALLOCATABLE :: I(:)
    INTEGER :: N
    N = SIZE(A)
    ALLOCATE(I(N))
    IF (PRESENT(METRIC)) THEN
       I = METRIC(:N)
    ELSE
       I(1:3) = IDENTITY(:N)
    END IF
    ISO = SUM(I * A) / THREE * I
    DEALLOCATE(I)
    RETURN
  END FUNCTION ISO
  ! ************************************************************************* !
  FUNCTION DEV(A)
    REAL(KIND=DP), INTENT(IN) :: A(:)
    REAL(KIND=DP) :: DEV(SIZE(A))
    DEV = A - ISO(A)
    RETURN
  END FUNCTION DEV
  ! ************************************************************************* !
  FUNCTION UNITDEV(A)
    REAL(KIND=DP), INTENT(IN) :: A(:)
    REAL(KIND=DP) :: UNITDEV(SIZE(A))
    REAL(KIND=DP) :: DEV(SIZE(A))
    DEV = A - ISO(A)
    UNITDEV = DEV / MAG(DEV)
    RETURN
  END FUNCTION UNITDEV
  ! ************************************************************************* !
  FUNCTION DBD(A, B)
    REAL(KIND=DP) :: DBD
    REAL(KIND=DP), INTENT(IN) :: A(:), B(:)
    REAL(KIND=DP), ALLOCATABLE :: W(:)
    INTEGER :: N
    N = SIZE(A)
    ALLOCATE(W(N))
    W(1:3) = ONE
    W(4:) = TWO
    DBD = SUM(A * B * W)
    DEALLOCATE(W)
    RETURN
  END FUNCTION DBD
  ! ************************************************************************* !
  FUNCTION DOT(A, B)
    ! ----------------------------------------------------------------------- !
    ! DOT PRODUCT OF SECOND ORDER TENSOR A WITH B
    ! ----------------------------------------------------------------------- !
    REAL(KIND=DP), INTENT(IN) :: A(6), B(:)
    REAL(KIND=DP) :: DOT(SIZE(B))
    INTEGER :: M
    M = SIZE(B)
    IF (M == 6) THEN
       ! DOT PRODUCT OF 2 SYMMETRIC MATRICES STORED AS ARRAYS
       DOT(1) = A(1)*B(1) + A(4)*B(4) + A(5)*B(5)
       DOT(2) = A(2)*B(2) + A(4)*B(4) + A(6)*B(6)
       DOT(3) = A(3)*B(3) + A(5)*B(5) + A(6)*B(6)
       DOT(4) = A(1)*B(4) + A(4)*B(2) + A(5)*B(6)
       DOT(5) = A(1)*B(5) + A(4)*B(6) + A(5)*B(3)
       DOT(6) = A(2)*B(6) + A(4)*B(5) + A(6)*B(3)
    ELSE IF (M == 3) THEN
       DOT(1) = A(1) * B(1) + A(4) * B(2) + A(5) * B(3)
       DOT(2) = A(4) * B(1) + A(2) * B(2) + A(6) * B(3)
       DOT(3) = A(5) * B(1) + A(6) * B(2) + A(3) * B(3)
    ELSE
       PRINT *, "ERROR: EXPECTED DOT PRODUCT OF SYMMETRIC MATRICES AND ARRAYS"
       CALL XIT
    END IF
    RETURN
  END FUNCTION DOT
  ! ************************************************************************* !
  FUNCTION DYAD(A, B)
    ! ----------------------------------------------------------------------- !
    ! DYADIC PRODUCT OF A AND B
    ! ----------------------------------------------------------------------- !
    REAL(KIND=DP), INTENT(IN) :: A(:), B(:)
    REAL(KIND=DP) :: DYAD(SIZE(A),SIZE(B))
    INTEGER :: I, J, M, N
    N = SIZE(A); M = SIZE(B)
    FORALL(I=1:N, J=1:N) DYAD(I,J) = A(I) * B(J)
    RETURN
  END FUNCTION DYAD
  ! ************************************************************************* !
  FUNCTION SYMLEAF(A, B)
    ! ----------------------------------------------------------------------- !
    ! SYMMETRIC LEAF OF A AND B
    ! ----------------------------------------------------------------------- !
    REAL(KIND=DP), INTENT(IN) :: A(6), B(6)
    REAL(KIND=DP) :: SYMLEAF(6,6)
    REAL(KIND=DP) :: FAC
    FAC = HALF
    SYMLEAF(1,1) = A(1) * B(1)
    SYMLEAF(1,2) = A(4) * B(4)
    SYMLEAF(1,3) = A(5) * B(5)
    SYMLEAF(1,4) = FAC * A(1) * B(4)  +  FAC * A(4) * B(1)
    SYMLEAF(1,5) = FAC * A(1) * B(5)  +  FAC * A(5) * B(1)
    SYMLEAF(1,6) = FAC * A(4) * B(5)  +  FAC * A(5) * B(4)
    SYMLEAF(2,1) = A(4) * B(4)
    SYMLEAF(2,2) = A(2) * B(2)
    SYMLEAF(2,3) = A(6) * B(6)
    SYMLEAF(2,4) = FAC * A(2) * B(4)  +  FAC * A(4) * B(2)
    SYMLEAF(2,5) = FAC * A(4) * B(6)  +  FAC * A(6) * B(4)
    SYMLEAF(2,6) = FAC * A(2) * B(6)  +  FAC * A(6) * B(2)
    SYMLEAF(3,1) = A(5) * B(5)
    SYMLEAF(3,2) = A(6) * B(6)
    SYMLEAF(3,3) = A(3) * B(3)
    SYMLEAF(3,4) = FAC * A(5) * B(6)  +  FAC * A(6) * B(5)
    SYMLEAF(3,5) = FAC * A(3) * B(5)  +  FAC * A(5) * B(3)
    SYMLEAF(3,6) = FAC * A(3) * B(6)  +  FAC * A(6) * B(3)
    SYMLEAF(4,1) = A(4) * B(1)
    SYMLEAF(4,2) = A(2) * B(4)
    SYMLEAF(4,3) = A(6) * B(5)
    SYMLEAF(4,4) = FAC * A(2) * B(1)  +  FAC * A(4) * B(4)
    SYMLEAF(4,5) = FAC * A(4) * B(5)  +  FAC * A(6) * B(1)
    SYMLEAF(4,6) = FAC * A(2) * B(5)  +  FAC * A(6) * B(4)
    SYMLEAF(5,1) = A(5) * B(1)
    SYMLEAF(5,2) = A(6) * B(4)
    SYMLEAF(5,3) = A(3) * B(5)
    SYMLEAF(5,4) = FAC * A(5) * B(4)  +  FAC * A(6) * B(1)
    SYMLEAF(5,5) = FAC * A(3) * B(1)  +  FAC * A(5) * B(5)
    SYMLEAF(5,6) = FAC * A(3) * B(4)  +  FAC * A(6) * B(5)
    SYMLEAF(6,1) = A(5) * B(4)
    SYMLEAF(6,2) = A(6) * B(2)
    SYMLEAF(6,3) = A(3) * B(6)
    SYMLEAF(6,4) = FAC * A(5) * B(2)  +  FAC * A(6) * B(4)
    SYMLEAF(6,5) = FAC * A(3) * B(4)  +  FAC * A(5) * B(6)
    SYMLEAF(6,6) = FAC * A(3) * B(2)  +  FAC * A(6) * B(6)
    RETURN
  END FUNCTION SYMLEAF
  ! ************************************************************************* !
  FUNCTION TRANSQ(T)
    ! ----------------------------------------------------------------------- !
    ! CONSTRUCT TRANFORMATION MATRIX Q FROM T
    ! ----------------------------------------------------------------------- !
    ! NOTES
    ! -----
    ! FOR AN ARBITRARY CHANGE OF BASIS, 2ND AND 4TH ORDER TENSORS TRANSFORM
    ! ACCORDING TO
    !                             S' = T.S.TT
    !                           C' = T.T.C.TT.TT
    ! IN VOIGHT NOTATION
    !                               S' = Q.S
    !                             C' = Q.C.QT
    ! COMPARING EQUATIONS, WE GET THE FOLLOWING FOR Q
    ! ----------------------------------------------------------------------- !
    REAL(KIND=DP) :: TRANSQ(6,6)
    REAL(KIND=DP), INTENT(IN) :: T(3,3)
    TRANSQ = ZERO
    TRANSQ(1,1) = T(1,1) ** 2
    TRANSQ(1,2) = T(1,2) ** 2
    TRANSQ(1,3) = T(1,3) ** 2
    TRANSQ(1,4) = TWO * T(1,1) * T(1,2)
    TRANSQ(1,5) = TWO * T(1,1) * T(1,3)
    TRANSQ(1,6) = TWO * T(1,2) * T(1,3)
    TRANSQ(2,1) = T(2,1) ** 2
    TRANSQ(2,2) = T(2,2) ** 2
    TRANSQ(2,3) = T(2,3) ** 2
    TRANSQ(2,4) = TWO * T(2,1) * T(2,2)
    TRANSQ(2,5) = TWO * T(2,1) * T(2,3)
    TRANSQ(2,6) = TWO * T(2,2) * T(2,3)
    TRANSQ(3,1) = T(3,1) ** 2
    TRANSQ(3,2) = T(3,2) ** 2
    TRANSQ(3,3) = T(3,3) ** 2
    TRANSQ(3,4) = TWO * T(3,1) * T(3,2)
    TRANSQ(3,5) = TWO * T(3,1) * T(3,3)
    TRANSQ(3,6) = TWO * T(3,2) * T(3,3)
    TRANSQ(4,1) = T(1,1) * T(2,1)
    TRANSQ(4,2) = T(1,2) * T(2,2)
    TRANSQ(4,3) = T(1,3) * T(2,3)
    TRANSQ(4,4) = T(1,1) * T(2,2) + T(1,2) * T(2,1)
    TRANSQ(4,5) = T(1,1) * T(2,3) + T(1,3) * T(2,1)
    TRANSQ(4,6) = T(1,2) * T(2,3) + T(1,3) * T(2,2)
    TRANSQ(5,1) = T(1,1) * T(3,1)
    TRANSQ(5,2) = T(1,2) * T(3,2)
    TRANSQ(5,3) = T(1,3) * T(3,3)
    TRANSQ(5,4) = T(1,1) * T(3,2) + T(1,2) * T(3,1)
    TRANSQ(5,5) = T(1,1) * T(3,3) + T(1,3) * T(3,1)
    TRANSQ(5,6) = T(1,2) * T(3,3) + T(1,3) * T(3,2)
    TRANSQ(6,1) = T(2,1) * T(3,1)
    TRANSQ(6,2) = T(2,2) * T(3,2)
    TRANSQ(6,3) = T(2,3) * T(3,3)
    TRANSQ(6,4) = T(2,1) * T(3,2) + T(2,2) * T(3,1)
    TRANSQ(6,5) = T(2,1) * T(3,3) + T(2,3) * T(3,1)
    TRANSQ(6,6) = T(2,2) * T(3,3) + T(2,3) * T(3,2)
  END FUNCTION TRANSQ
  ! ************************************************************************* !
  SUBROUTINE INVARS(A, I1, I2, I3)
    ! ----------------------------------------------------------------------- !
    ! INVARIANTS OF SYMMETRIC SECOND ORDER TENSOR A
    ! ----------------------------------------------------------------------- !
    REAL(KIND=DP), INTENT(IN) :: A(:)
    REAL(KIND=DP), INTENT(OUT) :: I1, I2, I3
    REAL(KIND=DP) :: TRASQ
    I1 = TRACE(A)
    TRASQ = DBD(A, A)
    I2 = HALF * (I1 ** 2 - TRASQ)
    I3 = SYMDET(A)
  END SUBROUTINE INVARS
  ! ************************************************************************* !
  FUNCTION SYMDET(A)
    REAL(KIND=DP) :: SYMDET
    REAL(KIND=DP), INTENT(IN) :: A(6)
    SYMDET = A(1) * A(2) * A(3) - A(1) * A(5) ** 2 &
           - A(2) * A(6) ** 2 - A(3) * A(4) ** 2 &
           + TWO * A(4) * A(5) * A(6)
  END FUNCTION SYMDET
  ! ************************************************************************* !
  FUNCTION ASMAT(A)
    REAL(KIND=DP) :: ASMAT(3,3)
    REAL(KIND=DP), INTENT(IN) :: A(:)
    ASMAT = ZERO
    ASMAT(1,1) = A(1)
    ASMAT(2,2) = A(2)
    ASMAT(3,3) = A(3)
    ASMAT(1,2) = A(4)
    IF (SIZE(A) > 4) THEN
       ASMAT(1,3) = A(5)
       ASMAT(2,3) = A(6)
    END IF
    ASMAT(2,1) = ASMAT(1,2)
    ASMAT(3,1) = ASMAT(1,3)
    ASMAT(3,2) = ASMAT(2,3)
  END FUNCTION ASMAT
  ! ************************************************************************* !
  FUNCTION ASARRAY(A, N)
    INTEGER :: N
    REAL(KIND=DP) :: ASARRAY(N)
    REAL(KIND=DP), INTENT(IN) :: A(3,3)
    ASARRAY = ZERO
    ASARRAY(1) = A(1,1)
    ASARRAY(2) = A(2,2)
    ASARRAY(3) = A(3,3)
    ASARRAY(4) = A(1,2)
    IF (N > 4) THEN
       ASARRAY(5) = A(1,3)
       ASARRAY(6) = A(2,3)
    END IF
  END FUNCTION ASARRAY
  ! ************************************************************************* !
  FUNCTION SYMINV(AARG)
    REAL(KIND=DP), INTENT(IN) :: AARG(:)
    REAL(KIND=DP) :: SYMINV(SIZE(AARG))
    INTEGER :: N
    REAL(KIND=DP) :: A(3,3)
    A = ASMAT(AARG)
    N = SIZE(AARG)
    SYMINV = ASARRAY(INV(A), N)
  END FUNCTION SYMINV
  ! ************************************************************************* !
  FUNCTION DET(A)
    REAL(KIND=DP) :: DET
    REAL(KIND=DP), INTENT(IN) :: A(3,3)
    DET = A(1,1) * A(2,2) * A(3,3) - A(1,2) * A(2,1) * A(3,3) &
        + A(1,2) * A(2,3) * A(3,1) + A(1,3) * A(3,2) * A(2,1) &
        - A(1,3) * A(3,1) * A(2,2) - A(2,3) * A(3,2) * A(1,1)
  END FUNCTION DET
  ! ************************************************************************* !
  FUNCTION INV(A)
    REAL(KIND=DP) :: INV(3,3)
    REAL(KIND=DP), INTENT(IN) :: A(3,3)
    REAL(KIND=DP) :: DETA
    DETA = DET(A)
    INV(1,1) =  A(2,2) * A(3,3) - A(2,3) * A(3,2)
    INV(1,2) = -A(1,2) * A(3,3) + A(1,3) * A(3,2)
    INV(1,3) =  A(1,2) * A(2,3) - A(1,3) * A(2,2)
    INV(2,1) = -A(2,1) * A(3,3) + A(2,3) * A(3,1)
    INV(2,2) =  A(1,1) * A(3,3) - A(1,3) * A(3,1)
    INV(2,3) = -A(1,1) * A(2,3) + A(1,3) * A(2,1)
    INV(3,1) =  A(2,1) * A(3,2) - A(2,2) * A(3,1)
    INV(3,2) = -A(1,1) * A(3,2) + A(1,2) * A(3,1)
    INV(3,3) =  A(1,1) * A(2,2) - A(1,2) * A(2,1)
    INV = INV / DETA
    RETURN
  END FUNCTION INV
END MODULE HYPERELASTIC

! *************************************************************************** !

SUBROUTINE UMAT(STRESS,STATEV,DDSDDE,SSE,SPD,SCD, &
     RPL,DDSDDT,DRPLDE,DRPLDT, &
     STRAN,DSTRAN,TIME,DTIME,TEMP,DTEMP,PREDEF,DPRED,CMNAME, &
     NDI,NSHR,NTENS,NSTATEV,PROPS,NPROPS,COORDS,DROT,PNEWDT, &
     CELENT,F0,F1,NOEL,NPT,LAYER,KSPT,KSTEP,KINC)
  ! ------------------------------------------------------------------------- !
  !     UMAT INTERFACE FOR FINITE STRAIN PROPELLANT MODEL
  ! ------------------------------------------------------------------------- !
  !     CANNOT BE USED FOR PLANE STRESS
  !     PROPS(1)  - C10
  !     PROPS(2)  - K1
  ! ------------------------------------------------------------------------- !
  USE HYPERELASTIC, ONLY: DP, ONE, TWO, THREE, HYPEREL, JACOBIAN, TRANSQ
  IMPLICIT NONE

  ! --- PASSED ARGUMENTS
  ! ------------------------------------------------------------------------- !
  CHARACTER*8, INTENT(IN) :: CMNAME
  INTEGER, INTENT(IN) :: NDI, NSHR, NTENS, NSTATEV, NPROPS
  INTEGER, INTENT(IN) :: NOEL, NPT, LAYER, KSPT, KSTEP, KINC
  REAL(KIND=DP), INTENT(IN) :: SSE, SPD, SCD, RPL, DRPLDT, TIME, DTIME
  REAL(KIND=DP), INTENT(IN) :: TEMP, DTEMP, PNEWDT, CELENT
  REAL(KIND=DP), INTENT(INOUT) :: STRESS(NTENS), STATEV(NSTATEV)
  REAL(KIND=DP), INTENT(INOUT) :: DDSDDE(NTENS, NTENS)
  REAL(KIND=DP), INTENT(INOUT) :: DDSDDT(NTENS), DRPLDE(NTENS)
  REAL(KIND=DP), INTENT(IN) :: STRAN(NTENS), DSTRAN(NTENS)
  REAL(KIND=DP), INTENT(IN) :: PREDEF(1), DPRED(1), PROPS(NPROPS), COORDS(3)
  REAL(KIND=DP), INTENT(IN) :: DROT(3, 3), F0(3, 3), F1(3, 3)
  ! --- LOCAL VARIABLES
  INTEGER, PARAMETER :: NFIELDV=1, INCMPFLAG=1
  REAL(KIND=DP) :: JAC, C(6), T(6), Q(6,6), DDTDDC(6,6)
  REAL(KIND=DP) :: FIELDV(NFIELDV), FIELDVINC(NFIELDV)
  ! ---------------------------------------------------------------- UMAT --- !

  ! DEFORMATION TENSOR
  ! C = FT.F
  C(1) = F1(1,1)**2 + F1(2,1)**2 + F1(3,1)**2
  C(2) = F1(1,2)**2 + F1(2,2)**2 + F1(3,2)**2
  C(3) = F1(1,3)**2 + F1(2,3)**2 + F1(3,3)**2
  C(4) = F1(1,1)*F1(1,2) + F1(2,1)*F1(2,2) + F1(3,1)*F1(3,2)
  C(5) = F1(1,1)*F1(1,3) + F1(2,1)*F1(2,3) + F1(3,1)*F1(3,3)
  C(6) = F1(1,2)*F1(1,3) + F1(2,2)*F1(2,3) + F1(3,2)*F1(3,3)

  ! JACOBIAN
  ! JAC = DET(F)
  JAC = F1(1,1) * F1(2,2) * F1(3,3) - F1(1,2) * F1(2,1) * F1(3,3) &
      + F1(1,2) * F1(2,3) * F1(3,1) + F1(1,3) * F1(3,2) * F1(2,1) &
      - F1(1,3) * F1(3,1) * F1(2,2) - F1(2,3) * F1(3,2) * F1(1,1)

  CALL HYPEREL(NPROPS, PROPS, TEMP, NOEL, CMNAME, INCMPFLAG, &
       NSTATEV, STATEV, NFIELDV, FIELDV, FIELDVINC, JAC, C, T, DDTDDC)
  !CALL JACOBIAN(NPROPS, PROPS, TEMP, NOEL, CMNAME, INCMPFLAG, &
  !     NSTATEV, STATEV, NFIELDV, FIELDV, FIELDVINC, C, T, DDTDDC)

  Q = TRANSQ(F1)
  DDSDDE = MATMUL(Q, MATMUL(DDTDDC, TRANSPOSE(Q))) / JAC

  IF (DTIME < EPSILON(ONE)) RETURN

  STRESS = MATMUL(Q, T) / JAC

END SUBROUTINE UMAT
