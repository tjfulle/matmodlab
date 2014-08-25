MODULE HYPERELASTIC

  IMPLICIT NONE
  INTEGER, PARAMETER :: DP=SELECTED_REAL_KIND(14)
  REAL(KIND=DP), PARAMETER :: HALF=.5_DP, MONE=-1._DP
  REAL(KIND=DP), PARAMETER :: ZERO=0._DP, ONE=1._DP, TWO=2._DP, THREE=3._DP
  REAL(KIND=DP), PARAMETER :: FOUR=4._DP
  REAL(KIND=DP), PARAMETER :: THIRD=ONE/THREE, P2THIRD=TWO/THREE, P3HALF=THREE/TWO
  REAL(KIND=DP), PARAMETER :: FOURTH=ONE/FOUR, SIX=6._DP
  REAL(KIND=DP), PARAMETER :: IDENTITY(6)=[ONE, ONE, ONE, ZERO, ZERO, ZERO]
  REAL(KIND=DP), PARAMETER :: IIMI4(6,6) = RESHAPE(&
                         [ZERO,  ONE,  ONE, ZERO, ZERO, ZERO, &
                           ONE, ZERO,  ONE, ZERO, ZERO, ZERO, &
                           ONE,  ONE, ZERO, ZERO, ZERO, ZERO, &
                          ZERO, ZERO, ZERO, MONE, ZERO, ZERO, &
                          ZERO, ZERO, ZERO, ZERO, MONE, ZERO, &
                          ZERO, ZERO, ZERO, ZERO, ZERO, MONE], SHAPE(IIMI4))

CONTAINS

  SUBROUTINE HYPEREL(NPROPS, PROPS, TEMP, F1, NOEL, CMNAME, INCMPFLAG, &
       NSTATV, STATEV, NFLDV, FIELDV, DFIELDV, STRESS, DDSDDE)
    ! ----------------------------------------------------------------------- !
    ! HYPERELASTIC MATERIAL MODEL
    ! CALLS A UYHPER MODEL AND FORMULATES THE STRESS AND STIFFNESS ARRAYS
    ! ----------------------------------------------------------------------- !
    ! --- PASSED ARGUMENTS
    CHARACTER*8, INTENT(IN) :: CMNAME
    INTEGER, INTENT(IN) :: NPROPS, NOEL, NSTATV, INCMPFLAG, NFLDV
    REAL(KIND=DP), INTENT(IN) :: PROPS(NPROPS), TEMP, F1(3,3)
    REAL(KIND=DP), INTENT(INOUT) :: STATEV(NSTATV), FIELDV(NFLDV), DFIELDV(NFLDV)
    REAL(KIND=DP), INTENT(OUT) :: STRESS(6)
    REAL(KIND=DP), INTENT(OUT), OPTIONAL :: DDSDDE(6,6)
    ! --- LOCAL VARIABLES
    INTEGER :: IJ, I
    REAL(KIND=DP) :: JAC, C(6), PK2(6), DDTDDC(6,6), Q(6,6)
    REAL(KIND=DP) :: U(2), DU(3), D2U(6), D3U(6)
    REAL(KIND=DP) :: I1, I2, I3, I1B, I2B
    REAL(KIND=DP) :: SCALE, CINV(6), A(3), DA(3,6), B(3,6), DB(3,6,6)
    REAL(KIND=DP), DIMENSION(6,6) :: TERM1, TERM2, CII, ICI, CICI, LCICI, CCI, CIC
    ! ----------------------------------------------------------- HYPEREL --- !

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

    ! INVARIANTS OF C
    CALL INVARS(C, I1, I2, I3)

    ! INVARIANTS OF CBAR
    SCALE = JAC ** (-ONE / THREE)
    I1B = I1 * (SCALE ** 2)
    I2B = I2 * (SCALE ** 4)

    CALL UHYPER(I1B, I2B, JAC, U, DU, D2U, D3U, TEMP, NOEL, CMNAME, &
         INCMPFLAG, NSTATV, STATEV, NFLDV, FIELDV, DFIELDV, &
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

    ! CAUCHY STRESS
    Q = TRANSQ(F1)
    STRESS = MATMUL(Q, PK2) / JAC

    IF (.NOT. PRESENT(DDSDDE)) RETURN

    CICI = DYAD(CINV, CINV)
    CIC = DYAD(CINV, C)
    CCI = DYAD(C, CINV)
    LCICI = OPROD(CINV, CINV)
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
    DDSDDE = MATMUL(MATMUL(Q, DDTDDC), TRANSPOSE(Q)) / JAC

  END SUBROUTINE HYPEREL

  ! ************************************************************************* !

  SUBROUTINE JACOBIAN(NPROPS, PROPS, TEMP, NOEL, CMNAME, INCMPFLAG, &
       NSTATV, STATEV, NFLDV, FIELDV, DFIELDV, F, PK2, DDSDDE)
    ! THIS PROCEDURE NUMERICALLY COMPUTES AND RETURNS THE JACOBIAN MATRIX
    !
    !                      J = (JIJ) = (DSIGI/DEPSJ).
    !
    ! THE COMPONENTS OF JSUB ARE COMPUTED NUMERICALLY USING A CENTERED
    ! DIFFERENCING SCHEME WHICH REQUIRES TWO CALLS TO THE MATERIAL MODEL
    ! SUBROUTINE FOR EACH ELEMENT OF PK2. THE CENTERING IS ABOUT THE POINT
    !                       EPS = EPSOLD + D * DT,
    ! WHERE D IS THE RATE-OF-STRAIN ARRAY.
    REAL(KIND=DP), PARAMETER :: TOL=1.E-6_DP
    CHARACTER*8, INTENT(IN) :: CMNAME
    INTEGER, INTENT(IN) :: NPROPS, NOEL, NSTATV, INCMPFLAG, NFLDV
    REAL(KIND=DP), INTENT(IN) :: PROPS(NPROPS), TEMP, STATEV(NSTATV)
    REAL(KIND=DP), INTENT(INOUT) :: FIELDV(NFLDV), DFIELDV(NFLDV)
    REAL(KIND=DP), INTENT(IN) :: PK2(6), F(3,3)
    REAL(KIND=DP), INTENT(OUT) :: DDSDDE(6,6)
    INTEGER :: N
    REAL(KIND=DP) :: EPS, DF(3,3)
    REAL(KIND=DP) :: SVP(NSTATV), FP(3,3), SP(6)
    REAL(KIND=DP) :: SVM(NSTATV), FM(3,3), SM(6)
    ! ---------------------------------------------------------- JACOBIAN --- !
    EPS = SQRT(EPSILON(ONE))
    DO N = 1, 6
       DF = DGEDDG(EPS/TWO, N, F)
       FP = F + DF
       SVP = STATEV
       CALL HYPEREL(NPROPS, PROPS, TEMP, FP, NOEL, CMNAME, INCMPFLAG, &
            NSTATV, SVP, NFLDV, FIELDV, DFIELDV, SP)
       FM = F - DF
       SVM = STATEV
       CALL HYPEREL(NPROPS, PROPS, TEMP, FM, NOEL, CMNAME, INCMPFLAG, &
            NSTATV, SVM, NFLDV, FIELDV, DFIELDV, SM)
       DDSDDE(:, N) = (SP - SM) / EPS
    END DO
    WHERE (ABS(DDSDDE) / MAXVAL(DDSDDE) < TOL)
       DDSDDE = ZERO
    END WHERE
    DDSDDE = HALF * (DDSDDE + TRANSPOSE(DDSDDE))
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
  FUNCTION OPROD(A, B)
    ! ----------------------------------------------------------------------- !
    ! "O" PRODUCT OF A AND B
    ! ----------------------------------------------------------------------- !
    REAL(KIND=DP), INTENT(IN) :: A(6), B(6)
    REAL(KIND=DP) :: OPROD(6,6)
    REAL(KIND=DP) :: FAC
    FAC = HALF
    OPROD(1,1) = A(1) * B(1)
    OPROD(1,2) = A(4) * B(4)
    OPROD(1,3) = A(5) * B(5)
    OPROD(1,4) = FAC * A(1) * B(4)  +  FAC * A(4) * B(1)
    OPROD(1,5) = FAC * A(1) * B(5)  +  FAC * A(5) * B(1)
    OPROD(1,6) = FAC * A(4) * B(5)  +  FAC * A(5) * B(4)
    OPROD(2,1) = A(4) * B(4)
    OPROD(2,2) = A(2) * B(2)
    OPROD(2,3) = A(6) * B(6)
    OPROD(2,4) = FAC * A(2) * B(4)  +  FAC * A(4) * B(2)
    OPROD(2,5) = FAC * A(4) * B(6)  +  FAC * A(6) * B(4)
    OPROD(2,6) = FAC * A(2) * B(6)  +  FAC * A(6) * B(2)
    OPROD(3,1) = A(5) * B(5)
    OPROD(3,2) = A(6) * B(6)
    OPROD(3,3) = A(3) * B(3)
    OPROD(3,4) = FAC * A(5) * B(6)  +  FAC * A(6) * B(5)
    OPROD(3,5) = FAC * A(3) * B(5)  +  FAC * A(5) * B(3)
    OPROD(3,6) = FAC * A(3) * B(6)  +  FAC * A(6) * B(3)
    OPROD(4,1) = A(4) * B(1)
    OPROD(4,2) = A(2) * B(4)
    OPROD(4,3) = A(6) * B(5)
    OPROD(4,4) = FAC * A(2) * B(1)  +  FAC * A(4) * B(4)
    OPROD(4,5) = FAC * A(4) * B(5)  +  FAC * A(6) * B(1)
    OPROD(4,6) = FAC * A(2) * B(5)  +  FAC * A(6) * B(4)
    OPROD(5,1) = A(5) * B(1)
    OPROD(5,2) = A(6) * B(4)
    OPROD(5,3) = A(3) * B(5)
    OPROD(5,4) = FAC * A(5) * B(4)  +  FAC * A(6) * B(1)
    OPROD(5,5) = FAC * A(3) * B(1)  +  FAC * A(5) * B(5)
    OPROD(5,6) = FAC * A(3) * B(4)  +  FAC * A(6) * B(5)
    OPROD(6,1) = A(5) * B(4)
    OPROD(6,2) = A(6) * B(2)
    OPROD(6,3) = A(3) * B(6)
    OPROD(6,4) = FAC * A(5) * B(2)  +  FAC * A(6) * B(4)
    OPROD(6,5) = FAC * A(3) * B(4)  +  FAC * A(5) * B(6)
    OPROD(6,6) = FAC * A(3) * B(2)  +  FAC * A(6) * B(6)
    RETURN
  END FUNCTION OPROD
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
  FUNCTION DGEDDG(EPS, N, F)
    ! ----------------------------------------------------------------------- !
    ! PERTURB THE DEFORMATION GRADIENT
    ! ----------------------------------------------------------------------- !
    INTEGER, INTENT(IN) :: N
    REAL(KIND=DP), INTENT(IN) :: EPS, F(3,3)
    REAL(KIND=DP) :: DGEDDG(3,3)
    REAL(KIND=DP), PARAMETER :: O=1._DP,Z=0._DP
    REAL(KIND=DP), PARAMETER :: E(3,3)=RESHAPE((/O,Z,Z,Z,O,Z,Z,Z,O/),(/3,3/))
    INTEGER, PARAMETER :: IJ(2,6)=RESHAPE((/1,1,2,2,3,3,1,2,1,3,2,3/),(/2,6/))
    REAL(KIND=DP) :: EI(3), EJ(3), EIJ(3,3), EJI(3,3)
    ! ----------------------------------------------------------------------- !
    EI = E(IJ(1,N),:3)
    EJ = E(IJ(2,N),:3)
    EIJ = DYAD(EI, EJ)
    EJI = DYAD(EJ, EI)
    DGEDDG = EPS / TWO * (MATMUL(EIJ, F) + MATMUL(EJI, F))
  END FUNCTION DGEDDG

  SUBROUTINE NEOHOOKE(NPROPS, PROPS, F, SIG, C)
    ! --------------------------------------------------------------------- !
    ! COMPRESSIBLE NEO-HOOKEAN HYPERELASTIC MATERIAL
    !
    ! NOTES
    ! -----
    ! SYMMETRIC TENSOR ORDERING : XX, YY, ZZ, XY, YZ, ZX
    ! --------------------------------------------------------------------- !
    INTEGER, INTENT(IN) :: NPROPS
    REAL(DP), INTENT(IN) :: PROPS(NPROPS), F(3,3)
    REAL(DP), INTENT(INOUT) :: SIG(6), C(6,6)
    INTEGER :: I, J
    REAL(DP) :: EE(6), EEP(3), BBP(3), BBN(3,3)
    REAL(DP) :: C10, D1, EG, EK, EG23, PR
    REAL(DP) :: JAC, SCALE, FB(3,3), BB(6), TRBBAR

    ! ELASTIC PROPERTIES
    C10 = PROPS(1)
    D1 = PROPS(2)

    ! JACOBIAN AND DISTORTION TENSOR
    JAC = F(1,1) * F(2,2) * F(3,3) - F(1,2) * F(2,1) * F(3,3) &
        + F(1,2) * F(2,3) * F(3,1) + F(1,3) * F(3,2) * F(2,1) &
        - F(1,3) * F(3,1) * F(2,2) - F(2,3) * F(3,2) * F(1,1)
    SCALE = JAC **(-ONE / THREE)
    FB = SCALE * F

    ! DEVIATORIC LEFT CAUCHY-GREEN DEFORMATION TENSOR
    BB(1) = FB(1,1) * FB(1,1) + FB(1,2) * FB(1,2) + FB(1,3) * FB(1,3)
    BB(2) = FB(2,1) * FB(2,1) + FB(2,2) * FB(2,2) + FB(2,3) * FB(2,3)
    BB(3) = FB(3,1) * FB(3,1) + FB(3,2) * FB(3,2) + FB(3,3) * FB(3,3)
    BB(4) = FB(2,1) * FB(1,1) + FB(2,2) * FB(1,2) + FB(2,3) * FB(1,3)
    BB(5) = FB(3,1) * FB(2,1) + FB(3,2) * FB(2,2) + FB(3,3) * FB(2,3)
    BB(6) = FB(3,1) * FB(1,1) + FB(3,2) * FB(1,2) + FB(3,3) * FB(1,3)
    TRBBAR = SUM(BB(1:3)) / THREE
    EG = TWO * C10 / JAC
    EK = TWO / D1 * (TWO * JAC - ONE)
    PR = TWO / D1 * (JAC - ONE)

    ! CAUCHY STRESS
    SIG = EG * (BB - TRBBAR * IDENTITY) + PR * IDENTITY

    ! SPATIAL STIFFNESS
    EG23 = EG * TWO / THREE
    C(1,1) =  EG23 * (BB(1) + TRBBAR) + EK
    C(1,2) = -EG23 * (BB(1) + BB(2)-TRBBAR) + EK
    C(1,3) = -EG23 * (BB(1) + BB(3)-TRBBAR) + EK
    C(1,4) =  EG23 * BB(4) / TWO
    C(1,5) = -EG23 * BB(5)
    C(1,6) =  EG23 * BB(6) / TWO
    C(2,2) =  EG23 * (BB(2) + TRBBAR) + EK
    C(2,3) = -EG23 * (BB(2) + BB(3)-TRBBAR) + EK
    C(2,4) =  EG23 * BB(4) / TWO
    C(2,5) =  EG23 * BB(5) / TWO
    C(2,6) = -EG23 * BB(6)
    C(3,3) =  EG23 * (BB(3) + TRBBAR) + EK
    C(3,4) = -EG23 * BB(4)
    C(3,5) =  EG23 * BB(5) / TWO
    C(3,6) =  EG23 * BB(6) / TWO
    C(4,4) =  EG * (BB(1) + BB(2)) / TWO
    C(4,5) =  EG * BB(6) / TWO
    C(4,6) =  EG * BB(5) / TWO
    C(5,5) =  EG * (BB(2) + BB(3)) / TWO
    C(5,6) =  EG * BB(4) / TWO
    C(6,6) =  EG * (BB(1) + BB(3)) / TWO
    FORALL(I=1:6,J=1:6,J<I) C(I,J) = C(J,I)
    RETURN
  END SUBROUTINE NEOHOOKE
END MODULE HYPERELASTIC

! *************************************************************************** !

SUBROUTINE UMAT(STRESS,STATEV,DDSDDE,SSE,SPD,SCD, &
     RPL,DDSDDT,DRPLDE,DRPLDT, &
     STRAN,DSTRAN,TIME,DTIME,TEMP,DTEMP,PREDEF,DPRED,CMNAME, &
     NDI,NSHR,NTENS,NSTATV,PROPS,NPROPS,COORDS,DROT,PNEWDT, &
     CELENT,F0,F1,NOEL,NPT,LAYER,KSPT,KSTEP,KINC)
  ! ------------------------------------------------------------------------- !
  !     UMAT INTERFACE FOR FINITE STRAIN PROPELLANT MODEL
  ! ------------------------------------------------------------------------- !
  !     CANNOT BE USED FOR PLANE STRESS
  !     PROPS(1)  - C10
  !     PROPS(2)  - K1
  ! ------------------------------------------------------------------------- !
  USE HYPERELASTIC, ONLY: DP, ONE, TWO, THREE, HYPEREL, JACOBIAN, TRANSQ, &
       NEOHOOKE
  IMPLICIT NONE

  ! --- PASSED ARGUMENTS
  ! ------------------------------------------------------------------------- !
  CHARACTER*8, INTENT(IN) :: CMNAME
  INTEGER, INTENT(IN) :: NDI, NSHR, NTENS, NSTATV, NPROPS
  INTEGER, INTENT(IN) :: NOEL, NPT, LAYER, KSPT, KSTEP, KINC
  REAL(KIND=DP), INTENT(IN) :: SSE, SPD, SCD, RPL, DRPLDT, TIME, DTIME
  REAL(KIND=DP), INTENT(IN) :: TEMP, DTEMP, PNEWDT, CELENT
  REAL(KIND=DP), INTENT(INOUT) :: STRESS(NTENS), STATEV(NSTATV)
  REAL(KIND=DP), INTENT(INOUT) :: DDSDDE(NTENS, NTENS)
  REAL(KIND=DP), INTENT(INOUT) :: DDSDDT(NTENS), DRPLDE(NTENS)
  REAL(KIND=DP), INTENT(IN) :: STRAN(NTENS), DSTRAN(NTENS)
  REAL(KIND=DP), INTENT(IN) :: PREDEF(1), DPRED(1), PROPS(NPROPS), COORDS(3)
  REAL(KIND=DP), INTENT(IN) :: DROT(3, 3), F0(3, 3), F1(3, 3)
  ! --- LOCAL VARIABLES
  INTEGER, PARAMETER :: NFLDV=1, INCMPFLAG=1
  REAL(KIND=DP) :: FIELDV(NFLDV), DFIELDV(NFLDV)
  real(kind=dp) :: foos(6), fooc1(6,6),fooc2(6,6)
  ! ---------------------------------------------------------------- UMAT --- !


  CALL HYPEREL(NPROPS, PROPS, TEMP, F1, NOEL, CMNAME, INCMPFLAG, &
       NSTATV, STATEV, NFLDV, FIELDV, DFIELDV, STRESS, DDSDDE)
  CALL JACOBIAN(NPROPS, PROPS, TEMP, NOEL, CMNAME, INCMPFLAG, &
       NSTATV, STATEV, NFLDV, FIELDV, DFIELDV, F1, STRESS, fooc1)

  CALL NEOHOOKE(2, (/PROPS(1),PROPS(2)/), F1, foos, fooc2)

  IF (DTIME < EPSILON(ONE)) RETURN

  print*
  print*, 'hyper stiff'
  print*, ddsdde(1,1:3)
  print*, ddsdde(2,1:3)
  print*, ddsdde(3,1:3)
  print*, '        ', ddsdde(4,4:6)
  print*, '        ', ddsdde(5,4:6)
  print*, '        ', ddsdde(6,4:6)
  print*, 'numerical stiff'
  print*, fooc1(1,1:3)
  print*, fooc1(2,1:3)
  print*, fooc1(3,1:3)
  print*, '        ', fooc1(4,4:6)
  print*, '        ', fooc1(5,4:6)
  print*, '        ', fooc1(6,4:6)
  print*, 'neohooke stiff'
  print*, fooc2(1,1:3)
  print*, fooc2(2,1:3)
  print*, fooc2(3,1:3)
  print*, '        ', fooc2(4,4:6)
  print*, '        ', fooc2(5,4:6)
  print*, '        ', fooc2(6,4:6)
  print*
  print*

END SUBROUTINE UMAT
