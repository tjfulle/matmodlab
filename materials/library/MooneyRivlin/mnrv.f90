! *************************************************************************** !
! Mooney-Rivlin material model
! Public procedures
! ------ ----------
! MNRVCP : Check properties
! MNRVXV : Request extra variables
! MNRVJM : Jacobian sub matrix
! MNRVUS : update state
! *************************************************************************** !

SUBROUTINE MNRVUS(NC, PROP, R, V, T, XTRA, SIG)
  ! ------------------------------------------------------------------------- !
  ! Update state
  ! ------------------------------------------------------------------------- !
  IMPLICIT NONE
  INCLUDE "mnrv.h"
  INCLUDE "extmod.h"
  INCLUDE "symdot.h"
  INTEGER, INTENT(IN) :: NC
  REAL(DP), INTENT(IN) :: PROP(NPROP), R(9,NC), V(6,NC), T(NC)
  REAL(DP), INTENT(INOUT) :: XTRA(NX,NC)
  REAL(DP), INTENT(OUT) :: SIG(6,NC)
  INTEGER :: I
  REAL(DP) :: J, CBRTJ, QBR
  REAL(DP) :: BB(6), BBS(6), I1B, I2B
  REAL(DP) :: C10, C01, NU, G, K, FAC, C1, C2, P
  ! ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ MNRVUS ~~~ !

  DO I = 1, NC

     IF (PROP(MC10) == SPECIAL) CALL EXTMOD(IC10, T(I), XTRA(KC10, I))
     IF (PROP(MC01) == SPECIAL) CALL EXTMOD(IC01, T(I), XTRA(KC01, I))
     C10 = XTRA(KC10, I)
     C01 = XTRA(KC01, I)
     NU = PROP(INU)
     G = 2._DP * (C10 + C01)
     K = 2._DP * G * (1._DP + NU) / 3._DP / (1 - 2._DP * NU)

     ! Mechanical volume change from Jm = det[Fm] = det[Vm] since det[R]=1
     J = SYMDET(V(1:6, I))

     ! Deviatoric stretch matrix [Bbar] (Left Cauchy-Green strain tensor)
     ! [Bbar] = [Fbar][Fbar]^T; [Fbar] = J^(-1/3) [F] = Jm^(-1/3)[Fm]
     ! [Bbar] = [Vbar][Vbar]^T; [Vbar] = J^(-1/3) [V]
     ! [Bbar] = J^(-2/3) [B], the deviatoric Left C-G strain tensor
     !          (J = Det(V))
     CBRTJ = SIGN((ABS(J)) ** (1._DP / 3._DP), J)
     QBR = (1._DP / (CBRTJ * CBRTJ))
     BB = QBR * SYMDOT(V(1:6,I), V(1:6,I))

     ! First strain invariant IB1 = trace(BB) = I:BB
     I1B = TRACE(BB)

     ! Second strain invariant I2B = 0.5*(IB1^2-trace(BB.BB))
     BBS = SYMDOT(BB, BB)
     I2B = 0.5_DP * (I1B * I1B - TRACE(BBS))

     ! Bulk response
     P = -K * (J - 1)

     ! Stress response
     !     2  / p                                     1                       \
     ! S = - | --I + (C10 + I1B C01) BB - C01 BB.BB - - (C10 I1B + 2C01 I2B) I |
     !     J  \ 2                                     3                       /
     FAC = 2.0_DP / J
     C1 = C10 + C01 * I1B
     C2 = (C10 * I1B + 2._DP * C01 * I2B) / 3._DP
     SIG(1:6, I) = FAC * (-P / 2._DP * I6 + C1 * BB - C01 * BBS - C2 * I6)

     ! Cauchy stresses in unrotated state
     CALL UNROTATE(R(1:9, I), SIG(1:6, I))

     ! Energy
     XTRA(KW, I) = V(1,I) !C10 * (I1B - 3._DP) + C01 * (I2B - 3._DP)

  END DO

CONTAINS

  FUNCTION TRACE(X)
    ! Trace of a second-order symmetric tensor
    REAL(DP) :: TRACE
    REAL(DP), INTENT(IN) :: X(6)
    TRACE = X(1) + X(2) + X(3)
    RETURN
  END FUNCTION TRACE

  SUBROUTINE UNROTATE(Q, X)
    ! Perform the unrotation operation XB = Q^T.X.Q
    REAL(DP), INTENT(IN) :: Q(3,3)
    REAL(DP), INTENT(INOUT) :: X(6)
    REAL(DP) :: Y(3,3)
    Y = RESHAPE((/X(1),X(4),X(6),X(4),X(2),X(5),X(6),X(5),X(3)/), SHAPE(Y))
    Y = MATMUL(MATMUL(TRANSPOSE(Q), Y), Q)
    Y = .5_DP * (Y + TRANSPOSE(Y))
    X = (/Y(1,1), Y(2,2), Y(3,3), Y(1,2), Y(2,3), Y(3,1)/)
    RETURN
  END SUBROUTINE UNROTATE

  FUNCTION SYMDET(X)
    ! Compute the determinant of a second-order symmetric tensor
    REAL(DP) :: SYMDET
    REAL(DP), INTENT(IN) :: X(6)
    SYMDET = X(1) * X(2) * X(3) + 2._DP * X(4) * X(5) * X(6) &
           - (X(1) * X(5) * X(5) + X(2) * X(6) * X(6) + X(3) * X(4) * X(4))
  END FUNCTION SYMDET

END SUBROUTINE MNRVUS

! *************************************************************************** !

SUBROUTINE MNRVCP(PROP)
  ! ------------------------------------------------------------------------- !
  ! Check properties
  ! ------------------------------------------------------------------------- !
  IMPLICIT NONE
  INCLUDE "mnrv.h"
  INCLUDE "extmod.h"
  REAL(DP), INTENT(INOUT) :: PROP(NPROP)
  REAL(DP) :: C10, C01, NU, T0
  ! ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ MNRVCP ~~~ !

  C10 = PROP(IC10)
  C01 = PROP(IC01)
  NU = PROP(INU)
  T0 = PROP(IT0)

  IF (ANY((/C10, C01/) == SPECIAL) .AND. T0 <= 0._DP) &
       CALL BOMBED("MNRVCP: T0 <= 0")

  IF (C10 == SPECIAL) THEN
     PROP(MC10) = SPECIAL
     CALL EXTMOD(1)
     CALL EXTMOD(1, T0, C10)
  END IF

  IF (C01 == SPECIAL) THEN
     PROP(MC01) = SPECIAL
     CALL EXTMOD(2)
     CALL EXTMOD(2, T0, C01)
  END IF

  IF (NU < -1._DP .OR. NU >= .5_DP) CALL BOMBED("MNRVCP: Bad NU")

  PROP(IC10) = C10
  PROP(IC01) = C01
  PROP(INU) = NU

END SUBROUTINE MNRVCP

! *************************************************************************** !

SUBROUTINE MNRVXV(PROP, NXTRA, KEYA, XTRA)
  ! ------------------------------------------------------------------------- !
  ! Initialize the Mooney-Rivlin material
  ! ------------------------------------------------------------------------- !
  IMPLICIT NONE
  INCLUDE "mnrv.h"
  REAL(DP), INTENT(IN) :: PROP(NPROP)
  INTEGER, INTENT(OUT) :: NXTRA
  CHARACTER(LEN=1), INTENT(OUT) :: KEYA(NX * 10)
  REAL(DP), INTENT(OUT) :: XTRA(NX)
  CHARACTER(LEN=9) :: KEYS(NX)
  ! ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ MNRVXV ~~~ !

  NXTRA = 1
  IF (NXTRA /= KC10) CALL BOMBED("KC10 pointer wrong")
  KEYS(NXTRA) = "C10"

  NXTRA = NXTRA + 1
  IF (NXTRA /= KC01) CALL BOMBED("KC01 pointer wrong")
  KEYS(NXTRA) = "C01"

  NXTRA = NXTRA + 1
  IF (NXTRA /= KW) CALL BOMBED("W pointer wrong")
  KEYS(NXTRA) = "W"

  IF (NX /= NXTRA) CALL BOMBED("NXTRA != NX")

  ! convert keys to character streams namea and keya
  CALL TOKENS(NXTRA, KEYS, KEYA)

  CALL MNRVINI(PROP, XTRA)

END SUBROUTINE MNRVXV

! *************************************************************************** !

SUBROUTINE MNRVINI(PROP, XTRA)
  ! ------------------------------------------------------------------------- !
  ! Initialize the Mooney-Rivlin material
  ! ------------------------------------------------------------------------- !
  IMPLICIT NONE
  INCLUDE "mnrv.h"
  INCLUDE "extmod.h"
  REAL(DP), INTENT(IN) :: PROP(NPROP)
  REAL(DP), INTENT(INOUT) :: XTRA(NX)
  ! ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ MNRVINI ~~~ !

  XTRA(KC10) = PROP(IC10)
  IF (PROP(MC10) == SPECIAL) CALL EXTMOD(1, PROP(IT0), XTRA(KC10))

  XTRA(KC01) = PROP(IC01)
  IF (PROP(MC01) == SPECIAL) CALL EXTMOD(2, PROP(IT0), XTRA(KC01))

  XTRA(KW) = 1._DP

END SUBROUTINE MNRVINI

! *************************************************************************** !

SUBROUTINE EXTMOD(IVAL, TVAL, CVAL)
  ! ------------------------------------------------------------------------- !
  ! Read and compute temperature dependent moduli from external file
  !
  ! Notes
  ! -----
  ! If TVAL and CVAL are not passed, the external file is read in and stored.
  ! This should always be done first before calling with TVAL and CVAL
  ! present.
  !
  ! If passed, TVAL is the temperature on input and CVAL the modulus at the
  ! temperature on output
  ! ------------------------------------------------------------------------- !
  IMPLICIT NONE
  INCLUDE "mnrv.h"
  INTEGER, INTENT(IN) :: IVAL
  REAL(DP), INTENT(IN), OPTIONAL :: TVAL
  REAL(DP), INTENT(OUT), OPTIONAL :: CVAL
  REAL(DP), ALLOCATABLE :: A(:,:)
  REAL(DP), ALLOCATABLE, SAVE :: C10(:,:), C01(:,:)
  INTEGER :: I, J, N
  CHARACTER(LEN=7) :: FILENAME
  LOGICAL :: EXISTS
  ! ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ EXTMOD ~~~ !

  IF (PRESENT(TVAL)) THEN
     ! Interpolate value of the modulus
     SELECT CASE (IVAL)
     CASE (1)
        N = SIZE(C10) / 2
        CVAL = INTERP1D(C10(1:N, 1), C10(1:N, 2), TVAL)
     CASE (2)
        N = SIZE(C01) / 2
        CVAL = INTERP1D(C01(1:N, 1), C01(1:N, 2), TVAL)
     END SELECT
     RETURN
  END IF

  ! To reach this point, the temperature dependent moduli must be read in from
  ! a file and saved to their respective arrays

  ! Read in moduli from external file
  SELECT CASE (IVAL)
  CASE (1)
     FILENAME = "C10.dat"
  CASE (2)
     FILENAME = "C01.dat"
  END SELECT

  ! Check that file exists
  INQUIRE(FILE=FILENAME, EXIST=EXISTS)
  IF (.NOT. EXISTS) CALL BOMBED("EXTMOD: " // FILENAME // ": no such file")

  ! Read in the data
  N = COUNTLINES(FILENAME)
  ALLOCATE(A(N, 2))
  OPEN(UNIT=15, FILE=FILENAME, STATUS="OLD")
  DO I = 1, N
     READ(15, *) (A(I, J), J=1,2)
     IF (I > 1 .AND. A(I, 1) < A(I-1, 1)) THEN
        CALL BOMBED("EXTMOD: " // FILENAME // &
                    ": temperature must be monotonically increasing")
     END IF
  END DO
  CLOSE(15)

  SELECT CASE (IVAL)
  CASE (1)
     ALLOCATE(C10(N, 2))
     C10(1:N, 1) = A(1:N, 1)
     C10(1:N, 2) = A(1:N, 2)
  CASE (2)
     ALLOCATE(C01(N, 2))
     C01(1:N, 1) = A(1:N, 1)
     C01(1:N, 2) = A(1:N, 2)
  END SELECT

  DEALLOCATE(A)

  CONTAINS

  INTEGER FUNCTION IPOS(X, XVAL)
    REAL(8), INTENT(IN) :: X(:), XVAL
    INTEGER :: IA(1)
    IA = MAXLOC(X, X <= XVAL)
    IPOS = IA(1)
    IF (IPOS >= SIZE(X)) IPOS = -1
    RETURN
  END FUNCTION IPOS

  REAL(DP) FUNCTION INTERP1D(X, Y, XVAL) RESULT(YVAL)
    REAL(DP), INTENT(IN) :: X(:), Y(:), XVAL
    INTEGER :: II
    REAL(DP) :: X1, X2, Y1, Y2, FRAC
    II = IPOS(X, XVAL)
    IF (II == 0) THEN
       YVAL = Y(1)
       RETURN
    ELSE IF (II == -1) THEN
       YVAL = Y(SIZE(Y))
       RETURN
    END IF
    X1 = X(II); X2 = X(II+1)
    Y1 = Y(II); Y2 = Y(II+1)
    FRAC = (XVAL - X1) / (X2 - X1)
    YVAL = Y1 + FRAC * (Y2 - Y1)
    RETURN
  END FUNCTION INTERP1D

  INTEGER FUNCTION COUNTLINES(FNAME) RESULT(NLINES)
    CHARACTER(LEN=7), INTENT(IN) :: FNAME
    OPEN(UNIT=15, FILE=FNAME, STATUS="OLD")
    NLINES = 0
    DO
       READ(15, *, END=10)
       NLINES = NLINES + 1
    END DO
10  CLOSE(15)
  END FUNCTION COUNTLINES

END SUBROUTINE EXTMOD

! *************************************************************************** !

SUBROUTINE MNRVJM(PROP, R, V, T, XTRA, NW, W, JSUB)
  ! ------------------------------------------------------------------------- !
  ! Compute the material Jacobian matrix dsig / deps
  !
  ! Notes
  ! -----
  ! This procedure numerically computes and returns a specified submatrix,
  ! Jsub, of the Jacobian matrix J = (Jij) = (dsigi/depsj). The submatrix
  ! returned is the one formed by the intersections of the rows and columns
  ! specified in the vector subscript array, v. That is, Jsub = J(v,v). The
  ! physical array con- taining this submatrix is assumed to be dimensioned
  ! Jsub(nv,nv), where nv is the number of elements in v. Note that in the
  ! special case v = /1,2,3,4,5,6/, with nv = 6, the matrix that is returned
  ! is the full Jacobian matrix, J.
  !
  ! The components of Jsub are computed numerically using a centered
  ! differencing scheme which requires two calls to the material model
  ! subroutine for each element of v. The centering is about the point eps =
  ! epsold + d * dt, where d is the rate-of-strain array.
  ! ------------------------------------------------------------------------- !
  IMPLICIT NONE
  INCLUDE "mnrv.h"
  INCLUDE "symdot.h"
  INTEGER, INTENT(IN) :: NW, W(NW)
  REAL(DP), INTENT(IN) :: PROP(NPROP), R(9), V(6), T(1), XTRA(NX)
  REAL(DP), INTENT(OUT) :: JSUB(NW,NW)
  INTEGER :: I, J, N1, N2
  REAL(DP) :: B(6), EPS(6), DEPS(6), D
  REAL(DP) :: VP(6), XP(NX), SP(6)
  REAL(DP) :: VM(6), XM(NX), SM(6)
  ! ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ MNRVJM ~~~ !
  D = SQRT(EPSILON(V))
  B = SYMDOT(V(1:6), V(1:6))
  EPS = ALMANSI(B)
  DEPS = 0._DP
  JSUB = 0._DP
  OUTER: DO N1 = 1, NW
     I = W(N1)
     INNER: DO N2 = 1, NW
        J = W(N2)
        IF (J < I) THEN
           ! Symmetric
           JSUB(J, I) = JSUB(I, J)
           CYCLE INNER
        END IF
        DEPS(J) = D / 2._DP
        VP = LEFTV(EPS + DEPS)
        XP = XTRA
        CALL MNRVUS(1, PROP, I9, VP, T, XP, SP)
        DEPS(J) = -D / 2._DP
        VM = LEFTV(EPS + DEPS)
        XM = XTRA
        CALL MNRVUS(1, PROP, I9, VM, T, XM, SM)
        JSUB(I, J) = (SP(I) - SM(I)) / D
        JSUB(J, I) = JSUB(I, J)
        DEPS(J) = 0._DP
     END DO INNER
  END DO OUTER

  WHERE (ABS(JSUB) / MAXVAL(JSUB) < 1.E-06_DP)
     JSUB = 0._DP
  END WHERE

CONTAINS

  FUNCTION LEFTV(A)
    ! V.V = INV(I - 2A)
    REAL(DP) :: LEFTV(6)
    REAL(DP), INTENT(IN) :: A(6)
    REAL(DP) :: VSQ(6)
    VSQ = SYMINV(I6 - 2._DP * A)
    LEFTV = SQRTT(VSQ)
  END FUNCTION LEFTV

  FUNCTION ALMANSI(B)
    ! 2A = I - INV(B) = I - INV(V.V)
    REAL(DP) :: ALMANSI(6)
    REAL(DP), INTENT(IN) :: B(6)
    REAL(DP) :: BI(6)
    BI = SYMINV(B)
    ALMANSI = .5_DP * (I6 - BI)
  END FUNCTION ALMANSI

  FUNCTION SYMINV(X)
    ! Inverse of symmetric second-order tensor
    IMPLICIT NONE
    INCLUDE "mnrv.h"
    REAL(DP) :: SYMINV(6)
    REAL(DP), INTENT(IN) :: X(6)
    REAL(DP) :: DNOM
    DNOM = X(3) * X(4) ** 2 + X(1) * (-(X(2) * X(3)) + X(5) ** 2) + &
           X(6) * (-2_DP * X(4) * X(5) + X(2) * X(6))
    SYMINV(1) = (X(5) * X(5) - X(2) * X(3)) / DNOM
    SYMINV(2) = (X(6) * X(6) - X(1) * X(3)) / DNOM
    SYMINV(3) = (X(4) * X(4) - X(1) * X(2)) / DNOM
    SYMINV(4) = (X(3) * X(4) - X(5) * X(6)) / DNOM
    SYMINV(5) = (X(1) * X(5) - X(4) * X(6)) / DNOM
    SYMINV(6) = (X(2) * X(6) - X(4) * X(5)) / DNOM
  END FUNCTION SYMINV

  FUNCTION SQRTT(X)
    ! Square root of symmetric second order tensor
    IMPLICIT NONE
    INCLUDE "mnrv.h"
    REAL(DP) :: SQRTT(6)
    REAL(DP), INTENT(IN) :: X(6)
    INTEGER, PARAMETER :: N=3, LWORK=3*N-1
    REAL(DP) :: W(N), WORK(LWORK), V(3,3), L(3,3), A(3,3)
    INTEGER :: INFO, I
    SQRTT = 0._DP
    IF (ALL(ABS(X(4:6)) < EPSILON(X))) THEN
       ! Diagonal
       SQRTT(1:3) = SQRT(X(1:3))
       RETURN
    END IF
    ! eigenvalues/vectors of a
    V(1,1) = X(1); V(1,2) = X(4); V(1,3) = X(6)
    V(2,1) = X(4); V(2,2) = X(2); V(2,3) = X(5)
    V(3,1) = X(6); V(3,2) = X(5); V(3,3) = X(3)
    CALL DSYEV("V", "L", 3, V, 3, W, WORK, LWORK, INFO)
    L = 0._DP
    FORALL(I=1:3) L(I,I) = SQRT(W(I))
    A = MATMUL(MATMUL(V, L ), TRANSPOSE(V))
    SQRTT = (/A(1,1), A(2,2), A(3,3), A(1,2), A(2,3), A(1,3)/)
  END FUNCTION SQRTT

END SUBROUTINE MNRVJM

! *************************************************************************** !

FUNCTION SYMDOT(X, Y)
  ! Dot product of two second-order symmetric tensors
  IMPLICIT NONE
  INCLUDE "mnrv.h"
  REAL(DP) :: SYMDOT(6)
  REAL(DP), INTENT(IN) :: X(6), Y(6)
  SYMDOT(1) = X(1) * Y(1) + X(4) * Y(4) + X(6) * Y(6)
  SYMDOT(2) = X(2) * Y(2) + X(4) * Y(4) + X(5) * Y(5)
  SYMDOT(3) = X(3) * Y(3) + X(5) * Y(5) + X(6) * Y(6)
  SYMDOT(4) = X(1) * Y(4) + X(2) * Y(4) + X(5) * Y(6)
  SYMDOT(5) = X(2) * Y(5) + X(3) * Y(5) + X(4) * Y(6)
  SYMDOT(6) = X(1) * Y(6) + X(3) * Y(6) + X(4) * Y(5)
END FUNCTION SYMDOT
