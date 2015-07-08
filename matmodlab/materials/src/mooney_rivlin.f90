! *************************************************************************** !
! Mooney-Rivlin material model
! *************************************************************************** !

MODULE MOONEY_RIVLIN

  IMPLICIT NONE
  INTEGER, PARAMETER :: DP=SELECTED_REAL_KIND(14)
  REAL(DP), PARAMETER :: &
       I6(6)=[1._DP, 1._DP, 1._DP, 0._DP, 0._DP, 0._DP], &
       I9(9)=[1._DP, 0._DP, 0._DP, 0._DP, 1._DP, 0._DP, 0._DP, 0._DP, 1._DP]


! Local Variables:
! mode: f90
! End:

CONTAINS

  SUBROUTINE UPDATE_STATE(NPROP, PROP, R, V, SIG)
    ! ------------------------------------------------------------------------- !
    ! Update state
    ! ------------------------------------------------------------------------- !
    INTEGER, INTENT(IN) :: NPROP
    REAL(DP), INTENT(IN) :: PROP(NPROP), R(9), V(6)
    REAL(DP), INTENT(OUT) :: SIG(6)
    REAL(DP) :: J, CBRTJ, QBR
    REAL(DP) :: BB(6), BDB(6), I1B, I2B
    REAL(DP) :: C10, C01, NU, G, K, FAC, C1, C2, P
    ! ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ UPDATE_STATE ~~~ !

    C10 = PROP(1)
    C01 = PROP(2)
    NU = PROP(3)
    G = 2._DP * (C10 + C01)
    K = 2._DP * G * (1._DP + NU) / 3._DP / (1 - 2._DP * NU)

    ! Mechanical volume change from Jm = det[Fm] = det[Vm] since det[R]=1
    J = SYMDET(V(1:6))

    ! Deviatoric stretch matrix [Bbar] (Left Cauchy-Green strain tensor)
    ! [Bbar] = [Fbar][Fbar]^T; [Fbar] = J^(-1/3) [F] = Jm^(-1/3)[Fm]
    ! [Bbar] = [Vbar][Vbar]^T; [Vbar] = J^(-1/3) [V]
    ! [Bbar] = J^(-2/3) [B], the Left C-G strain tensor
    !          (J = Det(V))
    CBRTJ = SIGN((ABS(J)) ** (1._DP / 3._DP), J)
    QBR = (1._DP / (CBRTJ * CBRTJ))
    BB = QBR * SYMDOT(V(1:6), V(1:6))

    ! First invariant of stretch IB1 = trace(BB) = I:BB
    I1B = TRACE(BB)

    ! Second invariant of stretch I2B = 0.5 (IB1^2 - trace(BB.BB))
    BDB = SYMDOT(BB, BB)
    I2B = 0.5_DP * (I1B * I1B - TRACE(BDB))

    ! Bulk response
    P = -K * (J - 1)

    ! Stress response
    !     2  /  p                                     1                       \
    ! S = - | - -I + (C10 + I1B C01) BB - C01 BB.BB - - (C10 I1B + 2C01 I2B) I |
    !     J  \  2                                     3                       /
    FAC = 2.0_DP / J
    C1 = C10 + C01 * I1B
    C2 = (C10 * I1B + 2._DP * C01 * I2B) / 3._DP
    SIG(1:6) = FAC * (-P / 2._DP * I6 + C1 * BB - C01 * BDB - C2 * I6)

    ! Cauchy stresses in unrotated state
    CALL UNROTATE(R(1:9), SIG(1:6))

  END SUBROUTINE UPDATE_STATE

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
    Y = RESHAPE([X(1),X(4),X(6),X(4),X(2),X(5),X(6),X(5),X(3)], SHAPE(Y))
    Y = MATMUL(MATMUL(TRANSPOSE(Q), Y), Q)
    Y = .5_DP * (Y + TRANSPOSE(Y))
    X = [Y(1,1), Y(2,2), Y(3,3), Y(1,2), Y(2,3), Y(3,1)]
    RETURN
  END SUBROUTINE UNROTATE

  FUNCTION SYMDET(X)
    ! Compute the determinant of a second-order symmetric tensor
    REAL(DP) :: SYMDET
    REAL(DP), INTENT(IN) :: X(6)
    SYMDET = X(1) * X(2) * X(3) + 2._DP * X(4) * X(5) * X(6) &
           - (X(1) * X(5) * X(5) + X(2) * X(6) * X(6) + X(3) * X(4) * X(4))
  END FUNCTION SYMDET

  ! *************************************************************************** !

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


  ! *************************************************************************** !

  SUBROUTINE JACOBIAN(NPROP, PROP, V, JSUB)
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
    INTEGER, INTENT(IN) :: NPROP
    REAL(DP), INTENT(IN) :: PROP(NPROP), V(6)
    REAL(DP), INTENT(OUT) :: JSUB(6,6)
    INTEGER :: I, J
    REAL(DP) :: B(6), EPS(6), DEPS(6), D
    REAL(DP) :: VP(6), SP(6)
    REAL(DP) :: VM(6), SM(6)
    INTEGER, PARAMETER :: W(6)=(/1,2,3,4,5,6/), NW=6
    ! ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ MNRVJM ~~~ !
    D = SQRT(EPSILON(V))
    B = SYMDOT(V(1:6), V(1:6))
    EPS = ALMANSI(B)
    DEPS = 0._DP
    JSUB = 0._DP

    OUTER: DO I = 1, NW
       INNER: DO J = 1, NW
          IF (J < I) THEN
             ! Symmetric
             JSUB(I, J) = JSUB(J, I)
             CYCLE INNER
          END IF
          DEPS(W(J)) = D / 2._DP
          VP = LEFTV(EPS + DEPS)
          CALL UPDATE_STATE(NPROP, PROP, I9, VP, SP)
          DEPS(W(J)) = -D / 2._DP
          VM = LEFTV(EPS + DEPS)
          CALL UPDATE_STATE(NPROP, PROP, I9, VM, SM)
          JSUB(I, J) = (SP(W(I)) - SM(W(I))) / D
          DEPS(W(J)) = 0._DP
       END DO INNER
    END DO OUTER

    WHERE (ABS(JSUB) / MAXVAL(JSUB) < 1.E-06_DP)
       JSUB = 0._DP
    END WHERE

  END SUBROUTINE JACOBIAN

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
    SQRTT = [A(1,1), A(2,2), A(3,3), A(1,2), A(2,3), A(1,3)]
  END FUNCTION SQRTT


  ! *************************************************************************** !

  FUNCTION SYMDOT(X, Y)
    ! Dot product of two second-order symmetric tensors
    REAL(DP) :: SYMDOT(6)
    REAL(DP), INTENT(IN) :: X(6), Y(6)
    SYMDOT(1) = X(1) * Y(1) + X(4) * Y(4) + X(6) * Y(6)
    SYMDOT(2) = X(2) * Y(2) + X(4) * Y(4) + X(5) * Y(5)
    SYMDOT(3) = X(3) * Y(3) + X(5) * Y(5) + X(6) * Y(6)
    SYMDOT(4) = X(1) * Y(4) + X(2) * Y(4) + X(5) * Y(6)
    SYMDOT(5) = X(2) * Y(5) + X(3) * Y(5) + X(4) * Y(6)
    SYMDOT(6) = X(1) * Y(6) + X(3) * Y(6) + X(4) * Y(5)
  END FUNCTION SYMDOT

END MODULE MOONEY_RIVLIN


SUBROUTINE MNRV_MAT(NPROP, PROP, R, V, SIG, C)
  USE MOONEY_RIVLIN, ONLY : UPDATE_STATE, JACOBIAN
  IMPLICIT NONE
  INTEGER, INTENT(IN) :: NPROP
  REAL(8), INTENT(IN) :: PROP(NPROP), R(9), V(6)
  REAL(8), INTENT(OUT) :: SIG(6), C(6,6)
  CALL UPDATE_STATE(NPROP, PROP, R, V, SIG)
  CALL JACOBIAN(NPROP, PROP, V, C)
END SUBROUTINE MNRV_MAT
