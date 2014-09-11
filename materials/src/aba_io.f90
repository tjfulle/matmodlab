! Include the following in each signature file
! python module gmd__user__routines
!     interface gmd_user_interface
!         subroutine log_message(message)
!             intent(callback) log_message
!             character*(*) :: message
!         end subroutine log_message
!         subroutine log_error(message)
!             intent(callback) log_error
!             character*(*) :: message
!             real intent(callback) :: log_message
!         end subroutine log_error
!     end interface gmd_user_interface
! end python module gmd__user__routines
!
! Then, in each fortran function
! use gmd__user__routines
! intent(callback) log_error
! external log_error
! intent(callback) log_message
! external log_message
SUBROUTINE STDB_ABQERR(IERR, MSG, INTV, REALV, CHARV)
  IMPLICIT NONE
  INTEGER, INTENT(IN) :: IERR
  CHARACTER(120), INTENT(IN) :: MSG
  INTEGER, INTENT(IN) :: INTV(*)
  REAL(8), INTENT(IN) :: REALV(*)
  CHARACTER(8), INTENT(IN) :: CHARV(*)
  CHARACTER(1) :: C(2)
  CHARACTER(12) :: S
  CHARACTER(220) :: STRING
  INTEGER :: I, J, K, N, LS, II, IR, IC
  EXTERNAL LOG_MESSAGE
  EXTERNAL LOG_WARNING
  EXTERNAL LOG_ERROR

  II=1; IR=1; IC=1; J=1; K=1
  FORALL(I=1:LEN(STRING)) STRING(I:I) = " "
  DO I=1,LEN_TRIM(MSG)
     C(1) = MSG(J:J); C(2) = MSG(J+1:J+1)
     N = 1
     IF (C(1) == "%") THEN
        N = 2
        IF (C(2) == "I" .OR. C(2) == "i") THEN
           ! REPLACE INTEGERS
           WRITE(S,"(I5)") INTV(II)
           II = II + 1
        ELSE IF (C(2) == "C" .OR. C(2) == "c") THEN
           ! REPLACE CHARACTERS
           S = CHARV(IC)
           IC = IC + 1
        ELSE IF (C(2) == "R" .OR. C(2) == "r") THEN
           ! REPLACE REALS
           WRITE(S,"(E12.6)") REALV(IR)
           IR = IR + 1
        ELSE
           S = C(2)
           N = 1
        END IF
        S = ADJUSTL(S)
        LS = LEN(TRIM(S))
        STRING(K:K+LS) = TRIM(S)
        K = K + LS
     ELSE
        STRING(K:K) = C(1)
        K = K + 1
     END IF
     J = J + N
  END DO

  IF (IERR == -3) THEN
     STRING = "ABA: BOMBED: " // ADJUSTL(STRING)
     CALL LOG_ERROR(STRING)
  ELSE IF (IERR == -1) THEN
     CALL LOG_WARNING(STRING)
  ELSE IF (IERR == -2) THEN
     STRING = "ABA: ERROR: " // ADJUSTL(STRING)
     CALL LOG_WARNING(TRIM(STRING))
  ELSE
     CALL LOG_MESSAGE(TRIM(STRING))
  END IF
  RETURN
END SUBROUTINE STDB_ABQERR
SUBROUTINE XIT
  IMPLICIT NONE
  CHARACTER*120 :: MSG
  EXTERNAL LOG_ERROR
  MSG = 'STOPPING DUE TO FORTRAN PROCEDURE ERROR'
  CALL LOG_ERROR(MSG)
END SUBROUTINE XIT

FUNCTION UPPER(S1)  RESULT (S2)
  CHARACTER(*) :: S1
  CHARACTER(LEN(S1)) :: S2
  CHARACTER :: CH
  INTEGER,PARAMETER :: DUC=ICHAR('A')-ICHAR('a')
  INTEGER :: I
  DO I=1,LEN(S1)
     CH = S1(I:I)
     IF (CH >= 'A'.AND.CH <= 'Z') CH = CHAR(ICHAR(CH)+DUC)
     S2(I:I) = CH
  END DO
END FUNCTION UPPER
