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
SUBROUTINE STDB_ABQERR(I, MSG, INTV, REALV, CHARV)
  IMPLICIT NONE
  INTEGER :: I
  CHARACTER*120 :: MSG
  INTEGER :: INTV(1), IDUM
  REAL(8) :: REALV(1), RDUM
  CHARACTER*8 :: CHARV(1), CDUM(1)
  CHARACTER*200 JNKSTR
  EXTERNAL LOG_MESSAGE
  EXTERNAL LOG_ERROR
  EXTERNAL LOG_WARNING
  RDUM=REALV(1)
  IDUM=INTV(1)
  CDUM=CHARV(1)
  IF (I == -3) THEN
     JNKSTR = "ABA: BOMBED: " // MSG
     CALL LOG_ERROR(JNKSTR)
  ELSE IF (I == -1) THEN
     CALL LOG_WARNING(MSG)
  ELSE IF (I == -2) THEN
     JNKSTR = "ABA: ERROR: " // MSG
     CALL LOG_WARNING(MSG)
  ELSE
     CALL LOG_MESSAGE(MSG)
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
