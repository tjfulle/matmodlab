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
subroutine stdb_abqerr(i, msg, intv, realv, charv)
  implicit none
  integer :: i
  character*120 :: msg
  integer :: intv(1), idum
  real(8) :: realv(1), rdum
  character*8 :: charv(1), cdum(1)
  character*200 jnkstr
  external log_message
  external log_error
  rdum=realv(1)
  idum=intv(1)
  cdum=charv(1)
  if (i == -3) then
     jnkstr = "abq: bombed: " // msg
     call log_error(msg)
  else
     call log_message(msg)
  end if
  return
end subroutine stdb_abqerr
subroutine xit
  implicit none
  stop
end subroutine xit
