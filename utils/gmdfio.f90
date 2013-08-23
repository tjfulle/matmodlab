! Include the following in each signature file
! python module gmd__user__routines
!     interface gmd_user_interface
!         subroutine log_message(message)
!             intent(callback) log_message
!             character*(*) :: message
!             real intent(callback) :: report_and_raise_error
!         end subroutine log_message
!         subroutine log_error(message)
!             intent(callback) log_error
!             character*(*) :: message
!             real intent(callback) :: log_message
!         end subroutine report_and_raise_error
!     end interface gmd_user_interface
! end python module gmd__user__routines
!
! Then, in each fortran function
! use gmd__user__routines
! intent(callback) log_error
! external log_error
! intent(callback) log_message
! external log_message

subroutine logmes(msg)
  character*(*) msg
  external log_message
  call log_message(msg)
  return
end subroutine logmes

subroutine bombed(msg)
  character*(*) msg
  character*200 jnkstr
  external log_error
  jnkstr = "BOMBED: " // msg
  call log_error(jnkstr)
  return
end subroutine bombed

subroutine faterr(caller, msg)
  character*200 jnkstr
  character*(*) caller
  character*(*) msg
  external log_error
  jnkstr = "FATAL ERROR: " // msg
  call log_error(jnkstr)
  return
end subroutine faterr

subroutine tokens(n, sa, ca)
  integer, intent(in) :: n
  character*(*), intent(in) :: sa(n)
  character(len=1), intent(out) :: ca(*)
  character(len=1), parameter :: pipe='|'
  character(len=1), parameter :: blank=' '
  integer :: i, knt, nchr, ichr
  knt = 0
  do 2 i = 1, n
      do 3 nchr = len(sa(i)), 1, -1
3   if(sa(i)(nchr:nchr).ne.blank) go to 7
7     do 1 ichr = 1, nchr
         knt = knt + 1
         ca(knt) = sa(i)(ichr:ichr)
1     continue
      knt = knt + 1
      ca(knt) = pipe
2  continue
  return
end subroutine tokens
