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
