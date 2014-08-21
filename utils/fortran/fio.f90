subroutine logmes(msg)
  character*(*) msg
  print*, msg
  return
end subroutine logmes

subroutine bombed(msg)
  character*(*) msg
  character*200 jnkstr
  jnkstr = "BOMBED: " // msg
  print*, "*********************************************************************"
  print*, "*********************************************************************"
  print*, jnkstr
  print*, "*********************************************************************"
  print*, "*********************************************************************"
  stop
  return
end subroutine bombed

subroutine faterr(caller, msg)
  character*200 jnkstr
  character*(*) caller
  character*(*) msg
  jnkstr = "FATAL ERROR: " // caller // ": " // msg
  print*, "*********************************************************************"
  print*, "*********************************************************************"
  print*, jnkstr
  print*, "*********************************************************************"
  print*, "*********************************************************************"
  stop
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
