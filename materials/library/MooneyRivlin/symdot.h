  INTERFACE
     FUNCTION SYMDOT(X, Y)
       INCLUDE "mnrv.h"
       REAL(DP) :: SYMDOT(6)
       REAL(DP), INTENT(IN) :: X(6), Y(6)
     END FUNCTION SYMDOT
  END INTERFACE

! Local Variables:
! mode: f90
! End:
