MODULE LINALG
  IMPLICIT NONE
  INTEGER, PARAMETER :: DP=SELECTED_REAL_KIND(14)
CONTAINS
  REAL(DP) FUNCTION DET(A)
    REAL(DP), INTENT(IN) :: A(3,3)
    DET = A(1,1) * A(2,2) * A(3,3) - A(1,2) * A(2,1) * A(3,3) &
        + A(1,2) * A(2,3) * A(3,1) + A(1,3) * A(3,2) * A(2,1) &
        - A(1,3) * A(3,1) * A(2,2) - A(2,3) * A(3,2) * A(1,1)
  END FUNCTION DET
END MODULE LINALG
