MODULE NEOHOOKE
  IMPLICIT NONE
  INTEGER, PARAMETER :: DP=SELECTED_REAL_KIND(14)
  INTEGER, PARAMETER :: NDI=3, NSHR=3
  INTEGER, PARAMETER :: NTENS=NDI+NSHR
  REAL(DP), PARAMETER :: ZERO=0._DP, ONE=1._DP, TWO=2._DP, THREE=3._DP
  REAL(DP), PARAMETER :: FOUR=4._DP, SIX=6._DP
  REAL(DP), PARAMETER :: IDENTITY(6)=RESHAPE([ONE,ONE,ONE,ZERO,ZERO,ZERO], &
                                             SHAPE(IDENTITY))
CONTAINS

  SUBROUTINE GET_STRESS(NUI, UI, F, NSV, SV, SIG, C)
    ! --------------------------------------------------------------------- !
    ! COMPRESSIBLE NEO-HOOKEAN HYPERELASTIC MATERIAL
    !
    ! NOTES
    ! -----
    ! SYMMETRIC TENSOR ORDERING : XX, YY, ZZ, XY, YZ, ZX
    ! --------------------------------------------------------------------- !
    USE TENSALG, ONLY : DET
    INTEGER, INTENT(IN) :: NUI, NSV
    REAL(DP), INTENT(IN) :: UI(NUI), F(3,3)
    REAL(DP), INTENT(INOUT) :: SV(NSV), SIG(NTENS), C(NTENS,NTENS)
    INTEGER :: I, J
    REAL(DP) :: EE(6), EEP(3), BBP(3), BBN(3,3)
    REAL(DP) :: EMOD, ENU, C10, D1, EG, EK, EG23, PR
    REAL(DP) :: JAC, SCALE, FB(3,3), BB(6), TRBBAR

    ! ELASTIC PROPERTIES
    EMOD = UI(1)
    ENU = UI(2)
    C10 = EMOD / (FOUR * (ONE + ENU))
    D1 = SIX * (ONE - TWO * ENU) / EMOD

    ! JACOBIAN AND DISTORTION TENSOR
    JAC = DET(F)
    SCALE = JAC **(-ONE / THREE)
    FB = SCALE * F

    ! DEVIATORIC LEFT CAUCHY-GREEN DEFORMATION TENSOR
    BB(1) = FB(1,1) * FB(1,1) + FB(1,2) * FB(1,2) + FB(1,3) * FB(1,3)
    BB(2) = FB(2,1) * FB(2,1) + FB(2,2) * FB(2,2) + FB(2,3) * FB(2,3)
    BB(3) = FB(3,1) * FB(3,1) + FB(3,2) * FB(3,2) + FB(3,3) * FB(3,3)
    BB(4) = FB(2,1) * FB(1,1) + FB(2,2) * FB(1,2) + FB(2,3) * FB(1,3)
    BB(5) = FB(3,1) * FB(2,1) + FB(3,2) * FB(2,2) + FB(3,3) * FB(2,3)
    BB(6) = FB(3,1) * FB(1,1) + FB(3,2) * FB(1,2) + FB(3,3) * FB(1,3)
    TRBBAR = SUM(BB(1:3)) / THREE
    EG = TWO * C10 / JAC
    EK = TWO / D1 * (TWO * JAC - ONE)
    PR = TWO / D1 * (JAC - ONE)

    ! CAUCHY STRESS
    SIG = EG * (BB - TRBBAR * IDENTITY) + PR * IDENTITY

    ! SPATIAL STIFFNESS
    EG23 = EG * TWO / THREE
    C(1,1) =  EG23 * (BB(1) + TRBBAR) + EK
    C(1,2) = -EG23 * (BB(1) + BB(2)-TRBBAR) + EK
    C(1,3) = -EG23 * (BB(1) + BB(3)-TRBBAR) + EK
    C(1,4) =  EG23 * BB(4) / TWO
    C(1,5) = -EG23 * BB(5)
    C(1,6) =  EG23 * BB(6) / TWO

    C(2,2) =  EG23 * (BB(2) + TRBBAR) + EK
    C(2,3) = -EG23 * (BB(2) + BB(3)-TRBBAR) + EK
    C(2,4) =  EG23 * BB(4) / TWO
    C(2,5) =  EG23 * BB(5) / TWO
    C(2,6) = -EG23 * BB(6)

    C(3,3) =  EG23 * (BB(3) + TRBBAR) + EK
    C(3,4) = -EG23 * BB(4)
    C(3,5) =  EG23 * BB(5) / TWO
    C(3,6) =  EG23 * BB(6) / TWO

    C(4,4) =  EG * (BB(1) + BB(2)) / TWO
    C(4,5) =  EG * BB(6) / TWO
    C(4,6) =  EG * BB(5) / TWO

    C(5,5) =  EG * (BB(2) + BB(3)) / TWO
    C(5,6) =  EG * BB(4) / TWO

    C(6,6) =  EG * (BB(1) + BB(3)) / TWO
    FORALL(I=1:NTENS,J=1:NTENS,J<I) C(I,J) = C(J,I)

    ! ! LOGARITHMIC STRAINS
    ! CALL SPRIND(BB, BBP, BBN, 1, 3, 3)
    ! EEP(1) = LOG(SQRT(BBP(1)) / SCALE)
    ! EEP(2) = LOG(SQRT(BBP(2)) / SCALE)
    ! EEP(3) = LOG(SQRT(BBP(3)) / SCALE)
    ! EE(1) = EEP(1) * BBN(1,1) ** 2 + EEP(2) * BBN(2,1) ** 2 &
    !          + EEP(3) * BBN(3,1) ** 2
    ! EE(2) = EEP(1) * BBN(1,2) ** 2 + EEP(2) * BBN(2,2) ** 2 &
    !          + EEP(3) * BBN(3,2) ** 2
    ! EE(3) = EEP(1) * BBN(1,3) ** 2 + EEP(2) * BBN(2,3) ** 2 &
    !          + EEP(3) * BBN(3,3) ** 2
    ! EE(4) = TWO * (EEP(1) * BBN(1,1) * BBN(1,2) &
    !          + EEP(2) * BBN(2,1) * BBN(2,2) &
    !          + EEP(3) * BBN(3,1) * BBN(3,2))
    ! EE(5) = TWO * (EEP(1) * BBN(1,1) * BBN(1,3) &
    !          + EEP(2) * BBN(2,1) * BBN(2,3) &
    !          + EEP(3) * BBN(3,1) * BBN(3,3))
    ! EE(6) = TWO * (EEP(1) * BBN(1,2) * BBN(1,3) &
    !          + EEP(2) * BBN(2,2) * BBN(2,3) &
    !          + EEP(3) * BBN(3,2) * BBN(3,3))
    ! SV(1:NTENS) = EE(K1)
    RETURN
  END SUBROUTINE GET_STRESS

END MODULE NEOHOOKE
