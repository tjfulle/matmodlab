      MODULE LINELAS
        IMPLICIT NONE
        INTEGER, PARAMETER :: DP=SELECTED_REAL_KIND(14)
        INTEGER, PARAMETER :: NDI=3, NSHR=3
        INTEGER, PARAMETER :: NTENS=NDI+NSHR
        REAL(DP), PARAMETER :: ZERO=0._DP, ONE=1._DP, TWO=2._DP
        REAL(DP), PARAMETER :: IDENTITY(6)=RESHAPE([ONE,ONE,ONE,
     &                                              ZERO,ZERO,ZERO], 
     &                                              SHAPE(IDENTITY))
      CONTAINS

        SUBROUTINE GET_STRESS(NUI, UI, EEINCIN, DT, NSV, SV, SIG, C)
          ! --------------------------------------------------------------------- !
          ! ISOTROPIC LINEAR-ELASTIC MATERIAL
          !
          ! NOTES
          ! -----
          ! SYMMETRIC TENSOR ORDERING : XX, YY, ZZ, XY, XZ, YZ
          ! --------------------------------------------------------------------- !
          INTEGER, INTENT(IN) :: NUI, NSV
          REAL(DP), INTENT(IN) :: UI(NUI), EEINCIN(NTENS), DT
          REAL(DP), INTENT(INOUT) :: SV(NSV), SIG(NTENS), C(NTENS,NTENS)
          INTEGER :: I, J
          REAL(DP) :: ELAM, EG, EEINC(NTENS)

          ! CONVERT THE ENGINEERING SHEAR STRAINS TO REGUlAR SHEAR STRAINS
          EEINC(:3) = EEINCIN(:3)
          EEINC(3:) = EEINCIN(3:) / TWO

          ! ELASTIC PROPERTIES
          ! UI(1) = E; UI(2) = NU
          ELAM = UI(1) * UI(2) / ((ONE + UI(2)) * (ONE - TWO * UI(2)))
          EG = UI(1) / (TWO * (ONE + UI(2)))

          ! CAUCHY STRESS
          SIG = SIG + ELAM*SUM(EEINC(:3))*IDENTITY + TWO*EG*EEINC

          ! SPATIAL STIFFNESS
          C(1,1) = TWO * EG + ELAM
          C(1,2) = ELAM
          C(1,3) = ELAM
          C(1,4) = ZERO
          C(1,5) = ZERO
          C(1,6) = ZERO

          C(2,2) = TWO * EG + ELAM
          C(2,3) = ELAM
          C(2,4) = ZERO
          C(2,5) = ZERO
          C(2,6) = ZERO

          C(3,3) = TWO * EG + ELAM
          C(3,4) = ZERO
          C(3,5) = ZERO
          C(3,6) = ZERO

          C(4,4) = EG
          C(4,5) = ZERO
          C(4,6) = ZERO

          C(5,5) = EG
          C(5,6) = ZERO

          C(6,6) = EG

          ! POPULATE THE OTHER HALF OF THE STIFFNESS
          FORALL(I=1:NTENS,J=1:NTENS,J<I) C(I,J) = C(J,I)

          RETURN
        END SUBROUTINE GET_STRESS

      END MODULE LINELAS


      SUBROUTINE UMAT(STRESS, STATEV, DDSDDE, SSE, SPD, SCD, RPL,
     &     DDSDDT, DRPLDE,DRPLDT,STRAN,DSTRAN,TIME,DTIME,TEMP,DTEMP,
     &     PREDEF,DPRED, CMNAME, NDI, NSHR, NTENS, NSTATV, PROPS,
     &     NPROPS, COORDS, DROT, PNEWDT, CELENT, DFGRD0, DFGRD1, NOEL,
     &     NPT, LAYER, KSPT, KSTEP, KINC)
        USE LINELAS, ONLY : GET_STRESS
        IMPLICIT DOUBLE PRECISION (A-H, O-Z)
        CHARACTER*8 CMNAME
        DIMENSION STRESS(NTENS), STATEV(NSTATV), DDSDDE(NTENS, NTENS),
     &       DDSDDT(NTENS), DRPLDE(NTENS), STRAN(NTENS), DSTRAN(NTENS),
     &       PREDEF(1), DPRED(1), PROPS(NPROPS), COORDS(3), DROT(3, 3),
     &       DFGRD0(3, 3), DFGRD1(3, 3)
        DIMENSION F(3,3), C(6,6), S(6)

        ! CHECK INPUTS
        IF (NDI /= 3) THEN
           PRINT *, 'UMAT REQUIRES NDI=3'
           CALL XIT
        END IF

        ! COPY THE STRESS TENSOR TO PASS IT IN
        S(1:NTENS) = STRESS(1:NTENS)

        ! CALL THE ROUTINE
        CALL GET_STRESS(NPROPS, PROPS, DSTRAN, DTIME,
     &                  NSTATV, STATEV, S, C)

        ! TRANSFER FOUND QUANTITIES
        STRESS(1:NTENS) = S(1:NTENS)
        DDSDDE(1:NTENS,1:NTENS) = C(1:NTENS,1:NTENS)

        RETURN
      END SUBROUTINE UMAT
