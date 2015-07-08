C-------------------------------------------------------------
      subroutine uanisohyper_inv (ainv, ua, zeta, nfibers, ninv,
     $     ui1, ui2, ui3, temp, noel, cmname, incmpflag, ihybflag,
     $     numstatev, statev, numfieldv, fieldv, fieldvinc,
     $     numprops, props)
C
      implicit double precision (a-h,o-z)
C
      character*80 cmname
      dimension ua(2), ainv(ninv), ui1(ninv),
     $     ui2(ninv*(ninv+1)/2), ui3(ninv*(ninv+1)/2),
     $     statev(numstatev), fieldv(numfieldv),
     $     fieldvinc(numfieldv), props(numprops)
C
C
C
c      if (cmname(1:10) .eq. 'UANISO_HGO') then
         call UANISOHYPER_INVHGO(ainv, ua, zeta, nfibers, ninv,
     $     ui1, ui2, ui3, temp, noel, cmname, incmpflag, ihybflag,
     $     numstatev, statev, numfieldv, fieldv, fieldvinc,
     $     numprops, props)
c      else if(cmname(1:13) .eq. 'UANISO_INVISO') then
c         call UANISOHYPER_INVISO(ainv, ua, zeta, nfibers, ninv,
c     $     ui1, ui2, ui3, temp, noel, cmname, incmpflag, ihybflag,
c     $     numstatev, statev, numfieldv, fieldv, fieldvinc,
c     $     numprops, props)
c      else if(cmname(1:16) .eq. 'UANISO_FUNGINV44') then
c         call UANISOHYPER_FUNGINV44(ainv, ua, zeta, nfibers, ninv,
c     $     ui1, ui2, ui3, temp, noel, cmname, incmpflag, ihybflag,
c     $     numstatev, statev, numfieldv, fieldv, fieldvinc,
c     $     numprops, props)
c      else if(cmname(1:16) .eq. 'UANISO_FUNGINV45') then
c         call UANISOHYPER_FUNGINV45(ainv, ua, zeta, nfibers, ninv,
c     $     ui1, ui2, ui3, temp, noel, cmname, incmpflag, ihybflag,
c     $     numstatev, statev, numfieldv, fieldv, fieldvinc,
c     $     numprops, props)
c      else
c         write(6,*)'ERROR: User subroutine UANISOHYPER_INV missing!'
c         call xit
c      end if
C
C
C
      return
      end
c------------------------------------------------------------------
c
c     HGO model
c
      subroutine uanisohyper_invhgo (ainv, ua, zeta, nfibers, ninv,
     $     ui1, ui2, ui3, temp, noel, cmname, incmpflag, ihybflag,
     $     numstatev, statev, numfieldv, fieldv, fieldvinc,
     $     numprops, props)
C
      implicit double precision (a-h,o-z)
C
      character*80 cmname
      dimension ua(2), ainv(ninv), ui1(ninv),
     $     ui2(ninv*(ninv+1)/2), ui3(ninv*(ninv+1)/2),
     $     statev(numstatev), fieldv(numfieldv),
     $     fieldvinc(numfieldv), props(numprops)
C
c     ainv: invariants
c     ua  : energies ua(1): utot, ua(2); udev
c     ui1 : dUdI
c     ui2 : d2U/dIdJ
c     ui3 : d3U/dIdJdJ, not used for regular elements
C
      parameter ( half = 0.5d0,
     *            zero = 0.d0,
     *            one  = 1.d0,
     *            two  = 2.d0,
     *            three= 3.d0,
     *            four = 4.d0,
     *            five = 5.d0,
     *            six  = 6.d0,
c
     *            index_I1 = 1,
     *            index_J  = 3,
     *            asmall   = 2.d-16  )
C
C     HGO model
C
      C10 = props(1)
      rk1 = props(3)
      rk2 = props(4)
      rkp = props(5)
c
      ua(2) = zero
      om3kp = one - three * rkp
      do k1 = 1, nfibers
**         index_i4 = 4 + k1*(k1-1) + 2*(k1-1)
         index_i4 = indxInv4(k1,k1)
         E_alpha1 = rkp  * (ainv(index_i1) - three)
     *           + om3kp * (ainv(index_i4) - one  )
         E_alpha = max(E_alpha1, zero)
         ht4a    = half + sign(half,E_alpha1 + asmall)
         aux     = exp(rk2*E_alpha*E_alpha)
c energy
         ua(2) = ua(2) +  aux - one
c ui1
         ui1(index_i1) = ui1(index_i1) + aux * E_alpha
         ui1(index_i4) = rk1 * om3kp * aux * E_alpha
c ui2
         aux2 = ht4a + two * rk2 * E_alpha * E_alpha
         ui2(indx(index_I1,index_I1)) = ui2(indx(index_I1,index_I1))
     *                                + aux * aux2
         ui2(indx(index_I1,index_i4)) = rk1*rkp*om3kp * aux * aux2
         ui2(indx(index_i4,index_i4)) = rk1*om3kp*om3kp*aux * aux2
      end do
c
c     deviatoric energy
c
      ua(2) = ua(2) * rk1 / (two * rk2)
      ua(2) = ua(2) + C10 * (ainv(index_i1) - three)
c
c     compute derivatives
c
      ui1(index_i1) = rk1 * rkp * ui1(index_i1) + C10
      ui2(indx(index_I1,index_I1))= ui2(indx(index_I1,index_I1))
     *                            * rk1 * rkp * rkp
c
c     compressible case
      if(props(2).gt.zero) then
         Dinv = one / props(2)
         det = ainv(index_J)
         ua(1) = ua(2) + Dinv *((det*det - one)/two - log(det))
         ui1(index_J) = Dinv * (det - one/det)
         ui2(indx(index_J,index_J))= Dinv * (one + one / det / det)
         if (hybflag.eq.1) then
           ui3(indx(index_J,index_J))= - Dinv * two / (det*det*det)
         end if
      end if
c
      return
      end
C-------------------------------------------------------------
C     Function to map index from Square to Triangular storage
C 		 of symmetric matrix
C
      integer function indx( i, j )
      implicit double precision (a-h,o-z)
      ii = min(i,j)
      jj = max(i,j)
      indx = ii + jj*(jj-1)/2
      return
      end
C-------------------------------------------------------------
C
C     Function to generate enumeration of scalar
C     Pseudo-Invariants of type 4

      integer function indxInv4( i, j )
      implicit double precision (a-h,o-z)
      ii = min(i,j)
      jj = max(i,j)
      indxInv4 = 4 + jj*(jj-1) + 2*(ii-1)
      return
      end
C-------------------------------------------------------------
C
C     Function to generate enumeration of scalar
C     Pseudo-Invariants of type 5
C
      integer function indxInv5( i, j )
      implicit double precision (a-h,o-z)
      ii = min(i,j)
      jj = max(i,j)
      indxInv5 = 5 + jj*(jj-1) + 2*(ii-1)
      return
      end
C-------------------------------------------------------------
c
c    generalized Fung Anisotropic Model
c    reformulated in terms of J and I4ab
c
c
      subroutine uanisohyper_funginv44(ainv, ua, zeta, nfibers, ninv,
     $     ui1, ui2, ui3, temp, noel, cmname, incmpflag, ihybflag,
     $     numstatev, statev, numfieldv, fieldv, fieldvinc,
     $     numprops, props)
C
      implicit double precision (a-h,o-z)
C
      character*80 cmname
      dimension ua(2), ainv(ninv), ui1(ninv),
     $     ui2(ninv*(ninv+1)/2), ui3(ninv*(ninv+1)/2),
     $     statev(numstatev), fieldv(numfieldv),
     $     fieldvinc(numfieldv), props(numprops)
C
c     ainv: invariants
c     ua  : energies ua(1): utot, ua(2); udev
c     ui1 : dUdI
c     ui2 : d2U/dIdJ
c     ui3 : not used for regular elements; for hybrid define d3U/dJ3
C
      parameter ( half = 0.5d0,
     *            zero = 0.d0,
     *            one  = 1.d0,
     *            two  = 2.d0,
     *            three= 3.d0,
     *            four = 4.d0,
     *            five = 5.d0,
     *            six  = 6.d0,
c
     *            indx_J = 3 )
C
      dimension bmx(6,6),Cg(6),unit(6),index4(6)
      dimension dQdC(6),d2QdC2(6,6),ind4C2I(6),ind4I2C(6)
      dimension eg(6)
C
C     Fung model in invariant form --- in terms of J and I4ab
C
c     active invariants:
c
c     1
c     2
c     3    yes   J
c     4    yes   I4(11)
c     5
c     6    yes   I4(12)
c     7
c     8    yes   I4(22)
c     9
c     10   yes   I4(13)
c     11
c     12   yes   I4(23)
c     13
c     14   yes   I4(33)
c     15
c
c     bmx 11, 22, 33, 12, 13, 23
c

c      write(*,*) 'inside 44 44 44'

      do k1=1,6
         do k2=1,k1
            ktmp = k2 + (k1-1)*k1/2
            if(k1.le.3) then
               bmx(k2,k1) = props(ktmp)
            else
               if(k2.le.3) then
                  bmx(k2,k1) = two * props(ktmp)
               else
                  bmx(k2,k1) = four * props(ktmp)
               end if
            end if
         end do
      end do
      do k1=1,6
         do k2=k1+1,6
            bmx(k2,k1) = bmx(k1,k2)
         end do
      end do
c
      Cval = props(22)
c
c      indx_411 = 4
c      indx_412 = 6
c      indx_422 = 8
c      indx_413 = 10
c      indx_423 = 12
c      indx_433 = 14
c
c      indxC_411 = 1
c      indxC_422 = 2
c      indxC_433 = 3
c      indxC_412 = 4  to be consistent with Fung input
c      indxC_413 = 5  to be consistent with Fung input
c      indxC_423 = 6  to be consistent with Fung input

c
c     index of I4ab (I4_11,12,22,13,23,33) in ainv
c
      index4(1) = 4
      index4(2) = 6
      index4(3) = 8
      index4(4) = 10
      index4(5) = 12
      index4(6) = 14
c
c     index of components in C to invariants
c
      ind4C2I(1) = 4
      ind4C2I(2) = 8
      ind4C2I(3) = 14
      ind4C2I(4) = 6
      ind4C2I(5) = 10
      ind4C2I(6) = 12
c
c     index of invariant number to components in C
c
      ind4I2C(1) = 1
      ind4I2C(2) = 4
      ind4I2C(3) = 2
      ind4I2C(4) = 5
      ind4I2C(5) = 6
      ind4I2C(6) = 3
c
      unit(1) = one
      unit(2) = one
      unit(3) = one
      unit(4) = zero
      unit(5) = zero
      unit(6) = zero
c
      det = ainv(indx_J)
      do k1=1,6
         Cg(k1) = ainv(ind4C2I(k1))
         eg(k1) = half * (Cg(k1) - unit(k1))
      end do
c
      Qval = zero
      do k1=1,6
         do k2=1,6
            Qval = Qval + eg(k1)*bmx(k1,k2)*eg(k2)
         end do
      end do
c
      ua(2) = half * Cval * (exp(Qval) - one)
      ua(1) = ua(2)
c
      if(props(23).gt.zero) then
         Dinv = one / props(23)
         ua(1) = ua(1) + Dinv * (half*(det*det-one) - log(det))
         ui1(indx_J) = Dinv * (det - one/det)
         ui2(indx(indx_J,indx_J)) = Dinv * (one + one / det / det)
         if (ihybflag.eq.1) then
           ui3(indx(indx_J,indx_J)) = - Dinv * two / (det*det*det)
         end if
      end if
c
      do k1=1,6
         dQdC(k1) = zero
         do k2=1,6
            d2QdC2(k1,k2) = half * bmx(k1,k2)
            dQdC(k1) = dQdC(k1) + bmx(k1,k2)*eg(k2)
         end do
      end do
      dUdQ = half * Cval * exp(Qval)
      d2UdQ2 = dUdQ
c
      do k1=1,6
         ui1(index4(k1)) = dUdQ * dQdC(ind4I2C(k1))
         do k2=1,k1
            ui2(indx(index4(k1),index4(k2))) =
     *        dUdQ   * d2QdC2(ind4I2C(k1),ind4I2C(k2))
     *      + d2UdQ2 * dQdC(ind4I2C(k1)) * dQdC(ind4I2C(k2))
         end do
      end do
c
      return
      end
C----------------------------------------------------------------------
c
c     isotropic hyperelasticity through UANISOHYPER_INV
c
      subroutine uanisohyper_inviso (ainv, ua, zeta, nfibers, ninv,
     $     ui1, ui2, ui3, temp, noel, cmname, incmpflag, ihybflag,
     $     numstatev, statev, numfieldv, fieldv, fieldvinc,
     $     numprops, props)
C
      implicit double precision (a-h,o-z)
C
      character*80 cmname
      dimension ua(2), ainv(ninv), ui1(ninv),
     $     ui2(ninv*(ninv+1)/2), ui3(ninv*(ninv+1)/2),
     $     statev(numstatev), fieldv(numfieldv),
     $     fieldvinc(numfieldv), props(numprops)
C
c     ainv: invariants
c     ua  : energies ua(1): utot, ua(2); udev
c     ui1 : dUdI
c     ui2 : d2U/dIdJ
c     ui3 : not used for regular elements; for hybrid define d3U/dJ3
C
      parameter ( half = 0.5d0,
     *            zero = 0.d0,
     *            one  = 1.d0,
     *            two  = 2.d0,
     *            three= 3.d0,
     *            four = 4.d0,
     *            five = 5.d0,
     *            six  = 6.d0,
     *            twt4 = 24.d0,
c
     *            index_I1 = 1,
     *            index_I2 = 2,
     *            index_J  = 3  )
C
C     -- Polynomial with N=2 --
C
      C10 = props(1)
      C01 = props(2)
      C20 = props(3)
      C11 = props(4)
      C02 = props(5)
      D1  = props(6)
      D2  = props(7)
c

      rI1 = ainv(index_I1)
      rI2 = ainv(index_I2)
      rJ  = ainv(index_J )
c
      rI1m3 = rI1 - three
      rI2m3 = rI2 - three
      rJm1  = rJ  - one
c
      ua(2) =   C10 * rI1m3 + C01 * rI2m3
     *        + C20 * rI1m3 * rI1m3 + C11 * rI1m3 * rI2m3
     *        + C02 * rI2m3 * rI2m3
c     compressible case
      if(D1.gt.zero) then
         Dinv1 = one / D1
         ua(1) = ua(2) + Dinv1 * rJm1 * rJm1
         ui1(index_J) = two * Dinv1 * rJm1
         ui2(indx(index_J,index_J)) = two * Dinv1
      end if
c
      if(D2.gt.zero) then
         Dinv2 = one / D2
         ua(1) = ua(1) + Dinv2 * rJm2**4
         ui1(index_J) = ui1(index_J) + four * Dinv2 * rJm1**3
         ui2(indx(index_J,index_J)) = ui2(indx(index_J,index_J))
     *                              + three * four * Dinv2 * rJm1**2
         if (ihybflag.eq.1) then
           ui3(indx(index_J,index_J)) = twt4 * Dinv2 * rJm1
         end if
      end if
c
      ui1(index_I1) = C10 + two * C20 * rI1m3 + C11 * rI2m3
      ui1(index_I2) = C01 + two * C02 * rI2m3 + C11 * rI1m3
c
      ui2(indx(index_I1,index_I1)) = two * C20
      ui2(indx(index_I1,index_I2)) = C11
      ui2(indx(index_I2,index_I2)) = two * C02
c
      return
      end
c-------------------------------------------------------------------------
c
c    generalized Fung Anisotropic Model
c    reformulated in terms of J, I4ab, and I5aa
c
      subroutine uanisohyper_funginv45(ainv, ua, zeta, nfibers, ninv,
     $     ui1, ui2, ui3, temp, noel, cmname, incmpflag, ihybflag,
     $     numstatev, statev, numfieldv, fieldv, fieldvinc,
     $     numprops, props)
C
      implicit double precision (a-h,o-z)
C
      character*80 cmname
      dimension ua(2), ainv(ninv), ui1(ninv),
     $     ui2(ninv*(ninv+1)/2), ui3(ninv*(ninv+1)/2),
     $     statev(numstatev), fieldv(numfieldv),
     $     fieldvinc(numfieldv), props(numprops)
C
c     ainv: invariants
c     ua  : energies ua(1): utot, ua(2); udev
c     ui1 : dUdI
c     ui2 : d2U/dIdJ
c     ui3 : not used for regular elements; for hybrid define d3U/dJ3
C
      parameter ( half = 0.5d0,
     *            zero = 0.d0,
     *            one  = 1.d0,
     *            two  = 2.d0,
     *            three= 3.d0,
     *            four = 4.d0,
     *            five = 5.d0,
     *            six  = 6.d0,
     *            quat = 0.25d0,
c
     *            indx_J = 3 )
C
      dimension bmx(6,6),Cg(6),unit(6),indI(6),ind4C2I(6),eg(6)
      dimension dQdC(6),d2QdC2(6,6),aI(6),d2QdCdI(6,6)
      dimension dQdI(6),d2QdI2(6,6),dCdI(6,6),d2CdI2(6,6,6)
C
C     Fung model in invariant form --- in terms of J and I4ab
C
c     active invariants:
c
c     1
c     2
c     3    yes   J
c     4
c     5    yes   I5(11)
c     6    yes   I4(12)
c     7
c     8
c     9    yes   I5(22)
c     10   yes   I4(13)
c     11
c     12   yes   I4(23)
c     13
c     14
c     15   yes   I5(33)
c
c     bmx 11, 22, 33, 12, 13, 23
c
c      write(*,*) 'inside 45 45 45'

      do k1=1,6
         do k2=1,k1
            ktmp = k2 + (k1-1)*k1/2
            if(k1.le.3) then
               bmx(k2,k1) = props(ktmp)
            else
               if(k2.le.3) then
                  bmx(k2,k1) = two * props(ktmp)
               else
                  bmx(k2,k1) = four * props(ktmp)
               end if
            end if
         end do
      end do
      do k1=1,6
         do k2=k1+1,6
            bmx(k2,k1) = bmx(k1,k2)
         end do
      end do
c
      Cval = props(22)
c
c      indx_411 = 4
c      indx_412 = 6
c      indx_422 = 8
c      indx_413 = 10
c      indx_423 = 12
c      indx_433 = 14
c
c      indxC_411 = 1
c      indxC_422 = 2
c      indxC_433 = 3
c      indxC_412 = 4  to be consistent with Fung input
c      indxC_413 = 5  to be consistent with Fung input
c      indxC_423 = 6  to be consistent with Fung input

c
c     index of invariants
c
      indI(1) = 5
      indI(2) = 6
      indI(3) = 9
      indI(4) = 10
      indI(5) = 12
      indI(6) = 15
c
      ind4C2I(1) = 4
      ind4C2I(2) = 8
      ind4C2I(3) = 14
      ind4C2I(4) = 6
      ind4C2I(5) = 10
      ind4C2I(6) = 12
c
      unit(1) = one
      unit(2) = one
      unit(3) = one
      unit(4) = zero
      unit(5) = zero
      unit(6) = zero
c
      det = ainv(indx_J)
      do k1=1,6
         aI(k1) = ainv(indI(k1))
         Cg(k1) = ainv(ind4C2I(k1))
         eg(k1) = half * (Cg(k1) - unit(k1))
      end do
c
      Qval = zero
      do k1=1,6
         do k2=1,6
            Qval = Qval + eg(k1)*bmx(k1,k2)*eg(k2)
         end do
      end do
      dUdQ = half * Cval * exp(Qval)
      d2UdQ2 = dUdQ
c
      ua(2) = half * Cval * (exp(Qval) - one)
      ua(1) = ua(2)
      if(props(23).gt.zero) then
         Dinv = one / props(23)
         ua(1) = ua(1) + Dinv * (half*(det*det-one) - log(det))
         ui1(indx_J) = Dinv * (det - one/det)
         ui2(indx(indx_J,indx_J)) = Dinv * (one + one / det / det)
         if (ihybflag.eq.1) then
           ui3(indx(indx_J,indx_J)) = - Dinv * two / (det*det*det)
         end if
      end if
c
c     dQdC and d2QdC2
      do k1=1,6
         dQdC(k1) = zero
         do k2=1,6
            d2QdC2(k1,k2) = half * bmx(k1,k2)
            dQdC(k1) = dQdC(k1) + bmx(k1,k2)*eg(k2)
         end do
      end do
c
c     compute dCdI(6,6) and d2CdI2(6,6,6)
c     not the best way to do.......
      do k3=1,6
         do k2=1,6
            dCdI(k2,k3) = zero
            do k1=1,6
               d2CdI2(k1,k2,k3) = zero
            end do
         end do
      end do
c
c     dC1dI(6)
      dCdI(1,1) = one / (two * Cg(1))
      dCdI(1,2) = - aI(2) / Cg(1)
      dCdI(1,4) = - aI(4) / Cg(1)
c     d2C1dI2(1,6,6)
      aux = - one / (two * Cg(1) * Cg(1))
      d2CdI2(1,1,1) = aux * dCdI(1,1)
      d2CdI2(1,1,2) = aux * dCdI(1,2)
      d2CdI2(1,1,4) = aux * dCdI(1,4)
      d2CdI2(1,2,1) = d2CdI2(1,1,2)
      d2CdI2(1,4,1) = d2CdI2(1,1,4)

      aux1 = - one / Cg(1)
      aux  = one / (Cg(1) * Cg(1))
      d2CdI2(1,2,2) = aux1 + aux * aI(2) * dCdI(1,2)
      d2CdI2(1,2,4) =        aux * aI(2) * dCdI(1,4)
      d2CdI2(1,4,2) = d2CdI2(1,2,4)
c
      d2CdI2(1,4,4) = aux1 + aux * aI(4) * dCdI(1,4)
c
c
c     dC2dI(6)
      dCdI(2,2) = - aI(2) / Cg(2)
      dCdI(2,3) = one / (two * Cg(2))
      dCdI(2,5) = - aI(5) / Cg(2)
c     d2C2dI(2,6,6)
      aux1 = - one / Cg(2)
      aux  = one / (Cg(2) * Cg(2))
      d2CdI2(2,2,2) = aux1 + aux * aI(2) * dCdI(2,2)
      d2CdI2(2,2,3) =        aux * aI(2) * dCdI(2,3)
      d2CdI2(2,2,5) =        aux * aI(2) * dCdI(2,5)
      d2CdI2(2,3,2) = d2CdI2(2,2,3)
      d2CdI2(2,5,2) = d2CdI2(2,2,5)

      d2CdI2(2,3,3) =  - half * aux * dCdI(2,3)
      d2CdI2(2,3,5) =  - half * aux * dCdI(2,5)
      d2CdI2(2,5,3) = d2CdI2(2,3,5)

      d2CdI2(2,5,5) = aux1 + aux * aI(5) * dCdI(2,5)
c
c
c     dC3dI(6)
      dCdI(3,4) = - aI(4) / Cg(3)
      dCdI(3,5) = - aI(5) / Cg(3)
      dCdI(3,6) = one / (two * Cg(3))
c     d2C3dI(3,6,6)
      aux1 = - one / Cg(3)
      aux  = one / (Cg(3) * Cg(3))
      d2CdI2(3,4,4) = aux1 + aux * aI(4) * dCdI(3,4)
      d2CdI2(3,4,5) =        aux * aI(4) * dCdI(3,5)
      d2CdI2(3,4,6) =        aux * aI(4) * dCdI(3,6)
      d2CdI2(3,5,4) = d2CdI2(3,4,5)
      d2CdI2(3,6,4) = d2CdI2(3,4,6)

      d2CdI2(3,5,5) = aux1 + aux * aI(5) * dCdI(3,5)
      d2CdI2(3,5,6) =        aux * aI(5) * dCdI(3,6)
      d2CdI2(3,6,5) = d2CdI2(3,5,6)

      d2CdI2(3,6,6) =   - half * aux * dCdI(3,6)
c
      dCdI(4,2) = one
      dCdI(5,4) = one
      dCdI(6,5) = one
c
      do k1=1,6
         dQdI(k1) = zero
         do k2=1,6
            dQdI(k1) = dQdI(k1) + dQdC(k2) * dCdI(k2,k1)
         end do
      end do

      call aprd(d2QdC2,dCdI,d2QdCdI,6,6,6)
      do k1=1,6
         do k2=1,6
            d2QdI2(k1,k2) = zero
            do k3=1,6
               d2QdI2(k1,k2) = d2QdI2(k1,k2)
     *                       + dCdI(k3,k1)*d2QdCdI(k3,k2)
     *                       + dQdC(k3)*d2CdI2(k3,k1,k2)
            end do
         end do
      end do
c
      do k1=1,6
         ui1(indI(k1)) = dUdQ * dQdI(k1)
      end do
      do k1=1,6
         do k2=1,6
            ui2(indx(indI(k1),indI(k2))) = d2UdQ2 * dQdI(k1) * dQdI(k2)
     *           + dUdQ * d2QdI2(k1,k2)
         end do
      end do
c
      return
      end
C----------------------------------------------------------------------
      subroutine aprd(A,B,C,n,m,k)
c
      implicit double precision (a-h,o-z)
c
      parameter (zero = 0.d0)
      dimension A(n,m),B(m,k),C(n,k)
c
      do k1=1,n
         do k2=1,k
            C(k1,k2) = zero
            do k3=1,m
               C(k1,k2)=C(k1,k2)+A(k1,k3)*B(k3,k2)
            end do
         end do
      end do
c
      return
      end
C----------------------------------------------------------------------
      subroutine aTprd(A,B,C,n,m,k)
c
      implicit double precision (a-h,o-z)
c
      parameter (zero = 0.d0)
      dimension A(n,m),B(m,k),C(n,k)
c
      do k1=1,n
         do k2=1,k
            C(k1,k2) = zero
            do k3=1,m
               C(k1,k2)=C(k1,k2)+A(k1,k3)*B(k2,k3)
            end do
         end do
      end do
c
      return
      end
C
C User subroutine uanisohyper_strain
c verification using built-in Fung orthotropic model
c
      subroutine uanisohyper_strain (
     *     ebar, aj, ua, du1, du2, du3, temp, noel, cmname,
     *     incmpFlag, ihybFlag, ndi, nshr, ntens,
     *     numStatev, statev, numFieldv, fieldv, fieldvInc,
     *     numProps, props)
c
      implicit double precision (a-h,o-z)
c
      dimension ebar(ntens), ua(2), du1(ntens+1)
      dimension du2((ntens+1)*(ntens+2)/2),du3((ntens+1)*(ntens+2)/2)
      dimension statev(numStatev), fieldv(numFieldv)
      dimension fieldvInc(numFieldv), props(numProps)
c
      double precision geW(6)
c
      character*80 cmname
c
      parameter ( half    = 0.5d0,
     $            one     = 1.d0,
     $            two     = 2.d0,
     $            four    = 4.d0 )
*
      c10=props(10)
      cod=props(11)
      dinv=one/cod
*
* --- strain energy function
*
      geW(1) =  props(1)*ebar(1)
     $        + props(2)*ebar(2)
     $        + props(4)*ebar(3)
      geW(2) =  props(2)*ebar(1)
     $        + props(3)*ebar(2)
     $        + props(5)*ebar(3)
      geW(3) =  props(4)*ebar(1)
     $        + props(5)*ebar(2)
     $        + props(6)*ebar(3)
      geW(4) =  props(7)*ebar(4)*two
      geW(5) =  props(8)*ebar(5)*two
      geW(6) =  props(9)*ebar(6)*two
      expQ =  geW(1)*ebar(1)
     $      + geW(2)*ebar(2)
     $      + geW(3)*ebar(3)
     $      + geW(4)*ebar(4)*two
     $      + geW(5)*ebar(5)*two
     $      + geW(6)*ebar(6)*two
      expQ = exp(expQ)
      udev = half*c10*(expQ - one)
      tmpC = c10*expQ
* ---- derivatives ue1R wrt E_ij
      du1(1) = tmpC*geW(1)
      du1(2) = tmpC*geW(2)
      du1(3) = tmpC*geW(3)
      du1(4) = tmpC*geW(4)
      du1(5) = tmpC*geW(5)
      du1(6) = tmpC*geW(6)
*  ---- derivatives ue2R, ue3R wrt E_ij / J
      du2( 1) = (props(1)+two*geW(1)*geW(1))*tmpC
      du2( 2) = (props(2)+two*geW(1)*geW(2))*tmpC
      du2( 3) = (props(3)+two*geW(2)*geW(2))*tmpC
      du2( 4) = (props(4)+two*geW(1)*geW(3))*tmpC
      du2( 5) = (props(5)+two*geW(2)*geW(3))*tmpC
      du2( 6) = (props(6)+two*geW(3)*geW(3))*tmpC
      du2( 7) =           two*geW(1)*geW(4) *tmpC
      du2( 8) =           two*geW(2)*geW(4) *tmpC
      du2( 9) =           two*geW(3)*geW(4) *tmpC
      du2(10) = (props(7)+two*geW(4)*geW(4))*tmpC
      du2(11) =           two*geW(1)*geW(5) *tmpC
      du2(12) =           two*geW(2)*geW(5) *tmpC
      du2(13) =           two*geW(3)*geW(5) *tmpC
      du2(14) =           two*geW(4)*geW(5) *tmpC
      du2(15) = (props(8)+two*geW(5)*geW(5))*tmpC
      du2(16) =           two*geW(1)*geW(6) *tmpC
      du2(17) =           two*geW(2)*geW(6) *tmpC
      du2(18) =           two*geW(3)*geW(6) *tmpC
      du2(19) =           two*geW(4)*geW(6) *tmpC
      du2(20) =           two*geW(5)*geW(6) *tmpC
      du2(21) = (props(9)+two*geW(6)*geW(6))*tmpC
*
* --- volumetric contribution and derivatives
*
      ntensp1 = ntens+1
      ntensp2 = ntensp1*(ntensp1+1)/2
      detJ  = aj
      detJi = one/detJ
      utot = udev + dinv*(half*(detJ*detJ-one)-log(detJ))
      du1(ntensp1) = (detJ - detJi)*dinv
      du2(ntensp2) = (one + detJi*detJi)*dinv
*
      return
      end
