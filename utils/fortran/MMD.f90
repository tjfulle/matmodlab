

program MMD_main

 ! Fortran Material Model Driver (MMD) main program. This is primarily a
 ! management program that sets up the overall structure. The real driver
 ! is subroutine MMD.

   use MMD_m
   use MM1_m
!   use MM2_m
   use MM3_m
   use MM4_m
   use MM5_m
   use MM6_m
   use MM7_m
   use MM8_m
   use MM9_m
   use MM10_m
   use MM11_m
   use MM12_m

   implicit none

   integer MM_index

 ! -----------------------------------------------------------------------------

 ! Check to see if Excel-generated input file exists, and open
 ! appropriate file units accordingly

   inquire (file=Excel_Input_File, exist=Excel)
   if (Excel) then
      open (in , file=Excel_Input_File,  action='read' )
      open (out, file=Excel_Output_File, action='write')
   else
      print  *
      write (*, '(a)', advance='no' ) 'Enter input/output file name: '
      read  (*, '(a)', advance='yes')  Console_InOut_File
      print  *
      open  (in , status='scratch'       , action='readwrite')
      open  (out, file=Console_InOut_File, action='readwrite')
      call   remove_comments
   end if

 ! Input material model index number, and call Material Model Driver subroutine
 ! with pointers to appropriate material model and its initialization routine

   read (in, *)  MM_index
   select case  (MM_index)
      case (1)
         call MMD (MM1,  MM1_init)
!      case (2)
!         call MMD (MM2,  MM2_init)
      case (3)
         call MMD (MM3,  MM3_init)
      case (4)
         call MMD (MM4,  MM4_init)
      case (5)
         call MMD (MM5,  MM5_init)
      case (6)
         call MMD (MM6,  MM6_init)
      case (7)
         call MMD (MM7,  MM7_init)
      case (8)
         call MMD (MM8,  MM8_init)
      case (9)
         call MMD (MM9,  MM9_init)
      case (10)
         call MMD (MM10, MM10_init)
      case (11)
         call MMD (MM11, MM11_init)
      case (12)
         call MMD (MM12, MM12_init)
      case default
         print *
         print *, '   ***Error***  Material model index =', MM_index
         print *, '                is not a valid value'
         print *
   end select

   end program MMD_main


!*******************************************************************************


subroutine MMD (MM, MM_init)

 ! This is the real driver.

   use MMD_m
   implicit none

   external                MM, MM_init

   integer, parameter   :: maxit1=20, maxit2=30
   real(8), allocatable :: auxin(:,:), auxout(:), svsave(:)
   real(8), allocatable :: Js(:,:), Jsi(:,:), sigspec(:,:), sigerr(:)
   integer, allocatable :: v(:)
   real(8)              :: d(6), dsave(6), eps(6), sig(6), sigsave(6), J0(6,6)
   real(8)              :: dt, t, tleg(2), dum(11), delt, c(6,2), sigdum(6,2)
   real(8)              :: a1, a2, relerr, tol1, tol2
   integer              :: leg, npts, nprt, ltype(6), i, n, vdum(6)
   integer              :: icond, iconv, ierr=0, it, derr
   character            :: fmt*20, ltype_ch*6

 ! -----------------------------------------------------------------------------

 ! Initialize

   call MM_init ! in:
                ! out: sv,nsv,nauxin,nauxout

   ! allocate storage
   allocate ( auxin(nauxin,3), auxout(nauxout), svsave(nsv) )
   if (Excel) write (out, *) nauxout
   if (Excel) then
      fmt = '(i5, 2x, 100es  .  )'
      write (fmt(15:16), '(i2)') precision(1.d0) + 7
      write (fmt(18:19), '(i2)') precision(1.d0) - 1
   else
      fmt = '(i5, 2x, 100es11. 3)'
   endif

   !initial leg
   leg     = 0
   tleg(2) = 0
   t       = 0
   d       = 0
   dt      = 0
   eps     = 0
   sig     = 0
   c(:,2)  = 0

   ! call material model with zero state
   read  ( in, *, end=9) dum, auxin(:,2)
   svsave = sv
   call MM (d, dt, sig, svsave, auxin(:,2), auxout)
   write (out, fmt) leg, t, eps, sig, auxout

   ! compute initial jacobian matrix

   ! nv is the dimension of the needed jacobian matrix (nv x nv)
   ! for the intial jacobian, we need the whole thing
   nv = 6
   deallocate (   v   , stat=derr); allocate (   v (nv)    )

   ! v array is an array of integers that contains the rows and columns of
   ! the slice needed in the jacobian subroutine.
   v  = (/1,2,3,4,5,6/)
   dt = 1
   call Jacobian (MM, d, dt, sig, sv, auxin(:,2), v, J0)
   nv = 0
   deallocate (   v   , stat=derr); allocate (   v (nv)    )
   deallocate (sigspec, stat=derr); allocate (sigspec(nv,3))

 ! Process each leg

   legs: do

    ! Read inputs and initialize for this leg

      tleg  (  1) = tleg   (  2)
      sigdum(:,1) = sig    (:  )
      sigdum(v,1) = sigspec(:,2)
      c     (:,1) = c      (:,2)
      auxin (:,1) = auxin  (:,2)
      read (in, *, end=9) leg, tleg(2), npts, nprt, ltype_ch, c(:,2), auxin(:,2)
      read (ltype_ch, '(6i1)') ltype
      delt = tleg(2) - tleg(1)
      nv = 0

      do i = 1, 6
         select case (ltype(i))
            case (1)                                       ! strain rate
               d(i) =  c(i,2)
            case (2)                                       ! strain
               d(i) = (c(i,2) - eps(i)) / delt
            case (3)                                       ! stress rate
               sigdum(i,2) = sigdum(i,1) + c(i,2) * delt
               nv = nv + 1
               vdum(nv) = i
            case (4)                                       ! stress
               sigdum(i,2) = c(i,2)
               nv = nv + 1
               vdum(nv) = i
            case default
               print *
               print *, '   ***Error***  Invalid load type (ltype) parameter'
               print *, '                specified for leg', leg
               print *
               stop
         end select
      end do

      deallocate (   v   , stat=derr); allocate (   v (nv)    )
      deallocate (  Js   , stat=derr); allocate ( Js (nv,nv)  )
      deallocate (  Jsi  , stat=derr); allocate ( Jsi(nv,nv)  )
      deallocate (sigspec, stat=derr); allocate (sigspec(nv,3))
      deallocate (sigerr , stat=derr); allocate (sigerr (nv  ))

      v  = vdum(1:nv)
      sigspec(:,1:2) = sigdum(v,1:2)
      Js = J0(v,v)
      call matinv (Js, Jsi, nv, icond)
      d(v) = matmul ( Jsi, (sigspec(:,2)-sigspec(:,1))/delt )
      t  = tleg(1)
      dt = delt / npts

    ! Process this leg

      steps: do n = 1, npts

         t = t + dt
         a1 = dble(npts - n) / npts
         a2 = dble(   n    ) / npts
         auxin  (:,3) = a1 * auxin  (:,1) + a2 * auxin  (:,2)
         sigspec(:,3) = a1 * sigspec(:,1) + a2 * sigspec(:,2)

         if (nv == 0) then

          ! Only strains are specified. Simply evaluate material model:

            call MM (d, dt, sig, sv, auxin(:,3), auxout)

         else

          ! One or more stresses, sig(v), are specified. Need to solve for the
          ! unknown strain rates, d(v), such that sig(v) = sigspec. First save
          ! some quantnties:

            dsave   = d
            sigsave = sig
            svsave  = sv

          ! Try Newton's method with initial d(v) = values from previous time
          ! step:

            call Newton (MM, d, dt, sig, sv, auxin(:,3), auxout, v,            &
                                                         sigspec(:,3), iconv)
            if (iconv > 0) go to 1

          ! Didn't converge. Try Newton's method with initial d(v) = 0:

            d(v) = 0.d0
            sig  = sigsave
            sv   = svsave
            call Newton (MM, d, dt, sig, sv, auxin(:,3), auxout, v,            &
                                                         sigspec(:,3), iconv)
            if (iconv > 0) go to 1

          ! Still didn't converge. Try downhill simplex method with initial
          ! d(v) = 0. Accept whatever answer it returns:

            d(v) = 0.d0
            sig  = sigsave
            sv   = svsave
            call simplex (MM, d, dt, sig, sv, auxin(:,3), auxout, v,           &
                                                                sigspec(:,3))

       1 end if

         eps = eps + d * dt
         eps    = (eps    + 1.d-80) - 1.d-80
         sig    = (sig    + 1.d-80) - 1.d-80
         auxout = (auxout + 1.d-80) - 1.d-80
         if (mod(npts-n, nprt) == 0) then
            write (out, fmt) leg, t, eps, sig, auxout
         end if

      end do steps

   end do legs


 9 if (Excel) write (out, *) ierr

   deallocate ( auxin ,  stat=derr)
   deallocate ( auxout,  stat=derr)
   deallocate ( svsave,  stat=derr)
   deallocate (   v   ,  stat=derr)
   deallocate (  Js   ,  stat=derr)
   deallocate (  Jsi  ,  stat=derr)
   deallocate (sigspec,  stat=derr)
   deallocate (sigerr ,  stat=derr)

end subroutine MMD


!*******************************************************************************


subroutine Newton (MM, d, dt, sig, sv, auxin, auxout, v, sigspec, iconv)

 ! This procedure seeks to determine the unknown strain rates, d(v), needed to
 ! satisfy
 !
 !         sig(d(v)) = sigspec(:)
 !
 ! where v is a vector subscript array containing the components for which
 ! stresses (or stress rates) are specified, and sigspec(:) are the specified
 ! values at the current time.
 !
 ! The approach is an iterative scheme employing a multidimensional Newton's
 ! method. Each iteration begins with a call to subroutine Jacobian, which
 ! numerically computes the Jacobian submatrix Js = J(v,v), where J(:,;) is the
 ! full Jacobian matrix J = (Jij) = (dsigi/depsj). The inverse of this sub-
 ! matrix, Jsi, is then computed via a call to subroutine matinv. The value of
 ! d(v) is then updated according to
 !
 !         d(v) = d(v) - matmul (Jsi, sigerr(d(v))) / dt
 !
 ! where sigerr(d(v)) = sig(d(v)) - sigspec(:). This process is repeated until
 ! a convergence critierion is satisfied. The argument iconv is a flag indicat-
 ! ing whether or not the procedure converged:
 !
 !         iconv = 0  did not converge
 !                 1  converged based on tol1 (more stringent)
 !                 2  converged based on tol2 (less stringent)

   use MMD_m, only: nsv, nauxin, nauxout, nv
   implicit none

   external    MM
   real(8)  :: d(6), dt, sig(6), sv(nsv), auxin(nauxin), auxout(nauxout)
   real(8)  :: sigspec(nv)
   integer  :: v(nv), iconv

   real(8)  :: sigsave(6), svsave(nsv), sigerr(nv), Js(nv,nv), Jsi(nv,nv)
   real(8)  :: relerr, tol1, tol2, depsmax
   integer  :: it, maxit1, maxit2, icond

 ! -----------------------------------------------------------------------------

 ! Initialize

   tol1    = 10 * epsilon(sig)
   tol2    = sqrt(epsilon(sig)) / 10
   maxit1  = 20
   maxit2  = 30
   depsmax = 0.2d0
   iconv   = 0
   sigsave = sig
   svsave  = sv
   if (depsmag() > depsmax) return
   call MM (d, dt, sig, sv, auxin, auxout)
   sigerr = sig(v) - sigspec

 ! Perform Newton iteration

   do it = 1, maxit2
      sig = sigsave
      sv  = svsave
      call Jacobian (MM, d, dt, sig, sv, auxin, v, Js)
      call matinv (Js, Jsi, nv, icond)
      if (icond > 0) exit
      d(v) = d(v) - matmul (Jsi, sigerr) / dt
      if (depsmag() > depsmax) return
      call MM (d, dt, sig, sv, auxin, auxout)
      sigerr = sig(v) - sigspec
      relerr = maxval (abs(sigerr) / max(abs(sigspec), 1.d0))
      if (it <= maxit1) then
         if (relerr < tol1) then
            iconv = 1
            exit
         end if
      else
         if (relerr < tol2) then
            iconv = 2
            exit
         end if
      end if
   end do

contains

   function depsmag ()
      real(8) :: depsmag
      depsmag = sqrt( sum(d(1:3)**2) + 2 * sum(d(4:6)**2) ) * dt
   end function depsmag

end subroutine Newton


!*******************************************************************************


subroutine Jacobian (MM, d, dt, sig, sv, auxin, v, Jsub)

 ! This procedure numerically computes and returns a specified submatrix, Jsub,
 ! of the Jacobian matrix J = (Jij) = (dsigi/depsj). The submatrix returned is
 ! the one formed by the intersections of the rows and columns specified in the
 ! vector subscript array, v. That is, Jsub = J(v,v). The physical array con-
 ! taining this submatrix is assumed to be dimensioned Jsub(nv,nv), where nv is
 ! the number of elements in v. Note that in the special case v = /1,2,3,4,5,6/,
 ! with nv = 6, the matrix that is returned is the full Jacobian matrix, J.
 !
 ! The components of Jsub are computed numerically using a centered differencing
 ! scheme which requires two calls to the material model subroutine for each
 ! element of v. The centering is about the point eps = epsold + d * dt, where
 ! d is the rate-of-strain array.

   use MMD_m, only: nsv, nauxin, nauxout, nv
   implicit none

   external    MM
   real(8)  :: d(6), dt, sig(6), sv(nsv), auxin(nauxin), Jsub(nv,nv)
   integer  :: v(nv)

   real(8)  :: dp(6), sigp(6), svp(nsv)
   real(8)  :: dm(6), sigm(6), svm(nsv)
   real(8)  :: auxout(nauxout), deps
   integer  :: n

 ! -----------------------------------------------------------------------------

   deps = sqrt( epsilon(d) )

   do n = 1, nv
      dp       = d
      dp(v(n)) = d(v(n)) + (deps / dt) / 2
      sigp     = sig
      svp      = sv
      call MM (dp, dt, sigp, svp, auxin, auxout)
      dm       = d
      dm(v(n)) = d(v(n)) - (deps / dt) / 2
      sigm     = sig
      svm      = sv
      call MM (dm, dt, sigm, svm, auxin, auxout)
      Jsub(:,n) = ( sigp(v) - sigm(v) ) / deps
   end do

end subroutine Jacobian


!*******************************************************************************


subroutine matinv (A, Ainv, n, icond)

 ! This procedure computes the inverse of a real, general matrix using Gauss-
 ! Jordan elimination with partial pivoting. The input matrix, A, is returned
 ! unchanged, and the inverted matrix is returned in Ainv. The procedure also
 ! returns an integer flag, icond, indicating whether A is well- or ill-
 ! conditioned. If the latter, the contents of Ainv will be garbage.
 !
 ! The logical dimensions of the matrices A(1:n,1:n) and Ainv(1:n,1:n) are
 ! assumed to be the same as the physical dimensions of the storage arrays
 ! A(1:np,1:np) and Ainv(1:np,1:np), i.e., n = np. If A is not needed, A and
 ! Ainv can share the same storage locations.
 !
 ! input arguments:
 !
 !     A         real  matrix to be inverted
 !     n         int   number of rows/columns
 !
 ! output arguments:
 !
 !     Ainv      real  inverse of A
 !     icond     int   conditioning flag:
 !                       = 0  A is well-conditioned
 !                       = 1  A is  ill-conditioned

   implicit none
   integer  :: n, icond
   real(8)  :: A(n,n), Ainv(n,n)
   real(8)  :: W(n,n), Wmax, dum(n), fac, Wcond=1.d-13
   integer  :: row, col, v(1)

 ! -----------------------------------------------------------------------------

 ! Initialize

   icond = 0
   Ainv  = 0
   do row = 1, n
      Ainv(row,row) = 1
   end do
   W = A
   do row = 1, n
      v = maxloc( abs( W(row,:) ) )
      Wmax = W(row,v(1))
      if (Wmax == 0) then
         icond = 1
         return
      end if
      W   (row,:) = W   (row,:) / Wmax
      Ainv(row,:) = Ainv(row,:) / Wmax
   end do

 ! Gauss-Jordan elimination with partial pivoting

   do col = 1, n
      v = maxloc( abs( W(col:,col) ) )
      row = v(1) + col - 1
      dum(col:)   = W(col,col:)
      W(col,col:) = W(row,col:)
      W(row,col:) = dum(col:)
      dum(:)      = Ainv(col,:)
      Ainv(col,:) = Ainv(row,:)
      Ainv(row,:) = dum(:)
      Wmax = W(col,col)
      if ( abs(Wmax) < Wcond ) then
         icond = 1
         return
      end if
      row = col
      W(row,col:) = W(row,col:) / Wmax
      Ainv(row,:) = Ainv(row,:) / Wmax
      do row = 1, n
         if (row == col) cycle
         fac = W(row,col)
         W(row,col:) = W(row,col:) - fac * W(col,col:)
         Ainv(row,:) = Ainv(row,:) - fac * Ainv(col,:)
      end do
   end do

end subroutine matinv


!*******************************************************************************


subroutine simplex (MM, d, dt, sig, sv, auxin, auxout, v, sigspec)

 ! This procedure seeks to determine the unknown strain rates, d(v), needed to
 ! satisfy
 !
 !         sig(d(v)) = sigspec(:)
 !
 ! where v is a vector subscript array containing the components for which
 ! stresses (or stress rates) are specified, and sigspec(:) are the specified
 ! values at the current time.
 !
 ! This is a backup procedure that is only called if the multidimensional
 ! Newton's method fails. If it is called, it is assumed that there is something
 ! funny about the material model, e.g., it may be nonsmooth or discontinuous.
 ! The approach that is used is a low-order approach which seeks to minimize an
 ! objective function, func(d(v)), given by the sum of the squares of the homo-
 ! geneous equations to be satisfied, i.e.,
 !
 !         func(d(v)) = sum ( (sig(d(v)) - sigspec(:))**2 ).
 !
 ! Note that this function is always nonnegative. For a continuous function
 ! with a valid solution, the minimum is zero and occurs at a solution. For
 ! a discontinuous function the minimum may be greater than zero, but *may*
 ! still provide a valid solution in the same sense as bracketing a disconti-
 ! nuity in one dimension.
 !
 ! The approach used to minimize this function is the downhill simplex method
 ! of Nelder and Mead. A discussion of this can be found in Numerical Recipes,
 ! but the coding in this procedure is based on the original paper by N&M.
 !
 ! One unfortunate feature of this (or any similar) approach is that it can
 ! sometimes get "stuck" at the wrong answer. To circumvent this, the basic
 ! method (inner loop) is placed in an outer loop which repeatedly restarts it
 ! until the answer no longer changes. Unfortunately, for functions that *may*
 ! be discontinuous, there is no good way of knowing whether the answer is
 ! correct, or what its precision is. Accordingly, the procedure simply returns
 ! the best answer it can, with no convergence flag.

   use MMD_m, only: nsv, nauxin, nauxout, nv
   implicit none

   external    MM
   real(8)  :: d(6), dt, sig(6), sv(nsv), auxin(nauxin), auxout(nauxout)
   real(8)  :: sigspec(nv)
   integer  :: v(nv)

   real(8)  :: sigsave(6), svsave(nsv)
   real(8)  :: p(nv+1,nv), y(nv+1), pold(nv+1,nv), ploold(nv), padd, pbar(nv)
   real(8)  :: ptest(nv), ytest, ptest2(nv), ytest2, ptest3(nv), ytest3
   integer  :: np, mp, nc, ncmax, i, ilo, ihi, inhi, mout, maxout
   logical  :: first

 ! -----------------------------------------------------------------------------

 ! Initialize

   sigsave = sig
   svsave  = sv
   np      = nv
   mp      = np + 1
   ncmax   = 100 * mp
   maxout  = 100
   p(1,:)  = d(v)
   y(1)    = func ( p(1,:) )
   first   = .true.

 ! Perform simplex search

   outer: do mout = 1, maxout
      ploold = p(1,:)
      if (first) then
         padd  = maxval( abs(d) ) / 100
         if (padd <= 0 ) padd = 1.d-5 / dt
         first = .false.
      else
         padd  = 5 * maxval( abs(p(2:mp,:)-spread(p(1,:),dim=1,ncopies=mp-1)) )
      end if
      p(2:mp,:) = spread(p(1,:), dim=1, ncopies=mp-1)
      do i = 2, mp
         p(i,i-1) = p(i,i-1) + padd
         y(i)     = func ( p(i,:) )
      end do
      nc = 0

      inner: do
         call reorder
         pold   = p
         pbar   = sum (p(1:mp-1,:), dim=1) / (mp - 1)
         ptest  = pbar + (pbar - p(ihi,:))
         ytest  = func (ptest)
         if ( ytest < y(ilo) ) then
            ptest2 = pbar + 2 * (pbar - p(ihi,:))
            ytest2 = func(ptest2)
            if ( ytest2 < y(ilo) ) then
               p(ihi,:) = ptest2
               y(ihi)   = ytest2
            else
               p(ihi,:) = ptest
               y(ihi)   = ytest
            end if
            go to 1
         end if
         if ( ytest <= y(inhi) ) then
            p(ihi,:) = ptest
            y(ihi)   = ytest
            go to 1
         end if
         if (ytest <= y(ihi)) then
            p(ihi,:) = ptest
            y(ihi)   = ytest
         end if
         ptest3 = (pbar + p(ihi,:)) / 2
         ytest3 = func(ptest3)
         if (ytest3 <= y(ihi)) then
            p(ihi,:) = ptest3
            y(ihi)   = ytest3
            go to 1
         end if
         p(2:mp,:) = ( p(2:mp,:) + spread(p(ilo,:),dim=1,ncopies=mp-1) ) / 2
         do i = 2, mp
            p(i,:) = (p(i,:) + p(ilo,:)) / 2
            y(i)   = func ( p(i,:) )
         end do
      1  if ( maxval(abs(p - pold)) <= 0 .or. nc >= ncmax) exit inner
      end do inner

      if ( maxval(abs(p(ilo,:) - ploold)) <= 0 ) exit outer
   end do outer

 ! Finalize result

   y(ilo) = func ( p(ilo,:) )

contains

   function func (x)
      real(8) :: func, x(nv), w(6)=(/1,1,1,2,2,2/)
      nc   = nc + 1
      d(v) = x
      sig  = sigsave
      sv   = svsave
      call MM (d, dt, sig, sv, auxin, auxout)
      func = sum ( w(v) * (sig(v) - sigspec)**2 )
   end function func

   subroutine reorder
      integer :: idum(1)
      idum = minloc (y)
      ilo  = idum(1)
      call swap (ilo, 1)
      ilo  = 1
      idum = maxloc (y(2:mp))
      ihi  = idum(1) + 1
      call swap (ihi, mp)
      ihi  = mp
      idum = maxloc (y(1:mp-1))
      inhi = idum(1)
   end subroutine reorder

   subroutine swap (i1, i2)
      integer :: i1, i2
      real(8) :: ptemp(np), ytemp
      ptemp   = p(i1,:)
      ytemp   = y(i1  )
      p(i1,:) = p(i2,:)
      y(i1  ) = y(i2  )
      p(i2,:) = ptemp
      y(i2  ) = ytemp
   end subroutine swap

end subroutine simplex


!*******************************************************************************


subroutine remove_comments

 ! This procedure reads the input portion of a console input/output file, and
 ! creates a new scratch input file with the comments removed. It also positions
 ! the input/output file at the correct position for the first output record.
 ! Note that it is not necessary to do this for an Excel-generated input file,
 ! since comments are already removed by the Excel VBA program.
 !
 ! File unit nomenclature:
 !
 !  unit    file
 !
 !  in      scratch input file with comments removed
 !  out     console input/output file

   use MMD_m, only: in, out, Excel
   implicit none
   character(1) :: c

 ! -----------------------------------------------------------------------------

   do
      read (out, '(a)', advance='no', eor=3, end=4) c
      if (c == '!') then
         read (out, '()', advance='yes')
      else if (c == '&') then
         rewind (in)
         if (Excel) return
         read (out, '()', advance='yes')
         read (out, '()', advance='yes', end=1)
         backspace (out)
         return
 1       backspace (out)
         write (out, '()', advance='yes')
         backspace (out)
         backspace (out)
         do
            read (out, '(a)', advance='yes') c
            if (c == '&') return
         end do
      else
         do
            write (in , '(a)', advance='no'       ) c
            read  (out, '(a)', advance='no', eor=2) c
         end do
 2       write (in, '()', advance='yes')
      endif
 3 end do

 4 print *
   print *, '   ***Error***  Input/Output file must have record with "&"'
   print *, '                in column 1 indicating End of Input (EOI)'
   print *
   stop

end subroutine remove_comments


!*******************************************************************************
