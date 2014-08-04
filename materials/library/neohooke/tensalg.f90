module tensalg
  ! TENSOR ALGEBRA PACKAGE
  ! SYMMETRIC TENSOR STORAGE:
  !   VOIGHT FORM WITH COMPONENTS
  !   XX, YY, ZZ, XY, YZ, XZ
  implicit none
  private
  public :: det, inv

  ! PARAMETER DECLARATIONS
  integer, parameter :: dp=selected_real_kind(14)
  real(kind=dp), parameter :: zero=0.0E+00_dp, one=1.0E+00_dp, two=2.0E+00_dp
  real(kind=dp), parameter :: voight(6)=(/one,one,one,two,two,two/)
  real(kind=dp), parameter :: root2=0.1414213562373095048801688724209698078E+01_dp
  real(kind=dp), parameter :: toor2=0.7071067811865475244008443621048490392E+00_dp

  ! PUBLIC INTERFACES
  interface det
     module procedure det_3x3, det_6x1
  end interface det
  interface dbd
     module procedure dbd_3x3, dbd_6x1
  end interface dbd
  interface inv
     module procedure inv_3x3, inv_6x1
  end interface inv
  interface mag
     module procedure mag_3x3, mag_6x1
  end interface mag
  interface cnvaba
     module procedure cnvaba_6x1, cnvaba_6x6
  end interface cnvaba

contains

  ! --------------------------------------------------------- DETERMINANT --- !
  real(kind=dp) function det_3x3(a)
    ! determinant of second order tensor
    real(kind=dp), intent(in) :: a(3,3)
    det_3x3 = a(1,1) * a(2,2) * a(3,3) - a(1,2) * a(2,1) * a(3,3) &
            + a(1,2) * a(2,3) * a(3,1) + a(1,3) * a(3,2) * a(2,1) &
            - a(1,3) * a(3,1) * a(2,2) - a(2,3) * a(3,2) * a(1,1)
  end function det_3x3
  real(kind=dp) function det_6x1(a)
    ! determinant of second order tensor stored as 6x1 array
    real(kind=dp), intent(in) :: a(6)
    det_6x1 = a(1) * a(2) * a(3) - a(1) * a(5) ** 2 &
            - a(2) * a(6) ** 2 - a(3) * a(4) ** 2 &
            + two * a(4) * a(5) * a(6)
  end function det_6x1

  ! ------------------------------------------------------------- INVERSE --- !
  function inv_6x1(a)
    ! Inverse of 3x3 symmetric tensor stored as 6x1 array
    real(kind=dp) :: inv_6x1(6)
    real(kind=dp), intent(in) :: a(6)
    inv_6x1(1) = a(2) * a(3) - a(5) ** 2
    inv_6x1(2) = a(1) * a(3) - a(6) ** 2
    inv_6x1(3) = a(1) * a(2) - a(4) ** 2
    inv_6x1(4) = -a(3) * a(4) + a(5) * a(6)
    inv_6x1(5) = -a(1) * a(5) + a(4) * a(6)
    inv_6x1(6) = -a(2) * a(6) + a(4) * a(5)
    inv_6x1 = inv_6x1 / det_6x1(a)
  end function inv_6x1
  function inv_3x3(a)
    ! Inverse of symmetric second order tensor stored as 3x3 matrix
    real(kind=dp) :: inv_3x3(3,3)
    real(kind=dp), intent(in) :: a(3,3)
    inv_3x3(1,1) =  a(2,2) * a(3,3) - a(2,3) * a(3,2)
    inv_3x3(1,2) = -a(1,2) * a(3,3) + a(1,3) * a(3,2)
    inv_3x3(1,3) =  a(1,2) * a(2,3) - a(1,3) * a(2,2)
    inv_3x3(2,1) = -a(2,1) * a(3,3) + a(2,3) * a(3,1)
    inv_3x3(2,2) =  a(1,1) * a(3,3) - a(1,3) * a(3,1)
    inv_3x3(2,3) = -a(1,1) * a(2,3) + a(1,3) * a(2,1)
    inv_3x3(3,1) =  a(2,1) * a(3,2) - a(2,2) * a(3,1)
    inv_3x3(3,2) = -a(1,1) * a(3,2) + a(1,2) * a(3,1)
    inv_3x3(3,3) =  a(1,1) * a(2,2) - a(1,2) * a(2,1)
    inv_3x3 = inv_3x3 / det_3x3(a)
  end function inv_3x3

  ! ----------------------------------------------------------- MAGNITUDE --- !
  real(kind=dp) function mag_6x1(a)
    ! L2-norm (euclidean magnitude) of second order tensor stored as 6x1 array
    real(kind=dp), intent(in) :: a(6)
    mag_6x1 = sqrt(dbd_6x1(a, a))
    return
  end function mag_6x1
  real(kind=dp) function mag_3x3(a)
    ! L2-norm (euclidean magnitude) of second order tensor stored as 6x1 array
    real(kind=dp), intent(in) :: a(3,3)
    mag_3x3 = sqrt(dbd_3x3(a, a))
    return
  end function mag_3x3

  ! ---------------------------------------------------------- DOUBLE DOT --- !
  real(kind=dp) function dbd_6x1(a, b)
    ! Double dot of second order tensors stored as 6x1 arrays
    real(kind=dp), intent(in) :: a(6), b(6)
    dbd_6x1 = sum(a * b * voight)
    return
  end function dbd_6x1
  real(kind=dp) function dbd_3x3(a, b)
    ! Double dot of second order tensors stored as 3x3 arrays
    real(kind=dp), intent(in) :: a(3,3), b(3,3)
    dbd_3x3 = sum(a * b)
    return
  end function dbd_3x3

  subroutine symleafff(f, ff)
    ! ----------------------------------------------------------------------- !
    ! Compute a 6x6 Mandel matrix that is the sym-leaf transformation of the
    ! input 3x3 matrix F.
    ! ----------------------------------------------------------------------- !
    !
    ! Input
    ! -----
    !    F: any 3x3 matrix (in conventional 3x3 storage)
    !
    ! Output
    ! ------
    !    FF: 6x6 Mandel matrix for the sym-leaf transformation matrix
    !
    ! Authors
    ! -------
    ! Rebecca Brannon, theory, algorithm, and code, Sep 15, 2006
    ! Tim Fuller, clean up, conversion to F90
    !
    ! Notes
    ! -----
    ! If A is any symmetric tensor, and if {A} is its 6x1 Mandel array, then
    ! the 6x1 Mandel array for the tensor B=F.A.Transpose[F] may be computed
    ! by
    !                       {B}=[FF]{A}
    !
    ! If F is a deformation F, then B is the "push" (spatial) transformation
    ! of the reference tensor A If F is Inverse[F], then B is the "pull"
    ! (reference) transformation of the spatial tensor A, and therefore B
    ! would be Inverse[FF]{A}.
    !
    ! If F is a rotation, then B is the rotation of A, and
    ! FF would be be a 6x6 orthogonal matrix, just as is F
    !
    !
    real(kind=dp), intent(in) :: F(3,3)
    real(kind=dp), intent(out) :: FF(6,6)
    integer :: i, j
    real(kind=dp) :: fac
    forall(i=1:3, j=1:3) FF(i,j) = F(i,j) ** 2
    fac = two ! root2
    do i=1,3
       FF(i,4) = fac * F(i,1) * F(i,2)
       FF(i,5) = fac * F(i,2) * F(i,3)
       FF(i,6) = fac * F(i,3) * F(i,1)
       FF(4,i) = fac * F(1,i) * F(2,i)
       FF(5,i) = fac * F(2,i) * F(3,i)
       FF(6,i) = fac * F(3,i) * F(1,i)
    enddo
    FF(4,4) = F(1,2) * F(2,1) + F(1,1) * F(2,2)
    FF(5,4) = F(2,2) * F(3,1) + F(2,1) * F(3,2)
    FF(6,4) = F(3,2) * F(1,1) + F(3,1) * F(1,2)

    FF(4,5) = F(1,3) * F(2,2) + F(1,2) * F(2,3)
    FF(5,5) = F(2,3) * F(3,2) + F(2,2) * F(3,3)
    FF(6,5) = F(3,3) * F(1,2) + F(3,2) * F(1,3)

    FF(4,6) = F(1,1) * F(2,3) + F(1,3) * F(2,1)
    FF(5,6) = F(2,1) * F(3,3) + F(2,3) * F(3,1)
    FF(6,6) = F(3,1) * F(1,3) + F(3,3) * F(1,1)

    return
  end subroutine symleafff

  ! ************************************************************************* !

  function dd66x6(a, x, job)
    ! ----------------------------------------------------------------------- !
    ! Multiply a fourth-order tensor A times a second-order tensor B (or vice
    ! versa if JOB=-1)
    ! ----------------------------------------------------------------------- !
    !
    ! Input
    ! -----
    ! A : ndarray (6,6)
    !     Mandel matrix for a general (not necessarily major-sym) fourth-order
    !     minor-sym matrix
    ! X : ndarray(6,)
    !     Voigt matrix
    !
    ! Output
    ! ------
    ! A:X if JOB=1
    ! X:A if JOB=-1
    integer, intent(in), optional :: job
    real(kind=dp) :: dd66x6(6)
    real(kind=dp), intent(in) :: X(6), A(6,6)
    integer :: ij, io
    real(kind=dp) :: T(6)
    real(kind=dp), parameter :: w(6)=(/zero,zero,zero,root2,root2,root2/)
    ! ------------------------------------------------------------ dd66x6 --- !
    T = X * w
    dd66x6 = zero
    io = 1
    if (present(job)) io = job
    select case(io)
    case(1)
       ! ...Compute the Mandel form of A:X
       forall(ij=1:6) dd66x6(ij) = sum(A(ij,:) * T(:))
    case(-1)
       ! ...Compute the Mandel form of X:A
       forall(ij=1:6) dd66x6(ij) = sum(T(:) * A(:, ij))
    case default
       call bombed('unknown job sent to DD66X6')
    end select
    ! ...Convert result to Voigt form
    dd66x6(4:6) = dd66x6(4:6) * toor2
    return
  end function dd66x6

  ! ************************************************************************* !
  ! conversion subroutines for abaqus. the following subroutines convert
  ! tensors from being ordered as
  !     xx, yy, zz, xy, yz, zx
  ! to the ordering required by abaqus standard
  !     xx, yy, zz, xy, zx, yz
  subroutine cnvaba_6x1(a)
    real(kind=dp), intent(inout) :: a(6)
    a((/1,2,3,4,5,6/)) = a((/1,2,3,4,6,5/))
  end subroutine cnvaba_6x1
  subroutine cnvaba_6x6(a)
    real(kind=dp), intent(inout) :: a(6,6)
    a(:, (/1,2,3,4,5,6/)) = a(:, (/1,2,3,4,6,5/))
    a((/5,6/), (/1,2,3,4/)) = a((/6,5/), (/1,2,3,4/))
  end subroutine cnvaba_6x6

end module tensalg
