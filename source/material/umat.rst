
User Model to Define a Material's Mechanical Response
#####################################################

Overview
========

*umat* is a material model that completely describes the material behavior.

Interface
=========

.. code:: fortran

   subroutine umat(stress,statev,ddsdde,sse,spd,scd,rpl,ddsddt,drplde,drpldt,&
        stran,dstran,time,dtime,temp,dtemp,predef,dpred,cmname, ndi,nshr,ntens,&
        nstatv,props,nprops,fiber_dir,nfibers,drot,pnewdt,celent,f0,f1,noel,npt,&
        layer,kspt,kstep,kinc)
     implicit none

     character*8, intent(in) :: cmname
     integer, intent(in) :: ndi, nshr, ntens, nstatv, nprops, nfibers
     integer, intent(in) :: noel, npt, layer, kspt, kstep, kinc
     integer, parameter :: dp=selected_real_kind(14)
     real(kind=dp), intent(in) :: sse, spd, scd, rpl, drpldt, time(2), dtime
     real(kind=dp), intent(in) :: temp, dtemp, pnewdt, celent
     real(kind=dp), intent(inout) :: stress(ntens), statev(nstatv)
     real(kind=dp), intent(inout) :: ddsdde(ntens, ntens)
     real(kind=dp), intent(inout) :: ddsddt(ntens), drplde(ntens)
     real(kind=dp), intent(in) :: stran(ntens), dstran(ntens)
     real(kind=dp), intent(in) :: predef(1), dpred(1)
     real(kind=dp), intent(in) :: props(nprops),fiber_dir(nfibers,3)
     real(kind=dp), intent(in) :: drot(3, 3), f0(3, 3), f1(3, 3)

     ! User coding

   end subroutine umat
