!---------------------------------------------------------------------- !
! COMPILE USING
!  f2py -m nmod -c thisfile.f   

subroutine int_field_border(N_mp,xn,yn, bias_x,bias_y, dx,dy, &
                            efx, efy, Nxg, Nyg, Ex_n, Ey_n, inside_mat)
!f2py intent(in)  N_mp   
!f2py intent(in)  xn                                      
!f2py intent(in)  yn   
!f2py intent(in)  bias_x
!f2py intent(in)  bias_y 
!f2py intent(in)  dx
!f2py intent(in)  dy
!f2py intent(in)  efx
!f2py intent(in)  efy
!f2py intent(in)  inside_mat
!f2py intent(in)  Nxg
!f2py intent(in)  Nyg
!f2py intent(out) Ex_n
!f2py intent(out) Ey_n


implicit none
integer  N_mp
real(kind=8),dimension(N_mp)        ::   xn, yn
real(kind=8)                        :: bias_x, bias_y, dx, dy
integer                             :: Nxg,Nyg 
real(kind=8),dimension(Nxg, Nyg)    ::  efx(Nxg, Nyg), efy(Nxg, Nyg)
integer(kind=1),dimension(Nxg, Nyg) :: inside_mat(Nxg, Nyg)
integer                             :: p
real(kind=8)                        :: fi, fj, hx, hy
integer                             :: i, j
real(kind=8),dimension(N_mp)        :: Ex_n(N_mp), Ey_n(N_mp)
real(kind=8)                        :: wei_ij,  wei_i1j, wei_ij1, wei_i1j1 
real(kind=8)                        :: fact_correct  
logical                             :: anyexternal 
real(kind=8)                        :: tmpx00, tmpx10, tmpx01, tmpx11
real(kind=8)                        :: tmpy00, tmpy10, tmpy01, tmpy11

do p=1,N_mp
fi = 1+(xn(p)-bias_x)/dx;             !i index of particle's cell 
i  = int(fi);
hx = fi-dble(i);                      !fractional x position in cell


fj = 1+(yn(p)-bias_y)/dy;             !j index of particle' cell(C-like!!!!)
j = int(fj);
hy = fj-dble(j);                      !fractional y position in cell

anyexternal = .false.
if (inside_mat(i, j)==0) then
    wei_ij =0.
    anyexternal = .true.
else
    wei_ij = (1-hx)*(1-hy)
end if

if (inside_mat(i+1, j)==0) then
    wei_i1j =0.
    anyexternal = .true.
else       
    wei_i1j = hx*(1-hy)
end if        


if (inside_mat(i, j+1)==0) then
    wei_ij1 =0.
    anyexternal = .true.
else          
    wei_ij1 = (1-hx)*hy
end if
    
if (inside_mat(i+1, j+1)==0) then
    wei_i1j1 = 0.
    anyexternal = .true.
else
    wei_i1j1  =  hx*hy  
end if

!gather electric field
if (i>0 .and. j>0 .and. i<Nxg .and. j<Nyg) then

    if (anyexternal .eqv. .true.) then
        if ((wei_ij+wei_i1j+wei_ij1+wei_i1j1)>0.) then
            fact_correct = 1./(wei_ij+wei_i1j+wei_ij1+wei_i1j1)
        else
            fact_correct = 0.
            fact_correct = 0.
        end if
    else
        fact_correct = 1.
    end if
    
    tmpx00 = efx(i,j);
    tmpx10 = efx(i+1,j);
    tmpx01 = efx(i,j+1);
    tmpx11 = efx(i+1,j+1);

    Ex_n(p) = fact_correct * (tmpx00*wei_ij + tmpx10*wei_i1j + tmpx01*wei_ij1 + tmpx11*wei_i1j1);

    tmpy00 = efy(i,j);
    tmpy10 = efy(i+1,j);
    tmpy01 = efy(i,j+1);
    tmpy11 = efy(i+1,j+1);

    Ey_n(p) = fact_correct * (tmpy00*wei_ij + tmpy10*wei_i1j + tmpy01*wei_ij1 + tmpy11*wei_i1j1);   

end if
end do



end subroutine int_field_border

