! ---------------------------------------------------------------------- !
!  COMPILE USING
!   f2py -m int_field_for -c interp_field_for.f 

subroutine int_field(N_mp,xn,yn, bias_x,bias_y, dx,dy,efx, efy, &
                        Nxg, Nyg, Ex_n, Ey_n)
!f2py intent(in)  N_mp   
!f2py intent(in)  xn                                      
!f2py intent(in)  yn   
!f2py intent(in)  bias_x
!f2py intent(in)  bias_y 
!f2py intent(in)  dx
!f2py intent(in)  dy
!f2py intent(in)  efx
!f2py intent(in)  efy
!f2py intent(in)  Nxg
!f2py intent(in)  Nyg
!f2py intent(out) Ex_n
!f2py intent(out) Ey_n


implicit none
integer                          :: N_mp
real(kind=8),dimension(N_mp)     :: xn, yn
real(kind=8)                     :: bias_x, bias_y, dx, dy
integer                          :: Nxg,Nyg 
real(kind=8),dimension(Nxg, Nyg) :: efx, efy
integer                          :: p
real(kind=8)                     :: fi, fj, hx, hy
integer                          :: i, j
real(kind=8),dimension(N_mp)     :: Ex_n, Ey_n
real(kind=8)                     :: tmpx00, tmpx10, tmpx01, tmpx11
real(kind=8)                     :: tmpy00, tmpy10, tmpy01, tmpy11
    
    
    
do p=1,N_mp
fi = 1+(xn(p)-bias_x)/dx;             !i index of particle's cell 
i  = int(fi);
hx = fi-dble(i);                      !fractional x position in cell


fj = 1+(yn(p)-bias_y)/dy;             !j index of particle' cell(C-like!!!!)
j = int(fj);
hy = fj-dble(j);                      !fractional y position in cell


!gather electric field
if (i>0 .and. j>0 .and. i<Nxg .and. j<Nyg) then
tmpx00 = efx(i,j);
tmpx10 = efx(i+1,j);
tmpx01 = efx(i,j+1);
tmpx11 = efx(i+1,j+1);

Ex_n(p)=tmpx00*(1-hx)*(1-hy) + tmpx10*hx*(1-hy) + tmpx01*(1-hx)*hy + tmpx11*hx*hy;   

tmpy00 = efy(i,j);
tmpy10 = efy(i+1,j);
tmpy01 = efy(i,j+1);
tmpy11 = efy(i+1,j+1);

Ey_n(p)=tmpy00*(1-hx)*(1-hy) + tmpy10*hx*(1-hy) + tmpy01*(1-hx)*hy + tmpy11*hx*hy;
end if
end do

end subroutine int_field

