#----------------------------------------------------------------------
#                                                                      
#                           CERN                                       
#                                                                      
#     European Organization for Nuclear Research                       
#                                                                      
#     
#     This file is part of the code:
#                                                                      
# 
#                  PyPIC Version 2.4.5                     
#                  
#                                                                       
#     Author and contact:   Giovanni IADAROLA 
#                           BE-ABP Group                               
#                           CERN                                       
#                           CH-1211 GENEVA 23                          
#                           SWITZERLAND  
#                           giovanni.iadarola@cern.ch                  
#                                                                      
#                contact:   Giovanni RUMOLO                            
#                           BE-ABP Group                               
#                           CERN                                      
#                           CH-1211 GENEVA 23                          
#                           SWITZERLAND  
#                           giovanni.rumolo@cern.ch                    
#                                                                      
#
#                                                                      
#     Copyright  CERN,  Geneva  2011  -  Copyright  and  any   other   
#     appropriate  legal  protection  of  this  computer program and   
#     associated documentation reserved  in  all  countries  of  the   
#     world.                                                           
#                                                                      
#     Organizations collaborating with CERN may receive this program   
#     and documentation freely and without charge.                     
#                                                                      
#     CERN undertakes no obligation  for  the  maintenance  of  this   
#     program,  nor responsibility for its correctness,  and accepts   
#     no liability whatsoever resulting from its use.                  
#                                                                      
#     Program  and documentation are provided solely for the use  of   
#     the organization to which they are distributed.                  
#                                                                      
#     This program  may  not  be  copied  or  otherwise  distributed   
#     without  permission. This message must be retained on this and   
#     any other authorized copies.                                     
#                                                                      
#     The material cannot be sold. CERN should be  given  credit  in   
#     all references.                                                  
#----------------------------------------------------------------------

import numpy as np
import scipy.sparse as scsp
from scipy.sparse.linalg import spsolve
import scipy.sparse.linalg as ssl
from .PyPIC_Scatter_Gather import PyPIC_Scatter_Gather
from scipy.constants import e, epsilon_0
import cupy as cp
from line_profiler import profile

from . import int_field_for_border as iffb


na = lambda x:np.array([x])

qe = e
eps0 = epsilon_0

cuda_src = r'''
extern "C" __global__
void int_field_border(const long N_mp,
               const double* __restrict__ xn,
               const double* __restrict__ yn,
               const double bias_x,
               const double bias_y,
               const double dx,
               const double dy,
               const double* __restrict__ efx,
               const double* __restrict__ efy,
               const int Nxg,
               const int Nyg,
               const long stride_i,
               const long stride_j,
               const unsigned char* __restrict__ inside_mat, // NEW: byte mask (0 = outside)
               double* __restrict__ Ex_n,
               double* __restrict__ Ey_n)
{
    long p = blockDim.x * blockIdx.x + threadIdx.x;
    if (p >= N_mp) return;

    // Compute cell indices like Fortran INT() (truncate toward zero)
    double fi = 1.0 + (xn[p] - bias_x) / dx;
    double fj = 1.0 + (yn[p] - bias_y) / dy;

    int i = (int)fi;
    int j = (int)fj;

    double hx = fi - (double)i;
    double hy = fj - (double)j;

    double Ex = 0.0, Ey = 0.0;

    // Bounds: equivalent to Fortran (i>0 .and. j>0 .and. i<Nxg .and. j<Nyg)
    if (i > 0 && j > 0 && i < Nxg && j < Nyg) {
        // For corner nodes we use (i-1,j-1) base like your existing kernel
        int i0 = i - 1, j0 = j - 1;

        long idx00 = (long)i0       * stride_i + (long)j0       * stride_j; // (i,   j)
        long idx10 = (long)(i0 + 1) * stride_i + (long)j0       * stride_j; // (i+1, j)
        long idx01 = (long)i0       * stride_i + (long)(j0 + 1) * stride_j; // (i,   j+1)
        long idx11 = (long)(i0 + 1) * stride_i + (long)(j0 + 1) * stride_j; // (i+1, j+1)

        // Base bilinear weights
        double w00 = (1.0 - hx) * (1.0 - hy);
        double w10 = hx * (1.0 - hy);
        double w01 = (1.0 - hx) * hy;
        double w11 = hx * hy;

        // Apply inside_mat mask per-corner; track if any corner is external
        bool anyExternal = false;

        unsigned char m00 = inside_mat[idx00];
        unsigned char m10 = inside_mat[idx10];
        unsigned char m01 = inside_mat[idx01];
        unsigned char m11 = inside_mat[idx11];

        if (m00 == 0) { w00 = 0.0; anyExternal = true; }
        if (m10 == 0) { w10 = 0.0; anyExternal = true; }
        if (m01 == 0) { w01 = 0.0; anyExternal = true; }
        if (m11 == 0) { w11 = 0.0; anyExternal = true; }

        // Gather E with possibly zeroed weights
        Ex = efx[idx00]*w00 + efx[idx10]*w10 + efx[idx01]*w01 + efx[idx11]*w11;
        Ey = efy[idx00]*w00 + efy[idx10]*w10 + efy[idx01]*w01 + efy[idx11]*w11;

        // If any corner was external, renormalize by sum of remaining weights (if > 0)
        if (anyExternal) {
            double sumw = w00 + w10 + w01 + w11;
            if (sumw > 0.0) {
                double inv = 1.0 / sumw;
                Ex *= inv;
                Ey *= inv;
            } else {
                Ex = 0.0;
                Ey = 0.0;
            }
        }
    }

    Ex_n[p] = Ex;
    Ey_n[p] = Ey;
}''';

mod = cp.RawModule(code=cuda_src, options=('-std=c++11','-O3','--use_fast_math','--gpu-architecture=sm_70',), backend='nvcc')
int_field_kernel = mod.get_function('int_field_border')

@profile
def _strides_in_elements(arr2d):
    return (arr2d.strides[0] // arr2d.itemsize,
            arr2d.strides[1] // arr2d.itemsize)

@profile
def int_field_border_cu(xn, yn, bias_x, bias_y, dx, dy,
                 efx, efy, inside_mat,
                 *, Ex_n=None, Ey_n=None, stream=None):
    """
    CuPy wrapper for the int_field kernel (with inside_mat masking).

    Parameters
    ----------
    xn, yn : (N_mp,) cupy.ndarray, float64
    bias_x, bias_y, dx, dy : float
    efx, efy : (Nxg, Nyg) cupy.ndarray, float64
    inside_mat : (Nxg, Nyg) cupy.ndarray, uint8/byte (0 => outside, nonzero => inside)
    Ex_n, Ey_n : optional output buffers (float64); if None they are allocated
    stream : cp.cuda.Stream or None

    Returns
    -------
    Ex_n, Ey_n : (N_mp,) cupy.ndarray, float64
    """
    # basic shape checks
    Nxg, Nyg = map(int, efx.shape)
    # assert efy.shape == (Nxg, Nyg)
    # assert inside_mat.shape == (Nxg, Nyg), "inside_mat must match efx/efy shape"
    # inside_mat must be byte/uint8 for the kernel's unsigned char*
    if inside_mat.dtype != cp.uint8:
        inside_mat = inside_mat.astype(cp.uint8, copy=False)

    N_mp = int(xn.size)

    # allocate outputs if needed
    # if Ex_n is None: Ex_n = cp.empty_like(xn)
    # if Ey_n is None: Ey_n = cp.empty_like(xn)

    # ensure strides are consistent across the three 2D fields
    stride_i, stride_j = _strides_in_elements(efx)
    # si2, sj2 = _strides_in_elements(efy)
    # sim, sjm = _strides_in_elements(inside_mat)
    # assert (si2, sj2) == (stride_i, stride_j), "efy must share layout with efx"
    # assert (sim, sjm) == (stride_i, stride_j), "inside_mat must share layout with efx"

    threads = 512
    blocks = (N_mp + threads - 1) // threads

    args = (
        cp.int64(N_mp),
        xn, yn,
        float(bias_x), float(bias_y),
        float(dx), float(dy),
        efx, efy,
        cp.int32(Nxg), cp.int32(Nyg),
        cp.int64(stride_i), cp.int64(stride_j),
        inside_mat,                # <-- NEW: mask goes before outputs
        Ex_n, Ey_n
    )

    if stream is None:
        int_field_kernel((blocks,), (threads,), args)
    else:
        with stream:
            int_field_kernel((blocks,), (threads,), args)
    # cp.cuda.get_current_stream().synchronize()
    return Ex_n, Ey_n


class FiniteDifferences_ShortleyWeller_SquareGrid(PyPIC_Scatter_Gather):
    #@profile
    def __init__(self,chamb, Dh, sparse_solver = 'scipy_slu', tol_stem = 0.01, tol_der = 0.1, include_solver=True):

        print('Start PIC init.:')
        print('Finite Differences, Shortley-Weller, Square Grid')
        print('Using Shortley-Weller boundary approx.')

        self.Dh = Dh
        super(FiniteDifferences_ShortleyWeller_SquareGrid, self).__init__(chamb.x_aper, chamb.y_aper, self.Dh, self.Dh)
        Nyg, Nxg = self.Nyg, self.Nxg

        [xn, yn]=np.meshgrid(self.xg,self.yg)

        xn=xn.T
        xn=xn.flatten()

        yn=yn.T
        yn=yn.flatten()
        #% xn and yn are stored such that the external index is on x 

        flag_outside_n=chamb.is_outside(xn,yn)
        flag_inside_n=~(flag_outside_n)

        flag_outside_n_mat=np.reshape(flag_outside_n,(Nyg,Nxg),'F');
        flag_outside_n_mat=flag_outside_n_mat.T
        [gx,gy]=np.gradient(np.double(flag_outside_n_mat));
        gradmod=abs(gx)+abs(gy);
        flag_border_mat=np.logical_and((gradmod>0), flag_outside_n_mat);
        flag_border_n = flag_border_mat.flatten()


        if include_solver:
            A=scsp.lil_matrix((Nxg*Nyg,Nxg*Nyg)); #allocate a sparse matrix
            Dx=scsp.lil_matrix((Nxg*Nyg,Nxg*Nyg)); #allocate a sparse matrix
            Dy=scsp.lil_matrix((Nxg*Nyg,Nxg*Nyg)); #allocate a sparse matrix

            list_internal_force_zero = []
            # Build A Dx Dy matrices 
            for u in range(0,Nxg*Nyg):
                if np.mod(u, Nxg*Nyg//20)==0:
                    print(('Mat. assembly %.0f'%(float(u)/ float(Nxg*Nyg)*100)+"""%"""))
                if flag_inside_n[u]:

                    #Compute Shortley-Weller coefficients
                    if flag_inside_n[u-1]: #phi(i-1,j)
                        hw = Dh
                    else:
                        x_int,y_int,z_int,Nx_int,Ny_int, i_found_int = chamb.impact_point_and_normal(na(xn[u]), na(yn[u]), na(0.), na(xn[u-1]), na(yn[u-1]), na(0.), resc_fac=.995, flag_robust=False)
                        hw = np.abs(y_int[0]-yn[u])

                    if flag_inside_n[u+1]: #phi(i+1,j)
                        he = Dh
                    else:
                        x_int,y_int,z_int,Nx_int,Ny_int, i_found_int = chamb.impact_point_and_normal(na(xn[u]), na(yn[u]), na(0.), na(xn[u+1]), na(yn[u+1]), na(0.), resc_fac=.995, flag_robust=False)
                        he = np.abs(y_int[0]-yn[u])

                    if flag_inside_n[u-Nyg]: #phi(i,j-1)
                        hs = Dh
                    else:
                        x_int,y_int,z_int,Nx_int,Ny_int, i_found_int = chamb.impact_point_and_normal(na(xn[u]), na(yn[u]), na(0.), na(xn[u-Nyg]), na(yn[u-Nyg]), na(0.), resc_fac=.995, flag_robust=False)
                        hs = np.abs(x_int[0]-xn[u])
                        #~ print hs

                    if flag_inside_n[u+Nyg]: #phi(i,j+1)
                        hn = Dh
                    else:
                        x_int,y_int,z_int,Nx_int,Ny_int, i_found_int = chamb.impact_point_and_normal(na(xn[u]), na(yn[u]), na(0.), na(xn[u+Nyg]), na(yn[u+Nyg]), na(0.), resc_fac=.995, flag_robust=False)
                        hn = np.abs(x_int[0]-xn[u])
                        #~ print hn


                    # Build A matrix
                    if hn<Dh*tol_stem or hs<Dh*tol_stem or hw<Dh*tol_stem or he<Dh*tol_stem: # nodes very close to the bounday
                        A[u,u] =1.
                        list_internal_force_zero.append(u)
                        #print u, xn[u], yn[u]
                    else:
                        A[u,u] = -(2./(he*hw)+2/(hs*hn))
                        A[u,u-1]=2./(hw*(hw+he));     #phi(i-1,j)nx
                        A[u,u+1]=2./(he*(hw+he));     #phi(i+1,j)
                        A[u,u-Nyg]=2./(hs*(hs+hn));    #phi(i,j-1)
                        A[u,u+Nyg]=2./(hn*(hs+hn));    #phi(i,j+1)

                    # Build Dx matrix
                    if hn<Dh*tol_der:
                        if hs>=Dh*tol_der:
                            Dx[u,u] = -1./hs
                            Dx[u,u-Nyg]=1./hs
                    elif hs<Dh*tol_der:
                        if hn>=Dh*tol_der:
                            Dx[u,u] = 1./hn
                            Dx[u,u+Nyg]=-1./hn
                    else:
                        Dx[u,u] = (1./(2*hn)-1./(2*hs))
                        Dx[u,u-Nyg]=1./(2*hs)
                        Dx[u,u+Nyg]=-1./(2*hn)


                    # Build Dy matrix	
                    if he<Dh*tol_der:
                        if hw>=Dh*tol_der:
                            Dy[u,u] = -1./hw
                            Dy[u,u-1]=1./hw
                    elif hw<Dh*tol_der:
                        if he>=Dh*tol_der:
                            Dy[u,u] = 1./he
                            Dy[u,u+1]=-1./(he)
                    else:
                        Dy[u,u] = (1./(2*he)-1./(2*hw))
                        Dy[u,u-1]=1./(2*hw)
                        Dy[u,u+1]=-1./(2*he)

                else:
                    # external nodes
                    A[u,u]=1.


            flag_force_zero = flag_outside_n.copy()
            for ind in list_internal_force_zero:
                flag_force_zero[ind] = True

            flag_force_zero_mat=np.reshape(flag_force_zero,(Nyg,Nxg),'F');
            flag_force_zero_mat=flag_force_zero_mat.T

            print('Internal nodes with 0 potential')
            print(list_internal_force_zero)

            A=A.tocsr() #convert to csr format

            #Remove trivial equtions 
            diagonal = A.diagonal()
            N_full = len(diagonal)
            indices_non_id = np.where(diagonal!=1.)[0]
            N_sel = len(indices_non_id)

            Msel = scsp.lil_matrix((N_full, N_sel))
            for ii, ind in enumerate(indices_non_id):
                Msel[ind, ii] =1.

            Msel = Msel.tocsc()

            Asel = Msel.T*A*Msel
            Asel=Asel.tocsc()
            # print('Asel shape:', Asel.shape)


            self.xn = xn
            self.yn = yn

            self.flag_inside_n = flag_inside_n
            self.flag_outside_n = flag_outside_n
            self.flag_outside_n_mat = flag_outside_n_mat
            self.flag_force_zero = flag_force_zero
            self.Asel = Asel

            self.Dx = Dx.tocsc()

            self.Dy = Dy.tocsc()

            self.sparse_solver = sparse_solver

            self.U_sc_eV_stp=0.;


            self.Msel = Msel.tocsc()
            self.Msel_T = (Msel.T).tocsc()

            #initialize self.luobj
            self.build_sparse_solver()


            self.tol_der = tol_der
            self.tol_stem = tol_stem

            print('Done PIC init.')

        else:

            self.solve = self._solve_for_states
            self.sparse_solver = None
            self.tol_stem = None
            self.tol_der = None

        self.flag_inside_n_mat = np.logical_not(flag_outside_n_mat)
        self.chamb = chamb
        self.rho = np.zeros((self.Nxg,self.Nyg));
        self.phi = np.zeros((self.Nxg,self.Nyg));
        self.efx = np.zeros((self.Nxg,self.Nyg));
        self.efy = np.zeros((self.Nxg,self.Nyg));

    #@profile    
    def solve(self, rho = None, flag_verbose = False):

        if rho is None:
            rho = self.rho
        self._solve_core(self, rho)

    @profile
    def gather(self, x_mp, y_mp):

        if not (len(x_mp)==len(y_mp)):
            raise ValueError('x_mp, y_mp should have the same length!!!')

        if len(x_mp)>0:
            ## compute beam electric field
            nar = lambda x: cp.asnumpy(x)
            car = lambda x: cp.asarray(x)

            x_mp_gpu = car(x_mp)
            y_mp_gpu = car(y_mp)
            efx = car(self.efx)
            efy = car(self.efy)
            Exn = cp.zeros_like(x_mp_gpu)
            Eyn = cp.zeros_like(x_mp_gpu)
            inside_mat_GPU = car(self.flag_inside_n_mat)
            inside_mat_GPU = inside_mat_GPU.astype(cp.uint8, copy=False)

            Ex_sc_n, Ey_sc_n = iffb.int_field_border(x_mp,y_mp,self.bias_x,self.bias_y,self.Dh,
                                         self.Dh, self.efx, self.efy, self.flag_inside_n_mat)
            # cp.cuda.get_current_stream().synchronize()
            Ex_sc_n_gpu, Ey_sc_n_gpu = int_field_border_cu(x_mp_gpu,y_mp_gpu,self.bias_x,self.bias_y,self.dx,
                                         self.dy, efx, efy, inside_mat_GPU, Ex_n=Exn, Ey_n=Eyn)
            # cp.cuda.get_current_stream().synchronize()
            np.testing.assert_allclose(Ex_sc_n,nar(Ex_sc_n_gpu),atol=1e-7,rtol = 1e-4)
            np.testing.assert_allclose(Ey_sc_n,nar(Ey_sc_n_gpu),atol=1e-7,rtol = 1e-4)
        else:
            Ex_sc_n=0.
            Ey_sc_n=0.

        return Ex_sc_n, Ey_sc_n

    def build_sparse_solver(self):

        if self.sparse_solver == 'scipy_slu':
            print("Using scipy superlu solver...")
            luobj = ssl.splu(self.Asel.tocsc())
        elif self.sparse_solver == 'PyKLU':
            print("Using klu solver...")
            try:
                import PyKLU.klu as klu
                luobj = klu.Klu(self.Asel.tocsc())
            except Exception as e:
                print("Got exception: ", e)
                print("Falling back on scipy superlu solver:")
                luobj = ssl.splu(self.Asel.tocsc())
        else:
            raise ValueError('Solver not recognized!!!!\nsparse_solver must be "scipy_slu" or "PyKLU"\n')

        self.luobj = luobj

    def get_state_object(self):
        state = FiniteDifferences_ShortleyWeller_SquareGrid(chamb=self.chamb, Dh=self.Dh,
                    sparse_solver = self.sparse_solver, tol_stem = self.tol_stem, tol_der = self.tol_der,
                    include_solver=False)

        state.rho = self.rho.copy()
        state.phi = self.phi.copy()
        state.efx = self.efx.copy()
        state.efy = self.efy.copy()

        return state


    def solve_states(self, states):

        states = np.atleast_1d(states)
        for ii in range(len(states)):
            state = states[ii]
            self._solve_core(state, state.rho)


    def _solve_core(self, state, rho):

        b=-rho.flatten()/eps0;
        b[(self.flag_force_zero)]=0;
        b_sel = self.Msel_T*b
        phi_sel = self.luobj.solve(b_sel)
        phi = self.Msel*phi_sel

        efx = self.Dx*phi
        efy = self.Dy*phi
        phi=np.reshape(phi,(self.Nxg,self.Nyg))
        efx=np.reshape(efx,(self.Nxg,self.Nyg))
        efy=np.reshape(efy,(self.Nxg,self.Nyg))
        state.efx = efx
        state.efy = efy
        state.phi = phi


