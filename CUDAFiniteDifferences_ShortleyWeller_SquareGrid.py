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
# from scipy.sparse.linalg import spsolve
# import scipy.sparse.linalg as ssl
from .PyPIC_Scatter_Gather import PyPIC_Scatter_Gather
from scipy.constants import e, epsilon_0
import cupy as cp
from cupyx.scipy.sparse import csc_matrix, csr_matrix
from cupyx.scipy.sparse.linalg import splu, spsolve
from cupyx.scipy.linalg import lu_factor, lu_solve
from cupyx.scipy.sparse.linalg import cg, gmres, LinearOperator
from line_profiler import profile
from tqdm import tqdm
from cupy import fuse
from cupyx.time import repeat
from .luLU import luLU
from .cuDSSLU import SpMDVSolver
from . import rhocompute as rhocom
# from . import int_field_for_border as iffb


na = lambda x:np.array([x])

qe = cp.array(e)
eps0 = cp.array(epsilon_0)

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

    # return Ex_n, Ey_n

kernel_code_f = r'''
extern "C" {

#if __CUDA_ARCH__ < 600 && defined(__CUDA_ARCH__)
__device__ double atomicAdd_double(double* address, double val) {
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}
#define ATOMIC_ADD(addr, v) atomicAdd_double((addr), (v))
#else
#define ATOMIC_ADD(addr, v) atomicAdd((addr), (v))
#endif

// Drop-in: matches Fortran INT/1-based logic and (i,j) indexing
__global__ void compute_sc_rho_kernel_f(
    const long long N_mp,
    const double* __restrict__ x_mp,
    const double* __restrict__ y_mp,
    const double* __restrict__ nel_mp,
    const double bias_x, const double bias_y,
    const double dx, const double dy,
    const int Nxg, const int Nyg,
    double* __restrict__ rho,
    const long long sx,  // stride (elements) for dim-0 (i)
    const long long sy   // stride (elements) for dim-1 (j)
){
    long long p = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= N_mp) return;

    double x = x_mp[p];
    double y = y_mp[p];
    double w = nel_mp[p];

    if (!isfinite(x) || !isfinite(y) || !isfinite(w)) return;
    if (!(dx > 0.0) || !(dy > 0.0)) return;

    // Fortran-style: 1-based indices via INT (truncate toward zero)
    double fi = 1.0 + (x - bias_x) / dx;
    int i = (int)(fi);         // INT in Fortran (trunc toward zero)
    double hx = fi - (double)i;

    double fj = 1.0 + (y - bias_y) / dy;
    int j = (int)(fj);
    double hy = fj - (double)j;

    // Fortran guard: if (i>0 .and. j>0 .and. i<Nxg .and. j<Nyg)
    if (i > 0 && j > 0 && i < Nxg && j < Nyg) {
        // Convert to 0-based offsets
        long long i0 = (long long)(i - 1);
        long long j0 = (long long)(j - 1);

        long long idx00 = i0 * sx + j0 * sy;

        double w00 = w * (1.0 - hx) * (1.0 - hy);
        double w10 = w * (      hx) * (1.0 - hy);
        double w01 = w * (1.0 - hx) * (      hy);
        double w11 = w * (      hx) * (      hy);

        ATOMIC_ADD(&rho[idx00             ], w00);
        ATOMIC_ADD(&rho[idx00 + sx        ], w10);
        ATOMIC_ADD(&rho[idx00 + sy        ], w01);
        ATOMIC_ADD(&rho[idx00 + sx + sy   ], w11);
    }
}
} // extern "C"
'''
mod = cp.RawModule(code=kernel_code_f,
                   options=('-std=c++14','-O3','--use_fast_math','--gpu-architecture=sm_70',), backend='nvcc')
compute_sc_rho_kernel_f = mod.get_function('compute_sc_rho_kernel_f')

@profile
def compute_rho_gpu_dropin(
    x_mp, y_mp, nel_mp,
    bias_x, bias_y, dx, dy,
    Nxg, Nyg,
    rho=None
):
    # Device arrays, float64
    # x_mp = cp.asarray(x_mp, dtype=cp.float64)
    # y_mp = cp.asarray(y_mp, dtype=cp.float64)
    # nel_mp = cp.asarray(nel_mp, dtype=cp.float64)

    # rho shaped (Nxg, Nyg) like Fortran; default Fortran-order
    # if rho is None:
    #     rho = cp.zeros((Nxg, Nyg), dtype=cp.float64, order='F')
    # else:
    #     assert isinstance(rho, cp.ndarray)
    #     assert rho.dtype == cp.float64
    #     assert rho.shape == (Nxg, Nyg)
    #     rho.fill(0.0)

    # Strides in ELEMENTS (not bytes)
    sx = rho.strides[0] // rho.itemsize
    sy = rho.strides[1] // rho.itemsize

    N_mp = int(x_mp.size)
    threads = 512
    blocks = (N_mp + threads - 1) // threads

    compute_sc_rho_kernel_f(
        (blocks,), (threads,),
        (
            cp.int64(N_mp),
            x_mp, y_mp, nel_mp,
            float(bias_x), float(bias_y),
            float(dx), float(dy),
            cp.int32(Nxg), cp.int32(Nyg),
            rho,
            cp.int64(sx), cp.int64(sy)
        )
    )
    # return rho

class FiniteDifferences_ShortleyWeller_SquareGrid(PyPIC_Scatter_Gather):
    @profile
    def __init__(self,chamb, Dh, sparse_solver = 'cupy_splu', tol_stem = 0.01, tol_der = 0.1, include_solver=True):

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
            for u in tqdm(range(0,Nxg*Nyg), desc="Mat Assembly"):
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


            self.xn = xn
            self.yn = yn

            self.flag_inside_n = cp.array(flag_inside_n)
            self.flag_outside_n = cp.array(flag_outside_n)
            self.flag_outside_n_mat = cp.array(flag_outside_n_mat)
            self.flag_force_zero = cp.array(flag_force_zero)
            self.Asel = csc_matrix(Asel)
            self.Acsr = csr_matrix(Asel)

            self.Dx = csc_matrix(Dx)

            self.Dy = csc_matrix(Dy)

            self.sparse_solver = sparse_solver

            if self.sparse_solver == 'cupy_custom': 
                self.A = csc_matrix(A)

            self.U_sc_eV_stp=0.;


            self.Msel = csc_matrix(Msel)
            self.Msel_T = csc_matrix(Msel.T)

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

        self.flag_inside_n_mat = cp.logical_not(self.flag_outside_n_mat).astype(cp.uint8, copy=False)
        self.chamb = chamb
        self.rho = cp.zeros((self.Nxg,self.Nyg));
        self.phi = cp.zeros((self.Nxg,self.Nyg));
        self.efx = cp.zeros((self.Nxg,self.Nyg));
        self.efy = cp.zeros((self.Nxg,self.Nyg));

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
            Ex_sc_n = cp.empty_like(x_mp)
            Ey_sc_n = cp.empty_like(x_mp)
            
            int_field_border_cu(x_mp,y_mp,self.bias_x,self.bias_y,self.dx,
                                self.dy, self.efx, self.efy, self.flag_inside_n_mat, Ex_n=Ex_sc_n, Ey_n=Ey_sc_n)
        else:
            Ex_sc_n=0.
            Ey_sc_n=0.

        return Ex_sc_n, Ey_sc_n
    
    @profile
    def scatter(self, x_mp, y_mp, nel_mp, charge = -qe, flag_add=False):
        
        if not (len(x_mp)==len(y_mp)==len(nel_mp)):
            raise ValueError('x_mp, y_mp, nel_mp should have the same length!!!')
        
        if len(x_mp)>0:
            rho = cp.zeros((self.Nxg, self.Nyg), dtype=cp.float64)
            compute_rho_gpu_dropin(x_mp,y_mp,nel_mp,self.bias_x,self.bias_y,self.dx,self.dy,self.Nxg,self.Nyg, rho=rho)
        else:
            rho=self.rho*0.

        denom = cp.array(self.dx*self.dy)
        if flag_add:
            self.rho+=charge*rho/denom;
        else:
            self.rho=charge*rho/denom;

    def build_sparse_solver(self):

        if self.sparse_solver == 'cupy_splu':
            print("[Solver INIT]: Using CuPy splu solver")
            luobj = splu(self.Asel, permc_spec="MMD_AT_PLUS_A") #,diag_pivot_thresh=1.0
        elif self.sparse_solver == 'cupy_custom':
            print("[Solver INIT]: Using CuPy no selection")
            # luobj = None
            luobj = splu(self.A, permc_spec="MMD_AT_PLUS_A")
            self._solve_core = self._solve_core_iter
        elif self.sparse_solver == 'luLU':
            print("[Solver INIT]: Using luLU solver")
            luobj = luLU(self.Asel, permc_spec="MMD_AT_PLUS_A")
        elif self.sparse_solver == 'cuDSS':
            print("[Solver INIT]: Using cuDSS solver")
            luobj = SpMDVSolver(self.Asel.tocsr())
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

    @profile
    def _solve_core(self, state, rho):

        b=-rho.flatten()/eps0;
        b[(self.flag_force_zero)]=0;
        b_sel = self.Msel_T@b
        phi_sel = self.luobj.solve(b_sel)
        phi = self.Msel@phi_sel

        efx = self.Dx@phi
        efy = self.Dy@phi
        phi=cp.reshape(phi,(self.Nxg,self.Nyg))
        efx=cp.reshape(efx,(self.Nxg,self.Nyg))
        efy=cp.reshape(efy,(self.Nxg,self.Nyg))
        state.efx = efx
        state.efy = efy
        state.phi = phi
    
    @profile
    def _solve_core_iter(self, state, rho):
        
        b=-rho.flatten()/eps0;
        b[(self.flag_force_zero)]=0;
        phi = self.luobj.solve(b)

        efx = self.Dx@phi
        efy = self.Dy@phi
        phi=cp.reshape(phi,(self.Nxg,self.Nyg))
        efx=cp.reshape(efx,(self.Nxg,self.Nyg))
        efy=cp.reshape(efy,(self.Nxg,self.Nyg))
        state.efx = efx
        state.efy = efy
        state.phi = phi


