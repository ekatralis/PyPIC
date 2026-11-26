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
import scipy.sparse.linalg as ssl
from .PyPIC_Scatter_Gather import PyPIC_Scatter_Gather
from scipy.constants import e, epsilon_0
import cupy as cp
from cupyx.scipy.sparse import csc_matrix, csr_matrix
from cupyx.scipy.sparse.linalg import splu
from line_profiler import profile
from .luLU import luLU
from .cuDSSLU import SpMDVSolver
from tqdm import tqdm

na = lambda x:np.array([x])

qe = cp.array(e)
eps0 = cp.array(epsilon_0)

cuda_src = r'''
extern "C" __global__
void int_field(const long N_mp,
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
               double* __restrict__ Ex_n,
               double* __restrict__ Ey_n)
{
    long p = blockDim.x * blockIdx.x + threadIdx.x;
    if (p >= N_mp) return;

    double fi = 1.0 + (xn[p] - bias_x) / dx;
    double fj = 1.0 + (yn[p] - bias_y) / dy;

    // Match Fortran INT(): truncate toward zero
    int i = (int)fi;
    int j = (int)fj;

    double hx = fi - (double)i;
    double hy = fj - (double)j;

    double Ex = 0.0, Ey = 0.0;
    if (i > 0 && j > 0 && i < Nxg && j < Nyg) {
        int i0 = i - 1, j0 = j - 1;

        long idx00 = (long)i0 * stride_i + (long)j0 * stride_j;
        long idx10 = (long)(i0+1) * stride_i + (long)j0 * stride_j;
        long idx01 = (long)i0 * stride_i + (long)(j0+1) * stride_j;
        long idx11 = (long)(i0+1) * stride_i + (long)(j0+1) * stride_j;

        double w00 = (1.0 - hx) * (1.0 - hy);
        double w10 = hx * (1.0 - hy);
        double w01 = (1.0 - hx) * hy;
        double w11 = hx * hy;

        Ex = efx[idx00]*w00 + efx[idx10]*w10 + efx[idx01]*w01 + efx[idx11]*w11;
        Ey = efy[idx00]*w00 + efy[idx10]*w10 + efy[idx01]*w01 + efy[idx11]*w11;
    }
    Ex_n[p] = Ex;
    Ey_n[p] = Ey;
}
''';

mod = cp.RawModule(
    code=cuda_src,
    options=('--std=c++11', '--use_fast_math'),
    backend='nvrtc')
int_field_kernel = mod.get_function('int_field')

def _strides_in_elements(arr2d):
    return (arr2d.strides[0] // arr2d.itemsize,
            arr2d.strides[1] // arr2d.itemsize)

@profile
def int_field_cu(xn, yn, bias_x, bias_y, dx, dy, efx, efy, *, Ex_n=None, Ey_n=None, stream=None):
    """
    CuPy wrapper for the int_field kernel.

    Parameters
    ----------
    xn, yn : (N_mp,) cupy.ndarray, float64
    bias_x, bias_y, dx, dy : float (or float64)
    efx, efy : (Nxg, Nyg) cupy.ndarray, float64      # matches Fortran shapes
    stream : cp.cuda.Stream or None

    Returns
    -------
    Ex_n, Ey_n : (N_mp,) cupy.ndarray, float64
    """
    # Type/shape checks (lightweight)
    # assert xn.dtype == yn.dtype == cp.float64
    # assert efx.dtype == efy.dtype == cp.float64
    # assert efx.shape == efy.shape and efx.ndim == 2

    Nxg, Nyg = map(int, efx.shape)  # shape is (Nxg, Nyg) to mirror Fortran
    N_mp = int(xn.size)

    # Ex_n = cp.zeros_like(xn)
    # Ey_n = cp.zeros_like(xn)

    stride_i, stride_j = _strides_in_elements(efx)  # supports C- or F-order

    threads = 512
    blocks = (N_mp + threads - 1) // threads

    args = (
        N_mp,
        xn, yn,
        float(bias_x), float(bias_y),
        float(dx), float(dy),
        efx, efy,
        cp.int32(Nxg), cp.int32(Nyg),
        cp.int64(stride_i), cp.int64(stride_j),
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
mod = cp.RawModule(
    code=kernel_code_f,
    options=('--std=c++14', '--use_fast_math'),
    backend='nvrtc')
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

class FiniteDifferences_Staircase_SquareGrid(PyPIC_Scatter_Gather):
    #@profile
    def __init__(self, chamb, Dh, sparse_solver = 'cupy_splu', remove_external_nodes_from_mat=True, include_solver = True):
        
        print('Start PIC init.:')
        print('Finite Differences, Square Grid')


        self.Dh = Dh
        if hasattr(chamb, 'x_min') and hasattr(chamb, 'x_max') and hasattr(chamb, 'y_min') and hasattr(chamb, 'y_max'):
            super(FiniteDifferences_Staircase_SquareGrid, self).__init__(dx = self.Dh, dy = self.Dh, 
                x_min = chamb.x_min, x_max = chamb.x_max, y_min = chamb.y_min, y_max = chamb.y_max)
        else:
            super(FiniteDifferences_Staircase_SquareGrid, self).__init__(chamb.x_aper, chamb.y_aper, self.Dh, self.Dh)

        Nyg, Nxg = self.Nyg, self.Nxg
        
        
        [xn, yn]=np.meshgrid(self.xg,self.yg)

        xn=xn.T
        xn=xn.flatten()

        yn=yn.T
        yn=yn.flatten()
        #% xn and yn are stored such that the external index is on x 

        if hasattr(chamb, 'use_gpu') and chamb.use_gpu:
            xn_gpu = cp.asarray(xn)
            yn_gpu = cp.asarray(yn)
            flag_outside_n = cp.asnumpy(chamb.is_outside(xn_gpu, yn_gpu))
        else:
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

            list_internal_force_zero = []

            # Build A matrix
            for u in tqdm(range(0,Nxg*Nyg)):
                if flag_inside_n[u]:
                    A[u,u] = -(4./(Dh*Dh))
                    A[u,u-1]=1./(Dh*Dh);     #phi(i-1,j)nx
                    A[u,u+1]=1./(Dh*Dh);     #phi(i+1,j)
                    A[u,u-Nyg]=1./(Dh*Dh);    #phi(i,j-1)
                    A[u,u+Nyg]=1./(Dh*Dh);    #phi(i,j+1)
                else:
                    # external nodes
                    A[u,u]=1.
                    
            A=A.tocsr() #convert to csr format
            
            #Remove trivial equtions 
            if remove_external_nodes_from_mat:
                diagonal = A.diagonal()
                N_full = len(diagonal)
                indices_non_id = np.where(diagonal!=1.)[0]
                N_sel = len(indices_non_id)
                Msel = scsp.lil_matrix((N_full, N_sel))
                for ii, ind in enumerate(indices_non_id):
                    Msel[ind, ii] =1.
            else:
                diagonal = A.diagonal()
                N_full = len(diagonal)
                Msel = scsp.lil_matrix((N_full, N_full))
                for ii in range(N_full):
                    Msel[ii, ii] =1.
            Msel = Msel.tocsc()
            Asel = Msel.T*A*Msel
            Asel = csc_matrix(Asel)
            # print('Asel shape:', Asel.shape)

            self.sparse_solver = sparse_solver
                
            self.xn = xn
            self.yn = yn
            
            self.flag_inside_n = cp.array(flag_inside_n)
            self.flag_outside_n = cp.array(flag_outside_n)
            self.flag_outside_n_mat = cp.array(flag_outside_n_mat)
            self.flag_inside_n_mat = cp.logical_not(self.flag_outside_n_mat)
            self.flag_border_mat = cp.array(flag_border_mat)
            self.Asel = Asel
            self.U_sc_eV_stp=0.;
            self.Msel = csc_matrix(Msel)
            self.Msel_T = csc_matrix(Msel.T)
            self.build_sparse_solver()
            self.x_border = cp.array(xn[flag_border_n])
            self.y_border = cp.array(yn[flag_border_n])
            self.flag_border_n = cp.array(flag_border_n)
            
            print('Done PIC init.')
            
        else:
            self.solve = self._solve_for_states          
        # print(self.Asel.shape)
        self.unitarea = cp.array(self.dx*self.dy)
        self.Dhcp = cp.array(Dh)
        self.rho = cp.zeros((self.Nxg,self.Nyg));
        self.phi = cp.zeros((self.Nxg,self.Nyg));
        self.efx = cp.zeros((self.Nxg,self.Nyg));
        self.efy = cp.zeros((self.Nxg,self.Nyg));
        self.chamb = chamb
        
    def build_sparse_solver(self):

        if self.sparse_solver == 'cupy_splu':
            print("[Solver INIT]: Using CuPy splu solver")
            luobj = splu(self.Asel, permc_spec="MMD_AT_PLUS_A") #,diag_pivot_thresh=1.0
        elif self.sparse_solver == 'luLU':
            print("[Solver INIT]: Using luLU solver")
            luobj = luLU(self.Asel, permc_spec="MMD_AT_PLUS_A")
        elif self.sparse_solver == 'cuDSS':
            print("[Solver INIT]: Using cuDSS solver")
            luobj = SpMDVSolver(self.Asel.tocsr())
        else:
            raise ValueError('Solver not recognized!!!!\nsparse_solver must be "cupy_splu", "cuDSS" or "luLU"\n')

        self.luobj = luobj
        
    #@profile    
    def solve(self, rho = None, flag_verbose = False, pic_external = None):

        if rho is None:
            rho = self.rho
            
        self._solve_core(self, rho, pic_external) #change 2

    def gather(self, x_mp, y_mp):
        
        if not (len(x_mp)==len(y_mp)):
            raise ValueError('x_mp, y_mp should have the same length!!!')

        if len(x_mp)>0:    
            ## compute beam electric field
            Ex_sc_n = cp.zeros_like(x_mp)
            Ey_sc_n = cp.zeros_like(x_mp)
            
            int_field_cu(x_mp,y_mp,self.bias_x,self.bias_y,self.dx,
                         self.dy, self.efx, self.efy, Ex_n=Ex_sc_n, Ey_n=Ey_sc_n)
        else:
            Ex_sc_n=cp.array(0.)
            Ey_sc_n=cp.array(0.)
            
        return Ex_sc_n, Ey_sc_n

    def gather_phi(self, x_mp, y_mp):
        
        if not (len(x_mp)==len(y_mp)):
            raise ValueError('x_mp, y_mp should have the same length!!!')

        if len(x_mp)>0:    
            ## compute beam potential
            phi_sc_n = cp.zeros_like(x_mp)
            phi_sc_n2 = cp.zeros_like(x_mp)
            int_field_cu(x_mp,y_mp,self.bias_x,self.bias_y,self.dx,
                         self.dy, self.phi, self.phi, Ex_n=phi_sc_n, Ey_n=phi_sc_n2)
                       
        else:
            phi_sc_n=cp.array(0.)
            
        return phi_sc_n
    
    def scatter(self, x_mp, y_mp, nel_mp, charge = -qe, flag_add=False):
        
        if not (len(x_mp)==len(y_mp)==len(nel_mp)):
            raise ValueError('x_mp, y_mp, nel_mp should have the same length!!!')
        
        if len(x_mp)>0:
            rho = cp.zeros((self.Nxg, self.Nyg), dtype=cp.float64)
            compute_rho_gpu_dropin(x_mp,y_mp,nel_mp,self.bias_x,self.bias_y,self.dx,self.dy,self.Nxg,self.Nyg, rho=rho)
        else:
            rho=self.rho*0.

        if flag_add:
            self.rho+=charge*rho/self.unitarea;
        else:
            self.rho=charge*rho/self.unitarea;
        
    def get_state_object(self):
        state = FiniteDifferences_Staircase_SquareGrid(chamb=self.chamb, Dh=self.Dh, include_solver=False)
        
        state.rho = self.rho.copy()
        state.phi = self.phi.copy()
        state.efx = self.efx.copy()
        state.efy = self.efy.copy()
        
        return state		
        
    def solve_states(self, states, pic_s_external = None):
        
        states = np.atleast_1d(states)
        if pic_s_external is None:
            pic_s_external = len(states)*[None]
        else:
            pic_s_external = np.atleast_1d(pic_s_external)
            
        if len(pic_s_external) != len(states):
            raise ValueError('Found len(pic_s_external) != len(states)!!!!')
        
        for ii in range(len(states)):
            state = states[ii]
            pic_external = pic_s_external[ii]
            self._solve_core(state, state.rho, pic_external)
                

    @profile            
    def _solve_core(self, state, rho, pic_external):

        b=-rho.flatten()/eps0;
        b[(self.flag_outside_n)]=0.; #boundary condition

        if pic_external is not None:
            phi_border = pic_external.gather_phi(self.x_border, self.y_border)
            b[self.flag_border_n] = phi_border

        b_sel = self.Msel_T@b
        phi_sel = self.luobj.solve(b_sel)
        phi = self.Msel@phi_sel
        phi=cp.reshape(phi,(self.Nxg,self.Nyg))

        efx = state.efx
        efy = state.efy

        efx[1:self.Nxg-1,:] = phi[0:self.Nxg-2,:] - phi[2:self.Nxg,:];  #central difference on internal nodes
        efy[:,1:self.Nyg-1] = phi[:,0:self.Nyg-2] - phi[:,2:self.Nyg];  #central difference on internal nodes

        efx[self.flag_border_mat]=efx[self.flag_border_mat]*2;
        efy[self.flag_border_mat]=efy[self.flag_border_mat]*2;

        state.efx = efx / (2*self.Dhcp);    #divide grid size
        state.efy = efy / (2*self.Dhcp);
        state.rho = rho
        state.phi = phi
        state.b = b
        

        
        






