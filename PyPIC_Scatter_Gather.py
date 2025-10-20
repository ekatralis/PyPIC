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

import cupy as cp
import numpy as np
from . import rhocompute as rhocom
from . import int_field_for as iff
#~ from abc import abstractmethod, ABCMeta
from line_profiler import profile

import cupy as cp

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

mod = cp.RawModule(code=cuda_src, options=('-std=c++11','-O3','--use_fast_math','--gpu-architecture=sm_70',), backend='nvcc')
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

    return Ex_n, Ey_n

kernel_code = r'''
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

__global__ void compute_sc_rho_kernel(
    const long long N_mp,
    const double* __restrict__ x_mp,
    const double* __restrict__ y_mp,
    const double* __restrict__ nel_mp,
    const double bias_x, const double bias_y,
    const double dx, const double dy,
    const int Nxg, const int Nyg,
    double* __restrict__ rho
){
    long long p = blockDim.x * (long long)blockIdx.x + threadIdx.x;
    if (p >= N_mp) return;

    // fractional indices
    double fi = (x_mp[p] - bias_x) / dx;
    int i = (int)floor(fi);
    double hx = fi - (double)i;

    double fj = (y_mp[p] - bias_y) / dy;
    int j = (int)floor(fj);
    double hy = fj - (double)j;

    double w = nel_mp[p];

    // require i in [0, Nxg-2], j in [0, Nyg-2]
    if (i >= 0 && j >= 0 && i < (Nxg - 1) && j < (Nyg - 1)) {
        // Row-major indexing: rho has shape (Nyg, Nxg) -> index = j*Nxg + i
        size_t idx00 = (size_t)j * (size_t)Nxg + (size_t)i;

        double w00 = w * (1.0 - hx) * (1.0 - hy);
        double w10 = w * (      hx) * (1.0 - hy);
        double w01 = w * (1.0 - hx) * (      hy);
        double w11 = w * (      hx) * (      hy);

        ATOMIC_ADD(&rho[idx00],           w00);
        ATOMIC_ADD(&rho[idx00 + 1],       w10);
        ATOMIC_ADD(&rho[idx00 + Nxg],     w01);
        ATOMIC_ADD(&rho[idx00 + Nxg + 1], w11);
    }
}

} // extern "C"
'''
mod = cp.RawModule(code=kernel_code, options=('-std=c++11','-O3','--use_fast_math','--gpu-architecture=sm_70',), backend='nvcc')
compute_sc_rho_kernel = mod.get_function('compute_sc_rho_kernel')

@profile
def compute_rho_gpu(x_mp,y_mp,nel_mp,bias_x,bias_y,dx,dy,Nxg,Nyg,rho = None):
    # Inputs (example shapes/dtypes)
    # x_mp, y_mp, nel_mp: (N_mp,) float64 (on GPU)
    # bias_x, bias_y, dx, dy: float64
    # Nxg, Nyg: int32
    # rho: (Nyg, Nxg) float64 (on GPU), zero-initialized

    N_mp = x_mp.size
    # rho = cp.zeros((Nyg, Nxg), dtype=cp.float64)

    threads = 256
    blocks = (int(N_mp) + threads - 1) // threads

    compute_sc_rho_kernel(
        (blocks,), (threads,),
        (
            cp.int64(N_mp),
            x_mp, y_mp, nel_mp,
            cp.float64(bias_x), cp.float64(bias_y),
            cp.float64(dx), cp.float64(dy),
            cp.int32(Nxg), cp.int32(Nyg),
            rho
        )
    )

    # rho[j, i] now matches your Fortran's rho(i,j) (bearing in mind C vs Fortran memory layout)
    return rho


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
    return rho


kernel_code_fast = r'''
extern "C" {

#if __CUDA_ARCH__ < 600 && defined(__CUDA_ARCH__)
__device__ double atomicAdd_double(double* address, double val) {
    unsigned long long int* a = (unsigned long long int*)address;
    unsigned long long int old = *a, assumed;
    do {
        assumed = old;
        old = atomicCAS(a, assumed,
                        __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}
#define ATOMIC_ADD(addr, v) atomicAdd_double((addr), (v))
#else
#define ATOMIC_ADD(addr, v) atomicAdd((addr), (v))
#endif

__global__ void compute_sc_rho_kernel_f_fast(
    const long long N_mp,
    const double* __restrict__ x_mp,
    const double* __restrict__ y_mp,
    const double* __restrict__ nel_mp,
    const double bias_x, const double bias_y,
    const double dx, const double dy,
    const int Nxg, const int Nyg,
    double* __restrict__ rho  // Fortran layout: shape (Nxg, Nyg), i-fastest
){
    long long p = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= N_mp) return;

    double x = x_mp[p];
    double y = y_mp[p];
    double w = nel_mp[p];
    if (!isfinite(x) || !isfinite(y) || !isfinite(w)) return;
    if (!(dx > 0.0) || !(dy > 0.0)) return;

    // Fortran INT (truncate toward 0) with 1-based indices
    double fi = 1.0 + (x - bias_x) / dx;
    int i = (int)(fi);
    double hx = fi - (double)i;

    double fj = 1.0 + (y - bias_y) / dy;
    int j = (int)(fj);
    double hy = fj - (double)j;

    if (i > 0 && j > 0 && i < Nxg && j < Nyg) {
        int i0 = i - 1;
        int j0 = j - 1;

        long long idx00 = (long long)i0 + (long long)j0 * (long long)Nxg;

        double w00 = w * (1.0 - hx) * (1.0 - hy);
        double w10 = w * (      hx) * (1.0 - hy);
        double w01 = w * (1.0 - hx) * (      hy);
        double w11 = w * (      hx) * (      hy);

        ATOMIC_ADD(&rho[idx00],               w00);
        ATOMIC_ADD(&rho[idx00 + 1],           w10);
        ATOMIC_ADD(&rho[idx00 + Nxg],         w01);
        ATOMIC_ADD(&rho[idx00 + Nxg + 1],     w11);
    }
}
} // extern "C"
'''
mod_fast = cp.RawModule(code=kernel_code_fast,
                        options=('-std=c++14',),
                        name_expressions=['compute_sc_rho_kernel_f_fast'])
compute_sc_rho_kernel_f_fast = mod_fast.get_function('compute_sc_rho_kernel_f_fast')

def compute_rho_gpu_dropin_fast(
    x_mp, y_mp, nel_mp,
    bias_x, bias_y, dx, dy,
    Nxg, Nyg,
    rho=None
):
    # x_mp = cp.asarray(x_mp, dtype=cp.float64)
    # y_mp = cp.asarray(y_mp, dtype=cp.float64)
    # nel_mp = cp.asarray(nel_mp, dtype=cp.float64)

    # if rho is None:
    #     rho = cp.zeros((Nxg, Nyg), dtype=cp.float64, order='F')
    # else:
    #     assert rho.shape == (Nxg, Nyg) and rho.dtype == cp.float64
    #     rho.fill(0.0)

    N_mp = int(x_mp.size)
    threads = 256
    blocks = (N_mp + threads - 1) // threads

    compute_sc_rho_kernel_f_fast(
        (blocks,), (threads,),
        (
            cp.int64(N_mp),
            x_mp, y_mp, nel_mp,
            float(bias_x), float(bias_y),
            float(dx), float(dy),
            cp.int32(Nxg), cp.int32(Nyg),
            rho
        )
    )
    return rho


na = lambda x:np.array([x])

qe=1.602176565e-19;
eps0=8.8541878176e-12;

class PyPIC_Scatter_Gather(object):
    #__metadata__ = ABCMeta

    def __init__(self, x_aper=None, y_aper=None, dx=None, dy=None, xg=None, yg=None, 
                x_min=None, x_max=None, y_min=None, y_max=None, *args, **kwargs):

        print('PyPIC Version 2.4.5')
        
        if xg is not None and yg is not None:
            assert(x_aper is None and y_aper is None and dx is None and dy is None)
            assert(x_min is None and x_max is None and y_min is None and y_max is None)

            Nxg=len(xg);
            bias_x=min(xg);

            Nyg=len(yg);
            bias_y=min(yg);
            
            dx = xg[1]-xg[0]
            dy = yg[1]-yg[0]

        elif dx is not None and dy is not None:
            assert(xg is None and yg is None)
            # box given
            if x_min is not None and x_max is not None and y_min is not None and y_max is not None:
                assert(x_aper is None and y_aper is None)

                x_aper = (x_max-x_min)/2.
                x_center = (x_max+x_min)/2.

                y_aper = (y_max-y_min)/2.
                y_center = (y_max+y_min)/2.
            # aperture given
            elif x_aper is not None and y_aper is not None:
                assert(x_min is None and x_max is None and y_min is None and y_max is None)

                x_center = 0.
                y_center = 0.

            else:
                raise ValueError('x_aper and y_aper, or x_min, x_max and y_min, y_max must be specified!!!')

            xg=np.arange(0, x_aper+5.*dx,dx,float)  
            xgr=xg[1:]
            xgr=xgr[::-1]#reverse array
            xg=np.concatenate((-xgr,xg),0)
            xg = xg + x_center
            Nxg=len(xg);
            bias_x=min(xg);

            yg=np.arange(0,y_aper+4.*dy,dy,float)  
            ygr=yg[1:]
            ygr=ygr[::-1]#reverse array
            yg=np.concatenate((-ygr,yg),0)
            yg = yg + y_center
            Nyg=len(yg);
            bias_y=min(yg);	

        else:
            raise ValueError('dx and dy, or xg and yg must be specified!!!')


        self.dx = dx
        self.xg = xg
        self.Nxg = Nxg
        self.bias_x = bias_x
        self.dy = dy
        self.yg = yg
        self.Nyg = Nyg
        self.bias_y = bias_y

                        
    @profile
    def scatter(self, x_mp, y_mp, nel_mp, charge = -qe, flag_add=False):
        
        if not (len(x_mp)==len(y_mp)==len(nel_mp)):
            raise ValueError('x_mp, y_mp, nel_mp should have the same length!!!')
        
        if len(x_mp)>0:
            nar = lambda x: cp.asnumpy(x)
            car = lambda x: cp.asarray(x)
            x_mp_gpu = car(x_mp)
            y_mp_gpu = car(y_mp)
            nel_mp_gpu = car(nel_mp)
            rho_gpu = cp.zeros((self.Nxg, self.Nyg), dtype=cp.float64)
            rho=rhocom.compute_sc_rho(x_mp,y_mp,nel_mp,self.bias_x,self.bias_y,self.dx,self.dy,self.Nxg,self.Nyg)

            rho_gpuT = cp.zeros((self.Nyg, self.Nxg), dtype=cp.float64)
            rho_gpuT = compute_rho_gpu(x_mp_gpu,y_mp_gpu,nel_mp_gpu,self.bias_x,self.bias_y,self.dx,self.dy,self.Nxg,self.Nyg, rho=rho_gpuT)
            rho_gpu = compute_rho_gpu_dropin(x_mp_gpu,y_mp_gpu,nel_mp_gpu,self.bias_x,self.bias_y,self.dx,self.dy,self.Nxg,self.Nyg, rho=rho_gpu)
            # rho_gpuF = cp.zeros((self.Nxg, self.Nyg), dtype=cp.float64)
            # rho_gpuF = compute_rho_gpu_dropin_fast(x_mp_gpu,y_mp_gpu,nel_mp_gpu,self.bias_x,self.bias_y,self.dx,self.dy,self.Nxg,self.Nyg, rho=rho_gpuF)

            np.testing.assert_allclose(rho,nar(rho_gpu),atol=1e-7,rtol = 1e-4)
            # np.testing.assert_allclose(rho,nar(rho_gpuF),atol=1e-7,rtol = 1e-4)
            np.testing.assert_allclose(rho.T,nar(rho_gpuT),atol=1e-7,rtol = 1e-4)
        else:
            rho=self.rho*0.

        if flag_add:
            self.rho+=charge*rho/(self.dx*self.dy);
        else:
            self.rho=charge*rho/(self.dx*self.dy);

         
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
            Ex_sc_n, Ey_sc_n = iff.int_field(x_mp,y_mp,self.bias_x,self.bias_y,self.dx,
                                         self.dy, self.efx, self.efy)
            
            Ex_sc_n_gpu, Ey_sc_n_gpu = int_field_cu(x_mp_gpu,y_mp_gpu,self.bias_x,self.bias_y,self.dx,
                                         self.dy, efx, efy, Ex_n=Exn, Ey_n=Eyn)
            np.testing.assert_allclose(Ex_sc_n,nar(Ex_sc_n_gpu),atol=1e-7,rtol = 1e-4)
            np.testing.assert_allclose(Ey_sc_n,nar(Ey_sc_n_gpu),atol=1e-7,rtol = 1e-4)
            
            cp._default_memory_pool.free_all_blocks()
        else:
            Ex_sc_n=0.
            Ey_sc_n=0.
            
        return Ex_sc_n, Ey_sc_n
        
    def gather_phi(self, x_mp, y_mp):
        
        if not (len(x_mp)==len(y_mp)):
            raise ValueError('x_mp, y_mp should have the same length!!!')

        if len(x_mp)>0:    
            ## compute beam potential
            phi_sc_n, _ = iff.int_field(x_mp,y_mp,self.bias_x,self.bias_y,self.dx,
                                         self.dy, self.phi, self.phi)
                       
        else:
            phi_sc_n=0.
            
        return phi_sc_n
        
    def gather_rho(self, x_mp, y_mp):
        
        if not (len(x_mp)==len(y_mp)):
            raise ValueError('x_mp, y_mp should have the same length!!!')

        if len(x_mp)>0:    
            ## compute beam distribution
            rho_sc_n, _ = iff.int_field(x_mp,y_mp,self.bias_x,self.bias_y,self.dx,
                                         self.dy, self.rho, self.rho)
                       
        else:
            rho_sc_n=0.
            
        return rho_sc_n

    #@abstractmethod
    def solve(self, *args, **kwargs):
        '''Computes the electric field maps from the stored 
        charge distribution (self.rho) and stores them in
        self.efx, self.efy.'''
        pass
        
    #@profile
    def scatter_and_solve(self, x_mp, y_mp, nel_mp, charge = -qe, flag_add=False):
        self.scatter(x_mp, y_mp, nel_mp, charge, flag_add)
        self.solve()


    def _solve_for_states(self,*args, **kwargs):
        raise ValueError('I am a state, I cannot solve!')
