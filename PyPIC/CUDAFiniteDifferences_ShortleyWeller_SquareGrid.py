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
from cupyx.scipy.linalg import lu_factor, lu_solve
from cupyx.scipy.sparse.linalg import cg, gmres, LinearOperator
try:
    from line_profiler import profile
except ImportError:
    def profile(func):
        return func
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable
from cupy import fuse
from cupyx.time import repeat
import xobjects as xo


na = lambda x:np.array([x])

qe = cp.array(e)
eps0 = cp.array(epsilon_0)

class FiniteDifferences_ShortleyWeller_SquareGrid(PyPIC_Scatter_Gather):
    @profile
    def __init__(self,chamb, Dh, sparse_solver = 'cupy_splu', tol_stem = 0.01, tol_der = 0.1, include_solver=True,
                 context=None):

        print('Start PIC init.:')
        print('Finite Differences, Shortley-Weller, Square Grid')
        print('Using Shortley-Weller boundary approx.')

        self.Dh = Dh
        super(FiniteDifferences_ShortleyWeller_SquareGrid, self).__init__(chamb.x_aper, chamb.y_aper, self.Dh, self.Dh,
                                                                          context=context or xo.ContextCupy())
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

            self.U_sc_eV_stp=0.;


            self.idx_sel = cp.asarray(Msel.indices)  # Msel selects one node per column
            # Boundary condition as precomputed factor, avoiding boolean indexing at each solve
            self.not_force_zero_factor = cp.asarray(~flag_force_zero, dtype=float)

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
        # print(self.Asel.shape)
        self.flag_inside_n_mat = cp.asarray(np.logical_not(flag_outside_n_mat), dtype=cp.int8)
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

        return self._int_field(x_mp, y_mp, self.efx, self.efy, self.flag_inside_n_mat)
    
    def build_sparse_solver(self):
        xo_solvers = {'cupy_splu': 'cupySLU', 'luLU': 'CachedSLU', 'cuDSS': 'cuDSS'}
        if self.sparse_solver not in xo_solvers:
            raise ValueError('sparse_solver must be one of %s' % list(xo_solvers))
        print("[Solver INIT]: Using %s solver" % self.sparse_solver)
        self.luobj = xo.sparse.factorized_sparse_solver(
            self.Asel.tocsr(), force_solver=xo_solvers[self.sparse_solver], context='cupy')

    def get_state_object(self):
        state = FiniteDifferences_ShortleyWeller_SquareGrid(chamb=self.chamb, Dh=self.Dh,
                    sparse_solver = self.sparse_solver, tol_stem = self.tol_stem, tol_der = self.tol_der,
                    include_solver=False, context=self.context)

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

        b = -rho.flatten() / eps0 * self.not_force_zero_factor
        phi = cp.zeros(b.size)
        phi[self.idx_sel] = self.luobj.solve(b[self.idx_sel])

        efx = self.Dx@phi
        efy = self.Dy@phi
        phi=cp.reshape(phi,(self.Nxg,self.Nyg))
        efx=cp.reshape(efx,(self.Nxg,self.Nyg))
        efy=cp.reshape(efy,(self.Nxg,self.Nyg))
        state.efx = efx
        state.efy = efy
        state.phi = phi
