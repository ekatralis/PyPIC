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
try:
    from line_profiler import profile
except ImportError:
    def profile(func):
        return func
import xobjects as xo
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

na = lambda x:np.array([x])

qe = cp.array(e)
eps0 = cp.array(epsilon_0)

class FiniteDifferences_Staircase_SquareGrid(PyPIC_Scatter_Gather):
    #@profile
    def __init__(self, chamb, Dh, sparse_solver = 'cupy_splu', remove_external_nodes_from_mat=True, include_solver = True,
                 context=None):
        
        print('Start PIC init.:')
        print('Finite Differences, Square Grid')


        self.Dh = Dh
        context = context or xo.ContextCupy()
        if hasattr(chamb, 'x_min') and hasattr(chamb, 'x_max') and hasattr(chamb, 'y_min') and hasattr(chamb, 'y_max'):
            super(FiniteDifferences_Staircase_SquareGrid, self).__init__(dx = self.Dh, dy = self.Dh, 
                x_min = chamb.x_min, x_max = chamb.x_max, y_min = chamb.y_min, y_max = chamb.y_max, context=context)
        else:
            super(FiniteDifferences_Staircase_SquareGrid, self).__init__(chamb.x_aper, chamb.y_aper, self.Dh, self.Dh,
                                                                         context=context)

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
            self.idx_sel = cp.asarray(Msel.indices)  # Msel selects one node per column
            self.build_sparse_solver()
            self.x_border = cp.array(xn[flag_border_n])
            self.y_border = cp.array(yn[flag_border_n])
            self.flag_border_n = cp.array(flag_border_n)
            # Boundary conditions as precomputed factors/indices, avoiding
            # boolean indexing (a synchronization point) at each solve
            self.inside_factor_n = cp.asarray(~flag_outside_n, dtype=float)
            self.idx_border_n = cp.asarray(np.flatnonzero(flag_border_n))
            self.border_factor_mat = cp.asarray(np.where(flag_border_mat, 2., 1.))
            
            print('Done PIC init.')
            
        else:
            self.solve = self._solve_for_states          
        # print(self.Asel.shape)
        self.Dhcp = cp.array(Dh)
        self.rho = cp.zeros((self.Nxg,self.Nyg));
        self.phi = cp.zeros((self.Nxg,self.Nyg));
        self.efx = cp.zeros((self.Nxg,self.Nyg));
        self.efy = cp.zeros((self.Nxg,self.Nyg));
        self.chamb = chamb
        
    def build_sparse_solver(self):
        xo_solvers = {'cupy_splu': 'cupySLU', 'luLU': 'CachedSLU', 'cuDSS': 'cuDSS'}
        if self.sparse_solver not in xo_solvers:
            raise ValueError('sparse_solver must be one of %s' % list(xo_solvers))
        print("[Solver INIT]: Using %s solver" % self.sparse_solver)
        self.luobj = xo.sparse.factorized_sparse_solver(
            self.Asel.tocsr(), force_solver=xo_solvers[self.sparse_solver], context='cupy')

    def solve(self, rho = None, flag_verbose = False, pic_external = None):

        if rho is None:
            rho = self.rho
            
        self._solve_core(self, rho, pic_external) #change 2

    def get_state_object(self):
        state = FiniteDifferences_Staircase_SquareGrid(chamb=self.chamb, Dh=self.Dh, include_solver=False,
                                                       context=self.context)
        
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

        b = -rho.flatten() / eps0 * self.inside_factor_n  # boundary condition

        if pic_external is not None:
            b[self.idx_border_n] = pic_external.gather_phi(self.x_border, self.y_border)

        phi = cp.zeros(b.size)
        phi[self.idx_sel] = self.luobj.solve(b[self.idx_sel])
        phi=cp.reshape(phi,(self.Nxg,self.Nyg))

        efx = state.efx
        efy = state.efy

        efx[1:self.Nxg-1,:] = phi[0:self.Nxg-2,:] - phi[2:self.Nxg,:];  #central difference on internal nodes
        efy[:,1:self.Nyg-1] = phi[:,0:self.Nyg-2] - phi[:,2:self.Nyg];  #central difference on internal nodes

        efx *= self.border_factor_mat
        efy *= self.border_factor_mat

        state.efx = efx / (2*self.Dhcp);    #divide grid size
        state.efy = efy / (2*self.Dhcp);
        state.rho = rho
        state.phi = phi
        state.b = b
        

        
        






