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
import xobjects as xo
#~ from abc import abstractmethod, ABCMeta

na = lambda x:np.array([x])

_kernels = {
    "compute_sc_rho": xo.Kernel(
        args=[
            xo.Arg(xo.Int64, name="N_mp"),
            xo.Arg(xo.Float64, pointer=True, const=True, name="x_mp"),
            xo.Arg(xo.Float64, pointer=True, const=True, name="y_mp"),
            xo.Arg(xo.Float64, pointer=True, const=True, name="nel_mp"),
            xo.Arg(xo.Float64, name="bias_x"),
            xo.Arg(xo.Float64, name="bias_y"),
            xo.Arg(xo.Float64, name="dx"),
            xo.Arg(xo.Float64, name="dy"),
            xo.Arg(xo.Int64, name="Nxg"),
            xo.Arg(xo.Int64, name="Nyg"),
            xo.Arg(xo.Float64, pointer=True, name="rho"),
        ],
        n_threads="N_mp",
    ),
    "int_field": xo.Kernel(
        args=[
            xo.Arg(xo.Int64, name="N_mp"),
            xo.Arg(xo.Float64, pointer=True, const=True, name="x_mp"),
            xo.Arg(xo.Float64, pointer=True, const=True, name="y_mp"),
            xo.Arg(xo.Float64, name="bias_x"),
            xo.Arg(xo.Float64, name="bias_y"),
            xo.Arg(xo.Float64, name="dx"),
            xo.Arg(xo.Float64, name="dy"),
            xo.Arg(xo.Float64, pointer=True, const=True, name="efx"),
            xo.Arg(xo.Float64, pointer=True, const=True, name="efy"),
            xo.Arg(xo.Int64, name="Nxg"),
            xo.Arg(xo.Int64, name="Nyg"),
            xo.Arg(xo.Float64, pointer=True, name="Ex_n"),
            xo.Arg(xo.Float64, pointer=True, name="Ey_n"),
        ],
        n_threads="N_mp",
    ),
    "int_field_border": xo.Kernel(
        args=[
            xo.Arg(xo.Int64, name="N_mp"),
            xo.Arg(xo.Float64, pointer=True, const=True, name="x_mp"),
            xo.Arg(xo.Float64, pointer=True, const=True, name="y_mp"),
            xo.Arg(xo.Float64, name="bias_x"),
            xo.Arg(xo.Float64, name="bias_y"),
            xo.Arg(xo.Float64, name="dx"),
            xo.Arg(xo.Float64, name="dy"),
            xo.Arg(xo.Float64, pointer=True, const=True, name="efx"),
            xo.Arg(xo.Float64, pointer=True, const=True, name="efy"),
            xo.Arg(xo.Int8, pointer=True, const=True, name="inside_mat"),
            xo.Arg(xo.Int64, name="Nxg"),
            xo.Arg(xo.Int64, name="Nyg"),
            xo.Arg(xo.Float64, pointer=True, name="Ex_n"),
            xo.Arg(xo.Float64, pointer=True, name="Ey_n"),
        ],
        n_threads="N_mp",
    ),
}

qe=1.602176565e-19;
eps0=8.8541878176e-12;

class PyPIC_Scatter_Gather(object):
    #__metadata__ = ABCMeta

    def __init__(self, x_aper=None, y_aper=None, dx=None, dy=None, xg=None, yg=None, 
                x_min=None, x_max=None, y_min=None, y_max=None, *args, context=xo.context_default, **kwargs):

        print('PyPIC Version 2.4.5')

        self.context = context
        if 'compute_sc_rho' not in context.kernels:
            context.add_kernels(sources=['#include "PyPIC/headers/scatter_gather.h"'], kernels=_kernels)
        
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

                        
    #@profile
    def scatter(self, x_mp, y_mp, nel_mp, charge = -qe, flag_add=False):
        
        if not (len(x_mp)==len(y_mp)==len(nel_mp)):
            raise ValueError('x_mp, y_mp, nel_mp should have the same length!!!')
        
        rho = self.context.zeros((self.Nxg, self.Nyg))
        if len(x_mp)>0:
            self.context.kernels.compute_sc_rho(N_mp=len(x_mp), x_mp=x_mp, y_mp=y_mp, nel_mp=nel_mp,
                                                bias_x=self.bias_x, bias_y=self.bias_y, dx=self.dx, dy=self.dy,
                                                Nxg=self.Nxg, Nyg=self.Nyg, rho=rho)

        if flag_add:
            self.rho+=charge*rho/(self.dx*self.dy);
        else:
            self.rho=charge*rho/(self.dx*self.dy);

         
    def gather(self, x_mp, y_mp):
        
        if not (len(x_mp)==len(y_mp)):
            raise ValueError('x_mp, y_mp should have the same length!!!')

        return self._int_field(x_mp, y_mp, self.efx, self.efy)
        
    def gather_phi(self, x_mp, y_mp):
        
        if not (len(x_mp)==len(y_mp)):
            raise ValueError('x_mp, y_mp should have the same length!!!')

        return self._int_field(x_mp, y_mp, self.phi, self.phi)[0]
        
    def gather_rho(self, x_mp, y_mp):
        
        if not (len(x_mp)==len(y_mp)):
            raise ValueError('x_mp, y_mp should have the same length!!!')

        return self._int_field(x_mp, y_mp, self.rho, self.rho)[0]

    def _int_field(self, x_mp, y_mp, efx, efy, inside_mat=None):
        # Bilinear interpolation of grid quantities at the particle positions,
        # restricted to the nodes inside the chamber when inside_mat is given
        if len(x_mp)==0:
            return 0., 0.
        xp = self.context.nplike_lib
        Ex_n, Ey_n = self.context.zeros(len(x_mp)), self.context.zeros(len(x_mp))
        kwargs = dict(N_mp=len(x_mp), x_mp=x_mp, y_mp=y_mp, bias_x=self.bias_x, bias_y=self.bias_y,
                      dx=self.dx, dy=self.dy, efx=xp.ascontiguousarray(efx), efy=xp.ascontiguousarray(efy),
                      Nxg=self.Nxg, Nyg=self.Nyg, Ex_n=Ex_n, Ey_n=Ey_n)
        if inside_mat is None:
            self.context.kernels.int_field(**kwargs)
        else:
            self.context.kernels.int_field_border(inside_mat=inside_mat, **kwargs)
        return Ex_n, Ey_n

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
