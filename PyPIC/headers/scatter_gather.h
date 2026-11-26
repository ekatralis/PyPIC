// copyright ################################# //
// This file is part of the PyPIC Package.     //
// Copyright (c) CERN, 2011.                   //
// ########################################### //

// Particle-grid interpolation kernels for CPU and GPU (xobjects).
// Grid arrays are C-contiguous with shape (Nxg, Nyg). Cell indices are computed
// as in the original Fortran routines (1-based real index fi = 1 + (x - bias_x) / dx,
// i = int(fi), hx = fi - i), so that the results are unchanged.

#ifndef PYPIC_SCATTER_GATHER_H
#define PYPIC_SCATTER_GATHER_H

#include "xobjects/headers/common.h"
#include "xobjects/headers/atomicadd.h"


GPUKERN void compute_sc_rho(
    const int64_t N_mp,
    GPUGLMEM const double* x_mp, GPUGLMEM const double* y_mp, GPUGLMEM const double* nel_mp,
    const double bias_x, const double bias_y, const double dx, const double dy,
    const int64_t Nxg, const int64_t Nyg,
    GPUGLMEM double* rho)
{
    VECTORIZE_OVER(p, N_mp)
        const double fi = 1. + (x_mp[p] - bias_x) / dx;
        const int64_t i = (int64_t)fi;
        const double hx = fi - (double)i;
        const double fj = 1. + (y_mp[p] - bias_y) / dy;
        const int64_t j = (int64_t)fj;
        const double hy = fj - (double)j;
        const double nel = nel_mp[p];

        if (i > 0 && j > 0 && i < Nxg && j < Nyg) {
            const int64_t ij = (i - 1) * Nyg + (j - 1);
            atomicAdd(&rho[ij], nel * (1. - hx) * (1. - hy));
            atomicAdd(&rho[ij + Nyg], nel * hx * (1. - hy));
            atomicAdd(&rho[ij + 1], nel * (1. - hx) * hy);
            atomicAdd(&rho[ij + Nyg + 1], nel * hx * hy);
        }
    END_VECTORIZE
}


GPUKERN void int_field(
    const int64_t N_mp,
    GPUGLMEM const double* x_mp, GPUGLMEM const double* y_mp,
    const double bias_x, const double bias_y, const double dx, const double dy,
    GPUGLMEM const double* efx, GPUGLMEM const double* efy,
    const int64_t Nxg, const int64_t Nyg,
    GPUGLMEM double* Ex_n, GPUGLMEM double* Ey_n)
{
    VECTORIZE_OVER(p, N_mp)
        const double fi = 1. + (x_mp[p] - bias_x) / dx;
        const int64_t i = (int64_t)fi;
        const double hx = fi - (double)i;
        const double fj = 1. + (y_mp[p] - bias_y) / dy;
        const int64_t j = (int64_t)fj;
        const double hy = fj - (double)j;

        double Ex = 0., Ey = 0.;
        if (i > 0 && j > 0 && i < Nxg && j < Nyg) {
            const int64_t ij = (i - 1) * Nyg + (j - 1);
            Ex = efx[ij] * (1. - hx) * (1. - hy);
            Ex = Ex + efx[ij + Nyg] * hx * (1. - hy);
            Ex = Ex + efx[ij + 1] * (1. - hx) * hy;
            Ex = Ex + efx[ij + Nyg + 1] * hx * hy;

            Ey = efy[ij] * (1. - hx) * (1. - hy);
            Ey = Ey + efy[ij + Nyg] * hx * (1. - hy);
            Ey = Ey + efy[ij + 1] * (1. - hx) * hy;
            Ey = Ey + efy[ij + Nyg + 1] * hx * hy;
        }
        Ex_n[p] = Ex;
        Ey_n[p] = Ey;
    END_VECTORIZE
}


// As int_field, using only the nodes inside the chamber (inside_mat != 0) and
// renormalizing the weights when some of the cell nodes are outside
GPUKERN void int_field_border(
    const int64_t N_mp,
    GPUGLMEM const double* x_mp, GPUGLMEM const double* y_mp,
    const double bias_x, const double bias_y, const double dx, const double dy,
    GPUGLMEM const double* efx, GPUGLMEM const double* efy, GPUGLMEM const int8_t* inside_mat,
    const int64_t Nxg, const int64_t Nyg,
    GPUGLMEM double* Ex_n, GPUGLMEM double* Ey_n)
{
    VECTORIZE_OVER(p, N_mp)
        const double fi = 1. + (x_mp[p] - bias_x) / dx;
        const int64_t i = (int64_t)fi;
        const double hx = fi - (double)i;
        const double fj = 1. + (y_mp[p] - bias_y) / dy;
        const int64_t j = (int64_t)fj;
        const double hy = fj - (double)j;

        double Ex = 0., Ey = 0.;
        if (i > 0 && j > 0 && i < Nxg && j < Nyg) {
            const int64_t ij = (i - 1) * Nyg + (j - 1);
            const double wei_ij = inside_mat[ij] ? (1. - hx) * (1. - hy) : 0.;
            const double wei_i1j = inside_mat[ij + Nyg] ? hx * (1. - hy) : 0.;
            const double wei_ij1 = inside_mat[ij + 1] ? (1. - hx) * hy : 0.;
            const double wei_i1j1 = inside_mat[ij + Nyg + 1] ? hx * hy : 0.;
            const int anyexternal = !(inside_mat[ij] && inside_mat[ij + Nyg] && inside_mat[ij + 1] && inside_mat[ij + Nyg + 1]);

            Ex = efx[ij] * wei_ij;
            Ex = Ex + efx[ij + Nyg] * wei_i1j;
            Ex = Ex + efx[ij + 1] * wei_ij1;
            Ex = Ex + efx[ij + Nyg + 1] * wei_i1j1;

            Ey = efy[ij] * wei_ij;
            Ey = Ey + efy[ij + Nyg] * wei_i1j;
            Ey = Ey + efy[ij + 1] * wei_ij1;
            Ey = Ey + efy[ij + Nyg + 1] * wei_i1j1;

            if (anyexternal) {
                const double wei_sum = wei_ij + wei_i1j + wei_ij1 + wei_i1j1;
                if (wei_sum > 0.) {
                    const double fact_correct = 1. / wei_sum;
                    Ex = Ex * fact_correct;
                    Ey = Ey * fact_correct;
                } else {
                    Ex = 0.;
                    Ey = 0.;
                }
            }
        }
        Ex_n[p] = Ex;
        Ey_n[p] = Ey;
    END_VECTORIZE
}

#endif // PYPIC_SCATTER_GATHER_H
