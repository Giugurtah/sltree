/*
 * SPDX-License-Identifier: GPL-3.0-or-later
 * Copyright (C) 2026 Cristian Buoncompagni, Giulia Vannucci, Roberta Siciliano
 *
 * This file is part of SL-TREE. SL-TREE is free software: you can
 * redistribute it and/or modify it under the terms of the GNU General
 * Public License as published by the Free Software Foundation, either
 * version 3 of the License, or (at your option) any later version.
 * See the LICENSE file, or <https://www.gnu.org/licenses/>.
 */

#define FS(t,i,j) Fs[(t)*I*J + (i)*J + (j)]

#define ALPHA(t,i,k) alpha_out[(t)*I*2 + (i)*2 + (k)]
#define BETA(t,j,k)  beta_out[(t)*J*2 + (j)*2 + (k)]

#include "slbt_als.h"
#include "linalg.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <float.h>
#include <string.h>

//*------Internally used functions------
//Helper functions
void mat_tras_ptr(int I, int J, double** mat, double mat_res[J][I]){ //Function that given a matrix returns its transpose
    for(int i=0; i<I; i++){
        for(int j=0; j<J; j++){
            mat_res[j][i] = mat[i][j];
        }
    }
}
void prod_mat_ptr_vect(int I, int J, double **mat, double arr[J], double arr_res[I]){
    double sum;
    for(int i=0; i<I; i++){
        sum = 0;
        for(int j=0; j<J; j++){
            sum += mat[i][j]*arr[j];
        }
        arr_res[i] = sum;
    }
}
void prod_matrix_ptr(int I, int J, int Z, double** mat1, double mat2[J][Z], double mat_res[I][Z]){ //Given two matrices this function returns their product
    double sum;
    for(int i=0; i<I; i++){
        for(int z=0; z<Z; z++){
            sum = 0.0;
            for(int j=0; j<J; j++){
                sum += ((double)mat1[i][j] * (double)mat2[j][z]);
            }
            mat_res[i][z] = sum;
        }
    }
}
void mat_rand(int I, int J, double mat[I][J], lb_rng_t *rng){ //Function to randomly generate a double matrix
    // The generator is supplied by the caller (see csrc/common/lbrandom.h).
    // Using libc rand() here made the ALS start from a different point on
    // every run, so SLBT trees were not reproducible.
    for (int i=0; i<I; i++){
        for (int j=0; j<J; j++){
            mat[i][j] = lb_rng_double(rng);
        }
    }
}
void mat_tras(int I, int J, double mat[I][J], double mat_res[J][I]){ //Function that given a matrix returns its transpose
    for(int i=0; i<I; i++){
        for(int j=0; j<J; j++){
            mat_res[j][i] = mat[i][j];
        }
    }
}
void prod_matrix(int I, int J, int Z, double mat1[I][J], double mat2[J][Z], double mat_res[I][Z]){ //Given two matrices this function returns their product
    double sum;
    for(int i=0; i<I; i++){
        for(int z=0; z<Z; z++){
            sum = 0.0;
            for(int j=0; j<J; j++){
                sum += ((double)mat1[i][j] * (double)mat2[j][z]);
            }
            mat_res[i][z] = sum;
        }
    }
}
void prod_mat_vect(int I, int J, double mat[I][J], double arr[J], double arr_res[I]){
    double sum;
    for(int i=0; i<I; i++){
        sum = 0;
        for(int j=0; j<J; j++){
            sum += mat[i][j]*arr[j];
        }
        arr_res[i] = sum;
    }
}
void mat_zero(int I, int J, double mat[I][J]){ //Function to zero a double matrix
    for (int i=0; i<I; i++){
        for (int j=0; j<J; j++){
            mat[i][j] = 0.0;
        }
    }
}
double calc_tr(int I, double mat[I][I]){ //Function that given a matrix returns its trace
    double sum = 0;
    for(int i=0; i<I; i++){
        sum += mat[i][i];
    }
    return sum;
}

//Optimization-related functions
void conv_UA(int I, int J, double mat1[I][J], double mat2[I][J]){ //Function used, given U, to get A 
    double sum;
    for(int i=0; i<I; i++){
        sum = 0.0;
        for(int j=0; j<J; j++){
            sum += exp(mat1[i][j]);
        }
        for(int j=0; j<J; j++){
            mat2[i][j] = exp(mat1[i][j])/sum;
        }
    }
}
void conv_ZB(int I, int J, double mat1[I][J], double mat2[I][J]){ //Function used, given Z, to get B
    double sum;
    for(int j=0; j<J; j++){
        sum = 0.0;
        for(int i=0; i<I; i++){
            sum += exp(mat1[i][j]);
        }
        for(int i=0; i<I; i++){
            mat2[i][j] = exp(mat1[i][j])/sum;
        }
    }
}
void conv_df_dU(int Jb, int K, double mat1[Jb][K], double mat2[Jb][K], double mat_res[Jb][K]){ //Given df_dA this function returns df_dU
    double sum = 0;
    double check;
    for (int i=0; i<Jb; i++){
        for (int j=0; j<K; j++){
            sum = 0;
            for (int k=0; k<K; k++){
                check = 0.0;
                if(j == k){
                    check = 1.0;
                }
                sum += mat1[i][k]*(check*mat2[i][k] - mat2[i][k]*mat2[i][j]);

            }
            mat_res[i][j] = sum;
        }
    }
}
void conv_df_dZ(int Jy, int K, double mat1[Jy][K], double mat2[Jy][K], double mat_res[Jy][K]){ //Given df_dB this function returns df_dZ
    double sum = 0;
    double check;
    for (int i=0; i<Jy; i++){
        for (int j=0; j<K; j++){
            sum = 0;
            for (int k=0; k<K; k++){
                check = 0.0;
                if(i == k){
                    check = 1.0;
                }
                sum += mat1[k][j]*(check*mat2[k][j] - mat2[k][j]*mat2[i][j]);
            }
            mat_res[i][j] = sum;
        }
    }
}
void check_UZ(int J, int K, double mat[J][K]){ //Given U(or Z) this function checks if the rows sum to 1
    double sum;
    for(int i=0; i<J; i++){
        for(int j=0; j<K; j++){
            if(mat[i][j] > 20){
                mat[i][j] = 20;
            }
            if(mat[i][j] < -20){
                mat[i][j] = -20;
            }
        }
    }
}
void update_UZ(int J, int K, double rate, double mat1[J][K], double mat2[J][K]){ //Given U(or Z) and df_dU(or df_dZ) this function updates U(or Z) 
    for(int i=0; i<J; i++){
        for(int j=0; j<K; j++){
            mat1[i][j] = mat1[i][j] - rate*mat2[i][j];
        }
    }
}
void DX_calc(int I, int J, double mat1[I][J], double mat2[J][J]){ //Given the transpose of F this function returns DX
    double sum = 0.0;
    for(int j=0; j<J; j++){
        for(int i=0; i<J; i++){
            mat2[i][j] = 0.0;
        }
    }
    for(int j=0; j<J; j++){
        sum = 0.0;
        for(int i=0; i<I; i++){
            sum += mat1[i][j];
        }
        mat2[j][j] = sum;
    }
}


/* ====================================================================
 *  Least-squares fit of the (simultaneous) latent budget model
 * ====================================================================
 *
 * Objective (Eq. 12-13 of the SLBT paper, one term per stratum t):
 *
 *     f(A, B) = sum_t  N_t - 2 tr(F_t' A_t B_t') + tr(B_t A_t' D_t A_t B_t')
 *
 * where F_t is the I x J table handed in by the caller and D_t = diag(F_t 1).
 * What F_t contains decides which least-squares problem is solved:
 *
 *   - rows equal to the observed budgets p_{.|i(t)}  (D_t = I on observed
 *     rows): every observed budget weighs the same ("budget" weighting);
 *   - rows equal to w_{i(t)} p_{.|i(t)} with w proportional to the row
 *     frequency: Eq. 13 with D_t holding the row frequencies
 *     ("frequency" weighting).
 *
 * Up to an additive constant, f equals  sum_{t,i} w_{i(t)} || p_{.|i(t)} -
 * A_{i(t)} B_t' ||^2,  the quantity reported back as the fitted loss.
 *
 * Parameters A and B are reparameterised through softmax (Eq. 16-17) and
 * f is minimised by gradient descent on (U, Z) (Eq. 18-19).  Compared with
 * the original fixed-step loop, this version
 *
 *   1. uses Barzilai-Borwein step lengths with Armijo backtracking instead of
 *      a fixed step: still a gradient method on (U, Z), but monotone (it
 *      cannot diverge) and much faster near the boundary of the simplex,
 *      where the softmax flattens the gradient;
 *   2. stops on a projected-gradient / relative-decrease tolerance instead of
 *      the absolute thresholds 0.1 / 4e-4 / 0.05, which stopped far from the
 *      optimum;
 *   3. runs n_init random starts and keeps the one with the lowest loss;
 *   4. reports convergence diagnostics;
 *   (the alignment of the two latent components across strata in the
 *   unconstrained model is done afterwards by slba_c in slbt_core.c).
 */

#define SLBT_K 2

/* Evaluate f and, if grads != 0, its gradient w.r.t. A and B. */
static double slba_eval(int I, int J, int T, int TA, int TB,
                        double F[T][I][J], double Ft[T][J][I], double DX[T][I][I],
                        double Nsum,
                        double A[TA][I][SLBT_K], double B[TB][J][SLBT_K],
                        double df_dA[TA][I][SLBT_K], double df_dB[TB][J][SLBT_K],
                        int grads)
{
    const int K = SLBT_K;
    double f = Nsum;
    double At[K][I], Bt[K][J];
    double ABt[I][J], FtABt[J][J], DXABt[I][J], AtDXABt[K][J], BAtDXABt[J][J];
    double FB[I][K], BtB[K][K], ABtB[I][K], DXABtB[I][K];
    double FtA[J][K], DXA[I][K], AtDXA[K][K], BAtDXA[J][K];

    if (grads) {
        for (int t = 0; t < TA; t++) mat_zero(I, K, df_dA[t]);
        for (int t = 0; t < TB; t++) mat_zero(J, K, df_dB[t]);
    }

    for (int t = 0; t < T; t++) {
        int ta = (TA == 1) ? 0 : t;
        int tb = (TB == 1) ? 0 : t;

        mat_tras(J, K, B[tb], Bt);
        mat_tras(I, K, A[ta], At);
        prod_matrix(I, K, J, A[ta], Bt, ABt);

        prod_matrix(J, I, J, Ft[t], ABt, FtABt);
        f -= 2.0 * calc_tr(J, FtABt);

        prod_matrix(I, I, J, DX[t], ABt, DXABt);
        prod_matrix(K, I, J, At, DXABt, AtDXABt);
        prod_matrix(J, K, J, B[tb], AtDXABt, BAtDXABt);
        f += calc_tr(J, BAtDXABt);

        if (!grads) continue;

        /* Eq. 14 */
        prod_matrix(I, J, K, F[t], B[tb], FB);
        prod_matrix(K, J, K, Bt, B[tb], BtB);
        prod_matrix(I, K, K, A[ta], BtB, ABtB);
        prod_matrix(I, I, K, DX[t], ABtB, DXABtB);
        for (int i = 0; i < I; i++)
            for (int k = 0; k < K; k++)
                df_dA[ta][i][k] += 2.0 * DXABtB[i][k] - 2.0 * FB[i][k];

        /* Eq. 15 */
        prod_matrix(J, I, K, Ft[t], A[ta], FtA);
        prod_matrix(I, I, K, DX[t], A[ta], DXA);
        prod_matrix(K, I, K, At, DXA, AtDXA);
        prod_matrix(J, K, K, B[tb], AtDXA, BAtDXA);
        for (int j = 0; j < J; j++)
            for (int k = 0; k < K; k++)
                df_dB[tb][j][k] += 2.0 * BAtDXA[j][k] - 2.0 * FtA[j][k];
    }
    return f;
}

/* Left/right assignment of one alpha row: 1 if component 1 (left) wins.
   Ties (alpha exactly 0.5) go left, as in Algorithm 1. */
static inline int slba_goes_left(const double a0, const double a1) {
    return !(a0 < a1);
}

/* 1 if the split of start `cand` equals that of `best` on observed rows,
   up to the label switches the model cannot identify. */
static int slba_same_split(int I, int TA, int per_stratum_swap,
                           int obs[TA][I],
                           double Abest[TA][I][SLBT_K],
                           double Acand[TA][I][SLBT_K])
{
    int global_same = 1, global_swap = 1;
    for (int t = 0; t < TA; t++) {
        int same = 1, swap = 1;
        for (int i = 0; i < I; i++) {
            if (!obs[t][i]) continue;
            int sb = slba_goes_left(Abest[t][i][0], Abest[t][i][1]);
            int sc = slba_goes_left(Acand[t][i][0], Acand[t][i][1]);
            if (sb != sc) same = 0;
            if (sb == sc) swap = 0;
        }
        if (per_stratum_swap) {
            if (!same && !swap) return 0;
        } else {
            global_same &= same;
            global_swap &= swap;
        }
    }
    return per_stratum_swap ? 1 : (global_same || global_swap);
}

/* Euclidean projection of v (length n) onto the probability simplex
   (Duchi et al. 2008), in place. */
static void simplex_project(int n, double *v) {
    double u[n];
    memcpy(u, v, sizeof(double) * n);
    /* insertion sort, descending (n is the number of classes: small) */
    for (int a = 1; a < n; a++) {
        double key = u[a]; int b = a - 1;
        while (b >= 0 && u[b] < key) { u[b + 1] = u[b]; b--; }
        u[b + 1] = key;
    }
    double css = 0.0, theta = 0.0;
    for (int r = 0; r < n; r++) {
        css += u[r];
        double th = (css - 1.0) / (r + 1);
        if (u[r] - th > 0.0) theta = th;
    }
    for (int r = 0; r < n; r++) v[r] = v[r] - theta > 0.0 ? v[r] - theta : 0.0;
}

/* Project A (rows on the 2-simplex) and B (columns on the J-simplex). */
static void project_AB(int I, int J, int TA, int TB,
                       double A[TA][I][SLBT_K], double B[TB][J][SLBT_K]) {
    for (int t = 0; t < TA; t++)
        for (int i = 0; i < I; i++) {
            double a = A[t][i][0];
            a = a < 0.0 ? 0.0 : (a > 1.0 ? 1.0 : a);
            A[t][i][0] = a; A[t][i][1] = 1.0 - a;
        }
    for (int t = 0; t < TB; t++)
        for (int k = 0; k < SLBT_K; k++) {
            double col[J];
            for (int j = 0; j < J; j++) col[j] = B[t][j][k];
            simplex_project(J, col);
            for (int j = 0; j < J; j++) B[t][j][k] = col[j];
        }
}

/*  diag_out (length SLBA_NDIAG, see slbt_als.h):
 *    [0] fitted loss of the retained start  (sum_r w_r ||p_r - pi_r||^2)
 *    [1] largest loss among the n_init starts
 *    [2] iterations used by the retained start
 *    [3] 1 if the retained start met the tolerance before max_iter
 *    [4] number of starts that met the tolerance
 *    [5] number of starts whose split equals the retained one
 *    [6] n_init
 */
void slba_C_execute(int I, int J, int T, int TA, int TB, const double *Fs,
                    double *alpha_out, double *beta_out, uint64_t seed,
                    int n_init, int max_iter, double tol, int optimizer,
                    double *diag_out)
{
    const int K = SLBT_K;
    const double r0  = 0.5;
    const double gtol = 1e-8;

    if (n_init < 1)   n_init = 1;
    if (max_iter < 1) max_iter = 1;
    if (!(tol > 0.0)) tol = 1e-12;

    lb_rng_t rng;
    lb_rng_seed(&rng, seed);

    double F[T][I][J], Ft[T][J][I], DX[T][I][I];
    double Nsum = 0.0, cst = 0.0;

    for (int t = 0; t < T; t++)
        for (int i = 0; i < I; i++)
            for (int j = 0; j < J; j++) {
                F[t][i][j]  = FS(t, i, j);
                Ft[t][j][i] = FS(t, i, j);
                Nsum += FS(t, i, j);
            }

    /* D_t = diag(F_t 1); also the constant that separates f from the loss */
    for (int t = 0; t < T; t++)
        for (int i = 0; i < I; i++) {
            double w = 0.0, sq = 0.0;
            for (int j = 0; j < I; j++) DX[t][i][j] = 0.0;
            for (int j = 0; j < J; j++) { w += F[t][i][j]; sq += F[t][i][j] * F[t][i][j]; }
            DX[t][i][i] = w;
            if (w > 0.0) cst += w - sq / w;
        }

    /* Observed alpha rows: row i of stratum t has data. */
    int obs[TA][I];
    for (int ta = 0; ta < TA; ta++)
        for (int i = 0; i < I; i++) obs[ta][i] = 0;
    for (int t = 0; t < T; t++) {
        int ta = (TA == 1) ? 0 : t;
        for (int i = 0; i < I; i++)
            if (DX[t][i][i] > 0.0) obs[ta][i] = 1;
    }

    double U[TA][I][K], Z[TB][J][K], Uc[TA][I][K], Zc[TB][J][K];
    double A[TA][I][K], B[TB][J][K];
    double df_dA[TA][I][K], df_dB[TB][J][K], df_dU[TA][I][K], df_dZ[TB][J][K];
    double Abest[TA][I][K], Bbest[TB][J][K];
    double Aall[n_init][TA][I][K];
    double loss_start[n_init];

    double best_f = INFINITY, worst_f = -INFINITY;
    int best_s = 0, best_iter = 0, best_conv = 0, n_conv = 0;

    for (int s = 0; s < n_init; s++) {
        /* Start 0 draws exactly what the original single-start code drew,
           so n_init = 1 starts from the same matrices as before. */
        for (int t = 0; t < TA; t++) { mat_rand(I, K, U[t], &rng); check_UZ(I, K, U[t]); }
        for (int t = 0; t < TB; t++) { mat_rand(J, K, Z[t], &rng); check_UZ(J, K, Z[t]); }

        int iter = 0, conv = 0, small = 0;
        double f;
        if (optimizer == 1) {
            /* ---- projected gradient on (alpha_1, B) ------------------------
               Same objective and gradients (Eq. 14-15) without the softmax
               reparameterisation: alpha_{1|i} lives in [0,1] (alpha_2 =
               1 - alpha_1) and every latent budget on the J-simplex.  Optima
               on the boundary (some alpha or beta exactly 0 or 1) are reached
               exactly, where the softmax only approaches them. */
            const int nA = TA * I, nB = TB * J * K, nT = nA + nB;
            double x[nT], x_prev[nT], g[nT], g_prev[nT], xc[nT];
            double lambda = r0 / T;

            for (int t = 0; t < TA; t++) conv_UA(I, K, U[t], A[t]);   /* same start */
            for (int t = 0; t < TB; t++) conv_ZB(J, K, Z[t], B[t]);
            f = slba_eval(I, J, T, TA, TB, F, Ft, DX, Nsum, A, B, df_dA, df_dB, 1);

            while (iter < max_iter) {
                iter++;
                for (int t = 0; t < TA; t++)
                    for (int i = 0; i < I; i++) {
                        x[t * I + i] = A[t][i][0];
                        g[t * I + i] = df_dA[t][i][0] - df_dA[t][i][1];
                    }
                memcpy(x + nA, B, sizeof(double) * nB);
                memcpy(g + nA, df_dB, sizeof(double) * nB);

                /* stationarity: || x - P(x - g) ||_inf */
                {
                    double Ap[TA][I][K], Bp[TB][J][K], pg = 0.0;
                    for (int t = 0; t < TA; t++) for (int i = 0; i < I; i++)
                        Ap[t][i][0] = x[t * I + i] - g[t * I + i];
                    for (int q = 0; q < nB; q++) ((double *)Bp)[q] = x[nA + q] - g[nA + q];
                    project_AB(I, J, TA, TB, Ap, Bp);
                    for (int t = 0; t < TA; t++) for (int i = 0; i < I; i++)
                        pg = fmax(pg, fabs(x[t * I + i] - Ap[t][i][0]));
                    for (int q = 0; q < nB; q++) pg = fmax(pg, fabs(x[nA + q] - ((double *)Bp)[q]));
                    if (pg < gtol) { conv = 1; break; }
                }

                if (iter > 1) {
                    double sy = 0.0, ss = 0.0;
                    for (int q = 0; q < nT; q++) {
                        double sq = x[q] - x_prev[q], yq = g[q] - g_prev[q];
                        sy += sq * yq; ss += sq * sq;
                    }
                    if (sy > 1e-300 && ss > 0.0) lambda = ss / sy; else lambda *= 2.0;
                    if (lambda > 1e6)  lambda = 1e6;
                    if (lambda < 1e-12) lambda = 1e-12;
                }
                memcpy(x_prev, x, sizeof(double) * nT);
                memcpy(g_prev, g, sizeof(double) * nT);

                double f_new = f, Ac[TA][I][K], Bc[TB][J][K];
                int accepted = 0;
                while (lambda > 1e-14) {
                    for (int t = 0; t < TA; t++) for (int i = 0; i < I; i++)
                        Ac[t][i][0] = x[t * I + i] - lambda * g[t * I + i];
                    for (int q = 0; q < nB; q++) ((double *)Bc)[q] = x[nA + q] - lambda * g[nA + q];
                    project_AB(I, J, TA, TB, Ac, Bc);
                    for (int t = 0; t < TA; t++) for (int i = 0; i < I; i++) xc[t * I + i] = Ac[t][i][0];
                    memcpy(xc + nA, Bc, sizeof(double) * nB);
                    f_new = slba_eval(I, J, T, TA, TB, F, Ft, DX, Nsum, Ac, Bc, df_dA, df_dB, 0);
                    double gd = 0.0;
                    for (int q = 0; q < nT; q++) gd += g[q] * (x[q] - xc[q]);
                    if (isfinite(f_new) && f_new <= f - 1e-4 * gd && f_new < f) { accepted = 1; break; }
                    lambda *= 0.5;
                }
                if (!accepted) { conv = 1; break; }

                memcpy(A, Ac, sizeof(A));
                memcpy(B, Bc, sizeof(B));
                double decrease = f - f_new;
                f = slba_eval(I, J, T, TA, TB, F, Ft, DX, Nsum, A, B, df_dA, df_dB, 1);
                if (decrease <= tol * (1.0 + fabs(f - cst))) small++; else small = 0;
                if (small >= 5) { conv = 1; break; }
            }
        } else {
            const int nU = TA * I * K, nZ = TB * J * K, nT = nU + nZ;
            double theta[nT], theta_prev[nT], g[nT], g_prev[nT], cand[nT];
            double lambda = r0 / T;          /* first step as in the original code */

            for (int t = 0; t < TA; t++) conv_UA(I, K, U[t], A[t]);
            for (int t = 0; t < TB; t++) conv_ZB(J, K, Z[t], B[t]);
            f = slba_eval(I, J, T, TA, TB, F, Ft, DX, Nsum, A, B, df_dA, df_dB, 1);

            while (iter < max_iter) {
                iter++;

                /* gradient w.r.t. the unconstrained parameters (Eq. 18-19) */
                for (int t = 0; t < TA; t++) conv_df_dU(I, K, df_dA[t], A[t], df_dU[t]);
                for (int t = 0; t < TB; t++) conv_df_dZ(J, K, df_dB[t], B[t], df_dZ[t]);
                memcpy(theta, U, sizeof(double) * nU);
                memcpy(theta + nU, Z, sizeof(double) * nZ);
                memcpy(g, df_dU, sizeof(double) * nU);
                memcpy(g + nU, df_dZ, sizeof(double) * nZ);

                /* projected gradient: a coordinate pinned at the clipping bound
                   and pushed outwards does not count towards stationarity */
                double gmax = 0.0, gg = 0.0;
                for (int q = 0; q < nT; q++) {
                    if ((theta[q] >= 20.0 && g[q] < 0) || (theta[q] <= -20.0 && g[q] > 0)) continue;
                    if (fabs(g[q]) > gmax) gmax = fabs(g[q]);
                    gg += g[q] * g[q];
                }
                if (gmax < gtol) { conv = 1; break; }

                /* Barzilai-Borwein step from the last two iterates */
                if (iter > 1) {
                    double sy = 0.0, ss = 0.0;
                    for (int q = 0; q < nT; q++) {
                        double sq = theta[q] - theta_prev[q], yq = g[q] - g_prev[q];
                        sy += sq * yq; ss += sq * sq;
                    }
                    if (sy > 1e-300 && ss > 0.0) lambda = ss / sy;
                    else lambda *= 2.0;
                    if (lambda > 1e6)  lambda = 1e6;
                    if (lambda < 1e-12) lambda = 1e-12;
                }
                memcpy(theta_prev, theta, sizeof(double) * nT);
                memcpy(g_prev, g, sizeof(double) * nT);

                /* Armijo backtracking: guarantees a monotone decrease of f */
                double f_new = f;
                int accepted = 0;
                while (lambda > 1e-14) {
                    for (int q = 0; q < nT; q++) {
                        double v = theta[q] - lambda * g[q];
                        cand[q] = v > 20.0 ? 20.0 : (v < -20.0 ? -20.0 : v);
                    }
                    memcpy(Uc, cand, sizeof(double) * nU);
                    memcpy(Zc, cand + nU, sizeof(double) * nZ);
                    for (int t = 0; t < TA; t++) conv_UA(I, K, Uc[t], A[t]);
                    for (int t = 0; t < TB; t++) conv_ZB(J, K, Zc[t], B[t]);
                    f_new = slba_eval(I, J, T, TA, TB, F, Ft, DX, Nsum, A, B, df_dA, df_dB, 0);
                    double gd = 0.0;
                    for (int q = 0; q < nT; q++) gd += g[q] * (theta[q] - cand[q]);
                    if (isfinite(f_new) && f_new <= f - 1e-4 * gd && f_new < f) { accepted = 1; break; }
                    lambda *= 0.5;
                }
                if (!accepted) {
                    /* no descent left at machine precision */
                    for (int t = 0; t < TA; t++) conv_UA(I, K, U[t], A[t]);
                    for (int t = 0; t < TB; t++) conv_ZB(J, K, Z[t], B[t]);
                    conv = 1;
                    break;
                }

                memcpy(U, Uc, sizeof(U));
                memcpy(Z, Zc, sizeof(Z));
                double decrease = f - f_new;
                f = slba_eval(I, J, T, TA, TB, F, Ft, DX, Nsum, A, B, df_dA, df_dB, 1);

                /* stop after 5 consecutive negligible decreases */
                if (decrease <= tol * (1.0 + fabs(f - cst))) small++; else small = 0;
                if (small >= 5) { conv = 1; break; }
                (void)gg;
            }
        }

        double loss = f - cst;
        if (loss < 0.0 && loss > -1e-12) loss = 0.0;
        loss_start[s] = loss;
        memcpy(Aall[s], A, sizeof(A));
        if (conv) n_conv++;
        if (loss > worst_f) worst_f = loss;
        if (loss < best_f - 1e-15) {
            best_f = loss; best_s = s; best_iter = iter; best_conv = conv;
            memcpy(Abest, A, sizeof(A));
            memcpy(Bbest, B, sizeof(B));
        }
    }
    (void)loss_start;

    /* Alignment of the components across strata (unconstrained model) is
       done by the caller, slba_c, which knows the stratum frequencies. */
    int per_stratum_swap = (TA == T && TB == T && T > 1);

    int n_same = 0;
    for (int s = 0; s < n_init; s++)
        n_same += slba_same_split(I, TA, per_stratum_swap, obs, Abest, Aall[s]);

    for (int k = 0; k < K; k++) {
        for (int t = 0; t < TA; t++)
            for (int i = 0; i < I; i++) ALPHA(t, i, k) = Abest[t][i][k];
        for (int t = 0; t < TB; t++)
            for (int j = 0; j < J; j++) BETA(t, j, k) = Bbest[t][j][k];
    }

    if (diag_out) {
        diag_out[0] = best_f;
        diag_out[1] = worst_f;
        diag_out[2] = (double)best_iter;
        diag_out[3] = (double)best_conv;
        diag_out[4] = (double)n_conv;
        diag_out[5] = (double)n_same;
        diag_out[6] = (double)n_init;
    }
    (void)best_s;
}
