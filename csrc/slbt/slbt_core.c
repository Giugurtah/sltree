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

#include <stdlib.h>
#include <stdio.h>
#include "slbt_als.h"

#define Fij(i,j) F[(i)*J + (j)]
#define FnoNij(i,j) F_noN[(i)*J + (j)]
#define FsnoNkij(t,i,j) Fs_noN[(t)*I*J + (i)*J + (j)]
#define Fskij(t,i,j) Fs[(t)*I*J + (i)*J + (j)]

#define ALPHA(t,i,k) out_alpha[(t)*I*2 + (i)*2 + (k)]
#define BETA(t,j,k)  out_beta[(t)*J*2 + (j)*2 + (k)]

#define Ski(k,i) out_S[(k)*I + (i)]

#define FL(t,i,j)  F_left[(t)*I*J + (i)*J + (j)]
#define FR(t,i,j)  F_right[(t)*I*J + (i)*J + (j)]

// Internally used functions
double gini_impurity_complete(int I, int J, const double *F){
    double sumI, sumJ = 0.0;
    double N = 0.0;

    // Calculate N
    for(int i=0; i<I; i++){
        for(int j=0; j<J; j++){
            N += Fij(i,j);
        }
    }

    // Avoid division by zero
    if(N == 0.0){
        return 0.0;
    }

    // Calculate gini impurity
    for(int j=0; j<J; j++){
        sumI = 0.0;
        for(int i=0; i<I; i++){
            sumI += Fij(i,j); 
        }
        sumI /= N;
        sumJ += sumI*sumI;
    }

    // Return gini impurity
    return N*(1.0-sumJ);
}

double gini_impurity_strat_k(int I, int J, const double F[I][J]){
    double pyz;
    double impurity = 0.0;

    for(int j=0; j<J; j++){
        pyz = 0.0;
        for(int i=0; i<I; i++){
            pyz += F[i][j];
        }
        impurity += pyz*pyz;
    }

    return 1-impurity;
}

double gini_impurity_rows(int I, int J, const int S[I], const double *F){ //Function used to evaluate the gini impurity
    double sumI, sumJ = 0.0;
    double N = 0.0;
    
    // Calculate N
    for(int i=0; i<I; i++){
        if(S[i]){
            for(int j=0; j<J; j++){
                N += Fij(i,j);
            }
        }
    }
    
    // Avoid division by zero
    if(N == 0.0){
        return 0.0;
    }

    // Calculate gini impurity
    for(int j=0; j<J; j++){
        sumI = 0.0;
        for(int i=0; i<I; i++){
            if(S[i]){
                sumI += Fij(i,j);
            }
        }
        sumI /= N;
        sumJ += sumI*sumI;
    }

    // Return gini impurity
    return N*(1.0-sumJ);
}

// Out-called from slbt/_utils/criteria.py
double gpi_c(int K, int I, int J, const double *Fs){ //Given F this function returns the gpi
    double tau_yk = 0.0; // tau_Y|K
    double tau_ykx = 0.0; // tau_Y|(K,X)

    double P_k, P_jk, P_ik, P_ijk;

    // Calc tau_yk
    for(int k=0; k<K; k++){
        P_k = 0.0;
        for(int j=0; j<J; j++){
            for(int i=0; i<I; i++){
                P_k += Fskij(k,i,j);
            }
        }
        for(int j=0; j<J; j++){
            P_jk = 0.0;
            for(int i=0; i<I; i++){
                P_jk += Fskij(k,i,j);
            }
            if(P_k != 0.0){
                tau_yk += (P_jk*P_jk)/P_k;
            }
        }
    }

    // Calc tau_ykx
    for(int k=0; k<K; k++){
        for(int i=0; i<I; i++){
            P_ik = 0.0;
            P_ijk = 0.0;
            for(int j=0; j<J; j++){
                P_ik += Fskij(k,i,j);
            }
            for(int j=0; j<J; j++){
                P_ijk += Fskij(k,i,j)*Fskij(k,i,j);
            }
            if(P_ik != 0.0){
                tau_ykx += P_ijk/P_ik;
            }
        }
    }

    /* 1 - tau_yk is the within-stratum Gini impurity of Y.  It is zero
       when every stratum is pure: nothing left to predict, so tau = 0
       (the ratio would otherwise be 0/0). */
    if ((1.0 - tau_yk) <= 1e-12) {
        return 0.0;
    }
    return (tau_ykx - tau_yk)/(1 - tau_yk);
}

// Out-called from slbt/_tree/split.py
/*
 * Fit the SLBM with K = 2 on the current predictor, derive the split and
 * evaluate it.
 *
 *   Fs_noN : K x I x J joint proportions at the node (summing to 1)
 *   Fs     : K x I x J table handed to the least-squares fit (row budgets,
 *            possibly multiplied by row weights; see slbt_als.c)
 *
 * The split sends row i of stratum t to the left child when
 * alpha_{1|i(t)} >= 0.5 (Algorithm 1, line 14).
 *
 * Predictability Improvement (Section 2.3): the partial tau of Gray and
 * Williams (Eq. 1) computed on the K x 2 x J table obtained by collapsing
 * the predictor categories into the two sides of the split,
 *
 *     PI = tau_{Y | s, Z}(t).
 *
 * Because s is a coarsening of X within every stratum, PI <= GPI =
 * tau_{Y | X, Z}(t), which is what the FAST early stop relies on.
 *
 * NOTE: earlier versions returned  tau_{Y|X,Z}(parent) - (L tau_left +
 * R tau_right)  with tau computed on un-normalised child tables.  That
 * quantity is not tau_{Y|s,Z} and could rank splits differently.
 */
void slba_c(int K, int KA, int KB, int I, int J, // Input sizes
    const double *Fs_noN, const double *Fs, // Input matrices
    double *out_pi, double *out_S, double *out_alpha, double *out_beta, // Outputs
    uint64_t seed, int n_init, int max_iter, double tol, int optimizer, // fit settings
    double *out_diag // SLBA_NDIAG diagnostics
    ){
    double *F_split = calloc((size_t)K * 2 * J, sizeof(double));
    int S[KA][I];

    // Least-squares fit of the SLBM
    slba_C_execute(I, J, K, KA, KB, Fs, out_alpha, out_beta,
                   seed, n_init, max_iter, tol, optimizer, out_diag);

    // Unconstrained model (A and B both stratum-specific): nothing in the
    // model ties component 1 of one stratum to component 1 of another, so
    // without alignment the random start would decide which observations of
    // different strata share a child node.  Each stratum swaps its two
    // components when that brings its latent budgets closer (L2) to those of
    // the reference stratum, the one with the most observations at the node.
    if(KA == K && KB == K && K > 1){
        int ref = 0;
        double wref = -1.0;
        for(int t=0; t<K; t++){
            double w = 0.0;
            for(int i=0; i<I; i++) for(int j=0; j<J; j++) w += FsnoNkij(t,i,j);
            if(w > wref){ wref = w; ref = t; }
        }
        for(int t=0; t<K; t++){
            if(t == ref) continue;
            double d_same = 0.0, d_swap = 0.0;
            for(int j=0; j<J; j++){
                double b0 = BETA(t,j,0), b1 = BETA(t,j,1);
                double r0 = BETA(ref,j,0), r1 = BETA(ref,j,1);
                d_same += (b0-r0)*(b0-r0) + (b1-r1)*(b1-r1);
                d_swap += (b1-r0)*(b1-r0) + (b0-r1)*(b0-r1);
            }
            if(d_swap < d_same){
                for(int i=0; i<I; i++){ double a = ALPHA(t,i,0); ALPHA(t,i,0) = ALPHA(t,i,1); ALPHA(t,i,1) = a; }
                for(int j=0; j<J; j++){ double b = BETA(t,j,0);  BETA(t,j,0)  = BETA(t,j,1);  BETA(t,j,1)  = b; }
            }
        }
    }

    // Split: alpha_{1|i} >= 0.5 goes left
    for(int t=0; t<KA; t++){
        for(int i=0; i<I; i++){
            S[t][i] = (ALPHA(t,i,0) < ALPHA(t,i,1)) ? 0 : 1;
        }
    }

    // K x 2 x J table of the split: column 0 = left, column 1 = right
    for(int t=0; t<K; t++){
        int ta = (KA == 1) ? 0 : t;
        for(int i=0; i<I; i++){
            int side = S[ta][i] ? 0 : 1;
            for(int j=0; j<J; j++){
                F_split[(t*2 + side)*J + j] += FsnoNkij(t,i,j);
            }
        }
    }

    // PI = partial tau of Y on the split, given Z.  A split that sends a
    // whole stratum to one side contributes nothing for that stratum, and a
    // split that is one-sided everywhere yields exactly 0.
    *out_pi = gpi_c(K, 2, J, F_split);

    for(int t=0; t<KA; t++){
        for(int i=0; i<I; i++){
            Ski(t,i) = (double)S[t][i];
        }
    }

    free(F_split);
}
