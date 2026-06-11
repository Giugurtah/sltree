// Compiled as part of liblbtree
// EXTENDED VERSION: include funzioni weighted per AdaBoost
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "lbtree_core.h"

#define Fij(i, j)  F[(i) * J + (j)]

// ============================================================
//  SECTION 1 — IMPURITY (standard, unweighted)
// ============================================================

static double gini_impurity_rows(int I, int J, const int S[I], const double *F) {
    double N = 0.0, sumJ = 0.0, sumI;
    for (int i = 0; i < I; i++)
        if (S[i])
            for (int j = 0; j < J; j++)
                N += Fij(i, j);
    if (N == 0.0) return 0.0;
    for (int j = 0; j < J; j++) {
        sumI = 0.0;
        for (int i = 0; i < I; i++)
            if (S[i]) sumI += Fij(i, j);
        sumI /= N;
        sumJ += sumI * sumI;
    }
    return N * (1.0 - sumJ);
}

static double gini_impurity_complete(int I, int J, const double *F) {
    int S[I];
    for (int i = 0; i < I; i++) S[i] = 1;
    return gini_impurity_rows(I, J, S, F);
}

static double gini_impurity_twoing_rows(
    int I, int J,
    const int Sx[I], const int Sy[J], const int Sy_N[J],
    const double *F
) {
    double N = 0.0, p1 = 0.0, p0 = 0.0;
    for (int i = 0; i < I; i++)
        if (Sx[i])
            for (int j = 0; j < J; j++)
                N += Fij(i, j);
    if (N == 0.0) return 0.0;
    for (int i = 0; i < I; i++)
        if (Sx[i])
            for (int j = 0; j < J; j++) {
                p1 += (double)Sy[j]   * Fij(i, j) / N;
                p0 += (double)Sy_N[j] * Fij(i, j) / N;
            }
    return N * (1.0 - p1 * p1 - p0 * p0);
}

// ============================================================
//  SECTION 1b — IMPURITY (weighted, for AdaBoost)
// ============================================================

/*
 * gini_impurity_weighted
 * ----------------------
 * Gini impurity con pesi sulle righe.
 *
 *   G_w(S, W) = W_S * (1 - Σ_j p_j²)
 *
 * dove W_S   = Σ_{i:S[i]=1} W[i] * Σ_j F[i,j]
 *      p_j   = (Σ_{i:S[i]=1} W[i] * F[i,j]) / W_S
 *
 * W[i] = peso della riga i (fornito da AdaBoost).
 */
static double gini_impurity_weighted(
    int I, int J,
    const int S[I],
    const double *W,     // W[I]: peso per riga
    const double *F
) {
    double W_total = 0.0, sumJ = 0.0, sumI;

    // Calcola la somma pesata totale
    for (int i = 0; i < I; i++)
        if (S[i])
            for (int j = 0; j < J; j++)
                W_total += W[i] * Fij(i, j);

    if (W_total == 0.0) return 0.0;

    // Calcola p_j² sommato
    for (int j = 0; j < J; j++) {
        sumI = 0.0;
        for (int i = 0; i < I; i++)
            if (S[i])
                sumI += W[i] * Fij(i, j);
        sumI /= W_total;
        sumJ += sumI * sumI;
    }
    return W_total * (1.0 - sumJ);
}

static double gini_impurity_weighted_complete(int I, int J, const double *W, const double *F) {
    int S[I];
    for (int i = 0; i < I; i++) S[i] = 1;
    return gini_impurity_weighted(I, J, S, W, F);
}

/*
 * gini_impurity_twoing_weighted
 * ------------------------------
 * Twoing con pesi.
 */
static double gini_impurity_twoing_weighted(
    int I, int J,
    const int Sx[I], const int Sy[J], const int Sy_N[J],
    const double *W,
    const double *F
) {
    double W_total = 0.0, p1 = 0.0, p0 = 0.0;

    for (int i = 0; i < I; i++)
        if (Sx[i])
            for (int j = 0; j < J; j++)
                W_total += W[i] * Fij(i, j);

    if (W_total == 0.0) return 0.0;

    for (int i = 0; i < I; i++)
        if (Sx[i])
            for (int j = 0; j < J; j++) {
                p1 += (double)Sy[j]   * W[i] * Fij(i, j) / W_total;
                p0 += (double)Sy_N[j] * W[i] * Fij(i, j) / W_total;
            }

    return W_total * (1.0 - p1 * p1 - p0 * p0);
}

// ============================================================
//  SECTION 2 — GPI (unweighted)
// ============================================================

double gpi_c(int I, int J, const double *F) {
    int S[I];
    double I_Y = 0.0, I_YX = 0.0;
    for (int i = 0; i < I; i++) S[i] = 1;
    I_Y = gini_impurity_rows(I, J, S, F);
    for (int i = 0; i < I; i++) S[i] = 0;
    for (int i = 0; i < I; i++) {
        S[i] = 1;
        I_YX += gini_impurity_rows(I, J, S, F);
        S[i] = 0;
    }
    if (I_Y == 0.0) return 0.0;
    return (I_Y - I_YX) / I_Y;
}

// ============================================================
//  SECTION 2b — GPI (weighted)
// ============================================================

double gpi_weighted_c(int I, int J, const double *W, const double *F) {
    int S[I];
    double I_Y = 0.0, I_YX = 0.0;
    for (int i = 0; i < I; i++) S[i] = 1;
    I_Y = gini_impurity_weighted(I, J, S, W, F);
    for (int i = 0; i < I; i++) S[i] = 0;
    for (int i = 0; i < I; i++) {
        S[i] = 1;
        I_YX += gini_impurity_weighted(I, J, S, W, F);
        S[i] = 0;
    }
    if (I_Y == 0.0) return 0.0;
    return (I_Y - I_YX) / I_Y;
}

// ============================================================
//  SECTION 3 — SPLIT ENUMERATION HELPER
// ============================================================

static void incr_S(int I, int S[I], int S_N[I]) {
    for (int i = 0; i < I; i++) {
        if (S[i] == 0) {
            S[i]   = 1;
            S_N[i] = 0;
            return;
        } else {
            S[i]   = 0;
            S_N[i] = 1;
        }
    }
}

// ============================================================
//  SECTION 4 — SPLITTING (unweighted)
// ============================================================

void twoStage_c(int I, int J, const double *F, double *out_pi, double *out_S) {
    int S[I], S_N[I];
    int count = 1;
    double parent_imp, left_imp, right_imp, ppi;

    for (int i = 0; i < I; i++) { S[i] = 0; S_N[i] = 1; }
    parent_imp = gini_impurity_complete(I, J, F);
    *out_pi    = 0.0;
    for (int i = 0; i < I; i++) out_S[i] = 0.0;

    while (count < (int)pow(2, I - 1)) {
        incr_S(I, S, S_N);
        left_imp  = gini_impurity_rows(I, J, S,   F);
        right_imp = gini_impurity_rows(I, J, S_N, F);
        if (parent_imp == 0.0) { count++; continue; }
        ppi = (parent_imp - left_imp - right_imp) / parent_imp;
        if (ppi > *out_pi) {
            *out_pi = ppi;
            for (int i = 0; i < I; i++) out_S[i] = (double)S[i];
        }
        count++;
    }
}

void twoing_c(int I, int J, const double *F, double *out_pi, double *out_S) {
    int Sx[I], Sx_N[I];
    int Sy[J], Sy_N[J];
    int count_x = 1, count_y = 1;
    double parent_imp, left_imp, right_imp, ppi;

    for (int i = 0; i < I; i++) { Sx[i] = 0; Sx_N[i] = 1; }
    for (int j = 0; j < J; j++) { Sy[j] = 0; Sy_N[j] = 1; }
    parent_imp = gini_impurity_complete(I, J, F);
    *out_pi    = 0.0;
    for (int i = 0; i < I; i++) out_S[i] = 0.0;

    while (count_y < (int)pow(2, J - 1)) {
        incr_S(J, Sy, Sy_N);
        while (count_x < (int)pow(2, I - 1)) {
            incr_S(I, Sx, Sx_N);
            left_imp  = gini_impurity_twoing_rows(I, J, Sx,   Sy, Sy_N, F);
            right_imp = gini_impurity_twoing_rows(I, J, Sx_N, Sy, Sy_N, F);
            if (parent_imp == 0.0) { count_x++; continue; }
            ppi = (parent_imp - left_imp - right_imp) / parent_imp;
            if (ppi > *out_pi) {
                left_imp  = gini_impurity_rows(I, J, Sx,   F);
                right_imp = gini_impurity_rows(I, J, Sx_N, F);
                ppi = (parent_imp - left_imp - right_imp) / parent_imp;
                *out_pi = ppi;
                for (int i = 0; i < I; i++) out_S[i] = (double)Sx[i];
            }
            count_x++;
        }
        count_x = 1;
        for (int i = 0; i < I; i++) { Sx[i] = 0; Sx_N[i] = 1; }
        count_y++;
    }
}

// ============================================================
//  SECTION 4b — SPLITTING (weighted)
// ============================================================

void twoStage_weighted_c(int I, int J, const double *W, const double *F, double *out_pi, double *out_S) {
    int S[I], S_N[I];
    int count = 1;
    double parent_imp, left_imp, right_imp, ppi;

    for (int i = 0; i < I; i++) { S[i] = 0; S_N[i] = 1; }
    parent_imp = gini_impurity_weighted_complete(I, J, W, F);
    *out_pi    = 0.0;
    for (int i = 0; i < I; i++) out_S[i] = 0.0;

    while (count < (int)pow(2, I - 1)) {
        incr_S(I, S, S_N);
        left_imp  = gini_impurity_weighted(I, J, S,   W, F);
        right_imp = gini_impurity_weighted(I, J, S_N, W, F);
        if (parent_imp == 0.0) { count++; continue; }
        ppi = (parent_imp - left_imp - right_imp) / parent_imp;
        if (ppi > *out_pi) {
            *out_pi = ppi;
            for (int i = 0; i < I; i++) out_S[i] = (double)S[i];
        }
        count++;
    }
}

// ============================================================
//  SECTION 5 — TWO-CLASS: binary binarization of a numeric target
// ============================================================

/*
 * binarize_target_c 
 * -------------------------
 * Finds the optimal threshold T that splits the N_unique distinct sorted
 * target values into two groups:
 *   "0"  : y <  T
 *   "1" : y >= T
 * 
 * Parameters
 * ----------
 * N_unique      : number of distinct sorted target values
 * y_sorted      : array of N_unique values in ascending order
 * out_threshold : output — chosen split threshold T
 */
void binarize_target_c(int N_unique, const double *y_sorted, double *out_threshold)
{
    int N_d = N_unique, N_s = 0;
    double XM_d, XM_s;
    double XM_d2, XM_s2;
    double X_d = 0.0, X_s = 0.0;
    double X_d2 = 0.0, X_s2 = 0.0;
    double var_d, var_s, var_tot, var_best = 1e100;

    for(int i=0; i< N_unique; i++) {
        X_d += y_sorted[i];
        X_d2 += y_sorted[i] * y_sorted[i];
    }
    for(int i=0; i< N_unique - 1; i++) {
        N_d = N_d - 1;
        X_d -= y_sorted[i];
        X_d2 -= y_sorted[i] * y_sorted[i];
        XM_d = X_d / N_d;
        XM_d2 = XM_d*XM_d;
        var_d = N_d*XM_d2 - 2*XM_d*X_d + X_d2;

        N_s = N_s + 1;
        X_s += y_sorted[i];
        X_s2 += y_sorted[i] * y_sorted[i];
        XM_s = X_s / N_s;
        XM_s2 = XM_s*XM_s;
        var_s = N_s*XM_s2 - 2*XM_s*X_s + X_s2;
        
        var_tot = var_d + var_s;
        if (var_tot < var_best) {
            var_best = var_tot;
            // Il threshold è a metà tra y_sorted[i] e y_sorted[i+1]
            *out_threshold = (y_sorted[i] + y_sorted[i + 1]) / 2.0;
        }
    }
}

void twoing_weighted_c(int I, int J, const double *W, const double *F, double *out_pi, double *out_S) {
    int Sx[I], Sx_N[I];
    int Sy[J], Sy_N[J];
    int count_x = 1, count_y = 1;
    double parent_imp, left_imp, right_imp, ppi;

    for (int i = 0; i < I; i++) { Sx[i] = 0; Sx_N[i] = 1; }
    for (int j = 0; j < J; j++) { Sy[j] = 0; Sy_N[j] = 1; }
    parent_imp = gini_impurity_weighted_complete(I, J, W, F);
    *out_pi    = 0.0;
    for (int i = 0; i < I; i++) out_S[i] = 0.0;

    while (count_y < (int)pow(2, J - 1)) {
        incr_S(J, Sy, Sy_N);
        while (count_x < (int)pow(2, I - 1)) {
            incr_S(I, Sx, Sx_N);
            left_imp  = gini_impurity_twoing_weighted(I, J, Sx,   Sy, Sy_N, W, F);
            right_imp = gini_impurity_twoing_weighted(I, J, Sx_N, Sy, Sy_N, W, F);
            if (parent_imp == 0.0) { count_x++; continue; }
            ppi = (parent_imp - left_imp - right_imp) / parent_imp;
            if (ppi > *out_pi) {
                left_imp  = gini_impurity_weighted(I, J, Sx,   W, F);
                right_imp = gini_impurity_weighted(I, J, Sx_N, W, F);
                ppi = (parent_imp - left_imp - right_imp) / parent_imp;
                *out_pi = ppi;
                for (int i = 0; i < I; i++) out_S[i] = (double)Sx[i];
            }
            count_x++;
        }
        count_x = 1;
        for (int i = 0; i < I; i++) { Sx[i] = 0; Sx_N[i] = 1; }
        count_y++;
    }
}
