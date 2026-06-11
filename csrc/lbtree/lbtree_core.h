#ifndef SCTREE_CORE_H
#define SCTREE_CORE_H

// ============================================================
//  UNWEIGHTED FUNCTIONS (for SCTree + Random Forest)
// ============================================================

// GPI: Global Predictability Index
double gpi_c(int I, int J, const double *F);

// Two-stage splitting: maximize PPI over all 2^(I-1) partitions
void twoStage_c(int I, int J, const double *F, double *out_pi, double *out_S);

// Twoing: exhaustive row×column partition
void twoing_c(int I, int J, const double *F, double *out_pi, double *out_S);

// ============================================================
//  WEIGHTED FUNCTIONS (for AdaBoost)
// ============================================================

// GPI weighted: same as gpi_c but with sample weights W[I]
double gpi_weighted_c(int I, int J, const double *W, const double *F);

// Two-stage weighted
void twoStage_weighted_c(int I, int J, const double *W, const double *F, double *out_pi, double *out_S);

// Twoing weighted
void twoing_weighted_c(int I, int J, const double *W, const double *F, double *out_pi, double *out_S);

// ============================================================
//  TWO-CLASS — binary binarization of a numeric target
// ============================================================

// binarize_target_c: finds the optimal threshold T to split N_unique sorted
// numeric target values into two groups ("low": y < T, "high": y >= T).
// Called by _binarize_y() in Python before _find_best_predictor().
// NOTE: this is a stub — the optimization logic is not yet implemented.
//
// Parameters
// ----------
// N_unique   : number of distinct sorted target values
// y_sorted   : array of N_unique sorted distinct values of y (ascending)
// out_threshold : output — split threshold T
void binarize_target_c(int N_unique, const double *y_sorted, double *out_threshold);

#endif
