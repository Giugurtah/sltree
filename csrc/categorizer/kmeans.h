// csrc/categorizer/kmeans.h

#ifndef KMEANS_H
#define KMEANS_H

// ============================================================================
// CORE KMEANS
// ============================================================================

/**
 * Execute KMeans algorithm.
 * 
 * @param I         Number of samples
 * @param K         Number of clusters
 * @param X         Data array (I elements)
 * @param labels    Output: label array (I elements)
 * @param metric    Metric to calculate:
 *                  - 1.0: SSE (Sum of Squared Errors) for elbow
 *                  - 2.0: Silhouette score
 *                  - 3.0: Sequential (threshold-based)
 * @return          Computed metric value
 */
double kmeans_execute(int I, int K, double X[], int labels[], double metric);

// ============================================================================
// INITIALIZATION
// ============================================================================

/**
 * Initialize centroids using KMeans++.
 * 
 * @param K         Number of clusters
 * @param clusters  Output: centroid array (K elements)
 * @param I         Number of samples
 * @param X         Data array (I elements)
 */
void kplusplus_init(int K, double clusters[], int I, double X[]);

// ============================================================================
// METRICS & UTILITIES
// ============================================================================

/**
 * Check if all clusters have size >= minN.
 * 
 * @param I         Number of samples
 * @param K         Number of clusters
 * @param minN      Minimum required size
 * @param labels    Label array (I elements)
 * @return          1 if all clusters are too small, 0 otherwise
 */
int check_cluster_sizes(int I, int K, int minN, int labels[]);

// ============================================================================
// OPTIMAL K SELECTION
// ============================================================================

/**
 * Find optimal K using elbow method.
 * 
 * @param I             Number of samples
 * @param Kmax          Maximum K to test
 * @param Kmin          Minimum K to consider
 * @param minN          Minimum cluster size
 * @param X             Data array (I elements)
 * @param labels_temp1  Temporary buffer for labels (I elements)
 * @param labels_temp2  Temporary buffer for labels (I elements)
 * @return              Optimal K found
 */
int kmeans_elbow(int I, int Kmax, int Kmin, int minN, double X[], 
                 int labels_temp1[], int labels_temp2[]);

/**
 * Find optimal K using silhouette method.
 * 
 * @param I             Number of samples
 * @param Kmax          Maximum K to test
 * @param Kmin          Minimum K to consider
 * @param minN          Minimum cluster size
 * @param X             Data array (I elements)
 * @param labels_temp1  Temporary buffer for labels (I elements)
 * @param labels_temp2  Temporary buffer for labels (I elements)
 * @return              Optimal K (integer) + buffer indicator (0.0 or 0.5)
 */
double kmeans_silhouette(int I, int Kmax, int Kmin, int minN, double X[], 
                         int labels_temp1[], int labels_temp2[]);

// ============================================================================
// PYTHON INTERFACE (categorizer_core.c)
// ============================================================================

/**
 * Categorize data using KMeans with fixed K.
 */
void categorize_kmeans(int I, int K, double X[], int labels[]);

/**
 * Categorize data using KMeans with automatic K (elbow method).
 * @return Optimal K found
 */
int categorize_kmeans_elbow(int I, int Kmax, int Kmin, int minN, 
                             double X[], int labels[]);

/**
 * Categorize data using KMeans with automatic K (silhouette method).
 * @return Optimal K found
 */
int categorize_kmeans_silhouette(int I, int Kmax, int Kmin, int minN, 
                                  double X[], int labels[]);

/**
 * Get cluster centers for a given clustering.
 */
void get_cluster_centers(int I, int K, double X[], int labels[], double centers[]);

/**
 * Get cluster sizes.
 */
void get_cluster_sizes(int I, int K, int labels[], int sizes[]);

#endif // KMEANS_H
