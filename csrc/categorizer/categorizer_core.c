// csrc/categorizer/categorizer_core.c
//
// This file contains wrapper functions that are called from Python via ctypes.
// These functions provide a clean interface for the categorization algorithms.

#include "kmeans.h"
#include <stdlib.h>
#include <stdio.h>

// ============================================================================
// SIMPLE KMEANS CATEGORIZATION
// ============================================================================

/**
 * Categorize data using KMeans with a fixed K.
 * 
 * This is the simplest interface: you specify K and get back the labels.
 * 
 * @param I         Number of samples
 * @param K         Number of clusters
 * @param X         Input data array (I elements)
 * @param labels    Output labels array (I elements)
 */
void categorize_kmeans(int I, int K, double X[], int labels[]) {
    // Simply run KMeans and ignore the metric return value
    kmeans_execute(I, K, X, labels, 1.0);
}

// ============================================================================
// AUTOMATIC K SELECTION - ELBOW METHOD
// ============================================================================

/**
 * Categorize data using KMeans with automatic K selection via elbow method.
 * 
 * The function will test K values from Kmin to Kmax and select the optimal
 * K based on the elbow in the SSE curve.
 * 
 * @param I         Number of samples
 * @param Kmax      Maximum K to test
 * @param Kmin      Minimum K to consider optimal
 * @param minN      Minimum cluster size required
 * @param X         Input data array (I elements)
 * @param labels    Output labels array (I elements)
 * @return          Optimal K found
 */
int categorize_kmeans_elbow(int I, int Kmax, int Kmin, int minN, 
                             double X[], int labels[]) {
    // Allocate temporary buffers for labels during K search
    int *labels_temp1 = (int*)malloc(I * sizeof(int));
    int *labels_temp2 = (int*)malloc(I * sizeof(int));
    
    if (labels_temp1 == NULL || labels_temp2 == NULL) {
        fprintf(stderr, "Error: Failed to allocate memory for temporary labels\n");
        if (labels_temp1) free(labels_temp1);
        if (labels_temp2) free(labels_temp2);
        return -1;
    }
    
    // Find optimal K using elbow method
    int optimal_k = kmeans_elbow(I, Kmax, Kmin, minN, X, labels_temp1, labels_temp2);
    
    printf("Optimal K found: %d\n", optimal_k);
    
    // Run KMeans one final time with optimal K
    kmeans_execute(I, optimal_k, X, labels, 1.0);
    
    // Clean up
    free(labels_temp1);
    free(labels_temp2);
    
    return optimal_k;
}

// ============================================================================
// AUTOMATIC K SELECTION - SILHOUETTE METHOD
// ============================================================================

/**
 * Categorize data using KMeans with automatic K selection via silhouette method.
 * 
 * The function will test K values from Kmin to Kmax and select the optimal
 * K based on the highest average silhouette score.
 * 
 * @param I         Number of samples
 * @param Kmax      Maximum K to test
 * @param Kmin      Minimum K to consider optimal
 * @param minN      Minimum cluster size required
 * @param X         Input data array (I elements)
 * @param labels    Output labels array (I elements)
 * @return          Optimal K found
 */
int categorize_kmeans_silhouette(int I, int Kmax, int Kmin, int minN, 
                                  double X[], int labels[]) {
    // Allocate temporary buffers for labels during K search
    int *labels_temp1 = (int*)malloc(I * sizeof(int));
    int *labels_temp2 = (int*)malloc(I * sizeof(int));
    
    if (labels_temp1 == NULL || labels_temp2 == NULL) {
        fprintf(stderr, "Error: Failed to allocate memory for temporary labels\n");
        if (labels_temp1) free(labels_temp1);
        if (labels_temp2) free(labels_temp2);
        return -1;
    }
    
    // Find optimal K using silhouette method
    // Returns K + 0.5*buffer_indicator
    double result = kmeans_silhouette(I, Kmax, Kmin, minN, X, labels_temp1, labels_temp2);
    
    int optimal_k = (int)result;
    int buffer_used = (result - optimal_k > 0.25) ? 1 : 0;  // 0.5 indicates temp2
    
    printf("Optimal K found: %d\n", optimal_k);
    
    // Copy labels from the correct buffer
    if (buffer_used == 0) {
        for (int i = 0; i < I; i++) {
            labels[i] = labels_temp1[i];
        }
    } else {
        for (int i = 0; i < I; i++) {
            labels[i] = labels_temp2[i];
        }
    }
    
    // Clean up
    free(labels_temp1);
    free(labels_temp2);
    
    return optimal_k;
}

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

/**
 * Get cluster centers for a given clustering.
 * 
 * @param I         Number of samples
 * @param K         Number of clusters
 * @param X         Input data array (I elements)
 * @param labels    Cluster labels (I elements)
 * @param centers   Output cluster centers (K elements)
 */
void get_cluster_centers(int I, int K, double X[], int labels[], double centers[]) {
    int counts[K];
    
    // Initialize
    for (int k = 0; k < K; k++) {
        centers[k] = 0.0;
        counts[k] = 0;
    }
    
    // Sum values for each cluster
    for (int i = 0; i < I; i++) {
        centers[labels[i]] += X[i];
        counts[labels[i]]++;
    }
    
    // Calculate means
    for (int k = 0; k < K; k++) {
        if (counts[k] > 0) {
            centers[k] /= counts[k];
        }
    }
}

/**
 * Calculate cluster sizes.
 * 
 * @param I         Number of samples
 * @param K         Number of clusters
 * @param labels    Cluster labels (I elements)
 * @param sizes     Output cluster sizes (K elements)
 */
void get_cluster_sizes(int I, int K, int labels[], int sizes[]) {
    // Initialize
    for (int k = 0; k < K; k++) {
        sizes[k] = 0;
    }
    
    // Count elements per cluster
    for (int i = 0; i < I; i++) {
        sizes[labels[i]]++;
    }
}