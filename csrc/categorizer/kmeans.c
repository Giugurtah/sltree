// csrc/categorizer/kmeans.c
#include "kmeans.h"
#include <stdlib.h>
#include <float.h>
#include <math.h>

#define MAX_ITER 1000

double kmeans_execute(int I, int K, double X[], int labels[], double metric) {
    double centers[K], new_centers[K];
    double min_dist, dist;
    int count[K];
    int changed = 1;
    int counter = 0;

    // Centroids initialization (calls a function in /kmean_init.c)
    kplusplus_init(K, centers, I, X);
    
    for (int c = 0; c < K; c++) {
        new_centers[c] = 0.0;
        count[c] = 0;
    }

    // Iteraative execution of the KMeans algorithm
    while (counter < MAX_ITER && changed) {
        // Closest centroid assignment for each point
        for (int i = 0; i < I; i++) {
            min_dist = DBL_MAX;
            
            for (int c = 0; c < K; c++) {
                dist = fabs(centers[c] - X[i]);
                if (dist < min_dist) {
                    min_dist = dist;
                    labels[i] = c;
                }
            }
            
            new_centers[labels[i]] += X[i];
            count[labels[i]] += 1;
        }

        // Update the centroids
        changed = 0;
        for (int c = 0; c < K; c++) {
            if (count[c] > 0) {
                new_centers[c] = new_centers[c] / count[c];
                
                if (new_centers[c] != centers[c]) {
                    changed = 1;
                    centers[c] = new_centers[c];
                }
            }
            
            new_centers[c] = 0.0;
            count[c] = 0;
        }
        
        counter++;
    }

    // Based on the metrics selected either perform an elbow method or a silhouette method
    if (metric == 1.0) {
        // Elbow method
        double sse = 0.0;
        for (int i = 0; i < I; i++) {
            sse += pow(X[i] - centers[labels[i]], 2);
        }
        return sse;
    }
    else if (metric == 2.0) {
        // Silhouette method
        double dist_matrix[K];
        double min_b, total_score = 0.0;
        int cluster_counts[K];

        for (int i = 0; i < I; i++) {
            min_b = DBL_MAX;
            
            // 
            for (int c = 0; c < K; c++) {
                dist_matrix[c] = 0.0;
                cluster_counts[c] = 0;
            }
            
            // Average distance for each centroid
            for (int j = 0; j < I; j++) {
                dist_matrix[labels[j]] += fabs(X[i] - X[j]);
                cluster_counts[labels[j]] += 1;
            }
            
            // a(i): intra-cluster average distance
            cluster_counts[labels[i]] -= 1; 
            if (cluster_counts[labels[i]] > 0) {
                dist_matrix[labels[i]] /= cluster_counts[labels[i]];
            }
            
            // b(i): inter-cluster shortest distance
            for (int c = 0; c < K; c++) {
                if (c != labels[i] && cluster_counts[c] > 0) {
                    dist_matrix[c] /= cluster_counts[c];
                    if (dist_matrix[c] < min_b) {
                        min_b = dist_matrix[c];
                    }
                }
            }
            
            // Silhouette score 
            double a = dist_matrix[labels[i]];
            double s;
            
            if (min_b > a) {
                s = (min_b - a) / min_b;
            } else if (a > 0) {
                s = (min_b - a) / a;
            } else {
                s = 0.0;
            }
            
            total_score += s;
        }
        
        return total_score / I;  // Average silhouette
    }
    else if (metric == 3.0) {
        // Sequential: 
        double center;
        
        if (centers[0] > centers[1]) {
            center = centers[1] + (centers[0] - centers[1]) / 2.0;
        } else {
            center = centers[0] + (centers[1] - centers[0]) / 2.0;
        }
        
        // 
        for (int i = 0; i < I; i++) {
            if (X[i] > center) {
                return (double)(i - 1);
            }
        }
    }

    return 1.0;  // Default
}

int check_cluster_sizes(int I, int K, int minN, int labels[]) {
    int counts[K];
    
    // Conta elementi per cluster
    for (int c = 0; c < K; c++) {
        counts[c] = 0;
    }
    
    for (int i = 0; i < I; i++) {
        counts[labels[i]] += 1;
    }
    
    // Verifica se esiste almeno un cluster abbastanza grande
    for (int c = 0; c < K; c++) {
        if (counts[c] > minN) {
            return 0;  // Trovato un cluster grande
        }
    }
    
    return 1;  // Tutti i cluster sono troppo piccoli
}