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

// csrc/categorizer/kmeans_metrics.c

#include "kmeans.h"

int kmeans_elbow(int I, int Kmax, int Kmin, int minN, double X[], 
                 int labels_temp1[], int labels_temp2[], lb_rng_t *rng) {
    double ssq[Kmax];
    int current_k = 1;
    int alt = 0;  // Alternates between temp buffers
    int check;
    
    // Calculate SSE for K=1 and K=Kmax
    ssq[0] = kmeans_execute(I, current_k, X, labels_temp1, 1.0, rng);
    ssq[Kmax - 1] = kmeans_execute(I, Kmax, X, labels_temp2, 1.0, rng);
    
    // Calculate slope threshold
    double m = (ssq[Kmax - 1] - ssq[0]) / (Kmax - 1);
    
    current_k = 2;
    
    // Test each K from 2 to Kmax
    while (current_k <= Kmax) {
        if (alt == 0) {
            ssq[current_k - 1] = kmeans_execute(I, current_k, X, labels_temp2, 1.0, rng);
            check = check_cluster_sizes(I, current_k, minN, labels_temp2);
            
            // Check if elbow is found or clusters too small
            if (((ssq[current_k - 1] - ssq[current_k - 2]) > m && current_k > Kmin) || 
                check == 1) {
                return current_k;
            }
            alt = 1;
        } else {
            ssq[current_k - 1] = kmeans_execute(I, current_k, X, labels_temp1, 1.0, rng);
            check = check_cluster_sizes(I, current_k, minN, labels_temp1);
            
            // Check if elbow is found or clusters too small
            if (((ssq[current_k - 1] - ssq[current_k - 2]) > m && current_k > Kmin) || 
                check == 1) {
                return current_k;
            }
            alt = 0;
        }
        
        current_k++;
    }
    
    return Kmax;
}

int kmeans_silhouette(int I, int Kmax, int Kmin, int minN, double X[], 
                      int labels_temp1[], int labels_temp2[], lb_rng_t *rng) {
    int current_k = 2;
    int alt = 0;  // Alternates between temp buffers
    int check = 0;
    double silhouette_score, highest_silhouette = 0.0;
    int optimal_k = -1;   // -1 = no K reached a positive silhouette yet
    
    // Test each K from 2 to Kmax
    while (current_k <= Kmax && check == 0) {
        if (alt == 0) {
            silhouette_score = kmeans_execute(I, current_k, X, labels_temp2, 2.0, rng);
            check = check_cluster_sizes(I, current_k, minN, labels_temp2);
            
            if (silhouette_score > highest_silhouette && current_k >= Kmin) {
                highest_silhouette = silhouette_score;
                optimal_k = current_k;
            }
            alt = 1;
        } else {
            silhouette_score = kmeans_execute(I, current_k, X, labels_temp1, 2.0, rng);
            check = check_cluster_sizes(I, current_k, minN, labels_temp1);
            
            if (silhouette_score > highest_silhouette && current_k >= Kmin) {
                highest_silhouette = silhouette_score;
                optimal_k = current_k;
            }
            alt = 0;
        }
        
        current_k++;
    }
    
    // Only the chosen K is returned, not a labelling.
    //
    // The two temporary buffers alternate, so they hold at most the last two
    // clusterings tested: when the best K is more than two steps back its
    // labelling has already been overwritten and cannot be recovered. The
    // caller therefore re-runs kmeans_execute() with the K returned here,
    // exactly as the elbow branch already does, which guarantees that labels
    // and K always agree.
    //
    // If no K reached a positive silhouette (degenerate data, e.g. a constant
    // column), fall back to Kmin rather than leaving an unusable value.
    if (optimal_k < Kmin) {
        optimal_k = Kmin;
    }
    return optimal_k;
}