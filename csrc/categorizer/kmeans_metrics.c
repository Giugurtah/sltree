// csrc/categorizer/kmeans_metrics.c

#include "kmeans.h"
#include <stdio.h>

int kmeans_elbow(int I, int Kmax, int Kmin, int minN, double X[], 
                 int labels_temp1[], int labels_temp2[]) {
    double ssq[Kmax];
    int current_k = 1;
    int alt = 0;  // Alternates between temp buffers
    int check;
    
    // Calculate SSE for K=1 and K=Kmax
    ssq[0] = kmeans_execute(I, current_k, X, labels_temp1, 1.0);
    ssq[Kmax - 1] = kmeans_execute(I, Kmax, X, labels_temp2, 1.0);
    
    // Calculate slope threshold
    double m = (ssq[Kmax - 1] - ssq[0]) / (Kmax - 1);
    
    printf("Slope threshold m: %f\n", m);
    
    current_k = 2;
    
    // Test each K from 2 to Kmax
    while (current_k <= Kmax) {
        printf("Testing K=%d ", current_k);
        
        if (alt == 0) {
            ssq[current_k - 1] = kmeans_execute(I, current_k, X, labels_temp2, 1.0);
            check = check_cluster_sizes(I, current_k, minN, labels_temp2);
            
            // Check if elbow is found or clusters too small
            if (((ssq[current_k - 1] - ssq[current_k - 2]) > m && current_k > Kmin) || 
                check == 1) {
                return current_k;
            }
            alt = 1;
        } else {
            ssq[current_k - 1] = kmeans_execute(I, current_k, X, labels_temp1, 1.0);
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

double kmeans_silhouette(int I, int Kmax, int Kmin, int minN, double X[], 
                         int labels_temp1[], int labels_temp2[]) {
    int current_k = 2;
    int alt = 0;  // Alternates between temp buffers
    int check = 0;
    double silhouette_score, highest_silhouette = 0.0;
    int optimal_k = 1;
    
    // Test each K from 2 to Kmax
    while (current_k <= Kmax && check == 0) {
        printf("Testing K=%d ", current_k);
        
        if (alt == 0) {
            silhouette_score = kmeans_execute(I, current_k, X, labels_temp2, 2.0);
            check = check_cluster_sizes(I, current_k, minN, labels_temp2);
            
            printf("silhouette score: %f\n", silhouette_score);
            
            if (silhouette_score > highest_silhouette && current_k >= Kmin) {
                highest_silhouette = silhouette_score;
                optimal_k = current_k;
            }
            alt = 1;
        } else {
            silhouette_score = kmeans_execute(I, current_k, X, labels_temp1, 2.0);
            check = check_cluster_sizes(I, current_k, minN, labels_temp1);
            
            printf("silhouette score: %f\n", silhouette_score);
            
            if (silhouette_score > highest_silhouette && current_k >= Kmin) {
                highest_silhouette = silhouette_score;
                optimal_k = current_k;
            }
            alt = 0;
        }
        
        current_k++;
    }
    
    // Return K + buffer indicator (0.0 for temp1, 0.5 for temp2)
    return (double)optimal_k + 0.5 * alt;
}