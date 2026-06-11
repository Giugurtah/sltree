// csrc/categorizer/kmeans_init.c
#include "kmeans.h"
#include <stdlib.h>
#include <time.h>
#include <float.h>
#include <math.h>

void kplusplus_init(int K, double clusters[], int I, double X[]) {
    double sum, dist, dist_min;
    double distances[I];

    // Random seed initialization
    static int seeded = 0;
    if (!seeded) {
        srand(time(NULL));
        seeded = 1;
    }

    // Firs randomly selected centroid
    clusters[0] = X[rand() % I];

    // For cylce for the remaining K-1 centroids
    for (int k = 1; k < K; k++) {
        sum = 0.0;

        // Point by point evaluate the shortest distance from a centroid
        for (int i = 0; i < I; i++) {
            dist_min = DBL_MAX;
            
            for (int c = 0; c < k; c++) {
                dist = pow(X[i] - clusters[c], 2);
                if (dist < dist_min) {
                    dist_min = dist;
                }
            }
            distances[i] = dist_min * dist_min;  // D(x)^2
            sum += distances[i];
        }

        // Proportional selection
        double r = ((double)rand() / RAND_MAX) * sum;
        double cumulative = 0.0;
        
        for (int i = 0; i < I; i++) {
            cumulative += distances[i];
            if (cumulative >= r) {
                clusters[k] = X[i];
                break;
            }
        }
    }
}