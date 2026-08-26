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

// csrc/categorizer/kmeans_init.c
#include "kmeans.h"
#include <stdlib.h>
#include <float.h>
#include <math.h>

void kplusplus_init(int K, double clusters[], int I, double X[], lb_rng_t *rng) {
    double sum, dist, dist_min;
    double distances[I];

    // First randomly selected centroid. The generator is supplied by the
    // caller: no global state, so the result depends only on the seed.
    clusters[0] = X[lb_rng_below(rng, (uint64_t)I)];

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
        double r = lb_rng_double(rng) * sum;
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