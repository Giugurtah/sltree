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

/*
 * lbrandom.h — deterministic, reentrant pseudo-random generator for SL-TREE
 * =========================================================================
 * Header-only. Replaces libc rand()/srand(), which keep global process-wide
 * state: with rand(), seeding one module silently perturbs every other module
 * in the same process, and results cannot be reproduced.
 *
 * Every generator here carries its own explicit state, so two independent
 * lb_rng_t objects produce independent streams and the same seed always
 * produces the same sequence, on every platform and compiler.
 *
 * Algorithm: splitmix64 (Steele, Lea & Flood 2014). 64 bits of state, one
 * multiply-xor-shift chain per draw. Fast, and with a period of 2^64 it is
 * far beyond what tree initialisation needs.
 *
 * Usage:
 *     lb_rng_t rng;
 *     lb_rng_seed(&rng, seed);
 *     double u = lb_rng_double(&rng);      // uniform in [0, 1)
 *     int    i = (int)lb_rng_below(&rng, n);  // uniform in [0, n), unbiased
 */

#ifndef LB_RANDOM_H
#define LB_RANDOM_H

#include <stdint.h>

typedef struct { uint64_t s; } lb_rng_t;

/* Seed the generator. Any 64-bit value is valid, zero included. */
static inline void lb_rng_seed(lb_rng_t *rng, uint64_t seed) {
    rng->s = seed;
}

/* Next 64 random bits. */
static inline uint64_t lb_rng_next(lb_rng_t *rng) {
    uint64_t z = (rng->s += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

/* Uniform double in [0, 1). Uses the top 53 bits, which is exactly the
 * precision of a double's mantissa: every representable value is reachable
 * and none is favoured. */
static inline double lb_rng_double(lb_rng_t *rng) {
    return (double)(lb_rng_next(rng) >> 11) * (1.0 / 9007199254740992.0);
}

/* Uniform integer in [0, n), unbiased.
 *
 * Note that the idiom rand() % n — used by the current code — is biased
 * whenever n does not divide the generator's range: the first (range % n)
 * values come up slightly more often. Here the biased tail of the range is
 * rejected instead (Lemire 2019), so every value in [0, n) is equally
 * likely. For n = 0 the result is 0. */
static inline uint64_t lb_rng_below(lb_rng_t *rng, uint64_t n) {
    if (n == 0) return 0;
    uint64_t threshold = (uint64_t)(-(int64_t)n) % n;   /* 2^64 mod n */
    for (;;) {
        uint64_t r = lb_rng_next(rng);
        if (r >= threshold) return r % n;
    }
}

#endif /* LB_RANDOM_H */
