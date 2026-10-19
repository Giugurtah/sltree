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

#ifndef SLBT_ALS_H
#define SLBT_ALS_H

#include "lbrandom.h"

/* Number of diagnostics written by slba_C_execute into diag_out. */
#define SLBA_NDIAG 7

void slba_C_execute(
    int I,
    int J,
    int K,
    int KA,
    int KB,
    const double *Fs,
    double *out_alpha,
    double *out_beta,
    uint64_t seed,       /* seeds the random initialisation                */
    int n_init,          /* number of random starts; the best one is kept  */
    int max_iter,        /* iteration cap per start                        */
    double tol,          /* relative-decrease tolerance                    */
    int optimizer,       /* 0 = softmax gradient (Eq. 16-19),
                            1 = projected gradient on the simplex          */
    double *diag_out     /* SLBA_NDIAG diagnostics, may be NULL            */
);

#endif