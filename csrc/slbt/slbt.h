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

#include "lbrandom.h"

#include "slbt_als.h"

double gpi_c(int K, int I, int J, const double *Fs);
void slba_c(
    int K, int KA, int KB, int I, int J,
    const double *Fs_noN, /* K*I*J joint proportions at the node        */
    const double *Fs,     /* K*I*J table for the least-squares fit      */
    double *out_pi,       /* 1                                          */
    double *out_S,        /* KA*I                                       */
    double *out_alpha,    /* KA*I*2                                     */
    double *out_beta,     /* KB*J*2                                     */
    uint64_t seed,        /* seeds the random initialisation            */
    int n_init,           /* random starts                              */
    int max_iter,         /* iteration cap per start                    */
    double tol,           /* relative-decrease tolerance                */
    int optimizer,        /* 0 = softmax gradient, 1 = projected        */
    double *out_diag      /* SLBA_NDIAG diagnostics                     */
);
