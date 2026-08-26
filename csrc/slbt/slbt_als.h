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

void slba_C_execute(
    int I,
    int J,
    int K,
    int KA,
    int KB,
    const double *Fs,
    double *out_alpha,
    double *out_beta
);

#endif