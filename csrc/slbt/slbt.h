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

double gpi_c(int I, int J, const double *F);
void slba_c(
    int K, int KA, int KB, int I, int J,
    const double *F_noN,  //I*J (Contingency)
    const double *Fs_noN, //K*I*J (Stratified contingency)
    const double *F,      //I*J (Conditional contingency)
    const double *Fs,     //K*I*J (Stratified Conditional contingency)
    double *out_pi,      //1
    double *out_S,        //KA*I
    double *out_alpha,    //KA*I*2
    double *out_beta,     //KB*J*2
    uint64_t seed         //seeds the ALS random initialisation
);


