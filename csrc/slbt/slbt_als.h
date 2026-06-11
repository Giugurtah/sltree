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