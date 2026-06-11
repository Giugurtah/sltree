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
    double *out_beta      //KB*J*2
);


