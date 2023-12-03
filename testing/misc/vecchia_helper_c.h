#ifndef VECCHIA_HELPER_H
#define VECCHIA_HELPER_H


#define PI (3.141592653589793)
#define NGPU_MAX_NUM 99
#define BATCHCOUNT_MAX 99999999


typedef struct llh_data {
    bool strided;
    int ngpu;
    int nonUniform;

    int M, N;
    int Am, An, Cm, Cn;
    int lda, ldc, ldda, lddc;
    int ldacon, ldccon, Acon, Ccon;
    int lddacon, lddccon;
    int size_first, bs, cs;
    int ldda_first, lddc_first;
    int devices[NGPU_MAX_NUM];
    // TBD for non uniform 
    // int max_M, max_N;
    // int ISEED[4] = {0, 0, 0, 1};
    // int seed = 0;

    double *h_A, *h_C;
    double *d_A[NGPU_MAX_NUM], *d_C[NGPU_MAX_NUM];
    // int *h_M, *h_N,
    //     *d_M[NGPU_MAX_NUM], *d_N[NGPU_MAX_NUM];
    double **d_A_array[NGPU_MAX_NUM], **d_C_array[NGPU_MAX_NUM];
    int *d_ldda[NGPU_MAX_NUM], *d_lddc[NGPU_MAX_NUM];
    double *dot_result_h[NGPU_MAX_NUM];
    double *logdet_result_h[NGPU_MAX_NUM];
    double *logdet_result_h_first[NGPU_MAX_NUM];
    double *dot_result_h_first[NGPU_MAX_NUM];
    //  potrf used
    int *d_info[NGPU_MAX_NUM];
    location *locations;
    location *locations_con_boundary;
    location* locations_con;
    // location* locations_con[BATCHCOUNT_MAX];
    location* locations_copy;
    // no nugget
    double *localtheta;

    // vecchia offset
    double *h_A_conditioning, *h_A_cross, *h_C_conditioning;
    double *d_A_conditioning[NGPU_MAX_NUM], *d_A_cross[NGPU_MAX_NUM], *d_C_conditioning[NGPU_MAX_NUM];
    // vecchia first block 
    // vecchia first block 
    double *h_A_first, *h_C_first; 
    double *d_A_first[NGPU_MAX_NUM], *d_C_first[NGPU_MAX_NUM]; 
    // used for the store the memory of offsets for mu and sigma
    double *d_A_offset[NGPU_MAX_NUM], *d_mu_offset[NGPU_MAX_NUM];
    // double *d_C_copy[NGPU_MAX_NUM];


    int batchCount_gpu;
    int batchCount;

    // lapack flags
	char uplo;
	char transA;
	char transB;
	char side;
	char diag;

    // local theta for kernel in GPs
    double sigma;
    double beta;
    double nu;

    // vecchia
    int vecchia;
    int vecchia_cs;

    // iter
    int iterations;

    // openmp 
    int omp_threads;

    //extra config
    int kernel;
    int num_params;
    int num_loc;
    int seed;

    //vecchia time monitoring
    double vecchia_time_total;

    // bivariate
    int p;

    // real dataset
    int distance_metric; // 0 for euclidean; 1 for earth location.

    // performance && test
    int perf;
    kblasHandle_t *kblas_handle[NGPU_MAX_NUM];
} llh_data;

#endif /* VECCHIA_HELPER_H */

