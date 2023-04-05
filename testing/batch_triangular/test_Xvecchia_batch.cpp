/**
 * @copyright (c) 2012- King Abdullah University of Science and
 *                      Technology (KAUST). All rights reserved.
 **/

/**
 * @file testing/batch_triangular/test_Xtrsm_batch.cpp

 * KBLAS is a high performance CUDA library for subset of BLAS
 *    and LAPACK routines optimized for NVIDIA GPUs.
 * KBLAS is provided by KAUST.
 *
 * @version 3.0.0
 * @author Ali Charara
 * @date 2018-11-14
 **/

#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include <sys/time.h>
#include <ctime>

#if ((defined PREC_c) || (defined PREC_z)) && (defined USE_MKL)
// TODO need to handle MKL types properly
#undef USE_MKL
#endif

#include "testing_helper.h"
#include "testing_prec_def.h"
#include "flops.h"

#ifdef check_error
#undef check_error
#endif

#include "Xhelper_funcs.ch"  // TODO: need Xset_pointer_2 from this
#include "kblas_operators.h" // TODO: this has templates and C++ host/device functions (make_one and make_zero)
#include "kblas_common.h"    // TODO: this has templates and C++ host/device functions
#include "Xblas_core.ch"

// Used for llh
#include "ckernel.h"
#include "misc/llg.h"

#define PI (3.141592653589793)

//==============================================================================================
// #define DEBUG_DUMP
//==============================================================================================

#ifdef USING
#undef USING
#endif

#define USING printf("uplo %c, trans %c, batchCount %d, backDoor %d\n", opts.uplo, opts.transA, batchCount, opts.bd);

template <class T>
int test_Xvecchia_batch(kblas_opts &opts, T alpha)
{

    bool strided = opts.strided;
    int nruns = opts.nruns, ngpu = opts.ngpu;
    int nonUniform = opts.nonUniform;
    int M, N, max_M, max_N;
    int Am, An, Cm, Cn;
    int sizeA, sizeC;
    int lda, ldc, ldda, lddc;
    int ione = 1;
    int ISEED[4] = {0, 0, 0, 1};
    // int seed = 0;
    kblasHandle_t kblas_handle[ngpu];

    T *h_A, *h_C, *h_R;
    T *d_A[ngpu], *d_C[ngpu];
    int *h_M, *h_N,
        *d_M[ngpu], *d_N[ngpu];
    T **d_A_array[ngpu], **d_C_array[ngpu];
    int *d_ldda[ngpu], *d_lddc[ngpu];
    T *dot_result_h[ngpu];
    T *logdet_result_h[ngpu];
    T llk = 0;
    //  potrf used
    int *d_info[ngpu];
    location *locations;
    location *locations_con_boundary;
    // location* locations_con;
    location* locations_con[opts.batchCount];
    // no nugget
    T *localtheta;

    // vecchia offset
    T *h_A_copy, *h_A_conditioned, *h_C_conditioned;
    T *d_A_copy[ngpu], *d_A_conditioned[ngpu], *d_C_conditioned[ngpu];
    int ldacon, ldccon, Acon, Ccon;
    int lddacon, lddccon;
    // used for the store the memory of offsets for mu and sigma
    T *d_A_offset[ngpu], *d_mu_offset[ngpu];
    // geam operations
    T alpha_offset_1 = 1;
    T alpha_offset_n1 = 1;
    T alpha_offset_0 = 1;
    T beta_offset_n1 = -1;
    T beta_offset_0 = 0;
    T beta_offset_1 = 1;
    // T alpha_offset_mu = 0;
    // T beta_offset_mu = 1;
    // extra memory for mu, which can be optimized
    // T *h_mu;
    // T *d_mu[ngpu], *d_mu_copy[ngpu];
    T *d_C_copy[ngpu];

    double Cnorm;
    T c_one = make_one<T>(),
      c_neg_one = make_zero<T>() - make_one<T>();
    T work[1];
#ifdef DEBUG_DUMP
    FILE *outK, *outL, *outO;
#endif
    if (ngpu > 1)
        opts.check = 0;
    if (nonUniform)
        strided = 0;

    // USING
    cudaError_t err;
#ifdef USE_MAGMA
    if (opts.magma == 1)
    {
        magma_init(); // TODO is this in the proper place?
    }
#endif

    for (int g = 0; g < ngpu; g++)
    {
        err = cudaSetDevice(opts.devices[g]);
        kblasCreate(&kblas_handle[g]);
#ifdef USE_MAGMA
        if (opts.magma == 1)
        {
            kblasEnableMagma(kblas_handle[g]);
        }
#endif
    }

#ifdef USE_OPENMP
    int NUM_THREADS = opts.omp_numthreads;
#endif // USE_OPENMP

    for (int itest = 0; itest < opts.ntest; ++itest)
    {
        for (int iter = 0; iter < opts.niter; ++iter)
        {
            for (int btest = 0; btest < opts.btest; ++btest)
            {

                double gflops, perf,
                    ref_avg_perf = 0.0, ref_sdev_perf = 0.0, ref_avg_time = 0.0,
                    // rec_avg_perf = 0.0, rec_sdev_perf = 0.0, rec_avg_time = 0.0,
                    kblas_perf = 0.0, kblas_time = 0.0, kblas_time_1 = 0.0, dcmg_time = 0.0,
                    vecchia_time = 0.0;

                int batchCount = opts.batchCount;
                if (opts.btest > 1)
                    batchCount = opts.batch[btest];

                int batchCount_gpu = batchCount / ngpu;

                /*
                seed for location generation
                */
                int seed[batchCount];
                for (int i = 0; i < batchCount; i++)
                {
                    seed[i] = i + 1;
                }

                M = opts.msize[itest];
                N = opts.nsize[itest];

                fflush(stdout);

                /*
                flops in total

                if (nonUniform)
                {
                    max_M = max_N = 0;
                    TESTING_MALLOC_CPU(h_M, int, batchCount);
                    TESTING_MALLOC_CPU(h_N, int, batchCount);

                    for (int k = 0; k < batchCount; k++)
                    {
                        h_M[k] = 1 + (rand() % M);
                        h_N[k] = 1 + (rand() % N);
                        max_M = kmax(max_M, h_M[k]);
                        max_N = kmax(max_N, h_N[k]);
                        gflops += FLOPS_TRSM<T>(opts.side, h_M[k], h_N[k]) / 1e9;
                    }
                }
                else
                {
                    gflops = batchCount * FLOPS_TRSM<T>(opts.side, M, N) / 1e9;
                    gflops += batchCount * FLOPS_POTRF<T>(M) / 1e9;
                    gflops += batchCount * FLOPS_GEVV<T>(M) / 1e9;
                    // used for determinant of triangular matrix
                    gflops += batchCount * FLOPS_GEVV<T>(M) / 1e9;
                }
                */

                if (opts.side == KBLAS_Left)
                {
                    lda = Am = M;
                    An = M;
                }
                else
                {
                    lda = Am = N;
                    An = N;
                }
                ldc = Cm = M;
                Cn = N;

                // the number of conditioned points
                if (opts.vecchia_num > M)
                {
                    fprintf(stderr, "error: vecchia is invalid; ensure 0 <= vecchia_num <= M.\n");
                    exit(1);
                }
                else if (opts.vecchia_num == 0)
                {
                    ldacon = ldccon = Acon = Ccon = M;
                }
                else
                {
                    ldacon = ldccon = Acon = Ccon = opts.vecchia_num;
                }
                lddccon = ((ldccon + 31) / 32) * 32;
                lddacon = lddccon;

                ldda = ((lda + 31) / 32) * 32;
                lddc = ((ldc + 31) / 32) * 32;

                sizeA = lda * An;
                sizeC = ldc * Cn;
                // used for batched llh
                TESTING_MALLOC_PIN(h_A, T, lda * An * batchCount);
                TESTING_MALLOC_PIN(h_C, T, ldc * Cn * batchCount);
                if (opts.vecchia)
                {
                    // used for vecchia offset
                    // the first batch is no conditioned, but we still need the extra memeory for bound cases
                    TESTING_MALLOC_PIN(h_A_copy, T, ldacon * Acon * batchCount);
                    TESTING_MALLOC_PIN(h_A_conditioned, T, ldacon * An * batchCount);
                    TESTING_MALLOC_PIN(h_C_conditioned, T, ldccon * Cn * batchCount);
                    // extra memory for mu
                    // TESTING_MALLOC_PIN(h_mu, T, ldc * Cn * batchCount);
                }

                for (int g = 0; g < ngpu; g++)
                {
                    check_error(cudaSetDevice(opts.devices[g]));
                    TESTING_MALLOC_DEV(d_A[g], T, ldda * An * batchCount_gpu);
                    TESTING_MALLOC_DEV(d_C[g], T, lddc * Cn * batchCount_gpu);
                    TESTING_MALLOC_DEV(d_info[g], int, batchCount_gpu);
                    TESTING_MALLOC_CPU(dot_result_h[g], T, batchCount_gpu);
                    TESTING_MALLOC_CPU(logdet_result_h[g], T, batchCount_gpu);

                    if (opts.vecchia)
                    {
                        // used for vecchia offset
                        TESTING_MALLOC_DEV(d_A_copy[g], T, lddacon * Acon * batchCount_gpu);
                        TESTING_MALLOC_DEV(d_A_conditioned[g], T, lddacon * An * batchCount_gpu);
                        TESTING_MALLOC_DEV(d_C_conditioned[g], T, lddccon * Cn * batchCount_gpu);
                        // store the offset
                        TESTING_MALLOC_DEV(d_mu_offset[g], T, lddc * Cn * batchCount_gpu);
                        TESTING_MALLOC_DEV(d_A_offset[g], T, ldda * An * batchCount_gpu);
                        // extra memory for mu, in order to deal with the boundary value
                        TESTING_MALLOC_DEV(d_C_copy[g], T, lddc * Cn * batchCount_gpu);
                        // TESTING_MALLOC_DEV(d_mu[g], T, lddc * Cn * batchCount_gpu);
                        // TESTING_MALLOC_DEV(d_mu_copy[g], T, lddc * Cn * batchCount_gpu);
                    }

                    /* TODO
                    if (!strided)
                    {
                        TESTING_MALLOC_DEV(d_A_array[g], T *, batchCount_gpu);
                        TESTING_MALLOC_DEV(d_C_array[g], T *, batchCount_gpu);
                    }
                    if (nonUniform)
                    {
                        TESTING_MALLOC_DEV(d_M[g], int, batchCount_gpu);
                        TESTING_MALLOC_DEV(d_N[g], int, batchCount_gpu);
                        TESTING_MALLOC_DEV(d_ldda[g], int, batchCount_gpu);
                        TESTING_MALLOC_DEV(d_lddc[g], int, batchCount_gpu);
                    }
                    */
                }

                /* TODO
                if (opts.check || opts.time)
                {
                    TESTING_MALLOC_CPU(h_R, T, ldc * Cn * batchCount);

#ifdef DEBUG_DUMP
                    outO = fopen("outO.csv", "a");
                    outK = fopen("outK.csv", "a");
                    outL = fopen("outL.csv", "a");
#endif
                    if (opts.check)
                    {
                        opts.time = 0;
                        nruns = 1;
                    }
                }
                */
                // Xrand_matrix(Am, An * batchCount, h_A, lda);
                /*
                maftrix generation using kernel
                */
                // Uniform random generation for locations / read locations from disk
                TESTING_MALLOC_CPU(localtheta, T, 3); // no nugget effect
                localtheta[0] = opts.sigma;
                localtheta[1] = opts.beta;
                localtheta[2] = opts.nu;

                locations = GenerateXYLoc(batchCount * lda, 1);
                // printLocations(batchCount * lda, locations);
                if (opts.vecchia){
                    // Allocate memory for the copy, double is for locations not for computations
                    // locations_con->x = (double*) malloc(batchCount * lda * sizeof(double));
                    // locations_con->y = (double*) malloc(batchCount * lda * sizeof(double));
                    // locations_con->z = NULL;

                    
                    for (int i=0; i < batchCount; i ++){
                        locations_con[i] = (location*) malloc(sizeof(location));
                        locations_con[i]->x=&(locations->x[lda * i]);
                        locations_con[i]->y=&(locations->y[lda * i]); // 
                        locations_con[i]->z = NULL;
                    }                    


                    // printf("locations_con->x[1] is %lf\n", locations_con->x[1]);
                    // Copy the values from the original struct to the copy
                    // for (int i = 0; i < batchCount * lda; i++) {
                    //     locations_con->x[i] = locations->x[i];
                    //     locations_con->y[i] = locations->y[i];
                    // }
                    // printf("locations_con->x[1] is %lf\n", locations_con->x[1]);
                }
                // printLocations(batchCount * lda, locations_con);
                printf("[info] Starting Covariance Generation. \n");
                struct timespec start_dcmg, end_dcmg;
                clock_gettime(CLOCK_MONOTONIC, &start_dcmg);
                for (int i = 0; i < batchCount; i++)
                {   
                    core_dcmg(h_A + i * An * lda,
                              lda, An,
                              locations,
                              locations, localtheta, 0);
                    // printf("x is %lf \n",locations->x[0]);
                    // printf("y is %lf \n",locations->y[0]);
                    // printf("The conditioning covariance matrix.\n");
                    // printMatrixCPU(M, M, h_A + i * An * lda, lda, i); 

                    if (opts.vecchia)
                    {
                        // used for vecchia offset, the first one would not be used for offset
                        if(i==0)
                        {      
                            // this is used for the boundary, which no any meaning for llh
                            locations_con_boundary = GenerateXYLoc(lda, 1);
                            core_dcmg(h_A_conditioned, ldacon, An,
                                                        locations_con_boundary,
                                                        locations, localtheta, 0);
                        }    
                        else
                        {   
                            locations_con[i-1]->x += (Am - Acon);
                            locations_con[i-1]->y += (Am - Acon);
                            // printf("x is %lf \n",locations_con->x[0]);
                            // printf("y is %lf \n",locations_con->y[0]);
                            // printf("x is %lf \n",locations->x[0]);
                            // printf("y is %lf \n",locations->y[0]);
                            core_dcmg(h_A_conditioned + i * An * ldacon,
                                    ldacon, An,
                                    locations_con[i-1],
                                    locations, localtheta, 0); //matrix size: lda by An
                            // printf("The conditioned covariance matrix.\n");
                            // printMatrixCPU(M, M, h_A_conditioned + i * An * lda, lda, i);
                            // locations_con->x += Acon;
                            // locations_con->y += Acon;
                        }
                    }

                    // obeservation initialization
                    for (int j = 0; j < Cm; j++)
                    {
                        h_C[j + i * Cm] = 1;
                        // h_mu[j + i * Cm] = 0;
                    }
                    locations->x += An;
                    locations->y += An;
                }
                clock_gettime(CLOCK_MONOTONIC, &end_dcmg);
                dcmg_time = end_dcmg.tv_sec - start_dcmg.tv_sec + (end_dcmg.tv_nsec - start_dcmg.tv_nsec) / 1e9;
                printf("[info] Finished Covariance Generation with time %lf seconds. \n", dcmg_time);
                /*
                Xrand_matrix(Am, An * batchCount, h_A, lda);
                for (int i = 0; i < batchCount; i++)
                {
                    Xmatrix_make_hpd(Am, h_A + i * An * lda, lda);
                    // printMatrixCPU(M, M, h_A, lda, i);
                }
                */
                // Xrand_matrix(Cm, Cn * batchCount, h_C, ldc);
                // printMatrixCPU(M, M, h_A, lda, i);
                // if (opts.time)
                //     memcpy(h_R, h_C, sizeC * batchCount * sizeof(T));

                for (int g = 0; g < ngpu; g++)
                {
                    check_error(cudaSetDevice(opts.devices[g]));
                    check_cublas_error(cublasSetMatrixAsync(Am, An * batchCount_gpu, sizeof(T),
                                                            h_A + Am * An * batchCount_gpu * g, lda,
                                                            d_A[g], ldda, kblasGetStream(kblas_handle[g])));
                    check_cublas_error(cublasSetMatrixAsync(Cm, Cn * batchCount_gpu, sizeof(T),
                                                            h_C + Cm * Cn * batchCount_gpu * g, ldc,
                                                            d_C[g], lddc, kblasGetStream(kblas_handle[g])));
                    // check_cublas_error(cublasSetMatrixAsync(Cm, Cn * batchCount_gpu, sizeof(T),
                    //                                         h_mu + Cm * Cn * batchCount_gpu * g, ldc,
                    //                                         d_mu[g], lddc, kblasGetStream(kblas_handle[g])));
                    /* TODO
                    if (!strided)
                    {
                        check_kblas_error(Xset_pointer_2(d_A_array[g], d_A[g], ldda, An * ldda,
                                                         d_C_array[g], d_C[g], lddc, Cn * lddc,
                                                         batchCount_gpu, kblasGetStream(kblas_handle[g])));
                    }
                    if (nonUniform)
                    {
                        check_cublas_error(cublasSetVectorAsync(batchCount_gpu, sizeof(int),
                                                                h_M + batchCount_gpu * g, 1,
                                                                d_M[g], 1, kblasGetStream(kblas_handle[g])));
                        check_cublas_error(cublasSetVectorAsync(batchCount, sizeof(int),
                                                                h_N + batchCount_gpu * g, 1,
                                                                d_N[g], 1, kblasGetStream(kblas_handle[g])));
                        check_kblas_error(iset_value_1(d_ldda[g], ldda, batchCount, kblasGetStream(kblas_handle[g])));
                        check_kblas_error(iset_value_1(d_lddc[g], lddc, batchCount, kblasGetStream(kblas_handle[g])));
                    }
                    */
                }

                if (opts.vecchia)
                {   
                    printf("====================== The vecchia offset is starting now! ====================== \n");
                    // /* conditioned part, \sigma_{12} inv (\sigma_{22}) \sigma_{21} and \mu
                    // this part is removable in the real data 
                    for (int i = 0; i < batchCount; i++)
                    {   
                        // pair with conditioned covariance matrix
                        if (i==0){
                            // this does not matter, just some random number to make the matrix postive definite
                            core_dcmg(h_A_copy, Acon, Acon,
                                                        locations_con_boundary,
                                                        locations_con_boundary, 
                                                        localtheta, 0); 
                            memcpy(h_C_conditioned, h_C, sizeof(T) * ldccon * Cn);
                        }
                        else{
                            // varied conditioned number
                            for (int j = 0; j < Acon; j++)
                            {
                                memcpy(h_A_copy + Acon * j + i * Acon * Acon,
                                    h_A  + lda * An * (i -1) + (An - Acon + j) * An + (An - Acon), //  = i * An * An,
                                    sizeof(T) * Acon);                                       // the last one will not be used to offset
                            }
                            memcpy(h_C_conditioned + ldccon * Cn * i, 
                                    h_C + ldc * Cn * (i - 1) + (ldc - ldccon), 
                                    sizeof(T) * ldccon * Cn);
                        }
                        // printf("h_A_copy: \n");
                        // printMatrixCPU(Acon, Acon, h_A_copy + i * Acon * Acon, ldacon, i);
                    }
                    // extra memory for mu
                    // check_cublas_error(cublasSetMatrixAsync(Cm, Cn * batchCount_gpu, sizeof(T),
                    //                                         h_mu + Cm * Cn * batchCount_gpu * g, ldc,
                    //                                         d_mu_copy[g], lddc, kblasGetStream(kblas_handle[g])));
                    // conditioned part 1.1, matrix copy from host to device
                    for (int g = 0; g < ngpu; g++)
                    {
                        check_error(cudaSetDevice(opts.devices[g]));
                        check_cublas_error(cublasSetMatrixAsync(Acon, Acon * batchCount_gpu, sizeof(T),
                                                                h_A_copy + Acon * Acon * batchCount_gpu * g, ldacon,
                                                                d_A_copy[g], lddacon, kblasGetStream(kblas_handle[g])));
                        check_cublas_error(cublasSetMatrixAsync(Acon, An * batchCount_gpu, sizeof(T),
                                                                h_A_conditioned + Acon * An * batchCount_gpu * g, ldacon,
                                                                d_A_conditioned[g], lddacon, kblasGetStream(kblas_handle[g])));
                        check_cublas_error(cublasSetMatrixAsync(Ccon, Cn * batchCount_gpu, sizeof(T),
                                                                h_C_conditioned + Ccon * Cn * batchCount_gpu * g, ldccon,
                                                                d_C_conditioned[g], lddccon, kblasGetStream(kblas_handle[g])));
                        // check the device for offset, needed?  TODO test
                        // check_cublas_error(cublasSetMatrixAsync(Cm, Cn * batchCount_gpu, sizeof(T),
                        //                                         h_C_conditioned + Cm * Cn * batchCount_gpu * g, ldc,
                        //                                         d_mu_offset[g], lddc, kblasGetStream(kblas_handle[g])));
                        check_cublas_error(cublasSetMatrixAsync(Am, An * batchCount_gpu, sizeof(T),
                                                                h_A_copy + Am * An * batchCount_gpu * g, lda,
                                                                d_A_offset[g], ldda, kblasGetStream(kblas_handle[g])));
                        check_cublas_error(cublasSetMatrixAsync(Cm, Cn * batchCount_gpu, sizeof(T),
                                                                h_C + Cm * Cn * batchCount_gpu * g, ldc,
                                                                d_C_copy[g], lddc, kblasGetStream(kblas_handle[g])));
                        /*
                        TODO
                        if (!strided)
                        {
                            check_kblas_error(Xset_pointer_2(d_A_array[g], d_A[g], ldda, An * ldda,
                                                             d_C_array[g], d_C[g], lddc, Cn * lddc,
                                                             batchCount_gpu, kblasGetStream(kblas_handle[g])));
                        }
                        if (nonUniform)
                        {
                            check_cublas_error(cublasSetVectorAsync(batchCount_gpu, sizeof(int),
                                                                    h_M + batchCount_gpu * g, 1,
                                                                    d_M[g], 1, kblasGetStream(kblas_handle[g])));
                            check_cublas_error(cublasSetVectorAsync(batchCount, sizeof(int),
                                                                    h_N + batchCount_gpu * g, 1,
                                                                    d_N[g], 1, kblasGetStream(kblas_handle[g])));
                            check_kblas_error(iset_value_1(d_ldda[g], ldda, batchCount, kblasGetStream(kblas_handle[g])));
                            check_kblas_error(iset_value_1(d_lddc[g], lddc, batchCount, kblasGetStream(kblas_handle[g])));
                        }
                        */
                    }

                    // conditioned part 1.2, wsquery for potrf and trsm
                    for (int g = 0; g < ngpu; g++)
                    {
                        if (strided)
                        {
                            kblas_potrf_batch_strided_wsquery(kblas_handle[g], Acon, batchCount_gpu);
                            kblas_trsm_batch_strided_wsquery(kblas_handle[g], opts.side, Acon, N, batchCount_gpu);
                            kblas_gemm_batch_strided_wsquery(kblas_handle[g], batchCount);
                        }
                        else
                        {
                            return 0;
                        }
                        /* TODO
                        else if (nonUniform)
                            kblas_trsm_batch_nonuniform_wsquery(kblas_handle[g]);
                        else
                        {
                            kblas_potrf_batch_wsquery(kblas_handle[g], M, batchCount_gpu);
                            kblas_trsm_batch_wsquery(kblas_handle[g], opts.side, M, N, batchCount_gpu);
                        }
                        */
                        check_kblas_error(kblasAllocateWorkspace(kblas_handle[g]));
                        check_error(cudaGetLastError());
                    }

                    // conditioned part 2, for batched operationed
                    for (int g = 0; g < ngpu; g++)
                    {
                        check_error(cudaSetDevice(opts.devices[g]));
                        cudaDeviceSynchronize(); // TODO sync with streams instead
                    }

                    struct timespec start, end;
                    clock_gettime(CLOCK_MONOTONIC, &start);

                    for (int g = 0; g < ngpu; g++)
                    {
                        check_error(cudaSetDevice(opts.devices[g]));
                        /*
                        cholesky decomposition
                        */
                        // for (int i = 0; i < batchCount_gpu; i++)
                        // {
                        //     printf("%dth", i);
                        //     printMatrixGPU(Am, An, d_A_copy[g] + i * Am * ldda, ldda);
                        // } 
                        printf("[info] Starting Cholesky decomposition. \n");
                        if (strided)
                        {
                            check_kblas_error(kblas_potrf_batch(kblas_handle[g],
                                                                opts.uplo, Am,
                                                                d_A_copy[g], lddacon, Acon * lddacon,
                                                                batchCount_gpu,
                                                                d_info[g]));
                        }
                        /* TODO
                        else
                        {
                            check_kblas_error(kblas_potrf_batch(kblas_handle[g],
                                                                opts.uplo, Am,
                                                                d_A_array[g], ldda,
                                                                batchCount_gpu,
                                                                d_info[g]));
                        }
                        */
                        // for (int i = 0; i < batchCount_gpu; i++)
                        // {
                        //     printf("%dth", i);
                        //     printMatrixGPU(Am, An, d_A[g] + i * Am * ldda, ldda);
                        // }
                        printf("[info] Finished Cholesky decomposition. \n");
                        /*
                        triangular solution: L \Sigma_offset <- \Sigma_old && L z_offset <- z_old
                        */
                        printf("[info] Starting triangular solver. \n");
                        // for (int i = 0; i < batchCount_gpu; i++)
                        // {
                        //     printf("%dth", i);
                        //     printMatrixGPU(Am, An, d_A_copy[g] + i * Am * ldda, ldda);
                        //     printMatrixGPU(Am, An, d_A_conditioned[g] + i * Am * ldda, ldda);
                        // }

                        if (strided)
                        {
                            check_kblas_error(kblasXtrsm_batch_strided(kblas_handle[g],
                                                                       opts.side, opts.uplo, KBLAS_NoTrans, opts.diag,
                                                                       lddacon, An,
                                                                       alpha,
                                                                       d_A_copy[g], lddacon, Acon * lddacon, // A <- L
                                                                       d_A_conditioned[g], lddacon, An * lddacon,
                                                                       batchCount_gpu)); //d_A_conditioned <- 
                            check_kblas_error(kblasXtrsm_batch_strided(kblas_handle[g],
                                                                       opts.side, opts.uplo, KBLAS_NoTrans, opts.diag,
                                                                       lddccon, Cn,
                                                                       alpha,
                                                                       d_A_copy[g], lddacon, Acon * lddacon, // A <- L
                                                                       d_C_conditioned[g], lddccon, Cn * lddccon,
                                                                       batchCount_gpu));
                        }
                        // for (int i = 0; i < batchCount_gpu; i++)
                        // {
                        //     printf("%dth", i);
                        //     printMatrixGPU(Am, An, d_A_conditioned[g] + i * Am * ldda, ldda);
                        // }
                        /*TODO
                        else
                        {
                            check_cublas_error(
                                cublasDtrsmBatched(kblasGetCublasHandle(kblas_handle[g]),
                                                   CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_T, CUBLAS_DIAG_NON_UNIT,
                                                   An, Cn,
                                                   &alpha,
                                                   d_A_array[g], ldda,
                                                   d_C_array[g], lddc,
                                                   batchCount_gpu));
                            // check_kblas_error(kblasXtrsm_batch(kblas_handle[g],
                            //                                    opts.side, opts.uplo, opts.transA, opts.diag,
                            //                                    M, N,
                            //                                    alpha, (const T **)(d_A_array[g]), ldda,
                            //                                    d_C_array[g], lddc,
                            //                                    batchCount_gpu));
                        }
                        */
                        // printVecGPU(Cm, Cn, d_C[0], ldda);
                        printf("[info] Finished triangular solver. \n");
                        /*
                        GEMM and GEMV: \Sigma_offset^T %*% \Sigma_offset and \Sigma_offset^T %*% z_offset
                        */
                        printf("[info] Starting GEMM and GEMV. \n");
                        if (strided)
                        {
                            // \Sigma_offset^T %*% \Sigma_offset
                            // for (int i = 0; i < batchCount_gpu; i++)
                            // {
                            //     printf("%dth", i);
                            //     printMatrixGPU(Am, An, d_A_conditioned[g] + i * Acon * lddacon, lddacon);
                            // }
                            check_kblas_error(kblas_gemm_batch(kblas_handle[g],
                                                               KBLAS_Trans, KBLAS_NoTrans,
                                                               Am, An, Acon,
                                                               alpha,
                                                               d_A_conditioned[g], lddacon, An * lddacon,
                                                               d_A_conditioned[g], lddacon, An * lddacon,
                                                               0,
                                                               d_A_offset[g], ldda, An * ldda,
                                                               batchCount_gpu));
                            // for (int i = 0; i < batchCount_gpu; i++)
                            // {
                            //     printf("%dth", i);
                            //     printMatrixGPU(Am, An, d_A_offset[g] + i * An * ldda, ldda);
                            // } 
                            // \Sigma_offset^T %*% z_offset
                            check_kblas_error(kblas_gemm_batch(kblas_handle[g],
                                                               KBLAS_Trans, KBLAS_NoTrans,
                                                               Am, Cn, Acon,
                                                               alpha,
                                                               d_A_conditioned[g], lddacon, An * lddacon,
                                                               d_C_conditioned[g], lddccon, Cn * lddccon,
                                                               0,
                                                               d_mu_offset[g], lddc, Cn * lddc,
                                                               batchCount_gpu));
                            // for (int i = 0; i < batchCount_gpu; i++)
                            // {
                            //     printf("%dth", i);
                            //     printVecGPU(Cm, Cn, d_mu_offset[g] + i * lddc * Cn, lddc, i);
                            // } 
                        }
                        /*TODO non-strided*/
                        printf("[info] Finished GEMM and GEMV. \n");

                        /*
                        GEAD: \Sigma_new <- \Sigma - \Sigma_offset && \mu_new <- \mu - \mu_offset (not necessary)
                        */
                        printf("[info] Starting GEAD. \n");
                        // for (int i = 0; i < batchCount_gpu; i++)
                        // {
                        //     printf("%dth", i);
                        //     printMatrixGPU(Am, An, d_A_copy[g] + i * Am * ldda, ldda);
                        // }
                        // for (int i = 0; i < batchCount_gpu; i++)
                        // {
                        //     printf("%dth", i);
                        //     printMatrixGPU(Am, An, d_A_offset[g] + i * Am * ldda, ldda);
                        // }
                        if (strided)
                        {
                            for (int i = 1; i < batchCount_gpu; i++)
                            {
                                check_cublas_error(cublasXgeam(kblasGetCublasHandle(kblas_handle[g]),
                                                               CUBLAS_OP_N, CUBLAS_OP_N,
                                                               Am, An,
                                                               &alpha_offset_1, // 1
                                                               d_A[g] + ldda * An * i, ldda, // !!!!!not sure if it's ok, in order to save memory !!!!
                                                               &beta_offset_n1, // -1
                                                               d_A_offset[g] + ldda * An * i, ldda,
                                                               d_A[g] + ldda * An * i, ldda)); // + ldda * An * i means the first one does not change
                                // check_cublas_error(cublasXgeam(kblasGetCublasHandle(kblas_handle[g]),
                                //                                CUBLAS_OP_N, CUBLAS_OP_N,
                                //                                Cm, Cn,
                                //                                alpha_offset_mu,
                                //                                d_mu_copy[g] + lddc * Cn * i, ldda, ldda,
                                //                                beta_offset_mu,
                                //                                d_mu_offset[g], lddc, Cn * lddc,
                                //                                d_mu[g] + lddc * Cn * i, lddc));
                            }
                        }
                        // for (int i = 0; i < batchCount_gpu; i++)
                        // {
                        //     printf("%dth", i);
                        //     printMatrixGPU(Am, An, d_A[g] + i * Am * ldda, ldda);
                        // }
                        /*TODO non-strided*/
                        printf("[info] Finished GEAD. \n");
                    }

                    for (int g = 0; g < ngpu; g++)
                    {
                        check_error(cudaSetDevice(opts.devices[g]));
                        cudaDeviceSynchronize(); // TODO sync with streams instead
                    }
                    clock_gettime(CLOCK_MONOTONIC, &end);
                    vecchia_time = end.tv_sec - start.tv_sec + (end.tv_nsec - start.tv_nsec) / 1e9;

                    cudaFreeHost(h_A_copy);
                    cudaFreeHost(h_A_conditioned);
                    cudaFreeHost(h_C_conditioned);
                    for (int g = 0; g < ngpu; g++)
                    {
                        check_error(cudaSetDevice(opts.devices[g]));
                        check_error(cudaFree(d_A_copy[g]));
                        check_error(cudaFree(d_A_conditioned[g]));
                        check_error(cudaFree(d_C_conditioned[g]));
                        check_error(cudaFree(d_A_offset[g]));
                    }

                    // */conditioned part, \sigma_{12} inv (\sigma_{22}) \sigma_{21}
                    printf("====================== The vecchia offset is finished! ====================== \n");
                    printf("[Info] The time for vecchia offset is %lf seconds \n", vecchia_time);
                }

                printf("====================== Independent computing is starting now! ====================== \n");

                /*
                Independent computing
                */
                for (int g = 0; g < ngpu; g++)
                {
                    if (strided)
                    {
                        kblas_potrf_batch_strided_wsquery(kblas_handle[g], M, batchCount_gpu);
                        kblas_trsm_batch_strided_wsquery(kblas_handle[g], opts.side, M, N, batchCount_gpu);
                    }
                    /* TODO
                    else if (nonUniform)
                        kblas_trsm_batch_nonuniform_wsquery(kblas_handle[g]);
                    else
                    {
                        kblas_potrf_batch_wsquery(kblas_handle[g], M, batchCount_gpu);
                        kblas_trsm_batch_wsquery(kblas_handle[g], opts.side, M, N, batchCount_gpu);
                    }
                    */
                    check_kblas_error(kblasAllocateWorkspace(kblas_handle[g]));
                    check_error(cudaGetLastError());
                }

                struct timespec start, end;
                clock_gettime(CLOCK_MONOTONIC, &start);
                
                for (int g = 0; g < ngpu; g++)
                {
                    check_error(cudaSetDevice(opts.devices[g]));
                    cudaDeviceSynchronize(); // TODO sync with streams instead
                }
                for (int g = 0; g < ngpu; g++)
                {
                    check_error(cudaSetDevice(opts.devices[g]));
                    /*
                    cholesky decomposition
                    */
                    // printf("[info] Starting Cholesky decomposition. \n");
                    // for (int i = 0; i < batchCount_gpu; i++)
                    // {
                    //     printf("%dth", i);
                    //     printMatrixGPU(Am, An, d_A[g] + i * Am * ldda, ldda);
                    // }
                    if (strided)
                    {
                        check_kblas_error(kblas_potrf_batch(kblas_handle[g],
                                                            opts.uplo, Am,
                                                            d_A[g], ldda, An * ldda,
                                                            batchCount_gpu,
                                                            d_info[g]));
                    }
                    /*TODO
                    else
                    {
                        check_kblas_error(kblas_potrf_batch(kblas_handle[g],
                                                            opts.uplo, Am,
                                                            d_A_array[g], ldda,
                                                            batchCount_gpu,
                                                            d_info[g]));
                    }*/
                    // for (int i = 0; i < batchCount_gpu; i++)
                    // {
                    //     printf("%dth", i);
                    //     printMatrixGPU(Am, An, d_A[g] + i * Am * ldda, ldda);
                    // }
                    printf("[info] Finished Cholesky decomposition. \n");
                    /*
                    determinant
                    */
                    if (strided)
                    {
                        for (int i = 0; i < batchCount_gpu; i++)
                        {
                            core_Xlogdet<T>(d_A[g] + i * An * ldda, An, ldda, &(logdet_result_h[g][i]));
                            // printf("the det value is %lf \n", logdet_result_h[g][i]);
                            // cudaDeviceSynchronize();
                        }
                        // printf("The results during log-det.");
                        // printMatrixGPU(M, M, d_A[0] + An * ldda, ldda);
                    }
                    /* TODO
                    else
                    {
                        // for (int i = 0; i < batchCount_gpu; i++)
                        // {
                        //     core_Xdet<T>(d_A_array[g] + i * An * lda, An, lda, &(logdet_result_h[g][i]));
                        // }
                        return 0;
                    }
                    */
                    printf("[info] Finished determinant. \n");
                    /*
                    triangular solution: L Z_new <- Z_old
                    */
                    printf("[info] Starting triangular solver. \n");
                    if (opts.vecchia)
                    {
                        for (int i = 1; i < batchCount_gpu; i++)
                        {
                            // printf("The results before TRSM \n");
                            // printVecGPU(Cm, Cn, d_C[g], ldc, i);
                            // printf("The results before TRSM \n");
                            // printVecGPU(Cm, Cn, d_C_copy[g]+ lddc * Cn * i, ldc, i);
                            // printf("The results before TRSM \n");
                            // printVecGPU(Cm, Cn, d_mu_offset[g]+ lddc * Cn * i, ldc, i);
                            check_cublas_error(cublasXgeam(kblasGetCublasHandle(kblas_handle[g]),
                                                           CUBLAS_OP_N, CUBLAS_OP_N,
                                                           Cm, Cn,
                                                           &alpha_offset_1, // 1
                                                           d_C_copy[g] + lddc * Cn * i, lddc,
                                                           &beta_offset_n1, // -1
                                                           d_mu_offset[g] + lddc * Cn * i, lddc,
                                                           d_C[g] + lddc * Cn * i, lddc));
                            // printf("The results before TRSM \n");
                            // printVecGPU(Cm, Cn, d_C[g] + i * Cn * lddc, ldc, i);
                        }
                        check_error(cudaFree(d_mu_offset[g]));
                        // printf("The results before TRSM \n");
                        // printVecGPU(Cm, Cn, d_C[g] + Cn * lddc, ldc, 1);
                        // printf("The results before TRSM \n");
                        // printVecGPU(Cm, Cn, d_C[g], ldc, 0);
                    }
                    // printf("The results before TRSM.");
                    // printMatrixGPU(M, M, d_A[0] + An * ldda, ldda);
                    // for(int i=0; i<batchCount_gpu; i++){
                    //     printVecGPU(Cm, Cn, d_C[g] + i * Cn * lddc, ldc, i);
                    // }
                    // for (int i = 0; i < batchCount_gpu; i++)
                    // {
                    //     printf("%dth", i);
                    //     printMatrixGPU(Am, An, d_A[g] + i * Am * ldda, ldda);
                    // }
                    if (strided)
                    {
                        check_kblas_error(kblasXtrsm_batch_strided(kblas_handle[g],
                                                                   opts.side, opts.uplo, opts.transA, opts.diag,
                                                                   An, Cn,
                                                                   alpha,
                                                                   d_A[g], ldda, An * ldda,
                                                                   d_C[g], lddc, Cn * lddc,
                                                                   batchCount_gpu));
                    }
                    // for(int i=0; i<batchCount_gpu; i++){
                    //     printVecGPU(Cm, Cn, d_C[g] + i * Cn * lddc, ldc, i);
                    // }
                    /* TODO
                    else
                    {
                        check_cublas_error(
                            cublasDtrsmBatched(kblasGetCublasHandle(kblas_handle[g]),
                                               CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_T, CUBLAS_DIAG_NON_UNIT,
                                               An, Cn,
                                               &alpha,
                                               d_A_array[g], ldda,
                                               d_C_array[g], lddc,
                                               batchCount_gpu));
                        // check_kblas_error(kblasXtrsm_batch(kblas_handle[g],
                        //                                    opts.side, opts.uplo, opts.transA, opts.diag,
                        //                                    M, N,
                        //                                    alpha, (const T **)(d_A_array[g]), ldda,
                        //                                    d_C_array[g], lddc,
                        //                                    batchCount_gpu));
                    }
                    */
                    printf("[info] Finished triangular solver. \n");
                    /*
                    Dot scalar Z_new^T Z_new
                    */
                    printf("[info] Starting dot product. \n");
                    if (strided)
                    {
                        for (int i = 0; i < batchCount_gpu; i++)
                        {
                            // printVecGPU(Cm, Cn, d_C[g] + i * Cn * lddc, ldc, i);
                            check_cublas_error(cublasXdot(kblasGetCublasHandle(kblas_handle[g]), Cm,
                                                          d_C[g] + i * Cn * lddc, 1,
                                                          d_C[g] + i * Cn * lddc, 1,
                                                          &(dot_result_h[g][i])));
                            // printf("Dot product is %lf \n", dot_result_h[g][i]);
                        }
                        // cublasDnrm2( kblasGetCublasHandle(kblas_handle[g]), Cm, d_C[g], 1,  dot_result_h[g]);
                    }
                    /* TODO
                    else if (nonUniform)
                    {
                        return 0;
                    }
                    else
                    {
                        // for (int i = 0; i < batchCount_gpu; i++)
                        // {
                        //     check_cublas_error(cublasXdot(kblasGetCublasHandle(kblas_handle[g]), Cm,
                        //                                   d_C_array[g] + i * Cn * ldc, 1,
                        //                                   d_C_array[g] + i * Cn * ldc, 1,
                        //                                   &(dot_result_h[g][i])));
                        // }
                        return 0;
                    }
                    */
                    printf("[info] Finished dot product. \n");
                }
                for (int g = 0; g < ngpu; g++)
                {
                    check_error(cudaSetDevice(opts.devices[g]));
                    cudaDeviceSynchronize(); // TODO sync with streams instead
                }
                // time = get_elapsed_time(curStream);
                clock_gettime(CLOCK_MONOTONIC, &end);
                kblas_time_1 = end.tv_sec - start.tv_sec + (end.tv_nsec - start.tv_nsec) / 1e9;

                cudaFreeHost(h_A);
                cudaFreeHost(h_C);
                if (nonUniform)
                {
                    free(h_M);
                    free(h_N);
                }

                if (opts.check || opts.time)
                    free(h_R);
                for (int g = 0; g < ngpu; g++)
                {
                    check_error(cudaSetDevice(opts.devices[g]));
                    check_error(cudaFree(d_A[g]));
                    check_error(cudaFree(d_C[g]));
                    if (!strided)
                    {
                        check_error(cudaFree(d_A_array[g]));
                        check_error(cudaFree(d_C_array[g]));
                    }
                    if (nonUniform)
                    {
                        check_error(cudaFree(d_M[g]));
                        check_error(cudaFree(d_N[g]));
                        check_error(cudaFree(d_ldda[g]));
                        check_error(cudaFree(d_lddc[g]));
                    }
                }

                printf("-----------------------------------------\n");

                for (int g = 0; g < ngpu; g++)
                {
                    for (int k = 0; k < batchCount; k++)
                    {
                        T llk_temp = 0;
                        llk_temp = -(dot_result_h[g][k] + logdet_result_h[g][k] + Am * log(2 * PI)) * 0.5;
                        llk += llk_temp;
                        // printf("%dth log determinant is % lf\n", k, logdet_result_h[g][k]);
                        // printf("%dth dot product is % lf\n", k, dot_result_h[g][k]);
                        // printf("%dth pi is % lf\n", k, Am * log(2 * PI));
                        // printf("%dth log likelihood is % lf\n", k, llk_temp);
                        // printf("-------------------------------------\n");
                    }
                }
                printf("(True) Sigma: %lf beta:  %lf  nu: %lf\n", opts.sigma, opts.beta, opts.nu);
                printf("Log likelihood is %lf \n", llk);
                printf("====================== Independent computing is finished! ====================== \n");
                printf("[Info] The time for independent computing is %lf seconds\n", kblas_time_1);
                printf("[Info] The time for LLH is %lf seconds\n", kblas_time_1 + vecchia_time);
                for (int g; g < ngpu; g++)
                {
                    free(dot_result_h[g]);
                    free(logdet_result_h[g]);
                }

#ifdef DEBUG_DUMP
                if (opts.check)
                {
                    fclose(outO);
                    fclose(outL);
                    fclose(outK);
                }
#endif
            }
        }
        if (opts.niter > 1)
        {
            printf("\n");
        }
    }

    for (int g = 0; g < ngpu; g++)
    {
        kblasDestroy(&kblas_handle[g]);
    }
#ifdef USE_MAGMA
    if (opts.magma == 1)
    {
        magma_finalize(); // TODO is this in the proper place?
    }
#endif
    return 0;
}

//==============================================================================================
int main(int argc, char **argv)
{

    kblas_opts opts;
    parse_opts(argc, argv, &opts);

#if defined PREC_d
    check_error(cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte));
#endif

#if (defined PREC_s) || (defined PREC_d)
    TYPE alpha = 1.;
#elif defined PREC_c
    TYPE alpha = make_cuFloatComplex(1.2, -0.6);
#elif defined PREC_z
    TYPE alpha = make_cuDoubleComplex(1.2, -0.6);
#endif
    test_Xvecchia_batch<TYPE>(opts, alpha);
}
