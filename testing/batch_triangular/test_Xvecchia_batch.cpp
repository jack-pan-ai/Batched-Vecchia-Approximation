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

void printLocations(int N, location *locations)
{
    printf("\n---------------------------------\n");
    for (int i = 0; i < N; i++)
    {
        printf("%d th location: (%lf, %lf)\n", i, locations->x[i], locations->y[i]);
    }
    printf("-----------------------------------\n");
}

template <class T>
void printMatrixCPU(int m, int n, T *h_A, int lda, int i)
{
    printf("-------------------------------\n");
    printf("%d batch of all. (CPU)\n", i);
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n; j++)
        {
            // colunm-wised matrix
            // printf("(%d)", i + j * lda);
            printf("%lf ", (double)h_A[i + j * lda]);
            printf(", ");
        }
        printf("\n");
    }
    printf("-------------------------------\n");
}

template <class T>
void printMatrixGPU(int Am, int An, T *d_A, int lda)
{
    printf("-------------------------------\n");
    printf("Convariance matrix in batch. (GPU)\n");
    T *h_A = (T *)malloc(sizeof(T) * An * lda);
    cudaMemcpy(h_A, d_A, sizeof(T) * An * lda, cudaMemcpyDeviceToHost);
    // double sum = 0;
    for (int i = 0; i < Am; i++)
    {
        for (int j = 0; j < An; j++)
        {
            // colunm-wised matrix
            printf("(%d)", i + j * lda);
            printf("%lf ", (double)h_A[i + j * lda]);
            // sum += (double)h_A[i + j * lda];
        }
        // printf("\n");
    }
    // printf("The sum is %lf \n", sum);
    printf("-------------------------------\n");
    free(h_A);
}

template <class T>
void printVecGPU(int Cm, int Cn, T *d_C, int lda)
{
    printf("-------------------------------\n");
    printf("1st batch of all. (GPU) vector\n");
    T *h_C = (T *)malloc(sizeof(T) * Cn * lda);
    cudaMemcpy(h_C, d_C, sizeof(T) * Cn * lda, cudaMemcpyDeviceToHost);
    for (int i = 0; i < Cm; i++)
    {
        printf("(%d)", i);
        printf("%lf ", (double)h_C[i]);
    }
    printf("\n-------------------------------\n");
    free(h_C);
}

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
    // no nugget
    T *localtheta;

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

    // printf("batchCount    N     K     kblasSYRK GF/s (ms)  lapackSYRK GF/s (ms)  SP      Error\n");
    // printf("==================================================================================\n");
    for (int itest = 0; itest < opts.ntest; ++itest)
    {
        for (int iter = 0; iter < opts.niter; ++iter)
        {
            for (int btest = 0; btest < opts.btest; ++btest)
            {

                double gflops, perf,
                    ref_avg_perf = 0.0, ref_sdev_perf = 0.0, ref_avg_time = 0.0,
                    // rec_avg_perf = 0.0, rec_sdev_perf = 0.0, rec_avg_time = 0.0,
                    kblas_perf = 0.0, kblas_time = 0.0, kblas_time_1 = 0.0, cublas_perf = 0.0, cublas_time = 0.0,
                    ref_error = 0.0;

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

                ldda = ((lda + 31) / 32) * 32;
                lddc = ((ldc + 31) / 32) * 32;

                sizeA = lda * An;
                sizeC = ldc * Cn;
                TESTING_MALLOC_PIN(h_A, T, lda * An * batchCount);
                TESTING_MALLOC_PIN(h_C, T, ldc * Cn * batchCount);

                for (int g = 0; g < ngpu; g++)
                {
                    check_error(cudaSetDevice(opts.devices[g]));
                    TESTING_MALLOC_DEV(d_A[g], T, ldda * An * batchCount_gpu);
                    TESTING_MALLOC_DEV(d_C[g], T, lddc * Cn * batchCount_gpu);
                    TESTING_MALLOC_DEV(d_info[g], int, batchCount_gpu);
                    TESTING_MALLOC_CPU(dot_result_h[g], T, batchCount_gpu);
                    TESTING_MALLOC_CPU(logdet_result_h[g], T, batchCount_gpu);

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
                }

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
                // Xrand_matrix(Am, An * batchCount, h_A, lda);
                /*
                matrix generation using kernel
                */
                // uniform random generation for locations / read locations from disk
                TESTING_MALLOC_CPU(localtheta, T, 3); // no nugget effect
                localtheta[0] = opts.sigma;
                localtheta[1] = opts.beta;
                localtheta[2] = opts.nu;
                for (int i = 0; i < batchCount; i++)
                {
                    locations = GenerateXYLoc(Am, seed[i]);
                    core_dcmg(h_A + i * An * lda, Am, An, locations, locations, localtheta, 0);
                    printLocations(M, locations);
                    printMatrixCPU(M, M, h_A+ i * An * lda, lda, i);
                }
                /*
                Xrand_matrix(Am, An * batchCount, h_A, lda);
                for (int i = 0; i < batchCount; i++)
                {
                    Xmatrix_make_hpd(Am, h_A + i * An * lda, lda);
                    // printMatrixCPU(M, M, h_A, lda, i);
                }
                */

                for (int i = 0; i < batchCount * Cm; i++)
                {
                    h_C[i] = 1;
                }
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
                }

                // 
                for (int g = 0; g < ngpu; g++)
                {
                    if (nonUniform)
                        kblas_trsm_batch_nonuniform_wsquery(kblas_handle[g]);
                    else if (strided)
                    {
                        kblas_potrf_batch_strided_wsquery(kblas_handle[g], M, batchCount_gpu);
                        kblas_trsm_batch_strided_wsquery(kblas_handle[g], opts.side, M, N, batchCount_gpu);
                    }
                    else
                    {
                        kblas_potrf_batch_wsquery(kblas_handle[g], M, batchCount_gpu);
                        kblas_trsm_batch_wsquery(kblas_handle[g], opts.side, M, N, batchCount_gpu);
                    }
                    check_kblas_error(kblasAllocateWorkspace(kblas_handle[g]));
                    check_error(cudaGetLastError());
                }

                double time = 0;

                for (int r = 0; r < nruns; r++)
                {
                    for (int g = 0; g < ngpu; g++)
                    {
                        check_error(cudaSetDevice(opts.devices[g]));
                        cudaDeviceSynchronize(); // TODO sync with streams instead
                    }
                    // start_timing(curStream);
                    time = -gettime();
                    for (int g = 0; g < ngpu; g++)
                    {
                        check_error(cudaSetDevice(opts.devices[g]));
                        /*
                        cholesky decomposition
                        */
                        // for (int i = 0; i < batchCount_gpu; i++)
                        // {
                        //     printf("%dth", i);
                        //     printMatrixGPU(Am, An, d_A[g] + i * Am * ldda, ldda);
                        // }
                        printf("[info] Starting Cholesky decomposition. \n");
                        if (strided)
                        {
                            check_kblas_error(kblas_potrf_batch(kblas_handle[g],
                                                                opts.uplo, Am,
                                                                d_A[g], ldda, An * ldda,
                                                                batchCount_gpu,
                                                                d_info[g]));
                        }
                        else
                        {
                            check_kblas_error(kblas_potrf_batch(kblas_handle[g],
                                                                opts.uplo, Am,
                                                                d_A_array[g], ldda,
                                                                batchCount_gpu,
                                                                d_info[g]));
                        }
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
                                // cudaDeviceSynchronize(); 
                            }
                            // printf("the det value is %lf \n", logdet_result_h[g][0]);
                        }
                        else
                        {
                            // for (int i = 0; i < batchCount_gpu; i++)
                            // {
                            //     core_Xdet<T>(d_A_array[g] + i * An * lda, An, lda, &(logdet_result_h[g][i]));
                            // }
                            return 0;
                        }
                        printf("[info] Finished determinant. \n");
                        /*
                        triangular solution: L^T Z_new <- Z_old
                        */
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
                        printf("[info] Finished triangular solver. \n");
                        // printVecGPU(Cm, Cn, d_C[0], ldda);
                        /*
                        Dot scalar Z_new^T Z_new
                        */
                        if (nonUniform)
                        {
                            return 0;
                        }
                        else if (strided)
                        {
                            for (int i = 0; i < batchCount_gpu; i++)
                            {
                                check_cublas_error(cublasXdot(kblasGetCublasHandle(kblas_handle[g]), Cm,
                                                              d_C[g] + i * Cn * lddc, 1,
                                                              d_C[g] + i * Cn * lddc, 1,
                                                              &(dot_result_h[g][i])));
                            }
                            // printf("Dot product is %lf \n", dot_result_h[g][0]);
                            // cublasDnrm2( kblasGetCublasHandle(kblas_handle[g]), Cm, d_C[g], 1,  dot_result_h[g]);
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
                        printf("[info] Finished dot product. \n");
                    }
                    for (int g = 0; g < ngpu; g++)
                    {
                        check_error(cudaSetDevice(opts.devices[g]));
                        cudaDeviceSynchronize(); // TODO sync with streams instead
                    }
                    // time = get_elapsed_time(curStream);
                    time += gettime();
                    kblas_time_1 += time;
                }
                kblas_time_1 /= nruns;
                kblas_perf = gflops / kblas_time_1;
                kblas_time_1 *= 1000.0;

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

                if (opts.time)
                {
                    ref_sdev_perf = sqrt((ref_sdev_perf - (ref_avg_perf * ref_avg_perf / nruns)) / nruns);
                    // rec_sdev_perf = sqrt((rec_sdev_perf - (rec_avg_perf * rec_avg_perf / nruns))/nruns);
                }

                // printf(" %7.4f %7.4f       %7.4f %7.4f %7.4f %7.4f    %7.4f %7.4f %7.4f %7.4f    %.4e \n",
                // printf(" %7.4f %7.4f %7.4f %7.4f %7.4f       %7.4f %7.4f %7.4f %7.4f    %.4e \n",
                //        kblas_perf, kblas_time, kblas_time_1,
                //        cublas_perf, cublas_time,
                //        ref_avg_perf / nruns, ref_avg_time, ref_sdev_perf, ref_avg_time / kblas_time_1,
                //        // rec_avg_perf / nruns, rec_avg_time, rec_sdev_perf, rec_avg_time / kblas_time,
                //        ref_error);
                printf("-----------------------------------------\n");

                for (int g = 0; g < ngpu; g++)
                {
                    for (int k = 0; k < batchCount; k++)
                    {
                        T llk_temp = 0;
                        llk_temp = -(dot_result_h[g][k] + logdet_result_h[g][k] + Am * log(2 * PI)) * 0.5;
                        llk += llk_temp;
                        printf("%dth log determinant is % lf\n", k, logdet_result_h[g][k]);
                        printf("%dth dot product is % lf\n", k, dot_result_h[g][k]);
                        printf("%dth pi is % lf\n", k, Am * log(2 * PI));
                        printf("%dth log likelihood is % lf\n", k, llk_temp);
                        printf("-------------------------------------\n");
                    }
                }
                printf("(True) Sigma: %lf beta:  %lf  nu: %lf\n", opts.sigma, opts.beta, opts.nu);
                printf("Log likelihood is %lf \n", llk);
                printf("-----------------------------------------\n");
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
