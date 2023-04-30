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
#include <nlopt.hpp>
#include <vector>
#include <gsl/gsl_errno.h>


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
// used for vecchia
extern "C" {
    #include "misc/vecchia_helper_c.h"
}

// this is not formal statement and clarification, only for convenience
#include "misc/utils.h"
#include "misc/llh_Xvecchia_batch.h"

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
    llh_data data;

    // preconfig
    bool strided = opts.strided;
    int ngpu = opts.ngpu;
    int nonUniform = opts.nonUniform;

    // int nruns = opts.nruns;
    int M, N;
    int Am, An, Cm, Cn;
    int lda, ldc, ldda, lddc;
    int ISEED[4] = {0, 0, 0, 1};
    // TBD for non uniform
    // int max_M, max_N;
    // int *h_M, *h_N,
    //     *d_M[ngpu], *d_N[ngpu];
    // int seed = 0;
    kblasHandle_t kblas_handle[ngpu];

    T *h_A, *h_C, *h_R;
    T *d_A[ngpu], *d_C[ngpu];
    T **d_A_array[ngpu], **d_C_array[ngpu];
    // int *d_ldda[ngpu], *d_lddc[ngpu];
    T *dot_result_h[ngpu];
    T *logdet_result_h[ngpu];
    //  potrf used
    int *d_info[ngpu];
    location *locations;
    // location *locations_copy;
    location *locations_con_boundary;
    location *locations_con[opts.batchCount];
    // // no nugget
    // T *localtheta;
    // T *localtheta_initial;
    T *grad; // used for future gradient based optimization, please do not comment it

    // vecchia offset
    T *h_A_copy, *h_A_conditioned, *h_C_conditioned;
    T *d_A_copy[ngpu], *d_A_conditioned[ngpu], *d_C_conditioned[ngpu];
    int ldacon, ldccon, Acon, Ccon;
    int lddacon, lddccon;
    // used for the store the memory of offsets for mu and sigma
    T *d_A_offset[ngpu], *d_mu_offset[ngpu];
    T *d_C_copy[ngpu];

    // time 
    double whole_time = 0;
    
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

                int batchCount = opts.batchCount; 
                
                // if (opts.btest > 1)
                //     batchCount = opts.batch[btest];

                int batchCount_gpu = batchCount / ngpu;

                // /*
                // seed for location generation
                // */
                // int seed[batchCount];
                // for (int i = 0; i < batchCount; i++)
                // {
                //     seed[i] = i + 1;
                // }

                M = opts.msize[itest];
                N = opts.nsize[itest];
                data.M = M;
                data.N = N;

                fflush(stdout);

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

                
                createLogFileParams(opts.num_loc, M, opts.zvecs, opts.p);
                /* 
                Dataset: defined by yourself 
                */
                // Uniform random generation for locations / read locations from disk
                // random generate with seed as 1
                locations = GenerateXYLoc(batchCount * lda / opts.p, 1);
                // Xrand_matrix(Cm, Cn * batchCount, h_C, ldc);
                for(int i = 0; i < Cm * batchCount; i++) h_C[i] = 1.;
                // printLocations(opts.num_loc, locations);
                // for(int i = 0; i < Cm * batchCount; i++) printf("%ith %lf \n",i, h_C[i]);
                // univariate case
                // synthetic dataset (umcomment it if used)
                // std::string xy_path = "./data/synthetic_ds/LOC_" + std::to_string(opts.num_loc) + "_univariate_matern_stationary_" \
                //             + std::to_string(opts.zvecs);
                // std::string z_path = "./data/synthetic_ds/Z1_" + std::to_string(opts.num_loc) + "_univariate_matern_stationary_" \
                //             + std::to_string(opts.zvecs);
                // std::string xy_path = "./data/synthetic_ds/LOC_" + std::to_string(opts.num_loc) \
                //             + std::to_string(opts.zvecs);
                // std::string z_path = "./data/synthetic_ds/Z1_" + std::to_string(opts.num_loc) \
                //             + std::to_string(opts.zvecs);
                // locations = loadXYcsv(xy_path, int(opts.num_loc/opts.p)); 
                // loadObscsv<T>(z_path, int(opts.num_loc/opts.p), h_C);
                // data.distance_metric = 0;
                //// real dataset soil (umcomment it if used)
                // std::string xy_path = "./data/soil_moisture/R" + std::to_string(opts.zvecs) + \
                //                         "/METAinfo";
                // std::string z_path = "./data/soil_moisture/R" + std::to_string(opts.zvecs) + \
                //                         "/ppt.complete.Y001";
                // data.distance_metric = 1;
                // locations = loadXYcsv(xy_path, opts.num_loc); 
                // loadObscsv<T>(z_path, opts.num_loc, h_C);

                // bivaraite case (umcomment it if used)
                // std::string xy_path = "./data/synthetic_ds/LOC_" + std::to_string(int(opts.num_loc/opts.p)) + \
                //                         "_bivariate_matern_parsimonious_" + std::to_string(opts.zvecs);
                // std::string z_path = "./data/synthetic_ds/Z_" + std::to_string(int(opts.num_loc/opts.p)) + \
                //                         "_bivariate_matern_parsimonious_" + std::to_string(opts.zvecs);
                // locations = loadXYcsv(xy_path, int(opts.num_loc/opts.p)); 
                // loadObscsv<T>(z_path, int(opts.num_loc/opts.p), h_C);
                // data.distance_metric = 0;
                /*
                Dataset: defined by yourself 
                */

                if (batchCount * M != opts.num_loc) {
                    // printf("batchCount: ");
                    fprintf(stderr, "Error: batchCount * lda %d is not equal to %d\n", batchCount * M, opts.num_loc);
                    exit(0); // Exit the program with a non-zero status to indicate an error
                }  
                // printLocations(opts.num_loc, locations);
                // printLocations(batchCount * lda, locations);
                if (opts.vecchia){
                    // init for each iteration (necessary but low efficient)
                    // locations_copy = (location*) malloc(sizeof(location));
                    // data.locations_copy = locations_copy;
                    for (int i=0; i < batchCount; i ++){
                        locations_con[i] = (location*) malloc(sizeof(location));
                        data.locations_con[i] = locations_con[i];
                    }                
                }
                // // true parameter
                // TESTING_MALLOC_CPU(localtheta, T, opts.num_params); // no nugget effect
                // localtheta[0] = opts.sigma;
                // localtheta[1] = opts.beta;
                // localtheta[2] = opts.nu;
                // TESTING_MALLOC_CPU(localtheta_initial, T, opts.num_params); // no nugget effect
                // localtheta_initial[0] = 0.01;
                // localtheta_initial[1] = 0.01;
                // localtheta_initial[2] = 0.01;

                // prepare these for llh_Xvecchia_batch
                data.M = M;
                data.N = N;
                data.strided = opts.strided;
                data.ngpu = opts.ngpu;
                data.nonUniform = opts.nonUniform;
                data.Am = Am;
                data.An = An;
                data.Cm = Cm;
                data.Cn = Cn;
                data.lda = lda;
                data.ldc = ldc;
                data.ldda = ldda;
                data.lddc = lddc;
                data.ldacon = ldacon;
                data.ldccon = ldccon;
                data.Acon = Acon;
                data.Ccon = Ccon;
                data.lddacon = lddacon;
                data.lddccon =  lddccon;
                data.batchCount_gpu = batchCount_gpu;
                data.batchCount = batchCount;
                // int *h_M, *h_N,
                //     *d_M[ngpu], *d_N[ngpu];
                data.h_A = h_A;
                data.h_C = h_C;
                // // no nugget
                // data.localtheta = localtheta;
                data.locations = locations;
                data.locations_con_boundary = locations_con_boundary;
                // vecchia offset
                data.h_A_copy = h_A_copy;
                data.h_A_conditioned = h_A_conditioned;
                data.h_C_conditioned = h_C_conditioned;
 
                // opts
                // lapack flags
                data.uplo = opts.uplo;
                data.transA = opts.transA;
                data.transB = opts.transB;
                data.side = opts.side;
                data.diag = opts.diag;
                data.sigma = opts.sigma;
                data.beta = opts.beta;
                data.nu = opts.nu;
                data.vecchia = opts.vecchia;
                data.vecchia_num = opts.vecchia_num;
                data.iterations = 0;
                data.omp_threads = opts.omp_numthreads;

                data.num_loc = opts.num_loc;
                // kernel related
                data.kernel = opts.kernel;
                data.num_params = opts.num_params;
                data.zvecs = opts.zvecs;
                data.vecchia_time_total = 0; // used for accumulatet the time on vecchia
                data.p = opts.p; //bivariate = 2 or univariate = 1
                
                for (int i=0; i < ngpu; i++)
                {
                    data.kblas_handle[i] = &(kblas_handle[i]);
                    data.d_A[i] = d_A[i];
                    data.d_C[i] = d_C[i];
                    data.d_A_array[i] = d_A_array[i];
                    data.d_C_array[i] = d_C_array[i];
                    // data.d_ldda[i] = d_ldda[i];
                    // data.d_lddc[i] = d_lddc[i];
                    data.dot_result_h[i] = dot_result_h[i];
                    data.logdet_result_h[i] = logdet_result_h[i];
                    data.d_info[i] = d_info[i];
                    data.d_A_copy[i] = d_A_copy[i];
                    data.d_A_conditioned[i] = d_A_conditioned[i];
                    data.d_C_conditioned[i] = d_C_conditioned[i];
                    data.d_A_offset[i] = d_A_offset[i];
                    data.d_mu_offset[i] = d_mu_offset[i];
                    data.d_C_copy[i] = d_C_copy[i];
                    data.devices[i] = opts.devices[i];
                }

                struct timespec start_whole, end_whole;
                clock_gettime(CLOCK_MONOTONIC, &start_whole);
                // Set up the optimization problem
                nlopt::opt opt(nlopt::LN_BOBYQA, opts.num_params); // Use the BOBYQA algorithm in 2 dimensions
                std::vector<T> lb(opts.num_params, opts.lower_bound);
                std::vector<T> ub(opts.num_params, opts.upper_bound); 
                if (opts.kernel == 2){ // bivariate matern kernel 
                    ub.back() = 1. ;// beta should be constrained somehow
                }
                opt.set_lower_bounds(lb);
                opt.set_upper_bounds(ub);
                opt.set_ftol_rel(opts.tol);
                opt.set_maxeval(opts.maxiter);
                opt.set_max_objective(llh_Xvecchia_batch, &data); // Pass a pointer to the data structure
                // Set the initial guess from lower bound
                // std::vector<T> localtheta_initial = {0.481, 0.10434, 0.500};
                std::vector<T> localtheta_initial(opts.num_params, opts.lower_bound);
                // Optimize the log likelihood
                T maxf;
                nlopt::result result = opt.optimize(localtheta_initial, maxf);
                double max_llh = opt.last_optimum_value();
                int num_iterations = opt.get_numevals();

                clock_gettime(CLOCK_MONOTONIC, &end_whole);
                whole_time = end_whole.tv_sec - start_whole.tv_sec + (end_whole.tv_nsec - start_whole.tv_nsec) / 1e9;
                saveLogFileSum(num_iterations, localtheta_initial, 
                                max_llh, /* whole_time */data.vecchia_time_total, 
                                M, opts.num_loc, opts.zvecs);
                // int num_evals = 0;
                // num_evals = opt.get_numevals();
                printf("Done! \n");

                // vecchia
                cudaFreeHost(h_A_copy);
                cudaFreeHost(h_A_conditioned);
                cudaFreeHost(h_C_conditioned);
                if (opts.vecchia){
                    // init for each iteration (necessary but low efficient)
                    // cudaFreeHost(locations_copy);
                    for (int i=0; i < batchCount; i ++){
                        cudaFreeHost(locations_con[i]);
                    }                    
                }
                for (int g = 0; g < ngpu; g++)
                {
                    check_error(cudaFree(d_A_copy[g]));
                    check_error(cudaFree(d_A_conditioned[g]));
                    check_error(cudaFree(d_C_conditioned[g]));
                    check_error(cudaFree(d_A_offset[g]));
                    check_error(cudaFree(d_mu_offset[g]));
                }
                // independent
                cudaFreeHost(h_A);
                cudaFreeHost(h_C);
                // if (nonUniform)
                // {
                //     free(h_M);
                //     free(h_N);
                // }
                if (opts.check || opts.time)
                    free(h_R);
                for (int g = 0; g < ngpu; g++)
                {
                    check_error(cudaSetDevice(opts.devices[g]));
                    check_error(cudaFree(d_A[g]));
                    check_error(cudaFree(d_C[g]));
                    // if (!strided)
                    // {
                    //     check_error(cudaFree(d_A_array[g]));
                    //     check_error(cudaFree(d_C_array[g]));
                    // }
                    // if (nonUniform)
                    // {
                    //     check_error(cudaFree(d_M[g]));
                    //     check_error(cudaFree(d_N[g]));
                    //     check_error(cudaFree(d_ldda[g]));
                    //     check_error(cudaFree(d_lddc[g]));
                    // }
                }

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
    gsl_set_error_handler_off();
    return test_Xvecchia_batch<TYPE>(opts, alpha);
}
