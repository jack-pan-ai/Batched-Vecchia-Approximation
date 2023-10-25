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
#include <typeinfo>


#if ((defined PREC_c) || (defined PREC_z)) && (defined USE_MKL)
// TODO need to handle MKL types properly
#undef USE_MKL
#endif

// KBLAS helper
#include "testing_helper.h"
#include "Xblas_core.ch"

// flops
#include "flops.h"

// Used for llh
#include "ckernel.h"
#include "misc/llg.h"
// used for vecchia
extern "C" {
    #include "misc/vecchia_helper_c.h"
}
// used for nearest neighbor 
#include "misc/nearest_neighbor.h"
// this is not formal statement and clarification, only for convenience
#include "misc/utils.h"
#include "misc/llh_Xvecchia_batch.h"


template <class T>
int test_Xvecchia_batch(kblas_opts &opts, T alpha)
{
    llh_data data;

    // preconfig
    bool strided = opts.strided;
    int ngpu = opts.ngpu;
    int nonUniform = opts.nonUniform;
    
    long long batchCount;

    // int nruns = opts.nruns;
    // BLAS language, 
    // A* stands for the covariance matrix 
    // C* stands for the observations
    int M, N;
    int Am, An, Cm, Cn;
    int lda, ldc, ldda, lddc;
    int ISEED[4] = {0, 0, 0, 1};
    // vecchia language
    // bs: block size; cs: conditioning size
    int bs, cs; 
    kblasHandle_t kblas_handle[ngpu];

    T *h_A, *h_C;
    T *h_C_data;
    T *d_C[ngpu];
    T *dot_result_h[ngpu];
    T *logdet_result_h[ngpu];
    long long *batchCount_gpu;
    //  potrf used
    int *d_info[ngpu];
    location *locations;
    location *locations_con;
    
    // // no nugget
    std::vector<T> localtheta_initial;
    // T *localtheta;
    // T *localtheta_initial;
    T *grad; // used for future gradient based optimization, please do not comment it

    // vecchia offset
    T *h_A_conditioning, *h_A_cross, *h_C_conditioning;
    // T *h_A_offset_matrix, *h_mu_offset_matrix; 
    T *h_A_offset_vector, *h_mu_offset_vector; 
    T *d_A_conditioning[ngpu], *d_A_cross[ngpu], *d_C_conditioning[ngpu];
    int ldacon, ldccon, Acon, Ccon;
    int lddacon, lddccon;
    // used for the store the memory of offsets for mu and sigma
    // or, you could say that this is correction term
    T *d_A_offset_vector[ngpu], *d_mu_offset_vector[ngpu];

    // time 
    double whole_time = 0;
    
    // if (ngpu > 1)
    //     opts.check = 0;
    if (nonUniform)
        strided = 0;

    // USING
    cudaError_t err;
    for (int g = 0; g < ngpu; g++)
    {
        err = cudaSetDevice(opts.devices[g]);
        kblasCreate(&kblas_handle[g]);
        check_error(cudaDeviceSynchronize());
        check_error(cudaGetLastError());
    }

    M = opts.msize[0];
    N = opts.nsize[0];
    data.M = M;
    data.N = N;

    fflush(stdout);

    // Vecchia config
    lda = Am = M;
    An = M;
    ldc = Cm = M;
    Cn = N;
    bs = M;
    
    if (cs > opts.num_loc){
        fprintf(stderr, "Warning: your conditioning size is larger than the number of location in total!\n");
        cs = ldacon = ldccon = Acon = Ccon = opts.num_loc;
    }else{
        cs = ldacon = ldccon = Acon = Ccon = opts.vecchia_cs;
    }

    // the batchCount is choosen to the largest
    batchCount = opts.num_loc - cs + 1;
    fprintf(stderr, "Your totol batch count: %d, and the GPUs tobe used %d.\n", batchCount, ngpu);
    if (batchCount % ngpu != 0){
        fprintf(stderr, "Warning: your data is not assigned to each gpu equally.\n");
    }
    
    TESTING_MALLOC_PIN(batchCount_gpu, int, ngpu);
    if (ngpu > 1){
        for (int g = 0; g < ngpu; g++){
            if (g == (ngpu - 1)){
                // last one contain the rests
                batchCount_gpu[g] = batchCount / ngpu + batchCount % ngpu;
            }else{
                // the rest has the same 
                batchCount_gpu[g] = batchCount / ngpu;
            }
            // fprintf(stderr, "batchCount_gpu[g]: %d \n", batchCount_gpu[g]);
        }
    }else{
        batchCount_gpu[0] = batchCount / ngpu;
    }
    

    // Vecchia config for strided access
    ldda = ((lda + 31) / 32) * 32;
    lddc = ((ldc + 31) / 32) * 32;
    lddccon = ((ldccon + 31) / 32) * 32;
    lddacon = lddccon;

    // batched log-likelihood
    TESTING_MALLOC_CPU(h_A, T, lda * An * batchCount); 
    // h_C_data: original data; h_C overwritten in each iteration
    TESTING_MALLOC_CPU(h_C, T, ldc * opts.num_loc); 
    TESTING_MALLOC_CPU(h_C_data, T, ldc * opts.num_loc); 
    if (opts.vecchia)
    {
        // used for vecchia offset
        TESTING_MALLOC_CPU(h_A_conditioning, T,  static_cast<long long> (ldacon) * Acon * batchCount);
        TESTING_MALLOC_CPU(h_A_cross, T, ldacon * An * batchCount);
        TESTING_MALLOC_CPU(h_C_conditioning, T, ldccon * Cn * batchCount);
        // extra memory for mu
        TESTING_MALLOC_CPU(h_A_offset_vector, T, batchCount);
        TESTING_MALLOC_CPU(h_mu_offset_vector, T, batchCount);
    }

    for (int g = 0; g < ngpu; g++)
    {
        check_error(cudaSetDevice(opts.devices[g]));
        TESTING_MALLOC_CPU(dot_result_h[g], T, batchCount_gpu[g]);
        TESTING_MALLOC_CPU(logdet_result_h[g], T, batchCount_gpu[g]);
        if (opts.vecchia)
        {
            // used for vecchia offset
            TESTING_MALLOC_DEV(d_A_conditioning[g], T, static_cast<long long> (lddacon) * Acon * batchCount_gpu[g]);
            TESTING_MALLOC_DEV(d_A_cross[g], T, lddacon * An * batchCount_gpu[g]);
            TESTING_MALLOC_DEV(d_C_conditioning[g], T, lddccon * Cn * batchCount_gpu[g]);
            TESTING_MALLOC_DEV(d_mu_offset_vector[g], T, batchCount_gpu[g]);
            TESTING_MALLOC_DEV(d_A_offset_vector[g], T, batchCount_gpu[g]);
        }
        TESTING_MALLOC_DEV(d_C[g], T, lddc * Cn * batchCount_gpu[g]);
        TESTING_MALLOC_DEV(d_info[g], int, ngpu);
        check_error(cudaDeviceSynchronize());
        check_error(cudaGetLastError());
    }
    // fprintf(stderr, "%d %d %d", lddacon, Acon,  batchCount_gpu[0]);

    createLogFileParams(opts.num_loc, M, opts.zvecs, opts.p, cs);
    /* 
    Dataset: defined by yourself 
    */
    // Uniform random generation for locations / read locations from disk
    // random generate with seed as 1
    if (opts.perf == 1){
        locations = GenerateXYLoc(opts.num_loc / opts.p, 1); // 1 is the random seed
        // for(int i = 0; i < opts.num_loc; i++) h_C_data[i] = (T) rand()/(T)RAND_MAX;
        for(int i = 0; i < opts.num_loc; i++) h_C_data[i] = 3.0;
        // Xrand_matrix(Cm, Cn * batchCount, h_C, ldc);
        // printLocations(opts.num_loc, locations);
        // printVectorCPU(opts.num_loc, h_C, 1, 1);
        // for(int i = 0; i < Cm * batchCount; i++) printf("%ith %lf \n",i, h_C[i]);
        // // the optimization initial values
        localtheta_initial = {1.0, 0.1, 0.5};
    }else{
        // univariate case
        // synthetic dataset (umcomment it if used)
        if (opts.p == 1){
            std::string xy_path = "./data/synthetic_ds/LOC_" + std::to_string(opts.num_loc) + "_univariate_matern_stationary_" \
                    + std::to_string(opts.zvecs);
            std::string z_path = "./data/synthetic_ds/Z1_" + std::to_string(opts.num_loc) + "_univariate_matern_stationary_" \
                        + std::to_string(opts.zvecs);
            locations = loadXYcsv(xy_path, int(opts.num_loc/opts.p)); 
            loadObscsv<T>(z_path, int(opts.num_loc/opts.p), h_C_data);
        }
        if (opts.p == 2){
            // you have to preprocess the data
            std::string xy_path = "./data/synthetic_ds/LOC_" + std::to_string(int(opts.num_loc/opts.p)) + \
                                "_bivariate_matern_parsimonious_" + std::to_string(opts.zvecs);
            std::string z_path = "./data/synthetic_ds/Z_" + std::to_string(int(opts.num_loc/opts.p)) + \
                                    "_bivariate_matern_parsimonious_" + std::to_string(opts.zvecs);
            locations = loadXYcsv(xy_path, int(opts.num_loc/opts.p)); 
            loadObscsv<T>(z_path, int(opts.num_loc/opts.p), h_C_data);
        }
        // the optimization initial values
        // localtheta_initial(opts.num_params, opts.lower_bound);
        localtheta_initial = {0.01, 0.01, 0.01};
        // for (int i = 0; i < opts.num_params; i++) {
        //     localtheta_initial[i] = opts.lower_bound;
        // }
        data.distance_metric = 0;
    }
    if (opts.randomordering==1){
        random_locations(int(opts.num_loc/opts.p), locations, h_C_data);
    }
    // if (opts.morton==1){
    //     zsort_locations(opts.num_loc, locations, h_C_data);
    // }
    
    // std::string xy_path = "./data/synthetic_ds/LOC_" + std::to_string(opts.num_loc) \
    //             + "_" + std::to_string(opts.zvecs);
    // std::string z_path = "./data/synthetic_ds/Z_" + std::to_string(opts.num_loc) \
    //             + "_" + std::to_string(opts.zvecs);
    //// real dataset soil (umcomment it if used)
    // std::string xy_path = "./data/soil_moisture/R" + std::to_string(opts.zvecs) + \
    //                         "/METAinfo";
    // std::string z_path = "./data/soil_moisture/R" + std::to_string(opts.zvecs) + \
    //                         "/ppt.complete.Y001";
    // data.distance_metric = 1;
    // locations = loadXYcsv(xy_path, opts.num_loc); 
    // loadObscsv<T>(z_path, opts.num_loc, h_C_data);

    /*
    Locations preparation
    */


    // printLocations(opts.num_loc, locations);
    // printLocations(batchCount * lda, locations);
    memcpy(h_C, h_C_data, sizeof(T) * opts.num_loc);
    if (opts.vecchia){
        // Following can be deleted with adding nearest neighbors
        // init for each iteration (necessary but low efficient)
        // locations_copy = (location*) malloc(sizeof(location));
        // data.locations_copy = locations_copy;          
        locations_con = (location*) malloc(sizeof(location));
        locations_con->x = (T* ) malloc((batchCount - 1) * cs / opts.p * sizeof(double));
        locations_con->y = (T* ) malloc((batchCount - 1)* cs / opts.p * sizeof(double));
        locations_con->z = NULL;
        data.locations_con = locations_con;
        // copy for the first independent block
        memcpy(h_C_conditioning, h_C_data, sizeof(T) * cs);
        if (opts.knn){
            #pragma omp parallel for
            for (int i = 0; i < (batchCount - 1); i++){
                // how many previous points you would like to include in your nearest neighbor searching
                // int con_loc = std::max(i * bs - 10000 * bs, 0);
                int con_loc = 0;
                findNearestPoints(
                    h_C_conditioning, h_C_data, locations_con, locations,
                    con_loc , cs + i * bs, 
                    cs + (i + 1) * bs, cs, i
                );
                // printLocations(opts.num_loc, locations);
                // printLocations(cs * i, locations_con);
                // fprintf(stderr, "asdasda\n");
            }
        }else{
            #pragma omp parallel for
            for (int i = 0; i < (batchCount - 1); i++){
                memcpy(locations_con->x + i * cs, locations->x + i * bs, sizeof(T) * cs);
                memcpy(locations_con->y + i * cs, locations->y + i * bs, sizeof(T) * cs);
                memcpy(h_C_conditioning + (i + 1) * cs, h_C + i * bs, sizeof(T) * cs);
            }
        }
    }
    // printLocations(cs * batchCount, locations_con);

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
    data.batchCount = batchCount;
    data.bs = bs;
    data.cs = cs;
    data.h_A = h_A;
    data.h_C = h_C;
    data.h_C_data = h_C_data;
    // // no nugget
    data.locations = locations;
    // vecchia offset
    data.h_A_conditioning = h_A_conditioning;
    data.h_A_cross = h_A_cross;
    data.h_C_conditioning = h_C_conditioning;
    data.h_A_offset_vector = h_A_offset_vector;
    data.h_mu_offset_vector = h_mu_offset_vector;

    // opts
    data.sigma = opts.sigma;
    data.beta = opts.beta;
    data.nu = opts.nu;
    data.vecchia = opts.vecchia;
    data.iterations = 0;
    data.omp_threads = opts.omp_numthreads;

    data.num_loc = opts.num_loc;
    // kernel related
    data.kernel = opts.kernel;
    data.num_params = opts.num_params;
    data.zvecs = opts.zvecs;
    data.vecchia_time_total = 0; // used for accumulatet the time on vecchia
    data.p = opts.p; //bivariate = 2 or univariate = 1
    data.perf = opts.perf;
    
    for (int g=0; g < ngpu; g++)
    {
        data.kblas_handle[g] = &(kblas_handle[g]);

        data.d_C[g] = d_C[g];
        data.dot_result_h[g] = dot_result_h[g];
        data.logdet_result_h[g] = logdet_result_h[g];
        data.d_info[g] = d_info[g];
        data.d_A_conditioning[g] = d_A_conditioning[g];
        data.d_A_cross[g] = d_A_cross[g];
        data.d_C_conditioning[g] = d_C_conditioning[g];
        data.d_A_offset_vector[g] = d_A_offset_vector[g];
        data.d_mu_offset_vector[g] = d_mu_offset_vector[g];
        data.devices[g] = opts.devices[g];
        data.batchCount_gpu[g] = batchCount_gpu[g];
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
    // opt.set_maxeval(1);
    opt.set_max_objective(llh_Xvecchia_batch, &data); // Pass a pointer to the data structure
    // Optimize the log likelihood
    T maxf;
    
    try {
        // Cautious for future develop
        // Perform the optimization
        nlopt::result result = opt.optimize(localtheta_initial, maxf);
    } catch (const std::exception& e) {
        // Handle any other exceptions that may occur during optimization
        std::cerr << "Exception caught: " << e.what() << std::endl;
        // ...
    }

    double max_llh = opt.last_optimum_value();
    int num_iterations = opt.get_numevals();

    clock_gettime(CLOCK_MONOTONIC, &end_whole);
    whole_time = end_whole.tv_sec - start_whole.tv_sec + (end_whole.tv_nsec - start_whole.tv_nsec) / 1e9;
    saveLogFileSum(num_iterations, localtheta_initial, 
                max_llh, whole_time, 
                M, opts.num_loc, opts.zvecs, cs);
    // int num_evals = 0;
    // num_evals = opt.get_numevals();
    printf("Done! \n");

    // vecchia
    cudaFreeHost(h_A_conditioning);
    cudaFreeHost(h_A_cross);
    cudaFreeHost(h_C_conditioning);
    cudaFreeHost(locations->x);
    cudaFreeHost(locations->y);
    if (opts.vecchia){
        // init for each iteration (necessary but low efficient)
        // cudaFreeHost(locations_copy);
        // for (int i=0; i < batchCount; i ++){
        //     cudaFreeHost(locations_con[i]);
        // }
        cudaFreeHost(locations_con->x);
        cudaFreeHost(locations_con->y);
    }
    for (int g = 0; g < ngpu; g++)
    {
        check_error(cudaFree(d_A_conditioning[g]));
        check_error(cudaFree(d_A_cross[g]));
        check_error(cudaFree(d_C_conditioning[g]));
        check_error(cudaFree(d_A_offset_vector[g]));
        check_error(cudaFree(d_mu_offset_vector[g]));
    }
    // independent
    cudaFreeHost(h_A);
    cudaFreeHost(h_C);
    for (int g = 0; g < ngpu; g++)
    {
        check_error(cudaSetDevice(opts.devices[g]));
        check_error(cudaFree(d_C[g]));
    }

    for (int g=0; g < ngpu; g++)
    {
        free(dot_result_h[g]);
        free(logdet_result_h[g]);
    }

    for (int g = 0; g < ngpu; g++)
    {
        kblasDestroy(&kblas_handle[g]);
    }
    return 0;
}

//==============================================================================================
int main(int argc, char **argv)
{

    kblas_opts opts;
    parse_opts(argc, argv, &opts);

// #if defined PREC_d
//     check_error(cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte));
// #endif

// #if (defined PREC_s) || (defined PREC_d)
    // TYPE alpha = 1.;
// #elif defined PREC_c
//     TYPE alpha = make_cuFloatComplex(1.2, -0.6);
// #elif defined PREC_z
//     TYPE alpha = make_cuDoubleComplex(1.2, -0.6);
// #endif
    double alpha = 1.;
    gsl_set_error_handler_off();
    return test_Xvecchia_batch<double>(opts, alpha);
}
