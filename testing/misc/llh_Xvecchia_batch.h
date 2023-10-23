#ifndef LLH_XVECCHIA_BATCH_H
#define LLH_XVECCHIA_BATCH_H

#include <omp.h>

#include "../aux_operations.h"

template <class T>

T llh_Xvecchia_batch(unsigned n, const T* localtheta, T* grad, void* f_data)
{   
    T llk = 0;
    llh_data * data = static_cast<llh_data *>(f_data);
    double gflops_batch_potrf = 0.0, gflops_batch_trsm = 0.0, gflops_quadratic = 0.0;
    double indep_time = 0.0, dcmg_time = 0.0, vecchia_time_batch_potrf = 0.0, vecchia_time_batch_trsm = 0.0, vecchia_time_quadratic = 0.0;
    double time_copy = 0.0, time_copy_hd = 0.0, time_copy_dh = 0.0,vecchia_time_total = 0.0;
    double alpha_1 = 1.;
    double beta_n1 = -1.;
    double beta_0 =0.;
    int omp_threads = data->omp_threads;
    omp_set_num_threads(omp_threads);

    // printf("[info] Starting Covariance Generation. \n");
    struct timespec start_dcmg, end_dcmg;
    clock_gettime(CLOCK_MONOTONIC, &start_dcmg);

    memcpy(data->h_C, data->h_C_data, sizeof(T) * data->num_loc);

    // covariance matrix generation 
    #pragma omp parallel for
    for (int i=0; i < data->batchCount; i++)
    {
        // loc_batch: for example, p(y1|y2), the locations of y1 is the loc_batch
        location* loc_batch= (location *) malloc(sizeof (location));
        // h_A: \sigma_{11}
        if (i == 0){
            // the first independent block
            loc_batch->x = data->locations->x;
            loc_batch->y = data->locations->y;
            loc_batch->z = NULL;  
            // printLocations(data->cs, loc_batch); 
            core_dcmg(data->h_A_conditioning,
                        data->cs, data->cs,
                        loc_batch, loc_batch, 
                        localtheta, data->distance_metric);  
            // printVectorCPU(data->bs, data->h_C, data->ldc, 0);
            memcpy(data->h_C_conditioning, data->h_C, sizeof(T) * data->cs);
            // the first cs data->h_A_cross has to be treated carefully (replace with h_C_conditioning)
            // because we need its quadratic term
            memcpy(data->h_A_cross, data->h_C, sizeof(T) * data->cs);
            // note that we need copy the first block oberservation;
            // then h_C needs to point the location after the first block size;
            data->h_C = data->h_C + data->cs - 1;
            // printVectorCPU(data->bs, data->h_C, data->ldc, 0);
        }else{
            // the rest batched block
            loc_batch->x = data->locations->x + data->cs + i * data->bs;
            loc_batch->y = data->locations->y + data->cs + i * data->bs;
            loc_batch->z = NULL;  
            // printLocations(data->bs, loc_batch); 
            core_dcmg(data->h_A + i * data->bs * data->bs,
                        data->bs, data->bs,
                        loc_batch, loc_batch, 
                        localtheta, data->distance_metric);  
        }
        free(loc_batch);
    }
        #pragma omp parallel for
        for (long long i=0; i < (data->batchCount - 1); i++)
        {
            // for example, p(y1|y2), the locations of y2 is the loc_batch_con
            location* loc_batch_con= (location *) malloc(sizeof (location));
            location* loc_batch= (location *) malloc(sizeof (location));
            loc_batch_con->x = data->locations_con->x + i * data->cs;
            loc_batch_con->y = data->locations_con->y + i * data->cs;
            loc_batch_con->z = NULL;
            loc_batch->x = data->locations->x + data->cs + i * data->bs;
            loc_batch->y = data->locations->y + data->cs + i * data->bs;
            loc_batch->z = NULL;   
            // note that the h_A_conditioning[0] h_A_cross[0] is for first independent block
            //*h_A_conditioning: \sigma_{22}
            core_dcmg(data->h_A_conditioning + (i + 1) * data->cs * data->cs,
                        data->cs, data->cs,
                        loc_batch_con,
                        loc_batch_con, localtheta, data->distance_metric);
            // *h_A_cross: \sigma_{21}
            core_dcmg(data->h_A_cross + (i + 1) * data->cs * data->bs,
                        data->cs, data->bs,
                        loc_batch_con,
                        loc_batch, localtheta, data->distance_metric);
            free(loc_batch);
            free(loc_batch_con);
        }
    // }
    // printVectorCPU(data->bs, data->h_C, data->ldc, 0);
    clock_gettime(CLOCK_MONOTONIC, &end_dcmg);
    dcmg_time = end_dcmg.tv_sec - start_dcmg.tv_sec + (end_dcmg.tv_nsec - start_dcmg.tv_nsec) / 1e9;
    // printf("[info] Covariance Generation with time %lf seconds. \n", dcmg_time);
    // printMatrixCPU(data->M, data->M, data->h_A, data->lda, i);
    // covariance matrix copt from host to device
    check_error(cudaGetLastError());
    // exit(0);

    // timing
    struct timespec start_copy_hd, end_copy_hd;
    clock_gettime(CLOCK_MONOTONIC, &start_copy_hd);
    for (int g = 0; g < data->ngpu; g++)
    {
        check_error(cudaSetDevice(data->devices[g]));
        if (g==0){
            check_cublas_error(cublasSetMatrixAsync(
                data->Cm, data->Cn * data->batchCount_gpu[g], sizeof(T),
                data->h_C, data->ldc,
                data->d_C[g], data->lddc, 
                kblasGetStream(*(data->kblas_handle[g])))
            );
        }else{
            check_cublas_error(cublasSetMatrixAsync(
                data->Cm, data->Cn * data->batchCount_gpu[g], sizeof(T),
                data->h_C + data->Cm * data->Cn * data->batchCount_gpu[g-1], data->ldc,
                data->d_C[g], data->lddc, 
                kblasGetStream(*(data->kblas_handle[g])))
            );
        }
        check_error(cudaDeviceSynchronize());
        check_error(cudaGetLastError());
    }
    long long batch22count = 0;
    int batch21count = 0;
    int z2count = 0;

    for (int g = 0; g < data->ngpu; g++)
    {
        check_error(cudaSetDevice(data->devices[g]));
        // sigma_{22}
        check_cublas_error(cublasSetMatrixAsync(data->cs, data->cs * data->batchCount_gpu[g], sizeof(T),
                                            data->h_A_conditioning + batch22count,
                                            data->ldacon, 
                                            data->d_A_conditioning[g], 
                                            data->lddacon, 
                                            kblasGetStream(*(data->kblas_handle[g]))));
        batch22count += static_cast<long long> (data->cs) * data->cs * data->batchCount_gpu[g];
        // sigma{21}
        check_cublas_error(cublasSetMatrixAsync(data->cs, data->bs * data->batchCount_gpu[g], sizeof(T),
                                            data->h_A_cross + batch21count, 
                                            data->ldacon, 
                                            data->d_A_cross[g], 
                                            data->lddacon, 
                                            kblasGetStream(*(data->kblas_handle[g]))));
        batch21count += data->cs * data->bs * data->batchCount_gpu[g] ;
        // z_(2)
        check_cublas_error(cublasSetMatrixAsync(data->cs, data->Cn * data->batchCount_gpu[g], sizeof(T),
                                            data->h_C_conditioning + z2count,
                                            data->ldccon, 
                                            data->d_C_conditioning[g], 
                                            data->lddccon, 
                                            kblasGetStream(*(data->kblas_handle[g]))));
        z2count += data->cs * data->Cn * data->batchCount_gpu[g];
        check_error(cudaDeviceSynchronize()); // TODO sync with streams instead
        check_error(cudaGetLastError());
    }
    clock_gettime(CLOCK_MONOTONIC, &end_copy_hd);
    time_copy_hd = end_copy_hd.tv_sec - start_copy_hd.tv_sec + (end_copy_hd.tv_nsec - start_copy_hd.tv_nsec) / 1e9;

    // conditioning part 1.2, wsquery for potrf and trsm 
    for (int g = 0; g < data->ngpu; g++)
    {
        check_error(cudaSetDevice(data->devices[g]));
        if (data->strided)
        {   
            // data->Acon*2 instead of data->Acon is because for some cases, 
            // like 320/20 combination, 320 batchszie 20 batch count, cannot 
            // be allocated enough memory. But the reason is unclear now.
            kblas_potrf_batch_strided_wsquery(*(data->kblas_handle[g]), data->cs, data->batchCount_gpu[g]);
            kblas_trsm_batch_strided_wsquery(*(data->kblas_handle[g]), 'L', data->cs, data->N, data->batchCount_gpu[g]);
            kblas_trsm_batch_strided_wsquery(*(data->kblas_handle[g]), 'L', data->cs, data->N, data->batchCount_gpu[g]);
        }
        check_kblas_error(kblasAllocateWorkspace(*(data->kblas_handle[g])));
        check_error(cudaDeviceSynchronize()); // TODO sync with streams instead
        check_error(cudaGetLastError());
    }

    gflops_batch_potrf = data->batchCount * FLOPS_POTRF<T>(data->cs) / 1e9;
    gflops_batch_trsm = 2*data->batchCount * FLOPS_TRSM<T>('L', data->lddacon, data->An) / 1e9;
    gflops_quadratic = 2 * data->batchCount * FLOPS_DOTPRODUCT<T>(data->cs) / 1e9;
    // gflops_quadratic = 2*FLOPS_GEMM_v1<T>(data->batchCount, data->batchCount, data->cs) / 1e9;

    /*----------------------------*/
    /* correction terms */
    /*----------------------------*/

    struct timespec start_batch_potrf, end_batch_potrf;
    clock_gettime(CLOCK_MONOTONIC, &start_batch_potrf);
    for (int g = 0; g < data->ngpu; g++)
    {
        check_error(cudaSetDevice(data->devices[g]));
        /*
        cholesky decomposition
        */
        // for (int i = 0; i < data->batchCount_gpu[g]; i++)
        // {
        //     printf("%dth", i);
        //     printMatrixGPU(data->Am, data->An, data->d_A_conditioning[g] + i * data->Am * data->ldda, data->ldda);
        // } 
        // printf("[info] Starting Cholesky decomposition. \n");
        if (data->strided)
        {
            check_kblas_error(kblas_potrf_batch(*(data->kblas_handle[g]),
                                                'L', data->Acon,
                                                data->d_A_conditioning[g], data->lddacon, data->Acon * data->lddacon,
                                                data->batchCount_gpu[g],
                                                data->d_info[g]));
        }
        // printf("[info] Finished Cholesky decomposition. \n");
        // check_kblas_error(kblasAllocateWorkspace(*(data->kblas_handle[g])));
        check_error(cudaDeviceSynchronize()); // TODO sync with streams instead
        check_error(cudaGetLastError());
    }
    clock_gettime(CLOCK_MONOTONIC, &end_batch_potrf);
    vecchia_time_batch_potrf = end_batch_potrf.tv_sec - start_batch_potrf.tv_sec + (end_batch_potrf.tv_nsec - start_batch_potrf.tv_nsec) / 1e9;

    /*
    triangular solution: L \Sigma_offset <- \Sigma_old && L z_offset <- z_old
    */
    struct timespec start_batch_trsm, end_batch_trsm;
    clock_gettime(CLOCK_MONOTONIC, &start_batch_trsm);
    for (int g = 0; g < data->ngpu; g++)
    {
        check_error(cudaSetDevice(data->devices[g]));
        // printf("[info] Starting triangular solver. \n");
        if (data->strided)
        {
            check_kblas_error(kblasXtrsm_batch_strided(*(data->kblas_handle[g]),
                                                        'L', 'L', 'N', data->diag,
                                                        data->lddacon, data->An,
                                                        1.,
                                                        data->d_A_conditioning[g], data->lddacon, data->Acon * data->lddacon, // A <- L
                                                        data->d_A_cross[g], data->lddacon, data->An * data->lddacon,
                                                        data->batchCount_gpu[g])); 
            check_kblas_error(kblasXtrsm_batch_strided(*(data->kblas_handle[g]),
                                                        'L', 'L', 'N', data->diag,
                                                        data->lddccon, data->Cn,
                                                        1.,
                                                        data->d_A_conditioning[g], data->lddacon, data->Acon * data->lddacon, // A <- L
                                                        data->d_C_conditioning[g], data->lddccon, data->Cn * data->lddccon,
                                                        data->batchCount_gpu[g]));
        }
        // check_kblas_error(kblasAllocateWorkspace(*(data->kblas_handle[g])));
        check_error(cudaDeviceSynchronize()); // TODO sync with streams instead
        check_error(cudaGetLastError());
    }
    clock_gettime(CLOCK_MONOTONIC, &end_batch_trsm);
    vecchia_time_batch_trsm = end_batch_trsm.tv_sec - start_batch_trsm.tv_sec + (end_batch_trsm.tv_nsec - start_batch_trsm.tv_nsec) / 1e9;
    
    /*----------------------------*/
    /* quadratic term calculations*/
    /*----------------------------*/
    struct timespec start_qua, end_qua;
    clock_gettime(CLOCK_MONOTONIC, &start_qua);
    for (int g = 0; g < data->ngpu; g++)
    {
        /*
        GEMM and GEMV: \Sigma_offset^T %*% \Sigma_offset and \Sigma_offset^T %*% z_offset
        */
        // start_timing(curStream);
        check_error(cudaSetDevice(data->devices[g]));
        if (data->strided)
        {
            // Launch the kernel
            // fprintf(stderr, "--------------gpu: %d ------------\n", g);
            // printVecGPUv1(data->batchCount_gpu[g], data->d_A_offset_vector[g]);
            DgpuDotProducts_Strided(data->d_A_cross[g], data->d_A_cross[g], data->d_A_offset_vector[g], data->batchCount_gpu[g], data->cs, data->lddacon);
            DgpuDotProducts_Strided(data->d_A_cross[g], data->d_C_conditioning[g], data->d_mu_offset_vector[g], data->batchCount_gpu[g], data->cs, data->lddacon);
            
        }
        check_error(cudaDeviceSynchronize()); // TODO sync with streams instead
        check_error(cudaGetLastError());
    }
    clock_gettime(CLOCK_MONOTONIC, &end_qua);
    vecchia_time_quadratic = end_qua.tv_sec - start_qua.tv_sec + (end_qua.tv_nsec - start_qua.tv_nsec) / 1e9;
    
    // copy
    struct timespec start_copy_dh, end_copy_dh;
    clock_gettime(CLOCK_MONOTONIC, &start_copy_dh);
    for (int g = 0; g < data->ngpu; g++)
    {   
        check_error(cudaSetDevice(data->devices[g]));
        int _count = 0;
        for (int j=0; j < g; j++) _count += data->batchCount_gpu[j];
        // copy the mu' and sigma' from gpu to host
        cublasGetVectorAsync(data->batchCount_gpu[g], sizeof(T), 
                                data->d_A_offset_vector[g], 1,
                                data->h_A_offset_vector + _count, 1, 
                                kblasGetStream(*(data->kblas_handle[g])));
        cublasGetVectorAsync(data->batchCount_gpu[g], sizeof(T), 
                                data->d_mu_offset_vector[g], 1,
                                data->h_mu_offset_vector + _count, 1, 
                                kblasGetStream(*(data->kblas_handle[g])));

        // _mat_bit_count += data->batchCount_gpu[g]*data->batchCount_gpu[g];
        check_error(cudaDeviceSynchronize()); // TODO sync with streams instead
        check_error(cudaGetLastError());
    }
    clock_gettime(CLOCK_MONOTONIC, &end_copy_dh);
    time_copy_dh += end_copy_dh.tv_sec - start_copy_dh.tv_sec + (end_copy_dh.tv_nsec - start_copy_dh.tv_nsec) / 1e9;

    // synchronize the gpu
    for (int g = 0; g < data->ngpu; g++)
    {
        check_error(cudaSetDevice(data->devices[g]));
        check_error(cudaDeviceSynchronize()); // TODO sync with streams instead
        check_error(cudaGetLastError());
    }

    /*
    Independent computing
    */
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    // first independent block likelihood 
    core_Xlogdet<T>(data->d_A_conditioning[0], 
                data->cs, data->lddacon,
                &(data->logdet_result_h[0][0]));
    // data->dot_result_h[0][0] = data->h_mu_offset_matrix[0];
    data->dot_result_h[0][0] = data->h_mu_offset_vector[0];

    // scalar vecchia approximation
    // int _sum_batchcmat = 0;
    for (int g = 0; g < data->ngpu; g++)
    {   
        if (g==0){
            for (int i=1; i < data->batchCount_gpu[g]; i++){
                // data->h_C[i] -= data->h_mu_offset_matrix[_sum_batchcmat + i + i * data->batchCount_gpu[g]]; 
                // // the first is no meaning 
                // data->h_A[_sum_batchcvec + i] -= data->h_A_offset_matrix[_sum_batchcmat + i + i * data->batchCount_gpu[g]]; 
                // // llhi calulation
                // data->dot_result_h[g][i] = data->h_C[_sum_batchcvec + i] * data->h_C[_sum_batchcvec + i] / data->h_A[_sum_batchcvec + i];
                // data->logdet_result_h[g][i] = log(data->h_A[_sum_batchcvec + i] );
                // correction
                data->h_C[i] -= data->h_mu_offset_vector[i]; 
                data->h_A[i] -= data->h_A_offset_vector[i]; 
                // llhi calulation
                data->dot_result_h[g][i] = data->h_C[i] * data->h_C[i] / data->h_A[i];
                // fprintf(stderr, "The %d, %lf \n", i, log(data->h_A_offset_vector[i]));
                data->logdet_result_h[g][i] = log(data->h_A[i] );
            }
        }else{
            int _sum_batchcvec = 0;
            for (int j=0; j < g; j++) {_sum_batchcvec += data->batchCount_gpu[j];}
            for (int i=0; i < data->batchCount_gpu[g]; i++){
                // data->h_C[_sum_batchcvec + i] -= data->h_mu_offset_matrix[_sum_batchcmat + i + i * data->batchCount_gpu[g]]; 
                // // the first is no meaning 
                // data->h_A[_sum_batchcvec + i] -= data->h_A_offset_matrix[_sum_batchcmat + i + i * data->batchCount_gpu[g]]; 
                // // llhi calulation
                // data->dot_result_h[g][i] = data->h_C[_sum_batchcvec + i] * data->h_C[_sum_batchcvec + i] / data->h_A[_sum_batchcvec + i];
                // data->logdet_result_h[g][i] = log(data->h_A[_sum_batchcvec + i] );
                data->h_C[_sum_batchcvec + i] -= data->h_mu_offset_vector[_sum_batchcvec + i]; 
                // the first is no meaning 
                data->h_A[_sum_batchcvec + i] -= data->h_A_offset_vector[_sum_batchcvec + i]; 
                // llhi calulation
                data->dot_result_h[g][i] = data->h_C[_sum_batchcvec + i] * data->h_C[_sum_batchcvec + i] / data->h_A[_sum_batchcvec + i];
                data->logdet_result_h[g][i] = log(data->h_A[_sum_batchcvec + i] );
            }
        }
        // _sum_batchcvec += data->batchCount_gpu[g];
        // _sum_batchcmat += data->batchCount_gpu[g] * data->batchCount_gpu[g];
    }
    // time = get_elapsed_time(curStream);
    // printf("-----------------------------------------\n");
    for (int g = 0; g < data->ngpu; g++)
    {   
        // printf("----------------%dth GPU---------------\n", g);
        for (int k = 0; k < data->batchCount_gpu[g]; k++)
        {
            T llk_temp = 0;
            int _size_llh = 1;
            if (g==0 && k==0){
                _size_llh = data->cs;
            }
            llk_temp = -(data->dot_result_h[g][k] + data->logdet_result_h[g][k] + _size_llh * log(2 * PI)) * 0.5;
            llk += llk_temp;
            // printf("%dth log determinant is % lf\n", k, data->logdet_result_h[g][k]);
            // printf("%dth dot product is % lf\n", k, data->dot_result_h[g][k]);
            // printf("%dth pi is % lf\n", k, _size_llh * log(2 * PI));
            // printf("%dth log likelihood is % lf\n", k, llk_temp);
            // printf("-------------------------------------\n");
        }
    }
    // recover the h_C
    data->h_C = data->h_C - (data->cs - 1);
    clock_gettime(CLOCK_MONOTONIC, &end);
    indep_time = end.tv_sec - start.tv_sec + (end.tv_nsec - start.tv_nsec) / 1e9;
    
    vecchia_time_total = vecchia_time_batch_potrf + vecchia_time_batch_trsm + vecchia_time_quadratic;  
    time_copy = time_copy_dh + time_copy_hd;
    if (data->perf == 1){
        fprintf(stderr, "===================================Execution time (s)=======================================\n");
        fprintf(stderr, "   Time total      Time dcmg      Copy(CPU->GPU)     BatchPOTRF    BatchTRSM   Quadratic      Independent\n"); 
        fprintf(stderr, "   %8.6lf        %8.6lf        %8.6lf        %8.6lf      %8.6lf      %8.6lf       %8.6lf\n", 
                        dcmg_time + indep_time + vecchia_time_total + time_copy, 
                        dcmg_time, 
                        time_copy_hd,
                        vecchia_time_batch_potrf,
                        vecchia_time_batch_trsm, 
                        vecchia_time_quadratic,
                        indep_time);
        fprintf(stderr, "=============================Computing performance (Gflops/s)===================================\n");
        fprintf(stderr, "     Vecchia         BatchPOTRF         BatchTRSM       Quadratic\n");
        fprintf(stderr, "   %8.2lf           %8.2lf        %8.2lf        %8.2lf  \n",
                        (gflops_batch_potrf + gflops_batch_trsm + gflops_quadratic)/(vecchia_time_batch_potrf + vecchia_time_batch_trsm + vecchia_time_quadratic),
                        gflops_batch_potrf/vecchia_time_batch_potrf, 
                        gflops_batch_trsm/vecchia_time_batch_trsm, 
                        gflops_quadratic/vecchia_time_quadratic);
        // fprintf(stderr, "%lf ==============\n", llk);
        fprintf(stderr, "==========================================================================================\n");
    }
    if (data->perf != 1){
        saveLogFileParams(data->iterations, 
                        localtheta, llk, 
                        indep_time + vecchia_time_total, dcmg_time, 
                        data->num_loc, data->M,
                        data->zvecs, data->p,
                        data->cs); // this is log_tags for write a file
        if (data->kernel ==1){
            printf("%dth Model Parameters (Variance, range, smoothness): (%lf, %lf, %lf) -> Loglik: %lf \n", 
                data->iterations, localtheta[0], localtheta[1], localtheta[2], llk); 
        }else if (data->kernel ==2){
            printf("%dth Model Parameters (Variance1, Variance2, range, smoothness1, smoothness2, beta): (%lf, %lf, %lf, %lf, %lf, %lf) -> Loglik: %lf \n", 
                data->iterations, localtheta[0], localtheta[1], localtheta[2], 
                localtheta[3], localtheta[4], localtheta[5], llk); 
        }
    }
    data->iterations += 1;
    // printf("-----------------------------------------------------------------------------\n");
    return llk;
}

#endif