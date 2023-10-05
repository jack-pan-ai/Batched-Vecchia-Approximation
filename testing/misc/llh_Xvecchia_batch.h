#ifndef LLH_XVECCHIA_BATCH_H
#define LLH_XVECCHIA_BATCH_H

#include <omp.h>

template <class T>

T llh_Xvecchia_batch(unsigned n, const T* localtheta, T* grad, void* f_data)
{   
    T llk = 0;
    llh_data * data = static_cast<llh_data *>(f_data);
    double kblas_perf = 0.0, kblas_time = 0.0, indep_time = 0.0, dcmg_time = 0.0, vecchia_time = 0.0;
    double alpha_1 = 1.;
    double beta_n1 = -1.;
    double beta_0 =0.;
    int omp_threads = data->omp_threads;
    omp_set_num_threads(omp_threads);

    // printf("[info] Starting Covariance Generation. \n");
    struct timespec start_dcmg, end_dcmg;
    clock_gettime(CLOCK_MONOTONIC, &start_dcmg);

    /*
        Example,
        p(y1 | y2) with mean zero gaussian distribution
        \mu'_1 = \sigma_{12}  inv(\sigma_{22}) y_2
        \sigma'_{11} = \sigma_{11} - \sigma_{12}  inv(\sigma_{22}) \sigma_{12}^T

        
        : \sigma'_{11}/ \sigma_{11}
        d_A_offset: \sigma_{12}  inv(\sigma_{22}) \sigma_{12}^T because gemm cannot overwrite the value in place
        d_C: y1
        d_mu_offset: \mu'_1, used for pdf calculation 

        d_C_conditioning: y_2
        d_A_conditioning: \sigma_{22}
        d_A_cross: \sigma_{12}

        PS: there are 5 important generation parts
        d_A: \sigma_{11}
        *d_A_conditioning: \sigma_{22}
        *d_A_cross: \sigma_{12}

        *d_C_conditioning: y_2
        d_C: y1
    */


    // covariance matrix generation 
    #pragma omp parallel for
    for (int i=0; i < data->batchCount; i++)
    {
        // printf("x_copy is %lf \n",data->locations_copy->x[0]);
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
            // note that we need copy the first block oberservation;
            // then h_C needs to point the location after the first block size;
            // printVectorCPU(data->bs, data->h_C, data->ldc, 0);
            memcpy(data->h_C_conditioning, data->h_C, sizeof(T) * data->cs);
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
            // printVectorCPU(data->Cm, data->h_C, data->ldc, i); 
        }

        // h_A: \sigma_{11}
        // if (data->kernel == 1){
        //     core_dcmg(data->h_A + i * data->An * data->lda,
        //                 data->lda, data->An,
        //                 loc_batch,
        //                 loc_batch, localtheta, data->distance_metric);
        // }
        // else if (data->kernel == 2){
        //     core_dcmg_bivariate_parsimonious(data->h_A + i * data->An * data->lda,
        //                                     data->lda, data->An,
        //                                     loc_batch,
        //                                     loc_batch, localtheta, data->distance_metric);
        // }else{
        //     printf("The other kernel function has been developed.");
        //     exit(0);
        // }
        // printf("The conditioning covariance matrix.\n");
        // printMatrixCPU(data->M, data->M, data->h_A + i * data->An * data->lda, data->lda, i);
        // exit(0);
        // obeservation initialization
        // for (int j = 0; j < data->Cm; j++)
        // {
        //     data->h_C[j + i * data->Cm] = 1;
        //     // h_mu[j + i * data->Cm] = 0;
        // }
        // data->locations_copy->x += data->An;
        // data->locations_copy->y += data->An;
        free(loc_batch);
    }
    // h_A_cross: \sigma_{12} and h_A_conditioning: \sigma_{22}
    // if (data->vecchia)
    // {   
        // printVectorCPU(data->bs, data->h_C, data->ldc, 0);
        #pragma omp parallel for
        for (int i=0; i < (data->batchCount - 1); i++)
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
            // if (data->kernel == 1){
            //     // *h_A_conditioning: \sigma_{22}
            //     core_dcmg(data->h_A_conditioning + i * data->Acon * data->ldacon,
            //                 data->ldacon, data->Acon,
            //                 loc_batch_con,
            //                 loc_batch_con, localtheta, data->distance_metric);
            //     // *h_A_cross: \sigma_{21}
            //     core_dcmg(data->h_A_cross + i * data->An * data->ldacon,
            //                 data->ldacon, data->An,
            //                 loc_batch_con,
            //                 loc_batch, localtheta, data->distance_metric);
            // }else if (data->kernel == 2){
            //     // *h_A_conditioning: \sigma_{22}
            //     core_dcmg_bivariate_parsimonious(data->h_A_conditioning + i * data->Acon * data->ldacon,
            //                                     data->ldacon, data->Acon,
            //                                     loc_batch_con,
            //                                     loc_batch_con, localtheta, data->distance_metric);
            //     // *h_A_cross: \sigma_{21}
            //     core_dcmg_bivariate_parsimonious(data->h_A_cross + i * data->An * data->ldacon,
            //                                     data->ldacon, data->An,
            //                                     loc_batch_con,
            //                                     loc_batch, localtheta, data->distance_metric);
            // }else{
            //     printf("The other kernel function has been developed.");
            //     exit(0);
            // }
            free(loc_batch);
            free(loc_batch_con);
            //matrix size: data->lda by data->An
            // printf("The conditioning covariance matrix.\n");
            // printMatrixCPU(data->M, data->M, data->h_A_cross + i * data->An * data->lda, data->lda, i);
            // printMatrixCPU(data->M, data->M, data->h_A_conditioning + i * data->Acon * data->ldacon, data->ldacon, i);
        }
    // }
    // printVectorCPU(data->bs, data->h_C, data->ldc, 0);
    clock_gettime(CLOCK_MONOTONIC, &end_dcmg);
    dcmg_time = end_dcmg.tv_sec - start_dcmg.tv_sec + (end_dcmg.tv_nsec - start_dcmg.tv_nsec) / 1e9;
    // printf("[info] Covariance Generation with time %lf seconds. \n", dcmg_time);
    // printMatrixCPU(data->M, data->M, data->h_A, data->lda, i);
    // covariance matrix copt from host to device
    check_error(cudaGetLastError());
    for (int g = 0; g < data->ngpu; g++)
    {
        check_error(cudaSetDevice(data->devices[g]));
        check_cublas_error(cublasSetMatrixAsync(
            data->Cm, data->Cn * data->batchCount_gpu[g], sizeof(T),
            data->h_C + data->Cm * data->Cn * data->batchCount_gpu[g] , data->ldc,
            data->d_C[g], data->lddc, 
            kblasGetStream(*(data->kblas_handle[g])))
            );
        check_error(cudaDeviceSynchronize());
    }
    // if (data->vecchia)
    // {   
        // printf("[info] The vecchia offset is starting now!\n");
        // conditioning part 1.1, matrix copy from host to device
        int batch22count = 0;
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
            batch22count += data->cs * data->cs * data->batchCount_gpu[g];
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
        }

        // conditioning part 1.2, wsquery for potrf and trsm 
        for (int g = 0; g < data->ngpu; g++)
        {
            check_error(cudaSetDevice(data->devices[g]));
            if (data->strided)
            {   
                // data->Acon*2 instead of data->Acon is because for some cases, 
                // like 320/20 combination, 320 batchszie 20 batch count, cannot 
                // be allocated enough memory. But the reason is unclear now.
                kblas_potrf_batch_strided_wsquery(*(data->kblas_handle[g]), data->cs*2, data->batchCount_gpu[g]);
                kblas_trsm_batch_strided_wsquery(*(data->kblas_handle[g]), 'L', data->cs, data->N, data->batchCount_gpu[g]);
                kblas_gemm_batch_strided_wsquery(*(data->kblas_handle[g]), data->batchCount_gpu[g]);
            }
            check_kblas_error(kblasAllocateWorkspace(*(data->kblas_handle[g])));
            check_error(cudaGetLastError());
        }


        struct timespec start, end;
        clock_gettime(CLOCK_MONOTONIC, &start);

        for (int g = 0, _mat_bit_count = 0; g < data->ngpu; g++)
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
            /*
            triangular solution: L \Sigma_offset <- \Sigma_old && L z_offset <- z_old
            */
            // printf("[info] Starting triangular solver. \n");
            // for (int i = 0; i < data->batchCount_gpu[g]; i++)
            // {
            //     printf("%dth", i);
            //     // printMatrixGPU(data->Am, data->An, data->d_A_conditioning[g] + i * data->Am * data->ldda, data->ldda);
            //     printMatrixGPU(data->Am, data->An, data->d_A_cross[g] + i * data->Am * data->ldda, data->ldda);
            // }
            if (data->strided)
            {
                check_kblas_error(kblasXtrsm_batch_strided(*(data->kblas_handle[g]),
                                                            'L', 'L', 'N', data->diag,
                                                            data->lddacon, data->An,
                                                            1.,
                                                            data->d_A_conditioning[g], data->lddacon, data->Acon * data->lddacon, // A <- L
                                                            data->d_A_cross[g], data->lddacon, data->An * data->lddacon,
                                                            data->batchCount_gpu[g])); //data->d_A_cross <- 
                check_kblas_error(kblasXtrsm_batch_strided(*(data->kblas_handle[g]),
                                                            'L', 'L', 'N', data->diag,
                                                            data->lddccon, data->Cn,
                                                            1.,
                                                            data->d_A_conditioning[g], data->lddacon, data->Acon * data->lddacon, // A <- L
                                                            data->d_C_conditioning[g], data->lddccon, data->Cn * data->lddccon,
                                                            data->batchCount_gpu[g]));
            }
            // for (int i = 0; i < data->batchCount_gpu[g]; i++)
            // {
            //     printf("%dth", i);
            //     printMatrixGPU(data->Am, data->An, data->d_A_cross[g] + i * data->Am * data->ldda, data->ldda);
            // }
            // printVecGPU(data->Cm, data->Cn, data->d_C[0], data->ldda, i);
            // printf("[info] Finished triangular solver. \n");
            /*
            GEMM and GEMV: \Sigma_offset^T %*% \Sigma_offset and \Sigma_offset^T %*% z_offset
            */
            // printf("[info] Starting GEMM and GEMV. \n");
            if (data->strided)
            {
                // // \Sigma_offset^T %*% \Sigma_offset
                // for (int i = 0; i < data->batchCount_gpu[g]; i++)
                // {
                //     printf("%dth", i);
                //     printMatrixGPU(data->Am, data->An, data->d_A_cross[g] + i * data->Acon * data->lddacon, data->lddacon);
                // }
                // printMatrixGPU(data->batchCount_gpu[g], data->batchCount_gpu[g], data->d_mu_offset[g], data->batchCount_gpu[g]);
                // printMatrixGPU(data->batchCount_gpu[g], data->batchCount_gpu[g], data->d_A_offset[g], data->batchCount_gpu[g]);
                check_cublas_error(cublasXgemm(kblasGetCublasHandle(*(data->kblas_handle[g])),
                                                CUBLAS_OP_T, CUBLAS_OP_N,
                                                data->batchCount_gpu[g], data->batchCount_gpu[g], data->cs,
                                                &alpha_1, data->d_A_cross[g], data->lddacon,
                                                            data->d_A_cross[g], data->lddacon,
                                                &beta_0,  data->d_A_offset[g], data->batchCount_gpu[g]));
                 // for (int i = 0; i < data->batchCount_gpu[g]; i++)
                // {
                //     printf("%dth", i);
                //     printMatrixGPU(data->Am, data->An, data->d_A_offset[g] + i * data->An * data->ldda, data->ldda);
                // } 
                // for (int i = 0; i < data->batchCount_gpu[g]; i++)
                // {
                //     printf("%dth", i);
                //     printVecGPU(data->Cm, data->Cn, data->d_C_conditioning[g] + i * data->lddc * data->Cn, data->lddc, i);
                // } 
                // \Sigma_offset^T %*% z_offset
                check_cublas_error(cublasXgemm(kblasGetCublasHandle(*(data->kblas_handle[g])),
                                                CUBLAS_OP_T, CUBLAS_OP_N,
                                                data->batchCount_gpu[g], data->batchCount_gpu[g], data->cs,
                                                &alpha_1, data->d_A_cross[g], data->lddacon,
                                                          data->d_C_conditioning[g], data->lddacon,
                                                &beta_0,  data->d_mu_offset[g], data->batchCount_gpu[g]));
                // printMatrixGPU(data->cs, data->batchCount_gpu[g], data->d_A_cross[g], data->lddacon);
                // printMatrixGPU(data->cs, data->batchCount_gpu[g], data->d_C_conditioning[g], data->lddacon);
                // printMatrixGPU(data->batchCount_gpu[g], data->batchCount_gpu[g], data->d_mu_offset[g], data->batchCount_gpu[g]);
                // printMatrixGPU(data->batchCount_gpu[g], data->batchCount_gpu[g], data->d_A_offset[g], data->batchCount_gpu[g]);
            }
            
            // printf("The results before TRSM \n");
            // printVecGPU(data->Cm, data->Cn, data->d_C[g] + data->Cn * data->lddc, data->ldc, 1);
            // printf("The results before TRSM \n");
            // printVecGPU(data->Cm, data->Cn, data->d_C[g], data->ldc, 0);
            // copy the mu' and sigma' from gpu to host
            check_cublas_error(cublasGetMatrix(data->batchCount_gpu[g], data->batchCount_gpu[g], sizeof(T),
                                                data->d_A_offset[g], data->batchCount_gpu[g],
                                                data->h_A_offset_matrix + _mat_bit_count,
                                                data->batchCount_gpu[g]));
            check_cublas_error(cublasGetMatrix(data->batchCount_gpu[g], data->batchCount_gpu[g], sizeof(T),
                                                data->d_mu_offset[g], data->batchCount_gpu[g],
                                                data->h_mu_offset_matrix + _mat_bit_count, 
                                                data->batchCount_gpu[g]));
            _mat_bit_count += data->batchCount_gpu[g]*data->batchCount_gpu[g];
        }
        for (int g = 0; g < data->ngpu; g++)
        {
            check_error(cudaSetDevice(data->devices[g]));
            check_error(cudaDeviceSynchronize()); // TODO sync with streams instead
            check_error(cudaGetLastError());
        }
        clock_gettime(CLOCK_MONOTONIC, &end);
        vecchia_time = end.tv_sec - start.tv_sec + (end.tv_nsec - start.tv_nsec) / 1e9;
        // */conditioning part, \sigma_{12} inv (\sigma_{22}) \sigma_{21}
        // printf("[info] The vecchia offset is finished!  \n");
        // printf("[Info] The time for vecchia offset is %lf seconds \n", vecchia_time);
    // }
    // // printf("[info] Independent computing is starting now! \n");
    // for (int g = 0; g < data->ngpu; g++)
    // {
    //     check_error(cudaSetDevice(data->devices[g]));
    //     cudaDeviceSynchronize(); // TODO sync with streams instead
    // }
    
    
    /*
    Independent computing
    */
    clock_gettime(CLOCK_MONOTONIC, &start);
    

    // first independent block likelihood 
    core_Xlogdet<T>(data->d_A_conditioning[0], 
                data->cs, data->lddacon,
                &(data->logdet_result_h[0][0]));
    data->dot_result_h[0][0] = data->h_mu_offset_matrix[0];
    fprintf(stderr, "%lf \n", data->h_mu_offset_matrix[0]);

    // scalar vecchia approximation
    int _sum_batchcvec = 0;
    int _sum_batchcmat = 0;
    for (int g = 0; g < data->ngpu; g++)
    {   
        if (g==0){
            for (int i=1; i < data->batchCount_gpu[g]; i++){
                data->h_C[_sum_batchcvec + i] -= data->h_mu_offset_matrix[_sum_batchcmat + i + i * data->batchCount_gpu[g]]; 
                // the first is no meaning 
                data->h_A[_sum_batchcvec + i] -= data->h_A_offset_matrix[_sum_batchcmat + i + i * data->batchCount_gpu[g]]; 
                // llhi calulation
                data->dot_result_h[g][i] = data->h_C[_sum_batchcvec + i] * data->h_C[_sum_batchcvec + i] / data->h_A[_sum_batchcvec + i];
                data->logdet_result_h[g][i] = log(data->h_A[_sum_batchcvec + i] );
            }
        }else{
            for (int i=0; i < data->batchCount_gpu[g]; i++){
                data->h_C[_sum_batchcvec + i] -= data->h_mu_offset_matrix[_sum_batchcmat + i + i * data->batchCount_gpu[g]]; 
                // the first is no meaning 
                data->h_A[_sum_batchcvec + i] -= data->h_A_offset_matrix[_sum_batchcmat + i + i * data->batchCount_gpu[g]]; 
                // llhi calulation
                data->dot_result_h[g][i] = data->h_C[_sum_batchcvec + i] * data->h_C[_sum_batchcvec + i] / data->h_A[_sum_batchcvec + i];
                data->logdet_result_h[g][i] = log(data->h_A[_sum_batchcvec + i] );
            }
        }
        _sum_batchcvec += data->batchCount_gpu[g];
        _sum_batchcmat += data->batchCount_gpu[g] * data->batchCount_gpu[g];
    }
    // time = get_elapsed_time(curStream);
    clock_gettime(CLOCK_MONOTONIC, &end);
    indep_time = end.tv_sec - start.tv_sec + (end.tv_nsec - start.tv_nsec) / 1e9;

    // printf("-----------------------------------------\n");
    for (int g = 0; g < data->ngpu; g++)
    {   
        // printf("----------------%dth GPU---------------\n", g);
        for (int k = 0; k < data->batchCount_gpu[g]; k++)
        {
            T llk_temp = 0;
            llk_temp = -(data->dot_result_h[g][k] + data->logdet_result_h[g][k] + data->Am * log(2 * PI)) * 0.5;
            llk += llk_temp;
            printf("%dth log determinant is % lf\n", k, data->logdet_result_h[g][k]);
            printf("%dth dot product is % lf\n", k, data->dot_result_h[g][k]);
            printf("%dth pi is % lf\n", k, data->Am * log(2 * PI));
            printf("%dth log likelihood is % lf\n", k, llk_temp);
            printf("-------------------------------------\n");
        }
    }
    // printf("[info] Independent computing is finished! \n");
    // printf("[Info] The time for independent computing is %lf seconds\n", indep_time);
    // printf("[Info] The time for LLH is %lf seconds\n", indep_time + vecchia_time);
    // // printf("(Estimated) Sigma: %lf beta:  %lf  nu: %lf\n", localtheta[0], localtheta[1], localtheta[2]);
    // printf("Log likelihood is %lf \n", llk);
    if (data->perf != 1){
        saveLogFileParams(data->iterations, 
                        localtheta, llk, 
                        indep_time + vecchia_time, dcmg_time, 
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
    data->vecchia_time_total += (indep_time + vecchia_time);  
    // printf("-----------------------------------------------------------------------------\n");
    return llk;
}

#endif