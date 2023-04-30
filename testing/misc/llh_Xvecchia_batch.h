#include <omp.h>

template <class T>

T llh_Xvecchia_batch(unsigned n, const T* localtheta, T* grad, void* f_data)
{   
    T llk = 0;
    llh_data * data = static_cast<llh_data *>(f_data);
    double kblas_perf = 0.0, kblas_time = 0.0, indep_time = 0.0, dcmg_time = 0.0, vecchia_time = 0.0;
    double alpha_1 = 1.;
    double beta_n1 = -1.;
    int omp_threads = data->omp_threads;
    omp_set_num_threads(omp_threads);

    // printf("[info] Starting Covariance Generation. \n");
    struct timespec start_dcmg, end_dcmg;
    clock_gettime(CLOCK_MONOTONIC, &start_dcmg);

    if (data->vecchia){
        // init for each iteration (necessary but low efficient)
        // TESTING_MALLOC_CPU(data->locations_copy->x, double, data->batchCount * data->lda);
        // TESTING_MALLOC_CPU(data->locations_copy->y, double, data->batchCount * data->lda);
        // memcpy(data->locations_copy->x, data->locations->x, data->batchCount * data->lda * sizeof(double));
        // memcpy(data->locations_copy->y, data->locations->y, data->batchCount * data->lda * sizeof(double));
        for (int i=0; i < data->batchCount; i++){
            data->locations_con[i]->x=&(data->locations->x[int(data->lda * i/data->p)]);
            data->locations_con[i]->y=&(data->locations->y[int(data->lda * i/data->p)]); // 
            data->locations_con[i]->z = NULL;
            // fprintf(stderr, "%d, \n", int(data->lda * i /data->p));
        }                   
        // fprintf(stderr, "asdasdsa\n");
    }
    // printf("asdsadasda------------\n");
    // exit(0);

    // for(int i =0; i< data->batchCount * data->lda / data->p; i++){
    //     printf("(x, y) is (%lf, %lf) \n", data->locations->x[i], data->locations->y[i]);
    //     printf("COPY (x, y) is (%lf, %lf) \n", data->locations_copy->x[i], data->locations_copy->y[i]);
    //     // exit(0);
    // }
    #pragma omp parallel for
    for (int i=0; i < data->batchCount; i++)
    {
        // printf("x_copy is %lf \n",data->locations_copy->x[0]);
        location* loc_batch= (location *) malloc(sizeof (location)); // this is used for each batch 
        // TESTING_MALLOC_CPU(loc_batch->x, double, data->lda); 
        // TESTING_MALLOC_CPU(loc_batch->y, double, data->lda); 
        loc_batch->x=&(data->locations->x[int (i*data->lda / data->p)]);
        loc_batch->y=&(data->locations->y[int (i*data->lda / data->p)]);
        loc_batch->z= NULL;
        if (data->kernel == 1){
            core_dcmg(data->h_A + i * data->An * data->lda,
                        data->lda, data->An,
                        loc_batch,
                        loc_batch, localtheta, data->distance_metric);
        }else if (data->kernel == 2){
            core_dcmg_bivariate_parsimonious(data->h_A + i * data->An * data->lda,
                                            data->lda, data->An,
                                            loc_batch,
                                            loc_batch, localtheta, data->distance_metric);
        }else{
            printf("The other kernel function has been developed.");
            exit(0);
        }
        // printf("The conditioning covariance matrix.\n");
        // printMatrixCPU(data->M, data->M, data->h_A + i * data->An * data->lda, data->lda, i);
        // exit(0);
        if (data->vecchia)
        {
            // used for vecchia offset, the first one would not be used for offset
            if(i==0)
            {      
                // this is used for the boundary, which no any meaning for llh
                data->locations_con_boundary = GenerateXYLoc(data->lda, 1);
                if (data->kernel == 1){
                    core_dcmg(data->h_A_conditioned, data->ldacon, data->An,
                                data->locations_con_boundary,
                                loc_batch, localtheta, data->distance_metric);
                }else if (data->kernel == 2){
                    core_dcmg_bivariate_parsimonious(data->h_A_conditioned, data->ldacon, data->An,
                                                    data->locations_con_boundary,
                                                    loc_batch, localtheta, data->distance_metric);
                }else{
                    printf("The other kernel function has been developed.");
                    exit(0);
                }
            }    
            else
            {   
                data->locations_con[i-1]->x += (data->Am - data->Acon);
                data->locations_con[i-1]->y += (data->Am - data->Acon);
                // printf("x is %lf \n",data->locations_con->x[0]);
                // printf("y is %lf \n",data->locations_con->y[0]);
                // printf("x_copy is %lf \n",data->locations_copy->x[0]);
                // printf("y_copy is %lf \n",data->locations_copy->y[0]);
                if (data->kernel == 1){
                    core_dcmg(data->h_A_conditioned + i * data->An * data->ldacon,
                                data->ldacon, data->An,
                                data->locations_con[i-1],
                                loc_batch, localtheta, data->distance_metric);
                }else if (data->kernel == 2){
                    core_dcmg_bivariate_parsimonious(data->h_A_conditioned + i * data->An * data->ldacon,
                                                    data->ldacon, data->An,
                                                    data->locations_con[i-1],
                                                    loc_batch, localtheta, data->distance_metric);
                }else{
                    printf("The other kernel function has been developed.");
                    exit(0);
                }
                //matrix size: data->lda by data->An
                // printf("The conditioned covariance matrix.\n");
                // printMatrixCPU(data->M, data->M, data->h_A_conditioned + i * data->An * data->lda, data->lda, i);
                // data->locations_con->x += data->Acon;
                // data->locations_con->y += data->Acon;
            }
        }

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
    
    clock_gettime(CLOCK_MONOTONIC, &end_dcmg);
    dcmg_time = end_dcmg.tv_sec - start_dcmg.tv_sec + (end_dcmg.tv_nsec - start_dcmg.tv_nsec) / 1e9;
    // printf("[info] Covariance Generation with time %lf seconds. \n", dcmg_time);
    /*
    Xrand_matrix(data->Am, data->An * data->batchCount, data->h_A, data->lda);
    for (int i = 0; i < data->batchCount; i++)
    {
        Xmatrix_make_hpd(data->Am, data->h_A + i * data->An * data->lda, data->lda);
        // printMatrixCPU(data->M, data->M, data->h_A, data->lda, i);
    }
    */
    // Xrand_matrix(data->Cm, data->Cn * data->batchCount, data->h_C, data->ldc);
    // printMatrixCPU(data->M, data->M, data->h_A, data->lda, i);
    check_error(cudaGetLastError());
    for (int g = 0; g < data->ngpu; g++)
    {
        check_error(cudaSetDevice(data->devices[g]));
        check_cublas_error(cublasSetMatrixAsync(data->Am, data->An * data->batchCount_gpu, sizeof(T),
                                                data->h_A + data->Am * data->An * data->batchCount_gpu * g, data->lda,
                                                data->d_A[g], data->ldda, kblasGetStream(*(data->kblas_handle[g]))));
        check_cublas_error(cublasSetMatrixAsync(data->Cm, data->Cn * data->batchCount_gpu, sizeof(T),
                                                data->h_C + data->Cm * data->Cn * data->batchCount_gpu * g, data->ldc,
                                                data->d_C[g], data->lddc, kblasGetStream(*(data->kblas_handle[g]))));
        // check_cublas_error(cublasSetMatrixAsync(data->Cm, data->Cn * data->batchCount_gpu, sizeof(T),
        //                                         h_mu + data->Cm * data->Cn * data->batchCount_gpu * g, data->ldc,
        //                                         d_mu[g], data->lddc, kblasGetStream(*(data->kblas_handle[g]))));
        /* TODO
        if (!data->strided)
        {
            check_kblas_error(Xset_pointer_2(d_A_array[g], data->d_A[g], data->ldda, data->An * data->ldda,
                                                data->d_C_array[g], data->d_C[g], data->lddc, data->Cn * data->lddc,
                                                data->batchCount_gpu, kblasGetStream(*(data->kblas_handle[g]))));
        }
        if (data->nonUniform)
        {
            check_cublas_error(cublasSetVectorAsync(data->batchCount_gpu, sizeof(int),
                                                    h_M + data->batchCount_gpu * g, 1,
                                                    d_M[g], 1, kblasGetStream(*(data->kblas_handle[g]))));
            check_cublas_error(cublasSetVectorAsync(data->batchCount, sizeof(int),
                                                    h_N + data->batchCount_gpu * g, 1,
                                                    d_N[g], 1, kblasGetStream(*(data->kblas_handle[g]))));
            check_kblas_error(iset_value_1(d_ldda[g], data->ldda, data->batchCount, kblasGetStream(*(data->kblas_handle[g]))));
            check_kblas_error(iset_value_1(d_lddc[g], data->lddc, data->batchCount, kblasGetStream(*(data->kblas_handle[g]))));
        }
        */
    }
    if (data->vecchia)
    {   
        // printf("[info] The vecchia offset is starting now!\n");
        // /* conditioned part, \sigma_{12} inv (\sigma_{22}) \sigma_{21} and \mu
        // this part is removable in the real data 
        for (int i = 0; i < data->batchCount; i++)
        {   
            // pair with conditioned covariance matrix
            if (i==0){
                // this does not matter, just some random number to make the matrix postive definite
                if (data->kernel == 1){
                    core_dcmg(data->h_A_copy, data->Acon, data->Acon,
                                            data->locations_con_boundary,
                                            data->locations_con_boundary, 
                                            localtheta, data->distance_metric); 
                }else if (data->kernel == 2){
                    core_dcmg_bivariate_parsimonious(data->h_A_copy, data->Acon, data->Acon,
                                            data->locations_con_boundary,
                                            data->locations_con_boundary, 
                                            localtheta, data->distance_metric); 
                }else{
                    printf("The other kernel function has been developed.");
                    exit(0);
                }
                memcpy(data->h_C_conditioned, data->h_C, sizeof(T) * data->ldccon * data->Cn);
            }
            else{
                // varied conditioned number
                for (int j = 0; j < data->Acon; j++)
                {
                    memcpy(data->h_A_copy + data->Acon * j + i * data->Acon * data->Acon,
                        data->h_A  + data->lda * data->An * (i -1) + (data->An - data->Acon + j) * data->An + (data->An - data->Acon), //  = i * data->An * data->An,
                        sizeof(T) * data->Acon);                                       // the last one will not be used to offset
                }
                memcpy(data->h_C_conditioned + data->ldccon * data->Cn * i, 
                        data->h_C + data->ldc * data->Cn * (i - 1) + (data->ldc - data->ldccon), 
                        sizeof(T) * data->ldccon * data->Cn);
            }
            // printf("data->h_A_copy: \n");
            // printMatrixCPU(data->Acon, data->Acon, data->h_A_copy + i * data->Acon * data->Acon, data->ldacon, i);
        }
        // extra memory for mu
        // check_cublas_error(cublasSetMatrixAsync(data->Cm, data->Cn * data->batchCount_gpu, sizeof(T),
        //                                         h_mu + data->Cm * data->Cn * data->batchCount_gpu * g, data->ldc,
        //                                         d_mu_copy[g], data->lddc, kblasGetStream(*(data->kblas_handle[g]))));
        // conditioned part 1.1, matrix copy from host to device
        for (int g = 0; g < data->ngpu; g++)
        {
            check_error(cudaSetDevice(data->devices[g]));
            check_cublas_error(cublasSetMatrixAsync(data->Acon, data->Acon * data->batchCount_gpu, sizeof(T),
                                                    data->h_A_copy + data->Acon * data->Acon * data->batchCount_gpu * g, data->ldacon,
                                                    data->d_A_copy[g], data->lddacon, kblasGetStream(*(data->kblas_handle[g]))));
            check_cublas_error(cublasSetMatrixAsync(data->Acon, data->An * data->batchCount_gpu, sizeof(T),
                                                    data->h_A_conditioned + data->Acon * data->An * data->batchCount_gpu * g, data->ldacon,
                                                    data->d_A_conditioned[g], data->lddacon, kblasGetStream(*(data->kblas_handle[g]))));
            check_cublas_error(cublasSetMatrixAsync(data->Ccon, data->Cn * data->batchCount_gpu, sizeof(T),
                                                    data->h_C_conditioned + data->Ccon * data->Cn * data->batchCount_gpu * g, data->ldccon,
                                                    data->d_C_conditioned[g], data->lddccon, kblasGetStream(*(data->kblas_handle[g]))));
            // check the device for offset, needed?  TODO test
            // check_cublas_error(cublasSetMatrixAsync(data->Cm, data->Cn * data->batchCount_gpu, sizeof(T),
            //                                         data->h_C_conditioned + data->Cm * data->Cn * data->batchCount_gpu * g, data->ldc,
            //                                         data->d_mu_offset[g], data->lddc, kblasGetStream(*(data->kblas_handle[g]))));
            check_cublas_error(cublasSetMatrixAsync(data->Am, data->An * data->batchCount_gpu, sizeof(T),
                                                    data->h_A_copy + data->Am * data->An * data->batchCount_gpu * g, data->lda,
                                                    data->d_A_offset[g], data->ldda, kblasGetStream(*(data->kblas_handle[g]))));
            check_cublas_error(cublasSetMatrixAsync(data->Cm, data->Cn * data->batchCount_gpu, sizeof(T),
                                                    data->h_C + data->Cm * data->Cn * data->batchCount_gpu * g, data->ldc,
                                                    data->d_C_copy[g], data->lddc, kblasGetStream(*(data->kblas_handle[g]))));
            /*
            TODO
            if (!data->strided)
            {
                check_kblas_error(Xset_pointer_2(d_A_array[g], data->d_A[g], data->ldda, data->An * data->ldda,
                                                    data->d_C_array[g], data->d_C[g], data->lddc, data->Cn * data->lddc,
                                                    data->batchCount_gpu, kblasGetStream(*(data->kblas_handle[g]))));
            }
            if (data->nonUniform)
            {
                check_cublas_error(cublasSetVectorAsync(data->batchCount_gpu, sizeof(int),
                                                        h_M + data->batchCount_gpu * g, 1,
                                                        d_M[g], 1, kblasGetStream(*(data->kblas_handle[g]))));
                check_cublas_error(cublasSetVectorAsync(data->batchCount, sizeof(int),
                                                        h_N + data->batchCount_gpu * g, 1,
                                                        d_N[g], 1, kblasGetStream(*(data->kblas_handle[g]))));
                check_kblas_error(iset_value_1(d_ldda[g], data->ldda, data->batchCount, kblasGetStream(*(data->kblas_handle[g]))));
                check_kblas_error(iset_value_1(d_lddc[g], data->lddc, data->batchCount, kblasGetStream(*(data->kblas_handle[g]))));
            }
            */
        }

        // conditioned part 1.2, wsquery for potrf and trsm
        for (int g = 0; g < data->ngpu; g++)
        {
            check_error(cudaSetDevice(data->devices[g]));
            if (data->strided)
            {
                kblas_potrf_batch_strided_wsquery(*(data->kblas_handle[g]), data->Acon, data->batchCount_gpu);
                kblas_trsm_batch_strided_wsquery(*(data->kblas_handle[g]), data->side, data->Acon, data->N, data->batchCount_gpu);
                kblas_gemm_batch_strided_wsquery(*(data->kblas_handle[g]), data->batchCount);
            }
            else
            {
                return 0;
            }
            /* TODO
            else if (data->nonUniform)
                kblas_trsm_batch_nonuniform_wsquery(*(data->kblas_handle[g]));
            else
            {
                kblas_potrf_batch_wsquery(*(data->kblas_handle[g]), data->M, data->batchCount_gpu);
                kblas_trsm_batch_wsquery(*(data->kblas_handle[g]), data->side, data->M, data->N, data->batchCount_gpu);
            }
            */
            check_kblas_error(kblasAllocateWorkspace(*(data->kblas_handle[g])));
            check_error(cudaGetLastError());
        }


        struct timespec start, end;
        clock_gettime(CLOCK_MONOTONIC, &start);

        for (int g = 0; g < data->ngpu; g++)
        {
            check_error(cudaSetDevice(data->devices[g]));
            /*
            cholesky decomposition
            */
            // for (int i = 0; i < data->batchCount_gpu; i++)
            // {
            //     printf("%dth", i);
            //     printMatrixGPU(data->Am, data->An, data->d_A_copy[g] + i * data->Am * data->ldda, data->ldda);
            // } 
            // printf("[info] Starting Cholesky decomposition. \n");
            if (data->strided)
            {
                check_kblas_error(kblas_potrf_batch(*(data->kblas_handle[g]),
                                                    data->uplo, data->Am,
                                                    data->d_A_copy[g], data->lddacon, data->Acon * data->lddacon,
                                                    data->batchCount_gpu,
                                                    data->d_info[g]));
            }
            /* TODO
            else
            {
                check_kblas_error(kblas_potrf_batch(*(data->kblas_handle[g]),
                                                    data->uplo, data->Am,
                                                    d_A_array[g], data->ldda,
                                                    data->batchCount_gpu,
                                                    data->d_info[g]));
            }
            */
            // for (int i = 0; i < data->batchCount_gpu; i++)
            // {
            //     printf("%dth", i);
            //     printMatrixGPU(data->Am, data->An, data->d_A[g] + i * data->Am * data->ldda, data->ldda);
            // }
            // printf("[info] Finished Cholesky decomposition. \n");
            /*
            triangular solution: L \Sigma_offset <- \Sigma_old && L z_offset <- z_old
            */
            // printf("[info] Starting triangular solver. \n");
            // for (int i = 0; i < data->batchCount_gpu; i++)
            // {
            //     printf("%dth", i);
            //     printMatrixGPU(data->Am, data->An, data->d_A_copy[g] + i * data->Am * data->ldda, data->ldda);
            //     printMatrixGPU(data->Am, data->An, data->d_A_conditioned[g] + i * data->Am * data->ldda, data->ldda);
            // }
            if (data->strided)
            {
                check_kblas_error(kblasXtrsm_batch_strided(*(data->kblas_handle[g]),
                                                            data->side, data->uplo, KBLAS_NoTrans, data->diag,
                                                            data->lddacon, data->An,
                                                            1.,
                                                            data->d_A_copy[g], data->lddacon, data->Acon * data->lddacon, // A <- L
                                                            data->d_A_conditioned[g], data->lddacon, data->An * data->lddacon,
                                                            data->batchCount_gpu)); //data->d_A_conditioned <- 
                check_kblas_error(kblasXtrsm_batch_strided(*(data->kblas_handle[g]),
                                                            data->side, data->uplo, KBLAS_NoTrans, data->diag,
                                                            data->lddccon, data->Cn,
                                                            1.,
                                                            data->d_A_copy[g], data->lddacon, data->Acon * data->lddacon, // A <- L
                                                            data->d_C_conditioned[g], data->lddccon, data->Cn * data->lddccon,
                                                            data->batchCount_gpu));
            }
            // for (int i = 0; i < data->batchCount_gpu; i++)
            // {
            //     printf("%dth", i);
            //     printMatrixGPU(data->Am, data->An, data->d_A_conditioned[g] + i * data->Am * data->ldda, data->ldda);
            // }
            /*TODO
            else
            {
                check_cublas_error(
                    cublasDtrsmBatched(kblasGetCublasHandle(*(data->kblas_handle[g])),
                                        CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_T, CUBLAS_DIAG_NON_UNIT,
                                        data->An, data->Cn,
                                        &1.,
                                        d_A_array[g], data->ldda,
                                        data->d_C_array[g], data->lddc,
                                        data->batchCount_gpu));
                // check_kblas_error(kblasXtrsm_batch(*(data->kblas_handle[g]),
                //                                    data->side, data->uplo, data->transA, data->diag,
                //                                    data->M, data->N,
                //                                    1., (const T **)(d_A_array[g]), data->ldda,
                //                                    data->d_C_array[g], data->lddc,
                //                                    data->batchCount_gpu));
            }
            */
            // printVecGPU(data->Cm, data->Cn, data->d_C[0], data->ldda);
            // printf("[info] Finished triangular solver. \n");
            /*
            GEMM and GEMV: \Sigma_offset^T %*% \Sigma_offset and \Sigma_offset^T %*% z_offset
            */
            // printf("[info] Starting GEMM and GEMV. \n");
            if (data->strided)
            {
                // \Sigma_offset^T %*% \Sigma_offset
                // for (int i = 0; i < data->batchCount_gpu; i++)
                // {
                //     printf("%dth", i);
                //     printMatrixGPU(data->Am, data->An, data->d_A_conditioned[g] + i * data->Acon * data->lddacon, data->lddacon);
                // }
                check_kblas_error(kblas_gemm_batch(*(data->kblas_handle[g]),
                                                    KBLAS_Trans, KBLAS_NoTrans,
                                                    data->Am, data->An, data->Acon,
                                                    1.,
                                                    data->d_A_conditioned[g], data->lddacon, data->An * data->lddacon,
                                                    data->d_A_conditioned[g], data->lddacon, data->An * data->lddacon,
                                                    0,
                                                    data->d_A_offset[g], data->ldda, data->An * data->ldda,
                                                    data->batchCount_gpu));
                // for (int i = 0; i < data->batchCount_gpu; i++)
                // {
                //     printf("%dth", i);
                //     printMatrixGPU(data->Am, data->An, data->d_A_offset[g] + i * data->An * data->ldda, data->ldda);
                // } 
                // \Sigma_offset^T %*% z_offset
                check_kblas_error(kblas_gemm_batch(*(data->kblas_handle[g]),
                                                    KBLAS_Trans, KBLAS_NoTrans,
                                                    data->Am, data->Cn, data->Acon,
                                                    1.,
                                                    data->d_A_conditioned[g], data->lddacon, data->An * data->lddacon,
                                                    data->d_C_conditioned[g], data->lddccon, data->Cn * data->lddccon,
                                                    0,
                                                    data->d_mu_offset[g], data->lddc, data->Cn * data->lddc,
                                                    data->batchCount_gpu));
                // for (int i = 0; i < data->batchCount_gpu; i++)
                // {
                //     printf("%dth", i);
                //     printVecGPU(data->Cm, data->Cn, data->d_mu_offset[g] + i * data->lddc * data->Cn, data->lddc, i);
                // } 
            }
            /*TODO non-data->strided*/
            // printf("[info] Finished GEMM and GEMV. \n");

            /*
            GEAD: \Sigma_new <- \Sigma - \Sigma_offset && \mu_new <- \mu - \mu_offset (not necessary)
            */
            // printf("[info] Starting GEAD. \n");
            // for (int i = 0; i < data->batchCount_gpu; i++)
            // {
            //     printf("%dth", i);
            //     printMatrixGPU(data->Am, data->An, data->d_A_copy[g] + i * data->Am * data->ldda, data->ldda);
            // }
            // for (int i = 0; i < data->batchCount_gpu; i++)
            // {
            //     printf("%dth", i);
            //     printMatrixGPU(data->Am, data->An, data->d_A_offset[g] + i * data->Am * data->ldda, data->ldda);
            // }
            if (data->strided)
            {
                for (int i = 1; i < data->batchCount_gpu; i++)
                {   
                    check_cublas_error(cublasXgeam(kblasGetCublasHandle(*(data->kblas_handle[g])),
                                                    CUBLAS_OP_N, CUBLAS_OP_N,
                                                    data->Am, data->An,
                                                    &alpha_1, // 1
                                                    data->d_A[g] + data->ldda * data->An * i, data->ldda, // !!!!!not sure if it's ok, in order to save memory !!!!
                                                    &beta_n1, // -1
                                                    data->d_A_offset[g] + data->ldda * data->An * i, data->ldda,
                                                    data->d_A[g] + data->ldda * data->An * i, data->ldda)); // + data->ldda * data->An * i means the first one does not change
                }
            }
            // for (int i = 0; i < data->batchCount_gpu; i++)
            // {
            //     printf("%dth", i);
            //     printMatrixGPU(data->Am, data->An, data->d_A[g] + i * data->Am * data->ldda, data->ldda);
            // }
            /*TODO non-data->strided*/
            // printf("[info] Finished GEAD. \n");
        }

        for (int g = 0; g < data->ngpu; g++)
        {
            check_error(cudaSetDevice(data->devices[g]));
            check_error(cudaDeviceSynchronize()); // TODO sync with streams instead
            check_error(cudaGetLastError());
        }
        clock_gettime(CLOCK_MONOTONIC, &end);
        vecchia_time = end.tv_sec - start.tv_sec + (end.tv_nsec - start.tv_nsec) / 1e9;
        // */conditioned part, \sigma_{12} inv (\sigma_{22}) \sigma_{21}
        // printf("[info] The vecchia offset is finished!  \n");
        // printf("[Info] The time for vecchia offset is %lf seconds \n", vecchia_time);
    }
    // // printf("[info] Independent computing is starting now! \n");
    // for (int g = 0; g < data->ngpu; g++)
    // {
    //     check_error(cudaSetDevice(data->devices[g]));
    //     cudaDeviceSynchronize(); // TODO sync with streams instead
    // }
    /*
    Independent computing
    */
    for (int g = 0; g < data->ngpu; g++)
    {      
        check_error(cudaGetLastError());
        check_error(cudaSetDevice(data->devices[g]));
        if (data->strided)
        {
            kblas_potrf_batch_strided_wsquery(*(data->kblas_handle[g]), data->M, data->batchCount_gpu);
            kblas_trsm_batch_strided_wsquery(*(data->kblas_handle[g]), data->side, data->M, data->N, data->batchCount_gpu);
        }
        /* TODO
        else if (data->nonUniform)
            kblas_trsm_batch_nonuniform_wsquery(*(data->kblas_handle[g]));
        else
        {
            kblas_potrf_batch_wsquery(*(data->kblas_handle[g]), data->M, data->batchCount_gpu);
            kblas_trsm_batch_wsquery(*(data->kblas_handle[g]), data->side, data->M, data->N, data->batchCount_gpu);
        }
        */
        check_kblas_error(kblasAllocateWorkspace(*(data->kblas_handle[g])));
        check_error(cudaGetLastError());
    }

    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    
    for (int g = 0; g < data->ngpu; g++)
    {   
        check_error(cudaSetDevice(data->devices[g]));
        /*
        cholesky decomposition
        */
        // printf("[info] Starting Cholesky decomposition. \n");
        // for (int i = 0; i < data->batchCount_gpu; i++)
        // {
        //     printf("%dth", i);
        //     printMatrixGPU(data->Am, data->An, data->d_A[g] + i * data->Am * data->ldda, data->ldda);
        // }
        if (data->strided)
        {
            check_kblas_error(kblas_potrf_batch(*(data->kblas_handle[g]),
                                                data->uplo, data->Am,
                                                data->d_A[g], data->ldda, data->An * data->ldda,
                                                data->batchCount_gpu,
                                                data->d_info[g]));
        }
        /*TODO
        else
        {
            check_kblas_error(kblas_potrf_batch(*(data->kblas_handle[g]),
                                                data->uplo, data->Am,
                                                d_A_array[g], data->ldda,
                                                data->batchCount_gpu,
                                                data->d_info[g]));
        }*/
        // for (int i = 0; i < data->batchCount_gpu; i++)
        // {
        //     printf("%dth", i);
        //     printMatrixGPU(data->Am, data->An, data->d_A[g] + i * data->Am * data->ldda, data->ldda);
        // }
        // printf("[info] Finished Cholesky decomposition. \n");
        /*
        determinant
        */
        if (data->strided)
        {
            for (int i = 0; i < data->batchCount_gpu; i++)
            {
                core_Xlogdet<T>(data->d_A[g] + i * data->An * data->ldda, data->An, data->ldda, &(data->logdet_result_h[g][i]));
                // printf("the det value is %lf \n", data->logdet_result_h[g][i]);
                // cudaDeviceSynchronize();
            }
            // printf("The results during log-det.");
            // printMatrixGPU(data->M, data->M, data->d_A[0] + data->An * data->ldda, data->ldda);
        }
        /* TODO
        else
        {
            // for (int i = 0; i < data->batchCount_gpu; i++)
            // {
            //     core_Xdet<T>(d_A_array[g] + i * data->An * data->lda, data->An, data->lda, &(data->logdet_result_h[g][i]));
            // }
            return 0;
        }
        */
        // printf("[info] Finished determinant. \n");
        /*
        triangular solution: L Z_new <- Z_old
        */
        // printf("[info] Starting triangular solver. \n");
        if (data->vecchia)
        {
            for (int i = 1; i < data->batchCount_gpu; i++)
            {
                // printf("The results before TRSM \n");
                // printVecGPU(data->Cm, data->Cn, data->d_C[g], data->ldc, i);
                // printf("The results before TRSM \n");
                // printVecGPU(data->Cm, data->Cn, data->d_C_copy[g]+ data->lddc * data->Cn * i, data->ldc, i);
                // printf("The results before TRSM \n");
                // printVecGPU(data->Cm, data->Cn, data->d_mu_offset[g]+ data->lddc * data->Cn * i, data->ldc, i);
                check_cublas_error(cublasXgeam(kblasGetCublasHandle(*(data->kblas_handle[g])),
                                                CUBLAS_OP_N, CUBLAS_OP_N,
                                                data->Cm, data->Cn,
                                                &alpha_1, // 1
                                                data->d_C_copy[g] + data->lddc * data->Cn * i, data->lddc,
                                                &beta_n1, // -1
                                                data->d_mu_offset[g] + data->lddc * data->Cn * i, data->lddc,
                                                data->d_C[g] + data->lddc * data->Cn * i, data->lddc));
                // printf("The results before TRSM \n");
                // printVecGPU(data->Cm, data->Cn, data->d_C[g] + i * data->Cn * data->lddc, data->ldc, i);
            }
            // printf("The results before TRSM \n");
            // printVecGPU(data->Cm, data->Cn, data->d_C[g] + data->Cn * data->lddc, data->ldc, 1);
            // printf("The results before TRSM \n");
            // printVecGPU(data->Cm, data->Cn, data->d_C[g], data->ldc, 0);
        }
        // printf("The results before TRSM.");
        // printMatrixGPU(data->M, data->M, data->d_A[0] + data->An * data->ldda, data->ldda);
        // for(int i=0; i<data->batchCount_gpu; i++){
        //     printVecGPU(data->Cm, data->Cn, data->d_C[g] + i * data->Cn * data->lddc, data->ldc, i);
        // }
        // for (int i = 0; i < data->batchCount_gpu; i++)
        // {
        //     printf("%dth", i);
        //     printMatrixGPU(data->Am, data->An, data->d_A[g] + i * data->Am * data->ldda, data->ldda);
        // }
        if (data->strided)
        {
            check_kblas_error(kblasXtrsm_batch_strided(*(data->kblas_handle[g]),
                                                        data->side, data->uplo, data->transA, data->diag,
                                                        data->An, data->Cn,
                                                        1.,
                                                        data->d_A[g], data->ldda, data->An * data->ldda,
                                                        data->d_C[g], data->lddc, data->Cn * data->lddc,
                                                        data->batchCount_gpu));
        }
        // for(int i=0; i<data->batchCount_gpu; i++){
        //     printVecGPU(data->Cm, data->Cn, data->d_C[g] + i * data->Cn * data->lddc, data->ldc, i);
        // }
        /* TODO
        else
        {
            check_cublas_error(
                cublasDtrsmBatched(kblasGetCublasHandle(*(data->kblas_handle[g])),
                                    CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_T, CUBLAS_DIAG_NON_UNIT,
                                    data->An, data->Cn,
                                    &1.,
                                    d_A_array[g], data->ldda,
                                    data->d_C_array[g], data->lddc,
                                    data->batchCount_gpu));
            // check_kblas_error(kblasXtrsm_batch(*(data->kblas_handle[g]),
            //                                    data->side, data->uplo, data->transA, data->diag,
            //                                    data->M, data->N,
            //                                    1., (const T **)(d_A_array[g]), data->ldda,
            //                                    data->d_C_array[g], data->lddc,
            //                                    data->batchCount_gpu));
        }
        */
        // printf("[info] Finished triangular solver. \n");
        /*
        Dot scalar Z_new^T Z_new
        */
        // printf("[info] Starting dot product. \n");
        if (data->strided)
        {
            for (int i = 0; i < data->batchCount_gpu; i++)
            {
                // printVecGPU(data->Cm, data->Cn, data->d_C[g] + i * data->Cn * data->lddc, data->ldc, i);
                check_cublas_error(cublasXdot(kblasGetCublasHandle(*(data->kblas_handle[g])), data->Cm,
                                                data->d_C[g] + i * data->Cn * data->lddc, 1,
                                                data->d_C[g] + i * data->Cn * data->lddc, 1,
                                                &(data->dot_result_h[g][i])));
                // printf("Dot product is %lf \n", data->dot_result_h[g][i]);
            }
            // cublasDnrm2( kblasGetCublasHandle(*(data->kblas_handle[g])), data->Cm, data->d_C[g], 1,  data->dot_result_h[g]);
        }
        /* TODO
        else if (data->nonUniform)
        {
            return 0;
        }
        else
        {
            // for (int i = 0; i < data->batchCount_gpu; i++)
            // {
            //     check_cublas_error(cublasXdot(kblasGetCublasHandle(*(data->kblas_handle[g])), data->Cm,
            //                                   data->d_C_array[g] + i * data->Cn * data->ldc, 1,
            //                                   data->d_C_array[g] + i * data->Cn * data->ldc, 1,
            //                                   &(data->dot_result_h[g][i])));
            // }
            return 0;
        }
        */
        // printf("[info] Finished dot product. \n");
    }
    for (int g = 0; g < data->ngpu; g++)
    {
        check_error(cudaSetDevice(data->devices[g]));
        cudaDeviceSynchronize(); // TODO sync with streams instead
    }
    // time = get_elapsed_time(curStream);
    clock_gettime(CLOCK_MONOTONIC, &end);
    indep_time = end.tv_sec - start.tv_sec + (end.tv_nsec - start.tv_nsec) / 1e9;

    // printf("-----------------------------------------\n");
    for (int g = 0; g < data->ngpu; g++)
    {   
        // printf("----------------%dth GPU---------------\n", g);
        for (int k = 0; k < data->batchCount_gpu; k++)
        {
            T llk_temp = 0;
            llk_temp = -(data->dot_result_h[g][k] + data->logdet_result_h[g][k] + data->Am * log(2 * PI)) * 0.5;
            llk += llk_temp;
            // printf("%dth log determinant is % lf\n", k, data->logdet_result_h[g][k]);
            // printf("%dth dot product is % lf\n", k, data->dot_result_h[g][k]);
            // printf("%dth pi is % lf\n", k, data->Am * log(2 * PI));
            // printf("%dth log likelihood is % lf\n", k, llk_temp);
            // printf("-------------------------------------\n");
        }
    }
    // printf("[info] Independent computing is finished! \n");
    // printf("[Info] The time for independent computing is %lf seconds\n", indep_time);
    // printf("[Info] The time for LLH is %lf seconds\n", indep_time + vecchia_time);
    // // printf("(Estimated) Sigma: %lf beta:  %lf  nu: %lf\n", localtheta[0], localtheta[1], localtheta[2]);
    // printf("Log likelihood is %lf \n", llk);
    saveLogFileParams(data->iterations, 
                    localtheta, llk, 
                    indep_time + vecchia_time, dcmg_time, 
                    data->num_loc, data->M,
                    data->zvecs, data->p); // this is log_tags for write a file
    if (data->kernel ==1){
        printf("%dth Model Parameters (Variance, range, smoothness): (%lf, %lf, %lf) -> Loglik: %lf \n", 
            data->iterations, localtheta[0], localtheta[1], localtheta[2], llk); 
    }else if (data->kernel ==2){
        printf("%dth Model Parameters (Variance1, Variance2, range, smoothness1, smoothness2, beta): (%lf, %lf, %lf, %lf, %lf, %lf) -> Loglik: %lf \n", 
            data->iterations, localtheta[0], localtheta[1], localtheta[2], 
            localtheta[3], localtheta[4], localtheta[5], llk); 
    }
    data->iterations += 1;
    data->vecchia_time_total += (indep_time + vecchia_time);  
    // printf("-----------------------------------------------------------------------------\n");

    // if (data->vecchia){
    //     // init for each iteration (necessary but low efficient)
    //     cudaFreeHost(data->locations_copy->x);
    //     cudaFreeHost(data->locations_copy->y);
    // }
    return llk;
}