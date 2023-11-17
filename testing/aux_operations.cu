#include <stdio.h>
#include <stdlib.h>
// #include <cublas_v2.h>

#include "aux_operations.h"

#define CHUNKSIZE 32

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// gpuDotProducts
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// CUDA kernel to calculate multiple dot products in parallel
__global__ void DgpuDotProducts_kernel(
        double *a, double *b, 
        double *results, 
        int numDotProducts, 
        int vectorSize) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    // Iterate over dot products
    for (int i = tid; i < numDotProducts; i += blockDim.x * gridDim.x) {
        double result = 0.0f;

        // Perform the dot product in parallel
        for (int j = 0; j < vectorSize; ++j) {
            result += a[i * vectorSize + j] * b[i * vectorSize + j];
        }

        // Store the result
        results[i] = result;
    }
}

void DgpuDotProducts(
        double *a, double *b, 
        double *results, 
        int numDotProducts, 
        int vectorSize,
        cudaStream_t stream) {
    
    int block_dim = 256;
    int grid_dim = (numDotProducts + block_dim - 1) / block_dim;
    
    DgpuDotProducts_kernel<<<grid_dim, block_dim, 0, stream>>>(a, b, results, numDotProducts, vectorSize);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// gpuDotProducts - strided version
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// CUDA kernel to calculate multiple dot products in parallel

__global__ void DgpuDotProducts_Strided_kernel(
        double *a, double *b, double *results, 
        int numDotProducts, 
        int vectorSize, 
        int lddvectorSize) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    // Iterate over dot products
    for (int i = tid; i < numDotProducts; i += blockDim.x * gridDim.x) {
        double result = 0.0f;

        // Perform the dot product in parallel
        for (int j = 0; j < vectorSize; ++j) {
            result += a[i * lddvectorSize + j] * b[i * lddvectorSize + j];
        }

        // Store the result
        results[i] = result;
    }
}

void DgpuDotProducts_Strided(double *a, double *b, 
        double *results, 
        int numDotProducts, 
        int vectorSize, 
        int lddvectorSize,
        cudaStream_t stream) {
    
    int block_dim = 256;
    int grid_dim = (numDotProducts + block_dim - 1) / block_dim;
    
    DgpuDotProducts_Strided_kernel<<<grid_dim, block_dim, 0, stream>>>(a, b, results, numDotProducts, vectorSize, lddvectorSize);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Covariance matrix generation 1/2 3/2 5/2 - strided version
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void Dcmg_matern12_strided_kernel(
        double *A, 
        int m, int n, int lddm,
        // int m0, int n0, 
        double* l1_x_cuda, double* l1_y_cuda, 
        double* l2_x_cuda, double* l2_y_cuda,
        double localtheta0, double localtheta1, 
        int distance_metric)
    /*!
     * Returns covariance matrix tile using the aforementioned kernel.
     * @param[in] A: 1D array which saves the matrix entries by column.
     * @param[in] m: number of rows of tile.
     * @param[in] n: number of columns of tile.
     * @param[in] lddm: leading dimension of columns of tile.
     * @param[in] m0: Global row start point of tile.
     * @param[in] n0: Global column start point of tile.
     * @param[in] l1_x_cuda: value of x-axis of locaton vector l1.
     * @param[in] l1_y_cuda: value of y-axis of locaton vector l1.
     * @param[in] l2_x_cuda: value of x-axis of locaton vector l2.
     * @param[in] l2_y_cuda: value of y-axis of locaton vector l2.
     * @param[in] localtheta: there are three parameters to define this kernel.
     * @param[in] distance_metric: unused arguments
     * */
{
   int idx = blockIdx.x * blockDim.x + threadIdx.x;
   int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if(idx >= m || idy >= n) return;

    double scaled_distance  = 0.0;
    // double expr1 = 0.0;
    double sigma_square = localtheta0;
    scaled_distance = sqrt(
                pow((l2_x_cuda[idy] - l1_x_cuda[idx]), 2) +
                pow((l2_y_cuda[idy] - l1_y_cuda[idx]), 2)
            ) / localtheta1;

    A[idx + idy * lddm] = sigma_square *  
                    exp(-(scaled_distance)); // power-exp kernel
    // }
    

}

__global__ void Dcmg_matern32_strided_kernel(
        double *A, 
        int m, int n, int lddm,
        // int m0, int n0, 
        double* l1_x_cuda, double* l1_y_cuda, 
        double* l2_x_cuda, double* l2_y_cuda,
        double localtheta0, double localtheta1, 
        int distance_metric)
{
   int idx = blockIdx.x * blockDim.x + threadIdx.x;
   int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if(idx >= m || idy >= n) return;

    double scaled_distance = 0.0;
    // double expr1 = 0.0;
    double sigma_square = localtheta0;
    scaled_distance = sqrt(
                pow((l2_x_cuda[idy] - l1_x_cuda[idx]), 2) +
                pow((l2_y_cuda[idy] - l1_y_cuda[idx]), 2)
            ) / localtheta1;
    A[idx + idy * lddm] = sigma_square * 
                (1 + scaled_distance) *
                exp(-scaled_distance);
}

__global__ void Dcmg_matern52_strided_kernel(
        double *A, 
        int m, int n, int lddm,
        double* l1_x_cuda, double* l1_y_cuda, 
        double* l2_x_cuda, double* l2_y_cuda,
        double localtheta0, double localtheta1, 
        int distance_metric)
{
   int idx = blockIdx.x * blockDim.x + threadIdx.x;
   int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if(idx >= m || idy >= lddm) return;

    double scaled_distance  = 0.0;
    double sigma_square = localtheta0;
    scaled_distance = sqrt(
                pow((l2_x_cuda[idy] - l1_x_cuda[idx]), 2) +
                pow((l2_y_cuda[idy] - l1_y_cuda[idx]), 2)
            ) / localtheta1;
    A[idx + idy * lddm] = sigma_square * 
            (1 + scaled_distance + pow(scaled_distance, 2) / 3) * 
            exp(-scaled_distance); 
}


__global__ void Dcmg_powexp_strided_kernel(
        double *A, 
        int m, int n, int lddm,
        // int m0, int n0, 
        double* l1_x_cuda, double* l1_y_cuda, 
        double* l2_x_cuda, double* l2_y_cuda,
        double localtheta0, double localtheta1, 
        double localtheta2, 
        int distance_metric)
{
   int idx = blockIdx.x * blockDim.x + threadIdx.x;
   int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if(idx >= m || idy >= n) return;

    double expr  = 0.0;
    double expr1 = 0.0;
    double sigma_square = localtheta0;
    expr = sqrt(pow((l2_x_cuda[idy] - l1_x_cuda[idx]), 2) +
            pow((l2_y_cuda[idy] - l1_y_cuda[idx]), 2));
    expr1 = pow(expr, localtheta2);
    A[idx + idy * lddm] = sigma_square *  exp(-(expr1/localtheta1)); 
}


void cudaDcmg_matern135_2_strided( 
        double *A, 
        int m, int n, int lddm,
        // int m0, int n0, 
        double* l1_x_cuda, double* l1_y_cuda, 
        double* l2_x_cuda, double* l2_y_cuda,
        const double *localtheta, int distance_metric, 
        cudaStream_t stream){

    // Matern function with fraction 1/2, 3/2, 5/2

    int nBlockx = (m + CHUNKSIZE - 1)/CHUNKSIZE;
    int nBlocky = (n + CHUNKSIZE - 1)/CHUNKSIZE;
    dim3 dimBlock(CHUNKSIZE,CHUNKSIZE);
    dim3 dimGrid(nBlockx,nBlocky);

    if (localtheta[2] == 0.5){
        Dcmg_matern12_strided_kernel<<<dimGrid, dimBlock, 0, stream>>>(
            A, m, n, lddm, 
            l1_x_cuda, l1_y_cuda, 
            l2_x_cuda, l2_y_cuda, 
            localtheta[0], localtheta[1], 
            distance_metric
        );
    }else if (localtheta[2] == 1.5){
        Dcmg_matern32_strided_kernel<<<dimGrid, dimBlock, 0, stream>>>(
            A, m, n, lddm, 
            l1_x_cuda, l1_y_cuda, 
            l2_x_cuda, l2_y_cuda, 
            localtheta[0], localtheta[1], 
            distance_metric
        );
    }else if (localtheta[2] == 2.5){
        Dcmg_matern52_strided_kernel<<<dimGrid, dimBlock, 0, stream>>>(
            A, m, n, lddm, 
            l1_x_cuda, l1_y_cuda, 
            l2_x_cuda, l2_y_cuda, 
            localtheta[0], localtheta[1], 
            distance_metric
        );
    }else{
        fprintf(stderr, "Other smoothness setting are still developing. \n");
        exit(0);
    }
    
}


void cudaDcmg_powexp_strided( 
        double *A, 
        int m, int n, 
        int lddm,
        // int m0, int n0, 
        double* l1_x_cuda, double* l1_y_cuda, 
        double* l2_x_cuda, double* l2_y_cuda,
        const double *localtheta, int distance_metric, 
        cudaStream_t stream){

    int nBlockx= (m + CHUNKSIZE - 1) / CHUNKSIZE;
    int nBlocky= (n + CHUNKSIZE - 1) / CHUNKSIZE;
    dim3 dimBlock(CHUNKSIZE, CHUNKSIZE);
    dim3 dimGrid(nBlockx, nBlocky);

    Dcmg_powexp_strided_kernel<<<dimGrid, dimBlock, 0, stream>>>(
        A, m, n, lddm, 
        l1_x_cuda, l1_y_cuda, 
        l2_x_cuda, l2_y_cuda, 
        localtheta[0], localtheta[1], 
        localtheta[2],
        distance_metric);

    // cudaStreamSynchronize(stream);
}