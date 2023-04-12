/*
    -- vecchia (version 1.0.0) --
       King Abdullah Uni. of Sci. and Tech.
       @date April 2023

       @generated from tempplate of magmablas/zgeadd_batched.cu
       @author Qilong Pan
*/
#include "dcmg_helper.h"

#define MAX_BATCHCOUNT    (65534)
#define NB 1024

/******************************************************************************/
/*
    Batches dlacpy of multiple arrays;
    y-dimension of grid is different arrays,
    x-dimension of grid is blocks for each array.
    Matrix is m x n, and is divided into block rows, each NB x n.
    Each CUDA block has NB threads to handle one block row.
    Each thread adds one row, iterating across all columns.
    The bottom block of rows may be partially outside the matrix;
    if so, rows outside the matrix (i >= m) are disabled.

    TODO. Block in both directions, for large matrices.
    E.g., each block does 64x64 tile, instead of 64xN tile.
*/
__global__ void
ddcmg_batched_kernel(
    int m, int n,
    double alpha,
    const double * const *dAarray, int ldda,
    double              **dBarray, int lddb )
{
    // dA and dB iterate across row i
    const double *dA = dAarray[ blockIdx.y ];
    double       *dB = dBarray[ blockIdx.y ];
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if ( i < m ) {
        dA += i;
        dB += i;
        const double *dAend = dA + n*ldda;
        while( dA < dAend ) {
            *dB = alpha*(*dA) + (*dB);
            dA += ldda;
            dB += lddb;
        }
    }
}

__global__ void
ddcmg_batched_kernel(
    int m, int n,
    double *alpha,
    const double * const *dAarray, int ldda,
    double              **dBarray, int lddb )
{
    // dA and dB iterate across row i
    const double *dA = dAarray[ blockIdx.y ];
    double       *dB = dBarray[ blockIdx.y ];
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if ( i < m ) {
        dA += i;
        dB += i;
        const double *dAend = dA + n*ldda;
        while( dA < dAend ) {
            *dB = alpha*(*dA) + (*dB);
            dA += ldda;
            dB += lddb;
        }
    }
}


/***************************************************************************//**
    Purpose
    -------
    ZGEADD adds two sets of matrices, dAarray[i] = alpha*dAarray[i] + dBarray[i],
    for i = 0, ..., batchCount-1.

    Arguments
    ---------

    @param[in]
    m       INTEGER
            The number of rows of each matrix dAarray[i].  M >= 0.

    @param[in]
    n       INTEGER
            The number of columns of each matrix dAarray[i].  N >= 0.

    @param[in]
    alpha   DOUBLE PRECISION
            The scalar alpha.

    @param[in]
    dAarray array on GPU, dimension(batchCount), of pointers to arrays,
            with each array a DOUBLE PRECISION array, dimension (LDDA,N)
            The m by n matrices dAarray[i].

    @param[in]
    ldda    INTEGER
            The leading dimension of each array dAarray[i].  LDDA >= max(1,M).

    @param[in,out]
    dBarray array on GPU, dimension(batchCount), of pointers to arrays,
            with each array a DOUBLE PRECISION array, dimension (LDDB,N)
            The m by n matrices dBarray[i].

    @param[in]
    lddb    INTEGER
            The leading dimension of each array dBarray[i].  LDDB >= max(1,M).

    @param[in]
    batchCount INTEGER
            The number of matrices to add; length of dAarray and dBarray.
            batchCount >= 0.

    @param[in]
    queue   magma_queue_t
            Queue to execute in.

    @ingroup magma_geadd_batched
*******************************************************************************/
extern "C" void
ddcmg_batched(
    int const m, int const n,
    double const *alpha,
    location* l1,
	location* l2, 
    double* localtheta,
    double* A, int const ldda,
    int const batchCount,
    cudaStream_t stream)
{
    magma_int_t info = 0;
    if ( m < 0 )
        info = -1;
    else if ( n < 0 )
        info = -2;
    else if ( ldda < max(1,m))
        info = -5;
    else if ( lddb < max(1,m))
        info = -7;
    else if ( batchCount < 0 )
        info = -8;

    if ( info != 0 ) {
        fprintf(stderr, "The input size (m, n, ldda, lddb, batchCount) is illegal\n"); 
        return;
    }

    if ( m == 0 || n == 0 || batchCount == 0 )
        fprintf(stderr, "Hey, dude, please give me valid size (m, n, ldda, lddb, batchCount)\n"); 
        return;

    dim3 threads( NB ); // each thread compute the covariance for a row
    int max_batchCount = MAX_BATCHCOUNT;

    for(int i = 0; i < batchCount; i+=max_batchCount) {
        int ibatch = min(max_batchCount, batchCount-i);
        dim3 grid( ceildiv( m, NB ), ibatch );  // batched operation for each block, block -> a small matrix

        ddcmg_batched_kernel<<< grid, threads, 0, stream >>>
        (m, n, *alpha, dAarray+i, ldda, dBarray+i, lddb );
    }
}
