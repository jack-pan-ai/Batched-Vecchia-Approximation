#include <cmath>

/*
dot product strided version
*/
cublasStatus_t cublasXdot(cublasHandle_t handle, int n,
                          const double *x, int incx,
                          const double *y, int incy,
                          double *result)
{
  return cublasDdot(handle, n, x, incx, y, incy, result);
}

cublasStatus_t cublasXdot(cublasHandle_t handle, int n,
                          const float *x, int incx,
                          const float *y, int incy,
                          float *result)
{
  return cublasSdot(handle, n, x, incx, y, incy, result);
}
cublasStatus_t cublasXdot(cublasHandle_t handle, int n,
                          const cuComplex *x, int incx,
                          const cuComplex *y, int incy,
                          cuComplex *result)
{
  return cublasCdotc(handle, n, x, incx, y, incy, result);
}

cublasStatus_t cublasXdot(cublasHandle_t handle, int n,
                          const cuDoubleComplex *x, int incx,
                          const cuDoubleComplex *y, int incy,
                          cuDoubleComplex *result)
{
  return cublasZdotc(handle, n, x, incx, y, incy, result);
}


#define FMULS_GEVV(m_) ((m_) * (m_))
#define FADDS_GEVV(m_) ((m_) * (m_))

template <class T>
double FLOPS_GEVV(int m)
{
  return (is_complex(T) ? 6. : 1.) * FMULS_GEVV((double)(m)) + (is_complex(T) ? 2. : 1.) * FADDS_GEVV((double)(m));
}

/*
determinant for log(det(A)) = log(det(L)det(L^T))
strided version
*/

template <class T>
void core_Xlogdet(T *L, int An, int ldda, T *logdet_result_h)
{
  T *L_h = (T *)malloc(sizeof(T) * An * ldda);
  cudaMemcpy(L_h, L, sizeof(T) * An * ldda, cudaMemcpyDeviceToHost);
  * logdet_result_h = 0;
  for (int i = 0; i < An; i++)
  {
    // printf("%d L diagnal value %lf\n", i, L_h[i + i * ldda]);
    // printf("%d the value is %lf \n", i, * logdet_result_h);
    // printf("%d the value is %p \n", i, logdet_result_h);
    if (L_h[i + i * ldda] > 0)
      * logdet_result_h += log(L_h[i + i * ldda] * L_h[i + i * ldda]);
  }
  // printf("the value is %lf \n", * logdet_result_h);
  // printf("-----------------------------------");
  free(L_h);
}

/*
non-strided version

template <class T>
void core_Xlogdet(T **L, int An, int lda, T *logdet_result_h)
{
  T *L_h = (T *)malloc(sizeof(T) * An * lda);
  cudaMemcpy(L_h, L, sizeof(T) * An * lda, cudaMemcpyDeviceToHost);
  for (int i = 0; i < An; i++)
  {
    // printf("L diagnal value %lf\n", L_h[i + i * m]);
    if (L_h[i + i * lda] > 0)
      * logdet_result_h += log(L_h[i + i * lda] * L_h[i + i * lda]);
  }
  // printf("the value is %lf \n", logdet_result_h[0]);
  free(L_h);
}
*/