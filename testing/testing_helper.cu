/**
 * @copyright (c) 2012- King Abdullah University of Science and
 *                      Technology (KAUST). All rights reserved.
 **/


/**
 * @file testing/testing_helper.cu

 * KBLAS is a high performance CUDA library for subset of BLAS
 *    and LAPACK routines optimized for NVIDIA GPUs.
 * KBLAS is provided by KAUST.
 *
 * @version 3.0.0
 * @author Wajih Halim Boukaram
 * @author Ali Charara
 * @date 2018-11-14
 **/

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/transform.h>
#include <thrust/execution_policy.h>
#include <thrust/system/cuda/execution_policy.h>
#include <thrust/system/omp/execution_policy.h>
#include <thrust/random.h>
#include <thrust/copy.h>
#include <thrust/logical.h>

#include <sys/time.h>
#include <stdarg.h>

#include <cmath>

#include "testing_helper.h"

// ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// // Generating array of pointers from a strided array
// ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// template<class T>
// struct UnaryAoAAssign : public thrust::unary_function<int, T*>
// {
//   T* original_array;
//   int stride;
//   UnaryAoAAssign(T* original_array, int stride) { this->original_array = original_array; this->stride = stride; }
//   __host__ __device__
//   T* operator()(const unsigned int& thread_id) const { return original_array + thread_id * stride; }
// };

// template<class T>
// void generateArrayOfPointersT(T* original_array, T** array_of_arrays, int stride, int num_arrays, cudaStream_t stream)
// {
//   thrust::device_ptr<T*> dev_data(array_of_arrays);

//   thrust::transform(
//     thrust::cuda::par.on(stream),
//     thrust::counting_iterator<int>(0),
//     thrust::counting_iterator<int>(num_arrays),
//     dev_data,
//     UnaryAoAAssign<T>(original_array, stride)
//     );

//   check_error( cudaGetLastError() );
// }

// extern "C" void generateDArrayOfPointers(double* original_array, double** array_of_arrays, int stride, int num_arrays, cudaStream_t stream)
// { generateArrayOfPointersT<double>(original_array, array_of_arrays, stride, num_arrays, stream); }

// extern "C" void generateSArrayOfPointers(float* original_array, float** array_of_arrays, int stride, int num_arrays, cudaStream_t stream)
// { generateArrayOfPointersT<float>(original_array, array_of_arrays, stride, num_arrays, stream); }

// extern "C" void generateDArrayOfPointersHost(double* original_array, double** array_of_arrays, int stride, int num_arrays)
// { for(int i = 0; i < num_arrays; i++) array_of_arrays[i] = original_array + i * stride; }

// extern "C" void generateSArrayOfPointersHost(float* original_array, float** array_of_arrays, int stride, int num_arrays)
// { for(int i = 0; i < num_arrays; i++) array_of_arrays[i] = original_array + i * stride; }


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Error helpers
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

const char* cublasGetErrorString(cublasStatus_t error)
{
  switch(error)
  {
    case CUBLAS_STATUS_SUCCESS:
    return "success";
    case CUBLAS_STATUS_NOT_INITIALIZED:
    return "not initialized";
    case CUBLAS_STATUS_ALLOC_FAILED:
    return "out of memory";
    case CUBLAS_STATUS_INVALID_VALUE:
    return "invalid value";
    case CUBLAS_STATUS_ARCH_MISMATCH:
    return "architecture mismatch";
    case CUBLAS_STATUS_MAPPING_ERROR:
    return "memory mapping error";
    case CUBLAS_STATUS_EXECUTION_FAILED:
    return "execution failed";
    case CUBLAS_STATUS_INTERNAL_ERROR:
    return "internal error";
    default:
    return "unknown error code";
  }
}


extern "C" void gpuCublasAssert(cublasStatus_t code, const char *file, int line)
{
	if(code != CUBLAS_STATUS_SUCCESS)
	{
    printf("gpuCublasAssert: %s %s %d\n", cublasGetErrorString(code), file, line);
    exit(-1);
  }
}

extern "C" void gpuKblasAssert(int code, const char *file, int line)
{
	if(code != 1)  // TODO replace by KBlas_Success
	{
    printf("gpuKblasAssert: %s %s %d\n", kblasGetErrorString(code), file, line);
    exit(-1);
  }
}
////////////////////////////////////////////////////////////
// Command line parser
////////////////////////////////////////////////////////////
static const char *usage =
"  -N               Batchsize, e.g., 10:1, and batchsize is 10.\n"
"  --dev x          GPU device to use, default 0.\n"
"  -s               Strided version, the default"
"\n"
"examples: \n"
"to test trmm with matrix A[512,512], B[2000,512] do\n"
"   test_dtrmm -N 2000:512 \n"
"to test trmm for range of sizes starting at 1024, stoping at 4096, steping at 1024, sizes will be for both A and B, with A upper traingular and transposed, do\n"
"   test_dtrmm --range 1024:4096:1024 -U -T";


#define USAGE printf("usage: -N m[:n] --range m-start:m-end:m-step -m INT -n INT -L|U -SL|SR -DN|DU -[NTC][NTC] -c --niter INT --dev devID\n\n"); \
printf("%s\n", usage);

#define USING printf("side %c, uplo %c, trans %c, diag %c, db %d\n", opts.side, opts.uplo, opts.transA, opts.diag, opts.db);

void kblas_assert( int condition, const char* msg, ... )
{
	if ( ! condition ) {
		printf( "Assert failed: " );
		va_list va;
		va_start( va, msg );
		vprintf( msg, va );
		printf( "\n" );
		exit(1);
	}
}

extern "C" int parse_opts(int argc, char** argv, kblas_opts *opts)
{
  // fill in default values
  for(int d = 0; d < MAX_NGPUS; d++)
    opts->devices[d] = d;

  opts->nstream    = 1;
  opts->ngpu       = 1;
  opts->tolerance  = 0.;
  opts->time       = 0;
  opts->nonUniform = 0; // TBD
  // opts->batchCount = 4;
  opts->strided    = 1; // TBD

  // local theta for kernel in GPs
  opts->sigma     = 0.1;
  opts->beta     = 0.1;
  opts->nu     = 0.1;
  opts->nugget = 0.0;

  // local theta for kernel in GPs
  opts->sigma_init     = 0.01;
  opts->beta_init     = 0.01;
  opts->nu_init     = 0.01;
  opts->nugget_init = 0.01;

  // bivariate
  opts->sigma1     = 0.1;
  opts->sigma2     = 0.1;
  opts->alpha     = 0.1;
  opts->nu1     = 0.1;
  opts->nu2     = 0.1;
  
  // performance test
  opts->perf = 0;

  // vecchia conditioning
  opts->vecchia = 1; 
  opts->vecchia_cs =0; 
  opts->test =0;

  // optimization setting
  opts->tol = 1e-9;
  opts->maxiter = 2000;
  opts->lower_bound = 0.01;
  opts->upper_bound = 3.;

  // openmp
  opts->omp_numthreads = 40;

  //extra config
  opts->kernel = 1;
  opts->num_params = 3;
  opts->num_loc = 20000;

  // bivariate 
  opts->p = 1; // univaraite

  // k nearest neighbors
  opts->knn = 0;

  // random ordering
  opts->randomordering = 0;
  opts->mortonordering = 1;
  opts->kdtreeordering = 0;
  opts->hilbertordering = 0;
  opts->mmdordering = 0;

  // irregular locations generation 
  opts->seed = 0;

  int ndevices;
  cudaGetDeviceCount( &ndevices );
  int info;
  int ntest = 0;
  for( int i = 1; i < argc; ++i ) {
    // ----- matrix size
    // each -N fills in next entry of msize, nsize, ksize and increments ntest
    if ( strcmp("-N", argv[i]) == 0 && i+1 < argc ) {
      kblas_assert( ntest < MAX_NTEST, "error: -N %s, max number of tests exceeded, ntest=%d.\n", argv[i], ntest );
      i++;
      int m2, n2, k2, q2;
      info = sscanf( argv[i], "%d:%d:%d:%d", &m2, &n2, &k2, &q2 );
      if ( info == 4 && m2 >= 0 && n2 >= 0 && k2 >= 0 && q2 >= 0 ) {
        opts->msize[ ntest ] = m2;
        opts->nsize[ ntest ] = n2;
        opts->ksize[ ntest ] = k2;
        opts->rsize[ ntest ] = q2;
      }
      else
      if ( info == 3 && m2 >= 0 && n2 >= 0 && k2 >= 0 ) {
        opts->msize[ ntest ] = m2;
        opts->nsize[ ntest ] = n2;
        opts->ksize[ ntest ] = k2;
          opts->rsize[ ntest ] = k2;  // implicitly
      }
      else
      if ( info == 2 && m2 >= 0 && n2 >= 0 ) {
        opts->msize[ ntest ] = m2;
        opts->nsize[ ntest ] = n2;
        opts->ksize[ ntest ] = n2;  // implicitly
        opts->rsize[ ntest ] = n2;  // implicitly
      }
      else
      if ( info == 1 && m2 >= 0 ) {
        opts->msize[ ntest ] = m2;
        opts->nsize[ ntest ] = m2;  // implicitly
        opts->ksize[ ntest ] = m2;  // implicitly
        opts->rsize[ ntest ] = m2;  // implicitly
      }
      else {
        fprintf( stderr, "error: -N %s is invalid; ensure m >= 0, n >= 0, k >= 0, info=%d, m2=%d, n2=%d, k2=%d, q2=%d.\n",
          argv[i],info,m2,n2,k2,q2 );
        exit(1);
      }
      ntest++;
    }
    
    // ----- scalar arguments
    else if ( strcmp("--dev", argv[i]) == 0 && i+1 < argc ) {
      int n;
      info = sscanf( argv[++i], "%d", &n );
      if ( info == 1) {
        char inp[512];
        char * pch;
        int ngpus = 0;
        strcpy(inp, argv[i]);
        pch = strtok (inp,",");
        do{
          info = sscanf( pch, "%d", &n );
          if ( ngpus >= MAX_NGPUS ) {
            printf( "warning: selected number exceeds KBLAS max number of GPUs, ngpus=%d.\n", ngpus);
            break;
          }
          if ( ngpus >= ndevices ) {
            printf( "warning: max number of available devices reached, ngpus=%d.\n", ngpus);
            break;
          }
          if ( n >= ndevices || n < 0) {
            printf( "error: device %d is invalid; ensure dev in [0,%d].\n", n, ndevices-1 );
            break;
          }
          opts->devices[ ngpus++ ] = n;
          pch = strtok (NULL,",");
        }while(pch != NULL);
        opts->ngpu = ngpus;
      }
      else {
        fprintf( stderr, "error: --dev %s is invalid; ensure you have comma seperated list of integers.\n",
         argv[i] );
        exit(1);
      }
      kblas_assert( opts->ngpu > 0 && opts->ngpu <= ndevices,
       "error: --dev %s is invalid; ensure dev in [0,%d].\n", argv[i], ndevices-1 );
    }
    else if ( strcmp("--ngpu",    argv[i]) == 0 && i+1 < argc ) {
      opts->ngpu = atoi( argv[++i] );
      kblas_assert( opts->ngpu <= MAX_NGPUS ,
        "error: --ngpu %s exceeds MAX_NGPUS, %d.\n", argv[i], MAX_NGPUS  );
      kblas_assert( opts->ngpu <= ndevices,
       "error: --ngpu %s exceeds number of CUDA devices, %d.\n", argv[i], ndevices );
      kblas_assert( opts->ngpu > 0,
       "error: --ngpu %s is invalid; ensure ngpu > 0.\n", argv[i] );
    }
    else if ( strcmp("--nstream", argv[i]) == 0 && i+1 < argc ) {
      opts->nstream = atoi( argv[++i] );
      kblas_assert( opts->nstream > 0,
       "error: --nstream %s is invalid; ensure nstream > 0.\n", argv[i] );
    }
    else if ( strcmp("--omp_threads", argv[i]) == 0 && i+1 < argc ) {
      opts->omp_numthreads = atoi( argv[++i] );
      kblas_assert( opts->omp_numthreads >= 1,
        "error: --omp_numthreads %s is invalid; ensure omp_numthreads >= 1.\n", argv[i] );
    }
    else if ( strcmp("--strided",  argv[i]) == 0 || strcmp("-s",  argv[i]) == 0 ) { opts->strided = 1;  }
    // used for performance test
    else if ( strcmp("--perf", argv[i]) == 0 ) {
       opts->perf  = 1; 
       opts->maxiter = 1;
      }
    // used for vecchia conditioning
    else if ( strcmp("--test", argv[i]) == 0 ) { opts->test  = 1;    }
    // else if ( strcmp("--vecchia", argv[i]) == 0 ) {
    //    opts->vecchia  = 1;
    //   }
    else if ( (strcmp("--vecchia_cs",   argv[i]) == 0) && i+1 < argc ) {
      i++;
      int num;
      info = sscanf( argv[i], "%d", &num);
      if( info == 1 && num >= 0 ){
        opts->vecchia_cs = num;
        opts->vecchia = 1;
      // }else if(info == 1 && num == 0){
      //   opts->vecchia_cs = 0;
      //   opts->vecchia = 0;
      }else{
        fprintf( stderr, "error: --vecchia_cs %s is invalid; ensure only one number and 0 < vecchia_cs <= N.\n", argv[i]);
        exit(1);
      }
    }
    //used for optimization
    else if ( (strcmp("--maxiter",   argv[i]) == 0) && i+1 < argc ) {
      i++;
      int maxiter;
      info = sscanf( argv[i], "%d", &maxiter);
      if( info == 1 && maxiter > 0 ){
        opts->maxiter = maxiter;
      }else{
        fprintf( stderr, "error: --maxiter %s is invalid; ensure maxiter > 0 and be integer.\n", argv[i]);
        exit(1);
      }
    }
    else if ( (strcmp("--tol",   argv[i]) == 0) && i+1 < argc ) {
      i++;
      int tol;
      info = sscanf( argv[i], "%d", &tol);
      if( info == 1 && tol > 0 ){
        opts->tol = pow(10, -tol);
      }else{
        fprintf( stderr, "error: --tol %s is invalid; ensure tol > 0.\n", argv[i]);
        exit(1);
      }
    }
    else if ( (strcmp("--lower_bound",   argv[i]) == 0) && i+1 < argc ) {
      i++;
      double lower_bound;
      info = sscanf( argv[i], "%lf", &lower_bound);
      if( info == 1 && lower_bound > 0 ){
        opts->lower_bound = lower_bound;
      }else{
        fprintf( stderr, "error: --lower_bound %s is invalid; ensure lower_bound > 0.\n", argv[i]);
        exit(1);
      }
    }
    else if ( (strcmp("--upper_bound",   argv[i]) == 0) && i+1 < argc ) {
      i++;
      double upper_bound;
      info = sscanf( argv[i], "%lf", &upper_bound);
      if( info == 1 && upper_bound < 100 ){
        opts->upper_bound = upper_bound;
      }else{
        fprintf( stderr, "error: --upper_bound %s is invalid; ensure upper_bound < 100. (Or you fix 100 in opts file)\n", argv[i]);
        exit(1);
      }
    }
    // --- extra config
    else if ( (strcmp("--kernel", argv[i]) == 0) && i+1 < argc ) {
       i++;
        char* kernel_str = argv[i];

        if (strcmp(kernel_str, "univariate_matern_stationary_no_nugget") == 0) {
            fprintf(stderr, "You are using the Matern Kernel (sigma^2, range, smooth)!\n");
          opts->kernel = 1; // You can change this value as needed
          opts->num_params = 3; // Set appropriate values for the 'matern' kernel
          opts->p = 1; // You can modify this as per the requirement for 'matern'
        } else if (strcmp(kernel_str, "univariate_powexp_stationary_no_nugget") == 0) {
            fprintf(stderr, "You are using the Power exponential Kernel (sigma^2, range, smooth)!\n");
            opts->kernel = 2; // Change as per your requirement for 'powexp'
            opts->num_params = 3; // Set appropriate values for the 'powexp' kernel
            opts->p = 1; // Modify as needed for 'powexp'
        } else if (strcmp(kernel_str, "univariate_powexp_nugget_stationary") == 0) {
          fprintf(stderr, "You are using the Power exponential Kernel with nugget (sigma^2, range, smooth, nugget)!\n");
          opts->kernel = 3; // 
          opts->num_params = 4; // 
          opts->p = 1; // Modify as needed for 'powexp'
        }else {
            fprintf(stderr, "Unsupported kernel type: %s\n", kernel_str);
            exit(1);
        }
    }
    else if ( (strcmp("--num_loc",   argv[i]) == 0) && i+1 < argc ) {
      i++;
      int num_loc;
      info = sscanf( argv[i], "%d", &num_loc);
      opts->num_loc=num_loc;
    }
    // k nearest neighbors
    else if ( strcmp("--knn", argv[i]) == 0 ) {
      opts->knn  = 1; 
    }
    // ordering 
    else if ( strcmp("--randomordering", argv[i]) == 0 ){
      opts->randomordering  = 1;
      opts->mortonordering  = 0;
      opts->kdtreeordering  = 0;
      opts->hilbertordering = 0;
      opts->mmdordering = 0;
    }
    else if ( strcmp("--kdtreeordering", argv[i]) == 0 ){
      opts->randomordering  = 0;
      opts->mortonordering  = 0;
      opts->kdtreeordering  = 1;
      opts->hilbertordering = 0;
      opts->mmdordering = 0;
    }
    else if ( strcmp("--hilbertordering", argv[i]) == 0 ){
      opts->randomordering  = 0;
      opts->mortonordering  = 0;
      opts->kdtreeordering  = 0;
      opts->hilbertordering = 1;
      opts->mmdordering = 0;
    }
    else if ( strcmp("--mmdordering", argv[i]) == 0 ){
      opts->randomordering  = 0;
      opts->mortonordering  = 0;
      opts->kdtreeordering  = 0;
      opts->hilbertordering = 0;
      opts->mmdordering = 1;
    }
    // ture parameters
    else if ( strcmp("--ikernel", argv[i]) == 0 && i+1 < argc ) {
       i++;
     double a1 = -1, a2 = -1, a3 = -1, a4 = -1; // Initialize with default values indicating 'unknown'
     char s1[10], s2[10], s3[10], s4[10]; // Arrays to hold the string representations
     // Parse the input into string buffers
     int info = sscanf(argv[i], "%9[^:]:%9[^:]:%9[^:]:%9[^:]", s1, s2, s3, s4);
     if (info < 3 || info > 4) {
       printf("Other kernels have been developing on the way!");
       exit(0);
     }
     // Check and convert each value
     if (strcmp(s1, "?") != 0) a1 = atof(s1);
     if (strcmp(s2, "?") != 0) a2 = atof(s2);
     if (strcmp(s3, "?") != 0) a3 = atof(s3);
     if ( info == 4){
       if (strcmp(s4, "?") != 0) a4 = atof(s4);
     } 
     // Assign values to opts if they are not unknown
     if (a1 != -1) opts->sigma = a1;
     if (a2 != -1) opts->beta = a2;
     if (a3 != -1) opts->nu = a3;
     if ( info == 4){
       if (a4 != -1) opts->nugget = a4;
     } 
    }
    // initi parameters
    else if ( strcmp("--kernel_init", argv[i]) == 0 && i+1 < argc ) {
       i++;
     double a1 = -1, a2 = -1, a3 = -1, a4 = -1; // Initialize with default values indicating 'unknown'
     char s1[10], s2[10], s3[10], s4[10]; // Arrays to hold the string representations
     // Parse the input into string buffers
     int info = sscanf(argv[i], "%9[^:]:%9[^:]:%9[^:]:%9[^:]", s1, s2, s3, s4);
     if (info < 3 || info > 4) {
       printf("Other kernels have been developing on the way!");
       exit(0);
     }
     // Check and convert each value
     if (strcmp(s1, "?") != 0) a1 = atof(s1);
     if (strcmp(s2, "?") != 0) a2 = atof(s2);
     if (strcmp(s3, "?") != 0) a3 = atof(s3);
     if ( info == 4){
       if (strcmp(s4, "?") != 0) a4 = atof(s4);
     } 
     // Assign values to opts if they are not unknown
     if (a1 != -1) opts->sigma_init = a1;
     if (a2 != -1) opts->beta_init = a2;
     if (a3 != -1) opts->nu_init = a3;
     if ( info == 4){
       if (a4 != -1) opts->nugget_init = a4;
     } 
    }
    // iiregular locations generation seeds
    else if ( (strcmp("--seed",   argv[i]) == 0) && i+1 < argc ) {
      i++;
      int seed;
      info = sscanf( argv[i], "%d", &seed);
      opts->seed=seed;
    }
    // ----- usage
    else if ( strcmp("-h",     argv[i]) == 0 || strcmp("--help", argv[i]) == 0 ) {
      USAGE
      exit(0);
    }
    else {
      fprintf( stderr, "error: unrecognized option %s\n", argv[i] );
      exit(1);
    }
  }
  kblas_assert( ntest <= MAX_NTEST, "error: tests exceeded max allowed tests!\n" );
  opts->ntest = ntest;


  // set device
  cudaError_t ed = cudaSetDevice( opts->devices[0] );
  if(ed != cudaSuccess){
    printf("Error setting device : %s \n", cudaGetErrorString(ed) ); exit(-1);
  }

  return 1;
}// end parse_opts
