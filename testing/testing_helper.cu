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
  opts->sigma     = 0.;
  opts->beta     = 0.;
  opts->nu     = 0.;
  opts->sigma1     = 0.;
  opts->sigma2     = 0.;
  opts->alpha     = 0.;
  opts->nu1     = 0.;
  opts->nu2     = 0.;
  opts->beta     = 0.;
  
  // performance test
  opts->perf = 0;

  // vecchia conditioning
  opts->vecchia = 1; 
  opts->vecchia_cs =0; 
  opts->test =0;

  // optimization setting
  opts->tol = 1e-5;
  opts->maxiter = 2000;
  opts->lower_bound = 0.01;
  opts->upper_bound = 5.;

  // openmp
  opts->omp_numthreads = 40;

  //extra config
  opts->kernel = 1;
  opts->num_params = 3;
  opts->num_loc = 40000;
  opts->zvecs = 1;

  // bivariate 
  opts->p = 1; // univaraite

  // k nearest neighbors
  opts->knn = 0;

  // random ordering
  opts->randomordering = 0;

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
    // else if ( (strcmp("--batchCount",   argv[i]) == 0 || strcmp("--batch",   argv[i]) == 0) && i+1 < argc ) {
    //   i++;
    //   int start, stop, step;
    //   char op;
    //   info = sscanf( argv[i], "%d:%d%c%d", &start, &stop, &op, &step );
    //   if( info == 1 ){
    //     opts->batch[0] = opts->batchCount = start;
    //   }else {
    //     fprintf( stderr, "error: --batchCount %s is invalid; ensure start > 0.\n",
    //       argv[i] );
    //     exit(1);
    //   }
    //   //opts->batchCount = atoi( argv[++i] );
    //   //kblas_assert( opts->batchCount > 0, "error: --batchCount %s is invalid; ensure batchCount > 0.\n", argv[i] );
    // }
    // else if ( (strcmp("--rank",   argv[i]) == 0) && i+1 < argc ) {
    //   i++;
    //   int start, stop, step;
    //   char op, sep;
    //   info = sscanf( argv[i], "%d%c%d%c%d", &start, &sep, &stop, &op, &step );
    //   if( info == 1 ){
    //     opts->rank[0] = opts->rank[1] = start;
    //   }else
    //   if( info == 3 ){
    //     opts->rank[0] = start;
    //     opts->rank[1] = stop;
    //   }else
    //   if ( info == 5 && start >= 0 && stop >= 0 && step != 0 && (op == '+' || op == '*' || op == ':')) {
    //     opts->rtest = 0;
    //     for( int b = start; (step > 0 ? b <= stop : b >= stop); ) {
    //       opts->rank[ opts->rtest++ ] = b;
    //       if(op == '*') b *= step; else b += step;
    //     }
    //   }
    //   else {
    //     fprintf( stderr, "error: --range %s is invalid; ensure start >= 0, stop >= 0, step != 0 && op in (+,*,:).\n",
    //       argv[i] );
    //     exit(1);
    //   }
    //   //opts->batchCount = atoi( argv[++i] );
    //   //kblas_assert( opts->batchCount > 0, "error: --batchCount %s is invalid; ensure batchCount > 0.\n", argv[i] );
    // }
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
      if( info == 1 && num > 0 ){
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
    else if ( (strcmp("--kernel",   argv[i]) == 0) && i+1 < argc ) {
      i++;
      int kernel;
      info = sscanf( argv[i], "%d", &kernel);
      if( info == 1 && kernel == 1 ){
        opts->kernel = 1;
        opts->num_params = 3; 
        opts->p = 1; // univariate_matern_stationary
      }else if (info == 1 && kernel == 2){
        opts->kernel = 2;
        opts->num_params = 6; 
        opts->p = 2; // bivariate_matern_parsimonious
      }
      else{
        fprintf( stderr, "Other kernel is developing now!");
        exit(1);
      }
    }
    else if ( (strcmp("--num_loc",   argv[i]) == 0) && i+1 < argc ) {
      i++;
      int num_loc;
      info = sscanf( argv[i], "%d", &num_loc);
      opts->num_loc=num_loc;
    }
    else if ( (strcmp("--zvecs",   argv[i]) == 0) && i+1 < argc ) {
      i++;
      int zvecs;
      info = sscanf( argv[i], "%d", &zvecs);
      if( info == 1 && zvecs <= 10000 ){
        opts->zvecs=zvecs;
      }else{
        fprintf( stderr, "Your dataset does not contain the replicate more than 50!");
        exit(1);
      }
    }
    // k nearest neighbors
    else if ( strcmp("--knn", argv[i]) == 0 ) {
      opts->knn  = 1; 
    }
    // ordering 
    else if ( strcmp("--randomordering", argv[i]) == 0 ) {
      opts->randomordering  = 1; 
    }
    // ture parameters
    else if ( strcmp("--ikernel", argv[i]) == 0 && i+1 < argc ) {
      i++;
      double a1, a2, a3, a4, a5, a6;
      info = sscanf( argv[i], "%lf:%lf:%lf:%lf:%lf:%lf", &a1, &a2, &a3, &a4, &a5, &a6);
      if ( info == 3 ) {
        fprintf(stderr, "You are using the Matern Kernel in the univariate case now!\n");
        opts->sigma = a1;
        opts->beta = a2;
        opts->nu = a3;
      }else if (info ==6){
        fprintf(stderr, "You are using the Parsimonious Matern Kernel in the bivariate case now!\n");
        opts->sigma1 = a1;
        opts->sigma2 = a2;
        opts->alpha = a3;
        opts->nu1 = a4;
        opts->nu2 = a5;
        opts->beta = a6;
      }else{
        printf("Other kernels have been developing on the way!");
      }
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
