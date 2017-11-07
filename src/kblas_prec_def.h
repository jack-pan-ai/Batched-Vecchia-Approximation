 /**
  --* (C) Copyright 2013 King Abdullah University of Science and Technology
  Authors:
  Ali Charara (ali.charara@kaust.edu.sa)
  David Keyes (david.keyes@kaust.edu.sa)
  Hatem Ltaief (hatem.ltaief@kaust.edu.sa)

  Redistribution  and  use  in  source and binary forms, with or without
  modification,  are  permitted  provided  that the following conditions
  are met:

  * Redistributions  of  source  code  must  retain  the above copyright
  * notice,  this  list  of  conditions  and  the  following  disclaimer.
  * Redistributions  in  binary  form must reproduce the above copyright
  * notice,  this list of conditions and the following disclaimer in the
  * documentation  and/or other materials provided with the distribution.
  * Neither  the  name of the King Abdullah University of Science and
  * Technology nor the names of its contributors may be used to endorse
  * or promote products derived from this software without specific prior
  * written permission.
  *
  THIS  SOFTWARE  IS  PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
  ``AS IS''  AND  ANY  EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
  LIMITED  TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
  A  PARTICULAR  PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
  HOLDERS OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
  SPECIAL,  EXEMPLARY,  OR  CONSEQUENTIAL  DAMAGES  (INCLUDING,  BUT NOT
  LIMITED  TO,  PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
  DATA,  OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
  THEORY  OF  LIABILITY,  WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
  (INCLUDING  NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
  OF  THIS  SOFTWARE,  EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
  **/
#ifndef __KBLAS_PREC_DEF__
#define __KBLAS_PREC_DEF__

#if defined   PREC_s
  #define TYPE float

  #define kblasXgemm_batch kblasSgemm_batch
  #define Xgemm_batch_strided_wsquery sgemm_batch_strided_wsquery

  #define kblasXsyrk_batch_wsquery kblasSsyrk_batch_wsquery
  #define kblasXsyrk_batch kblasSsyrk_batch
  #define kblasXsyrk_batch_strided_wsquery kblasSsyrk_batch_strided_wsquery
  #define kblasXsyrk_batch_strided kblasSsyrk_batch_strided

  #define kblasXtrsm_batch kblasStrsm_batch
  #define kblasXtrsm_batch_wsquery kblasStrsm_batch_wsquery
  #define kblasXtrsm_batch_strided kblasStrsm_batch_strided
  #define kblasXtrsm_batch_strided_wsquery kblasStrsm_batch_strided_wsquery

  #define cublasXgemm cublasSgemm
  #define cublasXgemmBatched cublasSgemmBatched
  #define cublasXgemmStridedBatched cublasSgemmStridedBatched
  #define magmablas_Xgemm_batched magmablas_sgemm_batched

#elif defined PREC_d
  #define TYPE double

  #define kblasXgemm_batch kblasDgemm_batch
  #define Xgemm_batch_strided_wsquery dgemm_batch_strided_wsquery

  #define kblasXsyrk_batch_wsquery kblasDsyrk_batch_wsquery
  #define kblasXsyrk_batch kblasDsyrk_batch
  #define kblasXsyrk_batch_strided_wsquery kblasDsyrk_batch_strided_wsquery
  #define kblasXsyrk_batch_strided kblasDsyrk_batch_strided

  #define kblasXtrsm_batch kblasDtrsm_batch
  #define kblasXtrsm_batch_wsquery kblasDtrsm_batch_wsquery
  #define kblasXtrsm_batch_strided kblasDtrsm_batch_strided
  #define kblasXtrsm_batch_strided_wsquery kblasDtrsm_batch_strided_wsquery

  #define cublasXgemm cublasDgemm
  #define cublasXgemmBatched cublasDgemmBatched
  #define cublasXgemmStridedBatched cublasDgemmStridedBatched
  #define magmablas_Xgemm_batched magmablas_dgemm_batched

#elif defined PREC_c
  #define TYPE cuComplex

  #define kblasXgemm_batch kblasCgemm_batch
  #define Xgemm_batch_strided_wsquery cgemm_batch_strided_wsquery

  #define kblasXsyrk_batch_wsquery kblasCsyrk_batch_wsquery
  #define kblasXsyrk_batch kblasCsyrk_batch
  #define kblasXsyrk_batch_strided_wsquery kblasCsyrk_batch_strided_wsquery
  #define kblasXsyrk_batch_strided kblasCsyrk_batch_strided

  #define kblasXtrsm_batch kblasCtrsm_batch
  #define kblasXtrsm_batch_wsquery kblasCtrsm_batch_wsquery
  #define kblasXtrsm_batch_strided kblasCtrsm_batch_strided
  #define kblasXtrsm_batch_strided_wsquery kblasCtrsm_batch_strided_wsquery

  #define cublasXgemm cublasCgemm
  #define cublasXgemmBatched cublasCgemmBatched
  #define cublasXgemmStridedBatched cublasCgemmStridedBatched
  #define magmablas_Xgemm_batched magmablas_cgemm_batched

#elif defined PREC_z
  #define TYPE cuDoubleComplex

  #define kblasXgemm_batch kblasZgemm_batch
  #define Xgemm_batch_strided_wsquery zgemm_batch_strided_wsquery

  #define kblasXsyrk_batch_wsquery kblasZsyrk_batch_wsquery
  #define kblasXsyrk_batch kblasZsyrk_batch
  #define kblasXsyrk_batch_strided_wsquery kblasZsyrk_batch_strided_wsquery
  #define kblasXsyrk_batch_strided kblasZsyrk_batch_strided

  #define kblasXtrsm_batch kblasZtrsm_batch
  #define kblasXtrsm_batch_wsquery kblasZtrsm_batch_wsquery
  #define kblasXtrsm_batch_strided kblasZtrsm_batch_strided
  #define kblasXtrsm_batch_strided_wsquery kblasZtrsm_batch_strided_wsquery

  #define cublasXgemm cublasZgemm
  #define cublasXgemmBatched cublasZgemmBatched
  #define cublasXgemmStridedBatched cublasZgemmStridedBatched
  #define magmablas_Xgemm_batched magmablas_zgemm_batched

#else
  #error "No precesion defined"
#endif

#endif //__KBLAS_PREC_DEF__
