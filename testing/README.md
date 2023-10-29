## How to use?
1. run 'bash test_*.sh' in your terminal, such as `test_kl.sh`, `test_timing.sh`;
2. check the log-likelihood and Time total in the output
3. different machines may give different time (large)/log-likelihood (small)
   because of the performance of gpu/cpu, and the tolerance of machines;

## You can tune the following:
- N: the number of locations
- Ncs: the number of conditioning size 
- Nbs: (do not tune now) 1 for classic vecchia approximation, other vlaues are being developed.

## parameters of the command;
 - --kernel   :  1, matern kernel, others TBD
 - -N         :  fixed! NOT MODIFIED
 - --num_loc  :  number of locations, such as 180k
 - --omp_threads : number of CPU to be occupied
 - --perf     :  performance mode, without doing MLE
 - --vecchia_cs: number of conditioning size
 - --knn      : nearest neighbor (recommended)
 - --randomoredering: literally
 - --ngpu      : number of GPU to be occupied
