#!/bin/bash

N=2000
Nbs=1
Ncs=120

./bin/test_dvecchia_batch --ikernel ?:?:0.5 -N $Nbs:1 -s --kernel univariate_matern_stationary_no_nugget --num_loc $N --omp_threads 40 --vecchia_cs $Ncs --knn
# ./bin/test_dvecchia_batch --ikernel 0.001:1.2505:0.5 -N $Nbs:1 -s --kernel univariate_matern_stationary_no_nugget --num_loc $N --omp_threads 40 --vecchia_cs $Ncs --perf --randomordering --knn
