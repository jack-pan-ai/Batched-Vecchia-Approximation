#!/bin/bash

N=6400
Nbs=1
Ncs=30
echo "================ Morton ordering ========================"
# used for full likelihood
./bin/test_dvecchia_batch -N $Nbs:1 -s --kernel 1 --num_loc $N --omp_threads 1 --perf --vecchia_cs $Ncs --knn

echo "================ Random ordering ========================"
# used for full likelihood
./bin/test_dvecchia_batch -N $Nbs:1 -s --kernel 1 --num_loc $N --omp_threads 1 --perf --vecchia_cs $Ncs --knn --randomordering
