#!/bin/bash

N=100
Nbs=1
Ncs=100
# Test 1 - likelihood computation
echo "================ Full likelihood log f(x1, x2)========================"
# used for full likelihood
./bin/test_dvecchia_batch -N $Nbs:1 -s --kernel 1 --num_loc $N --omp_threads 1 --perf --vecchia_cs $Ncs

Ncs=99
echo "================ Vecchia likelihood log f(x1) + log f(x1|x2)========================"
# used for vecchia for full likelihood (1 batch)
./bin/test_dvecchia_batch -N $Nbs:1 -s --kernel 1 --num_loc $N --omp_threads 1 --perf --vecchia_cs $Ncs
# echo "================ Vecchia likelihood log f(x1) + log f(x1|x2) + knn========================"
# # used for vecchia for full likelihood (1 batch) + knn
# ./bin/test_dvecchia_batch -N $Nbs:1 -s --kernel 1 --num_loc $N --omp_threads 1 --perf --vecchia_cs $Ncs --knn
