#!/bin/bash

N=100
Nbs=1
Ncs=20

echo "================ Vecchia likelihood log f(x1) + log f(x1|x2) 1 gpu ========================"
# used for vecchia for full likelihood (2 batch) + knn
./bin/test_dvecchia_batch -N $Nbs:1 -s --kernel 1 --num_loc $N --omp_threads 1 --perf --vecchia_cs $Ncs --knn

echo "================ Vecchia likelihood log f(x1) + log f(x1|x2) 2 gpus ========================"
# used for vecchia for full likelihood (2 batch) + knn
./bin/test_dvecchia_batch -N $Nbs:1 -s --kernel 1 --num_loc $N --omp_threads 1 --perf --vecchia_cs $Ncs --knn --ngpu 2

