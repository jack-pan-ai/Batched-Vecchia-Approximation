#!/bin/bash

N=6400
Nbs=1
Ncs=30
Nmp=1

# Test 1 - likelihood computation
echo "================ Full likelihood log f(x1, x2)========================"
# used for full likelihood
./bin/test_dvecchia_batch -N $N:1 -s --kernel 1 --num_loc $N  --omp_threads $Nmp --perf
echo "================ Classic Vecchia + knn + Vsort========================"
# used for vecchia for full likelihood (1 batch) + knn
./bin/test_dvecchia_batch -N $Nbs:1 -s --kernel 1 --num_loc $N --omp_threads $Nmp --perf --vecchia_cs $Ncs --knn
echo "================ Classic Vecchia + knn + Random========================"
# used for vecchia for full likelihood (1 batch) + knn
./bin/test_dvecchia_batch -N $Nbs:1 -s --kernel 1 --num_loc $N --omp_threads $Nmp --randomordering --perf --vecchia_cs $Ncs --knn 