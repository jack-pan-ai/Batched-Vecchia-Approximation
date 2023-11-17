#!/bin/bash

N=900
Nbs=1
Ncs=900
# Test 1 - likelihood computation (1 batch)
echo "================ Full likelihood log f(x1, x2)========================"
# used for full likelihood
./bin/test_dvecchia_batch -N $Nbs:1 -s --kernel univariate_powexp_stationary_no_nugget --num_loc $N --omp_threads 1 --perf --vecchia_cs $Ncs --ikernel 1.0:0.1:0.5 --seed 0

Ncs=899
echo "================ Vecchia likelihood log f(x1) + log f(x1|x2)========================"
# used for vecchia for full likelihood (2 batch)
./bin/test_dvecchia_batch -N $Nbs:1 -s --kernel univariate_powexp_stationary_no_nugget --num_loc $N --omp_threads 1 --perf --vecchia_cs $Ncs --ikernel 1.0:0.1:0.5 --seed 0

echo "================ Vecchia likelihood log f(x1) + log f(x1|x2) + knn========================"
# used for vecchia for full likelihood (2 batch) + knn
./bin/test_dvecchia_batch -N $Nbs:1 -s --kernel univariate_powexp_stationary_no_nugget --num_loc $N --omp_threads 1 --perf --vecchia_cs $Ncs --ikernel 1.0:0.1:0.5 --seed 0 --knn

./bin/test_dvecchia_batch -N $Nbs:1 -s --kernel univariate_powexp_stationary_no_nugget --num_loc $N --omp_threads 1 --perf --vecchia_cs $Ncs --ikernel 1.0:0.1:0.5 --randomordering --seed 0 -knn

echo "================ Vecchia likelihood log f(x1) + log f(x1|x2) 2 gpus ========================"
# used for vecchia for full likelihood (2 batch) + knn
./bin/test_dvecchia_batch -N $Nbs:1 -s --kernel univariate_powexp_stationary_no_nugget --num_loc $N --omp_threads 1 --perf --vecchia_cs $Ncs --ngpu 2 --ikernel 1.0:0.1:0.5 --seed 0 -knn
