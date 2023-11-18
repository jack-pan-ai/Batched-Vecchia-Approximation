#!/bin/bash

N=260000
Nbs=1

echo "===================================================="
for Ncs in 16 32 48 64 108 120
do
./bin/test_dvecchia_batch --ikernel 1.0:0.1:0.5 -N $Nbs:1 -s --kernel univariate_matern_stationary_no_nugget --num_loc $N --omp_threads 40 --perf --vecchia_cs $Ncs
done

N=180000
Nbs=1
for Ncs in 16 32 48 64 108 120 128 140
do
./bin/test_dvecchia_batch --ikernel 1.0:0.1:0.5 -N $Nbs:1 -s --kernel univariate_matern_stationary_no_nugget --num_loc $N --omp_threads 40 --perf --vecchia_cs $Ncs
done