#!/bin/bash

Nbs=1
for num_N in 40000 80000 120000 160000 200000 300000 400000 500000 620000 
do
    for Ncs in 16 32 64 84 108 128
    do
    ./bin/test_dvecchia_batch --ikernel 1.0:0.1:0.5 -N $Nbs:1 -s --kernel univariate_matern_stationary_no_nugget --num_loc $num_N --omp_threads 40 --perf --vecchia_cs $Ncs
    done
done

mkdir ./log/a100
mv ./log/locs* ./log/a100
mv ./log/perf* ./log/a100