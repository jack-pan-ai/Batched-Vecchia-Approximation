#!/bin/bash

Nbs=1
for num_N in 40000 80000 120000 160000 200000 280000 360000 420000 500000 600000 700000 800000 900000 1000000
do
    for Ncs in 10 30 60 90 120 150
    do
    ./bin/test_dvecchia_batch --ikernel 1.0:0.1:0.5 -N $Nbs:1 -s --kernel univariate_matern_stationary_no_nugget --num_loc $num_N --omp_threads 40 --perf --vecchia_cs $Ncs
    done
done

./bin/test_dvecchia_batch --ikernel 1.0:0.1:0.5 -N 1:1 -s --kernel univariate_matern_stationary_no_nugget --num_loc 16000 --omp_threads 40 --perf --vecchia_cs 500
./bin/test_dvecchia_batch --ikernel 1.0:0.1:0.5 -N 1:1 -s --kernel univariate_matern_stationary_no_nugget --num_loc 4000 --omp_threads 40 --perf --vecchia_cs 1000

mkdir ./log/v100
rm ./log/locs*
mv ./log/perf* ./log/v100