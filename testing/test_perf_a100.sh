#!/bin/bash

for seed in 0 1 2 3 4
do
    Nbs=1
    for num_N in 40000 80000 120000 160000 200000 280000 360000 420000 500000 600000 700000 800000 900000 1000000
    do
        for Ncs in 10 30 60 90 120 150
        do
        ./bin/test_dvecchia_batch --ikernel 1.0:0.1:0.5 -N $Nbs:1 -s --kernel univariate_matern_stationary_no_nugget --num_loc $num_N --omp_threads 40 --perf --vecchia_cs $Ncs
        done
    done
done
mkdir ./log/a100
rm ./log/locs*
mv ./log/perf* ./log/a100