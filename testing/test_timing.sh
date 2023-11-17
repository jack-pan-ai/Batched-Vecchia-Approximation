#!/bin/bash

N=260000
Nbs=1
Ncs=120

./bin/test_dvecchia_batch -N $Nbs:1 -s --kernel univariate_matern_stationary_no_nugget --num_loc $N --omp_threads 40 --perf --vecchia_cs $Ncs  --ikernel 1.0:0.1:0.5
