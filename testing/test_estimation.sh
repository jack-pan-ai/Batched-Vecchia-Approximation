#!/bin/bash

N=6000
Nbs=1
Ncs=120

./bin/test_dvecchia_batch --ikernel 0.1:0.1:1.5 -N $Nbs:1 -s --kernel 1 --num_loc $N --omp_threads 40 --vecchia_cs $Ncs --knn
