#!/bin/bash

N=260000
Nbs=1
Ncs=120

./bin/test_dvecchia_batch -N $Nbs:1 -s --kernel 1 --num_loc $N --omp_threads 40 --perf --vecchia_cs $Ncs  
